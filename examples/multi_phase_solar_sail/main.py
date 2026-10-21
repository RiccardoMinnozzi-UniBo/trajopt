"""Solve the multi-phase solar-sail GEO disposal problem end to end.

    python main.py [ALGORITHM] [--revolutions N] [--no-search] [--set key=value ...]
                   [--output DIR]

ALGORITHM is any algorithm file -- a single method such as ``algorithms/ms.yaml``
or a flow such as ``algorithms/p_to_ms.yaml`` (see ``algorithms/README.md``) --
and defaults to ``config-ps.yaml``.  ``config.yaml`` and ``config-ps.yaml`` are
the original two, kept as aliases of ``algorithms/ms.yaml`` and
``algorithms/p.yaml``.

The revolution count cannot be optimised inside a single SCP solve -- it sets the
number of phases -- so this script searches over it, starting from the guess in
``problem.revolutions`` and bracketing the minimum from there (see ``search.py``).
That means changing the sail, the discretisation or anything else still lands on
the minimum-time solution without working out the right count by hand; a poor
guess only costs extra solves.

``--no-search`` (or ``problem.revolution_search.enabled: false``) solves once at
the guess, which is quicker when iterating on something else.  ``--set``
overrides any setting without editing a file, e.g.
``--set problem.sail.area_to_mass_m2_kg=5`` or, for one stage of a flow,
``--set p.method.hyperparams.convergence.iter_max=40``.
"""

import argparse
import sys
from pathlib import Path

from algorithm import Algorithm, parse_overrides
from search import search_minimum_revolutions, solve_once
from solution import (
    phase_timing_report,
    print_validation_report,
    save_solution,
)
from visualization import render_all, render_shadow_animation

EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXAMPLE_DIRECTORY / "config-ps.yaml"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else EXAMPLE_DIRECTORY / path


def main(config_file: str | Path = DEFAULT_CONFIG, revolutions: int | None = None,
         search: bool | None = None, assignments: list[str] | None = None,
         output_directory: str | Path | None = None) -> None:
    algorithm = Algorithm(_resolve(config_file))
    overrides = parse_overrides(assignments or [], [s.name for s in algorithm.stages])
    if revolutions is not None:
        overrides.setdefault("problem", {})["revolutions"] = int(revolutions)
    print(algorithm.describe(overrides))

    settings = algorithm.settings(overrides)
    search_settings = settings.revolution_search
    guess = int(settings.revolutions)
    if search is None:
        search = bool(search_settings.enabled)

    if search:
        result = search_minimum_revolutions(
            algorithm,
            start=guess,
            minimum=int(search_settings.minimum),
            maximum=int(search_settings.maximum),
            max_solves=int(search_settings.max_solves),
            overrides=overrides,
        )
        if not result.found:
            # Nothing validated; still save and draw the last attempt so the
            # failure can be looked at rather than just reported.
            print("\nNo feasible revolution count: reporting the last attempt.")
            attempt = result.attempts[-1]
            problem = None
        else:
            attempt, problem = result.best, result.problem
        title = f"Minimum-time solution (N = {attempt.revolutions})"
    else:
        attempt, problem = solve_once(algorithm, guess, overrides, quiet=True)
        title = f"Solution at the guess (N = {attempt.revolutions})"

    attempt.run.print_summary()
    solution, report = attempt.solution, attempt.report
    if report is None:
        print("\nThe final stage produced no trajectory; nothing to report or draw.")
        return

    if problem is not None:
        phase_timing_report(problem, title)
    print_validation_report(report)

    output_settings = algorithm.output_settings()
    output = _resolve(output_directory or output_settings.directory)
    files = [save_solution(solution, output / output_settings.solution_file)]

    # Figures are always produced: an unconverged run is exactly when you most
    # want to look at the trajectory.  The animation is expensive, so it is
    # reserved for solutions that pass validation.
    files.extend(render_all(solution, output, report))

    if report["numerically_feasible"]:
        files.append(render_shadow_animation(
            solution,
            output / output_settings.animation_file,
            int(output_settings.animation_frames),
        ))
    else:
        print("\nValidation failed: skipping the animation. "
              "The static figures above show what the solver returned.")

    print("\noutputs")
    for path in files:
        print(f"  {path}")


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("algorithm", nargs="?", default=str(DEFAULT_CONFIG),
                        help="algorithm file: a single method or a flow")
    parser.add_argument("--revolutions", "-n", type=int, default=None,
                        help="revolution count (the search's starting guess)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no-search", dest="search", action="store_false", default=None,
                       help="solve once at the revolution count instead of searching")
    group.add_argument("--search", dest="search", action="store_true",
                       help="search over the revolution count even if the config disables it")
    parser.add_argument("--set", dest="assignments", action="append", default=[],
                        metavar="KEY=VALUE", help="override a setting; repeatable")
    parser.add_argument("--output", default=None,
                        help="results directory (default: the file's output.directory, "
                             "else results/<file name>)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _arguments()
    main(args.algorithm, args.revolutions, args.search, args.assignments, args.output)
