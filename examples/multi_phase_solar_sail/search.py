"""Outer loop over the revolution count (the paper's Algorithm 1).

``revolutions`` sets the number of phases and seeds the initial guess, so it
cannot be optimised inside a single SCP solve; it has to be searched over.
``main.py`` runs this search on every invocation, so that changing the sail, the
discretisation or any other setting still lands on the minimum-time solution
without anyone having to work out the right count by hand.

The search brackets the minimum from wherever the starting guess lands:

* guess **feasible**   -> step down until a count fails.  The last feasible count
  is the minimum, and the failure is what proves it.
* guess **infeasible** -> step up until a count succeeds.  The first feasible
  count is the minimum.

Both rely on the same monotonicity: more revolutions means more time to reach the
graveyard orbit, so feasibility, once gained, is kept.  A good guess therefore
costs one extra solve to confirm; a poor guess just costs more steps, which is
the intended trade.

Only the best attempt's :class:`SolarSailProblem` is kept alive.  Each one owns a
compiled CVXPY subproblem that can run to gigabytes, so holding on to all of them
exhausts memory after a handful of solves.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

from problem import SolarSailProblem
from solution import SolarSailSolution, extract_solution, validation_report


@dataclass
class Attempt:
    """The outcome of one solve at a fixed revolution count.

    Deliberately holds no reference to the problem: see the module docstring.
    """

    revolutions: int
    solution: SolarSailSolution
    report: dict

    @property
    def feasible(self) -> bool:
        return bool(self.report["numerically_feasible"])

    @property
    def elapsed_days(self) -> float:
        return float(self.report["elapsed_days"])

    def summary(self) -> str:
        mark = "FEASIBLE  " if self.feasible else "infeasible"
        offset = self.report["final_perigee_km"] - self.solution.geo_radius_km
        return (f"N={self.revolutions:3d}  {mark}  "
                f"{self.elapsed_days:8.4f} days  "
                f"e_f={self.report['final_eccentricity']:.6f}  "
                f"r_p-r_GEO={offset:+8.2f} km  "
                f"defect={self.report['max_dynamics_defect_nd']:.2e}  "
                f"({self.report['iterations']:4d} it)")


@dataclass
class SearchResult:
    """What the search concluded."""

    best: Attempt | None
    problem: SolarSailProblem | None
    attempts: list[Attempt]
    bounded_by_minimum: bool = False

    @property
    def found(self) -> bool:
        return self.best is not None


def solve_once(config_path: str | Path, revolutions: int | None = None,
               overrides: dict | None = None, quiet: bool = False):
    """Build and solve for one revolution count.

    Returns ``(attempt, problem)``.  Drop the problem as soon as you no longer
    need it: it owns the compiled subproblem.
    """
    settings = dict(overrides or {})
    if revolutions is not None:
        settings["revolutions"] = int(revolutions)

    problem = SolarSailProblem(config_path, settings)
    problem.solve()
    solution = extract_solution(problem)
    attempt = Attempt(problem.revolutions, solution, validation_report(solution))
    if not quiet:
        print(f"  {attempt.summary()}", flush=True)
    return attempt, problem


def search_minimum_revolutions(
    config_path: str | Path,
    start: int,
    minimum: int = 1,
    maximum: int = 30,
    max_solves: int = 12,
    overrides: dict | None = None,
) -> SearchResult:
    """Bracket the fewest revolutions that still reach the graveyard orbit."""
    start = int(min(max(start, minimum), maximum))
    attempts: list[Attempt] = []
    best: Attempt | None = None
    best_problem: SolarSailProblem | None = None
    bounded = False

    def release(problem):
        del problem
        gc.collect()

    def keep(attempt, problem):
        nonlocal best, best_problem
        if best_problem is not None:
            release(best_problem)
        best, best_problem = attempt, problem

    print(f"\nrevolution search: starting from N = {start}, "
          f"bracketing within [{minimum}, {maximum}]", flush=True)

    attempt, problem = solve_once(config_path, start, overrides)
    attempts.append(attempt)

    if attempt.feasible:
        keep(attempt, problem)
        print("  feasible: stepping down until a count fails", flush=True)
        revolutions = start - 1
        proved = False
        while revolutions >= minimum and len(attempts) < max_solves:
            attempt, problem = solve_once(config_path, revolutions, overrides)
            attempts.append(attempt)
            if not attempt.feasible:
                release(problem)
                proved = True  # the failure is what proves minimality
                break
            keep(attempt, problem)
            revolutions -= 1
        # Without a failure below it, the best count is only the smallest one
        # tried, not the smallest one possible.
        bounded = not proved
    else:
        release(problem)
        print("  infeasible: stepping up until a count succeeds", flush=True)
        revolutions = start + 1
        while revolutions <= maximum and len(attempts) < max_solves:
            attempt, problem = solve_once(config_path, revolutions, overrides)
            attempts.append(attempt)
            if attempt.feasible:
                keep(attempt, problem)
                break
            release(problem)
            revolutions += 1

    _report(best, attempts, bounded, minimum)
    return SearchResult(best, best_problem, attempts, bounded)


def _report(best, attempts, bounded, minimum) -> None:
    print("\nsearch history")
    for attempt in sorted(attempts, key=lambda a: a.revolutions):
        print(f"  {attempt.summary()}")

    if best is None:
        print("\n  no feasible revolution count found in range; "
              "widen revolution_search.maximum, or relax the problem")
        return

    print(f"\n  minimum-time solution: N = {best.revolutions}, "
          f"{best.elapsed_days:.4f} days")
    if bounded:
        print(f"  note: no smaller count was shown to fail, so N = "
              f"{best.revolutions} is the smallest tried rather than the "
              f"smallest possible -- lower revolution_search.minimum, or raise "
              f"max_solves, to keep going")
