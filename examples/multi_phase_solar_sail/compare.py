"""Run several algorithms on the same problem and tabulate them.

    python compare.py ALGORITHM [ALGORITHM ...] [--revolutions N] [--set key=value ...]
                      [--csv FILE]

Each ALGORITHM is an algorithm file (a single method or a flow, see
``algorithms/README.md``).  All of them are solved at the same revolution count
-- the one in ``problem.yaml`` unless ``--revolutions`` is given -- with the same
``--set`` overrides, one after another so that only one compiled problem is in
memory at a time.  There is no revolution search: this compares algorithms on a
fixed problem, which is what a study wants.

Every stage of every algorithm gets a row: its status, iterations, wall time
split into build, JAX compilation and SCP iterations, and the independent
validation of what it returned.  ``--csv`` writes the same rows to a file.

``--set`` follows ``main.py``.  A ``<stage>.``-prefixed override applies to the
algorithms that have a stage of that name and is ignored by the others.

    python compare.py algorithms/ms.yaml algorithms/p.yaml algorithms/p_to_ms.yaml
    python compare.py algorithms/*.yaml -n 6 --set problem.graveyard.perigee_altitude_km=40
"""

import argparse
import csv
import sys
from pathlib import Path

from algorithm import Algorithm, parse_overrides

EXAMPLE_DIRECTORY = Path(__file__).resolve().parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COLUMNS = [
    "algorithm", "stage", "seed_from", "discretisation", "revolutions", "status",
    "iterations", "build_s", "compile_s", "solve_s", "validate_s", "total_s",
    "feasible", "elapsed_days", "true_perigee_alt_km", "true_eccentricity",
    "accumulated_defect_km", "max_defect_nd", "defect_tol_nd",
    "max_repropagation_error_nd", "reason",
]


def stage_rows(algorithm: Algorithm, run, overrides: dict) -> list[dict]:
    rows = []
    for spec, stage in zip(algorithm.stages, run.stages):
        report, solution = stage.report, stage.solution
        row = dict(
            algorithm=algorithm.name, stage=stage.name, seed_from=stage.seed_from or "",
            discretisation=algorithm.stage_label(spec, overrides),
            revolutions=run.revolutions, status=stage.status,
            iterations=stage.iterations, reason=stage.reason,
            **{key: round(stage.timings.get(key, 0.0), 2)
               for key in ("build_s", "compile_s", "solve_s", "validate_s", "total_s")},
        )
        if report is not None:
            row.update(
                feasible=stage.feasible,
                elapsed_days=report["elapsed_days"],
                true_perigee_alt_km=report["true_final_perigee_km"] - solution.geo_radius_km,
                true_eccentricity=report["true_final_eccentricity"],
                accumulated_defect_km=report["accumulated_defect_km"],
                max_defect_nd=report["max_dynamics_defect_nd"],
                defect_tol_nd=report["defect_tolerance_nd"],
                max_repropagation_error_nd=report["max_propagation_error_nd"],
            )
        rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    print("\ncomparison")
    header = (f"  {'algorithm':<16} {'stage':<14} {'disc.':<12} {'status':<13} {'it':>4} "
              f"{'compile':>8} {'SCP':>7} {'total':>7}  {'verdict':<10} {'days':>8} "
              f"{'r_p-r_GEO':>9} {'e':>8} {'acc km':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        verdict = "" if "feasible" not in row else ("FEASIBLE" if row["feasible"] else "infeasible")
        tail = ""
        if "elapsed_days" in row:
            tail = (f"{row['elapsed_days']:8.4f} {row['true_perigee_alt_km']:+9.2f} "
                    f"{row['true_eccentricity']:8.6f} {row['accumulated_defect_km']:+8.3f}")
        print(f"  {row['algorithm']:<16} {row['stage']:<14} {row['discretisation']:<12} "
              f"{row['status']:<13} {row['iterations']:4d} {row['compile_s']:7.0f}s "
              f"{row['solve_s']:6.0f}s {row['total_s']:6.0f}s  {verdict:<10} {tail}")


def main(argv=None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("algorithms", nargs="+", help="algorithm files")
    parser.add_argument("--revolutions", "-n", type=int, default=None)
    parser.add_argument("--set", dest="assignments", action="append", default=[],
                        metavar="KEY=VALUE")
    parser.add_argument("--csv", type=Path, default=None, help="write the rows here")
    parser.add_argument("--final-only", action="store_true",
                        help="validate only each algorithm's final stage (saves a "
                             "re-propagation per intermediate stage)")
    args = parser.parse_args(argv)

    algorithms = [Algorithm(path if Path(path).is_absolute() else EXAMPLE_DIRECTORY / path)
                  for path in args.algorithms]
    all_stages = sorted({stage.name for algorithm in algorithms for stage in algorithm.stages})
    overrides = parse_overrides(args.assignments, all_stages)

    rows: list[dict] = []
    for algorithm in algorithms:
        mine = dict(overrides)
        if "stages" in overrides:
            mine["stages"] = {name: value for name, value in overrides["stages"].items()
                              if name in {s.name for s in algorithm.stages}}
        print("\n" + "=" * 78 + f"\n{algorithm.describe(mine)}\n" + "=" * 78, flush=True)
        run = algorithm.run(args.revolutions, mine, validate_all=not args.final_only)
        run.print_summary()
        rows.extend(stage_rows(algorithm, run, mine))
        if args.csv is not None:   # rewritten after every algorithm, so a crash keeps the rest
            _write_csv(args.csv, rows)

    _print_table(rows)
    if args.csv is not None:
        print(f"\nrows written to {args.csv}")
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
