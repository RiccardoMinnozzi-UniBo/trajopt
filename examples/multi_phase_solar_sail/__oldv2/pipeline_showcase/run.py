"""Solve the showcase pipeline and save every stage.

    python run.py [--revolutions N]

Runs ``configs/pipeline.yaml`` (relaxed p -> ms -> ms-mirrored hp) and writes,
per stage, to ``results/``:

* ``<stage>_solution.npz``  the validated solution (``solution.load_solution``),
* ``<stage>_iterates.npz``  every SCP iterate of every phase, plus metadata,

and ``results/summary.json`` with each stage's status, timings, tolerances and
validation report.  ``figures.py`` draws from these files only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
sys.path.insert(0, str(EXAMPLE))
sys.path.insert(0, str(EXAMPLE / "pipeline_benchmark"))

from solution import save_solution  # noqa: E402
from solve import RecordingAlgorithm, _jsonable  # noqa: E402

CONFIG = HERE / "configs" / "pipeline.yaml"
RESULTS = HERE / "results"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--revolutions", "-n", type=int, default=None)
    args = parser.parse_args(argv)

    algorithm = RecordingAlgorithm(CONFIG)
    print(algorithm.describe(), flush=True)
    started = time.perf_counter()
    run = algorithm.run(args.revolutions)
    run.print_summary()

    RESULTS.mkdir(parents=True, exist_ok=True)
    stages = []
    for spec, outcome in zip(algorithm.stages, run.stages):
        problem_cfg, method_cfg = algorithm.stage_config(spec)
        entry = {
            "name": outcome.name,
            "label": algorithm.stage_label(spec),
            "seed_from": outcome.seed_from,
            "status": outcome.status,
            "reason": outcome.reason,
            "iterations": int(outcome.iterations),
            "timings": _jsonable(outcome.timings),
            "eps_cost": float(method_cfg.hyperparams.convergence.get("eps_cost", 1e-4)),
            "defect_budget_km": float(problem_cfg.max_accumulated_defect_km),
            "mesh": {key: int(problem_cfg.get(key, 1)) for key in
                     ("sunlit_nodes", "eclipse_nodes", "sunlit_hp_segments", "eclipse_hp_segments")},
            "nsub": int(method_cfg.hyperparams.discretize.get("nsub", 0) or 0),
            "mode": str(method_cfg.hyperparams.discretize.mode),
            "report": _jsonable(outcome.report) if outcome.report else None,
        }
        if outcome.solution is not None:
            save_solution(outcome.solution, RESULTS / f"{outcome.name}_solution.npz")
        if outcome.name in algorithm.records:
            arrays, meta = algorithm.records[outcome.name]
            np.savez_compressed(RESULTS / f"{outcome.name}_iterates.npz",
                                meta=np.asarray(json.dumps(meta)), **arrays)
        stages.append(entry)

    summary = {"config": str(CONFIG.relative_to(EXAMPLE)), "revolutions": int(run.revolutions),
               "completed": bool(run.completed), "failed_stage": run.failed_stage,
               "wall_s": time.perf_counter() - started, "stages": stages}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"saved {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
