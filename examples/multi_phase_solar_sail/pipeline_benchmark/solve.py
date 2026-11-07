"""Solve the benchmark approaches and save what the solver produced.

    python solve.py [APPROACH ...] [--revolutions N] [--data DIR]

APPROACH is any of ``p``, ``ms``, ``p_ms_hp`` (default: all three), each a file
in ``configs/``.  Every approach is solved once at the revolution count of
``problem.yaml`` (no revolution search), one after another.

For every stage the raw solver record is written to ``DIR/<approach>__<stage>.npz``
(default ``results/data``): every SCP iterate of every phase (node states,
longitudes, controls, the solver's own dynamics defect), the per-iteration
scalars the solver logs (cost, step size, trust region, convergence checks,
penalty weights, timings) and the metadata needed to rebuild the dynamics
(phase parameters, scales, mesh).  Nothing derived is saved: ``postprocess.py``
computes every metric and figure from these files, so it can be re-run on a new
solution without solving again.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
sys.path.insert(0, str(EXAMPLE))

from algorithm import Algorithm  # noqa: E402

APPROACHES = ("p", "ms", "p_ms_hp")
DEFAULT_DATA = HERE / "results" / "data"

# per-iteration, per-phase scalars copied from the solver's iteration record
_SCALARS = ("alpha", "s_mult", "tr_ratio", "cost", "penalty_cost",
            "solve_time", "parse_time", "discretization_time")
_CHECKS = ("dz", "dcost", "final_state", "nonconvex_inequality",
           "nonconvex_equality", "dynamics")
_PENALTY_FIELDS = ("W", "dual")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _indices(value) -> list[int]:
    return [int(i) for i in np.atleast_1d(np.asarray(value)).ravel()]


def _float(value) -> float:
    try:
        return float(np.asarray(value, dtype=float).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return np.nan


def capture(problem) -> tuple[dict, dict]:
    """Every iterate of every phase, as arrays, plus the per-phase metadata."""
    phases = list(problem.segments.items())
    count = min(len(segment.iter_data_list) for _, segment in phases)
    arrays: dict[str, np.ndarray] = {}
    phase_meta = []

    for index, (name, segment) in enumerate(phases):
        idx = segment.index_map.indices
        state_idx = _indices(idx.z.state)
        state_scales = np.asarray(segment.nondim.state_scales, dtype=float).ravel()
        time_scale = float(segment.nondim.time_scale)
        control_scale = float(np.asarray(segment.nondim.control_scales, dtype=float).ravel()[0])

        states, longitudes, controls, defects = [], [], [], []
        scalars = {key: [] for key in _SCALARS}
        checks = {key: [] for key in _CHECKS}
        penalties: dict[str, list] = {}
        converged, status = [], []
        for record in segment.iter_data_list[:count]:
            x_nd, L_nd, _, u_nd, _ = segment.index_map.unpack_znu(record.z_opt, record.nu_opt)
            states.append(np.asarray(x_nd, dtype=float) * state_scales)
            longitudes.append(np.asarray(L_nd, dtype=float).reshape(-1) * time_scale)
            controls.append(np.asarray(u_nd, dtype=float)[:, 0] * control_scale)
            defect = record.get("defect", None)
            n_int = states[-1].shape[0] - 1
            defects.append(np.full((n_int, len(state_idx)), np.nan) if defect is None
                           else np.asarray(defect, dtype=float)[:, state_idx])
            for key in _SCALARS:
                scalars[key].append(_float(record.get(key, np.nan)))
            chk = record.get("chk", None) or {}
            for key in _CHECKS:
                checks[key].append(_float(chk.get(key, np.nan)))
            for field in _PENALTY_FIELDS:
                for cname, value in (record.get(field, None) or {}).items():
                    arr = np.abs(np.asarray(value, dtype=float))
                    peak = float(np.max(arr)) if arr.size else np.nan
                    penalties.setdefault(f"{field}__{cname}", []).append(peak)
            converged.append(bool(record.get("bool_conv", False)))
            status.append(str(record.get("status", "")))

        prefix = f"phase{index:02d}__"
        arrays[prefix + "state"] = np.stack(states)          # (iterates, nodes, 6), dimensional
        arrays[prefix + "longitude"] = np.stack(longitudes)  # (iterates, nodes), rad
        arrays[prefix + "control"] = np.stack(controls)      # (iterates, nodes), rad
        arrays[prefix + "defect"] = np.stack(defects)        # (iterates, nodes-1, 6), solver units
        for key, values in {**scalars, **{f"chk__{k}": v for k, v in checks.items()}}.items():
            arrays[prefix + key] = np.asarray(values, dtype=float)
        for key, values in penalties.items():
            if len(values) == count:
                arrays[prefix + key] = np.asarray(values, dtype=float)
        arrays[prefix + "bool_conv"] = np.asarray(converged)
        arrays[prefix + "status"] = np.asarray(status)

        eps = {}
        for cname, constraint in segment.constraints.items():
            pen = getattr(constraint, "penalties", None)
            value = getattr(pen, "eps", None) if pen is not None else None
            if value is not None:
                eps[cname] = _jsonable(np.atleast_1d(np.asarray(value, dtype=float)).ravel())
        eps_dyn = np.asarray(segment.eps_dyn, dtype=float).ravel()

        phase_meta.append({
            "name": name,
            "sunlit": bool(segment.params.illumination),
            "params": _jsonable(dict(segment.params)),
            "state_scales": state_scales.tolist(),
            "eps_dyn_state": eps_dyn[state_idx].tolist(),
            "constraint_eps": eps,
            "discretize": str(segment.mesh.discretize),
            "segments": int(segment.mesh.segments),
            "n_nodes": int(states[-1].shape[0]),
        })

    return arrays, {"phases": phase_meta, "iterates": int(count)}


class RecordingAlgorithm(Algorithm):
    """An Algorithm that keeps each stage's raw solver record before releasing it."""

    def __init__(self, path):
        super().__init__(path)
        self.records: dict[str, tuple[dict, dict]] = {}

    def _run_stage(self, stage, overrides, exported, validate):
        outcome, problem = super()._run_stage(stage, overrides, exported, validate)
        if problem is not None:
            try:
                self.records[stage.name] = capture(problem)
            except Exception as exc:  # keep the run going; the stage just has no record
                print(f"capture of stage '{stage.name}' failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
        return outcome, problem


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=EXAMPLE, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def solve(approach: str, revolutions: int | None, data_dir: Path) -> None:
    algorithm = RecordingAlgorithm(HERE / "configs" / f"{approach}.yaml")
    print("\n" + "=" * 78 + f"\n{algorithm.describe()}\n" + "=" * 78, flush=True)
    started = time.perf_counter()
    run = algorithm.run(revolutions)
    print(f"[{approach}: {time.perf_counter() - started:.0f} s]", flush=True)
    run.print_summary()

    data_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    stage_names = [stage.name for stage in run.stages]
    for index, (spec, outcome) in enumerate(zip(algorithm.stages, run.stages)):
        if outcome.name not in algorithm.records:
            continue
        arrays, meta = algorithm.records[outcome.name]
        problem_cfg, method_cfg = algorithm.stage_config(spec)
        meta.update({
            "approach": approach,
            "stage": outcome.name,
            "stage_index": index,
            "stages": stage_names,
            "seed_from": outcome.seed_from,
            "discretisation": algorithm.stage_label(spec),
            "status": outcome.status,
            "reason": outcome.reason,
            "iterations": int(outcome.iterations),
            "timings": _jsonable(outcome.timings),
            "revolutions": int(run.revolutions),
            "git_commit": commit,
            "settings": {
                "geo_radius_km": float(problem_cfg.constants.geo_radius_km),
                "target_perigee_km": float(problem_cfg.constants.geo_radius_km
                                           + problem_cfg.graveyard.perigee_altitude_km),
                "target_eccentricity": float(problem_cfg.graveyard.eccentricity_max),
                "defect_budget_km": float(problem_cfg.max_accumulated_defect_km),
                "eps_cost": float(method_cfg.hyperparams.convergence.get("eps_cost", 1e-4)),
                "flag_conv": int(method_cfg.hyperparams.convergence.flag_conv),
            },
            "problem_config": _jsonable(dict(problem_cfg)),
            "method_config": _jsonable(dict(method_cfg)),
        })
        path = data_dir / f"{approach}__{outcome.name}.npz"
        np.savez_compressed(path, meta=np.asarray(json.dumps(meta)), **arrays)
        print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    algorithm.records.clear()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("approaches", nargs="*", help=f"any of {', '.join(APPROACHES)} (default: all)")
    parser.add_argument("--revolutions", "-n", type=int, default=None)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args(argv)
    unknown = set(args.approaches) - set(APPROACHES)
    if unknown:
        parser.error(f"unknown approach(es) {sorted(unknown)}; choose from {APPROACHES}")
    for approach in args.approaches or APPROACHES:
        solve(approach, args.revolutions, args.data)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Freeing the compiled problems at interpreter exit takes ~10 minutes and
    # does nothing useful once the files are written.
    os._exit(0)
