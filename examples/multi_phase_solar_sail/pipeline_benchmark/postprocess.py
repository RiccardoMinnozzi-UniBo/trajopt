"""Compare the benchmark approaches from the solver records written by solve.py.

    python postprocess.py [--data DIR] [--figures DIR] [--every K]

Reads every ``<approach>__<stage>.npz`` in ``--data`` (default ``results/data``)
and writes the figures, as .jpg, to ``--figures`` (default ``results/figures``).
Nothing computed here is saved apart from the figures: re-run it on a new set
of solver records to redo the whole comparison.

Everything is computed from the saved iterates, independently of the solver:

* the solver's own measures, as it logged them: objective, dynamics defect
  (each method on its own mesh), constraint buffers, the two stopping tests,
  line-search step, trust region, penalty weights, timings;
* the *true-dynamics* measures, by re-integrating each iterate end to end with
  the control model of its own discretisation (first-order hold between nodes
  for shooting, Lagrange through the Radau nodes of each segment for p and
  hp): the drift accumulated by the end of the transfer, and the terminal orbit
  the trajectory actually reaches.  Unlike the solver's defect these are the
  same yardstick for every method.  ``--every K`` re-integrates only every K-th
  iterate (plus the first and last of each stage).

The verdict in the summary table repeats ``solution.validation_report``'s tests
on these numbers: converged, graveyard reached on the propagated orbit,
accumulated drift within the stage's budget, solver defect within tolerance,
interval re-propagation error below 1e-3.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
sys.path.insert(0, str(EXAMPLE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import solar_sail  # noqa: E402
from trajopt.utils.tools import recursive_attrdict  # noqa: E402
from visualization import (_RC, BLUE, ECLIPSE_BAND, GREEN, GRID, MUTED, ORANGE,  # noqa: E402
                           PURPLE, TEXT)

DEFAULT_DATA = HERE / "results" / "data"
DEFAULT_FIGURES = HERE / "results" / "figures"
APPROACH_ORDER = ("p", "ms", "p_ms_hp")
APPROACH_LABEL = {"p": "p (Radau)", "ms": "ms (RK4 shooting)", "p_ms_hp": "p → ms → hp"}
APPROACH_COLOR = {"p": BLUE, "ms": ORANGE, "p_ms_hp": PURPLE}
STAGE_COLOR = {"p": BLUE, "ms": ORANGE, "hp": GREEN}
DAY = 86400.0
TWO_PI = 2.0 * np.pi
_SUBSTEPS = 16

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# loading
# =============================================================================

@dataclass
class Stage:
    approach: str
    name: str
    meta: dict
    data: dict
    history: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    @property
    def phases(self) -> list[dict]:
        return self.meta["phases"]

    @property
    def iterates(self) -> int:
        return int(self.meta["iterates"])

    @property
    def settings(self) -> dict:
        return self.meta["settings"]

    @property
    def timings(self) -> dict:
        return self.meta["timings"]

    @property
    def color(self) -> str:
        return STAGE_COLOR.get(self.name, APPROACH_COLOR.get(self.approach, TEXT))

    def phase(self, index: int, key: str) -> np.ndarray:
        return self.data[f"phase{index:02d}__{key}"]

    def nodes(self, index: int, iterate: int):
        """``(state, longitude, control)`` of one phase at one iterate."""
        return (self.phase(index, "state")[iterate], self.phase(index, "longitude")[iterate],
                self.phase(index, "control")[iterate])


def load(data_dir: Path) -> dict[str, list[Stage]]:
    approaches: dict[str, list[Stage]] = {}
    for path in sorted(data_dir.glob("*__*.npz")):
        with np.load(path, allow_pickle=False) as npz:
            data = {key: npz[key] for key in npz.files}
        meta = json.loads(str(data.pop("meta")))
        approaches.setdefault(meta["approach"], []).append(Stage(meta["approach"], meta["stage"], meta, data))
    for stages in approaches.values():
        stages.sort(key=lambda s: s.meta["stage_index"])
    order = [a for a in APPROACH_ORDER if a in approaches] + sorted(set(approaches) - set(APPROACH_ORDER))
    return {a: approaches[a] for a in order}


# =============================================================================
# true dynamics: re-integration with each discretisation's control model
# =============================================================================

@functools.lru_cache(maxsize=None)
def _params(params_json: str):
    return recursive_attrdict(json.loads(params_json))


def _rk4(f, x, L, h):
    k1 = f(x, L)
    k2 = f(x + 0.5 * h * k1, L + 0.5 * h)
    k3 = f(x + 0.5 * h * k2, L + 0.5 * h)
    k4 = f(x + h * k3, L + h)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@functools.lru_cache(maxsize=None)
def _phase_propagator(params_json: str, mode: str, n_nodes: int, segments: int, n_steps: int):
    """Jitted map ``(x0, node longitudes, node controls) -> state at the phase end``."""
    params = _params(params_json)

    def rhs(u_of):
        return lambda x, L: solar_sail.dynamics(x, jnp.atleast_1d(u_of(L)), L, params, None)

    if mode != "ps":
        @jax.jit
        def run(x0, Ln, un):
            h = (Ln[-1] - Ln[0]) / n_steps
            f = rhs(lambda L: jnp.interp(L, Ln, un))

            def step(x, i):
                return _rk4(f, x, Ln[0] + i * h, h), None
            xf, _ = jax.lax.scan(step, x0, jnp.arange(n_steps))
            return xf
        return run

    p = (n_nodes - 1) // segments
    n_seg = max(n_steps // segments, 1)

    @jax.jit
    def run(x0, Ln, un):
        def segment(x, s):
            nodes = jax.lax.dynamic_slice(Ln, (s * p + 1,), (p,))
            vals = jax.lax.dynamic_slice(un, (s * p + 1,), (p,))
            weights = 1.0 / jnp.prod(nodes[:, None] - nodes[None, :] + jnp.eye(p), axis=1)

            def u_of(L):
                d = L - nodes
                exact = jnp.argmin(jnp.abs(d))
                wd = weights / jnp.where(jnp.abs(d) < 1e-14, 1.0, d)
                return jnp.where(jnp.abs(d[exact]) < 1e-14, vals[exact],
                                 jnp.sum(wd * vals) / jnp.sum(wd))
            f = rhs(u_of)
            L0 = Ln[s * p]
            h = (Ln[(s + 1) * p] - L0) / n_seg

            def step(x, i):
                return _rk4(f, x, L0 + i * h, h), None
            xs, _ = jax.lax.scan(step, x, jnp.arange(n_seg))
            return xs, None
        xf, _ = jax.lax.scan(segment, x0, jnp.arange(segments))
        return xf
    return run


@functools.lru_cache(maxsize=None)
def _interval_propagator(params_json: str):
    """Jitted, vmapped: every node interval from its own node, linear control (as validation)."""
    params = _params(params_json)

    def one(x0, L0, L1, u0, u1):
        h = (L1 - L0) / _SUBSTEPS
        f = lambda x, L: solar_sail.dynamics(  # noqa: E731
            x, jnp.atleast_1d(u0 + (L - L0) / (L1 - L0) * (u1 - u0)), L, params, None)

        def step(x, i):
            return _rk4(f, x, L0 + i * h, h), None
        xf, _ = jax.lax.scan(step, x0, jnp.arange(_SUBSTEPS))
        return xf
    return jax.jit(jax.vmap(one))


def _params_json(phase: dict) -> str:
    return json.dumps(phase["params"], sort_keys=True)


def true_dynamics(stage: Stage, iterate: int) -> dict:
    """Chain the whole transfer from the initial state, never resetting to the nodes."""
    x = None
    for index, phase in enumerate(stage.phases):
        state, L, u = stage.nodes(index, iterate)
        if x is None:
            x = state[0]
        n = state.shape[0]
        run = _phase_propagator(_params_json(phase), phase["discretize"], n, int(phase["segments"]),
                                max(200, 40 * (n - 1)))
        x = np.asarray(run(jnp.asarray(x), jnp.asarray(L), jnp.asarray(u)))
    claimed = stage.nodes(len(stage.phases) - 1, iterate)[0][-1]
    geo = stage.settings["geo_radius_km"]
    e_true = float(np.hypot(x[1], x[2]))
    return {"accumulated_km": float(claimed[0] - x[0]),
            "true_perigee_km": float((x[0] + geo) / (1.0 + e_true)),
            "true_eccentricity": e_true}


def interval_error(stage: Stage, iterate: int) -> float:
    """Largest node-to-node re-propagation error, relative to the state scales."""
    worst = 0.0
    for index, phase in enumerate(stage.phases):
        state, L, u = stage.nodes(index, iterate)
        run = _interval_propagator(_params_json(phase))
        end = np.asarray(run(jnp.asarray(state[:-1]), jnp.asarray(L[:-1]), jnp.asarray(L[1:]),
                             jnp.asarray(u[:-1]), jnp.asarray(u[1:])))
        error = np.abs(end - state[1:]) / np.asarray(phase["state_scales"])
        worst = max(worst, float(np.max(error)))
    return worst


# =============================================================================
# control interpolation on an arbitrary longitude grid
# =============================================================================

def _lagrange(nodes, vals, query):
    weights = 1.0 / np.prod(nodes[:, None] - nodes[None, :] + np.eye(len(nodes)), axis=1)
    d = query[:, None] - nodes[None, :]
    exact = np.abs(d) < 1e-14
    d = np.where(exact, 1.0, d)
    wd = weights[None, :] / d
    out = (wd @ vals) / wd.sum(axis=1)
    hit = exact.any(axis=1)
    out[hit] = vals[np.argmax(exact[hit], axis=1)]
    return out


def control_on_grid(stage: Stage, iterate: int, query: np.ndarray) -> np.ndarray:
    """Steering angle at ``query`` longitudes, NaN in eclipse and outside the transfer."""
    out = np.full(query.shape, np.nan)
    for index, phase in enumerate(stage.phases):
        if not phase["sunlit"]:
            continue
        _, L, u = stage.nodes(index, iterate)
        inside = (query >= L[0]) & (query <= L[-1])
        if not inside.any():
            continue
        q = query[inside]
        if phase["discretize"] != "ps":
            out[inside] = np.interp(q, L, u)
            continue
        segs = int(phase["segments"])
        p = (len(L) - 1) // segs
        values = np.empty_like(q)
        for s in range(segs):
            lo, hi = L[s * p], L[(s + 1) * p]
            sel = (q >= lo) & (q <= hi)
            if sel.any():
                values[sel] = _lagrange(L[s * p + 1:(s + 1) * p + 1], u[s * p + 1:(s + 1) * p + 1], q[sel])
        out[inside] = values
    return out


def eclipse_intervals(stage: Stage, iterate: int = -1) -> list[tuple[float, float]]:
    spans = []
    for index, phase in enumerate(stage.phases):
        if not phase["sunlit"]:
            L = stage.nodes(index, iterate)[1]
            spans.append((L[0] / TWO_PI, L[-1] / TWO_PI))
    return spans


def wrap(angle):
    return (np.asarray(angle) + np.pi) % TWO_PI - np.pi


def gapped_deg(angle):
    deg = np.rad2deg(wrap(angle))
    jump = np.abs(np.diff(deg, prepend=deg[:1])) > 180.0
    return np.where(jump, np.nan, deg)


# =============================================================================
# per-iteration history and per-stage summary
# =============================================================================

def _max_over_phases(stage: Stage, key: str) -> np.ndarray:
    rows = [stage.phase(i, key) for i in range(len(stage.phases)) if f"phase{i:02d}__{key}" in stage.data]
    if not rows:
        return np.full(stage.iterates, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmax(np.vstack(rows), axis=0)


def build_history(stage: Stage, every: int) -> None:
    n = stage.iterates
    last = len(stage.phases) - 1
    h = {"iterate": np.arange(n)}
    h["objective_days"] = stage.phase(last, "state")[:, -1, 5] / DAY

    # solver's dynamics defect, absolute and against the stage's own tolerance
    absolute, ratio = [], []
    for i, phase in enumerate(stage.phases):
        defect = np.abs(stage.phase(i, "defect"))                     # (n, intervals, 6)
        eps = np.asarray(phase["eps_dyn_state"])
        with np.errstate(all="ignore"):
            absolute.append(np.nanmax(defect, axis=(1, 2)))
            ratio.append(np.nanmax(defect / eps, axis=(1, 2)))
    with np.errstate(all="ignore"):
        h["defect_nd"] = np.nanmax(np.vstack(absolute), axis=0)
        h["defect_ratio"] = np.nanmax(np.vstack(ratio), axis=0)
    h["defect_tol_nd"] = float(np.min([np.min(p["eps_dyn_state"]) for p in stage.phases]))

    for key in ("dz", "dcost", "final_state", "nonconvex_inequality", "nonconvex_equality", "dynamics"):
        h[f"chk_{key}"] = _max_over_phases(stage, f"chk__{key}")
    h["dcost_abs"] = h["chk_dcost"] * stage.settings["eps_cost"]
    with np.errstate(all="ignore"):
        h["constraint_ratio"] = np.nanmax(np.vstack([h["chk_final_state"], h["chk_nonconvex_inequality"],
                                                     h["chk_nonconvex_equality"]]), axis=0)
    h["alpha"] = stage.phase(0, "alpha")
    h["s_mult"] = stage.phase(0, "s_mult")
    h["W_dynamics"] = _max_over_phases(stage, "W__dynamics")

    # time per iteration: the logged pieces, scaled so that they add up to the
    # stage's measured SCP time (the rest is Python overhead between them)
    pieces = (np.nan_to_num(stage.phase(0, "solve_time")) + np.nan_to_num(stage.phase(0, "parse_time"))
              + sum(np.nan_to_num(stage.phase(i, "discretization_time")) for i in range(len(stage.phases))))
    pieces = pieces / 1000.0
    pieces[0] = 0.0
    total = float(np.sum(pieces))
    scale = stage.timings.get("solve_s", total) / total if total > 0 else 1.0
    h["iter_time_s"] = np.where(np.arange(n) == 0, np.nan, pieces * scale)
    h["solve_time_s"] = np.nan_to_num(stage.phase(0, "solve_time")) / 1000.0

    # feasibility as the solver sees it
    feasible = (h["defect_ratio"] <= 1.0) & (np.nan_to_num(h["constraint_ratio"]) <= 1.0)
    feasible[0] = False
    h["feasible"] = feasible

    # true dynamics, on a subset of iterates
    picks = sorted(set(range(0, n, max(every, 1))) | {0, n - 1})
    for key in ("accumulated_km", "true_perigee_km", "true_eccentricity"):
        h[key] = np.full(n, np.nan)
    for k in picks:
        for key, value in true_dynamics(stage, k).items():
            h[key][k] = value
    stage.history = h


def build_summary(stage: Stage) -> None:
    h, s, t = stage.history, stage.settings, stage.timings
    n = stage.iterates - 1
    feasible_its = np.flatnonzero(h["feasible"])
    first_feasible = int(feasible_its[0]) if feasible_its.size else None
    repro = interval_error(stage, -1)
    durations = [stage.phase(i, "state")[-1, -1, 5] - stage.phase(i, "state")[-1, 0, 5]
                 for i in range(len(stage.phases))]
    true_ok = (h["true_perigee_km"][-1] >= s["target_perigee_km"] - 0.1
               and h["true_eccentricity"][-1] <= s["target_eccentricity"] + 1e-6)
    validated = bool(stage.meta["status"] == "converged" and true_ok
                     and abs(h["accumulated_km"][-1]) <= s["defect_budget_km"]
                     and h["defect_ratio"][-1] <= 1.0 and repro <= 1e-3 and min(durations) > 0)
    stage.summary = {
        "status": stage.meta["status"],
        "iterations": n,
        "build_s": t.get("build_s", 0.0), "compile_s": t.get("compile_s", 0.0),
        "solve_s": t.get("solve_s", 0.0), "validate_s": t.get("validate_s", 0.0),
        "export_s": t.get("export_s", 0.0), "total_s": t.get("total_s", 0.0),
        "s_per_iter": t.get("solve_s", 0.0) / n if n else np.nan,
        "first_feasible": first_feasible,
        "stagnation_its": (n - first_feasible) if first_feasible is not None else None,
        "elapsed_days": float(h["objective_days"][-1]),
        "true_perigee_alt_km": float(h["true_perigee_km"][-1] - s["geo_radius_km"]),
        "true_eccentricity": float(h["true_eccentricity"][-1]),
        "accumulated_km": float(h["accumulated_km"][-1]),
        "budget_km": s["defect_budget_km"],
        "defect_ratio": float(h["defect_ratio"][-1]),
        "repro_error": repro,
        "mean_alpha": float(np.nanmean(h["alpha"][1:])) if n else np.nan,
        "validated": validated,
    }


def approach_summary(stages: list[Stage]) -> dict:
    final = stages[-1].summary
    out = dict(final)
    for key in ("iterations", "build_s", "compile_s", "solve_s", "validate_s", "export_s", "total_s"):
        out[key] = sum(st.summary[key] for st in stages)
    out["s_per_iter"] = out["solve_s"] / out["iterations"] if out["iterations"] else np.nan
    # feasible iterate counted along the whole approach
    offset, first = 0, None
    for st in stages:
        if st.summary["first_feasible"] is not None and st is stages[-1]:
            first = offset + st.summary["first_feasible"]
        offset += st.summary["iterations"]
    out["first_feasible"] = first
    out["stagnation_its"] = final["stagnation_its"]
    out["mean_alpha"] = float(np.nanmean(np.concatenate([st.history["alpha"][1:] for st in stages])))
    return out


# =============================================================================
# concatenated axes (a flow is drawn as one run, its stages one after another)
# =============================================================================

def concat(stages: list[Stage], key: str):
    """``(cumulative iteration, cumulative wall time [s], values, stage spans)``."""
    xs, ts, ys, spans = [], [], [], []
    it0, t0 = 0, 0.0
    for st in stages:
        h, t = st.history, st.timings
        n = st.iterates
        start_t = t0 + t.get("build_s", 0.0) + t.get("compile_s", 0.0)
        times = start_t + np.concatenate([[0.0], np.cumsum(np.nan_to_num(h["iter_time_s"][1:]))])
        xs.append(it0 + np.arange(n))
        ts.append(times)
        ys.append(np.asarray(h[key], dtype=float) if np.ndim(h[key]) else np.full(n, h[key]))
        spans.append((st, it0, it0 + n - 1, t0, t0 + t.get("total_s", times[-1] - t0)))
        it0 += n - 1
        t0 += t.get("total_s", times[-1] - t0)
    return xs, ts, ys, spans


def shade_stages(axis, spans, use_time=False, label=False) -> None:
    if len(spans) < 2:
        return
    for st, i0, i1, t0, t1 in spans:
        lo, hi = (t0 / 60.0, t1 / 60.0) if use_time else (i0, i1)
        axis.axvspan(lo, hi, color=st.color, alpha=0.07, lw=0)
        if label:
            axis.text(0.5 * (lo + hi), 0.97, f"{st.name} stage", transform=axis.get_xaxis_transform(),
                      ha="center", va="top", fontsize=8, color=st.color, weight="bold")


# =============================================================================
# figures
# =============================================================================

def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, pil_kwargs={"quality": 92})
    plt.close(figure)
    print(f"  {path}")
    return path


def fig_summary_table(approaches, path: Path) -> Path:
    rows, colors = [], []
    best = min(approach_summary(st)["elapsed_days"] for st in approaches.values())

    def fmt(summary, label, disc, color):
        ff = summary["first_feasible"]
        rows.append([
            label, disc, summary["status"], f"{summary['iterations']}",
            f"{summary['total_s']:.0f}", f"{summary['solve_s']:.0f}", f"{summary['s_per_iter']:.1f}",
            "-" if ff is None else f"{ff}",
            "-" if summary["stagnation_its"] is None else f"{summary['stagnation_its']}",
            f"{summary['elapsed_days']:.4f}", f"{(summary['elapsed_days'] - best) * 24 * 60:+.1f}",
            f"{summary['true_perigee_alt_km']:+.2f}", f"{summary['true_eccentricity']:.6f}",
            f"{summary['accumulated_km']:+.3f} / {summary['budget_km']:g}",
            f"{summary['defect_ratio']:.2g}", f"{summary['repro_error']:.1e}",
            f"{summary['mean_alpha']:.2f}", "yes" if summary["validated"] else "NO",
        ])
        colors.append(color)

    for name, stages in approaches.items():
        disc = " → ".join(st.meta["discretisation"] for st in stages)
        fmt(approach_summary(stages), APPROACH_LABEL.get(name, name), disc, APPROACH_COLOR.get(name, TEXT))
        if len(stages) > 1:
            for st in stages:
                fmt(st.summary, f"   stage {st.name}", st.meta["discretisation"], st.color)

    header = ["approach", "mesh", "status", "iter.", "wall\n[s]", "SCP\n[s]", "s /\niter.",
              "1st feas.\niter.", "iter. after\nfeasible", "elapsed\n[d]", "gap\n[min]",
              "true r_p - r_GEO\n[km]", "true e", "acc. drift / budget\n[km]",
              "max defect\n/ tol.", "re-prop.\nerror", "mean\nstep α", "valid"]
    fig, ax = plt.subplots(figsize=(19, 0.5 + 0.42 * (len(rows) + 2)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#EEF1F5")
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.5)
        else:
            if c == 0:
                cell.set_text_props(color=colors[r - 1], weight="bold", ha="left")
                cell._loc = "left"
            if rows[r - 1][0].startswith("   "):
                cell.set_facecolor("#FAFAFA")
            if c == len(header) - 1:
                cell.set_text_props(color=GREEN if rows[r - 1][-1] == "yes" else "#C0392B", weight="bold")
    table.auto_set_column_width(list(range(len(header))))
    revolutions = next(iter(approaches.values()))[0].meta["revolutions"]
    ax.set_title(f"Benchmark summary, {revolutions} revolutions  ·  stage rows judged against "
                 f"their own tolerances  ·  wall = build + compile + SCP + validation",
                 fontsize=10, color=MUTED, pad=4)
    return _save(fig, path)


def _plot_history(ax, approaches, key, use_time=False, log=False, marker=None, label_stages=False):
    for name, stages in approaches.items():
        xs, ts, ys, spans = concat(stages, key)
        color = APPROACH_COLOR.get(name, TEXT)
        # a flow is dashed, so that it stays visible where it retraces a single method
        ls = "--" if len(stages) > 1 else "-"
        for i, (x, t, y) in enumerate(zip(xs, ts, ys)):
            xx = t / 60.0 if use_time else x
            yy = np.abs(y) if log else y
            if marker:
                good = np.isfinite(yy)
                ax.plot(xx[good], yy[good], color=color, lw=1.1, ls=ls, marker=marker, ms=2.5,
                        label=APPROACH_LABEL.get(name, name) if i == 0 else None)
            else:
                ax.plot(xx, yy, color=color, lw=1.2, ls=ls, label=APPROACH_LABEL.get(name, name) if i == 0 else None)
        if len(stages) > 1:
            shade_stages(ax, spans, use_time, label=label_stages)
            for _, i0, _, t0, _ in spans[1:]:
                ax.axvline(t0 / 60.0 if use_time else i0, color=MUTED, lw=0.6, ls=":")
    if log:
        ax.set_yscale("log")


def _tolerance_steps(ax, approaches, key, use_time=False, scale=1.0):
    """Each stage's own tolerance as a dashed step line."""
    for name, stages in approaches.items():
        _, ts, _, spans = concat(stages, "objective_days")
        for (st, i0, i1, t0, t1), t in zip(spans, ts):
            value = (st.history[key] if key in st.history else st.settings[key]) * scale
            lo, hi = (t[0] / 60.0, t[-1] / 60.0) if use_time else (i0, i1)
            ax.hlines(value, lo, hi, colors=APPROACH_COLOR.get(name, TEXT), linestyles="--", lw=0.8, alpha=0.8)


def fig_history(approaches, path: Path, use_time: bool) -> Path:
    xlabel = "cumulative wall time [min]" if use_time else "cumulative SCP iteration"
    if use_time:
        fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
        panels = ["objective", "defect", "drift", "dcost"]
    else:
        fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True)
        panels = ["objective", "defect", "drift", "constraints", "dcost", "alpha", "weight", "time"]
    for ax, panel in zip(axes.ravel(), panels):
        top = ax in axes.ravel()[:2]
        if panel == "objective":
            _plot_history(ax, approaches, "objective_days", use_time, label_stages=top)
            ax.set_ylabel("elapsed time of the iterate [d]")
            ax.set_title("objective")
            finals = [st[-1].history["objective_days"][-1] for st in approaches.values()]
            ax.set_ylim(min(finals) - 0.25, max(finals) + 1.5)
        elif panel == "defect":
            _plot_history(ax, approaches, "defect_nd", use_time, log=True, label_stages=top)
            _tolerance_steps(ax, approaches, "defect_tol_nd", use_time)
            ax.set_ylabel("max dynamics defect (solver, nondim.)")
            ax.set_title("solver defect on its own mesh (dashed: stage tolerance)")
        elif panel == "drift":
            _plot_history(ax, approaches, "accumulated_km", use_time, log=True, marker="o")
            _tolerance_steps(ax, approaches, "defect_budget_km", use_time)
            ax.set_ylabel("|accumulated drift| [km]")
            ax.set_title("true dynamics: drift of the re-integrated transfer (dashed: budget)")
        elif panel == "constraints":
            _plot_history(ax, approaches, "constraint_ratio", use_time, log=True)
            ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
            ax.set_ylabel("max buffer / tolerance")
            ax.set_title("terminal, terminator and path constraints")
        elif panel == "dcost":
            _plot_history(ax, approaches, "dcost_abs", use_time, log=True)
            _tolerance_steps(ax, approaches, "eps_cost", use_time)
            ax.set_ylabel("|change in cost|")
            ax.set_title("cost stopping test (dashed: stage eps_cost)")
        elif panel == "alpha":
            _plot_history(ax, approaches, "alpha", use_time, log=True)
            ax.set_ylabel("line-search step α")
            ax.set_title("accepted step fraction")
        elif panel == "weight":
            _plot_history(ax, approaches, "W_dynamics", use_time, log=True)
            ax.set_ylabel("max dynamics penalty weight")
            ax.set_title("dynamics penalty weight (resets at a hand-off)")
        elif panel == "time":
            _plot_history(ax, approaches, "iter_time_s", use_time)
            ax.set_ylabel("wall time per iteration [s]")
            ax.set_title("cost of an iteration")
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.suptitle("Iteration history" + (" against wall time" if use_time else ""), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(1.0, 1.0))
    return _save(fig, path)


def _reference(approaches) -> Stage:
    finals = [st[-1] for st in approaches.values()]
    valid = [f for f in finals if f.summary["validated"]] or finals
    return min(valid, key=lambda f: f.summary["elapsed_days"])


def _control_figure(entries, reference: Stage, title: str, path: Path) -> Path:
    """entries: list of (label, color, stage, iterate)."""
    L_ref = np.concatenate([reference.nodes(i, -1)[1] for i in range(len(reference.phases))])
    lo = max(e[2].nodes(0, e[3])[1][0] for e in entries)
    hi = min(e[2].nodes(len(e[2].phases) - 1, e[3])[1][-1] for e in entries)
    grid = np.linspace(lo, hi, 60000)
    rev = grid / TWO_PI
    ref_u = control_on_grid(reference, -1, grid)
    eclipses = eclipse_intervals(reference)
    zoom_lo = max(lo, L_ref[-1] - 2.0 * TWO_PI) / TWO_PI

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    for ax in axes:
        for a, b in eclipses:
            ax.axvspan(a, b, color=ECLIPSE_BAND, lw=0)
    for label, color, stage, iterate in entries:
        u = control_on_grid(stage, iterate, grid)
        axes[0].plot(rev, gapped_deg(u), color=color, lw=0.7, label=label)
        if stage is not reference or iterate not in (-1, stage.iterates - 1):
            diff = np.rad2deg(np.abs(wrap(u - ref_u)))
            axes[1].plot(rev, diff, color=color, lw=0.7, label=label)
        sel = rev >= zoom_lo
        axes[2].plot(rev[sel], gapped_deg(u[sel]), color=color, lw=1.1, label=label)
    axes[0].set_ylabel("steering angle [deg]")
    axes[0].set_title("steering angle over the transfer (eclipses shaded)")
    axes[0].legend(loc="lower left", ncol=len(entries))
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-4, 360)
    axes[1].set_ylabel("|difference| [deg]")
    axes[1].set_title(f"difference from the reference ({_entry_label(entries, reference)})")
    if axes[1].lines:
        axes[1].legend(loc="lower left")
    axes[2].set_xlim(zoom_lo, hi / TWO_PI)
    axes[2].set_ylabel("steering angle [deg]")
    axes[2].set_title("last two revolutions (terminator and terminal constraints active)")
    for ax in axes:
        ax.set_xlabel("revolutions (true longitude / 2π)")
    for ax in axes[:2]:
        ax.set_xlim(lo / TWO_PI, hi / TWO_PI)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def _entry_label(entries, reference):
    for label, _, stage, iterate in entries:
        if stage is reference:
            return label
    return f"{APPROACH_LABEL.get(reference.approach, reference.approach)}"


def fig_controls_approaches(approaches, path: Path) -> Path:
    reference = _reference(approaches)
    entries = [(APPROACH_LABEL.get(name, name), APPROACH_COLOR.get(name, TEXT), stages[-1], -1)
               for name, stages in approaches.items()]
    return _control_figure(entries, reference, "Optimal control: the three approaches", path)


def fig_controls_pipeline(stages: list[Stage], path: Path) -> Path:
    entries = [(f"after stage {st.name} ({st.meta['discretisation']})", st.color, st, -1) for st in stages]
    return _control_figure(entries, stages[-1], "Optimal control along the p → ms → hp pipeline", path)


def fig_control_evolution(approaches, path: Path, columns: int = 1500) -> Path:
    reference = _reference(approaches)
    L_ref = np.concatenate([reference.nodes(i, -1)[1] for i in range(len(reference.phases))])
    fig, axes = plt.subplots(1, len(approaches), figsize=(5.2 * len(approaches), 6), squeeze=False)
    axes = axes[0]
    image = None
    for ax, (name, stages) in zip(axes, approaches.items()):
        final = stages[-1]
        lo = L_ref[0]
        hi = min(L_ref[-1], final.nodes(len(final.phases) - 1, -1)[1][-1])
        grid = np.linspace(lo, hi, columns)
        target = control_on_grid(final, -1, grid)
        rows, boundaries = [], []
        for st in stages:
            for k in range(st.iterates):     # iterate 0 of a stage is its (re-meshed) guess
                rows.append(np.rad2deg(np.abs(wrap(control_on_grid(st, k, grid) - target))))
            boundaries.append(len(rows))
        data = np.clip(np.vstack(rows), 1e-3, 180.0)
        image = ax.imshow(data, aspect="auto", origin="lower", cmap="magma_r",
                          norm=matplotlib.colors.LogNorm(1e-3, 180.0),
                          extent=(grid[0] / TWO_PI, grid[-1] / TWO_PI, 0, data.shape[0]))
        for st, b in zip(stages[:-1], boundaries[:-1]):
            ax.axhline(b, color="white", lw=1.0, ls="--")
        if len(stages) > 1:
            start = 0
            for st, b in zip(stages, boundaries):
                ax.text(grid[-1] / TWO_PI, 0.5 * (start + b), f" {st.name}", color=st.color,
                        va="center", ha="left", fontsize=9, weight="bold")
                start = b
        ax.set_title(APPROACH_LABEL.get(name, name))
        ax.set_xlabel("revolutions")
        ax.grid(False)
    axes[0].set_ylabel("iterate (all stages, in order)")
    fig.colorbar(image, ax=list(axes), label="|steering angle - final| [deg]  (eclipses blank)", shrink=0.85)
    fig.suptitle("How the control converges: distance of every iterate from the approach's final control",
                 fontsize=12)
    return _save(fig, path)


def fig_handoffs(stages: list[Stage], path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    names = [st.name for st in stages]

    # end of one stage against the start of the next, true dynamics
    ax = axes[0, 0]
    labels, end_vals, start_vals = [], [], []
    for prev, nxt in zip(stages[:-1], stages[1:]):
        labels.append(f"{prev.name} → {nxt.name}")
        end_vals.append(abs(prev.history["accumulated_km"][-1]))
        start_vals.append(abs(nxt.history["accumulated_km"][0]))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, end_vals, 0.36, color=[s.color for s in stages[:-1]], label="end of stage")
    ax.bar(x + 0.18, start_vals, 0.36, color=[s.color for s in stages[1:]], alpha=0.55,
           hatch="//", label="start of next (re-meshed guess)")
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_ylabel("|accumulated drift| [km]")
    ax.set_title("hand-off: true-dynamics drift before and after re-meshing")
    ax.legend(loc="upper right")

    # the objective along the whole flow: each stage's guess and its result
    ax = axes[0, 1]
    ref = stages[-1].history["objective_days"][-1]
    ticks, names_x = [], []
    for i, st in enumerate(stages):
        x0, x1 = 2 * i, 2 * i + 1
        y0, y1 = (np.array([st.history["objective_days"][0], st.history["objective_days"][-1]]) - ref) * 24 * 60
        ax.plot([x0, x1], [y0, y1], color=st.color, marker="o", lw=1.6, label=f"stage {st.name}")
        if i:
            ax.plot([x0 - 1, x0], [prev_y, y0], color=MUTED, lw=1.0, ls=":")
        prev_y = y1
        ticks += [x0, x1]
        names_x += [f"{st.name}\nguess", f"{st.name}\nresult"]
    ax.set_xticks(ticks, names_x)
    ax.axhline(0.0, color=MUTED, lw=0.6)
    ax.set_ylabel("elapsed time - final [min]")
    ax.set_title("objective through the flow (dotted: hand-off and re-meshing)")
    ax.legend()

    # how much each stage changes the solution, phase by phase
    ax = axes[1, 0]
    ax2 = axes[1, 1]
    for number, (prev, nxt) in enumerate(zip(stages[:-1], stages[1:])):
        style = dict(marker="os^"[number % 3], ms=3.5, lw=1.0, ls=["-", "--", ":"][number % 3],
                     color=nxt.color, label=f"{prev.name} → {nxt.name}")
        n_ph = len(nxt.phases)
        d_ctrl, d_state = np.full(n_ph, np.nan), np.full(n_ph, np.nan)
        for i in range(n_ph):
            sa, La, _ = prev.nodes(i, -1)
            sb, Lb, _ = nxt.nodes(i, -1)
            d_state[i] = np.max(np.abs(sb[-1] - sa[-1]) / np.asarray(nxt.phases[i]["state_scales"]))
            if nxt.phases[i]["sunlit"]:
                lo, hi = max(La[0], Lb[0]), min(La[-1], Lb[-1])
                q = np.linspace(lo, hi, 400)
                ua = control_on_grid(prev, -1, q)
                ub = control_on_grid(nxt, -1, q)
                d_ctrl[i] = np.nanmax(np.rad2deg(np.abs(wrap(ub - ua))))
        phases = np.arange(1, n_ph + 1)
        lit = np.isfinite(d_ctrl)
        ax.semilogy(phases[lit], d_ctrl[lit], **style)
        ax2.semilogy(phases, d_state, **style)
    ax.set_xlabel("phase")
    ax.set_ylabel("max |Δ steering| [deg]")
    ax.set_title("change of the control made by each stage (sunlit phases)")
    ax.legend()
    ax2.set_xlabel("phase")
    ax2.set_ylabel("max |Δ state| at phase end / state scale")
    ax2.set_title("change of the trajectory made by each stage")
    ax2.legend()
    fig.suptitle(f"Pipeline hand-offs ({' → '.join(names)})", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def fig_pipeline_budget(stages: list[Stage], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    parts = [("build_s", "build", "#B8C4D6"), ("compile_s", "JAX compile", "#8DA0BF"),
             ("solve_s", "SCP iterations", None), ("validate_s", "validation", "#E3E3E3"),
             ("export_s", "export", "#CCCCCC")]
    x = np.arange(len(stages))
    bottom = np.zeros(len(stages))
    for key, label, color in parts:
        vals = np.array([st.summary[key] for st in stages]) / 60.0
        axes[0].bar(x, vals, 0.6, bottom=bottom, label=label,
                    color=[st.color for st in stages] if color is None else color)
        bottom += vals
    for xi, total in zip(x, bottom):
        axes[0].text(xi, total, f"{total:.1f} min", ha="center", va="bottom", fontsize=8.5)
    axes[0].set_xticks(x, [f"{st.name}\n{st.meta['discretisation']}" for st in stages])
    axes[0].set_ylabel("wall time [min]")
    axes[0].set_title("where the pipeline's time goes")
    axes[0].legend(loc="upper left", fontsize=8)

    its = np.array([st.summary["iterations"] for st in stages])
    feas = [st.summary["first_feasible"] for st in stages]
    axes[1].bar(x, its, 0.6, color=[st.color for st in stages])
    for xi, n, f in zip(x, its, feas):
        if f is not None:
            axes[1].plot([xi - 0.3, xi + 0.3], [f, f], color=TEXT, lw=1.2)
        axes[1].text(xi, n, f"{n}", ha="center", va="bottom", fontsize=8.5)
    axes[1].set_xticks(x, [st.name for st in stages])
    axes[1].set_ylabel("SCP iterations")
    axes[1].set_title("iterations per stage (bar) and first feasible iterate (line)")
    fig.tight_layout()
    return _save(fig, path)


def fig_tradeoff(approaches, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    best = min(approach_summary(st)["elapsed_days"] for st in approaches.values())
    for name, stages in approaches.items():
        s = approach_summary(stages)
        color = APPROACH_COLOR.get(name, TEXT)
        ax.scatter(s["total_s"] / 60.0, (s["elapsed_days"] - best) * 24 * 60, s=90, color=color,
                   facecolors=color if s["validated"] else "none", linewidths=1.6, zorder=3,
                   label=f"{APPROACH_LABEL.get(name, name)} ({s['iterations']} it.)")
    ax.set_xlabel("total wall time [min]")
    ax.set_ylabel("elapsed time - best [min]")
    ax.set_title("cost against quality (filled: passes validation)")
    ax.legend()
    ax.margins(0.25)
    return _save(fig, path)


# =============================================================================

def print_table(approaches) -> None:
    print(f"\n{'approach/stage':<22}{'status':>14}{'it':>6}{'wall s':>9}{'elapsed d':>11}"
          f"{'r_p-r_GEO':>11}{'e':>10}{'drift km':>10}{'def/tol':>9}{'valid':>7}")
    for name, stages in approaches.items():
        rows = [(name, approach_summary(stages))]
        if len(stages) > 1:
            rows += [(f"  {st.name}", st.summary) for st in stages]
        for label, s in rows:
            print(f"{label:<22}{s['status']:>14}{s['iterations']:>6}{s['total_s']:>9.0f}"
                  f"{s['elapsed_days']:>11.4f}{s['true_perigee_alt_km']:>+11.2f}{s['true_eccentricity']:>10.6f}"
                  f"{s['accumulated_km']:>+10.3f}{s['defect_ratio']:>9.2g}{'yes' if s['validated'] else 'NO':>7}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--every", type=int, default=1,
                        help="re-integrate the true dynamics every K-th iterate (default: all)")
    args = parser.parse_args(argv)

    # early iterates carry no defect or checks yet: their all-NaN reductions are expected
    warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
    approaches = load(args.data)
    if not approaches:
        raise SystemExit(f"no solver records in {args.data}")
    for stages in approaches.values():
        for st in stages:
            print(f"processing {st.approach}/{st.name} ({st.iterates - 1} iterations)", flush=True)
            build_history(st, args.every)
            build_summary(st)
    print_table(approaches)

    plt.rcParams.update(_RC)
    out = args.figures
    print("\nfigures")
    fig_summary_table(approaches, out / "summary_table.jpg")
    fig_history(approaches, out / "history_iterations.jpg", use_time=False)
    fig_history(approaches, out / "history_walltime.jpg", use_time=True)
    fig_controls_approaches(approaches, out / "controls_approaches.jpg")
    fig_control_evolution(approaches, out / "control_evolution.jpg")
    fig_tradeoff(approaches, out / "tradeoff.jpg")
    for name, stages in approaches.items():
        if len(stages) > 1:
            fig_controls_pipeline(stages, out / f"controls_{name}_stages.jpg")
            fig_handoffs(stages, out / f"handoffs_{name}.jpg")
            fig_pipeline_budget(stages, out / f"budget_{name}.jpg")


if __name__ == "__main__":
    main()
