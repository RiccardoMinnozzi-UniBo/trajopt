"""Algorithms for the solar-sail problem: one method, or a flow of methods.

An *algorithm file* is one of two things, and everything downstream -- the
revolution search, validation, figures, ``compare.py`` -- treats them alike:

* a **single-method config**, e.g. ``algorithms/ms.yaml``: the problem
  (inherited from ``problem.yaml``) plus a mesh and a ``method:``;

* a **flow**, e.g. ``algorithms/p_to_ms.yaml``: an ordered list of such configs,
  each solved in turn and optionally warm-started from an earlier one's
  solution.  The schema is trajopt's ``flow.stages`` (see ``examples/algflow``),
  with inline overrides added so that a variant does not need a file of its own::

      flow:
        stages:
          - name: p
            config: p.yaml               # relative to this file
            problem: {sunlit_nodes: 16}  # optional, this stage only
            method:  {hyperparams.convergence.iter_max: 40}   # optional, may `inherit`
            allow_nonconverged: true     # optional: hand on an unconverged iterate
          - name: ms
            config: ms.yaml
            initial_guess:
              from: p                    # an earlier stage
              phases: [...]              # optional subset; the rest use the default guess
      problem: {...}                     # optional, every stage
      method:  {...}                     # optional, every stage
      output:  {...}                     # optional

Overrides apply in increasing precedence: the stage's config file, the flow's
``problem``/``method``, the stage's inline ``problem``/``method``, and finally
the run-time overrides passed to :meth:`Algorithm.run` (``--set`` on the
command line).  A single-method config is run as a one-stage flow whose stage is
named after the file.

The phase structure (and so the revolution count) must be the same in every
stage, since a guess is handed on phase by phase.  Only one stage's problem is
alive at a time: each owns a compiled CVXPY problem that can run to tens of
gigabytes, so a stage's solution is exported as a mesh-independent guess and
the stage released before the next is built.
"""

from __future__ import annotations

import copy
import gc
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from trajopt.utils import config_loader, tools
from trajopt.utils.tools import AttrDict, recursive_attrdict

from problem import SolarSailProblem
from solution import SolarSailSolution, extract_solution, validation_report

_STAGE_FIELDS = {"name", "config", "initial_guess", "allow_nonconverged", "problem", "method"}
_RESERVED_NAMES = {"problem", "method", "output", "flow"}


@dataclass(frozen=True)
class Stage:
    """One solve of the flow, as written in the algorithm file."""

    name: str
    config_path: Path
    problem: dict = field(default_factory=dict)
    method: dict = field(default_factory=dict)
    seed_from: str | None = None
    seed_phases: tuple[str, ...] | None = None
    allow_nonconverged: bool = False


@dataclass
class StageRun:
    """What one stage did."""

    name: str
    config_path: Path
    seed_from: str | None = None
    status: str = "failed"              # converged | not_converged | failed
    reason: str = ""
    iterations: int = 0
    timings: dict = field(default_factory=dict)
    solution: SolarSailSolution | None = None
    report: dict | None = None

    @property
    def converged(self) -> bool:
        return self.status == "converged"

    @property
    def feasible(self) -> bool:
        return bool(self.report is not None and self.report["numerically_feasible"])

    def summary(self) -> str:
        seed = f" <- {self.seed_from}" if self.seed_from else ""
        head = f"{self.name}{seed}"
        clock = (f"{self.timings.get('total_s', 0.0):6.0f} s "
                 f"(compile {self.timings.get('compile_s', 0.0):4.0f}, "
                 f"SCP {self.timings.get('solve_s', 0.0):4.0f})")
        if self.report is None:
            return f"{head:<22} {self.status:<13} {self.iterations:4d} it  {clock}  {self.reason}"
        r = self.report
        geo = self.solution.geo_radius_km
        verdict = "FEASIBLE" if self.feasible else "infeasible"
        return (f"{head:<22} {self.status:<13} {self.iterations:4d} it  {clock}  "
                f"{verdict:<10} {r['elapsed_days']:8.4f} d  "
                f"r_p-r_GEO={r['true_final_perigee_km'] - geo:+8.2f} km  "
                f"e={r['true_final_eccentricity']:.6f}  "
                f"acc={r['accumulated_defect_km']:+7.3f} km")


@dataclass
class Run:
    """The outcome of running an algorithm once at a fixed revolution count."""

    algorithm: str
    revolutions: int
    stages: list[StageRun] = field(default_factory=list)
    completed: bool = False
    failed_stage: str | None = None
    # the final stage's problem, if it was asked to be kept; owns the compiled
    # subproblem, so drop it as soon as it is no longer needed
    problem: SolarSailProblem | None = None

    @property
    def final(self) -> StageRun | None:
        return self.stages[-1] if self.stages else None

    @property
    def solution(self) -> SolarSailSolution | None:
        return None if self.final is None else self.final.solution

    @property
    def report(self) -> dict | None:
        return None if self.final is None else self.final.report

    @property
    def feasible(self) -> bool:
        """The whole flow ran and its last stage validates."""
        return self.completed and self.final is not None and self.final.feasible

    @property
    def iterations(self) -> int:
        return sum(stage.iterations for stage in self.stages)

    @property
    def total_s(self) -> float:
        return sum(stage.timings.get("total_s", 0.0) for stage in self.stages)

    def print_summary(self) -> None:
        print(f"\n{self.algorithm}  N = {self.revolutions}  "
              f"({len(self.stages)} stage{'s' if len(self.stages) != 1 else ''}, "
              f"{self.iterations} iterations, {self.total_s:.0f} s)")
        for stage in self.stages:
            print(f"  {stage.summary()}")
        if self.failed_stage is not None:
            print(f"  stopped at stage '{self.failed_stage}'")


class Algorithm:
    """An algorithm file, parsed and checked; :meth:`run` solves it."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.name = self.path.stem
        raw = config_loader.load_trajopt_config(str(self.path))
        self.output = recursive_attrdict(dict(raw.get("output", {}) or {}))

        if "flow" not in raw:
            # a plain config is a one-stage flow
            self.is_flow = False
            self.problem_overrides, self.method_overrides = {}, {}
            self.stages = [Stage(self.name, self.path)]
            return

        self.is_flow = True
        self.problem_overrides = self._resolve(raw.get("problem", {}))
        self.method_overrides = self._resolve(raw.get("method", {}))
        flow = raw["flow"]
        if not isinstance(flow, dict) or set(flow) != {"stages"}:
            raise ValueError(f"{self.path.name}: expected flow.stages")
        stages = flow["stages"]
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"{self.path.name}: flow.stages must be a nonempty list")

        self.stages = []
        for spec in stages:
            self.stages.append(self._stage(spec, {s.name for s in self.stages}))
        for stage in self.stages:
            if stage.seed_from is not None and self._dynamics_w_init(stage) < 10.0:
                print(f"warning: {self.path.name}: stage '{stage.name}' is seeded from "
                      f"'{stage.seed_from}' but starts its dynamics penalty at "
                      f"{self._dynamics_w_init(stage):g}; it may walk away from the guess to "
                      f"a worse optimum.  Add `method: {{inherit: warm_start.yaml}}` to the "
                      f"stage (see algorithms/warm_start.yaml).", file=sys.stderr)

    def _resolve(self, block) -> dict:
        """An inline override block, with any `inherit` resolved from this file."""
        if not block:
            return {}
        if not isinstance(block, dict):
            raise ValueError(f"{self.path.name}: override blocks must be mappings")
        return dict(config_loader._resolve_inheritance(dict(block), _source=str(self.path)))

    def _stage(self, spec, earlier: set[str]) -> Stage:
        where = f"{self.path.name}: stage {spec.get('name', '?')!r}" if isinstance(spec, dict) else self.path.name
        if not isinstance(spec, dict) or set(spec) - _STAGE_FIELDS:
            raise ValueError(f"{where}: stage fields are {sorted(_STAGE_FIELDS)}")
        name = spec.get("name")
        if not isinstance(name, str) or not name.strip() or name in earlier:
            raise ValueError(f"{where}: expected a unique stage name")
        if name in _RESERVED_NAMES or "." in name:
            raise ValueError(f"{where}: stage names cannot be {sorted(_RESERVED_NAMES)} or contain '.'")
        config = spec.get("config")
        if not isinstance(config, str) or not config:
            raise ValueError(f"{where}: needs a config path")
        config_path = (self.path.parent / config).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"{where}: config not found: {config_path}")
        if "flow" in config_loader.load_trajopt_config(str(config_path)):
            raise ValueError(f"{where}: a stage config cannot itself be a flow")
        allow = spec.get("allow_nonconverged", False)
        if type(allow) is not bool:
            raise ValueError(f"{where}: allow_nonconverged must be true or false")

        seed_from = seed_phases = None
        seed = spec.get("initial_guess")
        if seed is not None:
            if not isinstance(seed, dict) or set(seed) - {"from", "phases"} or "from" not in seed:
                raise ValueError(f"{where}: initial_guess needs `from` and optional `phases`")
            seed_from = seed["from"]
            if seed_from not in earlier:
                raise ValueError(f"{where}: initial_guess.from must name an earlier stage")
            if seed.get("phases") is not None:
                seed_phases = tuple(seed["phases"])

        return Stage(name, config_path, self._resolve(spec.get("problem")),
                     self._resolve(spec.get("method")), seed_from, seed_phases, allow)

    # ------------------------------------------------------------------ settings

    def stage_overrides(self, stage: Stage, overrides: dict | None = None) -> tuple[dict, dict]:
        """``(problem, method)`` overrides for one stage, in precedence order."""
        overrides = overrides or {}
        per_stage = (overrides.get("stages") or {}).get(stage.name, {})
        problem, method = {}, {}
        for layer_problem, layer_method in (
            (self.problem_overrides, self.method_overrides),
            (stage.problem, stage.method),
            (overrides.get("problem") or {}, overrides.get("method") or {}),
            (per_stage.get("problem") or {}, per_stage.get("method") or {}),
        ):
            problem = tools.deep_merge(problem, copy.deepcopy(dict(layer_problem)))
            method = tools.deep_merge(method, copy.deepcopy(dict(layer_method)))
        return problem, method

    def settings(self, overrides: dict | None = None) -> AttrDict:
        """Problem settings of the final stage, which the search and output read."""
        stage = self.stages[-1]
        problem, _ = self.stage_overrides(stage, overrides)
        settings = config_loader.load_trajopt_config(str(stage.config_path)).problem
        return recursive_attrdict(tools.deep_merge(dict(settings), problem))

    def output_settings(self) -> AttrDict:
        """Where and what to write; the directory defaults to results/<file name>.

        A flow takes the final stage's settings but never its directory, so that
        a flow does not overwrite the results of the algorithm it ends with.
        """
        if self.is_flow:
            final = config_loader.load_trajopt_config(str(self.stages[-1].config_path))
            base = dict(final.get("output", {}) or {})
            base.pop("directory", None)
            output = tools.deep_merge(base, dict(self.output))
        else:
            output = dict(self.output)
        output.setdefault("directory", f"results/{self.name}")
        return recursive_attrdict(output)

    def stage_config(self, stage: Stage, overrides: dict | None = None) -> tuple[AttrDict, AttrDict]:
        """The ``(problem, method)`` a stage will actually run with."""
        config = config_loader.load_trajopt_config(str(stage.config_path))
        problem_over, method_over = self.stage_overrides(stage, overrides)
        return (recursive_attrdict(tools.deep_merge(dict(config.problem), problem_over)),
                recursive_attrdict(tools.deep_merge(dict(config.method), method_over)))

    def _dynamics_w_init(self, stage: Stage) -> float:
        _, method = self.stage_config(stage)
        penalties = method.hyperparams.penalties
        dynamics = penalties.get("dynamics", {}).get("W", {}).get("init")
        return float(dynamics if dynamics is not None else penalties.default.W.init)

    def stage_label(self, stage: Stage, overrides: dict | None = None) -> str:
        """Discretisation and mesh of a stage, e.g. 'ms 12/3', 'ps-p 24/4', 'ps-hp4 25/4'."""
        problem, method = self.stage_config(stage, overrides)
        mode = method.hyperparams.discretize.mode
        if mode == "ps":
            segments = int(problem.get("sunlit_hp_segments", 1))
            mode = "ps-p" if segments == 1 else f"ps-hp{segments}"
        return f"{mode} {problem.get('sunlit_nodes', '?')}/{problem.get('eclipse_nodes', '?')}"

    def describe(self, overrides: dict | None = None) -> str:
        lines = [f"{self.name}  ({'flow' if self.is_flow else 'single method'}: {self.path.name})"]
        for stage in self.stages:
            seed = f"  <- {stage.seed_from}" if stage.seed_from else ""
            lines.append(f"  {stage.name:<14} {stage.config_path.name:<18} "
                         f"{self.stage_label(stage, overrides):<14}{seed}")
        return "\n".join(lines)

    # ----------------------------------------------------------------------- run

    def run(self, revolutions: int | None = None, overrides: dict | None = None,
            keep_problem: bool = False, validate_all: bool = True) -> Run:
        """Solve every stage in turn.

        ``overrides`` is ``{"problem": {...}, "method": {...}, "stages": {name:
        {"problem": {...}, "method": {...}}}}``; ``revolutions`` overrides
        ``problem.revolutions`` in every stage.  With ``keep_problem`` the final
        stage's :class:`SolarSailProblem` is left on ``Run.problem``.  Stages
        before the last are validated too unless ``validate_all`` is false; that
        costs a re-propagation each and is what shows how good the hand-over was.
        """
        overrides = copy.deepcopy(overrides or {})
        if revolutions is not None:
            overrides.setdefault("problem", {})["revolutions"] = int(revolutions)
        nrev = int(self.settings(overrides).revolutions)

        result = Run(self.name, nrev)
        exported: dict[str, tuple[dict, dict]] = {}   # stage name -> (guess, layouts)
        last = len(self.stages) - 1

        for index, stage in enumerate(self.stages):
            outcome, problem = self._run_stage(stage, overrides, exported,
                                               validate=validate_all or index == last)
            result.stages.append(outcome)

            keep_going = outcome.converged or (
                outcome.status == "not_converged" and stage.allow_nonconverged)
            if problem is not None and index < last and keep_going and self._needed_later(stage, index):
                started = time.perf_counter()
                exported[stage.name] = (problem.export_guess(), problem.phase_layouts())
                outcome.timings["export_s"] = time.perf_counter() - started

            if index == last and keep_problem:
                result.problem = problem
            else:
                del problem
                gc.collect()

            if not keep_going:
                result.failed_stage = stage.name
                break
        else:
            result.completed = True
        return result

    def _needed_later(self, stage: Stage, index: int) -> bool:
        return any(s.seed_from == stage.name for s in self.stages[index + 1:])

    def _run_stage(self, stage: Stage, overrides: dict, exported: dict, validate: bool):
        outcome = StageRun(stage.name, stage.config_path, stage.seed_from)
        timings = outcome.timings
        started = time.perf_counter()
        problem = None
        try:
            problem_over, method_over = self.stage_overrides(stage, overrides)

            guess = layouts = None
            if stage.seed_from is not None:
                guess, layouts = exported[stage.seed_from]
                if stage.seed_phases is not None:
                    unknown = set(stage.seed_phases) - set(guess)
                    if unknown:
                        raise ValueError(f"initial_guess.phases not in stage "
                                         f"'{stage.seed_from}': {sorted(unknown)}")
                    guess = {name: guess[name] for name in stage.seed_phases}
                    layouts = {name: layouts[name] for name in stage.seed_phases}

            before = time.perf_counter()
            problem = SolarSailProblem(stage.config_path, problem_over, method_over,
                                       initial_guess=guess, guess_layouts=layouts)
            timings["build_s"] = time.perf_counter() - before

            # Compile the JAX kernels up front so that the SCP time below is the
            # iterations alone (plus CVXPY's one-off canonicalisation).  solve()
            # repeats the call, which is then only a parameter update.
            before = time.perf_counter()
            problem.method.warmup_jax()
            timings["compile_s"] = time.perf_counter() - before

            before = time.perf_counter()
            try:
                problem.solve()
            finally:
                timings["solve_s"] = time.perf_counter() - before

            method = problem.method
            outcome.iterations = max(int(p.current_iter_data.iter_num) for p in problem.segments.values())
            status = method.cp_subproblem.status
            if method._converged:
                outcome.status, outcome.reason = "converged", "convergence criteria satisfied"
            elif outcome.iterations > 0 and status in {"optimal", "optimal_inaccurate"}:
                outcome.status, outcome.reason = "not_converged", "stopped before convergence"
            else:
                outcome.reason = f"solve failed: {status}, {outcome.iterations} accepted iterations"

            if validate and outcome.iterations > 0:
                before = time.perf_counter()
                outcome.solution = extract_solution(problem)
                outcome.report = validation_report(outcome.solution)
                timings["validate_s"] = time.perf_counter() - before
        except Exception as exc:  # recorded, so that a search or a comparison carries on
            outcome.status = "failed"
            outcome.reason = f"{type(exc).__name__}: {exc}"
            print(f"\nstage '{stage.name}' failed:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        timings["total_s"] = time.perf_counter() - started
        return outcome, problem


def load_algorithm(path: str | Path) -> Algorithm:
    return Algorithm(path)


def parse_overrides(assignments: list[str], stage_names: list[str]) -> dict:
    """``--set`` assignments into the ``overrides`` mapping of :meth:`Algorithm.run`.

    ``problem.a.b=1`` and ``method.a.b=1`` apply to every stage;
    ``<stage>.problem.a.b=1`` and ``<stage>.method.a.b=1`` to one stage.  Values
    are parsed as YAML, so ``1e-4``, ``true``, ``[12, 3]`` all work.
    """
    import yaml

    overrides: dict = {}
    for assignment in assignments:
        key, sep, raw = assignment.partition("=")
        if not sep or not key:
            raise ValueError(f"--set expects key=value, got {assignment!r}")
        parts = key.strip().split(".")
        if parts[0] in stage_names and len(parts) > 2 and parts[1] in ("problem", "method"):
            target = overrides.setdefault("stages", {}).setdefault(parts[0], {})
            parts = parts[1:]
        elif parts[0] in ("problem", "method") and len(parts) > 1:
            target = overrides
        else:
            raise ValueError(
                f"--set {key}: start with problem., method., or <stage>.problem. / "
                f"<stage>.method. where <stage> is one of {stage_names}")
        node = target.setdefault(parts[0], {})
        for part in parts[1:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return overrides
