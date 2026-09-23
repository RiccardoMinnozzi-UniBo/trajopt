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
        # the propagated orbit, not the node values: see validation_report
        offset = self.report["true_final_perigee_km"] - self.solution.geo_radius_km
        return (f"N={self.revolutions:3d}  {mark}  "
                f"{self.elapsed_days:8.4f} days  "
                f"e_f={self.report['true_final_eccentricity']:.6f}  "
                f"true r_p-r_GEO={offset:+8.2f} km  "
                f"acc={self.report['accumulated_defect_km']:+8.2f} km  "
                f"({self.report['iterations']:4d} it"
                f"{'' if self.report['solver_converged'] else ', UNCONVERGED'})")


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
    maximum: int = 60,
    max_solves: int = 14,
    overrides: dict | None = None,
) -> SearchResult:
    """Fewest revolutions that still reach the graveyard orbit.

    Feasibility is monotone -- more revolutions means more time under thrust,
    so once the graveyard is reachable it stays reachable -- which makes this a
    search for a single threshold.  The walk therefore brackets the threshold
    and bisects it, rather than stepping one revolution at a time:

      * expand outward from the guess, doubling the stride, until one feasible
        and one infeasible count are known;
      * bisect until they are adjacent.  The infeasible side is what proves the
        feasible side is minimal.

    Stepping by one costs O(distance) solves, which matters because the honest
    minimum for the paper sail is near thirty and a solve at that size is
    expensive.  Bracketing costs O(log(distance)), so a poor guess is cheap and
    the answer no longer depends on starting near it.
    """
    start = int(min(max(start, minimum), maximum))
    attempts: list[Attempt] = []
    best: Attempt | None = None
    best_problem: SolarSailProblem | None = None

    def release(problem):
        del problem
        gc.collect()

    def keep(attempt, problem):
        nonlocal best, best_problem
        if best_problem is not None:
            release(best_problem)
        best, best_problem = attempt, problem

    def evaluate(revolutions: int) -> bool:
        """Solve at this count, keeping the problem only if it is the best yet."""
        attempt, problem = solve_once(config_path, revolutions, overrides)
        attempts.append(attempt)
        if attempt.feasible and (best is None or revolutions < best.revolutions):
            keep(attempt, problem)
        else:
            release(problem)
        return attempt.feasible

    def budget_left() -> bool:
        return len(attempts) < max_solves

    print(f"\nrevolution search: starting from N = {start}, "
          f"bracketing within [{minimum}, {maximum}]", flush=True)

    lo: int | None = None   # largest count shown to be infeasible
    hi: int | None = None   # smallest count shown to be feasible

    if evaluate(start):
        hi = start
    else:
        lo = start

    if hi is None:
        print("  infeasible: expanding upward until a count succeeds", flush=True)
        stride, count = 1, start
        while count < maximum and budget_left():
            count = min(maximum, count + stride)
            if evaluate(count):
                hi = count
                break
            lo = count
            stride *= 2

    if lo is None:
        print("  feasible: expanding downward until a count fails", flush=True)
        stride, count = 1, start
        while count > minimum and budget_left():
            count = max(minimum, count - stride)
            if evaluate(count):
                hi = count
            else:
                lo = count
                break
            stride *= 2

    if lo is not None and hi is not None and hi - lo > 1:
        print(f"  bracketed: N = {lo} fails, N = {hi} succeeds -- bisecting",
              flush=True)
    while lo is not None and hi is not None and hi - lo > 1 and budget_left():
        middle = (lo + hi) // 2
        if evaluate(middle):
            hi = middle
        else:
            lo = middle

    # Without a failure below it, the best count is only the smallest tried.
    bounded = best is not None and lo is None
    exhausted = (best is not None and lo is not None and hi is not None
                 and hi - lo > 1)

    _report(best, attempts, bounded, minimum, exhausted)
    return SearchResult(best, best_problem, attempts, bounded)


def _report(best, attempts, bounded, minimum, exhausted=False) -> None:
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
    elif exhausted:
        print(f"  note: the bracket was not closed before max_solves ran out, "
              f"so N = {best.revolutions} is feasible but not proven minimal "
              f"-- raise revolution_search.max_solves to finish the bisection")
