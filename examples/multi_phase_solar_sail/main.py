import jax

from trajopt.methods.stable.algflows import AlgorithmFlow
from trajopt.trajectory_analyzer import TrajectoryAnalyzer

jax.config.update("jax_enable_x64", True)

ALGORITHM = "p_ms_hp"   # "p", "ms" or "p_ms_hp"

if ALGORITHM == "p_ms_hp":
    result = AlgorithmFlow("p_ms_hp.yaml").solve()
    for name, stage in result.stages.items():
        final_arc = stage.trajectory.get("final_arc")
        days = final_arc.sample(final_arc.native_tau).x[-1, 5] / 86400 if final_arc is not None else float("nan")
        print(f"{name}: {stage.status}, {stage.iterations} iterations, "
              f"{stage.timings['total_s']:.0f} s, transfer {days:.4f} days")
else:
    traj = TrajectoryAnalyzer(f"config_{ALGORITHM}.yaml")
    traj.solve()
    traj.analyze()
    traj.plot()
