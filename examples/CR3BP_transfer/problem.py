"""Low-thrust transfer from Earth-Moon L1 to L2 in the planar CR3BP, minimum control effort, fixed time.

Nondimensional rotating frame: x = [px, py, vx, vy], u = [ux, uy] (thrust acceleration).
"""
from pathlib import Path
from types import SimpleNamespace

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from trajopt.common.trajectory import TrajectoryData

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve()
MU = 0.01215058560962404
U_MAX = 0.5             # loose bound, ~1.4e-3 m/s^2
U_SCALE = 0.05
R_MIN = 0.05            # ~19000 km from the Moon centre
TU_DAYS = 4.342


def dynamics(x, u, t, params, fcns=None):
    mu = params.mu
    px, py, vx, vy = x
    r1 = jnp.sqrt((px + mu) ** 2 + py ** 2)
    r2 = jnp.sqrt((px - 1 + mu) ** 2 + py ** 2)
    ax = 2 * vy + px - (1 - mu) * (px + mu) / r1 ** 3 - mu * (px - 1 + mu) / r2 ** 3
    ay = -2 * vx + py - (1 - mu) * py / r1 ** 3 - mu * py / r2 ** 3
    return jnp.array([vx, vy, ax + u[0], ay + u[1]])


_f = jax.jit(lambda x, u: dynamics(x, u, 0.0, SimpleNamespace(mu=MU)))


def crash(t, x):
    return min(np.hypot(x[0] + MU, x[1]) - 0.0166, np.hypot(x[0] - 1 + MU, x[1]) - 0.0045)


crash.terminal = True


def fly(x0, span, control):
    """Integrate open loop under the control law u(t), stopping on the Earth or Moon surface."""
    return solve_ivp(lambda t, x: np.asarray(_f(x, control(np.array([t]))[0])), span, x0, method="DOP853",
                     rtol=1e-12, atol=1e-12, dense_output=True, events=crash)


def moon_distance(x, u, t, params, fcns=None):
    return jnp.atleast_1d(jnp.sqrt((x[0] - 1 + params.mu) ** 2 + x[1] ** 2))


def effort(x, u, t, params):
    return cp.sum_squares(u)


def potential(px, py, mu=MU):
    r1 = np.sqrt((px + mu) ** 2 + py ** 2)
    r2 = np.sqrt((px - 1 + mu) ** 2 + py ** 2)
    return 0.5 * (px ** 2 + py ** 2) + (1 - mu) / r1 + mu / r2


def lagrange_points(mu=MU):
    slope = lambda px: px - (1 - mu) * (px + mu) / abs(px + mu) ** 3 - mu * (px - 1 + mu) / abs(px - 1 + mu) ** 3
    return brentq(slope, 0.5, 1 - mu - 1e-3), brentq(slope, 1 - mu + 1e-3, 1.5)


L1, L2 = lagrange_points()
X0 = np.array([L1, 0.0, 0.0, 0.0])
XF = np.array([L2, 0.0, 0.0, 0.0])


def trajectory(T):
    """The physical problem: transfer from rest at L1 to rest at L2 in T time units."""
    return {
        "fcns": {"dynamics": f"{HERE}:dynamics", "effort": f"{HERE}:effort", "moon_distance": f"{HERE}:moon_distance"},
        "params": {"mu": MU},
        "state": {"px": {"idx": [0], "scale": 1.0}, "py": {"idx": [1], "scale": 1.0},
                  "vx": {"idx": [2], "scale": 1.0}, "vy": {"idx": [3], "scale": 1.0}},
        "control": {"ux": {"idx": [0], "scale": U_SCALE}, "uy": {"idx": [1], "scale": U_SCALE}},
        "time": {"scale": 1.0},
        "constraints": {
            "dynamics": {"type": "dynamics", "fcn": "fcns.dynamics"},
            "initial_state": {"type": "initial_state", "value": X0},
            "final_state": {"type": "final_state", "value": XF},
            "initial_time": {"type": "initial_time", "value": 0.0},
            "final_time": {"type": "final_time", "value": float(T)},
            "thrust_limits": {"type": "control_limits", "lower": -U_MAX * np.ones(2), "upper": U_MAX * np.ones(2)},
            "moon_clearance": {"type": "nonconvex_inequality", "fcn": "fcns.moon_distance", "lower": np.array([R_MIN])},
        },
        "costs": {"effort": {"type": "convex_running", "fcn": "fcns.effort", "w": 1.0}},
    }


def initial_guess(T, transit=4.0, n=1001):
    """Wait at L1, half a circle around the Moon to L2 in the middle `transit` time units, wait at L2. No thrust."""
    t = np.linspace(0.0, T, n)
    transit = min(transit, T)
    s = np.clip((t - 0.5 * (T - transit)) / transit, 0.0, 1.0)
    angle = np.pi * (1 + 3 * s ** 2 - 2 * s ** 3)
    rate = np.pi * (6 * s - 6 * s ** 2) / transit
    centre, radius = 0.5 * (L1 + L2), 0.5 * (L2 - L1)
    x = np.column_stack([centre + radius * np.cos(angle), radius * np.sin(angle),
                         -radius * np.sin(angle) * rate, radius * np.cos(angle) * rate])
    return TrajectoryData.from_samples(t=t, x=x, u=np.zeros((n, 2)))
