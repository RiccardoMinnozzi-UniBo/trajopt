"""The library's propagated guess with a fixed number of RK4 steps per arc.
The library takes 10 steps per mesh interval, so its guess would change with the mesh."""
import dataclasses

from trajopt.methods.common import initial_guess

STEPS_PER_ARC = 400


def propagation(context):
    integrate = context.integrate
    fixed = dataclasses.replace(context, integrate=lambda *args, n_steps: integrate(*args, n_steps=STEPS_PER_ARC))
    return initial_guess.propagation(fixed)
