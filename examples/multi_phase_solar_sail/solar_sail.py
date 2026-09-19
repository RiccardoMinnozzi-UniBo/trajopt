"""Dynamics and terminal constraints for the fixed-Sun solar-sail deorbit example.

State and independent variable follow Kelly and Bevilacqua (2021) after their
change of independent variable from physical time to true longitude::

    x = [p, f, g, h, k, elapsed_time]        (km, -, -, -, -, s)
    independent variable: L, the true longitude (rad)

The single control is the **steering angle** ``theta``: the direction of the
paper's primer vector in the local orbital frame,

    primer = cos(theta) * radial + sin(theta) * transverse

Only the *direction* of the primer enters the steering law, so parameterising it
by one angle (instead of two components plus a unit-norm equality) removes the
problem's only nonconvex path constraint at no modelling cost.

The sail normal is then recovered from the closed-form locally optimal steering
law of the paper's Equation (24), which maximises the acceleration projected
onto the primer.
"""

import cvxpy as cp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _scalar_longitude(longitude):
    """Accept trajopt's one-element time vector as well as a scalar."""
    return jnp.ravel(jnp.asarray(longitude))[0]


def orbital_frame(h, k, longitude):
    """Radial, transverse and orbit-normal unit vectors in ECI."""
    s_squared = 1.0 + h**2 + k**2

    equinoctial_x = jnp.array([1.0 - k**2 + h**2, 2.0 * h * k, -2.0 * k]) / s_squared
    equinoctial_y = jnp.array([2.0 * h * k, 1.0 + k**2 - h**2, 2.0 * h]) / s_squared

    cosine = jnp.cos(longitude)
    sine = jnp.sin(longitude)
    radial = cosine * equinoctial_x + sine * equinoctial_y
    transverse = -sine * equinoctial_x + cosine * equinoctial_y
    return radial, transverse, jnp.cross(radial, transverse)


def position_eci(x, longitude):
    """Spacecraft position in ECI coordinates, in km."""
    longitude = _scalar_longitude(longitude)
    p, f, g = x[0], x[1], x[2]
    q = 1.0 + f * jnp.cos(longitude) + g * jnp.sin(longitude)
    radial, _, _ = orbital_frame(x[3], x[4], longitude)
    return (p / q) * radial


def _unit_vector(vector, fallback):
    squared_norm = jnp.dot(vector, vector)
    safe_norm = jnp.sqrt(squared_norm + 1.0e-24)
    return jnp.where(squared_norm > 1.0e-24, vector / safe_norm, fallback)


def light_direction(params):
    """Unit vector along photon travel, i.e. pointing away from the Sun.

    ``params.sun_direction_eci`` is the Earth-to-Sun direction, so sunlight
    travels along its negative.  Every sail quantity below is referred to this
    direction: a sail can only ever be pushed along it, never against it.
    """
    sun_direction = jnp.asarray(params.sun_direction_eci)
    return -sun_direction / jnp.linalg.norm(sun_direction)


def sail_normal_from_primer(x, u, longitude, params):
    """Locally optimal sail normal and cone angle, from Equations (23)-(25).

    The cone angle ``alpha`` is measured from the light direction and maximises
    the acceleration projected onto the primer.  When the primer points into the
    Sun the law returns ``alpha = pi/2``: the sail turns edge-on and produces no
    thrust, which is the correct feathered behaviour.
    """
    longitude = _scalar_longitude(longitude)
    radial, transverse, _ = orbital_frame(x[3], x[4], longitude)
    primer = jnp.cos(u[0]) * radial + jnp.sin(u[0]) * transverse

    sunward = light_direction(params)

    # Avoid the coordinate singularities at exactly parallel/antiparallel
    # vectors.  The 1e-10 clipping changes the angle by less than 1e-3 deg but
    # keeps automatic derivatives finite at arbitrary SCP guesses.
    cos_gamma_raw = jnp.clip(jnp.dot(sunward, primer), -1.0, 1.0)
    cos_gamma = jnp.clip(cos_gamma_raw, -1.0 + 1.0e-10, 1.0 - 1.0e-10)
    sin_gamma = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_gamma**2))

    # Equation (24a), evaluated with atan2 to retain the correct quadrant.
    numerator = jnp.sqrt(cos_gamma**2 + 8.0) - 3.0 * cos_gamma
    denominator = 4.0 * sin_gamma
    alpha = jnp.arctan2(numerator, denominator)

    reference = jnp.where(
        jnp.abs(sunward[2]) < 0.9,
        jnp.array([0.0, 0.0, 1.0]),
        jnp.array([0.0, 1.0, 0.0]),
    )
    fallback = _unit_vector(jnp.cross(reference, sunward), transverse)
    e2 = _unit_vector(primer - cos_gamma_raw * sunward, fallback)

    sail_normal = jnp.cos(alpha) * sunward + jnp.sin(alpha) * e2
    return sail_normal, alpha


def solar_sail_acceleration_lvlh(x, u, longitude, params):
    """Sail acceleration resolved in the local radial/transverse/normal frame.

    Absorbed photons push along the light direction, specularly reflected ones
    along the sail normal::

        a = (P A / m) cos(alpha) [ (1 - eps) s_hat + 2 eps cos(alpha) n_hat ]

    ``params.illumination`` is 1 on a sunlit arc and 0 inside the Earth shadow.
    """
    longitude = _scalar_longitude(longitude)
    radial, transverse, normal = orbital_frame(x[3], x[4], longitude)
    sail_normal, alpha = sail_normal_from_primer(x, u, longitude, params)

    pressure_acceleration = (
        params.solar_pressure_n_m2 * params.area_to_mass_m2_kg / 1000.0
    )  # N/kg -> km/s^2

    epsilon = params.reflectivity
    sunward = light_direction(params)
    cosine = jnp.cos(alpha)

    acceleration_eci = (
        params.illumination
        * pressure_acceleration
        * cosine
        * ((1.0 - epsilon) * sunward + 2.0 * epsilon * cosine * sail_normal)
    )

    return jnp.array(
        [
            jnp.dot(radial, acceleration_eci),
            jnp.dot(transverse, acceleration_eci),
            jnp.dot(normal, acceleration_eci),
        ]
    )


def dynamics(x, u, longitude, params, fcns):
    """Modified-equinoctial dynamics with longitude as the independent variable."""
    del fcns
    longitude = _scalar_longitude(longitude)
    p, f, g, h, k = x[:5]
    mu = params.earth_mu_km3_s2

    cosine = jnp.cos(longitude)
    sine = jnp.sin(longitude)
    q = 1.0 + f * cosine + g * sine
    s_squared = 1.0 + h**2 + k**2
    out_of_plane_factor = h * sine - k * cosine

    radial_accel, transverse_accel, normal_accel = solar_sail_acceleration_lvlh(
        x, u, longitude, params
    )

    common = jnp.sqrt(p / mu) / q

    # Standard modified-equinoctial Gauss equations.  Equation (15) in the
    # typeset paper shows 2p/q inside an already 1/q-scaled matrix; that would
    # introduce a spurious extra q.  The form below is dimensionally and
    # algebraically consistent with the remaining MEE equations.
    p_dot = common * (2.0 * p) * transverse_accel
    f_dot = common * (
        q * sine * radial_accel
        + ((q + 1.0) * cosine + f) * transverse_accel
        - g * out_of_plane_factor * normal_accel
    )
    g_dot = common * (
        -q * cosine * radial_accel
        + ((q + 1.0) * sine + g) * transverse_accel
        + f * out_of_plane_factor * normal_accel
    )
    h_dot = common * (0.5 * s_squared * cosine) * normal_accel
    k_dot = common * (0.5 * s_squared * sine) * normal_accel
    # The q^2 Keplerian term is the standard, dimensionally consistent MEE
    # expression (the superscript is not visible in the paper's typeset Eq. 16).
    longitude_dot = (
        common * out_of_plane_factor * normal_accel + jnp.sqrt(mu * p) * (q / p) ** 2
    )

    # Convert d/dt to d/dL; dt/dL is the sixth state derivative.
    return jnp.array([p_dot, f_dot, g_dot, h_dot, k_dot, 1.0]) / longitude_dot


# ---------------------------------------------------------------------------
# phase boundaries
# ---------------------------------------------------------------------------

def shadow_boundary(x, u, longitude, params):
    """Zero exactly on the cylindrical Earth-shadow terminator.

    The terminator is the edge of the Earth's shadow -- the instant the
    spacecraft crosses into or out of eclipse, and therefore the instant the
    sail switches between producing thrust and producing none.  That is what
    physically defines a phase boundary, so this is imposed as an equality at
    the end of every arc except the last.

    The residual is the spacecraft's squared perpendicular distance from the
    Earth-Sun line, measured against the Earth's radius and normalised by it:
    zero when that distance is exactly R_Earth, negative inside the shadow
    cylinder, positive outside.  Because the distance depends on the orbit
    radius, the crossing longitude moves as the orbit is raised; imposing this
    equality lets the solver place each boundary where the crossing actually
    occurs rather than pinning it to a constant taken from the initial orbit.

    Note this is the *cylinder* condition alone, which several longitudes per
    revolution satisfy -- shadow entry, shadow exit, and the two sunward points
    at the same perpendicular distance.  The arc-endpoint window in problem.py
    is what keeps each boundary on the root it belongs to.
    """
    del u
    position = position_eci(x, longitude)
    sunward = light_direction(params)
    axial = jnp.dot(position, sunward)
    transverse_squared = jnp.dot(position, position) - axial**2
    radius_squared = params.earth_radius_km**2
    return jnp.atleast_1d((transverse_squared - radius_squared) / radius_squared)


def phase_end_longitude(x, u, longitude, params):
    """Expose the true longitude so an arc endpoint can be bounded."""
    del x, u, params
    return jnp.atleast_1d(_scalar_longitude(longitude))


# ---------------------------------------------------------------------------
# terminal (graveyard) conditions -- exact convex cones for CVXPY
# ---------------------------------------------------------------------------

def terminal_eccentricity_cone(x, u, params):
    """Exact convex form of ``e_f <= e_max``."""
    del u
    return cp.norm(x[:, 1:3], axis=1) - float(params.final_eccentricity_max)


def terminal_perigee_cone(x, u, params):
    """Exact convex form of ``r_p,f >= r_target``.

    ``r_p = p / (1 + e)`` is nonconvex, but the equivalent
    ``r_target (1 + e) - p <= 0`` is a second-order cone constraint in (p, f, g).
    """
    del u
    target = float(params.final_perigee_radius_km)
    return target * (1.0 + cp.norm(x[:, 1:3], axis=1)) - x[:, 0]
