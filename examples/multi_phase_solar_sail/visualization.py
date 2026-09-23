"""Publication-quality figures for a solar-sail disposal solution.

Everything is drawn on a white background with a colourblind-safe palette
(Okabe-Ito) at 300 dpi, sized for a one- or two-column figure.

``render_all`` writes the whole set; each figure is also available on its own:

``render_elements``      orbital element histories against the requirements
``render_control``       steering angle, cone angle and thrust magnitude
``render_transfer_plane`` polar plan view of the spiral, with the shadow sector
``render_trajectory_3d`` 3D view with Earth and the shadow cylinder
``render_phases``        phase timeline and durations
``render_convergence``   SCP convergence history
``render_summary``       everything on one sheet, for a talk
``render_rotating_gif``  rotating version of the 3D view

All are safe to call on a solution that failed validation; they are annotated
with the outcome so an unconverged run is still readable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from solution import SolarSailSolution, load_solution

_SECONDS_PER_DAY = 86400.0

# --- style -----------------------------------------------------------------
# Okabe-Ito: distinguishable in grayscale and for all common colour vision types.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
TEXT = "#1A1A1A"
MUTED = "#5A5A5A"
GRID = "#C8C8C8"
ECLIPSE_BAND = "#DCE3EC"
EARTH = "#4E79A7"
TRAJECTORY_CMAP = "viridis"

_RC = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": TEXT,
    "axes.linewidth": 0.8,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 10.5,
    "axes.titleweight": "normal",
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.alpha": 0.7,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": GRID,
    "font.family": "DejaVu Sans",
    "lines.linewidth": 1.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}


def _save(figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


#: Nodes are drawn as small hollow markers on top of the propagated curve, so
#: the discretisation stays visible without competing with the trajectory.
_NODE_STYLE = dict(s=7, facecolors="none", linewidths=0.6, zorder=5)


def _wrap_deg(angle_rad) -> np.ndarray:
    """Angle in degrees wrapped to (-180, 180].

    The steering angle winds by -2*pi per revolution, so after 24 revolutions it
    reaches about -8500 deg.  Plotted unwrapped it flattens every other curve on
    the axis; wrapped, the per-revolution structure is visible and it shares an
    axis with the cone angle sensibly.
    """
    return np.rad2deg((np.asarray(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi)


def _wrap_deg_gapped(angle_rad) -> np.ndarray:
    """``_wrap_deg`` with NaN inserted at the wraps, so no vertical stripes."""
    wrapped = _wrap_deg(angle_rad)
    jump = np.abs(np.diff(wrapped, prepend=wrapped[0])) > 180.0
    return np.where(jump, np.nan, wrapped)


def _eclipse_intervals_dense(solution: SolarSailSolution) -> list[tuple[float, float]]:
    """Eclipse spans in days, from the propagated trajectory."""
    days = solution.dense_elapsed_days
    lit = solution.dense_illuminated
    intervals, start = [], None
    for index, illuminated in enumerate(lit):
        if not illuminated and start is None:
            start = days[index]
        elif illuminated and start is not None:
            intervals.append((start, days[index]))
            start = None
    if start is not None:
        intervals.append((start, days[-1]))
    return intervals


def _eclipse_intervals(solution: SolarSailSolution) -> list[tuple[float, float]]:
    """Eclipse arcs in elapsed days, for shading time-axis plots."""
    longitude, days = solution.longitude_rad, solution.elapsed_days
    order = np.argsort(longitude)
    return [
        (float(np.interp(start, longitude[order], days[order])),
         float(np.interp(stop, longitude[order], days[order])))
        for start, stop, sunlit in zip(solution.phase_start_rad,
                                       solution.phase_stop_rad,
                                       solution.phase_sunlit)
        if not sunlit
    ]


def _shade_eclipses(axis, intervals, label=True) -> None:
    for index, (start, stop) in enumerate(intervals):
        axis.axvspan(start, stop, color=ECLIPSE_BAND, linewidth=0, zorder=0,
                     label="eclipse" if (label and index == 0) else None)


def _status(report: dict | None) -> tuple[str, str]:
    if report is None:
        return "", MUTED
    if report["numerically_feasible"]:
        return "validated", GREEN
    if report.get("graveyard_reached", False):
        return "graveyard reached, validation incomplete", ORANGE
    return "not validated", VERMILLION


def _caption(solution: SolarSailSolution, report: dict | None) -> str:
    rise = float(solution.perigee_radius_km[-1]) - solution.geo_radius_km
    text = (f"A/m = {solution.area_to_mass_m2_kg:g} m$^2$/kg  ·  "
            f"N = {solution.revolutions}  ·  "
            f"{len(solution.phase_duration_h)} phases  ·  "
            f"{float(solution.elapsed_days[-1]):.3f} days  ·  "
            f"$r_p-r_{{GEO}}$ = {rise:+.1f} km  ·  "
            f"$e_f$ = {float(solution.eccentricity[-1]):.5f}")
    if report is not None:
        text += f"  ·  {report['iterations']} SCP iterations"
    return text


def _annotate(figure, solution, report, y=1.0, annotated=True) -> None:
    """Caption above the figure, validation status stacked above it.

    Stacked rather than side by side: the caption is long enough that a
    right-aligned status would collide with it on a narrow figure.

    ``annotated=False`` draws nothing at all, which is the point: the caption
    sits outside the axes, so with a tight bounding box it changes the saved
    figure's extent.  Leaving it out gives a clean crop that drops into a paper
    without disturbing the layout around it.
    """
    if not annotated:
        return
    figure.text(0.0, y, _caption(solution, report), fontsize=8.2, color=MUTED,
                ha="left", va="bottom", transform=figure.transFigure)
    label, colour = _status(report)
    if label:
        figure.text(0.0, y + 0.038, label, fontsize=8.2, color=colour,
                    ha="left", va="bottom", weight="bold",
                    transform=figure.transFigure)


# ---------------------------------------------------------------------------
# individual figures
# ---------------------------------------------------------------------------

def render_elements(solution: SolarSailSolution, path: str | Path,
                    report: dict | None = None, annotated: bool = True) -> Path:
    """Orbital element histories against the graveyard requirements."""
    days = solution.dense_elapsed_days
    node_days = solution.elapsed_days
    eclipses = _eclipse_intervals_dense(solution)

    with plt.rc_context(_RC):
        figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))

        axis = axes[0]
        _shade_eclipses(axis, eclipses)
        axis.axhline(solution.geo_radius_km, color=MUTED, linestyle="--",
                     linewidth=1.0, label="GEO")
        axis.axhline(solution.target_perigee_km, color=VERMILLION, linestyle=":",
                     linewidth=1.3, label="perigee requirement")
        axis.plot(days, solution.dense_apogee_radius_km, color=ORANGE, label="apogee")
        axis.plot(days, solution.dense_perigee_radius_km, color=BLUE, label="perigee")
        axis.scatter(node_days, solution.perigee_radius_km, edgecolors=BLUE,
                     label="nodes", **_NODE_STYLE)
        axis.set_ylabel("radius [km]")
        axis.set_title("Apsides")
        axis.legend(loc="lower right")

        axis = axes[1]
        _shade_eclipses(axis, eclipses)
        axis.axhline(solution.target_eccentricity, color=VERMILLION,
                     linestyle=":", linewidth=1.3, label="requirement")
        axis.plot(days, solution.dense_eccentricity, color=BLUE, label="eccentricity")
        axis.scatter(node_days, solution.eccentricity, edgecolors=BLUE,
                     label="nodes", **_NODE_STYLE)
        axis.set_ylabel("eccentricity [-]")
        axis.set_title("Eccentricity")
        axis.legend(loc="lower right")

        axis = axes[2]
        _shade_eclipses(axis, eclipses)
        axis.plot(days, solution.dense_state[:, 0], color=GREEN)
        axis.scatter(node_days, solution.state[:, 0], edgecolors=GREEN,
                     **_NODE_STYLE)
        axis.set_ylabel(r"$p - r_{\mathrm{GEO}}$ [km]")
        axis.set_title("Semi-latus rectum")

        for axis in axes:
            axis.set_xlabel("elapsed time [days]")
            axis.set_xlim(days[0], days[-1])
        figure.tight_layout()
        _annotate(figure, solution, report, y=1.01, annotated=annotated)
        return _save(figure, path)


def render_control(solution: SolarSailSolution, path: str | Path,
                   report: dict | None = None, annotated: bool = True) -> Path:
    """Steering angle, sail cone angle and the resulting thrust magnitude."""
    days = solution.dense_elapsed_days
    node_days = solution.elapsed_days
    eclipses = _eclipse_intervals_dense(solution)
    cone = np.rad2deg(solution.dense_cone_angle_rad).astype(float)
    cone[~solution.dense_illuminated] = np.nan  # the sail is inactive in eclipse

    with plt.rc_context(_RC):
        figure, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)

        axis = axes[0]
        _shade_eclipses(axis, eclipses)
        axis.plot(days, _wrap_deg_gapped(solution.dense_steering_angle_rad), color=BLUE,
                  label=r"steering angle $\theta$")
        axis.plot(days, cone, color=ORANGE, label=r"cone angle $\alpha$")
        axis.scatter(node_days, _wrap_deg(solution.steering_angle_rad),
                     edgecolors=BLUE, label="nodes", **_NODE_STYLE)
        axis.axhline(90.0, color=MUTED, linewidth=0.8, linestyle=":")
        axis.annotate("feathered", xy=(days[-1], 90.0), xytext=(-4, 3),
                      textcoords="offset points", ha="right", fontsize=7.5,
                      color=MUTED)
        axis.set_ylim(-190.0, 190.0)
        axis.set_yticks([-180, -90, 0, 90, 180])
        axis.set_ylabel("angle [deg]")
        axis.set_title("Sail steering")
        axis.legend(loc="upper left", ncol=3)

        axis = axes[1]
        _shade_eclipses(axis, eclipses, label=False)
        axis.plot(days, solution.dense_acceleration_mm_s2, color=GREEN)
        axis.fill_between(days, 0.0, solution.dense_acceleration_mm_s2, color=GREEN,
                          alpha=0.18, linewidth=0)
        axis.set_ylabel(r"$|a_{\mathrm{SRP}}|$ [mm/s$^2$]")
        axis.set_xlabel("elapsed time [days]")
        axis.set_title("Thrust magnitude (zero in eclipse, zero when feathered)")
        axis.set_ylim(bottom=0.0)

        for axis in axes:
            axis.set_xlim(days[0], days[-1])
        figure.tight_layout()
        _annotate(figure, solution, report, y=1.01, annotated=annotated)
        return _save(figure, path)


def render_transfer_plane(solution: SolarSailSolution, path: str | Path,
                          report: dict | None = None,
                          annotated: bool = True) -> Path:
    """Plan view of the transfer in the orbit plane.

    The radial axis is the orbit radius *relative to GEO*, which is the only way
    the spiral is visible at all: the whole transfer changes the radius by well
    under one per cent.
    """
    longitude = solution.dense_longitude_rad
    radius = solution.dense_radius_km - solution.geo_radius_km
    days = solution.dense_elapsed_days

    # the shadow sector sweeps with the Sun, so show where it starts and ends
    sun_start = solution.sun_direction_at(0.0)
    sun_end = solution.sun_direction_at(float(solution.dense_state[-1, 5]))
    sun_longitude = float(np.arctan2(-sun_start[1], -sun_start[0]))
    sun_longitude_end = float(np.arctan2(-sun_end[1], -sun_end[0]))
    half_shadow = float(np.arcsin(solution.earth_radius_km / solution.geo_radius_km))

    with plt.rc_context(_RC):
        figure = plt.figure(figsize=(6.4, 5.6))
        axis = figure.add_subplot(111, projection="polar")

        # shadow sector, centred on the anti-Sun longitude
        axis.bar(sun_longitude, 1.0, width=2.0 * half_shadow, bottom=0.0,
                 transform=axis.get_xaxis_transform(), color=ECLIPSE_BAND,
                 linewidth=0, zorder=0, label="Earth shadow (start)")
        if abs(sun_longitude_end - sun_longitude) > 1e-3:
            axis.bar(sun_longitude_end, 1.0, width=2.0 * half_shadow, bottom=0.0,
                     transform=axis.get_xaxis_transform(), facecolor="none",
                     edgecolor=MUTED, linewidth=0.8, linestyle="--", zorder=1,
                     label="Earth shadow (end)")

        points = np.column_stack((longitude, radius))
        segments = np.stack((points[:-1], points[1:]), axis=1)
        collection = LineCollection(segments, cmap=TRAJECTORY_CMAP, linewidth=1.7,
                                    zorder=3)
        collection.set_array(0.5 * (days[:-1] + days[1:]))
        axis.add_collection(collection)

        axis.scatter(solution.longitude_rad,
                     solution.radius_km - solution.geo_radius_km,
                     edgecolors=MUTED, label="nodes", **_NODE_STYLE)
        axis.plot(longitude[0], radius[0], "o", color=BLUE, markersize=6,
                  zorder=4, label="start (GEO)")
        axis.plot(longitude[-1], radius[-1], "*", color=VERMILLION,
                  markersize=12, zorder=4, label="end (graveyard)")
        axis.plot(np.linspace(0, 2 * np.pi, 200), np.zeros(200), color=MUTED,
                  linewidth=0.9, linestyle="--", zorder=2, label="GEO radius")

        axis.set_rmin(min(0.0, float(radius.min())) - 15.0)
        axis.set_rmax(float(radius.max()) * 1.08 + 10.0)
        axis.set_theta_zero_location("E")
        axis.set_title(r"Transfer in the orbit plane:  $r - r_{\mathrm{GEO}}$ [km]",
                       pad=16)
        axis.tick_params(axis="y", labelsize=7.5)
        axis.set_rlabel_position(108.0)  # clear of the trajectory and the shadow
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2)

        bar = figure.colorbar(collection, ax=axis, pad=0.10, shrink=0.75)
        bar.set_label("elapsed time [days]")
        bar.outline.set_edgecolor(GRID)

        _annotate(figure, solution, report, y=1.00, annotated=annotated)
        return _save(figure, path)


def render_phases(solution: SolarSailSolution, path: str | Path,
                  report: dict | None = None, annotated: bool = True) -> Path:
    """Phase structure: when each arc runs, and how long each lasts."""
    numbers = np.arange(1, len(solution.phase_duration_h) + 1)
    colours = [ORANGE if lit else BLUE for lit in solution.phase_sunlit]

    # arc start times in elapsed days
    order = np.argsort(solution.longitude_rad)
    starts = np.interp(solution.phase_start_rad,
                       solution.longitude_rad[order],
                       solution.elapsed_days[order])

    with plt.rc_context(_RC):
        figure, axes = plt.subplots(2, 1, figsize=(7.0, 4.4))

        axis = axes[0]
        axis.barh(numbers, solution.phase_duration_h / 24.0, left=starts,
                  color=colours, height=0.75, edgecolor="white", linewidth=0.4)
        axis.set_xlabel("elapsed time [days]")
        axis.set_ylabel("phase")
        axis.set_title("Phase timeline")
        step = max(1, len(numbers) // 12)  # keep the labels readable when N is large
        axis.set_yticks(numbers[::step])
        axis.invert_yaxis()
        axis.grid(axis="y", visible=False)

        axis = axes[1]
        axis.bar(numbers, solution.phase_duration_h, color=colours, width=0.78)
        axis.axhline(0.0, color=TEXT, linewidth=0.8)
        shortest = float(np.min(solution.phase_duration_h))
        tallest = float(np.max(solution.phase_duration_h))
        axis.set_ylim(min(0.0, shortest) * 1.15 - 0.05, tallest * 1.28)
        axis.set_xlabel("phase")
        axis.set_ylabel("duration [h]")
        axis.set_title(f"Phase durations (shortest {shortest:.3f} h)")
        axis.grid(axis="x", visible=False)

        handles = [Line2D([], [], color=ORANGE, linewidth=6, label="sunlit"),
                   Line2D([], [], color=BLUE, linewidth=6, label="eclipse")]
        axes[0].legend(handles=handles, loc="lower right", ncol=2)

        figure.tight_layout()
        _annotate(figure, solution, report, y=1.01, annotated=annotated)
        return _save(figure, path)


def render_convergence(solution: SolarSailSolution, path: str | Path,
                       report: dict | None = None,
                       annotated: bool = True) -> Path:
    """SCP convergence history."""
    defect = np.asarray(solution.defect_history, dtype=float)

    with plt.rc_context(_RC):
        figure, axis = plt.subplots(figsize=(6.4, 3.6))
        if defect.size:
            iterations = np.arange(1, defect.size + 1)
            axis.semilogy(iterations, np.maximum(defect, 1e-16), color=BLUE,
                          label="max dynamics defect")
            steps = np.asarray(solution.step_history, dtype=float)
            if steps.size == defect.size and np.isfinite(steps).any():
                axis.semilogy(iterations, np.maximum(steps, 1e-16), color=ORANGE,
                              label=r"state step / $\epsilon$")
            axis.axhline(1.0e-4, color=VERMILLION, linestyle=":", linewidth=1.3,
                         label=r"tolerance $10^{-4}$")
            axis.set_xlim(1, defect.size)
            axis.set_xlabel("SCP iteration")
            axis.set_ylabel("nondimensional")
            axis.legend(loc="upper right")
        else:
            axis.text(0.5, 0.5, "no iteration history", ha="center", va="center",
                      color=MUTED, transform=axis.transAxes)
            axis.set_axis_off()
        axis.set_title("Sequential convex programming convergence")
        figure.tight_layout()
        _annotate(figure, solution, report, y=1.01, annotated=annotated)
        return _save(figure, path)


# ---------------------------------------------------------------------------
# 3D scene
# ---------------------------------------------------------------------------

def _orthonormal_basis(axis_vector: np.ndarray):
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(axis_vector @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis_vector, reference)
    first /= np.linalg.norm(first)
    return first, np.cross(axis_vector, first)


def _shadow_surfaces(axis, solution: SolarSailSolution, elapsed_seconds: float):
    """Draw the shadow cylinder for the Sun direction at a given time.

    Returns the artists so an animation can remove and redraw them: the Sun line
    sweeps about +Z as the transfer proceeds, so the cylinder is not static.
    """
    earth_radius = solution.earth_radius_km
    geo_radius = solution.geo_radius_km

    sun = solution.sun_direction_at(elapsed_seconds)
    sun = sun / np.linalg.norm(sun)
    light = -sun
    first, second = _orthonormal_basis(light)

    angle = np.linspace(0.0, 2.0 * np.pi, 60)
    length = np.linspace(0.0, 1.35 * geo_radius, 2)
    angle_grid, length_grid = np.meshgrid(angle, length)
    ring = (np.cos(angle_grid)[..., None] * first
            + np.sin(angle_grid)[..., None] * second) * earth_radius
    cylinder = ring + length_grid[..., None] * light

    artists = [axis.plot_surface(
        cylinder[..., 0], cylinder[..., 1], cylinder[..., 2],
        color="#9AA7B8", alpha=0.30, linewidth=0, shade=False)]
    for end in (0.0, 1.35 * geo_radius):
        edge = (np.cos(angle)[:, None] * first + np.sin(angle)[:, None] * second)
        edge = edge * earth_radius + end * light
        artists.extend(axis.plot(edge[:, 0], edge[:, 1], edge[:, 2], color=MUTED,
                                 linewidth=0.7, alpha=0.6))

    tail = -light * geo_radius * 1.35
    artists.append(axis.quiver(*tail, *light, length=geo_radius * 0.40,
                               color=ORANGE, linewidth=2.2, arrow_length_ratio=0.18))
    artists.append(axis.text(*(tail - light * geo_radius * 0.20), "sunlight",
                             color=ORANGE, fontsize=9))
    return artists


def _draw_scene(axis, solution: SolarSailSolution, draw_shadow: bool = True):
    earth_radius = solution.earth_radius_km
    geo_radius = solution.geo_radius_km
    position = solution.dense_position_km

    lon = np.linspace(0.0, 2.0 * np.pi, 120)
    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 60)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    axis.plot_surface(
        earth_radius * np.cos(lat_grid) * np.cos(lon_grid),
        earth_radius * np.cos(lat_grid) * np.sin(lon_grid),
        earth_radius * np.sin(lat_grid),
        color=EARTH, alpha=1.0, linewidth=0, antialiased=True, shade=True,
        zorder=1,
    )

    if draw_shadow:
        _shadow_surfaces(axis, solution, 0.0)

    ring_angle = np.linspace(0.0, 2.0 * np.pi, 400)
    axis.plot(geo_radius * np.cos(ring_angle), geo_radius * np.sin(ring_angle),
              np.zeros_like(ring_angle), color=MUTED, linewidth=0.9, alpha=0.7,
              linestyle="--", label="initial GEO")

    segments = np.stack((position[:-1], position[1:]), axis=1)
    dense_days = solution.dense_elapsed_days
    midpoint_days = 0.5 * (dense_days[:-1] + dense_days[1:])
    lit = solution.dense_illuminated
    sunlit = lit[:-1] & lit[1:]
    collection = Line3DCollection(segments[sunlit], cmap=TRAJECTORY_CMAP,
                                  linewidth=1.4)
    collection.set_array(midpoint_days[sunlit])
    axis.add_collection3d(collection)
    if (~sunlit).any():
        axis.add_collection3d(Line3DCollection(segments[~sunlit],
                                               colors=VERMILLION, linewidth=2.2))

    nodes = solution.position_km
    axis.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=4, facecolors="none",
                 edgecolors=MUTED, linewidths=0.5, depthshade=False,
                 label="nodes")

    axis.scatter(*position[0], color=BLUE, s=42, marker="o", depthshade=False,
                 label="start")
    axis.scatter(*position[-1], color=VERMILLION, s=95, marker="*",
                 depthshade=False, label="end")

    limit = 1.12 * max(float(np.linalg.norm(position, axis=1).max()), geo_radius)
    vertical = 1.6 * earth_radius
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-vertical, vertical)
    axis.set_box_aspect((1.0, 1.0, 0.30))
    axis.set_xlabel("ECI $x$ [km]")
    axis.set_ylabel("ECI $y$ [km]")
    axis.set_zlabel("ECI $z$ [km]")
    for pane_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        pane_axis.pane.set_facecolor("white")
        pane_axis.pane.set_edgecolor(GRID)
        pane_axis.pane.set_alpha(1.0)
    axis.grid(True)
    return collection


def _build_3d_figure(solution: SolarSailSolution, report: dict | None,
                     annotated: bool = True, draw_shadow: bool = True):
    figure = plt.figure(figsize=(7.6, 4.9))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("white")
    collection = _draw_scene(axis, solution, draw_shadow=draw_shadow)

    bar = figure.colorbar(collection, ax=axis, shrink=0.55, pad=0.02)
    bar.set_label("elapsed time [days]")
    bar.outline.set_edgecolor(GRID)

    axis.legend(loc="upper left", frameon=False)
    axis.view_init(elev=26, azim=42)
    _annotate(figure, solution, report, y=0.99, annotated=annotated)
    return figure, axis


def render_trajectory_3d(solution: SolarSailSolution, path: str | Path,
                         report: dict | None = None,
                         annotated: bool = True) -> Path:
    """3D view with the Earth and the cylindrical shadow."""
    with plt.rc_context(_RC):
        figure, _ = _build_3d_figure(solution, report, annotated)
        return _save(figure, path)


#: Animation frames per revolution.  Below ~12 the sail jumps between widely
#: separated points and the orbit is impossible to follow; the cost is linear in
#: this, so it is the main lever on file size and render time.
_FRAMES_PER_REVOLUTION = 16


def render_shadow_animation(solution: SolarSailSolution, path: str | Path,
                            frames: int | None = None,
                            report: dict | None = None) -> Path:
    """Animate the transfer from a fixed viewpoint, with the shadow moving.

    The camera does not rotate. What moves is the physics: the Sun line sweeps
    about +Z at roughly one degree per day, so the shadow cylinder turns with it
    and the eclipse crossings migrate around the orbit. A rotating camera hides
    exactly that, because it is impossible to tell which motion is the scene and
    which is the viewpoint.

    The trajectory is drawn progressively up to the current time, so the sail
    and the shadow it is crossing are always in the same frame.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    revolutions = float(
        (solution.dense_longitude_rad[-1] - solution.dense_longitude_rad[0])
        / (2.0 * np.pi))
    if frames is None or frames < _FRAMES_PER_REVOLUTION * revolutions:
        frames = int(round(_FRAMES_PER_REVOLUTION * revolutions))

    seconds = solution.dense_state[:, 5]
    position = solution.dense_position_km
    lit = solution.dense_illuminated
    days = solution.dense_elapsed_days
    total = float(seconds[-1])

    # a tight bounding box would resize the canvas frame to frame
    with plt.rc_context({**_RC, "savefig.bbox": None}):
        # draw_shadow=False: the animation owns the shadow, and drawing it
        # once here as well would leave the initial cone frozen in every frame
        # alongside the moving one
        figure, axis = _build_3d_figure(solution, report, draw_shadow=False)

        shadow = _shadow_surfaces(axis, solution, 0.0)
        # a Line3DCollection must be non-empty when it is added to a 3D axis,
        # so both trails start as the first (degenerate) segment
        seed = [np.stack((position[0], position[0]))]
        trail_lit = Line3DCollection(seed, cmap=TRAJECTORY_CMAP, linewidth=1.6)
        trail_lit.set_array(np.array([days[0]]))
        trail_lit.set_clim(float(days[0]), float(days[-1]))
        axis.add_collection3d(trail_lit)
        trail_dark = Line3DCollection(list(seed), colors=VERMILLION, linewidth=2.2)
        axis.add_collection3d(trail_dark)
        marker = axis.scatter(*position[0], color=BLUE, s=34, depthshade=False)
        clock = axis.text2D(0.02, 0.02, "", transform=axis.transAxes,
                            fontsize=9, color=MUTED)

        def draw(frame: int):
            nonlocal shadow, marker
            now = total * (frame + 1) / frames
            # clamp both ways: the final frame lands exactly on the last sample
            upto = int(np.searchsorted(seconds, now)) + 1
            upto = min(max(upto, 2), len(position))

            for artist in shadow:
                artist.remove()
            shadow = _shadow_surfaces(axis, solution, now)

            points = position[:upto]
            segments = np.stack((points[:-1], points[1:]), axis=1)
            mid = 0.5 * (days[:upto][:-1] + days[:upto][1:])
            sunlit = lit[:upto][:-1] & lit[:upto][1:]
            lit_segments = list(segments[sunlit])
            trail_lit.set_segments(lit_segments or seed)
            trail_lit.set_array(mid[sunlit] if lit_segments else np.array([days[0]]))
            trail_dark.set_segments(list(segments[~sunlit]) or seed)

            marker.remove()
            marker = axis.scatter(*position[upto - 1], color=BLUE, s=34,
                                  depthshade=False, zorder=6)
            clock.set_text(f"t = {now / _SECONDS_PER_DAY:5.2f} d"
                           f"   Sun line swept {np.rad2deg(solution.sun_angular_rate_rad_s * now):4.1f} deg")
            return (axis,)

        movie = animation.FuncAnimation(figure, draw, frames=frames, blit=False)
        movie.save(path, writer=animation.PillowWriter(fps=14), dpi=110)
        plt.close(figure)
    return path


#: Kept under the old name so existing callers and configs keep working.
render_rotating_gif = render_shadow_animation


# ---------------------------------------------------------------------------
# combined sheet
# ---------------------------------------------------------------------------

def render_summary(solution: SolarSailSolution, path: str | Path,
                   report: dict | None = None, annotated: bool = True) -> Path:
    """Everything on one sheet -- convenient for a talk or a quick look."""
    days = solution.dense_elapsed_days
    node_days = solution.elapsed_days
    eclipses = _eclipse_intervals_dense(solution)

    with plt.rc_context(_RC):
        figure = plt.figure(figsize=(13.5, 7.4))
        # Without the header the panels reclaim the space it occupied, so the
        # clean version is a full sheet of plots rather than one with a gap.
        grid = figure.add_gridspec(2, 3, hspace=0.42, wspace=0.28, left=0.06,
                                   right=0.98, top=0.86 if annotated else 0.96,
                                   bottom=0.09)

        axis = figure.add_subplot(grid[0, 0])
        _shade_eclipses(axis, eclipses)
        axis.axhline(solution.target_perigee_km, color=VERMILLION, linestyle=":",
                     linewidth=1.3, label="perigee requirement")
        axis.plot(days, solution.dense_apogee_radius_km, color=ORANGE, label="apogee")
        axis.plot(days, solution.dense_perigee_radius_km, color=BLUE, label="perigee")
        axis.scatter(node_days, solution.perigee_radius_km, edgecolors=BLUE, **_NODE_STYLE)
        axis.set_ylabel("radius [km]")
        axis.set_title("Apsides")
        axis.legend(loc="lower right")

        axis = figure.add_subplot(grid[0, 1])
        _shade_eclipses(axis, eclipses)
        axis.axhline(solution.target_eccentricity, color=VERMILLION,
                     linestyle=":", linewidth=1.3, label="requirement")
        axis.plot(days, solution.dense_eccentricity, color=BLUE)
        axis.scatter(node_days, solution.eccentricity, edgecolors=BLUE, **_NODE_STYLE)
        axis.set_ylabel("eccentricity [-]")
        axis.set_title("Eccentricity")
        axis.legend(loc="lower right")

        axis = figure.add_subplot(grid[0, 2])
        _shade_eclipses(axis, eclipses)
        axis.plot(days, solution.dense_state[:, 0], color=GREEN)
        axis.scatter(node_days, solution.state[:, 0], edgecolors=GREEN, **_NODE_STYLE)
        axis.set_ylabel(r"$p - r_{\mathrm{GEO}}$ [km]")
        axis.set_title("Semi-latus rectum")

        axis = figure.add_subplot(grid[1, 0])
        _shade_eclipses(axis, eclipses)
        cone = np.rad2deg(solution.dense_cone_angle_rad).astype(float)
        cone[~solution.dense_illuminated] = np.nan
        axis.plot(days, _wrap_deg_gapped(solution.dense_steering_angle_rad), color=BLUE,
                  label=r"$\theta$")
        axis.plot(days, cone, color=ORANGE, label=r"$\alpha$")
        axis.axhline(90.0, color=MUTED, linewidth=0.8, linestyle=":")
        axis.set_ylim(-190.0, 190.0)
        axis.set_yticks([-180, -90, 0, 90, 180])
        axis.set_ylabel("angle [deg]")
        axis.set_title("Sail steering")
        axis.legend(loc="upper left", ncol=3)

        axis = figure.add_subplot(grid[1, 1])
        numbers = np.arange(1, len(solution.phase_duration_h) + 1)
        colours = [ORANGE if lit else BLUE for lit in solution.phase_sunlit]
        axis.bar(numbers, solution.phase_duration_h, color=colours, width=0.78)
        axis.axhline(0.0, color=TEXT, linewidth=0.8)
        shortest = float(np.min(solution.phase_duration_h))
        axis.set_ylim(min(0.0, shortest) * 1.15 - 0.05,
                      float(np.max(solution.phase_duration_h)) * 1.28)
        axis.set_ylabel("duration [h]")
        axis.set_xlabel("phase")
        axis.set_title(f"Phase durations (shortest {shortest:.3f} h)")
        axis.grid(axis="x", visible=False)
        axis.legend(handles=[Line2D([], [], color=ORANGE, linewidth=6, label="sunlit"),
                             Line2D([], [], color=BLUE, linewidth=6, label="eclipse")],
                    loc="upper center", ncol=2)

        axis = figure.add_subplot(grid[1, 2])
        defect = np.asarray(solution.defect_history, dtype=float)
        if defect.size:
            iterations = np.arange(1, defect.size + 1)
            axis.semilogy(iterations, np.maximum(defect, 1e-16), color=BLUE,
                          label="max dynamics defect")
            steps = np.asarray(solution.step_history, dtype=float)
            if steps.size == defect.size and np.isfinite(steps).any():
                axis.semilogy(iterations, np.maximum(steps, 1e-16), color=ORANGE,
                              label=r"step / $\epsilon$")
            axis.axhline(1.0e-4, color=VERMILLION, linestyle=":", linewidth=1.3,
                         label=r"tolerance $10^{-4}$")
            axis.set_xlim(1, defect.size)
            axis.set_xlabel("SCP iteration")
            axis.legend(loc="upper right")
        axis.set_title("SCP convergence")

        for index in (0, 1, 2):
            figure.axes[index].set_xlabel("elapsed time [days]")
        figure.axes[3].set_xlabel("elapsed time [days]")

        if annotated:
            figure.suptitle("Multi-phase solar-sail GEO disposal", fontsize=15,
                            x=0.06, ha="left", y=0.965)
            figure.text(0.06, 0.915, _caption(solution, report), fontsize=9.2,
                        color=MUTED, ha="left")
            label, colour = _status(report)
            if label:
                figure.text(0.98, 0.955, label, fontsize=10, color=colour,
                            ha="right", weight="bold")
        return _save(figure, path)


# ---------------------------------------------------------------------------
# convenience
# ---------------------------------------------------------------------------

FIGURES = {
    "elements.png": render_elements,
    "control.png": render_control,
    "transfer_plane.png": render_transfer_plane,
    "trajectory_3d.png": render_trajectory_3d,
    "phases.png": render_phases,
    "convergence.png": render_convergence,
    "summary.png": render_summary,
}


PAPER_SUBDIRECTORY = "paper"


def render_all(solution: SolarSailSolution, directory: str | Path,
               report: dict | None = None) -> list[Path]:
    """Write every static figure twice.

    ``directory``            annotated: each figure carries the settings it came
                             from (A/m, revolutions, elapsed time, validation
                             status), so a figure found later is self-describing.
    ``directory/paper``      unannotated: identical axes, no text outside them.
                             The annotation sits beyond the axes and so changes
                             the tight bounding box; dropping it gives a clean
                             crop that drops straight into a paper without
                             disturbing the surrounding layout.
    """
    directory = Path(directory)
    paper = directory / PAPER_SUBDIRECTORY
    written = []
    for name, render in FIGURES.items():
        written.append(render(solution, directory / name, report, annotated=True))
        written.append(render(solution, paper / name, report, annotated=False))
    return written


def render_saved_solution(solution_file: str | Path, directory: str | Path):
    """Regenerate the figures from a saved .npz without rerunning the solver."""
    return render_all(load_solution(solution_file), directory)
