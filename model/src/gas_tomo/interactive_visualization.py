from __future__ import annotations

"""Browser-based 3-D visualisation for dual-view gas tomography.

The numerical reconstruction always remains at full resolution.  Only the copy
sent to the WebGL viewer may be decimated when a very large volume would make a
standalone HTML file unnecessarily heavy.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable
import webbrowser

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from .grid import VolumeGrid
from .surfaces import resolve_levels, surface_mesh


@dataclass(frozen=True)
class InteractiveViewInfo:
    output_path: Path
    original_shape_zyx: tuple[int, int, int]
    display_shape_zyx: tuple[int, int, int]
    maximum: float
    maximum_location_m: tuple[float, float, float]
    centroid_location_m: tuple[float, float, float] | None


def load_reconstruction_bundle(
    bundle_path: str | Path,
) -> tuple[np.ndarray, VolumeGrid]:
    """Load ``concentration`` and its physical grid from a result bundle."""
    path = Path(bundle_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reconstruction bundle not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {"concentration", "bounds_m", "shape_nx_ny_nz"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"Bundle {path} is missing required arrays: {sorted(missing)}"
            )
        volume = np.asarray(data["concentration"], dtype=np.float32)
        bounds = np.asarray(data["bounds_m"], dtype=np.float64)
        shape = tuple(int(value) for value in data["shape_nx_ny_nz"][:3])

    grid = VolumeGrid(bounds=bounds, shape=shape)
    if volume.shape != grid.array_shape:
        raise ValueError(
            f"Volume shape {volume.shape} does not match grid array shape "
            f"{grid.array_shape}."
        )
    return volume, grid


def _normalise_plotly_include(value: Any) -> bool | str:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {
        "inline",
        "true",
        "yes",
        "1",
        "embedded",
        "embed",
        "self-contained",
        "self_contained",
    }:
        return True
    if text in {"cdn", "directory"} or text.endswith(".js"):
        return text
    if text in {"false", "no", "0", "none"}:
        return False
    raise ValueError(
        "interactive_include_plotlyjs must be inline, cdn, directory, a .js path, "
        "or a boolean."
    )


def _index_with_endpoint(length: int, step: int) -> np.ndarray:
    indices = np.arange(0, length, step, dtype=np.int64)
    if indices.size == 0 or indices[-1] != length - 1:
        indices = np.append(indices, length - 1)
    return np.unique(indices)


def _decimate_for_display(
    volume: np.ndarray,
    grid: VolumeGrid,
    max_voxels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if 0 < max_voxels < 8:
        raise ValueError("A 3-D display retaining endpoints needs >=8 voxels.")
    x, y, z = grid.centers_1d()
    if max_voxels <= 0 or volume.size <= max_voxels:
        return volume, x, y, z

    step = max(1, int(np.ceil((volume.size / max_voxels) ** (1.0 / 3.0))))
    while True:
        ix = _index_with_endpoint(grid.nx, step)
        iy = _index_with_endpoint(grid.ny, step)
        iz = _index_with_endpoint(grid.nz, step)
        display = volume[np.ix_(iz, iy, ix)]
        if display.size <= max_voxels:
            return display, x[ix], y[iy], z[iz]
        step += 1


def _box_trace(bounds: np.ndarray, name: str = "reconstruction domain") -> go.Scatter3d:
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    zmin, zmax = bounds[2]
    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float64,
    )
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for start, end in edges:
        xs.extend([float(corners[start, 0]), float(corners[end, 0]), None])
        ys.extend([float(corners[start, 1]), float(corners[end, 1]), None])
        zs.extend([float(corners[start, 2]), float(corners[end, 2]), None])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": "rgba(35,35,35,0.75)", "width": 3},
        name=name,
        hoverinfo="skip",
        showlegend=True,
    )


def _centroid(volume: np.ndarray, grid: VolumeGrid) -> tuple[float, float, float] | None:
    positive = np.clip(
        np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        None,
    )
    total = float(np.sum(positive, dtype=np.float64))
    if total <= 0.0:
        return None
    x, y, z = grid.centers_1d()
    marginal_x = np.sum(positive, axis=(0, 1), dtype=np.float64)
    marginal_y = np.sum(positive, axis=(0, 2), dtype=np.float64)
    marginal_z = np.sum(positive, axis=(1, 2), dtype=np.float64)
    return (
        float(np.dot(marginal_x, x) / total),
        float(np.dot(marginal_y, y) / total),
        float(np.dot(marginal_z, z) / total),
    )


def _point_cloud_trace(
    volume: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    minimum: float,
    maximum_points: int,
) -> go.Scatter3d:
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    flat_values = volume.ravel()
    eligible = np.flatnonzero(flat_values >= minimum)
    if eligible.size > maximum_points > 0:
        values = flat_values[eligible]
        keep_local = np.argpartition(values, -maximum_points)[-maximum_points:]
        eligible = eligible[keep_local]
    return go.Scatter3d(
        x=xx.ravel()[eligible],
        y=yy.ravel()[eligible],
        z=zz.ravel()[eligible],
        mode="markers",
        marker={
            "size": 3.5,
            "opacity": 0.38,
            "color": flat_values[eligible],
            "coloraxis": "coloraxis",
        },
        name="high-concentration voxels",
        visible=False,
        hovertemplate=(
            "X=%{x:.3f} m<br>Y=%{y:.3f} m<br>Z=%{z:.3f} m"
            "<br>Concentration=%{marker.color:.5g}<extra></extra>"
        ),
    )


def create_interactive_3d_figure(
    volume: np.ndarray,
    grid: VolumeGrid,
    visual_cfg: dict[str, Any] | None = None,
    title: str = "Dual-view reconstructed 3-D gas concentration field",
) -> tuple[go.Figure, InteractiveViewInfo]:
    """Build an interactive volume/isosurface/slice/voxel Plotly figure."""
    cfg = dict(visual_cfg or {})
    array = np.asarray(volume, dtype=np.float32)
    if array.shape != grid.array_shape:
        raise ValueError(f"Expected volume shape {grid.array_shape}; got {array.shape}.")
    clean = np.clip(
        np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        None,
    )
    maximum = float(np.max(clean, initial=0.0))
    maximum_location = (
        grid.max_location(clean)
        if maximum > 0.0
        else tuple(float(np.mean(axis_bounds)) for axis_bounds in grid.bounds)
    )
    centroid = _centroid(clean, grid)

    display, x, y, z = _decimate_for_display(
        clean,
        grid,
        max_voxels=int(
            cfg.get("interactive_max_voxels", cfg.get("interactive_max_points", 180_000))
        ),
    )
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    low_fraction = float(
        np.clip(cfg.get("interactive_threshold_fraction", 0.03), 0.0, 0.99)
    )
    iso_low_fraction = float(
        np.clip(cfg.get("interactive_isosurface_low_fraction", 0.10), 0.0, 0.99)
    )
    iso_high_fraction = float(
        np.clip(
            cfg.get("interactive_isosurface_high_fraction", 0.85),
            iso_low_fraction + 1.0e-4,
            1.0,
        )
    )
    surface_count = max(2, int(cfg.get("interactive_surface_count", 20)))
    iso_surface_count = max(2, int(cfg.get("interactive_isosurface_count", 6)))
    volume_opacity = float(
        np.clip(cfg.get("interactive_volume_opacity", 0.12), 0.01, 1.0)
    )
    iso_opacity = float(
        np.clip(cfg.get("interactive_isosurface_opacity", 0.38), 0.01, 1.0)
    )
    colorscale = str(cfg.get("interactive_colorscale", "Turbo"))
    point_limit = max(100, int(cfg.get("interactive_point_limit", 20_000)))

    figure = go.Figure()
    trace_names: list[str] = []

    if maximum > 0.0:
        minimum = maximum * low_fraction
        figure.add_trace(
            go.Volume(
                x=xx.ravel(),
                y=yy.ravel(),
                z=zz.ravel(),
                value=display.ravel(),
                isomin=minimum,
                isomax=maximum,
                surface_count=surface_count,
                opacity=volume_opacity,
                opacityscale=[
                    [0.0, 0.0],
                    [0.10, 0.0],
                    [0.25, 0.025],
                    [0.50, 0.10],
                    [0.75, 0.25],
                    [1.0, 0.62],
                ],
                caps={"x_show": False, "y_show": False, "z_show": False},
                slices={"x_show": False, "y_show": False, "z_show": False},
                coloraxis="coloraxis",
                name="volume rendering",
                hovertemplate=(
                    "X=%{x:.3f} m<br>Y=%{y:.3f} m<br>Z=%{z:.3f} m"
                    "<br>Concentration=%{value:.5g}<extra></extra>"
                ),
            )
        )
        trace_names.append("volume")

        for level in resolve_levels(clean, cfg):
            vertices, faces = surface_mesh(clean, grid, level)
            if faces.size:
                figure.add_trace(go.Mesh3d(
                    x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
                    i=faces[:,0], j=faces[:,1], k=faces[:,2],
                    intensity=np.full(len(vertices), level), coloraxis="coloraxis",
                    name=f"isosurface {level:.6g}", opacity=iso_opacity, visible=False,
                    hovertemplate="X=%{x:.4f} m<br>Y=%{y:.4f} m<br>Z=%{z:.4f} m<extra></extra>"))
                trace_names.append("isosurface")

        iz, iy, ix = np.unravel_index(np.argmax(display), display.shape)
        x2d_xy, y2d_xy = np.meshgrid(x, y, indexing="xy")
        figure.add_trace(
            go.Surface(
                x=x2d_xy,
                y=y2d_xy,
                z=np.full_like(x2d_xy, z[iz], dtype=np.float64),
                surfacecolor=display[iz, :, :],
                coloraxis="coloraxis",
                opacity=0.92,
                name=f"XY slice, z={z[iz]:.3f} m",
                visible=False,
                showscale=False,
                hovertemplate=(
                    "X=%{x:.3f} m<br>Y=%{y:.3f} m<br>Z=%{z:.3f} m"
                    "<br>Concentration=%{surfacecolor:.5g}<extra></extra>"
                ),
            )
        )
        trace_names.append("slice_xy")

        x2d_xz, z2d_xz = np.meshgrid(x, z, indexing="xy")
        figure.add_trace(
            go.Surface(
                x=x2d_xz,
                y=np.full_like(x2d_xz, y[iy], dtype=np.float64),
                z=z2d_xz,
                surfacecolor=display[:, iy, :],
                coloraxis="coloraxis",
                opacity=0.92,
                name=f"XZ slice, y={y[iy]:.3f} m",
                visible=False,
                showscale=False,
                hovertemplate=(
                    "X=%{x:.3f} m<br>Y=%{y:.3f} m<br>Z=%{z:.3f} m"
                    "<br>Concentration=%{surfacecolor:.5g}<extra></extra>"
                ),
            )
        )
        trace_names.append("slice_xz")

        y2d_yz, z2d_yz = np.meshgrid(y, z, indexing="xy")
        figure.add_trace(
            go.Surface(
                x=np.full_like(y2d_yz, x[ix], dtype=np.float64),
                y=y2d_yz,
                z=z2d_yz,
                surfacecolor=display[:, :, ix],
                coloraxis="coloraxis",
                opacity=0.92,
                name=f"YZ slice, x={x[ix]:.3f} m",
                visible=False,
                showscale=False,
                hovertemplate=(
                    "X=%{x:.3f} m<br>Y=%{y:.3f} m<br>Z=%{z:.3f} m"
                    "<br>Concentration=%{surfacecolor:.5g}<extra></extra>"
                ),
            )
        )
        trace_names.append("slice_yz")

        figure.add_trace(
            _point_cloud_trace(display, x, y, z, minimum, point_limit)
        )
        trace_names.append("points")

        figure.add_trace(
            go.Scatter3d(
                x=[maximum_location[0]],
                y=[maximum_location[1]],
                z=[maximum_location[2]],
                mode="markers+text",
                marker={
                    "size": 7,
                    "symbol": "diamond",
                    "color": "white",
                    "line": {"color": "black", "width": 2},
                },
                text=["maximum"],
                textposition="top center",
                name="maximum concentration",
                hovertemplate=(
                    f"Maximum={maximum:.6g}<br>X={maximum_location[0]:.4f} m"
                    f"<br>Y={maximum_location[1]:.4f} m"
                    f"<br>Z={maximum_location[2]:.4f} m<extra></extra>"
                ),
            )
        )
        trace_names.append("maximum")

        if centroid is not None:
            figure.add_trace(
                go.Scatter3d(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    z=[centroid[2]],
                    mode="markers+text",
                    marker={
                        "size": 6,
                        "symbol": "circle",
                        "color": "black",
                        "line": {"color": "white", "width": 1},
                    },
                    text=["centroid"],
                    textposition="bottom center",
                    name="concentration centroid",
                    hovertemplate=(
                        f"Centroid<br>X={centroid[0]:.4f} m"
                        f"<br>Y={centroid[1]:.4f} m"
                        f"<br>Z={centroid[2]:.4f} m<extra></extra>"
                    ),
                )
            )
            trace_names.append("centroid")

    figure.add_trace(_box_trace(grid.bounds))
    trace_names.append("box")

    buttons: list[dict[str, Any]] = []
    if maximum > 0.0:
        always = {"maximum", "centroid", "box"}

        def visibility(active: Iterable[str]) -> list[bool]:
            selected = set(active)
            return [name in selected or name in always for name in trace_names]

        buttons = [
            {
                "label": "体渲染",
                "method": "update",
                "args": [
                    {"visible": visibility({"volume"})},
                    {"title": f"{title} — 体渲染"},
                ],
            },
            {
                "label": "多层等值面",
                "method": "update",
                "args": [
                    {"visible": visibility({"isosurface"})},
                    {"title": f"{title} — 多层等值面"},
                ],
            },
            {
                "label": "正交切片",
                "method": "update",
                "args": [
                    {"visible": visibility({"slice_xy", "slice_xz", "slice_yz"})},
                    {"title": f"{title} — 最大值处正交切片"},
                ],
            },
            {
                "label": "体素点云",
                "method": "update",
                "args": [
                    {"visible": visibility({"points"})},
                    {"title": f"{title} — 高浓度体素点云"},
                ],
            },
            {
                "label": "体渲染 + 切片",
                "method": "update",
                "args": [
                    {
                        "visible": visibility(
                            {"volume", "slice_xy", "slice_xz", "slice_yz"}
                        )
                    },
                    {"title": f"{title} — 体渲染与正交切片"},
                ],
            },
        ]

    extents = grid.bounds[:, 1] - grid.bounds[:, 0]
    aspect = extents / np.max(extents)
    shape_note = (
        f"full grid: {grid.nz}×{grid.ny}×{grid.nx}; "
        f"display grid: {display.shape[0]}×{display.shape[1]}×{display.shape[2]}"
    )
    if maximum > 0.0:
        centroid_text = (
            "none"
            if centroid is None
            else f"({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}) m"
        )
        summary = (
            f"max={maximum:.6g} at "
            f"({maximum_location[0]:.3f}, {maximum_location[1]:.3f}, "
            f"{maximum_location[2]:.3f}) m"
            f"<br>centroid={centroid_text}<br>{shape_note}"
        )
    else:
        summary = f"The volume contains no positive concentration.<br>{shape_note}"

    figure.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        coloraxis={
            "colorscale": colorscale,
            "cmin": (cfg.get("color_range") or [0.0, 1.0])[0],
            "cmax": (cfg.get("color_range") or [0.0, maximum if maximum > 0 else 1.0])[1],
            "colorbar": {"title": cfg.get("concentration_unit", "relative response / m"), "thickness": 18},
        },
        scene={
            "xaxis": {"title": "X (m)", "range": list(grid.bounds[0])},
            "yaxis": {"title": "Y (m)", "range": list(grid.bounds[1])},
            "zaxis": {"title": "Z (m)", "range": list(grid.bounds[2])},
            "aspectmode": "manual",
            "aspectratio": {
                "x": float(aspect[0]),
                "y": float(aspect[1]),
                "z": float(aspect[2]),
            },
            "camera": {"eye": {"x": 1.55, "y": -1.55, "z": 1.15}},
            "dragmode": "orbit",
        },
        margin={"l": 0, "r": 0, "t": 85, "b": 0},
        legend={"x": 0.01, "y": 0.98, "bgcolor": "rgba(255,255,255,0.72)"},
        annotations=[
            {
                "text": summary,
                "xref": "paper",
                "yref": "paper",
                "x": 0.99,
                "y": 0.99,
                "xanchor": "right",
                "yanchor": "top",
                "showarrow": False,
                "align": "left",
                "bgcolor": "rgba(255,255,255,0.78)",
                "bordercolor": "rgba(80,80,80,0.35)",
                "borderwidth": 1,
                "font": {"size": 12},
            },
            {
                "text": "鼠标左键旋转 · 滚轮缩放 · 悬停读取坐标/浓度 · 双击重置",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.01,
                "xanchor": "center",
                "yanchor": "bottom",
                "showarrow": False,
                "font": {"size": 12},
                "bgcolor": "rgba(255,255,255,0.70)",
            },
        ],
        updatemenus=(
            [
                {
                    "type": "dropdown",
                    "direction": "down",
                    "x": 0.01,
                    "y": 1.08,
                    "xanchor": "left",
                    "yanchor": "top",
                    "showactive": True,
                    "active": 0,
                    "buttons": buttons,
                }
            ]
            if buttons
            else []
        ),
        uirevision="dualview-gas-3d",
    )

    info = InteractiveViewInfo(
        output_path=Path(),
        original_shape_zyx=tuple(int(value) for value in clean.shape),
        display_shape_zyx=tuple(int(value) for value in display.shape),
        maximum=maximum,
        maximum_location_m=tuple(float(value) for value in maximum_location),
        centroid_location_m=(
            None if centroid is None else tuple(float(value) for value in centroid)
        ),
    )
    return figure, info


def save_interactive_3d_html(
    volume: np.ndarray,
    grid: VolumeGrid,
    output_path: str | Path,
    visual_cfg: dict[str, Any] | None = None,
    title: str = "Dual-view reconstructed 3-D gas concentration field",
    self_contained: bool | None = None,
    include_plotlyjs: bool | str | None = None,
) -> InteractiveViewInfo:
    """Write a rotatable WebGL viewer as an HTML file."""
    cfg = dict(visual_cfg or {})
    figure, info = create_interactive_3d_figure(volume, grid, cfg, title=title)
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if include_plotlyjs is None:
        if self_contained is not None:
            include_plotlyjs = bool(self_contained)
        else:
            include_plotlyjs = _normalise_plotly_include(
                cfg.get("interactive_include_plotlyjs", "inline")
            )

    pio.write_html(
        figure,
        file=str(path),
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        auto_open=False,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": path.stem,
                "height": 900,
                "width": 1200,
                "scale": 2,
            },
        },
    )
    return InteractiveViewInfo(
        output_path=path,
        original_shape_zyx=info.original_shape_zyx,
        display_shape_zyx=info.display_shape_zyx,
        maximum=info.maximum,
        maximum_location_m=info.maximum_location_m,
        centroid_location_m=info.centroid_location_m,
    )


def open_interactive_html(path: str | Path) -> bool:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Interactive HTML not found: {resolved}")
    return bool(webbrowser.open(resolved.as_uri()))


def _ray_box_segment(
    origin: np.ndarray,
    direction: np.ndarray,
    bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a forward ray to an axis-aligned reconstruction box."""
    t_enter = -np.inf
    t_exit = np.inf
    for axis in range(3):
        component = float(direction[axis])
        if abs(component) < 1.0e-12:
            if origin[axis] < bounds[axis, 0] or origin[axis] > bounds[axis, 1]:
                return None
            continue
        t0 = (bounds[axis, 0] - origin[axis]) / component
        t1 = (bounds[axis, 1] - origin[axis]) / component
        t_enter = max(t_enter, min(t0, t1))
        t_exit = min(t_exit, max(t0, t1))
        if t_exit <= t_enter:
            return None
    start_t = max(float(t_enter), 0.0)
    if not np.isfinite(t_exit) or t_exit <= start_t:
        return None
    return origin + start_t * direction, origin + float(t_exit) * direction


def create_camera_geometry_figure(
    cameras: Any,
    grid: VolumeGrid,
    visual_cfg: dict[str, Any] | None = None,
) -> go.Figure:
    """Create an interactive figure showing both cameras and representative rays."""
    cfg = dict(visual_cfg or {})
    camera_list = list(cameras or [])
    if not camera_list:
        raise ValueError("At least one camera is required for camera geometry output.")
    ray_grid_raw = cfg.get("geometry_ray_grid", [9, 9])
    rows, cols = int(ray_grid_raw[0]), int(ray_grid_raw[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("geometry_ray_grid must contain two positive integers.")

    figure = go.Figure()
    palette = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD"]
    all_points = [grid.bounds[:, 0], grid.bounds[:, 1]]

    for camera_index, camera in enumerate(camera_list):
        color = palette[camera_index % len(palette)]
        origins, directions, _ = camera.measurement_rays(
            (rows, cols), samples_per_bin=1
        )
        line_x: list[float | None] = []
        line_y: list[float | None] = []
        line_z: list[float | None] = []
        for origin, direction in zip(origins, directions):
            segment = _ray_box_segment(origin, direction, grid.bounds)
            if segment is None:
                continue
            entry, exit_ = segment
            line_x.extend([float(origin[0]), float(entry[0]), float(exit_[0]), None])
            line_y.extend([float(origin[1]), float(entry[1]), float(exit_[1]), None])
            line_z.extend([float(origin[2]), float(entry[2]), float(exit_[2]), None])
        figure.add_trace(
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                line={"width": 1.4, "color": color},
                opacity=0.30,
                name=f"{camera.name} 代表性射线",
                hoverinfo="skip",
            )
        )
        center = np.asarray(camera.center_world, dtype=np.float64)
        all_points.append(center)
        figure.add_trace(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode="markers+text",
                marker={"size": 8, "symbol": "square", "color": color},
                text=[camera.name],
                textposition="top center",
                name=camera.name,
                hovertemplate=(
                    f"{camera.name}<br>X={center[0]:.4f} m"
                    f"<br>Y={center[1]:.4f} m<br>Z={center[2]:.4f} m"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_trace(_box_trace(grid.bounds, name="一立方米重建域"))
    points = np.vstack(all_points)
    data_min = np.min(points, axis=0)
    data_max = np.max(points, axis=0)
    padding = 0.06 * np.maximum(data_max - data_min, 1.0e-6)
    extents = data_max - data_min + 2.0 * padding
    aspect = extents / np.max(extents)
    figure.update_layout(
        title={
            "text": "双相机三维几何与代表性重建射线",
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        scene={
            "xaxis": {
                "title": "X (m)",
                "range": [data_min[0] - padding[0], data_max[0] + padding[0]],
            },
            "yaxis": {
                "title": "Y (m)",
                "range": [data_min[1] - padding[1], data_max[1] + padding[1]],
            },
            "zaxis": {
                "title": "Z (m)",
                "range": [data_min[2] - padding[2], data_max[2] + padding[2]],
            },
            "aspectmode": "manual",
            "aspectratio": {
                "x": float(aspect[0]),
                "y": float(aspect[1]),
                "z": float(aspect[2]),
            },
            "camera": {"eye": {"x": 1.55, "y": -1.55, "z": 1.15}},
            "dragmode": "orbit",
        },
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.72)"},
        annotations=[
            {
                "text": (
                    f"每台相机显示 {rows}×{cols} 条代表性射线；"
                    "系统矩阵仍使用配置中的全部观测格与子射线。"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.01,
                "xanchor": "center",
                "yanchor": "bottom",
                "showarrow": False,
                "bgcolor": "rgba(255,255,255,0.70)",
            }
        ],
        uirevision="dualview-camera-geometry",
    )
    return figure


def save_camera_geometry_3d_html(
    cameras: Any,
    grid: VolumeGrid,
    output_path: str | Path,
    *,
    ray_grid: tuple[int, int] = (9, 9),
    include_plotlyjs: bool | str = True,
) -> Path:
    """Save camera centres, representative rays and the reconstruction box."""
    figure = create_camera_geometry_figure(
        cameras,
        grid,
        {"geometry_ray_grid": [int(ray_grid[0]), int(ray_grid[1])]},
    )
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(
        figure,
        file=str(path),
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        auto_open=False,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": path.stem,
                "height": 900,
                "width": 1200,
                "scale": 2,
            },
        },
    )
    return path



def save_interactive_3d_outputs(
    *,
    volume: np.ndarray,
    grid: VolumeGrid,
    cameras: Any,
    output_dir: str | Path,
    visual_cfg: dict[str, Any] | None = None,
    prefix: str = "reconstruction",
) -> dict[str, str]:
    """Create the concentration viewer, geometry viewer and metadata JSON."""
    cfg = dict(visual_cfg or {})
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    fractions = [
        float(value)
        for value in cfg.get(
            "interactive_isosurface_fractions", [0.10, 0.25, 0.50]
        )
    ]
    viewer_cfg = dict(cfg)
    viewer_cfg.setdefault(
        "interactive_threshold_fraction", float(cfg.get("volume_min_fraction", 0.03))
    )
    viewer_cfg.setdefault(
        "interactive_volume_opacity", float(cfg.get("volume_opacity", 0.08))
    )
    viewer_cfg.setdefault(
        "interactive_surface_count", int(cfg.get("volume_surface_count", 22))
    )
    viewer_cfg.setdefault(
        "interactive_isosurface_low_fraction", min(fractions) if fractions else 0.10
    )
    viewer_cfg.setdefault(
        "interactive_isosurface_high_fraction", max(fractions) if fractions else 0.85
    )
    viewer_cfg.setdefault(
        "interactive_isosurface_count", max(len(fractions), 3)
    )
    viewer_cfg.setdefault(
        "interactive_isosurface_opacity",
        float(cfg.get("interactive_isosurface_opacity", 0.38)),
    )
    viewer_cfg.setdefault(
        "interactive_max_voxels", int(cfg.get("interactive_max_points", 180_000))
    )
    viewer_cfg.setdefault(
        "interactive_point_limit", int(cfg.get("interactive_point_limit", 20_000))
    )
    viewer_cfg.setdefault(
        "interactive_colorscale", str(cfg.get("interactive_colorscale", "Turbo"))
    )

    include_option = _normalise_plotly_include(
        cfg.get("interactive_include_plotlyjs", "inline")
    )
    viewer_path = output_root / f"{prefix}_3d_interactive.html"
    if cfg.get("dashboard", True):
        from .viewer_dashboard import write_dashboard
        write_dashboard([{"volume": volume, "grid": grid, "cameras": cameras or cfg.get("_viewer_cameras", []),
            "metadata": cfg.get("_viewer_metadata", {}),
            "observed": cfg.get("_observed"), "predicted": cfg.get("_predicted"),
            "weights": cfg.get("_weights"), "measurement_shape": cfg.get("_measurement_shape"),
            "measurement_shapes": cfg.get("_measurement_shapes",{})}],
            viewer_path, cfg)
        display,*_ = _decimate_for_display(np.asarray(volume),grid,int(cfg.get("interactive_max_voxels",180000)))
        info = InteractiveViewInfo(viewer_path,tuple(volume.shape),tuple(display.shape),
            float(np.max(volume)),grid.max_location(volume),_centroid(volume,grid))
    else:
        info = save_interactive_3d_html(volume,grid,viewer_path,visual_cfg=viewer_cfg,
            title="二视角三维气体浓度重建",include_plotlyjs=include_option)
    outputs: dict[str, str] = {"viewer": str(info.output_path)}

    if bool(cfg.get("save_camera_geometry_3d", True)) and cameras:
        ray_grid_raw = cfg.get("geometry_ray_grid", [9, 9])
        ray_grid = (int(ray_grid_raw[0]), int(ray_grid_raw[1]))
        geometry_path = output_root / "camera_geometry_3d_interactive.html"
        save_camera_geometry_3d_html(
            cameras,
            grid,
            geometry_path,
            ray_grid=ray_grid,
            include_plotlyjs=include_option,
        )
        outputs["camera_geometry"] = str(geometry_path)

    metadata_path = output_root / f"{prefix}_3d_viewer_metadata.json"
    metadata = {
        "viewer": outputs["viewer"],
        "camera_geometry": outputs.get("camera_geometry"),
        "original_shape_zyx": list(info.original_shape_zyx),
        "display_shape_zyx": list(info.display_shape_zyx),
        "maximum": info.maximum,
        "requested_isosurface_levels": resolve_levels(volume, cfg),
        "concentration_unit": cfg.get("concentration_unit", "relative response / m"),
        "maximum_location_m": list(info.maximum_location_m),
        "centroid_location_m": (
            None
            if info.centroid_location_m is None
            else list(info.centroid_location_m)
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    outputs["metadata"] = str(metadata_path)
    return outputs
