from __future__ import annotations

import csv
import gzip
import json
import os
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mpl_colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from .config import dump_resolved_config
from .evaluation import reconstruction_metrics
from .grid import VolumeGrid
from .interactive_visualization import save_interactive_3d_outputs
from .reconstruction import ReconstructionResult
from .surfaces import resolve_levels, surface_mesh


def _save_image(path: Path, dpi: int) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_projection_diagnostics(
    result: ReconstructionResult,
    output_dir: Path,
    dpi: int,
) -> None:
    for camera in result.cameras:
        rows, cols = result.measurement_shapes.get(camera.name, result.measurement_shape)
        camera_slice = result.camera_slices[camera.name]
        observed = result.observed[camera_slice].reshape(rows, cols)
        predicted = result.predicted[camera_slice].reshape(rows, cols)
        residual = predicted - observed
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, image, title in zip(
            axes,
            (observed, predicted, residual),
            ("Observed", "Forward projection", "Residual"),
        ):
            display = axis.imshow(image, origin="upper", aspect="equal")
            axis.set_title(title)
            axis.set_xlabel("grid column")
            axis.set_ylabel("grid row")
            figure.colorbar(display, ax=axis, shrink=0.8)
        figure.suptitle(camera.name)
        _save_image(output_dir / f"projection_{camera.name}.png", dpi)


def save_volume_maps(
    volume: np.ndarray,
    result: ReconstructionResult,
    output_dir: Path,
    prefix: str,
    dpi: int,
) -> None:
    spacing = result.grid.spacing
    xy = np.sum(volume, axis=0) * spacing[2]
    xz = np.sum(volume, axis=1) * spacing[1]
    yz = np.sum(volume, axis=2) * spacing[0]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    maps = [xy, xz, yz]
    titles = ["XY, integrated along Z", "XZ, integrated along Y", "YZ, integrated along X"]
    labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]
    extents = [np.concatenate((result.grid.bounds[a],result.grid.bounds[b])).tolist() for a,b in ((0,1),(0,2),(1,2))]
    for axis, image, title, (xlabel, ylabel), extent in zip(axes, maps, titles, labels, extents):
        display = axis.imshow(image, origin="lower", aspect="auto", extent=extent)
        axis.set_title(title)
        axis.set_xlabel(xlabel+" (m)")
        axis.set_ylabel(ylabel+" (m)")
        figure.colorbar(display, ax=axis, shrink=0.8)
    _save_image(output_dir / f"{prefix}_integrated_projections.png", dpi)

    iz, iy, ix = np.unravel_index(np.argmax(volume), volume.shape)
    slices = [volume[iz, :, :], volume[:, iy, :], volume[:, :, ix]]
    titles = [f"XY slice z-index={iz}", f"XZ slice y-index={iy}", f"YZ slice x-index={ix}"]
    labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, image, title, (xlabel, ylabel), extent in zip(axes, slices, titles, labels, extents):
        display = axis.imshow(image, origin="lower", aspect="auto", extent=extent)
        axis.set_title(title)
        axis.set_xlabel(xlabel+" (m)")
        axis.set_ylabel(ylabel+" (m)")
        figure.colorbar(display, ax=axis, shrink=0.8)
    _save_image(output_dir / f"{prefix}_max_slices.png", dpi)


def _write_ply(path: Path, vertices_xyz: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {vertices_xyz.shape[0]}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write(f"element face {faces.shape[0]}\n")
        file.write("property list uchar int vertex_indices\nend_header\n")
        for vertex in vertices_xyz:
            file.write(f"{vertex[0]:.8g} {vertex[1]:.8g} {vertex[2]:.8g}\n")
        for face in faces:
            file.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def _surface_world_coordinates(
    volume: np.ndarray,
    grid: VolumeGrid,
    level: float,
) -> tuple[np.ndarray, np.ndarray]:
    return surface_mesh(volume, grid, level)


def _normalise_fractions(
    fractions: Iterable[float],
    primary_fraction: float,
) -> list[float]:
    values = [float(np.clip(value, 1.0e-4, 0.9999)) for value in fractions]
    values.append(float(np.clip(primary_fraction, 1.0e-4, 0.9999)))
    return sorted(set(round(value, 8) for value in values))


def save_multilevel_isosurfaces(
    volume: np.ndarray,
    grid: VolumeGrid,
    output_dir: str | Path,
    fractions: Sequence[float],
    primary_fraction: float,
    dpi: int,
    prefix: str = "reconstruction",
    absolute_levels: Sequence[float] | None = None,
) -> list[Path]:
    """Save a publication-ready static 3-D multi-isosurface PNG and PLY meshes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    volume = np.asarray(volume, dtype=np.float32)
    if volume.shape != grid.array_shape:
        raise ValueError(f"Expected volume shape {grid.array_shape}, got {volume.shape}.")

    maximum = float(np.max(volume, initial=0.0))
    minimum = float(np.min(volume))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return []

    fractions_clean = _normalise_fractions(fractions, primary_fraction)
    surfaces: list[tuple[float, np.ndarray, np.ndarray]] = []
    output_paths: list[Path] = []
    levels = [f * maximum for f in fractions_clean] if absolute_levels is None else absolute_levels
    for level in levels:
        fraction = level / maximum
        if not (minimum < level < maximum):
            continue
        vertices_xyz, faces = _surface_world_coordinates(volume, grid, level)
        surfaces.append((fraction, vertices_xyz, faces))
        mesh_path = output_dir / (f"{prefix}_isosurface_{fraction*100:.6g}pct.ply" if absolute_levels is None else f"{prefix}_isosurface_level_{level:.9g}.ply")
        _write_ply(mesh_path, vertices_xyz, faces)
        output_paths.append(mesh_path)
        if abs(fraction - primary_fraction) < 1.0e-7:
            compatibility_path = output_dir / f"{prefix}_isosurface.ply"
            _write_ply(compatibility_path, vertices_xyz, faces)
            output_paths.append(compatibility_path)

    if not surfaces:
        return output_paths

    figure = plt.figure(figsize=(8.2, 7.0))
    axis = figure.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis")
    normalizer = mpl_colors.Normalize(
        vmin=min(item[0] for item in surfaces),
        vmax=max(item[0] for item in surfaces),
    )

    for index, (fraction, vertices_xyz, faces) in enumerate(surfaces):
        progress = index / max(len(surfaces) - 1, 1)
        mesh = Poly3DCollection(
            vertices_xyz[faces],
            alpha=0.12 + 0.38 * progress,
            facecolor=cmap(normalizer(fraction)),
            edgecolor="none",
        )
        axis.add_collection3d(mesh)

    max_x, max_y, max_z = grid.max_location(volume)
    axis.scatter([max_x], [max_y], [max_z], s=36, marker="*", label="maximum")
    axis.set_xlim(*grid.bounds[0])
    axis.set_ylim(*grid.bounds[1])
    axis.set_zlim(*grid.bounds[2])
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title("3-D concentration: multi-level isosurfaces")
    axis.set_box_aspect(grid.bounds[:, 1] - grid.bounds[:, 0])
    axis.view_init(elev=24, azim=-58)
    axis.legend(loc="upper right")
    scalar = ScalarMappable(norm=normalizer, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axis, fraction=0.03, pad=0.08)
    colorbar.set_label("fraction of maximum concentration")
    png_path = output_dir / f"{prefix}_isosurface.png"
    _save_image(png_path, dpi)
    output_paths.append(png_path)
    return output_paths



def save_vtk_legacy(path: Path, volume: np.ndarray, result: ReconstructionResult) -> None:
    nx, ny, nz = result.grid.shape
    dx, dy, dz = result.grid.spacing
    origin = result.grid.bounds[:, 0] + 0.5 * result.grid.spacing
    with path.open("w", encoding="utf-8") as file:
        file.write("# vtk DataFile Version 3.0\n")
        file.write("dual-view gas concentration\n")
        file.write("ASCII\n")
        file.write("DATASET STRUCTURED_POINTS\n")
        file.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        file.write(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n")
        file.write(f"SPACING {dx} {dy} {dz}\n")
        file.write(f"POINT_DATA {nx * ny * nz}\n")
        file.write("SCALARS concentration float 1\n")
        file.write("LOOKUP_TABLE default\n")
        flat = volume.astype(np.float32).ravel()
        for start in range(0, flat.size, 8):
            file.write(" ".join(f"{value:.8g}" for value in flat[start:start + 8]) + "\n")


def save_csv_gz(path: Path, volume: np.ndarray, result: ReconstructionResult) -> None:
    centers = result.grid.centers()
    values = volume.ravel()
    with gzip.open(path, "wt", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x_m", "y_m", "z_m", "concentration"])
        for point, value in zip(centers, values):
            writer.writerow([f"{point[0]:.9g}", f"{point[1]:.9g}", f"{point[2]:.9g}", f"{value:.9g}"])


def save_convergence(result: ReconstructionResult, output_dir: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    offset = 0
    for level_index, level in enumerate(result.levels, start=1):
        entries = level.solver.history
        if not entries:
            continue
        iterations = [offset + entry.iteration for entry in entries]
        objectives = [entry.objective for entry in entries]
        axis.semilogy(iterations, objectives, marker="o", label=f"level {level_index}")
        offset += level.solver.iterations
    axis.set_xlabel("cumulative iteration")
    axis.set_ylabel("objective")
    axis.set_title("Solver convergence")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    _save_image(output_dir / "convergence.png", dpi)


def _write_json(path: Path, value: Any):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def save_result_bundle(
    result: ReconstructionResult, cfg: dict[str, Any], output_dir: str | Path,
    truth: np.ndarray | None = None, *, save_interactive_html_override: bool | None = None,
    export_profile: str | None = None,
) -> dict[str, Any]:
    """Save numerical results first; isolate all optional plotting/export failures.

    data = arrays + metrics; preview = data + HTML; full = all report exports.
    A zero-filled no-estimate placeholder is explicitly flagged in both NPZ and JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis = dict(cfg["visualization"])
    profile = export_profile or vis.get("export_profile", "full")
    if profile not in {"data", "preview", "full"}:
        raise ValueError("export_profile must be data, preview or full.")
    if not np.isfinite(result.volume).all():
        raise ValueError("Numerical reconstruction contains nonfinite values.")
    # This is a dedicated result directory. Remove only names owned by this exporter
    # so a failed/reduced re-export cannot leave old graphs next to new numbers.
    owned_patterns = ["reconstruction_isosurface*.ply", "reconstruction_isosurface.png",
        "reconstruction_3d_interactive.html", "reconstruction_3d_viewer_metadata.json",
        "camera_geometry_3d_interactive.html", "reconstruction_integrated_projections.png",
        "reconstruction_max_slices.png", "truth_integrated_projections.png", "truth_max_slices.png",
        "visual_hull_support_integrated_projections.png", "visual_hull_support_max_slices.png",
        "convergence.png", "concentration_3d.vtk", "concentration_points.csv.gz"]
    owned_patterns += [f"projection_{camera.name}.png" for camera in result.cameras]
    for pattern in owned_patterns:
        for stale in output_dir.glob(pattern):
            if stale.is_file():
                stale.unlink()
    metrics = reconstruction_metrics(result, truth)
    metrics["scene"] = cfg.get("scene", {})
    metrics["measurement_shapes"] = {n:list(v) for n,v in result.measurement_shapes.items()}
    metrics["units"] = cfg.get("units", {"calibrated": False, "concentration": "relative response / m"})
    metrics["levels"] = [{"level": i+1, "grid_shape_nx_ny_nz": list(level.grid.shape),
        "matrix_id": level.matrix.matrix_id, "matrix_nnz": int(level.matrix.matrix.nnz),
        "iterations": level.solver.iterations, "converged": level.solver.converged,
        "status": level.solver.status, "termination_reason": level.solver.termination_reason,
        "initial_norm": level.solver.initial_norm, "lipschitz": level.solver.lipschitz,
        "projected_gradient_relative": level.solver.projected_gradient_relative,
        "backtracking_steps": level.solver.backtracking_steps,
        "history": [e.__dict__ for e in level.solver.history]}
        for i,level in enumerate(result.levels)]
    _write_json(output_dir/"metrics.json", metrics)
    with (output_dir/"concentration_3d.npy.tmp").open("wb") as f:
        np.save(f, result.volume.astype(np.float32))
    os.replace(output_dir/"concentration_3d.npy.tmp", output_dir/"concentration_3d.npy")
    with (output_dir/"reconstruction_bundle.npz.tmp").open("wb") as f:
        np.savez_compressed(f, concentration=result.volume.astype(np.float32),
            observed=result.observed.astype(np.float32), predicted=result.predicted.astype(np.float32),
            weights=result.weights.astype(np.float32), bounds_m=result.grid.bounds,
            shape_nx_ny_nz=np.array(result.grid.shape, dtype=np.int32),
            measurement_shape=np.array(result.measurement_shape, dtype=np.int32),
            measurement_shapes_json=np.array(json.dumps({n:list(v) for n,v in result.measurement_shapes.items()})),
            has_estimate=np.array(result.quality.get("has_estimate", True)),
            metadata_json=np.array(json.dumps(metrics, ensure_ascii=False, allow_nan=False)),
            cameras_json=np.array(json.dumps([c.metadata() for c in result.cameras])))
    os.replace(output_dir/"reconstruction_bundle.npz.tmp", output_dir/"reconstruction_bundle.npz")
    dump_resolved_config(cfg, output_dir/"resolved_config.yaml")
    if truth is not None:
        np.save(output_dir/"truth_3d.npy", np.asarray(truth, dtype=np.float32))
    errors, completed = [], []
    def optional(name, fn):
        try:
            fn()
            completed.append(name)
        except Exception as error:
            errors.append({"stage": name, "type": type(error).__name__, "message": str(error)})
            warnings.warn(f"Optional export {name} failed: {error}", RuntimeWarning)
            import matplotlib.pyplot as plt
            plt.close("all")
    dpi = int(vis.get("dpi",160))
    if profile == "full":
        optional("projection_diagnostics", lambda: save_projection_diagnostics(result, output_dir, dpi))
        optional("volume_maps", lambda: save_volume_maps(result.volume, result, output_dir, "reconstruction", dpi))
        if truth is not None:
            optional("truth_maps", lambda: save_volume_maps(truth, result, output_dir, "truth", dpi))
        if result.levels:
            optional("support_maps", lambda: save_volume_maps(result.grid.as_volume(result.levels[-1].support.volume), result, output_dir,"visual_hull_support",dpi))
        optional("isosurfaces", lambda: save_multilevel_isosurfaces(result.volume,result.grid,output_dir,
            vis.get("isosurface_fractions",[.1,.25,.5]),float(vis.get("isosurface_fraction",.25)),dpi,
            absolute_levels=resolve_levels(result.volume,vis)))
        optional("convergence", lambda: save_convergence(result,output_dir,dpi))
        if vis.get("save_vtk",True):
            optional("vtk", lambda: save_vtk_legacy(output_dir/"concentration_3d.vtk",result.volume,result))
        if vis.get("save_csv_gz",True):
            optional("csv", lambda: save_csv_gz(output_dir/"concentration_points.csv.gz",result.volume,result))
    interactive = vis.get("save_interactive_3d",True) if save_interactive_html_override is None else save_interactive_html_override
    if profile != "data" and interactive:
        vis.update({"_viewer_metadata": metrics, "_observed": result.observed,
            "_predicted": result.predicted, "_weights": result.weights,
            "_measurement_shape": result.measurement_shape,
            "_measurement_shapes": result.measurement_shapes})
        optional("interactive_3d", lambda: save_interactive_3d_outputs(volume=result.volume,grid=result.grid,
            cameras=result.cameras,output_dir=output_dir,visual_cfg=vis))
    _write_json(output_dir/"export_report.json", {"profile":profile, "completed":completed, "errors":errors,
        "requested_isosurface_levels":resolve_levels(result.volume,vis),
        "note":"No internal isosurface is expected for a constant volume; numerical data remain available."})
    return metrics
