from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .camera import CameraModel, load_cameras
from .grid import VolumeGrid
from .image_processing import MeasurementData, prepare_measurements
from .solvers import SolverResult, solve
from .support import SupportData, build_visual_hull_support
from .system_matrix import SystemMatrixBundle, build_or_load_system_matrix, _matrix_id, _matrix_payload
from .sampling import measurement_shapes, shape_for
from hashlib import sha256
import json


@dataclass
class LevelResult:
    grid: VolumeGrid
    solver: SolverResult
    matrix: SystemMatrixBundle
    support: SupportData


@dataclass
class ReconstructionResult:
    grid: VolumeGrid
    volume: np.ndarray  # physical/original measurement scale, shape (nz, ny, nx)
    volume_normalized: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray
    weights: np.ndarray
    measurement_scale: float
    cameras: list[CameraModel]
    measurement_shape: tuple[int, int]
    camera_slices: dict[str, slice]
    levels: list[LevelResult] = field(default_factory=list)
    signal_grids: dict[str, np.ndarray] = field(default_factory=dict)
    confidence_grids: dict[str, np.ndarray] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    timestamp_s: float | None = None
    frame_id: str | None = None
    measurement_shapes: dict[str, tuple[int, int]] = field(default_factory=dict)


def resize_volume(volume: np.ndarray, output_shape: tuple[int, int, int],
                  input_grid: VolumeGrid | None = None,
                  output_grid: VolumeGrid | None = None) -> np.ndarray:
    """Interpolate at physical cell centers; nearest-center extension at boundaries.

    Array order is (z,y,x). This is not a conservative remapping of cell averages.
    Without explicit grids both arrays are assumed to span the same physical box.
    """
    volume = np.asarray(volume, dtype=np.float32)
    output_shape = tuple(map(int, output_shape))
    if volume.ndim != 3 or len(output_shape) != 3 or min(output_shape) < 2:
        raise ValueError("resize_volume requires three-dimensional cell grids.")
    if not np.isfinite(volume).all():
        raise ValueError("Cannot interpolate a nonfinite volume.")
    if (input_grid is None) != (output_grid is None):
        raise ValueError("Supply both input_grid and output_grid, or neither.")
    if input_grid is None:
        axes = [(np.arange(new, dtype=float)+0.5)*old/new - 0.5
                for new, old in zip(output_shape, volume.shape)]
    else:
        if volume.shape != input_grid.array_shape or output_shape != output_grid.array_shape:
            raise ValueError("Volume shape does not match grid metadata.")
        axes_xyz = [(c-input_grid.bounds[k,0])/input_grid.spacing[k] - 0.5
                    for k,c in enumerate(output_grid.centers_1d())]
        axes = axes_xyz[::-1]
    coordinates = np.meshgrid(*axes, indexing="ij")
    return ndimage.map_coordinates(volume, coordinates, order=1, mode="nearest", prefilter=False).astype(np.float32)


def _solver_config(cfg: dict[str, Any]) -> dict[str, Any]:
    solver_cfg = dict(cfg["solver"])
    solver_cfg["lambda_support"] = float(
        cfg["support"].get("lambda_support", solver_cfg.get("lambda_support", 0.0))
    )
    solver_cfg["random_seed"] = int(cfg["project"].get("random_seed", 0))
    return solver_cfg


def reconstruct_measurements(
    cfg: dict[str, Any],
    cameras: list[CameraModel],
    measurements: MeasurementData,
    previous_volume: np.ndarray | None = None,
    force_matrix_rebuild: bool = False,
    verbose: bool = True,
    matrix_cache: dict | None = None,
    projection_cache: dict | None = None,
) -> ReconstructionResult:
    if len(cameras) < 2:
        raise ValueError("At least two cameras are required.")
    if np.asarray(measurements.vector).ndim != 1 or np.asarray(measurements.weights).shape != np.asarray(measurements.vector).shape:
        raise ValueError("Measurement vector and weights must be matching one-dimensional arrays.")
    if not np.isfinite(measurements.vector).all():
        raise ValueError("Measurement vector must be finite; mark unavailable observations with value 0 and weight 0.")
    if not np.isfinite(measurements.weights).all() or np.any(measurements.weights < 0):
        raise ValueError("Measurement weights must be finite and nonnegative.")
    measurement_shape = measurement_shapes(cfg, cameras)
    samples_per_bin = int(cfg["measurement"]["ray_samples_per_bin"])
    levels = [tuple(int(v) for v in level) for level in cfg["volume"]["levels"]]
    iterations_per_level = [int(v) for v in cfg["solver"]["iterations_per_level"]]
    cache_dir = Path(cfg["measurement"]["matrix_cache_dir"])
    solver_cfg = _solver_config(cfg)
    if not np.isfinite(measurements.scale) or measurements.scale <= 0:
        raise ValueError("Measurement scale must be finite and positive.")
    # u=x/s, b'=b/s, objective'=objective/s^2. Huber is not quadratic:
    # lambda_tv'=lambda_tv/s and delta'=delta/s; quadratic lambdas unchanged.
    solver_cfg["lambda_tv"] /= float(measurements.scale)
    solver_cfg["huber_delta"] /= float(measurements.scale)
    support_cfg = dict(cfg["support"])
    support_cfg["minimum_threshold"] = float(support_cfg.get("minimum_threshold", 0))/float(measurements.scale)
    matrix_cache = {} if matrix_cache is None else matrix_cache
    projection_cache = {} if projection_cache is None else projection_cache


    expected_measurements = sum(r*c for r,c in measurement_shape.values())
    if measurements.vector.size != expected_measurements:
        raise ValueError(
            f"Expected {expected_measurements} measurements, got {measurements.vector.size}."
        )

    for camera in cameras:
        expected = measurement_shape[camera.name]
        if camera.name not in measurements.grids or np.shape(measurements.grids[camera.name]) != expected:
            raise ValueError(f"Missing/mismatched signal grid for {camera.name}: expected {expected}.")
        if camera.name not in measurements.confidence_grids or np.shape(measurements.confidence_grids[camera.name]) != expected:
            raise ValueError(f"Missing/mismatched confidence grid for {camera.name}.")
    if measurements.geometry_signature is not None and measurements.geometry_signature != geometry_signature(cfg, cameras):
        raise ValueError("Projection geometry signature does not match this scene/camera configuration.")
    previous_normalized_final: np.ndarray | None = None
    if previous_volume is not None:
        previous_array = np.asarray(previous_volume, dtype=np.float32)
        final_grid = VolumeGrid.from_config(cfg["volume"], levels[-1])
        if previous_array.shape != final_grid.array_shape:
            raise ValueError(
                f"previous_volume must have shape {final_grid.array_shape}; "
                f"got {previous_array.shape}."
            )
        previous_normalized_final = previous_array / np.float32(measurements.scale)

    level_results: list[LevelResult] = []
    current_volume: np.ndarray | None = None
    previous_grid: VolumeGrid | None = None

    for level_index, (shape, iteration_count) in enumerate(
        zip(levels, iterations_per_level), start=1
    ):
        grid = VolumeGrid.from_config(cfg["volume"], shape)
        if verbose:
            print(
                f"[pipeline] level {level_index}/{len(levels)}: "
                f"grid={grid.nx}x{grid.ny}x{grid.nz}"
            )
        cache_key = _matrix_id(_matrix_payload(cameras, grid, measurement_shape, samples_per_bin))
        if cache_key not in matrix_cache or force_matrix_rebuild:
            matrix_cache[cache_key] = build_or_load_system_matrix(
                cameras=cameras, grid=grid, measurement_shape=measurement_shape,
                samples_per_bin=samples_per_bin, cache_dir=cache_dir,
                force_rebuild=force_matrix_rebuild, verbose=verbose)
        matrix_bundle = matrix_cache[cache_key]
        matrix = matrix_bundle.matrix
        valid_rows = np.asarray(matrix.sum(axis=1)).ravel() > 1.0e-9
        weights = measurements.weights * valid_rows.astype(np.float32)
        quality_cfg = cfg.get("quality", {})
        minimum_fraction = float(quality_cfg.get("minimum_valid_fraction_per_camera", 0.1))
        camera_quality = {}
        for camera in cameras:
            cs = matrix_bundle.camera_slices[camera.name]
            geometric_count = int(np.count_nonzero(valid_rows[cs]))
            valid_count = int(np.count_nonzero(weights[cs] > 0))
            fraction = valid_count / geometric_count if geometric_count else 0.0
            camera_quality[camera.name] = {"valid_fraction_in_volume": fraction,
                "valid_rows": valid_count, "geometric_rows": geometric_count,
                "usable": bool(valid_count > 0 and fraction >= minimum_fraction)}
        usable_count = sum(q["usable"] for q in camera_quality.values())
        both_usable = all(q["usable"] for q in camera_quality.values())
        required_views = int(quality_cfg.get("minimum_usable_views", 2 if quality_cfg.get("require_two_views", True) else 1))
        usable_confidence = {n: values.copy() for n, values in measurements.confidence_grids.items()}
        for camera in cameras:
            if not camera_quality[camera.name]["usable"]:
                weights[matrix_bundle.camera_slices[camera.name]] = 0
                usable_confidence[camera.name][:] = 0
        if usable_count < required_views:
            weights[:] = 0
        support_data = build_visual_hull_support(
            cameras=cameras,
            signal_grids=measurements.grids,
            grid=grid,
            measurement_shape=measurement_shape,
            cfg=support_cfg,
            confidence_grids=usable_confidence,
            projection_cache=projection_cache,
            external_scores=measurements.support_grids,
        )

        x0: np.ndarray | None = None
        if current_volume is not None and previous_grid is not None:
            x0 = resize_volume(current_volume, grid.array_shape).ravel()

        previous_level: np.ndarray | None = None
        if previous_normalized_final is not None:
            previous_level = resize_volume(
                previous_normalized_final, grid.array_shape
            ).ravel()
        if x0 is None and previous_level is not None:
            x0 = previous_level.copy()

        solver_result = solve(
            matrix=matrix,
            b=measurements.vector,
            weights=weights,
            grid=grid,
            config=solver_cfg,
            iterations=iteration_count,
            x0=x0,
            previous=previous_level,
            support=support_data.volume,
            verbose=verbose,
        )
        current_volume = grid.as_volume(solver_result.x)
        previous_grid = grid
        level_results.append(
            LevelResult(
                grid=grid,
                solver=solver_result,
                matrix=matrix_bundle,
                support=support_data,
            )
        )

    assert current_volume is not None and previous_grid is not None
    final_bundle = level_results[-1].matrix
    final_weights = weights.copy()
    final_solver = level_results[-1].solver
    quality = {"data_status": "valid_two_view" if both_usable else "insufficient_dual_view_data",
               "solver_status": final_solver.status,
               "has_estimate": final_solver.has_estimate,
               "measurement_supported": bool(np.any(final_weights > 0)),
               "per_camera": camera_quality,
               "preprocess": measurements.diagnostics}
    if both_usable and len(cameras) > 2:
        quality["data_status"] = "valid_multi_view"
    if not both_usable and np.any(final_weights > 0):
        quality["data_status"] = "degraded_single_view" if usable_count == 1 else "degraded_multi_view"
    quality["usable_views"] = usable_count
    quality["required_views"] = required_views
    quality["scene_id"] = cfg.get("scene", {}).get("id", "default")
    predicted_normalized = np.asarray(
        final_bundle.matrix @ current_volume.ravel(), dtype=np.float32
    )
    volume_physical = current_volume * np.float32(measurements.scale)
    observed_physical = measurements.vector * np.float32(measurements.scale)
    predicted_physical = predicted_normalized * np.float32(measurements.scale)

    return ReconstructionResult(
        grid=previous_grid,
        volume=volume_physical,
        volume_normalized=current_volume,
        observed=observed_physical,
        predicted=predicted_physical,
        weights=final_weights,
        measurement_scale=float(measurements.scale),
        cameras=cameras,
        measurement_shape=measurement_shape[cameras[0].name],
        measurement_shapes=measurement_shape,
        camera_slices=final_bundle.camera_slices,
        levels=level_results,
        signal_grids={
            name: grid_values * np.float32(measurements.scale)
            for name, grid_values in measurements.grids.items()
        },
        confidence_grids=measurements.confidence_grids,
        quality=quality,
    )


def reconstruct_images(
    cfg: dict[str, Any],
    image_paths: dict[str, str | Path],
    background_paths: dict[str, str | Path | None] | None = None,
    previous_volume: np.ndarray | None = None,
    force_matrix_rebuild: bool = False,
    verbose: bool = True,
    support_paths: dict | None = None,
    validity_paths: dict | None = None,
) -> ReconstructionResult:
    cameras = load_cameras(cfg)
    measurement_shape = measurement_shapes(cfg, cameras)
    from .inputs import prepare_input_measurements
    measurements = prepare_input_measurements(cfg, cameras, image_paths,
        background_paths=background_paths, support_paths=support_paths, validity_paths=validity_paths,
        kind="raw_images")
    return reconstruct_measurements(
        cfg=cfg,
        cameras=cameras,
        measurements=measurements,
        previous_volume=previous_volume,
        force_matrix_rebuild=force_matrix_rebuild,
        verbose=verbose,
    )


def geometry_signature(cfg: dict, cameras: list[CameraModel]) -> str:
    payload = {"cameras": [c.metadata() for c in cameras],
               "shapes": measurement_shapes(cfg, cameras),
               "samples": cfg["measurement"]["ray_samples_per_bin"],
               "volume": {"bounds_m": cfg["volume"]["bounds_m"],
                          "final_shape": cfg["volume"]["levels"][-1]}}
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def state_signature(cfg: dict, cameras: list[CameraModel]) -> str:
    data = {"geometry": geometry_signature(cfg, cameras), "scene": cfg.get("scene", {}),
            "preprocess": cfg.get("preprocess", {}), "input": cfg.get("input", {}),
            "units": cfg.get("units", {})}
    return sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class ReconstructionSession:
    """Fixed-geometry sequence session, keeping camera models and matrices in RAM.

    A missing/invalid pair does not silently become a measured zero field. A gap
    resets temporal state when configured; fine-grid-only continuation is optional.
    """
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = deepcopy(cfg)
        self.cameras = load_cameras(cfg)
        self.matrix_cache: dict = {}
        self.projection_cache: dict = {}
        self.previous: np.ndarray | None = None
        self.previous_timestamp: float | None = None
        self._signature = state_signature(self.cfg, self.cameras)
        self.last_reset_reason: str | None = "new_session"

    def configure(self, cfg: dict[str, Any], *, reset: bool = False) -> bool:
        """Switch scene/calibration safely. Changed physical inputs always reset state.

        No implicit warp of a historical volume after geometry/calibration changes.
        Calibration files are read again so replacing a file at the same path is detected.
        """
        cameras = load_cameras(cfg)
        signature = state_signature(cfg, cameras)
        changed = signature != self._signature
        if reset or changed:
            self.reset_temporal()
            self.matrix_cache.clear()
            self.projection_cache.clear()
            self.last_reset_reason = "configuration_changed" if changed else "explicit_reset"
        self.cfg, self.cameras, self._signature = deepcopy(cfg), cameras, signature
        return changed

    def reset_temporal(self):
        self.previous = None
        self.previous_timestamp = None

    def reconstruct(self, measurements: MeasurementData, *, timestamp_s: float | None = None,
                    frame_id: str | None = None, force_matrix_rebuild: bool = False,
                    verbose: bool = True) -> ReconstructionResult:
        # Fail closed on callers mutating cfg or replacing calibration files in-place.
        self.configure(self.cfg)
        cfg = deepcopy(self.cfg)
        seq = cfg.get("sequence", {})
        dt = None
        if timestamp_s is not None:
            if not np.isfinite(timestamp_s):
                raise ValueError("timestamp must be finite.")
            if self.previous_timestamp is not None:
                dt = timestamp_s-self.previous_timestamp
                if dt <= 0:
                    raise ValueError("Sequence timestamps must increase strictly.")
                max_gap = seq.get("max_gap_s")
                if max_gap is not None and dt > float(max_gap):
                    self.reset_temporal()
                elif seq.get("reference_dt_s") is not None:
                    cfg["solver"]["lambda_temporal"] *= float(seq["reference_dt_s"])/dt
        if self.previous is not None and seq.get("fine_level_only_after_first", True):
            cfg["volume"]["levels"] = cfg["volume"]["levels"][-1:]
            cfg["solver"]["iterations_per_level"] = cfg["solver"]["iterations_per_level"][-1:]
        result = reconstruct_measurements(cfg, self.cameras, measurements,
            previous_volume=self.previous, force_matrix_rebuild=force_matrix_rebuild,
            verbose=verbose, matrix_cache=self.matrix_cache, projection_cache=self.projection_cache)
        result.timestamp_s, result.frame_id = timestamp_s, frame_id
        result.quality["dt_s"] = dt
        result.quality["state_reset_reason"] = self.last_reset_reason
        self.last_reset_reason = None
        if result.quality["has_estimate"]:
            self.previous = result.volume.copy()
        if timestamp_s is not None:
            self.previous_timestamp = timestamp_s
        return result

    def reconstruct_inputs(self, paths, background_paths=None, support_paths=None,
                           validity_paths=None, kind=None, **kwargs):
        from .inputs import prepare_input_measurements
        self.configure(self.cfg)
        measurements = prepare_input_measurements(self.cfg, self.cameras, paths,
            background_paths=background_paths, support_paths=support_paths,
            validity_paths=validity_paths, kind=kind)
        return self.reconstruct(measurements, **kwargs)

    def reconstruct_images(self, image_paths, background_paths=None, **kwargs):
        return self.reconstruct_inputs(image_paths, background_paths, kind="raw_images", **kwargs)
