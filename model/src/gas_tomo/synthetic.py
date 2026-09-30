from __future__ import annotations

from typing import Any

import numpy as np

from .camera import CameraModel
from .grid import VolumeGrid
from .image_processing import MeasurementData
from .system_matrix import build_or_load_system_matrix
from .sampling import measurement_shapes, shape_for


def _gaussian3d(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    center: tuple[float, float, float],
    sigma: tuple[float, float, float],
    amplitude: float,
) -> np.ndarray:
    return amplitude * np.exp(
        -0.5
        * (
            ((xx - center[0]) / sigma[0]) ** 2
            + ((yy - center[1]) / sigma[1]) ** 2
            + ((zz - center[2]) / sigma[2]) ** 2
        )
    )


def make_synthetic_truth(grid: VolumeGrid, scenario: str = "plume", config: dict | None = None) -> np.ndarray:
    config = dict(config or {})
    x, y, z = grid.centers_1d()
    # Built-in demo shapes live in local [0,1]^3 coordinates, not a fixed world box.
    lo, widths = grid.bounds[:,0], grid.bounds[:,1]-grid.bounds[:,0]
    x, y, z = ((axis-lo[i])/widths[i] for i,axis in enumerate((x,y,z)))
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    if "gaussians" in config:
        return make_parametric_truth(grid, config)
    scenario = scenario.lower()

    if scenario == "single_gaussian":
        volume = _gaussian3d(
            xx,
            yy,
            zz,
            center=(0.42, 0.58, 0.42),
            sigma=(0.11, 0.09, 0.16),
            amplitude=1.0,
        )
    elif scenario == "two_clouds":
        volume = _gaussian3d(
            xx,
            yy,
            zz,
            center=(0.34, 0.42, 0.35),
            sigma=(0.09, 0.11, 0.14),
            amplitude=1.0,
        )
        volume += _gaussian3d(
            xx,
            yy,
            zz,
            center=(0.64, 0.68, 0.64),
            sigma=(0.12, 0.08, 0.11),
            amplitude=0.55,
        )
    elif scenario == "plume":
        source_z = 0.08
        height = np.maximum(zz - source_z, 0.0)
        center_x = 0.38 + 0.08 * np.sin(2.8 * height)
        center_y = 0.33 + 0.24 * height
        sigma_x = 0.045 + 0.11 * height
        sigma_y = 0.050 + 0.095 * height
        cross_section = np.exp(
            -0.5
            * (
                ((xx - center_x) / sigma_x) ** 2
                + ((yy - center_y) / sigma_y) ** 2
            )
        )
        vertical = np.exp(-height / 0.75) * (zz >= source_z)
        source = _gaussian3d(
            xx,
            yy,
            zz,
            center=(0.38, 0.33, 0.10),
            sigma=(0.045, 0.045, 0.055),
            amplitude=1.0,
        )
        volume = 0.85 * cross_section * vertical + 0.7 * source
    else:
        raise ValueError("scenario must be single_gaussian, two_clouds or plume.")

    volume = np.maximum(volume, 0.0)
    maximum = float(np.max(volume, initial=0.0))
    if maximum > 0:
        volume /= maximum
    amplitude = float(config.get("amplitude", 1.0))
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Synthetic amplitude must be finite and nonnegative.")
    return (volume*amplitude).astype(np.float32)


def make_parametric_truth(grid: VolumeGrid, config: dict) -> np.ndarray:
    points = grid.centers()
    widths = grid.bounds[:,1]-grid.bounds[:,0]
    local = config.get("coordinates", "relative")
    if local not in {"relative", "world_m"}:
        raise ValueError("synthetic.coordinates must be relative or world_m.")
    values = np.zeros(grid.n_voxels, dtype=np.float64)
    for item in config["gaussians"]:
        center = np.asarray(item["center"], dtype=float)
        sigma = np.asarray(item["sigma"], dtype=float)
        amp = float(item.get("amplitude", 1.0))
        if center.shape != (3,) or sigma.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0) or not np.isfinite(amp) or amp < 0:
            raise ValueError("Each Gaussian needs finite center[3], positive sigma[3], nonnegative amplitude.")
        if local == "relative":
            center, sigma = grid.bounds[:,0]+widths*center, widths*sigma
        values += amp*np.exp(-0.5*np.sum(((points-center)/sigma)**2, axis=1))
    return grid.as_volume(values.astype(np.float32))


def _scale_measurement(vector: np.ndarray, mode: str) -> float:
    positive = vector[vector > 0]
    if mode == "none" or positive.size == 0:
        return 1.0
    if mode == "max":
        return max(float(np.max(positive)), 1.0e-12)
    if mode == "p99":
        return max(float(np.quantile(positive, 0.99)), 1.0e-12)
    if mode == "p995":
        return max(float(np.quantile(positive, 0.995)), 1.0e-12)
    if mode == "l2":
        return max(float(np.linalg.norm(positive)), 1.0e-12)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def make_synthetic_measurements(
    cfg: dict[str, Any],
    cameras: list[CameraModel],
    truth: np.ndarray,
    noise_fraction: float = 0.01,
    seed: int = 42,
    force_matrix_rebuild: bool = False,
    verbose: bool = True,
) -> MeasurementData:
    if not np.isfinite(noise_fraction) or noise_fraction < 0:
        raise ValueError("noise_fraction must be finite and nonnegative.")
    final_shape = tuple(int(v) for v in cfg["volume"]["levels"][-1])
    grid = VolumeGrid.from_config(cfg["volume"], final_shape)
    truth = np.asarray(truth, dtype=np.float32)
    if truth.shape != grid.array_shape:
        raise ValueError(f"Truth must have shape {grid.array_shape}; got {truth.shape}.")
    measurement_shape = measurement_shapes(cfg, cameras)
    bundle = build_or_load_system_matrix(
        cameras=cameras,
        grid=grid,
        measurement_shape=measurement_shape,
        samples_per_bin=int(cfg["measurement"]["ray_samples_per_bin"]),
        cache_dir=cfg["measurement"]["matrix_cache_dir"],
        force_rebuild=force_matrix_rebuild,
        verbose=verbose,
    )
    clean = np.asarray(bundle.matrix @ truth.ravel(), dtype=np.float32)
    rng = np.random.default_rng(seed)
    sigma = float(noise_fraction) * max(float(np.max(clean, initial=0.0)), 1.0e-12)
    noisy = (clean + rng.normal(0.0, sigma, size=clean.size)).astype(
        np.float32
    )
    normalization = str(
        cfg["preprocess"].get("measurement_normalization", "none")
    ).lower()
    scale = _scale_measurement(noisy, normalization)
    if cfg["preprocess"].get("fixed_measurement_scale") is not None:
        scale = float(cfg["preprocess"]["fixed_measurement_scale"])
    normalized = noisy / np.float32(scale)
    grids: dict[str, np.ndarray] = {}
    confidence_grids: dict[str, np.ndarray] = {}
    for index, camera in enumerate(cameras):
        rows, cols = shape_for(measurement_shape, camera.name)
        camera_slice = bundle.camera_slices[camera.name]
        grids[camera.name] = normalized[camera_slice].reshape(rows, cols)
        confidence_grids[camera.name] = np.ones((rows, cols), dtype=np.float32)
    return MeasurementData(
        vector=normalized,
        weights=np.ones_like(normalized, dtype=np.float32),
        grids=grids,
        confidence_grids=confidence_grids,
        scale=scale,
        normalization=normalization,
    )
