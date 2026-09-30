from __future__ import annotations

import numpy as np

from .grid import VolumeGrid


def forward_gradient(
    volume: np.ndarray, spacing: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward finite differences for volume stored as (z, y, x)."""
    dz, dy, dx = float(spacing[2]), float(spacing[1]), float(spacing[0])
    gx = np.zeros_like(volume, dtype=np.float32)
    gy = np.zeros_like(volume, dtype=np.float32)
    gz = np.zeros_like(volume, dtype=np.float32)
    gx[..., :-1] = (volume[..., 1:] - volume[..., :-1]) / dx
    gy[:, :-1, :] = (volume[:, 1:, :] - volume[:, :-1, :]) / dy
    gz[:-1, ...] = (volume[1:, ...] - volume[:-1, ...]) / dz
    return gx, gy, gz


def gradient_adjoint(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """Adjoint of forward_gradient under the standard Euclidean inner product."""
    dz, dy, dx = float(spacing[2]), float(spacing[1]), float(spacing[0])
    output = np.zeros_like(qx, dtype=np.float32)
    output[..., :-1] -= qx[..., :-1] / dx
    output[..., 1:] += qx[..., :-1] / dx
    output[:, :-1, :] -= qy[:, :-1, :] / dy
    output[:, 1:, :] += qy[:, :-1, :] / dy
    output[:-1, ...] -= qz[:-1, ...] / dz
    output[1:, ...] += qz[:-1, ...] / dz
    return output


def huber_tv_value_gradient(
    flat: np.ndarray,
    grid: VolumeGrid,
    delta: float,
) -> tuple[float, np.ndarray]:
    """Isotropic Huber-TV integral and gradient."""
    if delta <= 0:
        raise ValueError("Huber delta must be positive.")
    volume = grid.as_volume(flat).astype(np.float32, copy=False)
    gx, gy, gz = forward_gradient(volume, grid.spacing)
    magnitude = np.sqrt(gx * gx + gy * gy + gz * gz)
    quadratic = magnitude <= delta
    value_density = np.empty_like(magnitude, dtype=np.float32)
    value_density[quadratic] = 0.5 * magnitude[quadratic] ** 2 / delta
    value_density[~quadratic] = magnitude[~quadratic] - 0.5 * delta

    denominator = np.maximum(magnitude, np.float32(delta))
    qx = gx / denominator
    qy = gy / denominator
    qz = gz / denominator
    gradient = gradient_adjoint(qx, qy, qz, grid.spacing)
    voxel_volume = np.float32(grid.voxel_volume)
    return (
        float(voxel_volume * np.sum(value_density, dtype=np.float64)),
        (voxel_volume * gradient).ravel(),
    )


def huber_tv_lipschitz_bound(grid: VolumeGrid, delta: float) -> float:
    """Conservative Lipschitz bound for the Huber-TV gradient."""
    inverse_squared = np.sum(1.0 / np.square(grid.spacing))
    return float(grid.voxel_volume * 4.0 * inverse_squared / delta)
