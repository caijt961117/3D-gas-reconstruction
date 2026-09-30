"""One source of truth for HTML, PNG and PLY isosurface thresholds."""
from __future__ import annotations
from typing import Any
import numpy as np
from skimage.measure import marching_cubes
from .grid import VolumeGrid


def resolve_levels(volume: np.ndarray, cfg: dict[str, Any]) -> list[float]:
    a = np.asarray(volume)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return []
    levels = cfg.get("isosurface_levels")
    if levels is None:
        peak = float(np.max(finite))
        fractions = cfg.get("isosurface_fractions", cfg.get("interactive_isosurface_fractions", [0.10,0.25,0.50]))
        levels = [float(v)*peak for v in fractions]
    levels = [float(v) for v in levels]
    if any(not np.isfinite(v) or v < 0 for v in levels):
        raise ValueError("Isosurface levels must be finite and nonnegative.")
    return sorted(set(levels))


def surface_mesh(volume: np.ndarray, grid: VolumeGrid, level: float):
    a = np.asarray(volume, dtype=np.float32)
    if a.shape != grid.array_shape:
        raise ValueError("Surface volume does not match grid.")
    if not np.isfinite(a).all():
        raise ValueError("Surface input contains nonfinite values.")
    if not np.isfinite(level) or not float(a.min()) < level < float(a.max()):
        return np.empty((0,3)), np.empty((0,3), dtype=np.int32)
    dx,dy,dz = grid.spacing
    try:
        vertices, faces, _, _ = marching_cubes(a, level=float(level), spacing=(dz,dy,dx), allow_degenerate=False)
    except (ValueError, RuntimeError) as error:
        # Only absence of an internal surface is a normal, skippable condition.
        if "No surface found" in str(error):
            return np.empty((0,3)), np.empty((0,3), dtype=np.int32)
        raise
    xyz = vertices[:, [2,1,0]] + grid.bounds[:,0] + grid.spacing*0.5
    return xyz, faces
