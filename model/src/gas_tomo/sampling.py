"""Per-camera observation layouts. Rows/columns are never world X/Y axes."""
from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np

ShapeSpec = tuple[int, int] | dict[str, tuple[int, int]]

def shape_for(spec, camera_name: str) -> tuple[int, int]:
    value = spec[camera_name] if isinstance(spec, Mapping) else spec
    if len(value) != 2 or any(not np.isfinite(v) or int(v) != v or int(v) < 2 for v in value):
        raise ValueError(f"Observation grid for {camera_name} must be [rows, cols], integer values >=2.")
    return tuple(map(int, value))

def measurement_shapes(cfg: dict, cameras=None) -> dict[str, tuple[int,int]]:
    configs = {c["name"]: c for c in cfg["cameras"]}
    names = [c.name for c in cameras] if cameras is not None else list(configs)
    default = cfg["measurement"]["grid_shape"]
    return {n: shape_for(configs[n].get("measurement_grid_shape", default), n) for n in names}

def layout_slices(names: Sequence[str], shapes) -> dict[str, slice]:
    result, start = {}, 0
    for name in names:
        rows, cols = shape_for(shapes, name)
        result[name] = slice(start, start + rows*cols)
        start += rows*cols
    return result
