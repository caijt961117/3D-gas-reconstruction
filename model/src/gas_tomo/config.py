from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import re
import yaml
import numpy as np
from .sampling import shape_for
from .grid import VolumeGrid


DEFAULTS: dict[str, Any] = {
    "project": {
        "name": "dualview_gas_tomography",
        "output_dir": "../outputs",
        "random_seed": 42,
    },
    "scene": {"id": "default", "world_unit": "m", "description": ""},
    "input": {"kind": "raw_images", "geometry": "measurement_grid", "column_density_unit": None},
    "volume": {
        "bounds_m": {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]},
        "levels": [[24, 24, 24], [40, 40, 40]],
    },
    "measurement": {
        "grid_shape": [40, 40],
        "ray_samples_per_bin": 2,
        "matrix_cache_dir": "../cache",
    },
    "preprocess": {
        "mode": "inverse_intensity",
        "gas_is_darker": True,
        "intensity_scale": None,
        "denoise_sigma_px": 0.8,
        "background_epsilon": 1.0e-5,
        "baseline_quantile": None,
        "calibration_coefficients": [0.0, 1.0],
        "clip_min": None,
        "clip_max": None,
        "saturation_low": 0.002,
        "saturation_high": 0.998,
        "measurement_normalization": "p99",
        "partial_bin_policy": "reject",
        "minimum_valid_fraction": 0.8,
        "fixed_measurement_scale": None,
    },
    "quality": {"minimum_valid_fraction_per_camera": 0.10, "require_two_views": True},
    "units": {"calibrated": False, "concentration": "relative response / m", "column_density": "relative response"},
    "sequence": {"fine_level_only_after_first": True, "reference_dt_s": None, "max_gap_s": None},
    "support": {
        "enabled": True,
        "source": "auto",
        "external_geometry": "measurement_grid",
        "threshold_fraction": 0.10,
        "minimum_threshold": 0.0,
        "closing_cells": 1,
        "dilation_cells": 2,
        "gaussian_sigma_cells": 1.0,
        "combine": "geometric_mean",
        "lambda_support": 0.02,
    },
    "solver": {
        "name": "fista_huber_tv",
        "iterations_per_level": [160, 260],
        "tolerance": 2.0e-5,
        "lambda_tv": 0.000004,
        "huber_delta": 0.06,
        "lambda_temporal": 0.015,
        "lambda_l2": 1.0e-6,
        "step_safety": 0.92,
        "power_iterations": 25,
        "log_every": 10,
        "restart": True,
        "sart_relaxation": 0.8,
        "backtracking": True,
        "max_backtracking": 25,
        "kkt_tolerance": 2.0e-4,
    },
    "visualization": {
        "isosurface_fraction": 0.25,
        "isosurface_fractions": [0.10, 0.25, 0.50],
        "save_interactive_3d": True,
        "interactive_isosurface_fractions": [0.10, 0.25, 0.50],
        "interactive_isosurface_opacity": 0.38,
        "interactive_colorscale": "Turbo",
        "interactive_max_voxels": 180000,
        "interactive_point_limit": 20000,
        "interactive_include_plotlyjs": "inline",
        "volume_min_fraction": 0.03,
        "volume_opacity": 0.08,
        "volume_surface_count": 22,
        "save_camera_geometry_3d": True,
        "geometry_ray_grid": [9, 9],
        "export_profile": "full",
        "color_range": None,
        "isosurface_levels": None,
        "concentration_unit": "relative response / m",
        "dashboard": True,
        "save_csv_gz": True,
        "save_vtk": True,
        "dpi": 160,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(value: str | Path, config_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return str(path)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load, merge defaults, resolve paths and validate a YAML configuration."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        user_cfg = yaml.safe_load(file) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError("Top-level YAML content must be a mapping.")

    cfg = _deep_merge(DEFAULTS, user_cfg)
    # Only EXPLICIT user keys choose grid resolution. Never let default levels
    # override a requested shape. Conflicting specifications are rejected.
    vol = user_cfg.get("volume", {})
    selectors = [k for k in ("levels", "shape", "voxel_size_m") if k in vol]
    if len(selectors) > 1:
        raise ValueError("Set exactly one of volume.levels, volume.shape, volume.voxel_size_m.")
    if selectors == ["levels"] and not vol["levels"]:
        raise ValueError("volume.levels must not be empty.")
    if selectors == ["shape"]:
        cfg["volume"]["levels"] = [vol["shape"]]
        cfg["volume"].pop("shape", None)
    elif selectors == ["voxel_size_m"]:
        h = np.asarray(vol["voxel_size_m"], dtype=float)
        if h.ndim == 0:
            h = np.repeat(h, 3)
        if h.shape != (3,) or not np.isfinite(h).all() or np.any(h <= 0):
            raise ValueError("volume.voxel_size_m must be a positive scalar or [hx,hy,hz].")
        b = cfg["volume"]["bounds_m"]
        widths = np.array([b[a][1]-b[a][0] for a in "xyz"])
        shape = np.maximum(2, np.ceil(widths/h - 1e-10)).astype(int)
        cfg["volume"]["levels"] = [shape.tolist()]
        cfg["volume"].pop("voxel_size_m", None)
        cfg["volume"]["requested_voxel_size_m"] = h.tolist()
    if "iterations_per_level" not in user_cfg.get("solver", {}):
        cfg["solver"]["iterations_per_level"] = [260]*len(cfg["volume"]["levels"])

    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)

    config_dir = config_path.parent
    cfg["project"]["output_dir"] = _resolve_path(
        cfg["project"]["output_dir"], config_dir
    )
    cfg["measurement"]["matrix_cache_dir"] = _resolve_path(
        cfg["measurement"]["matrix_cache_dir"], config_dir
    )

    cameras = cfg.get("cameras", [])
    if len(cameras) < 2:
        raise ValueError(f"At least two cameras are required; got {len(cameras)}.")
    for camera in cameras:
        if "name" not in camera:
            raise ValueError("Every camera needs a unique 'name'.")
        camera.setdefault("roi", None)
        camera.setdefault("rotate_k", 0)
        camera.setdefault("flip_horizontal", False)
        camera.setdefault("flip_vertical", False)
        if camera.get("type", "calibrated") == "calibrated":
            if "calibration_file" not in camera:
                raise ValueError(
                    f"Calibrated camera '{camera['name']}' needs calibration_file."
                )
            camera["calibration_file"] = _resolve_path(
                camera["calibration_file"], config_dir
            )

    levels = cfg["volume"].get("levels")
    if not levels:
        shape = cfg["volume"].get("shape", [40, 40, 40])
        levels = [shape]
        cfg["volume"]["levels"] = levels
    for shape in levels:
        if len(shape) != 3 or any(not np.isfinite(v) or int(v) != v or int(v) <= 1 for v in shape):
            raise ValueError("Each volume level must be [nx, ny, nz], all > 1.")

    measurement_shape = cfg["measurement"]["grid_shape"]
    if len(measurement_shape) != 2 or any(int(v) <= 1 for v in measurement_shape):
        raise ValueError("measurement.grid_shape must be [rows, cols], both > 1.")

    iterations = cfg["solver"].get("iterations_per_level", [])
    if len(iterations) == 1 and len(levels) > 1:
        cfg["solver"]["iterations_per_level"] = iterations * len(levels)
    elif len(iterations) != len(levels):
        raise ValueError(
            "solver.iterations_per_level must contain one value per volume level."
        )

    camera_names = [camera["name"] for camera in cameras]
    if len(set(camera_names)) != len(cameras):
        raise ValueError("Camera names must be unique.")

    for key in ("lambda_tv", "lambda_temporal", "lambda_l2", "huber_delta"):
        if not float(cfg["solver"][key]) >= 0:
            raise ValueError(f"solver.{key} must be nonnegative.")
    if cfg["solver"]["huber_delta"] <= 0:
        raise ValueError("solver.huber_delta must be positive.")
    if any(v < 1 for v in cfg["solver"]["iterations_per_level"]):
        raise ValueError("Every iteration budget must be positive.")
    if not 0 < float(cfg["solver"]["step_safety"]) <= 1:
        raise ValueError("step_safety must be in (0,1].")
    budget = int(cfg["visualization"]["interactive_max_voxels"])
    if 0 < budget < 8:
        raise ValueError("interactive_max_voxels must be 0 (unlimited) or >=8.")
    cfg["visualization"]["concentration_unit"] = cfg["units"]["concentration"]
    for camera in cameras:
        pc = cfg["preprocess"].get("per_camera", {}).get(camera["name"], {})
        if pc.get("bad_pixel_mask"):
            pc["bad_pixel_mask"] = _resolve_path(pc["bad_pixel_mask"], config_dir)
    if cfg["preprocess"].get("bad_pixel_mask"):
        cfg["preprocess"]["bad_pixel_mask"] = _resolve_path(cfg["preprocess"]["bad_pixel_mask"], config_dir)

    grid = VolumeGrid.from_config(cfg["volume"], levels[-1])
    cfg["volume"]["resolved_voxel_size_m"] = grid.spacing.tolist()
    for camera in cameras:
        shape_for(camera.get("measurement_grid_shape", measurement_shape), camera["name"])
    if cfg["scene"].get("world_unit", "m") != "m":
        raise ValueError("World coordinates and path lengths must be in meters. Convert inputs explicitly.")
    if cfg["input"].get("kind") not in {"raw_images", "column_density"}:
        raise ValueError("input.kind must be raw_images or column_density.")
    if cfg["input"].get("geometry") not in {"raw_pixels", "processed_pixels", "measurement_grid"}:
        raise ValueError("input.geometry must be raw_pixels, processed_pixels or measurement_grid.")
    if cfg["support"].get("source", "auto") not in {"auto", "external", "none"}:
        raise ValueError("support.source must be auto, external or none.")
    if cfg["support"].get("source") == "none":
        cfg["support"]["enabled"] = False
    if cfg["input"]["kind"] == "column_density":
        unit = cfg["input"].get("column_density_unit")
        if not unit or unit != cfg["units"].get("column_density"):
            raise ValueError("Direct projections require input.column_density_unit matching units.column_density; no implicit unit conversion.")
    min_views = int(cfg["quality"].get("minimum_usable_views", 2 if cfg["quality"].get("require_two_views", True) else 1))
    if not 1 <= min_views <= len(cameras):
        raise ValueError("quality.minimum_usable_views must be between 1 and the camera count.")
    if not isinstance(cfg["scene"].get("id"), str) or not re.fullmatch(r"[\w-]+",cfg["scene"]["id"]):
        raise ValueError("scene.id must contain only letters, digits, underscores or hyphens.")
    if not float(cfg["support"]["lambda_support"]) >= 0:
        raise ValueError("support.lambda_support must be nonnegative.")
    Path(cfg["project"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["measurement"]["matrix_cache_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def dump_resolved_config(cfg: dict[str, Any], path: str | Path) -> None:
    serializable = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(serializable, file, sort_keys=False, allow_unicode=True)
