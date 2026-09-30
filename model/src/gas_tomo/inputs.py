"""Scene-independent observation input. Intensities, supports and validity are distinct.

Direct projections retain their signed noise and declared units. No image inversion,
polynomial calibration or baseline subtraction is applied to column-density input.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from .camera import CameraModel
from .image_processing import (MeasurementData, prepare_measurements, read_image,
                               to_grayscale, _measurement_scale)
from .sampling import measurement_shapes


def read_array_2d(path: str | Path, *, key: str = "column_density") -> tuple[np.ndarray, str | None]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    unit = None
    if path.suffix.lower() == ".npy":
        a = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise ValueError(f"{path.name} needs NPZ array '{key}'.")
            a = data[key].copy()
            unit = str(data["unit"].item()) if "unit" in data else None
    elif path.suffix.lower() == ".csv":
        a = np.loadtxt(path, delimiter=",")
    else:
        a = read_image(path)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.number) and a.dtype != bool:
        raise ValueError(f"Expected a 2-D numeric array, not RGB/stacked frames: {path} {a.shape}")
    return a, unit


def score_array(path: str | Path, key: str = "support") -> np.ndarray:
    """Binary 0/1, uint8 0/255 or floating [0,1] score maps; NaN=unknown."""
    original, _ = read_array_2d(path, key=key)
    a = np.asarray(original, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if original.dtype == np.uint8 and finite.size and finite.max() > 1:
        a /= 255.0
    elif original.dtype == np.uint16 and finite.size and finite.max() > 1:
        a /= 65535.0
    finite = a[np.isfinite(a)]
    if np.any((finite < 0) | (finite > 1)):
        raise ValueError("Support/validity arrays must be float [0,1], binary 0/1, or full-range uint8/uint16.")
    return a


def _map_to_processed(a: np.ndarray, camera: CameraModel, geometry: str) -> np.ndarray:
    if geometry == "raw_pixels":
        return camera.preprocess_geometry(np.asarray(a, dtype=np.float32))
    if geometry == "processed_pixels":
        expected = camera.processed_size[::-1]
        if a.shape != expected:
            raise ValueError(f"Processed map for {camera.name} must have shape {expected}, got {a.shape}.")
        return np.asarray(a, dtype=np.float32)
    raise ValueError(f"Unsupported image geometry {geometry!r}.")


def auxiliary_grid(a: np.ndarray, camera: CameraModel, shape: tuple[int,int],
                   geometry: str, *, support: bool) -> np.ndarray:
    # Unknown support is neutral (1); unknown validity means invalid (0).
    a = np.where(np.isfinite(a), a, 1.0 if support else 0.0).astype(np.float32)
    if geometry == "measurement_grid":
        if a.shape != shape:
            raise ValueError(f"Map for {camera.name} must match measurement grid {shape}, got {a.shape}.")
        return a
    a = _map_to_processed(a, camera, geometry)
    rows, cols = shape
    if rows > a.shape[0] or cols > a.shape[1]:
        raise ValueError("Cannot upsample a mask to create additional independent observation cells.")
    a = cv2.resize(a, (cols,rows), interpolation=cv2.INTER_AREA)
    if not support:
        # Reject any incompletely valid footprint; never change its path kernel silently.
        a = np.where(a >= 1.0-1e-6, 1.0, 0.0)
    return a.astype(np.float32)


def _direct_grid(values, validity, camera, shape, geometry):
    valid = np.isfinite(values)
    if validity is not None:
        if validity.shape != values.shape:
            raise ValueError("Direct projection validity must share the projection's geometry and shape.")
        valid &= np.isfinite(validity) & (validity > 0)
    if geometry == "measurement_grid":
        if values.shape != shape:
            raise ValueError(f"Column-density grid for {camera.name} must be {shape}, got {values.shape}.")
        weights = valid.astype(np.float32)
        if validity is not None:
            weights *= np.where(np.isfinite(validity), validity, 0)
        return np.where(valid, values, 0).astype(np.float32), weights
    coverage = _map_to_processed(valid.astype(np.float32), camera, geometry)
    numerator = _map_to_processed(np.where(valid, values, 0).astype(np.float32), camera, geometry)
    processed = np.divide(numerator, coverage, out=np.zeros_like(numerator), where=coverage>1e-8)
    complete = coverage >= 1-1e-6
    if validity is not None:
        quality = _map_to_processed(np.where(np.isfinite(validity),validity,0),camera,geometry)
        complete &= quality >= 1-1e-6
    rows, cols = shape
    if rows>processed.shape[0] or cols>processed.shape[1]:
        raise ValueError("Observation grid exceeds the available column-density image resolution.")
    fraction = cv2.resize(complete.astype(np.float32),(cols,rows),interpolation=cv2.INTER_AREA)
    signal = cv2.resize(np.where(complete,processed,0),(cols,rows),interpolation=cv2.INTER_AREA)
    accepted = fraction>=1-1e-6
    return np.where(accepted,signal,0).astype(np.float32),accepted.astype(np.float32)


def prepare_input_measurements(cfg: dict, cameras: list[CameraModel], paths: dict,
                               *, background_paths: dict | None = None,
                               support_paths: dict | None = None,
                               validity_paths: dict | None = None,
                               kind: str | None = None) -> MeasurementData:
    """Create observations from raw images OR already-calibrated line integrals.

    No result from a segmenter is interpreted as concentration. Missing raw frames
    are rejected rather than synthesized. Each supplied filename is used read-only.
    """
    from .reconstruction import geometry_signature
    names = {c.name for c in cameras}
    if set(paths) != names:
        raise ValueError(f"Provide one input per configured camera: {sorted(names)}.")
    for label, mapping in (("background",background_paths),("support",support_paths),("validity",validity_paths)):
        if set(mapping or {})-names:
            raise ValueError(f"Unknown {label} camera name(s): {set(mapping)-names}.")
    shapes = measurement_shapes(cfg,cameras)
    settings = cfg.get("input",{})
    kind = kind or settings.get("kind","raw_images")
    valid_arrays = {n: score_array(v,"validity") for n,v in (validity_paths or {}).items()}
    if kind == "raw_images":
        vg = settings.get("validity_geometry", "raw_pixels")
        if valid_arrays and vg != "raw_pixels":
            raise ValueError("Raw-image validity must use raw_pixels so invalid pixels are removed before filtering.")
        data = prepare_measurements(cameras, paths, background_paths, shapes, cfg["preprocess"],
                                    raw_validity_masks=valid_arrays)
    elif kind == "column_density":
        if background_paths:
            raise ValueError("Direct column-density input must not include a background (no double calibration).")
        unit = settings.get("column_density_unit")
        if not unit or unit != cfg["units"].get("column_density"):
            raise ValueError("Declare input.column_density_unit equal to units.column_density; convert units explicitly upstream.")
        geometry = settings.get("geometry","measurement_grid")
        if settings.get("validity_geometry",geometry) != geometry and valid_arrays:
            raise ValueError("Direct input validity_geometry must match input.geometry.")
        grids, confidence, diagnostic = {}, {}, {}
        for camera in cameras:
            values, stored_unit = read_array_2d(paths[camera.name], key=settings.get("npz_key","column_density"))
            if stored_unit is not None and stored_unit != unit:
                raise ValueError(f"Projection unit '{stored_unit}' in {camera.name} does not match '{unit}'.")
            signal, weight = _direct_grid(np.asarray(values,dtype=np.float32), valid_arrays.get(camera.name),
                                         camera, shapes[camera.name], geometry)
            grids[camera.name], confidence[camera.name] = signal, weight
            diagnostic[camera.name] = {"input_kind":kind,"geometry":geometry,"column_density_unit":unit,
                                       "invalid_bins":int(np.count_nonzero(weight<=0))}
        vector = np.concatenate([grids[c.name].ravel() for c in cameras])
        weights = np.concatenate([confidence[c.name].ravel() for c in cameras])
        norm = cfg["preprocess"].get("measurement_normalization","none")
        fixed = cfg["preprocess"].get("fixed_measurement_scale")
        scale = _measurement_scale(vector[weights>0],norm) if fixed is None else float(fixed)
        if not np.isfinite(scale) or scale<=0:
            raise ValueError("Measurement normalization scale must be finite and positive.")
        data = MeasurementData(vector/scale,weights,{n:g/scale for n,g in grids.items()},confidence,
                               scale,norm,diagnostic)
    else:
        raise ValueError("Input kind must be raw_images or column_density.")
    support_geometry = cfg.get("support",{}).get("external_geometry","measurement_grid")
    for camera in cameras:
        if camera.name in (support_paths or {}):
            a = score_array(support_paths[camera.name])
            score = auxiliary_grid(a,camera,shapes[camera.name],support_geometry,support=True)
            score[data.confidence_grids[camera.name]<=0] = 1.0
            data.support_grids[camera.name] = score
    if cfg["support"].get("enabled",True) and cfg["support"].get("source")=="external" and set(data.support_grids)!=names:
        raise ValueError("External support mode requires one support map per camera.")
    data.geometry_signature = geometry_signature(cfg,cameras)
    data.diagnostics["input_contract"] = {"kind":kind,"unit":cfg.get("units",{}).get("column_density"),
        "external_support_cameras":sorted(data.support_grids),"geometry_signature":data.geometry_signature}
    return data
