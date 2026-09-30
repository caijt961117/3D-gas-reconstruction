from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from .camera import CameraModel
from .sampling import shape_for


@dataclass
class CameraMeasurement:
    camera_name: str
    signal_grid: np.ndarray
    confidence_grid: np.ndarray
    processed_intensity: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasurementData:
    vector: np.ndarray
    weights: np.ndarray
    grids: dict[str, np.ndarray]
    confidence_grids: dict[str, np.ndarray]
    scale: float
    normalization: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    support_grids: dict[str, np.ndarray] = field(default_factory=dict)
    geometry_signature: str | None = None


def _merge_camera_preprocess(base: dict[str, Any], camera_name: str) -> dict[str, Any]:
    merged = {key: value for key, value in base.items() if key != "per_camera"}
    camera_specific = base.get("per_camera", {}).get(camera_name, {})
    merged.update(camera_specific)
    return merged


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = tifffile.imread(path)
        # tifffile returns RGB/RGBA, whereas OpenCV uses BGR/BGRA. Normalize here.
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[..., ::-1]
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[..., [2, 1, 0, 3]]
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"OpenCV could not read image: {path}")
    if image.ndim > 3:
        image = np.squeeze(image)
    if image.ndim not in {2, 3}:
        raise ValueError(f"Unsupported image shape {image.shape} for {path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[..., 0]
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported channel count: {image.shape[2]}")


def to_float01(image: np.ndarray, intensity_scale: float | None = None) -> np.ndarray:
    """Use a fixed sensor scale; never guess a floating-point image's range.

    Float data without a scale must already be in [0,1]. Invalid values remain
    invalid until the validity mask is built, not clipped into valid readings.
    For 14-bit DN in uint16 containers explicitly set intensity_scale=16383.
    """
    array = np.asarray(image)
    if intensity_scale is not None:
        scale = float(intensity_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("intensity_scale must be finite and positive.")
    elif np.issubdtype(array.dtype, np.integer):
        scale = float(np.iinfo(array.dtype).max)
    else:
        finite = array[np.isfinite(array)]
        if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
            raise ValueError("Floating-point DN/radiance requires a fixed intensity_scale; "
                             "only pre-normalized floats in [0,1] may omit it.")
        scale = 1.0
    return np.asarray(array, dtype=np.float32) / np.float32(scale)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    kernel = max(3, int(np.ceil(6.0 * sigma)) | 1)
    return cv2.GaussianBlur(
        image,
        (kernel, kernel),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT101,
    )


def _compute_signal(
    gas: np.ndarray,
    background: np.ndarray | None,
    config: dict[str, Any],
) -> np.ndarray:
    mode = str(config.get("mode", "inverse_intensity")).lower()
    darker = bool(config.get("gas_is_darker", True))
    eps = float(config.get("background_epsilon", 1.0e-5))

    if mode == "inverse_intensity":
        signal = 1.0 - gas if darker else gas.copy()
    else:
        if background is None:
            raise ValueError(f"Preprocess mode '{mode}' requires a background image.")
        if background.shape != gas.shape:
            raise ValueError("Gas and background images have different processed shapes.")
        if darker:
            difference = background - gas
            numerator, denominator = background, gas
        else:
            difference = gas - background
            numerator, denominator = gas, background

        if mode == "difference":
            signal = difference
        elif mode == "normalized_difference":
            signal = difference / (background + eps)
        elif mode == "log_ratio":
            signal = np.log((numerator + eps) / (denominator + eps))
        else:
            raise ValueError(
                "preprocess.mode must be inverse_intensity, difference, "
                "normalized_difference or log_ratio."
            )
    # Preserve signed noise. The unknown concentration, not the observation, is nonnegative.
    return signal.astype(np.float32, copy=False)


def preprocess_camera_image(
    camera: CameraModel,
    image_path: str | Path,
    background_path: str | Path | None,
    grid_shape: tuple[int, int],
    preprocess_cfg: dict[str, Any],
    raw_validity_mask: np.ndarray | None = None,
) -> CameraMeasurement:
    """Mask raw invalid pixels BEFORE geometry, filtering and response averaging.

    Default partial_bin_policy='reject' removes every partially invalid bin,
    preserving agreement with the cached full-bin forward operator. An explicit
    masked_mean mode is available, but its area mismatch is reported, not hidden.
    """
    camera_cfg = _merge_camera_preprocess(preprocess_cfg, camera.name)
    raw = to_grayscale(read_image(image_path))
    gas = to_float01(raw, camera_cfg.get("intensity_scale"))
    background = None
    if background_path is not None:
        raw_bg = to_grayscale(read_image(background_path))
        if raw_bg.shape != raw.shape:
            raise ValueError("Gas and background raw image shapes differ.")
        if camera_cfg.get("intensity_scale") is None and raw_bg.dtype != raw.dtype:
            raise ValueError("Mixed gas/background dtypes require an explicit common intensity_scale.")
        background = to_float01(raw_bg, camera_cfg.get("intensity_scale"))

    lo = float(camera_cfg.get("saturation_low", 0.002))
    hi = float(camera_cfg.get("saturation_high", 0.998))
    if not lo < hi:
        raise ValueError("saturation_low must be less than saturation_high.")
    valid = np.isfinite(gas) & (gas > lo) & (gas < hi)
    if background is not None:
        valid &= np.isfinite(background) & (background > lo) & (background < hi)
    bad_mask_path = camera_cfg.get("bad_pixel_mask")
    if bad_mask_path:
        bad = to_grayscale(read_image(bad_mask_path))
        if bad.shape != gas.shape:
            raise ValueError("bad_pixel_mask must match the raw image shape (nonzero=bad).")
        valid &= bad == 0

    if raw_validity_mask is not None:
        external = np.asarray(raw_validity_mask)
        if external.shape != gas.shape:
            raise ValueError("Raw validity mask must match the raw image shape.")
        valid &= np.isfinite(external) & (external >= 1.0-1e-6)
    # Remapping numerator and mask with identical kernels excludes bad-pixel DN.
    raw_mask = valid.astype(np.float32)
    coverage = np.clip(camera.preprocess_geometry(raw_mask), 0, 1)
    def remap_masked(image):
        numerator = camera.preprocess_geometry(np.where(valid, image, 0.0).astype(np.float32))
        return np.divide(numerator, coverage, out=np.zeros_like(numerator), where=coverage > 1e-8)
    gas = remap_masked(gas)
    if background is not None:
        background = remap_masked(background)
    # Require complete interpolation neighborhoods, so dead pixels cannot spread.
    valid_pixels = (coverage >= 1.0 - 1e-6).astype(np.float32)
    sigma = float(camera_cfg.get("denoise_sigma_px", 0.0))
    blur_den = _gaussian_blur(valid_pixels, sigma)
    def masked_blur(image):
        numerator = _gaussian_blur(image * valid_pixels, sigma)
        return np.divide(numerator, blur_den, out=np.zeros_like(numerator), where=blur_den > 1e-8)
    gas_filtered = masked_blur(gas)
    bg_filtered = masked_blur(background) if background is not None else None
    response = _compute_signal(gas_filtered, bg_filtered, camera_cfg)
    valid_pixels *= np.isfinite(response).astype(np.float32)
    response = np.where(valid_pixels > 0, response, 0).astype(np.float32)
    rows, cols = map(int, grid_shape)
    if rows > response.shape[0] or cols > response.shape[1]:
        raise ValueError("Measurement grid cannot be larger than the processed image.")
    def area(image):
        return cv2.resize(image, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32)
    fraction = np.clip(area(valid_pixels), 0, 1)
    signal_grid = np.divide(area(response), fraction, out=np.zeros_like(fraction), where=fraction > 1e-8)
    policy = str(camera_cfg.get("partial_bin_policy", "reject"))
    min_fraction = float(camera_cfg.get("minimum_valid_fraction", 0.8))
    if not 0 < min_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be in (0,1].")
    if policy not in {"reject", "masked_mean"}:
        raise ValueError("partial_bin_policy must be reject or masked_mean.")
    accepted = fraction >= ((1.0 - 1e-6) if policy == "reject" else min_fraction)
    confidence = np.where(accepted, fraction, 0).astype(np.float32)

    baseline = 0.0
    q = camera_cfg.get("baseline_quantile")
    if q is not None:
        q = float(q)
        if not 0 <= q <= 1:
            raise ValueError("baseline_quantile must lie in [0,1].")
        values = signal_grid[accepted]
        if values.size:
            baseline = float(np.quantile(values, q))
            signal_grid -= baseline
    coefficients = np.asarray(camera_cfg.get("calibration_coefficients", [0.0, 1.0]), dtype=float)
    if coefficients.ndim != 1 or not coefficients.size or not np.isfinite(coefficients).all():
        raise ValueError("Calibration coefficients must be a finite, nonempty list.")
    signal_grid = np.polynomial.polynomial.polyval(signal_grid, coefficients).astype(np.float32)
    cmin, cmax = camera_cfg.get("clip_min"), camera_cfg.get("clip_max")
    if cmin is not None:
        signal_grid = np.maximum(signal_grid, float(cmin))
    if cmax is not None:
        signal_grid = np.minimum(signal_grid, float(cmax))
    confidence[~np.isfinite(signal_grid)] = 0
    signal_grid = np.where(confidence > 0, signal_grid, 0).astype(np.float32)
    diagnostics = {
        "valid_raw_pixel_fraction": float(np.mean(valid)),
        "valid_bin_fraction": float(np.mean(confidence > 0)),
        "rejected_bins": int(np.count_nonzero(confidence == 0)),
        "baseline_subtracted": baseline,
        "partial_bin_policy": policy,
        "partial_area_operator_approximation": bool(policy == "masked_mean" and np.any((confidence > 0) & (fraction < 1-1e-6))),
        "signed_observations_preserved": cmin is None,
        "mode": camera_cfg.get("mode", "inverse_intensity"),
    }
    return CameraMeasurement(camera.name, signal_grid, confidence, gas, diagnostics)


def _measurement_scale(vector: np.ndarray, mode: str) -> float:
    positive = np.abs(vector[np.isfinite(vector)])
    positive = positive[positive > 0]
    if positive.size == 0 or mode == "none":
        return 1.0
    if mode == "max":
        value = float(np.max(positive))
    elif mode == "p99":
        value = float(np.quantile(positive, 0.99))
    elif mode == "p995":
        value = float(np.quantile(positive, 0.995))
    elif mode == "l2":
        value = float(np.linalg.norm(positive))
    else:
        raise ValueError("measurement_normalization must be none, max, p99, p995 or l2.")
    return value if value > 1.0e-12 else 1.0


def prepare_measurements(
    cameras: list[CameraModel],
    image_paths: dict[str, str | Path],
    background_paths: dict[str, str | Path | None] | None,
    grid_shape: tuple[int, int],
    preprocess_cfg: dict[str, Any],
    raw_validity_masks: dict[str, np.ndarray] | None = None,
) -> MeasurementData:
    if len(cameras) < 2:
        raise ValueError("At least two cameras are required.")
    background_paths = background_paths or {}
    measurements: list[CameraMeasurement] = []
    for camera in cameras:
        if camera.name not in image_paths:
            raise KeyError(f"Missing image path for camera '{camera.name}'.")
        measurements.append(
            preprocess_camera_image(
                camera=camera,
                image_path=image_paths[camera.name],
                background_path=background_paths.get(camera.name),
                grid_shape=shape_for(grid_shape, camera.name),
                preprocess_cfg=preprocess_cfg,
                raw_validity_mask=(raw_validity_masks or {}).get(camera.name),
            )
        )

    original_vector = np.concatenate(
        [measurement.signal_grid.ravel() for measurement in measurements]
    ).astype(np.float32)
    weights = np.concatenate(
        [measurement.confidence_grid.ravel() for measurement in measurements]
    ).astype(np.float32)
    normalization = str(preprocess_cfg.get("measurement_normalization", "none")).lower()
    scale = _measurement_scale(original_vector[weights > 0], normalization)
    if preprocess_cfg.get("fixed_measurement_scale") is not None:
        scale = float(preprocess_cfg["fixed_measurement_scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("fixed_measurement_scale must be finite and positive.")
    vector = (original_vector / np.float32(scale)).astype(np.float32)
    grids = {
        measurement.camera_name: measurement.signal_grid / np.float32(scale)
        for measurement in measurements
    }
    confidence_grids = {
        measurement.camera_name: measurement.confidence_grid
        for measurement in measurements
    }
    return MeasurementData(
        vector=vector,
        weights=weights,
        grids=grids,
        confidence_grids=confidence_grids,
        scale=scale,
        normalization=normalization,
        diagnostics={m.camera_name: m.diagnostics for m in measurements},
    )


def make_median_background(
    image_paths: list[str | Path], output_path: str | Path
) -> dict[str, Any]:
    """Create a static no-gas background by a per-pixel median."""
    if len(image_paths) < 3:
        raise ValueError("Use at least three no-gas frames to build a median background.")
    images = [read_image(path) for path in image_paths]
    first_shape = images[0].shape
    first_dtype = images[0].dtype
    if any(image.shape != first_shape for image in images):
        raise ValueError("All background frames must have identical shape.")
    stack = np.stack([np.asarray(image, dtype=np.float32) for image in images], axis=0)
    median = np.median(stack, axis=0)
    if np.issubdtype(first_dtype, np.integer):
        info = np.iinfo(first_dtype)
        median = np.clip(np.rint(median), info.min, info.max).astype(first_dtype)
    else:
        median = median.astype(first_dtype)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".tif", ".tiff"}:
        tifffile.imwrite(output_path, median)
    else:
        if not cv2.imwrite(str(output_path), median):
            raise RuntimeError(f"Could not write background image: {output_path}")
    return {
        "output": str(output_path),
        "frames": len(images),
        "shape": list(first_shape),
        "dtype": str(first_dtype),
    }
