from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .camera import CameraModel
from .grid import VolumeGrid
from .sampling import shape_for
from hashlib import sha256
import json


@dataclass
class SupportData:
    volume: np.ndarray  # flat, in [0, 1]
    camera_scores: dict[str, np.ndarray]
    camera_masks: dict[str, np.ndarray]


def _camera_support_score(signal: np.ndarray, cfg: dict, confidence: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32)
    known = np.ones_like(signal, dtype=bool) if confidence is None else (np.asarray(confidence) > 0)
    signal = np.where(known & np.isfinite(signal), signal, 0)
    maximum = float(np.max(signal, initial=0.0))
    threshold = max(
        float(cfg.get("minimum_threshold", 0.0)),
        float(cfg.get("threshold_fraction", 0.1)) * maximum,
    )
    if maximum <= 0.0:
        mask = np.zeros_like(signal, dtype=bool)
    else:
        mask = signal >= threshold

    mask[~known] = True  # Missing observations are unknown, never evidence of no gas.
    closing_cells = int(cfg.get("closing_cells", 0))
    dilation_cells = int(cfg.get("dilation_cells", 0))
    if closing_cells > 0:
        structure = ndimage.generate_binary_structure(2, 2)
        mask = ndimage.binary_closing(mask, structure=structure, iterations=closing_cells, border_value=1)
    if dilation_cells > 0:
        structure = ndimage.generate_binary_structure(2, 2)
        mask = ndimage.binary_dilation(mask, structure=structure, iterations=dilation_cells)

    score = mask.astype(np.float32)
    sigma = float(cfg.get("gaussian_sigma_cells", 0.0))
    if sigma > 0:
        score = ndimage.gaussian_filter(score, sigma=sigma, mode="nearest")
    max_score = float(score.max(initial=0.0))
    if max_score > 0:
        score /= max_score
    score[~known] = 1.0
    return score.astype(np.float32), mask


def build_visual_hull_support(
    cameras: list[CameraModel],
    signal_grids: dict[str, np.ndarray],
    grid: VolumeGrid,
    measurement_shape: tuple[int, int],
    cfg: dict,
    confidence_grids: dict[str, np.ndarray] | None = None,
    projection_cache: dict | None = None,
    external_scores: dict[str, np.ndarray] | None = None,
) -> SupportData:
    if not bool(cfg.get("enabled", True)) or cfg.get("source") == "none":
        ones = np.ones(grid.n_voxels, dtype=np.float32)
        return SupportData(volume=ones, camera_scores={}, camera_masks={})

    centers = grid.centers()
    sampled_scores: list[np.ndarray] = []
    camera_scores: dict[str, np.ndarray] = {}
    camera_masks: dict[str, np.ndarray] = {}

    for camera in cameras:
        if camera.name not in signal_grids:
            raise KeyError(f"Missing signal grid for camera '{camera.name}'.")
        confidence = None if confidence_grids is None else confidence_grids.get(camera.name)
        cam_shape = shape_for(measurement_shape, camera.name)
        local_cfg = {**cfg, **cfg.get("per_camera", {}).get(camera.name, {})}
        if camera.name in (external_scores or {}):
            score = np.asarray(external_scores[camera.name], dtype=np.float32).copy()
            if score.shape != cam_shape:
                raise ValueError(f"External support for {camera.name} must match {cam_shape}.")
            finite = np.isfinite(score)
            if np.any((score[finite] < 0) | (score[finite] > 1)):
                raise ValueError("External support scores must be in [0,1].")
            unknown = ~finite
            if confidence is not None:
                unknown |= np.asarray(confidence) <= 0
            score[unknown] = 1.0
            # Scores are used as supplied; no implicit thresholding or max-normalization.
            mask = score >= 0.5
        elif local_cfg.get("source") == "external":
            raise ValueError(f"support.source=external requires a mask for {camera.name}.")
        else:
            score, mask = _camera_support_score(signal_grids[camera.name], local_cfg, confidence)
        camera_scores[camera.name] = score
        camera_masks[camera.name] = mask
        camera_key = sha256(json.dumps(camera.metadata(), sort_keys=True).encode()).hexdigest()
        key = (camera_key, grid.shape, tuple(grid.bounds.ravel()), cam_shape)
        if projection_cache is not None and key in projection_cache:
            grid_coordinates, valid = projection_cache[key]
        else:
            grid_coordinates, valid = camera.project_world_to_grid(centers, cam_shape)
            if projection_cache is not None:
                projection_cache[key] = (grid_coordinates, valid)
        sampled = ndimage.map_coordinates(
            score,
            [np.where(valid, grid_coordinates[:, 0], 0), np.where(valid, grid_coordinates[:, 1], 0)],
            order=1,
            mode="constant",
            cval=1.0,
            prefilter=False,
        ).astype(np.float32)
        sampled[~valid] = 1.0
        sampled_scores.append(sampled)

    combine = str(cfg.get("combine", "geometric_mean")).lower()
    if combine == "product":
        support = np.prod(sampled_scores, axis=0)
    elif combine == "minimum":
        support = np.min(sampled_scores, axis=0)
    elif combine == "geometric_mean":
        support = np.maximum(np.prod(sampled_scores, axis=0), 0.0) ** (1.0/len(sampled_scores))
    elif combine == "arithmetic_mean":
        support = np.mean(sampled_scores, axis=0)
    else:
        raise ValueError(
            "support.combine must be product, minimum, geometric_mean or arithmetic_mean."
        )

    support = np.clip(support, 0.0, 1.0).astype(np.float32)
    return SupportData(
        volume=support,
        camera_scores=camera_scores,
        camera_masks=camera_masks,
    )
