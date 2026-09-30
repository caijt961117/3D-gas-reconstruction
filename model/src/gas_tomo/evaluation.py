from __future__ import annotations

from typing import Any
import numpy as np
from .reconstruction import ReconstructionResult


def _centroid(volume: np.ndarray, centers: np.ndarray) -> list[float] | None:
    weights = np.maximum(np.asarray(volume, dtype=float).ravel(), 0)
    total = float(weights.sum())
    if total <= 0:
        return None
    return ((centers*weights[:, None]).sum(axis=0)/total).tolist()


def _residual_metrics(observed, predicted, weights):
    observed, predicted, weights = [np.asarray(v, dtype=float) for v in (observed, predicted, weights)]
    valid = (weights > 0) & np.isfinite(observed) & np.isfinite(predicted)
    if not np.any(valid):
        return {"weighted_relative_residual": None, "weighted_rmse": None,
                "rmse": None, "valid_rows": 0, "evaluation_status": "no_valid_data"}
    b, r, w = observed[valid], (predicted-observed)[valid], weights[valid]
    bn = float(np.linalg.norm(np.sqrt(w)*b))
    return {"weighted_relative_residual": float(np.linalg.norm(np.sqrt(w)*r)/bn) if bn > 0 else None,
            "weighted_rmse": float(np.sqrt(np.dot(w, r*r)/w.sum())),
            "rmse": float(np.sqrt(np.mean(r*r))), "valid_rows": int(valid.sum()),
            "evaluation_status": "valid" if bn > 0 else "valid_zero_signal_relative_residual_undefined"}


def reconstruction_metrics(result: ReconstructionResult, truth: np.ndarray | None = None) -> dict[str, Any]:
    quality = dict(result.quality)
    has_estimate = quality.get("has_estimate", True)
    volume = np.asarray(result.volume, dtype=float)
    centers = result.grid.centers()
    residuals = _residual_metrics(result.observed, result.predicted, result.weights)
    peak = float(np.max(volume)) if has_estimate else None
    metrics = {
        "measurement_scale": float(result.measurement_scale),
        "quality": quality, "frame_id": result.frame_id, "timestamp_s": result.timestamp_s,
        "numeric_placeholder": not has_estimate,
        "weighted_relative_projection_residual": residuals["weighted_relative_residual"],
        "weighted_projection_rmse": residuals["weighted_rmse"],
        "projection_rmse": residuals["rmse"],
        "projection_evaluation_status": residuals["evaluation_status"],
        "maximum_concentration": peak,
        "maximum_location_m": list(result.grid.max_location(volume)) if peak is not None and peak > 0 else None,
        "concentration_integral": float(volume.sum()*result.grid.voxel_volume) if has_estimate else None,
        "positive_voxel_fraction": float(np.mean(volume > 0)) if has_estimate else None,
        "centroid_m": _centroid(volume, centers) if has_estimate else None,
        "grid_shape_nx_ny_nz": list(result.grid.shape),
        "voxel_spacing_m": result.grid.spacing.tolist(),
        "per_camera": {},
    }
    for camera in result.cameras:
        cs = result.camera_slices[camera.name]
        cm = _residual_metrics(result.observed[cs], result.predicted[cs], result.weights[cs])
        cm["shape_rows_cols"] = list(result.measurement_shape)
        metrics["per_camera"][camera.name] = cm
    touching = []
    if peak is not None and peak > 0:
        faces = {"x_min": volume[:, :, 0], "x_max": volume[:, :, -1],
                 "y_min": volume[:, 0, :], "y_max": volume[:, -1, :],
                 "z_min": volume[0, :, :], "z_max": volume[-1, :, :]}
        touching = [name for name, face in faces.items() if np.any(face >= 0.05*peak)]
    metrics["boundary_touch_at_5pct_peak"] = touching
    if truth is not None:
        truth = np.asarray(truth, dtype=float)
        if truth.shape != volume.shape or not np.isfinite(truth).all():
            raise ValueError("Truth must be finite and match the reconstruction shape.")
        if not has_estimate:
            metrics["truth"] = {"nrmse": None, "correlation": None, "reason": "no_estimate"}
        else:
            tn, tm, ts = float(np.linalg.norm(truth)), float(np.max(truth)), float(truth.sum())
            mask_true, mask_rec = truth >= 0.1*tm, volume >= 0.1*tm
            intersection, union = (mask_true & mask_rec).sum(), (mask_true | mask_rec).sum()
            ct, cr = _centroid(truth, centers), _centroid(volume, centers)
            metrics["truth"] = {
                "nrmse": float(np.linalg.norm(volume-truth)/tn) if tn > 0 else None,
                "mae": float(np.mean(np.abs(volume-truth))),
                "correlation": float(np.corrcoef(truth.ravel(), volume.ravel())[0,1]) if np.std(truth)>0 and np.std(volume)>0 else None,
                "maximum_relative_error": abs(peak-tm)/tm if tm > 0 else None,
                "peak_signed_relative_error": (peak-tm)/tm if tm > 0 else None,
                "maximum_location_error_m": float(np.linalg.norm(np.asarray(result.grid.max_location(volume))-np.asarray(result.grid.max_location(truth)))) if peak>0 and tm>0 else None,
                "integral_relative_error": abs(float(volume.sum())-ts)/ts if ts>0 else None,
                "truth_centroid_m": ct,
                "centroid_error_m": float(np.linalg.norm(np.asarray(ct)-np.asarray(cr))) if ct is not None and cr is not None else None,
                "iou_at_10pct_truth_peak": float(intersection/union) if union else None,
            }
    return metrics
