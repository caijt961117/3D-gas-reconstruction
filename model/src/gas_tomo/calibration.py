from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .image_processing import read_image, to_grayscale


def _detection_image(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(to_grayscale(image), dtype=np.float32)
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        raise ValueError("Calibration image contains no finite pixels.")
    low, high = np.quantile(finite, [0.01, 0.99])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((gray - low) / (high - low), 0.0, 1.0)
    return np.round(normalized * 255.0).astype(np.uint8)


def calibrate_intrinsics(
    image_paths: Iterable[str | Path],
    board_cols: int,
    board_rows: int,
    square_size_m: float,
    output_path: str | Path,
) -> dict:
    paths = [Path(path) for path in image_paths]
    if len(paths) < 5:
        raise ValueError("Use at least five calibration images; more varied views are better.")
    board_size = (int(board_cols), int(board_rows))
    object_template = np.zeros((board_cols * board_rows, 3), dtype=np.float32)
    object_template[:, :2] = (
        np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * float(square_size_m)
    )

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    used_paths: list[str] = []

    for path in paths:
        image = read_image(path)
        detection = _detection_image(image)
        current_size = (detection.shape[1], detection.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError("All intrinsic calibration images must have identical size.")

        found = False
        corners = None
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                detection,
                board_size,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
            )
        if not found:
            found, corners = cv2.findChessboardCorners(
                detection,
                board_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                corners = cv2.cornerSubPix(
                    detection,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria=(
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        40,
                        1.0e-4,
                    ),
                )
        if found and corners is not None:
            object_points.append(object_template.copy())
            image_points.append(corners.astype(np.float32))
            used_paths.append(str(path))

    if image_size is None or len(object_points) < 5:
        raise RuntimeError(
            f"Checkerboard was detected in only {len(object_points)} images; at least 5 are required."
        )

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    reprojection_errors = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        error = np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1)))
        reprojection_errors.append(float(error))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        K=K.astype(np.float64),
        dist=dist.reshape(-1).astype(np.float64),
        image_size=np.array(image_size, dtype=np.int32),
        rms=np.array([rms], dtype=np.float64),
    )
    return {
        "output": str(output_path),
        "images_provided": len(paths),
        "images_used": len(used_paths),
        "used_images": used_paths,
        "rms": float(rms),
        "mean_reprojection_error_px": float(np.mean(reprojection_errors)),
        "max_reprojection_error_px": float(np.max(reprojection_errors)),
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "image_size": list(image_size),
    }


def _read_correspondences(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    required = {"x_m", "y_m", "z_m", "u_px", "v_px"}
    object_points: list[list[float]] = []
    image_points: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Correspondence CSV needs a header row.")
        normalized_names = {name.lower().strip(): name for name in reader.fieldnames}
        missing = required - set(normalized_names)
        if missing:
            raise ValueError(
                f"Correspondence CSV is missing columns: {sorted(missing)}. "
                "Required: x_m,y_m,z_m,u_px,v_px"
            )
        for row in reader:
            object_points.append(
                [
                    float(row[normalized_names["x_m"]]),
                    float(row[normalized_names["y_m"]]),
                    float(row[normalized_names["z_m"]]),
                ]
            )
            image_points.append(
                [
                    float(row[normalized_names["u_px"]]),
                    float(row[normalized_names["v_px"]]),
                ]
            )
    if len(object_points) < 6:
        raise ValueError("At least six 3D-2D point correspondences are required.")
    return np.asarray(object_points, dtype=np.float64), np.asarray(image_points, dtype=np.float64)


def calibrate_extrinsics(
    intrinsics_path: str | Path,
    correspondences_csv: str | Path,
    output_path: str | Path,
) -> dict:
    intrinsics = np.load(Path(intrinsics_path), allow_pickle=False)
    K = intrinsics["K"].astype(np.float64)
    dist = (
        intrinsics["dist"].astype(np.float64).reshape(-1)
        if "dist" in intrinsics.files
        else np.zeros(5, dtype=np.float64)
    )
    image_size = intrinsics["image_size"].astype(np.int32).reshape(-1)[:2]
    object_points, image_points = _read_correspondences(correspondences_csv)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=500,
        reprojectionError=4.0,
        confidence=0.999,
    )
    if not success:
        raise RuntimeError("solvePnPRansac failed to estimate camera pose.")
    if inliers is None or len(inliers) < 6:
        raise RuntimeError("Too few PnP inliers; recheck world/image point correspondences.")
    inlier_indices = inliers.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points[inlier_indices],
        image_points[inlier_indices],
        K,
        dist,
        rvec,
        tvec,
    )
    R, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    camera_center = -R.T @ tvec.reshape(3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        K=K,
        dist=dist,
        R=R.astype(np.float64),
        t=tvec.reshape(3).astype(np.float64),
        image_size=image_size,
        reprojection_errors=errors,
        inliers=inlier_indices.astype(np.int32),
    )
    return {
        "output": str(output_path),
        "points": int(object_points.shape[0]),
        "inliers": int(inlier_indices.size),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "median_reprojection_error_px": float(np.median(errors)),
        "max_reprojection_error_px": float(np.max(errors)),
        "camera_center_m": camera_center.tolist(),
        "R": R.tolist(),
        "t": tvec.reshape(3).tolist(),
    }


def draw_volume_overlay(
    calibration_path: str | Path,
    image_path: str | Path,
    bounds: np.ndarray,
    output_path: str | Path,
) -> None:
    data = np.load(Path(calibration_path), allow_pickle=False)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64).reshape(-1)
    R = data["R"].astype(np.float64)
    t = data["t"].astype(np.float64).reshape(3)
    rvec, _ = cv2.Rodrigues(R)
    image = read_image(image_path)
    gray = _detection_image(image)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    zmin, zmax = bounds[2]
    corners = np.array(
        [
            [xmin, ymin, zmin], [xmax, ymin, zmin],
            [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax],
            [xmax, ymax, zmax], [xmin, ymax, zmax],
        ],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(corners, rvec, t, K, dist)
    projected = np.round(projected.reshape(-1, 2)).astype(int)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for first, second in edges:
        cv2.line(canvas, tuple(projected[first]), tuple(projected[second]), (0, 255, 0), 2)
    for index, point in enumerate(projected):
        cv2.circle(canvas, tuple(point), 4, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            str(index),
            tuple(point + np.array([5, -5])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write overlay image: {output_path}")


def fit_response_polynomial(
    csv_path: str | Path,
    degree: int = 2,
    output_json: str | Path | None = None,
) -> dict:
    """
    Fit column_density = c0 + c1*signal + ... using calibration measurements.

    CSV columns: signal,column_density. Coefficients are returned in ascending order,
    matching numpy.polynomial.polynomial.polyval and the YAML configuration.
    """
    signals: list[float] = []
    densities: list[float] = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Response calibration CSV needs a header row.")
        names = {name.lower().strip(): name for name in reader.fieldnames}
        required = {"signal", "column_density"}
        missing = required - set(names)
        if missing:
            raise ValueError("CSV columns must include signal,column_density.")
        for row in reader:
            signals.append(float(row[names["signal"]]))
            densities.append(float(row[names["column_density"]]))
    if len(signals) < degree + 2:
        raise ValueError("Not enough calibration points for the requested polynomial degree.")
    x = np.asarray(signals, dtype=np.float64)
    y = np.asarray(densities, dtype=np.float64)
    coefficients = np.polynomial.polynomial.polyfit(x, y, deg=int(degree))
    predicted = np.polynomial.polynomial.polyval(x, coefficients)
    residual = predicted - y
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    report = {
        "degree": int(degree),
        "points": int(x.size),
        "coefficients_ascending": coefficients.tolist(),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        "signal_min": float(np.min(x)),
        "signal_max": float(np.max(x)),
        "column_density_min": float(np.min(y)),
        "column_density_max": float(np.max(y)),
    }
    if output_json is not None:
        import json

        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        report["output"] = str(output_path)
    return report
