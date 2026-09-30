from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np


def _normalize(vector: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


@dataclass(frozen=True)
class CameraModel:
    """
    Pinhole camera using OpenCV convention.

    R and t transform a world point into camera coordinates:
        p_camera = R @ p_world + t
    Camera axes are x-right, y-down and z-forward.
    Images are undistorted with the original K before ROI/orientation transforms.
    """

    name: str
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_size: tuple[int, int]  # (width, height)
    roi: tuple[int, int, int, int]  # (x, y, width, height) in undistorted image
    rotate_k: int = 0  # np.rot90 counter-clockwise turns after crop
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[\w-]+", self.name):
            raise ValueError("Camera names must contain only letters, digits, underscores or hyphens.")
        K = np.asarray(self.K, dtype=np.float64)
        dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)
        R = np.asarray(self.R, dtype=np.float64)
        t = np.asarray(self.t, dtype=np.float64).reshape(3)
        image_size = tuple(int(v) for v in self.image_size)
        roi = tuple(int(v) for v in self.roi)
        rotate_k = int(self.rotate_k) % 4

        if not all(np.isfinite(v).all() for v in (K, dist, R, t)):
            raise ValueError("Camera calibration must contain only finite values.")
        if K.shape != (3, 3):
            raise ValueError("K must have shape (3, 3).")
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
            raise ValueError("K must have positive focal lengths and last row [0,0,1].")
        if not np.isclose(K[0,1], 0) or not np.isclose(K[1,0], 0):
            raise ValueError("This pinhole implementation requires zero skew in K.")
        if R.shape != (3, 3):
            raise ValueError("R must have shape (3, 3).")
        if len(image_size) != 2 or min(image_size) <= 1:
            raise ValueError("image_size must be (width, height).")
        if len(roi) != 4 or roi[2] <= 1 or roi[3] <= 1:
            raise ValueError("roi must be (x, y, width, height).")
        x0, y0, width, height = roi
        if x0 < 0 or y0 < 0 or x0 + width > image_size[0] or y0 + height > image_size[1]:
            raise ValueError(f"ROI {roi} lies outside image size {image_size}.")
        if not np.allclose(R @ R.T, np.eye(3), atol=1.0e-5):
            raise ValueError("R must be approximately orthonormal.")
        if np.linalg.det(R) < 0.99:
            raise ValueError("R must be a proper rotation matrix.")

        object.__setattr__(self, "K", K)
        object.__setattr__(self, "dist", dist)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "roi", roi)
        object.__setattr__(self, "rotate_k", rotate_k)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CameraModel":
        camera_type = cfg.get("type", "calibrated")
        if camera_type == "calibrated":
            data = np.load(Path(cfg["calibration_file"]), allow_pickle=False)
            required = {"K", "R", "t", "image_size"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(
                    f"Calibration file for '{cfg['name']}' is missing: {sorted(missing)}"
                )
            K = data["K"]
            dist = data["dist"] if "dist" in data.files else np.zeros(5)
            R = data["R"]
            t = data["t"]
            image_size = tuple(int(v) for v in data["image_size"].reshape(-1)[:2])
        elif camera_type == "look_at":
            image_size = tuple(int(v) for v in cfg["image_size"])
            position = np.asarray(cfg["position_m"], dtype=np.float64)
            target = np.asarray(cfg["target_m"], dtype=np.float64)
            up_hint = np.asarray(cfg.get("up_world", [0.0, 0.0, 1.0]), dtype=np.float64)
            forward = _normalize(target - position)
            right = _normalize(np.cross(forward, up_hint))
            true_up = _normalize(np.cross(right, forward))
            down = -true_up
            R = np.stack((right, down, forward), axis=0)
            t = -R @ position

            fov_deg = float(cfg["fov_y_deg"])
            if not np.isfinite(fov_deg) or not 0 < fov_deg < 179:
                raise ValueError("fov_y_deg must be in (0,179).")
            fov_y = np.deg2rad(fov_deg)
            width, height = image_size
            fy = 0.5 * height / np.tan(0.5 * fov_y)
            pixel_aspect = float(cfg.get("pixel_aspect", 1.0))
            fx = fy * pixel_aspect
            K = np.array(
                [[fx, 0.0, (width - 1.0) / 2.0],
                 [0.0, fy, (height - 1.0) / 2.0],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            dist = np.zeros(5, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")

        roi_cfg = cfg.get("roi")
        if roi_cfg is None:
            roi = (0, 0, image_size[0], image_size[1])
        else:
            roi = tuple(int(v) for v in roi_cfg)

        return cls(
            name=str(cfg["name"]),
            K=K,
            dist=dist,
            R=R,
            t=t,
            image_size=image_size,
            roi=roi,
            rotate_k=int(cfg.get("rotate_k", 0)),
            flip_horizontal=bool(cfg.get("flip_horizontal", False)),
            flip_vertical=bool(cfg.get("flip_vertical", False)),
        )

    @property
    def center_world(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def processed_size(self) -> tuple[int, int]:
        _, _, width, height = self.roi
        if self.rotate_k % 2 == 0:
            return width, height
        return height, width

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if image.shape[1] != self.image_size[0] or image.shape[0] != self.image_size[1]:
            raise ValueError(
                f"Camera '{self.name}' expects image size {self.image_size}, "
                f"got {(image.shape[1], image.shape[0])}."
            )
        if self.dist.size == 0 or np.allclose(self.dist, 0.0):
            return image.copy()
        return cv2.undistort(image, self.K, self.dist, None, self.K)

    def crop_and_orient(self, image: np.ndarray) -> np.ndarray:
        x0, y0, width, height = self.roi
        cropped = image[y0:y0 + height, x0:x0 + width]
        transformed = np.rot90(cropped, k=self.rotate_k)
        if self.flip_horizontal:
            transformed = np.fliplr(transformed)
        if self.flip_vertical:
            transformed = np.flipud(transformed)
        return np.ascontiguousarray(transformed)

    def preprocess_geometry(self, image: np.ndarray) -> np.ndarray:
        return self.crop_and_orient(self.undistort(image))

    def _processed_to_crop_pixels(
        self, u_processed: np.ndarray, v_processed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map processed-image coordinates back to undistorted cropped image."""
        u = np.asarray(u_processed, dtype=np.float64).copy()
        v = np.asarray(v_processed, dtype=np.float64).copy()
        processed_width, processed_height = self.processed_size
        _, _, crop_width, crop_height = self.roi

        if self.flip_horizontal:
            u = (processed_width - 1.0) - u
        if self.flip_vertical:
            v = (processed_height - 1.0) - v

        k = self.rotate_k
        if k == 0:
            u_crop, v_crop = u, v
        elif k == 1:
            u_crop = (crop_width - 1.0) - v
            v_crop = u
        elif k == 2:
            u_crop = (crop_width - 1.0) - u
            v_crop = (crop_height - 1.0) - v
        else:  # k == 3
            u_crop = v
            v_crop = (crop_height - 1.0) - u
        return u_crop, v_crop

    def _crop_to_processed_pixels(
        self, u_crop: np.ndarray, v_crop: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map undistorted cropped-image coordinates to processed image."""
        u = np.asarray(u_crop, dtype=np.float64)
        v = np.asarray(v_crop, dtype=np.float64)
        _, _, crop_width, crop_height = self.roi

        k = self.rotate_k
        if k == 0:
            u_processed, v_processed = u.copy(), v.copy()
        elif k == 1:
            u_processed = v.copy()
            v_processed = (crop_width - 1.0) - u
        elif k == 2:
            u_processed = (crop_width - 1.0) - u
            v_processed = (crop_height - 1.0) - v
        else:  # k == 3
            u_processed = (crop_height - 1.0) - v
            v_processed = u.copy()

        processed_width, processed_height = self.processed_size
        if self.flip_horizontal:
            u_processed = (processed_width - 1.0) - u_processed
        if self.flip_vertical:
            v_processed = (processed_height - 1.0) - v_processed
        return u_processed, v_processed

    def measurement_rays(
        self,
        grid_shape: tuple[int, int],
        samples_per_bin: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return origins, normalized world directions and measurement row IDs.

        Multiple sub-rays approximate the finite area of each downsampled image bin.
        All sub-rays belonging to one bin share a row ID and are averaged later.
        """
        rows, cols = (int(grid_shape[0]), int(grid_shape[1]))
        samples = int(samples_per_bin)
        if rows <= 0 or cols <= 0 or samples <= 0:
            raise ValueError("grid_shape and samples_per_bin must be positive.")

        processed_width, processed_height = self.processed_size
        n_bins = rows * cols
        n_sub = samples * samples
        total = n_bins * n_sub
        u_processed = np.empty(total, dtype=np.float64)
        v_processed = np.empty(total, dtype=np.float64)
        row_ids = np.empty(total, dtype=np.int32)

        index = 0
        for row in range(rows):
            for col in range(cols):
                row_id = row * cols + col
                for sy in range(samples):
                    for sx in range(samples):
                        frac_x = (sx + 0.5) / samples
                        frac_y = (sy + 0.5) / samples
                        u_processed[index] = (
                            (col + frac_x) * processed_width / cols - 0.5
                        )
                        v_processed[index] = (
                            (row + frac_y) * processed_height / rows - 0.5
                        )
                        row_ids[index] = row_id
                        index += 1

        u_crop, v_crop = self._processed_to_crop_pixels(u_processed, v_processed)
        x0, y0, _, _ = self.roi
        pixels = np.vstack((u_crop + x0, v_crop + y0, np.ones(total)))
        directions_camera = np.linalg.solve(self.K, pixels).T
        directions_world = directions_camera @ self.R
        directions_world /= np.linalg.norm(directions_world, axis=1, keepdims=True)
        origins = np.repeat(self.center_world[None, :], total, axis=0)
        return origins, directions_world, row_ids

    def project_world(
        self, points_world: np.ndarray, processed: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project world points into undistorted raw or processed image coordinates."""
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        camera_points = points @ self.R.T + self.t
        depth = camera_points[:, 2]
        valid = depth > 1.0e-9

        u_raw = np.full(points.shape[0], np.nan, dtype=np.float64)
        v_raw = np.full(points.shape[0], np.nan, dtype=np.float64)
        normalized = camera_points[valid, :2] / depth[valid, None]
        u_raw[valid] = self.K[0, 0] * normalized[:, 0] + self.K[0, 2]
        v_raw[valid] = self.K[1, 1] * normalized[:, 1] + self.K[1, 2]

        if not processed:
            return np.column_stack((u_raw, v_raw)), valid

        x0, y0, crop_width, crop_height = self.roi
        u_crop = u_raw - x0
        v_crop = v_raw - y0
        in_crop = (
            valid
            & (u_crop >= -0.5)
            & (u_crop <= crop_width - 0.5)
            & (v_crop >= -0.5)
            & (v_crop <= crop_height - 0.5)
        )
        u_processed, v_processed = self._crop_to_processed_pixels(u_crop, v_crop)
        processed_width, processed_height = self.processed_size
        in_processed = (
            in_crop
            & (u_processed >= -0.5)
            & (u_processed <= processed_width - 0.5)
            & (v_processed >= -0.5)
            & (v_processed <= processed_height - 0.5)
        )
        return np.column_stack((u_processed, v_processed)), in_processed

    def project_world_to_grid(
        self, points_world: np.ndarray, grid_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        pixels, valid = self.project_world(points_world, processed=True)
        rows, cols = grid_shape
        processed_width, processed_height = self.processed_size
        grid_col = (pixels[:, 0] + 0.5) * cols / processed_width - 0.5
        grid_row = (pixels[:, 1] + 0.5) * rows / processed_height - 0.5
        valid &= (
            (grid_col >= -0.5)
            & (grid_col <= cols - 0.5)
            & (grid_row >= -0.5)
            & (grid_row <= rows - 0.5)
        )
        return np.column_stack((grid_row, grid_col)), valid

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "K": self.K.round(12).tolist(),
            "dist": self.dist.round(12).tolist(),
            "R": self.R.round(12).tolist(),
            "t": self.t.round(12).tolist(),
            "image_size": list(self.image_size),
            "roi": list(self.roi),
            "rotate_k": self.rotate_k,
            "flip_horizontal": self.flip_horizontal,
            "flip_vertical": self.flip_vertical,
        }


def load_cameras(cfg: dict[str, Any]) -> list[CameraModel]:
    return [CameraModel.from_config(camera_cfg) for camera_cfg in cfg["cameras"]]
