from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class VolumeGrid:
    """
    Cartesian cell-centered grid.

    Public shape is (nx, ny, nz). Volume arrays use (nz, ny, nx), so C-order
    flattening keeps x as the fastest-changing index:
        flat_index = ix + nx * (iy + ny * iz)
    """

    bounds: np.ndarray  # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    shape: tuple[int, int, int]  # (nx, ny, nz)

    def __post_init__(self) -> None:
        bounds = np.asarray(self.bounds, dtype=np.float64)
        shape = tuple(int(v) for v in self.shape)
        if not np.isfinite(bounds).all():
            raise ValueError("Grid bounds must be finite.")
        if bounds.shape != (3, 2):
            raise ValueError("bounds must have shape (3, 2).")
        if np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("Every upper bound must exceed the lower bound.")
        if len(shape) != 3 or any(v <= 1 for v in shape):
            raise ValueError("shape must be (nx, ny, nz), all greater than 1.")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "shape", shape)

    @classmethod
    def from_config(cls, volume_cfg: dict, shape: Iterable[int]) -> "VolumeGrid":
        b = volume_cfg["bounds_m"]
        bounds = np.array([b["x"], b["y"], b["z"]], dtype=np.float64)
        return cls(bounds=bounds, shape=tuple(int(v) for v in shape))

    @property
    def nx(self) -> int:
        return self.shape[0]

    @property
    def ny(self) -> int:
        return self.shape[1]

    @property
    def nz(self) -> int:
        return self.shape[2]

    @property
    def array_shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def n_voxels(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def spacing(self) -> np.ndarray:
        return (self.bounds[:, 1] - self.bounds[:, 0]) / np.array(self.shape)

    @property
    def voxel_volume(self) -> float:
        return float(np.prod(self.spacing))

    def centers_1d(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.spacing
        x = self.bounds[0, 0] + (np.arange(self.nx) + 0.5) * d[0]
        y = self.bounds[1, 0] + (np.arange(self.ny) + 0.5) * d[1]
        z = self.bounds[2, 0] + (np.arange(self.nz) + 0.5) * d[2]
        return x, y, z

    def centers(self) -> np.ndarray:
        x, y, z = self.centers_1d()
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    def as_volume(self, flat: np.ndarray) -> np.ndarray:
        flat = np.asarray(flat)
        if flat.size != self.n_voxels:
            raise ValueError(
                f"Expected {self.n_voxels} values, got {flat.size}."
            )
        return flat.reshape(self.array_shape)

    def as_flat(self, volume: np.ndarray) -> np.ndarray:
        volume = np.asarray(volume)
        if volume.shape != self.array_shape:
            raise ValueError(
                f"Expected volume shape {self.array_shape}, got {volume.shape}."
            )
        return volume.ravel()

    def index(self, ix: int, iy: int, iz: int) -> int:
        return int(ix + self.nx * (iy + self.ny * iz))

    def max_location(self, volume: np.ndarray) -> tuple[float, float, float]:
        iz, iy, ix = np.unravel_index(np.argmax(volume), self.array_shape)
        x, y, z = self.centers_1d()
        return float(x[ix]), float(y[iy]), float(z[iz])
