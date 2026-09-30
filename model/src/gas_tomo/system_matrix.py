from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from .camera import CameraModel
from .grid import VolumeGrid
from .raytrace import trace_rays_to_csr
from .sampling import shape_for, layout_slices


MATRIX_FORMAT_VERSION = 2


@dataclass
class SystemMatrixBundle:
    matrix: sparse.csr_matrix
    matrix_id: str
    cache_path: Path
    camera_slices: dict[str, slice]
    metadata: dict[str, Any]


def _matrix_payload(
    cameras: list[CameraModel],
    grid: VolumeGrid,
    measurement_shape: tuple[int, int],
    samples_per_bin: int,
) -> dict[str, Any]:
    return {
        "format_version": MATRIX_FORMAT_VERSION,
        "grid_bounds": grid.bounds.round(12).tolist(),
        "grid_shape_nx_ny_nz": list(grid.shape),
        "measurement_shapes_rows_cols": {c.name: list(shape_for(measurement_shape,c.name)) for c in cameras},
        "samples_per_bin": int(samples_per_bin),
        "cameras": [camera.metadata() for camera in cameras],
    }


def _matrix_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()[:16]


def build_or_load_system_matrix(
    cameras: list[CameraModel],
    grid: VolumeGrid,
    measurement_shape: tuple[int, int],
    samples_per_bin: int,
    cache_dir: str | Path,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> SystemMatrixBundle:
    if len(cameras) < 2:
        raise ValueError("At least two cameras are required.")
    if int(samples_per_bin) != samples_per_bin or samples_per_bin < 1:
        raise ValueError("samples_per_bin must be a positive integer.")
    shapes = {c.name: shape_for(measurement_shape, c.name) for c in cameras}
    payload = _matrix_payload(cameras, grid, measurement_shape, samples_per_bin)
    matrix_id = _matrix_id(payload)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shape_text = f"{grid.nx}x{grid.ny}x{grid.nz}"
    cache_path = cache_dir / f"A_{shape_text}_{matrix_id}.npz"
    metadata_path = cache_path.with_suffix(".json")

    camera_slices = layout_slices([c.name for c in cameras], shapes)
    n_measurements = sum(r*c for r,c in shapes.values())

    if cache_path.exists() and metadata_path.exists() and not force_rebuild:
        matrix = sparse.load_npz(cache_path).tocsr().astype(np.float32)
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        expected_shape = (n_measurements, grid.n_voxels)
        if matrix.shape != expected_shape:
            raise RuntimeError(
                f"Cached matrix has shape {matrix.shape}; expected {expected_shape}. "
                "Delete the cache file or use --force."
            )
        if verbose:
            print(
                f"[matrix] loaded {cache_path.name}: shape={matrix.shape}, "
                f"nnz={matrix.nnz:,}"
            )
        return SystemMatrixBundle(
            matrix=matrix,
            matrix_id=matrix_id,
            cache_path=cache_path,
            camera_slices=camera_slices,
            metadata=metadata,
        )

    start_time = perf_counter()
    camera_matrices: list[sparse.csr_matrix] = []
    for camera_index, camera in enumerate(cameras, start=1):
        if verbose:
            print(
                f"[matrix] tracing camera {camera_index}/{len(cameras)} '{camera.name}' "
                f"with {samples_per_bin}x{samples_per_bin} sub-rays per bin..."
            )
        camera_shape = shapes[camera.name]
        measurements_per_camera = camera_shape[0]*camera_shape[1]
        origins, directions, row_ids = camera.measurement_rays(
            camera_shape, samples_per_bin=samples_per_bin
        )
        camera_matrix = trace_rays_to_csr(
            origins=origins,
            directions=directions,
            row_ids=row_ids,
            n_measurements=measurements_per_camera,
            grid=grid,
        )
        camera_matrices.append(camera_matrix)

    matrix = sparse.vstack(camera_matrices, format="csr", dtype=np.float32)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    row_lengths = np.asarray(matrix.sum(axis=1)).ravel()
    valid_rows = row_lengths > 1.0e-9
    elapsed = perf_counter() - start_time
    metadata = {
        **payload,
        "matrix_id": matrix_id,
        "shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
        "valid_rows": int(np.count_nonzero(valid_rows)),
        "invalid_rows": int(valid_rows.size - np.count_nonzero(valid_rows)),
        "mean_path_length_valid_m": float(row_lengths[valid_rows].mean())
        if np.any(valid_rows)
        else 0.0,
        "max_path_length_m": float(row_lengths.max(initial=0.0)),
        "build_seconds": float(elapsed),
    }
    sparse.save_npz(cache_path, matrix, compressed=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    if verbose:
        dense_bytes = matrix.shape[0] * matrix.shape[1] * 4
        sparse_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        print(
            f"[matrix] built shape={matrix.shape}, nnz={matrix.nnz:,}, "
            f"time={elapsed:.2f}s, CSR={sparse_bytes / 2**20:.2f} MiB, "
            f"dense-float32-equivalent={dense_bytes / 2**30:.2f} GiB"
        )
        print(f"[matrix] cached at {cache_path}")

    return SystemMatrixBundle(
        matrix=matrix,
        matrix_id=matrix_id,
        cache_path=cache_path,
        camera_slices=camera_slices,
        metadata=metadata,
    )
