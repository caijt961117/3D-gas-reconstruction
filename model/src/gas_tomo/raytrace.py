from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy import sparse

from .grid import VolumeGrid


@njit(cache=True, parallel=True)
def _trace_rays_to_buffers(
    origins: np.ndarray,
    directions: np.ndarray,
    bounds: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Siddon-style exact path lengths through a regular Cartesian grid."""
    n_rays = origins.shape[0]
    max_segments = nx + ny + nz + 3
    indices = np.full((n_rays, max_segments), -1, dtype=np.int32)
    lengths = np.zeros((n_rays, max_segments), dtype=np.float32)
    counts = np.zeros(n_rays, dtype=np.int32)

    xmin, xmax = bounds[0, 0], bounds[0, 1]
    ymin, ymax = bounds[1, 0], bounds[1, 1]
    zmin, zmax = bounds[2, 0], bounds[2, 1]
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz
    eps = 1.0e-12
    t_tol = 1.0e-10

    for ray in prange(n_rays):
        ox, oy, oz = origins[ray, 0], origins[ray, 1], origins[ray, 2]
        vx, vy, vz = directions[ray, 0], directions[ray, 1], directions[ray, 2]
        norm = np.sqrt(vx * vx + vy * vy + vz * vz)
        if norm < eps:
            continue
        vx, vy, vz = vx / norm, vy / norm, vz / norm

        t_enter = -1.0e30
        t_exit = 1.0e30
        valid = True

        # X slab.
        if abs(vx) < eps:
            if ox < xmin or ox > xmax:
                valid = False
        else:
            tx0 = (xmin - ox) / vx
            tx1 = (xmax - ox) / vx
            if tx0 > tx1:
                tx0, tx1 = tx1, tx0
            if tx0 > t_enter:
                t_enter = tx0
            if tx1 < t_exit:
                t_exit = tx1

        # Y slab.
        if abs(vy) < eps:
            if oy < ymin or oy > ymax:
                valid = False
        else:
            ty0 = (ymin - oy) / vy
            ty1 = (ymax - oy) / vy
            if ty0 > ty1:
                ty0, ty1 = ty1, ty0
            if ty0 > t_enter:
                t_enter = ty0
            if ty1 < t_exit:
                t_exit = ty1

        # Z slab.
        if abs(vz) < eps:
            if oz < zmin or oz > zmax:
                valid = False
        else:
            tz0 = (zmin - oz) / vz
            tz1 = (zmax - oz) / vz
            if tz0 > tz1:
                tz0, tz1 = tz1, tz0
            if tz0 > t_enter:
                t_enter = tz0
            if tz1 < t_exit:
                t_exit = tz1

        if not valid:
            continue
        if t_enter < 0.0:
            t_enter = 0.0
        if t_exit <= t_enter + t_tol:
            continue

        t_values = np.empty(max_segments + 2, dtype=np.float64)
        n_t = 0
        t_values[n_t] = t_enter
        n_t += 1
        t_values[n_t] = t_exit
        n_t += 1

        if abs(vx) >= eps:
            for ix_plane in range(1, nx):
                plane = xmin + ix_plane * dx
                value = (plane - ox) / vx
                if value > t_enter + t_tol and value < t_exit - t_tol:
                    t_values[n_t] = value
                    n_t += 1
        if abs(vy) >= eps:
            for iy_plane in range(1, ny):
                plane = ymin + iy_plane * dy
                value = (plane - oy) / vy
                if value > t_enter + t_tol and value < t_exit - t_tol:
                    t_values[n_t] = value
                    n_t += 1
        if abs(vz) >= eps:
            for iz_plane in range(1, nz):
                plane = zmin + iz_plane * dz
                value = (plane - oz) / vz
                if value > t_enter + t_tol and value < t_exit - t_tol:
                    t_values[n_t] = value
                    n_t += 1

        sorted_t = np.sort(t_values[:n_t])
        count = 0
        previous_index = -1
        for segment in range(n_t - 1):
            t0 = sorted_t[segment]
            t1 = sorted_t[segment + 1]
            length = t1 - t0
            if length <= t_tol:
                continue
            middle = 0.5 * (t0 + t1)
            px = ox + middle * vx
            py = oy + middle * vy
            pz = oz + middle * vz
            ix = int(np.floor((px - xmin) / dx))
            iy = int(np.floor((py - ymin) / dy))
            iz = int(np.floor((pz - zmin) / dz))
            if ix < 0:
                ix = 0
            elif ix >= nx:
                ix = nx - 1
            if iy < 0:
                iy = 0
            elif iy >= ny:
                iy = ny - 1
            if iz < 0:
                iz = 0
            elif iz >= nz:
                iz = nz - 1

            flat_index = ix + nx * (iy + ny * iz)
            if flat_index == previous_index and count > 0:
                lengths[ray, count - 1] += np.float32(length)
            else:
                indices[ray, count] = flat_index
                lengths[ray, count] = np.float32(length)
                previous_index = flat_index
                count += 1
        counts[ray] = count

    return indices, lengths, counts


def trace_rays_to_csr(
    origins: np.ndarray,
    directions: np.ndarray,
    row_ids: np.ndarray,
    n_measurements: int,
    grid: VolumeGrid,
) -> sparse.csr_matrix:
    """
    Trace sub-rays and average all sub-rays belonging to each measurement bin.
    """
    origins = np.ascontiguousarray(origins, dtype=np.float64)
    directions = np.ascontiguousarray(directions, dtype=np.float64)
    row_ids = np.asarray(row_ids, dtype=np.int32).reshape(-1)
    if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("origins and directions must both have shape (n, 3).")
    if row_ids.size != origins.shape[0]:
        raise ValueError("row_ids must contain one value per ray.")
    if np.any(row_ids < 0) or np.any(row_ids >= n_measurements):
        raise ValueError("row_ids contain out-of-range measurement indices.")

    indices_buffer, lengths_buffer, counts = _trace_rays_to_buffers(
        origins, directions, grid.bounds, grid.nx, grid.ny, grid.nz
    )
    samples_per_row = np.bincount(row_ids, minlength=n_measurements).astype(np.float32)
    if np.any(samples_per_row == 0):
        raise ValueError("Every measurement row must have at least one sub-ray.")

    total_nonzero = int(np.sum(counts))
    rows = np.empty(total_nonzero, dtype=np.int32)
    cols = np.empty(total_nonzero, dtype=np.int32)
    data = np.empty(total_nonzero, dtype=np.float32)
    cursor = 0
    for ray in range(origins.shape[0]):
        count = int(counts[ray])
        if count == 0:
            continue
        next_cursor = cursor + count
        row_id = int(row_ids[ray])
        rows[cursor:next_cursor] = row_id
        cols[cursor:next_cursor] = indices_buffer[ray, :count]
        data[cursor:next_cursor] = (
            lengths_buffer[ray, :count] / samples_per_row[row_id]
        )
        cursor = next_cursor

    matrix = sparse.coo_matrix(
        (data[:cursor], (rows[:cursor], cols[:cursor])),
        shape=(n_measurements, grid.n_voxels),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix
