from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from gas_tomo.grid import VolumeGrid
from gas_tomo.raytrace import trace_rays_to_csr
from gas_tomo.solvers import solve_fista_huber_tv


INPUT_DIR = Path(r"D:\Desktop\gh\data")
RESULT_ROOT = Path(r"D:\Desktop\gh\result")

GRID_SHAPE = (41, 41, 41)
GRID_BOUNDS = np.array([[-20.5, 20.5], [-20.5, 20.5], [-20.5, 20.5]], dtype=np.float64)
GRID_COORDS = np.arange(-20.0, 21.0, 1.0, dtype=np.float64)
PROJECTION_SIZE = 160
NUM_REPROJECT_SAMPLES = 300
COORD_SHIFT = np.array([20.0, 20.0, 20.0], dtype=np.float64)
DOMAIN_HALF_WIDTH = 20.0
PLANE_BACKOFF = 100.0

BASE_SOLVER_CONFIG = {
    "huber_delta": 0.06,
    "lambda_temporal": 0.0,
    "lambda_support": 0.0,
    "tolerance": 2.0e-5,
    "kkt_tolerance": 2.0e-4,
    "step_safety": 0.92,
    "power_iterations": 20,
    "restart": True,
    "backtracking": True,
    "max_backtracking": 25,
}

THREE_VIEW_MODEL = {
    "name": "three_view_latest",
    "iterations": 800,
    "solver_config": {
        **BASE_SOLVER_CONFIG,
        "lambda_tv": 0.0,
        "lambda_l2": 0.0,
        "log_every": 50,
    },
}

TWO_VIEW_MODEL = {
    "name": "two_view_previous",
    "iterations": 300,
    "solver_config": {
        **BASE_SOLVER_CONFIG,
        "lambda_tv": 4.0e-6,
        "lambda_l2": 1.0e-6,
        "log_every": 25,
    },
}

THREE_VIEW_SCHEMES = {
    "scheme_1": [(0, 20, 40), (20, 0, 30), (40, 20, 30)],
    "scheme_2": [(10, 20, 40), (20, 0, 30), (40, 20, 30)],
    "scheme_3": [(20, 20, 40), (20, 0, 30), (40, 20, 30)],
    "scheme_4": [(30, 20, 40), (20, 0, 30), (40, 20, 30)],
    "scheme_5": [(40, 20, 40), (20, 0, 30), (40, 20, 30)],
}

FILENAME_RE = re.compile(r"view_(\d+)_user_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.mat$")


def mat_for_point(point: tuple[int, int, int]) -> tuple[str, Path]:
    matches: list[tuple[str, Path]] = []
    for mat_path in INPUT_DIR.glob("view_*_user_*.mat"):
        match = FILENAME_RE.match(mat_path.name)
        if not match:
            continue
        coords = tuple(int(float(match.group(i))) for i in (2, 3, 4))
        if coords == point:
            matches.append((f"view_{int(match.group(1)):02d}", mat_path))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one MAT file for point {point}, found {len(matches)}")
    return matches[0]


def make_projection_basis(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([0.0, 0.0, 1.0]) if abs(float(np.dot(w, [0, 0, 1]))) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(w, ref)
    u /= np.linalg.norm(u)
    v = np.cross(u, w)
    v /= np.linalg.norm(v)
    return u, v


def geometry_for_point(point: tuple[int, int, int]) -> dict:
    view_user = np.array(point, dtype=np.float64)
    eye_csv = view_user - COORD_SHIFT
    direction = -eye_csv
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError(f"View point {point} is at projection center.")
    w = direction / norm
    u, v = make_projection_basis(w)
    return {
        "point_user": list(point),
        "eye_csv": eye_csv,
        "central_direction": w,
        "u": u,
        "v": v,
        "axis_values": np.linspace(-DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH, PROJECTION_SIZE, dtype=np.float64),
    }


def mat_projection(path: Path) -> np.ndarray:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    projection = np.asarray(data["projection"], dtype=np.float32)
    if projection.shape != (PROJECTION_SIZE, PROJECTION_SIZE):
        raise ValueError(f"{path} has projection shape {projection.shape}")
    return projection


def parallel_rays(geometry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = geometry["axis_values"]
    rows = cols = axes.size
    origins = np.empty((rows * cols, 3), dtype=np.float64)
    directions = np.empty_like(origins)
    row_ids = np.arange(rows * cols, dtype=np.int32)
    cursor = 0
    for r in range(rows):
        for c in range(cols):
            plane_point = geometry["u"] * axes[c] + geometry["v"] * axes[r]
            origins[cursor] = plane_point - geometry["central_direction"] * PLANE_BACKOFF
            directions[cursor] = geometry["central_direction"]
            cursor += 1
    return origins, directions, row_ids


def reproject_parallel_like_matlab(volume_zyx: np.ndarray, point: tuple[int, int, int]) -> np.ndarray:
    volume_xyz = np.transpose(volume_zyx.astype(np.float64), (2, 1, 0))
    interpolator = RegularGridInterpolator(
        (GRID_COORDS, GRID_COORDS, GRID_COORDS),
        volume_xyz,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    geometry = geometry_for_point(point)
    axes = geometry["axis_values"]
    s, t = np.meshgrid(axes, axes)
    x0 = s * geometry["u"][0] + t * geometry["v"][0]
    y0 = s * geometry["u"][1] + t * geometry["v"][1]
    z0 = s * geometry["u"][2] + t * geometry["v"][2]
    lambdas = np.linspace(-np.sqrt(3.0) * DOMAIN_HALF_WIDTH, np.sqrt(3.0) * DOMAIN_HALF_WIDTH, NUM_REPROJECT_SAMPLES)
    dlambda = lambdas[1] - lambdas[0]
    projected = np.zeros((PROJECTION_SIZE, PROJECTION_SIZE), dtype=np.float64)
    w = geometry["central_direction"]
    for i, lam in enumerate(lambdas):
        x = x0 + lam * w[0]
        y = y0 + lam * w[1]
        z = z0 + lam * w[2]
        values = interpolator(np.column_stack((x.ravel(), y.ravel(), z.ravel()))).reshape(PROJECTION_SIZE, PROJECTION_SIZE)
        projected += (0.5 if i == 0 or i == len(lambdas) - 1 else 1.0) * values * dlambda
    return projected.astype(np.float32)


def e_field_percent(projected: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm((projected - reference).ravel()) / np.linalg.norm(reference.ravel()) * 100.0)


def run_pipeline(
    schemes: dict[str, list[tuple[int, int, int]]],
    output_dir: Path,
    summary_name: str,
    model: dict,
    selection: str = "min",
) -> pd.DataFrame:
    if selection not in {"min", "max", "mean"}:
        raise ValueError("selection must be 'min', 'max', or 'mean'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = VolumeGrid(bounds=GRID_BOUNDS, shape=GRID_SHAPE)
    rows = []

    for scheme, points in schemes.items():
        scheme_dir = output_dir / scheme
        scheme_dir.mkdir(parents=True, exist_ok=True)
        matrices = []
        observations = []
        views_metadata = []

        for point in points:
            view_name, mat_path = mat_for_point(point)
            geometry = geometry_for_point(point)
            projection = mat_projection(mat_path)
            origins, directions, row_ids = parallel_rays(geometry)
            matrices.append(trace_rays_to_csr(origins, directions, row_ids, row_ids.size, grid))
            observations.append(projection.ravel())
            views_metadata.append({"view_name": view_name, "point_user": list(point), "mat": str(mat_path)})

        system_matrix = sparse.vstack(matrices, format="csr", dtype=np.float32)
        observed = np.concatenate(observations).astype(np.float32)
        weights = np.ones_like(observed, dtype=np.float32)
        result = solve_fista_huber_tv(
            system_matrix,
            observed,
            weights,
            grid,
            model["solver_config"],
            iterations=int(model["iterations"]),
            verbose=False,
        )
        reconstruction = grid.as_volume(result.x).astype(np.float32)

        np.save(scheme_dir / "concentration_3d.npy", reconstruction)
        np.savez_compressed(
            scheme_dir / "reconstruction_bundle.npz",
            concentration=reconstruction,
            observed=observed,
            predicted=(system_matrix @ result.x).astype(np.float32),
            bounds_m=grid.bounds,
            shape_nx_ny_nz=np.array(grid.shape, dtype=np.int32),
        )

        view_rows = []
        for view in views_metadata:
            point = tuple(view["point_user"])
            reference = mat_projection(Path(view["mat"]))
            projected = reproject_parallel_like_matlab(reconstruction, point)
            value = e_field_percent(projected, reference)
            np.save(scheme_dir / f"{view['view_name']}_reprojected.npy", projected)
            view_rows.append({**view, "E_field_percent": value})

        (scheme_dir / "scheme_metadata.json").write_text(
            json.dumps({"views": views_metadata, "model": model, "geometry": "parallel"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if selection == "mean":
            value = float(np.mean([item["E_field_percent"] for item in view_rows]))
        else:
            selected = (min if selection == "min" else max)(view_rows, key=lambda item: item["E_field_percent"])
            value = selected["E_field_percent"]
        rows.append({"scheme": scheme, "E_field": value})

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / summary_name, index=False)
    return summary


def print_min_only(summary: pd.DataFrame) -> None:
    print("scheme,E_field")
    for row in summary.itertuples(index=False):
        print(f"{row.scheme},{row.E_field:.4f}%")
