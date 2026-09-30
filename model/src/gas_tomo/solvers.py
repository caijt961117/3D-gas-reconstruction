from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from .grid import VolumeGrid
from .regularization import huber_tv_lipschitz_bound, huber_tv_value_gradient


@dataclass
class SolverHistoryEntry:
    iteration: int
    objective: float
    data_term: float
    tv_term: float
    temporal_term: float
    support_term: float
    l2_term: float
    relative_change: float
    relative_residual: float | None
    elapsed_seconds: float


@dataclass
class SolverResult:
    x: np.ndarray
    history: list[SolverHistoryEntry] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    lipschitz: float = 0.0
    status: str = "max_iterations"
    termination_reason: str = "iteration_budget"
    has_estimate: bool = True
    initial_norm: float = 0.0
    projected_gradient_relative: float | None = None
    backtracking_steps: int = 0


def estimate_data_lipschitz(
    matrix: sparse.csr_matrix,
    weights: np.ndarray,
    iterations: int = 25,
    seed: int = 0,
) -> float:
    """Power iteration for ||sqrt(W) A||_2^2 / sum(W)."""
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    weight_sum = float(np.sum(weights, dtype=np.float64))
    if weight_sum <= 1.0e-12 or matrix.nnz == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(matrix.shape[1]).astype(np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1.0e-12)
    eigenvalue = 0.0
    for _ in range(max(2, int(iterations))):
        projected = matrix @ vector
        next_vector = matrix.T @ (weights * projected)
        next_vector = np.asarray(next_vector, dtype=np.float32) / np.float32(weight_sum)
        norm = float(np.linalg.norm(next_vector))
        if norm <= 1.0e-20:
            return 0.0
        vector = next_vector / np.float32(norm)
        eigenvalue = norm
    projected = matrix @ vector
    eigenvalue = float(
        np.dot(weights * projected, projected) / max(weight_sum, 1.0e-12)
    )
    return max(eigenvalue, 0.0)


def _objective_and_gradient(
    x: np.ndarray,
    matrix: sparse.csr_matrix,
    b: np.ndarray,
    weights: np.ndarray,
    grid: VolumeGrid,
    lambda_tv: float,
    huber_delta: float,
    lambda_temporal: float,
    previous: np.ndarray | None,
    lambda_support: float,
    support: np.ndarray | None,
    lambda_l2: float,
) -> tuple[float, np.ndarray, dict[str, float], np.ndarray]:
    weight_sum = max(float(np.sum(weights, dtype=np.float64)), 1.0e-12)
    residual = matrix @ x - b
    weighted_residual = weights * residual
    data_term = 0.5 * float(np.dot(weighted_residual, residual)) / weight_sum
    gradient = np.asarray(matrix.T @ weighted_residual, dtype=np.float32) / np.float32(
        weight_sum
    )

    tv_value = 0.0
    if lambda_tv > 0.0:
        raw_tv, tv_gradient = huber_tv_value_gradient(x, grid, huber_delta)
        tv_value = lambda_tv * raw_tv
        gradient += np.float32(lambda_tv) * tv_gradient

    voxel_volume = np.float32(grid.voxel_volume)
    temporal_term = 0.0
    if lambda_temporal > 0.0 and previous is not None:
        difference = x - previous
        temporal_term = (
            0.5
            * lambda_temporal
            * float(voxel_volume)
            * float(np.dot(difference, difference))
        )
        gradient += np.float32(lambda_temporal) * voxel_volume * difference

    support_term = 0.0
    if lambda_support > 0.0 and support is not None:
        outside = 1.0 - support
        penalized = outside * x
        support_term = (
            0.5
            * lambda_support
            * float(voxel_volume)
            * float(np.dot(penalized, penalized))
        )
        gradient += (
            np.float32(lambda_support)
            * voxel_volume
            * outside
            * outside
            * x
        )

    l2_term = 0.0
    if lambda_l2 > 0.0:
        l2_term = (
            0.5
            * lambda_l2
            * float(voxel_volume)
            * float(np.dot(x, x))
        )
        gradient += np.float32(lambda_l2) * voxel_volume * x

    components = {
        "data": data_term,
        "tv": tv_value,
        "temporal": temporal_term,
        "support": support_term,
        "l2": l2_term,
    }
    objective = sum(components.values())
    return objective, gradient, components, residual


def _validate_inputs(matrix, b, weights, grid, x0, previous, support):
    matrix = matrix.tocsr().astype(np.float32)
    b = np.asarray(b, dtype=np.float32).reshape(-1).copy()
    weights = np.asarray(weights, dtype=np.float32).reshape(-1).copy()
    if matrix.shape != (b.size, grid.n_voxels) or weights.size != b.size:
        raise ValueError("Matrix, observation, weights and grid dimensions are incompatible.")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("Optical path matrix must have finite nonnegative values.")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Observation weights must be finite and nonnegative.")
    invalid = ~np.isfinite(b)
    weights[invalid] = 0
    b[invalid] = 0
    # Rays missing the volume are not observations of its unknown concentration.
    weights[np.asarray(matrix.sum(axis=1)).ravel() <= 0] = 0
    def flat(value, name):
        if value is None:
            return None
        value = np.asarray(value, dtype=np.float32).reshape(-1).copy()
        if value.size != grid.n_voxels or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain one finite value per voxel.")
        return value
    x0, previous, support = flat(x0, "x0"), flat(previous, "previous"), flat(support, "support")
    x = np.zeros(grid.n_voxels, dtype=np.float32) if x0 is None else np.maximum(x0, 0)
    if previous is not None:
        previous = np.maximum(previous, 0)
    if support is not None:
        support = np.clip(support, 0, 1)
    return matrix, b, weights, x, previous, support


def _no_data_result(x, previous):
    # Explicit hold-last prediction policy: no false claim of measured concentration.
    if previous is not None:
        return SolverResult(x=previous.copy(), status="prior_only_prediction",
                            termination_reason="no_valid_observations_hold_previous",
                            converged=False, has_estimate=True, initial_norm=float(np.linalg.norm(x)))
    return SolverResult(x=np.zeros_like(x), status="insufficient_valid_data",
                        termination_reason="no_valid_observations_no_estimate",
                        converged=False, has_estimate=False, initial_norm=float(np.linalg.norm(x)))


def solve_fista_huber_tv(
    matrix: sparse.csr_matrix, b: np.ndarray, weights: np.ndarray,
    grid: VolumeGrid, config: dict[str, Any], iterations: int,
    x0: np.ndarray | None = None, previous: np.ndarray | None = None,
    support: np.ndarray | None = None, verbose: bool = True,
) -> SolverResult:
    """Projected FISTA with Huber-TV, monotone backtracking and KKT checking.

    W contains precision-like weights: data term is sum(W*r^2)/(2 sum W).
    Measurement values may be signed; only x is constrained nonnegative.
    No-data frames use the documented hold-last policy, not a fitted zero field.
    """
    matrix, b, weights, x, previous, support = _validate_inputs(
        matrix, b, weights, grid, x0, previous, support)
    if int(iterations) < 1:
        raise ValueError("iterations must be >=1.")
    initial_norm = float(np.linalg.norm(x))
    if not np.any(weights > 0):
        return _no_data_result(x, previous)
    zero_signal = not np.any(b[weights > 0] != 0)
    lambda_tv = float(config.get("lambda_tv", 0.0))
    huber_delta = float(config.get("huber_delta", 0.5))
    lambda_temporal = float(config.get("lambda_temporal", 0.0))
    lambda_support = float(config.get("lambda_support", 0.0))
    lambda_l2 = float(config.get("lambda_l2", 0.0))
    if min(lambda_tv, lambda_temporal, lambda_support, lambda_l2) < 0 or huber_delta <= 0:
        raise ValueError("Regularization coefficients must be nonnegative and huber_delta positive.")
    tolerance = float(config.get("tolerance", 2e-5))
    kkt_tolerance = float(config.get("kkt_tolerance", 2e-4))
    step_safety = float(config.get("step_safety", 0.92))
    if not 0 < step_safety <= 1:
        raise ValueError("step_safety must be in (0,1].")
    log_every = max(1, int(config.get("log_every", 10)))
    data_lipschitz = estimate_data_lipschitz(
        matrix, weights, int(config.get("power_iterations", 25)), int(config.get("random_seed", 0)))
    lipschitz = data_lipschitz
    if lambda_tv > 0:
        lipschitz += lambda_tv * huber_tv_lipschitz_bound(grid, huber_delta)
    if previous is not None:
        lipschitz += lambda_temporal * grid.voxel_volume
    if support is not None:
        lipschitz += lambda_support * grid.voxel_volume
    lipschitz += lambda_l2 * grid.voxel_volume
    step = step_safety / max(lipschitz, 1e-20)

    def evaluate(v):
        return _objective_and_gradient(v, matrix, b, weights, grid, lambda_tv,
            huber_delta, lambda_temporal, previous, lambda_support, support, lambda_l2)
    # Reference gradient at zero transforms consistently under numerical rescaling.
    _, g0, _, _ = evaluate(np.zeros_like(x))
    grad_reference = max(float(np.linalg.norm(g0)), np.finfo(np.float32).tiny)
    b_norm = float(np.linalg.norm(np.sqrt(weights) * b))
    y, momentum, history = x.copy(), 1.0, []
    stable, converged, bt_total, kkt = 0, False, 0, None
    start = perf_counter()
    if verbose:
        print(f"[solver] FISTA-Huber-TV: {grid.n_voxels:,} voxels; "
              f"iterations={iterations}; initial_norm={initial_norm:.4g}; L={lipschitz:.4e}")

    for iteration in range(1, int(iterations) + 1):
        fy, gradient, _, _ = evaluate(y)
        for bt in range(int(config.get("max_backtracking", 25)) + 1):
            x_new = np.maximum(y - np.float32(step) * gradient, 0).astype(np.float32)
            fx, g_new, components, residual = evaluate(x_new)
            diff = (x_new - y).astype(np.float64)
            upper = fy + float(np.dot(gradient, diff)) + float(np.dot(diff, diff)) / (2 * step)
            allowance = 2e-6 * max(abs(fy), abs(fx), abs(upper), np.finfo(np.float32).tiny)
            if fx <= upper + allowance or not bool(config.get("backtracking", True)):
                break
            step *= 0.5
            bt_total += 1
        else:
            raise RuntimeError("Backtracking failed; verify units and regularization parameters.")
        dx = float(np.linalg.norm(x_new-x)) / max(float(np.linalg.norm(x)), float(np.linalg.norm(x_new)), 1e-30)
        pg = np.where((x_new > 0) | (g_new < 0), g_new, 0)
        kkt = float(np.linalg.norm(pg)) / grad_reference
        stable = stable+1 if dx < tolerance and kkt < kkt_tolerance else 0
        converged = stable >= 5
        next_m = 0.5*(1 + np.sqrt(1+4*momentum*momentum))
        y_new = x_new + np.float32((momentum-1)/next_m)*(x_new-x)
        if bool(config.get("restart", True)) and float(np.dot(y-x_new, x_new-x)) > 0:
            next_m, y_new = 1.0, x_new.copy()
        x, y, momentum = x_new, y_new, next_m
        if iteration == 1 or iteration % log_every == 0 or iteration == iterations or converged:
            rr = float(np.linalg.norm(np.sqrt(weights)*residual))/b_norm if b_norm > 0 else None
            history.append(SolverHistoryEntry(iteration, fx, components["data"], components["tv"],
                components["temporal"], components["support"], components["l2"], dx, rr, perf_counter()-start))
            if verbose:
                rr_text = "N/A (zero signal)" if rr is None else f"{rr:.5f}"
                print(f"[solver] iter={iteration:4d} obj={fx:.6e} res={rr_text} dx={dx:.3e} kkt={kkt:.3e}")
        if converged:
            break
    status = ("valid_zero_signal" if zero_signal else "converged") if converged else "max_iterations"
    return SolverResult(x=x, history=history, iterations=iteration, converged=converged,
        lipschitz=float(step_safety/step), status=status,
        termination_reason="relative_change_and_projected_gradient" if converged else "iteration_budget",
        initial_norm=initial_norm, projected_gradient_relative=kkt, backtracking_steps=bt_total)


def solve_sart(
    matrix: sparse.csr_matrix,
    b: np.ndarray,
    weights: np.ndarray,
    grid: VolumeGrid,
    config: dict[str, Any],
    iterations: int,
    x0: np.ndarray | None = None,
    verbose: bool = True,
    **_: Any,
) -> SolverResult:
    """Weighted SIRT baseline (legacy function name solve_sart retained)."""
    matrix = matrix.tocsr().astype(np.float32)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    x = (
        np.zeros(grid.n_voxels, dtype=np.float32)
        if x0 is None
        else np.maximum(np.asarray(x0, dtype=np.float32).reshape(-1), 0.0)
    )
    matrix, b, weights, x, previous, _support = _validate_inputs(matrix, b, weights, grid, x, _.get("previous"), None)
    if iterations < 1:
        raise ValueError("iterations must be >=1.")
    if not np.any(weights > 0):
        return _no_data_result(x, previous)
    row_sum = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    row_inv = np.zeros_like(row_sum)
    valid_rows = row_sum > 1.0e-12
    row_inv[valid_rows] = 1.0 / row_sum[valid_rows]
    effective_weights = weights * valid_rows.astype(np.float32)
    col_sum = np.asarray(matrix.T @ effective_weights).ravel().astype(np.float32)
    col_inv = np.zeros_like(col_sum)
    valid_cols = col_sum > 1.0e-12
    col_inv[valid_cols] = 1.0 / col_sum[valid_cols]
    relaxation = float(config.get("sart_relaxation", 0.8))
    tolerance = float(config.get("tolerance", 1.0e-5))
    log_every = max(1, int(config.get("log_every", 10)))
    history: list[SolverHistoryEntry] = []
    start_time = perf_counter()
    b_norm = max(float(np.linalg.norm(np.sqrt(weights) * b)), 1.0e-12)
    converged = False

    for iteration in range(1, int(iterations) + 1):
        residual = b - matrix @ x
        correction = matrix.T @ (effective_weights * row_inv * residual)
        x_new = np.maximum(
            x + np.float32(relaxation) * col_inv * np.asarray(correction).ravel(),
            0.0,
        ).astype(np.float32)
        relative_change = float(np.linalg.norm(x_new - x)) / max(
            float(np.linalg.norm(x)), 1.0e-8
        )
        x = x_new
        if iteration == 1 or iteration % log_every == 0 or iteration == iterations:
            residual_after = matrix @ x - b
            relative_residual = float(
                np.linalg.norm(np.sqrt(weights) * residual_after)
            ) / b_norm
            data_term = 0.5 * float(np.dot(weights * residual_after, residual_after)) / max(
                float(np.sum(weights)), 1.0e-12
            )
            history.append(
                SolverHistoryEntry(
                    iteration=iteration,
                    objective=data_term,
                    data_term=data_term,
                    tv_term=0.0,
                    temporal_term=0.0,
                    support_term=0.0,
                    l2_term=0.0,
                    relative_change=relative_change,
                    relative_residual=relative_residual,
                    elapsed_seconds=perf_counter() - start_time,
                )
            )
            if verbose:
                print(
                    f"[solver] SART iter={iteration:4d} "
                    f"res={relative_residual:.4f} dx={relative_change:.3e}"
                )
        if relative_change < tolerance:
            converged = True
            break

    return SolverResult(
        x=x,
        history=history,
        iterations=iteration,
        converged=converged,
        lipschitz=0.0,
        status="converged" if converged else "max_iterations",
        termination_reason="relative_change" if converged else "iteration_budget",
    )


def solve(
    matrix: sparse.csr_matrix,
    b: np.ndarray,
    weights: np.ndarray,
    grid: VolumeGrid,
    config: dict[str, Any],
    iterations: int,
    x0: np.ndarray | None = None,
    previous: np.ndarray | None = None,
    support: np.ndarray | None = None,
    verbose: bool = True,
) -> SolverResult:
    name = str(config.get("name", "fista_huber_tv")).lower()
    if name == "fista_huber_tv":
        return solve_fista_huber_tv(
            matrix,
            b,
            weights,
            grid,
            config,
            iterations,
            x0=x0,
            previous=previous,
            support=support,
            verbose=verbose,
        )
    if name in {"sart", "sirt"}:
        return solve_sart(
            matrix,
            b,
            weights,
            grid,
            config,
            iterations,
            x0=x0,
            previous=previous,
            verbose=verbose,
        )
    raise ValueError("solver.name must be 'fista_huber_tv' or 'sirt' (legacy alias: 'sart').")

# Correct algorithm name; the old import remains available.
solve_sirt = solve_sart
