"""Scene diagnostics and portable metadata, not an accuracy/identifiability proof."""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import json
import numpy as np
from .camera import load_cameras
from .grid import VolumeGrid
from .sampling import measurement_shapes
from .system_matrix import build_or_load_system_matrix


def inspect_scene(cfg: dict, *, build_matrices: bool = False, verbose: bool = False):
    cameras = load_cameras(cfg)
    grid = VolumeGrid.from_config(cfg["volume"], cfg["volume"]["levels"][-1])
    shapes = measurement_shapes(cfg, cameras)
    centers = grid.centers()
    fov = np.zeros(grid.n_voxels, np.uint16)
    ray_count = np.zeros(grid.n_voxels, np.uint16)
    camera_reports, warnings = {}, []
    n_rows = sum(r*c for r,c in shapes.values())
    samples = int(cfg["measurement"]["ray_samples_per_bin"])
    bundle = None
    if build_matrices:
        bundle = build_or_load_system_matrix(cameras,grid,shapes,samples,cfg["measurement"]["matrix_cache_dir"],verbose=verbose)
    for camera in cameras:
        _, visible = camera.project_world(centers)
        fov += visible.astype(np.uint16)
        info = {"measurement_grid_rows_cols":list(shapes[camera.name]),
                "image_size_width_height":list(camera.image_size),
                "processed_size_width_height":list(camera.processed_size),
                "position_m":camera.center_world.tolist(),
                "fraction_voxel_centers_in_fov":float(np.mean(visible))}
        if not np.any(visible):
            warnings.append(f"{camera.name}: no voxel center lies in the camera field of view.")
        if bundle is not None:
            A = bundle.matrix[bundle.camera_slices[camera.name]]
            covered = np.asarray(A.sum(axis=0)).ravel()>0
            ray_count += covered.astype(np.uint16)
            info["sampled_ray_coverage_fraction"] = float(np.mean(covered))
            info["rays_intersecting_domain"] = int(np.count_nonzero(np.asarray(A.sum(axis=1)).ravel()>0))
        camera_reports[camera.name] = info
    center = np.mean(grid.bounds,axis=1)
    angles=[]
    for a,b in combinations(cameras,2):
        va,vb = center-a.center_world,center-b.center_world
        denom = np.linalg.norm(va)*np.linalg.norm(vb)
        angle = None if denom < 1e-12 else float(np.degrees(np.arccos(np.clip(abs(np.dot(va,vb))/denom,0,1))))
        angles.append({"cameras":[a.name,b.name],"unsigned_center_ray_separation_deg":angle})
        if angle is not None and angle<10:
            warnings.append(f"{a.name}/{b.name}: center rays are nearly parallel/opposed ({angle:.2f} deg unsigned); depth ambiguity may be severe.")
    if np.mean(fov>=2)<0.8:
        warnings.append("Less than 80% of voxel centers lie in two camera fields of view; inspect coverage before interpreting the full volume.")
    if cfg.get("units",{}).get("calibrated",False) is False:
        warnings.append("Uncalibrated/synthetic relative result; not an absolute gas concentration measurement.")
    nnz_upper = min(n_rows*grid.n_voxels,n_rows*samples*samples*(sum(grid.shape)-2))
    report={"scene":cfg.get("scene",{}),"bounds_m":grid.bounds.tolist(),
        "grid_shape_nx_ny_nz":list(grid.shape),"array_shape_nz_ny_nx":list(grid.array_shape),
        "voxel_size_m":grid.spacing.tolist(),"n_voxels":grid.n_voxels,"n_observations":n_rows,
        "dense_float32_bytes":int(4*n_rows*grid.n_voxels),
        "csr_upper_bound_bytes":int(8*nnz_upper+4*(n_rows+1)),
        "fraction_fov_at_least_two":float(np.mean(fov>=2)),
        "fraction_fov_all_cameras":float(np.mean(fov==len(cameras))),
        "camera_pairs":angles,"cameras":camera_reports,"warnings":warnings,
        "interpretation":"Field-of-view and sampled-ray coverage diagnostics are NOT uncertainty, uniqueness or quantitative accuracy guarantees. No obstacle occlusion model is applied."}
    if bundle is not None:
        A=bundle.matrix
        report.update({"matrix_id":bundle.matrix_id,"matrix_nnz":int(A.nnz),
            "matrix_bytes":int(A.data.nbytes+A.indices.nbytes+A.indptr.nbytes),
            "fraction_sampled_rays_at_least_two":float(np.mean(ray_count>=2))})
    arrays={"fov_view_count":grid.as_volume(fov),"bounds_m":grid.bounds,
            "shape_nx_ny_nz":np.array(grid.shape,dtype=np.int32)}
    if bundle is not None:
        arrays["sampled_ray_view_count"] = grid.as_volume(ray_count)
    return report, arrays


def save_scene_report(cfg, output_dir, *, build_matrices=False, verbose=False):
    report, arrays=inspect_scene(cfg,build_matrices=build_matrices,verbose=verbose)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    (output/"scene_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    np.savez_compressed(output/"coverage.npz",**arrays)
    return report
