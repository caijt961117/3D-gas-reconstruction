from __future__ import annotations

import argparse
from datetime import datetime
import glob
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .calibration import (
    calibrate_extrinsics,
    calibrate_intrinsics,
    draw_volume_overlay,
    fit_response_polynomial,
)
from .camera import load_cameras
from .config import DEFAULTS, load_config
from .sampling import measurement_shapes
from .inputs import prepare_input_measurements
from .grid import VolumeGrid
from .interactive_visualization import (
    load_reconstruction_bundle,
    open_interactive_html,
    save_interactive_3d_outputs,
)
from .image_processing import make_median_background
from .reconstruction import reconstruct_images, reconstruct_measurements, ReconstructionSession
from .pairing import load_pair_manifest, pair_folders
from .viewer_dashboard import write_sequence_dashboard, load_dashboard_frame, write_dashboard
from .synthetic import make_synthetic_measurements, make_synthetic_truth
from .system_matrix import build_or_load_system_matrix
from .visualization import save_result_bundle


def _key_value_pairs(values: Iterable[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got: {value}")
        key, path = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty camera name in: {value}")
        if key in result:
            raise ValueError(f"Duplicate camera argument: {key}")
        result[key] = path.strip()
    return result


def _default_output(cfg: dict, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(cfg["project"]["output_dir"])
    return root / f"{cfg['project']['name']}_{suffix}_{timestamp}"


def _print_metrics(metrics: dict) -> None:
    summary = {
        "weighted_relative_projection_residual": metrics.get(
            "weighted_relative_projection_residual"
        ),
        "maximum_concentration": metrics.get("maximum_concentration"),
        "maximum_location_m": metrics.get("maximum_location_m"),
        "concentration_integral": metrics.get("concentration_integral"),
    }
    if "truth" in metrics:
        summary["truth"] = metrics["truth"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_build_matrix(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cameras = load_cameras(cfg)
    measurement_shape = measurement_shapes(cfg, cameras)
    for shape in cfg["volume"]["levels"]:
        grid = VolumeGrid.from_config(cfg["volume"], shape)
        build_or_load_system_matrix(
            cameras=cameras,
            grid=grid,
            measurement_shape=measurement_shape,
            samples_per_bin=int(cfg["measurement"]["ray_samples_per_bin"]),
            cache_dir=cfg["measurement"]["matrix_cache_dir"],
            force_rebuild=args.force,
            verbose=True,
        )


def command_synthetic(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cameras = load_cameras(cfg)
    final_grid = VolumeGrid.from_config(cfg["volume"], cfg["volume"]["levels"][-1])
    truth = make_synthetic_truth(final_grid, scenario=args.scenario, config=cfg.get("synthetic"))
    measurements = make_synthetic_measurements(
        cfg=cfg,
        cameras=cameras,
        truth=truth,
        noise_fraction=args.noise,
        seed=args.seed,
        force_matrix_rebuild=args.force,
        verbose=True,
    )
    result = reconstruct_measurements(
        cfg=cfg,
        cameras=cameras,
        measurements=measurements,
        force_matrix_rebuild=False,
        verbose=True,
    )
    output = Path(args.output) if args.output else _default_output(cfg, "synthetic")
    metrics = save_result_bundle(
        result,
        cfg,
        output,
        truth=truth,
        save_interactive_html_override=False if args.no_interactive_3d else None,
        export_profile=getattr(args, "export_profile", None),
    )
    print(f"[done] synthetic result: {output}")
    _print_metrics(metrics)


def command_reconstruct(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    image_paths = _key_value_pairs(args.image)
    background_paths = _key_value_pairs(args.background)
    camera_names = {camera["name"] for camera in cfg["cameras"]}
    if set(image_paths) != camera_names:
        raise ValueError(
            f"Provide exactly one --image NAME=PATH for each camera: {sorted(camera_names)}"
        )
    unexpected_backgrounds = set(background_paths) - camera_names
    if unexpected_backgrounds:
        raise ValueError(f"Unknown background camera names: {unexpected_backgrounds}")

    result = reconstruct_images(
        cfg=cfg,
        image_paths=image_paths,
        background_paths=background_paths,
        support_paths=_key_value_pairs(getattr(args,"support",None)),
        validity_paths=_key_value_pairs(getattr(args,"validity",None)),
        force_matrix_rebuild=args.force,
        verbose=True,
    )
    output = Path(args.output) if args.output else _default_output(cfg, "frame")
    metrics = save_result_bundle(
        result,
        cfg,
        output,
        save_interactive_html_override=False if args.no_interactive_3d else None,
        export_profile=getattr(args, "export_profile", None),
    )
    print(f"[done] reconstruction result: {output}")
    _print_metrics(metrics)


def command_sequence(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    backgrounds = _key_value_pairs(args.background)
    names = [camera["name"] for camera in cfg["cameras"]]
    if set(backgrounds)-set(names):
        raise ValueError("Unknown background camera name.")
    if args.manifest:
        if args.folder:
            raise ValueError("Use --manifest OR --folder, not both.")
        pairs = load_pair_manifest(args.manifest,names,args.max_time_skew_s)
    else:
        pairs = pair_folders(_key_value_pairs(args.folder),names,args.pattern)
    if args.limit is not None:
        if args.limit<1:
            raise ValueError("--limit must be positive.")
        pairs=pairs[:args.limit]
    if args.interactive_3d_every<1:
        raise ValueError("--interactive-3d-every must be positive.")
    output=Path(args.output) if args.output else _default_output(cfg,"sequence")
    output.mkdir(parents=True,exist_ok=True)
    session=ReconstructionSession(cfg)
    records=[];bundles=[]
    for index,pair in enumerate(pairs):
        if pair.reset_temporal:
            session.reset_temporal()
        print(f"[sequence] {index+1}/{len(pairs)}: ID={pair.frame_id}")
        result=session.reconstruct_inputs(pair.images, {**backgrounds,**pair.backgrounds},
            support_paths=pair.supports,validity_paths=pair.validity,
            timestamp_s=pair.timestamp_s,frame_id=pair.frame_id,
            force_matrix_rebuild=args.force and index==0,verbose=not args.quiet)
        result.quality["pairing_basis"]=pair.pairing_basis
        result.quality["camera_time_skew_s"]=pair.skew_s
        result.quality["temporal_reset_due_to_id_gap"]=pair.reset_temporal
        directory=output/f"frame_{index:06d}"
        metrics=save_result_bundle(result,cfg,directory,save_interactive_html_override=False,
                                   export_profile=args.export_profile)
        records.append({"frame":index,"frame_id":pair.frame_id,"images":pair.images,
            "timestamp_s":pair.timestamp_s,"camera_time_skew_s":pair.skew_s,
            "quality":result.quality,"maximum_concentration":metrics["maximum_concentration"]})
        bundles.append(directory/"reconstruction_bundle.npz")
        # Incremental manifest survives a later input/solver error.
        (output/"sequence_summary.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    if not args.no_interactive_3d:
        selected=bundles[::args.interactive_3d_every]
        if selected[-1]!=bundles[-1]:
            selected.append(bundles[-1])
        viewer=write_sequence_dashboard(selected,output/"sequence_3d_interactive.html",cfg["visualization"])
        print(f"[done] sequence viewer: {viewer}")
    print(f"[done] sequence numerical results: {output}")


def command_visualize_sequence(args):
    cfg=load_config(args.config) if args.config else None
    bundles=sorted(Path(args.folder).glob("frame_*/reconstruction_bundle.npz"))
    if not bundles:
        raise ValueError("No frame_*/reconstruction_bundle.npz files found.")
    output=Path(args.output) if args.output else Path(args.folder)/"sequence_3d_interactive.html"
    viewer=write_sequence_dashboard(bundles,output,cfg["visualization"] if cfg else None)
    print(f"[done] {viewer}")
    if args.open:
        open_interactive_html(viewer)


def command_synthetic_sequence(args):
    from scipy import ndimage
    cfg=load_config(args.config)
    if args.frames<2 or args.dt<=0:
        raise ValueError("Use at least two frames and positive --dt.")
    output=Path(args.output)
    session=ReconstructionSession(cfg)
    grid=VolumeGrid.from_config(cfg["volume"],cfg["volume"]["levels"][-1])
    initial=make_synthetic_truth(grid,"plume",config=cfg.get("synthetic"))
    bundles=[]
    for i in range(args.frames):
        # Kinematic demonstration, NOT a validated fluid-dynamic simulation.
        truth=ndimage.shift(initial,shift=(0.3*i,0.35*i,0.25*i),order=1,mode="constant",cval=0)*(0.55+0.45*i/(args.frames-1))
        measurements=make_synthetic_measurements(cfg,session.cameras,truth,noise_fraction=.01,seed=42+i,verbose=False)
        result=session.reconstruct(measurements,timestamp_s=i*args.dt,frame_id=str(i),verbose=not args.quiet)
        result.quality["synthetic_sequence_type"]="translated_scaled_field_not_CFD"
        directory=output/f"frame_{i:06d}"
        save_result_bundle(result,cfg,directory,truth=truth,export_profile="data")
        bundles.append(directory/"reconstruction_bundle.npz")
    viewer=write_sequence_dashboard(bundles,output/"sequence_3d_interactive.html",cfg["visualization"])
    print(f"[done] synthetic sequence: {viewer}")
    if args.open:
        open_interactive_html(viewer)


def command_make_background(args: argparse.Namespace) -> None:
    paths = sorted(glob.glob(args.images))
    report = make_median_background(paths, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_fit_response(args: argparse.Namespace) -> None:
    report = fit_response_polynomial(
        csv_path=args.csv,
        degree=args.degree,
        output_json=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

def command_calibrate_intrinsics(args: argparse.Namespace) -> None:
    paths = sorted(glob.glob(args.images))
    report = calibrate_intrinsics(
        image_paths=paths,
        board_cols=args.board_cols,
        board_rows=args.board_rows,
        square_size_m=args.square_size_m,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_calibrate_extrinsics(args: argparse.Namespace) -> None:
    report = calibrate_extrinsics(
        intrinsics_path=args.intrinsics,
        correspondences_csv=args.points,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify_calibration(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    b = cfg["volume"]["bounds_m"]
    bounds = np.array([b["x"], b["y"], b["z"]], dtype=np.float64)
    draw_volume_overlay(
        calibration_path=args.calibration,
        image_path=args.image,
        bounds=bounds,
        output_path=args.output,
    )
    print(f"[done] calibration overlay: {args.output}")



def command_query(args: argparse.Namespace) -> None:
    data = np.load(Path(args.bundle), allow_pickle=False)
    if "has_estimate" in data and not bool(data["has_estimate"]):
        raise ValueError("This frame has no concentration estimate; its zero array is only a flagged placeholder.")
    volume = data["concentration"]
    bounds = data["bounds_m"]
    nx, ny, nz = (int(v) for v in data["shape_nx_ny_nz"])
    grid = VolumeGrid(bounds=bounds, shape=(nx, ny, nz))
    x, y, z = grid.centers_1d()
    interpolator = RegularGridInterpolator(
        (z, y, x), volume, method="linear", bounds_error=True
    )
    value = float(interpolator([[args.z, args.y, args.x]])[0])
    print(
        json.dumps(
            {"x_m": args.x, "y_m": args.y, "z_m": args.z, "concentration": value},
            ensure_ascii=False,
            indent=2,
        )
    )


def command_visualize_3d(args: argparse.Namespace) -> None:
    volume, grid = load_reconstruction_bundle(args.bundle)
    bundle_path = Path(args.bundle).expanduser().resolve()

    cameras = None
    visual_cfg = dict(DEFAULTS["visualization"])
    if args.config:
        cfg = load_config(args.config)
        visual_cfg.update(cfg["visualization"])
        cameras = load_cameras(cfg)

    overrides = {
        "volume_min_fraction": args.threshold_fraction,
        "volume_surface_count": args.surface_count,
        "volume_opacity": args.opacity,
        "interactive_max_voxels": args.max_voxels,
        "interactive_point_limit": args.point_limit,
        "interactive_colorscale": args.colorscale,
    }
    for key, value in overrides.items():
        if value is not None:
            visual_cfg[key] = value
    if args.cdn:
        visual_cfg["interactive_include_plotlyjs"] = "cdn"

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else bundle_path.parent
    )
    frame=load_dashboard_frame(args.bundle)
    if cameras is not None:
        frame["cameras"]=cameras
    saved_units=frame.get("metadata",{}).get("units",{})
    if saved_units.get("concentration"):
        visual_cfg["concentration_unit"]=saved_units["concentration"]
    visual_cfg.update({"_viewer_cameras":frame["cameras"],"_viewer_metadata":frame["metadata"],"_observed":frame.get("observed"),
        "_predicted":frame.get("predicted"),"_weights":frame.get("weights"),"_measurement_shape":frame.get("measurement_shape"),
        "_measurement_shapes":frame.get("measurement_shapes",{})})
    paths = save_interactive_3d_outputs(volume=volume,grid=grid,cameras=cameras,
        output_dir=output_dir,visual_cfg=visual_cfg,prefix=args.prefix)
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    if args.open:
        open_interactive_html(paths["viewer"])



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gas-tomo",
        description="Scene-configurable gas concentration tomography (two views by default).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-matrix", help="Build/cache sparse system matrices.")
    build.add_argument("--config", required=True)
    build.add_argument("--force", action="store_true")
    build.set_defaults(func=command_build_matrix)

    synthetic = subparsers.add_parser("synthetic", help="Run end-to-end synthetic validation.")
    synthetic.add_argument("--config", required=True)
    synthetic.add_argument(
        "--scenario", default="plume", choices=["single_gaussian", "two_clouds", "plume"]
    )
    synthetic.add_argument("--noise", type=float, default=0.01)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--output")
    synthetic.add_argument("--force", action="store_true")
    synthetic.add_argument("--export-profile", choices=["data","preview","full"],default="full")
    synthetic.add_argument("--no-interactive-3d", action="store_true")
    synthetic.set_defaults(func=command_synthetic)

    reconstruct = subparsers.add_parser("reconstruct", help="Reconstruct one synchronized pair.")
    reconstruct.add_argument("--config", required=True)
    reconstruct.add_argument(
        "--image",
        action="append",
        required=True,
        help="Camera image as NAME=PATH; pass once per camera.",
    )
    reconstruct.add_argument(
        "--background",
        action="append",
        help="Optional static background as NAME=PATH.",
    )
    reconstruct.add_argument("--support",action="append",help="External support map NAME=PATH.")
    reconstruct.add_argument("--validity",action="append",help="Raw-pixel validity map NAME=PATH (1=valid).")
    reconstruct.add_argument("--output")
    reconstruct.add_argument("--force", action="store_true")
    reconstruct.add_argument("--export-profile",choices=["data","preview","full"],default="full")
    reconstruct.add_argument("--no-interactive-3d", action="store_true")
    reconstruct.set_defaults(func=command_reconstruct)

    sequence = subparsers.add_parser(
        "reconstruct-sequence", help="Reconstruct explicit ID/timestamp pairs with temporal continuity."
    )
    sequence.add_argument("--config", required=True)
    sequence.add_argument(
        "--folder", action="append", help="NAME=DIR; both folders must have identical numeric frame IDs."
    )
    sequence.add_argument(
        "--background", action="append", help="Static background as NAME=PATH."
    )
    sequence.add_argument("--manifest",help="CSV with frame_id and two camera path/timestamp columns.")
    sequence.add_argument("--max-time-skew-s",type=float,help="Required tolerance for timestamped manifests.")
    sequence.add_argument("--export-profile",choices=["data","preview","full"],default="data")
    sequence.add_argument("--pattern", default="*.png")
    sequence.add_argument("--limit", type=int)
    sequence.add_argument("--output")
    sequence.add_argument("--force", action="store_true")
    sequence.add_argument("--quiet", action="store_true")
    sequence.add_argument("--no-interactive-3d", action="store_true")
    sequence.add_argument(
        "--interactive-3d-every",
        type=int,
        default=1,
        help="Sample every N frames into ONE sequence viewer; numerical outputs keep all frames.",
    )
    sequence.set_defaults(func=command_sequence)


    viewseq=subparsers.add_parser("visualize-sequence",help="Create one offline sequence viewer from existing frame bundles.")
    viewseq.add_argument("--folder",required=True)
    viewseq.add_argument("--config")
    viewseq.add_argument("--output")
    viewseq.add_argument("--open",action="store_true")
    viewseq.set_defaults(func=command_visualize_sequence)
    synseq=subparsers.add_parser("synthetic-sequence",help="Demonstrate a moving synthetic field and temporal reconstruction.")
    synseq.add_argument("--config",required=True)
    synseq.add_argument("--output",default="outputs/demo_sequence")
    synseq.add_argument("--frames",type=int,default=5)
    synseq.add_argument("--dt",type=float,default=.1)
    synseq.add_argument("--quiet",action="store_true")
    synseq.add_argument("--open",action="store_true")
    synseq.set_defaults(func=command_synthetic_sequence)

    background = subparsers.add_parser(
        "make-background", help="Build a median no-gas background image."
    )
    background.add_argument("--images", required=True, help="Glob of no-gas frames.")
    background.add_argument("--output", required=True)
    background.set_defaults(func=command_make_background)

    response = subparsers.add_parser(
        "fit-response", help="Fit signal-to-column-density polynomial."
    )
    response.add_argument("--csv", required=True, help="CSV: signal,column_density")
    response.add_argument("--degree", type=int, default=2)
    response.add_argument("--output", help="Optional JSON report path.")
    response.set_defaults(func=command_fit_response)

    intrinsic = subparsers.add_parser(
        "calibrate-intrinsics", help="Calibrate camera intrinsics from checkerboard images."
    )
    intrinsic.add_argument("--images", required=True, help="Glob, e.g. calib/*.tif")
    intrinsic.add_argument("--board-cols", type=int, required=True, help="Inner corners per row.")
    intrinsic.add_argument("--board-rows", type=int, required=True, help="Inner corners per column.")
    intrinsic.add_argument("--square-size-m", type=float, required=True)
    intrinsic.add_argument("--output", required=True)
    intrinsic.set_defaults(func=command_calibrate_intrinsics)

    extrinsic = subparsers.add_parser(
        "calibrate-extrinsics", help="Estimate world-to-camera pose from 3D-2D points."
    )
    extrinsic.add_argument("--intrinsics", required=True)
    extrinsic.add_argument("--points", required=True)
    extrinsic.add_argument("--output", required=True)
    extrinsic.set_defaults(func=command_calibrate_extrinsics)

    verify = subparsers.add_parser(
        "verify-calibration", help="Overlay the 3D reconstruction box on a raw image."
    )
    verify.add_argument("--config", required=True)
    verify.add_argument("--calibration", required=True)
    verify.add_argument("--image", required=True)
    verify.add_argument("--output", required=True)
    verify.set_defaults(func=command_verify_calibration)

    visualize = subparsers.add_parser(
        "visualize-3d",
        aliases=["visualize3d"],
        help="Create/open interactive 3-D HTML from an existing reconstruction bundle.",
    )
    visualize.add_argument("--bundle", required=True, help="reconstruction_bundle.npz")
    visualize.add_argument(
        "--config",
        help="Optional YAML; adds camera geometry and uses its visualization settings.",
    )
    visualize.add_argument("--output-dir")
    visualize.add_argument("--prefix", default="reconstruction")
    visualize.add_argument(
        "--threshold-fraction",
        type=float,
        help="Override the minimum displayed fraction of the maximum concentration.",
    )
    visualize.add_argument(
        "--surface-count", type=int, help="Override the number of volume-rendering shells."
    )
    visualize.add_argument("--opacity", type=float, help="Override volume opacity.")
    visualize.add_argument(
        "--max-voxels", type=int, help="Maximum voxel count sent to the browser viewer."
    )
    visualize.add_argument(
        "--point-limit", type=int, help="Maximum points in the voxel-cloud display mode."
    )
    visualize.add_argument("--colorscale", help="Plotly colorscale, for example Turbo.")
    visualize.add_argument(
        "--cdn",
        action="store_true",
        help="Use Plotly CDN to make HTML smaller (internet required when opening).",
    )
    visualize.add_argument(
        "--open", action="store_true", help="Open the generated viewer in the default browser."
    )
    visualize.set_defaults(func=command_visualize_3d)

    query = subparsers.add_parser("query", help="Interpolate concentration at a 3D point.")
    query.add_argument("--bundle", required=True, help="reconstruction_bundle.npz")
    query.add_argument("--x", type=float, required=True)
    query.add_argument("--y", type=float, required=True)
    query.add_argument("--z", type=float, required=True)
    query.set_defaults(func=command_query)
    from .scene_cli import add_scene_commands
    add_scene_commands(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if getattr(args, "debug", False):
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
