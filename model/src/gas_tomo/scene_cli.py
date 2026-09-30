"""User-facing generic scene workflows; legacy commands remain available."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import csv
import json
import numpy as np
import yaml
from .config import load_config, dump_resolved_config
from .camera import load_cameras
from .grid import VolumeGrid
from .inputs import prepare_input_measurements
from .reconstruction import ReconstructionSession, reconstruct_measurements
from .pairing import load_pair_manifest
from .scene import save_scene_report
from .synthetic import make_synthetic_truth, make_synthetic_measurements
from .visualization import save_result_bundle
from .viewer_dashboard import write_sequence_dashboard
from .interactive_visualization import open_interactive_html


def _kv(items):
    from .cli import _key_value_pairs
    return _key_value_pairs(items)


def command_check_scene(args):
    cfg=load_config(args.config)
    output=Path(args.output) if args.output else Path(cfg["project"]["output_dir"])/"scene_check"
    report=save_scene_report(cfg,output,build_matrices=args.build_matrix,verbose=True)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print(f"[done] scene diagnostics: {output}")


def command_projections(args):
    cfg=load_config(args.config)
    cameras=load_cameras(cfg)
    data=prepare_input_measurements(cfg,cameras,_kv(args.projection),kind="column_density",
        support_paths=_kv(args.support),validity_paths=_kv(args.validity))
    result=reconstruct_measurements(cfg,cameras,data,force_matrix_rebuild=args.force,verbose=True)
    output=Path(args.output) if args.output else Path(cfg["project"]["output_dir"])/"projection_result"
    metrics=save_result_bundle(result,cfg,output,export_profile=args.export_profile)
    print(json.dumps({k:metrics.get(k) for k in ("quality","maximum_concentration","weighted_relative_projection_residual")},ensure_ascii=False,indent=2))
    if args.open and args.export_profile!="data":
        open_interactive_html(output/"reconstruction_3d_interactive.html")


def command_run_scene(args):
    cfg=load_config(args.config)
    output=Path(args.output) if args.output else Path(cfg["project"]["output_dir"])/cfg.get("scene",{}).get("id","scene")
    output.mkdir(parents=True,exist_ok=True)
    report=save_scene_report(cfg,output,build_matrices=True,verbose=not args.quiet)
    print(f"[scene] grid={report['grid_shape_nx_ny_nz']}, spacing(m)={report['voxel_size_m']}, observations={report['n_observations']}")
    for message in report["warnings"]:
        print(f"[warning] {message}")
    names=[c["name"] for c in cfg["cameras"]]
    pairs=load_pair_manifest(args.manifest,names,args.max_time_skew_s)
    if args.limit is not None:
        if args.limit<1: raise ValueError("--limit must be positive.")
        pairs=pairs[:args.limit]
    session=ReconstructionSession(cfg)
    bundles=[];records=[]
    for index,pair in enumerate(pairs):
        if pair.reset_temporal:
            session.reset_temporal()
            session.last_reset_reason="manifest_sequence_or_frame_gap"
        result=session.reconstruct_inputs(pair.images,background_paths=pair.backgrounds,
            support_paths=pair.supports,validity_paths=pair.validity,
            timestamp_s=pair.timestamp_s,frame_id=pair.frame_id,verbose=not args.quiet)
        result.quality.update({"sequence_id":pair.sequence_id,"pairing_basis":pair.pairing_basis,
                               "camera_time_skew_s":pair.skew_s})
        directory=output/f"frame_{index:06d}"
        profile=args.export_profile
        # Single frame gets its own interactive page; sequence uses one shared page.
        metrics=save_result_bundle(result,cfg,directory,export_profile=profile,
                                  save_interactive_html_override=len(pairs)==1 and profile!="data")
        bundles.append(directory/"reconstruction_bundle.npz")
        records.append({"frame_id":pair.frame_id,"sequence_id":pair.sequence_id,
                        "timestamp_s":pair.timestamp_s,"quality":result.quality,
                        "maximum_concentration":metrics.get("maximum_concentration")})
        (output/"sequence_summary.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    if not args.no_viewer:
        viewer=write_sequence_dashboard(bundles,output/"sequence_3d_interactive.html",cfg["visualization"])
        print(f"[done] {viewer}")
        if args.open: open_interactive_html(viewer)
    print(f"[done] numerical output: {output}")


def example_scene_config(bounds=None, shape=None):
    bounds=np.asarray(bounds or [0,2,0,1.2,0,1.5],dtype=float).reshape(3,2)
    widths=bounds[:,1]-bounds[:,0];center=bounds.mean(axis=1)
    distance=1.7*max(widths)
    shape=shape or [32,24,24]
    cameras=[]
    for name,delta,image_size,obs in [
        ("camera_a",[-.12,-1.,.08],[320,240],[32,40]),
        ("camera_b",[1.,.28,.18],[256,256],[28,32])]:
        cameras.append({"name":name,"type":"look_at","image_size":image_size,
            "position_m":(center+distance*np.array(delta)).tolist(),"target_m":center.tolist(),
            "up_world":[0,0,1],"fov_y_deg":55.,"measurement_grid_shape":obs})
    return {"project":{"name":"generic_scene","output_dir":"./outputs"},
        "scene":{"id":"generic_scene","world_unit":"m","description":"Example virtual camera setup, NOT measured calibration."},
        "volume":{"bounds_m":{a:list(map(float,bounds[i])) for i,a in enumerate("xyz")},"shape":shape},
        "measurement":{"grid_shape":[32,32],"ray_samples_per_bin":2,"matrix_cache_dir":"./cache"},
        "cameras":cameras,"input":{"kind":"column_density","geometry":"measurement_grid",
            "column_density_unit":"relative response"},
        "preprocess":{"measurement_normalization":"p99"},
        "support":{"enabled":True,"source":"auto","external_geometry":"measurement_grid"},
        "solver":{"iterations_per_level":[200]},
        "units":{"calibrated":False,"concentration":"relative response / m","column_density":"relative response"},
        "sequence":{"max_gap_s":0.5,"fine_level_only_after_first":True},
        "visualization":{"export_profile":"preview"}}


def command_init_scene(args):
    target=Path(args.output)
    if target.exists() and any(target.iterdir()):
        raise ValueError("Choose an empty scene directory; existing files will not be overwritten.")
    target.mkdir(parents=True,exist_ok=True)
    config=example_scene_config(args.bounds,args.shape)
    # Validate physical domain before writing the template.
    VolumeGrid.from_config(config["volume"], config["volume"]["shape"])
    (target/"scene.yaml").write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding="utf-8")
    (target/"calibration").mkdir(exist_ok=True);(target/"data").mkdir(exist_ok=True)
    with (target/"frames_template.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(["frame_id","camera_a_path","camera_b_path","camera_a_support","camera_b_support"])
        w.writerow(["0","data/camera_a.npy","data/camera_b.npy","",""])
    (target/"README.txt").write_text("Replace example camera geometry with measured calibration. Select raw_images or column_density input. Image size is [width,height]; measurement grid is [rows,cols]; volume shape is [nx,ny,nz]; stored arrays are [nz,ny,nx]. No gas-source location is required.\n",encoding="utf-8")
    print(f"[done] editable scene: {target/'scene.yaml'}")


def command_make_demo_data(args):
    from scipy import ndimage
    cfg=load_config(args.config)
    if args.frames<1 or not np.isfinite(args.dt) or args.dt<=0:
        raise ValueError("frames>=1 and finite dt>0 are required.")
    target=Path(args.output)
    if target.exists() and any(target.iterdir()):
        raise ValueError("Demo-data output directory must be empty (no automatic overwrite).")
    target.mkdir(parents=True,exist_ok=True)
    cameras=load_cameras(cfg);grid=VolumeGrid.from_config(cfg["volume"],cfg["volume"]["levels"][-1])
    initial=make_synthetic_truth(grid,args.scenario,cfg.get("synthetic"))
    rows=[]
    for index in range(args.frames):
        # Kinematic regression input, NOT a gas-release physics simulation.
        truth=ndimage.shift(initial,shift=[index*.15,index*.2,index*.25],order=1,mode="constant",cval=0)
        m=make_synthetic_measurements(cfg,cameras,truth,noise_fraction=args.noise,seed=args.seed+index,verbose=False)
        np.save(target/f"truth_{index:04d}.npy",truth)
        record={"frame_id":str(index),"sequence_id":"synthetic_example"}
        for camera in cameras:
            name=camera.name
            observation=m.grids[name]*m.scale
            signal_path=f"{name}_{index:04d}.npz";mask_path=f"{name}_{index:04d}_support.npy"
            np.savez_compressed(target/signal_path,column_density=observation.astype(np.float32),
                unit=np.array(cfg["units"]["column_density"]))
            # An example EXTERNAL support map derived from observations, not truth masks.
            from .support import _camera_support_score
            support,_=_camera_support_score(observation,cfg["support"])
            np.save(target/mask_path,support)
            record.update({name+"_path":signal_path,name+"_support":mask_path,name+"_timestamp_s":index*args.dt})
        rows.append(record)
    with (target/"frames.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    portable=deepcopy(cfg)
    portable["input"].update({"kind":"column_density","geometry":"measurement_grid",
        "column_density_unit":cfg["units"]["column_density"]})
    portable["support"].update({"source":"external","external_geometry":"measurement_grid"})
    portable["project"]["output_dir"]="./outputs"
    portable["measurement"]["matrix_cache_dir"]="./cache"
    # Copy calibration into the portable dataset instead of leaking absolute file paths.
    import shutil
    for camera in portable["cameras"]:
        if camera.get("type")=="calibrated":
            caldir=target/"calibration";caldir.mkdir(exist_ok=True)
            dst=caldir/(camera["name"]+".npz");shutil.copyfile(camera["calibration_file"],dst)
            camera["calibration_file"]="calibration/"+dst.name
    dump_resolved_config(portable,target/"scene.yaml")
    (target/"README.txt").write_text("Synthetic line-integral regression dataset. No real experimental accuracy is claimed. Each NPZ contains column_density and unit; support arrays are separate float [0,1]. Load scene.yaml and frames.csv together.\n",encoding="utf-8")
    print(f"[done] input dataset: {target}")


def add_scene_commands(subparsers):
    check=subparsers.add_parser("check-scene",help="Inspect physical grid, camera coverage and memory estimates.")
    check.add_argument("--config",required=True);check.add_argument("--output")
    check.add_argument("--build-matrix",action="store_true");check.set_defaults(func=command_check_scene)
    direct=subparsers.add_parser("reconstruct-projections",help="Reconstruct declared column-density maps without recalibration.")
    direct.add_argument("--config",required=True);direct.add_argument("--projection",action="append",required=True)
    direct.add_argument("--support",action="append");direct.add_argument("--validity",action="append")
    direct.add_argument("--output");direct.add_argument("--force",action="store_true")
    direct.add_argument("--export-profile",choices=["data","preview","full"],default="preview")
    direct.add_argument("--open",action="store_true");direct.set_defaults(func=command_projections)
    scene=subparsers.add_parser("run-scene",help="Run a portable scene/manifest with images, projections, supports and timestamps.")
    scene.add_argument("--config",required=True);scene.add_argument("--manifest",required=True)
    scene.add_argument("--max-time-skew-s",type=float);scene.add_argument("--limit",type=int)
    scene.add_argument("--export-profile",choices=["data","preview","full"],default="data")
    scene.add_argument("--output");scene.add_argument("--no-viewer",action="store_true")
    scene.add_argument("--quiet",action="store_true");scene.add_argument("--open",action="store_true")
    scene.set_defaults(func=command_run_scene)
    init=subparsers.add_parser("init-scene",help="Create a standalone editable scene template (example geometry only).")
    init.add_argument("--output",required=True)
    init.add_argument("--bounds",nargs=6,type=float,metavar=("XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"))
    init.add_argument("--shape",nargs=3,type=int);init.set_defaults(func=command_init_scene)
    demo=subparsers.add_parser("make-demo-data",help="Generate portable direct-projection/mask sequence inputs.")
    demo.add_argument("--config",required=True);demo.add_argument("--output",required=True)
    demo.add_argument("--frames",type=int,default=3);demo.add_argument("--dt",type=float,default=.1)
    demo.add_argument("--scenario",choices=["plume","single_gaussian","two_clouds"],default="plume")
    demo.add_argument("--noise",type=float,default=.01);demo.add_argument("--seed",type=int,default=42)
    demo.set_defaults(func=command_make_demo_data)
