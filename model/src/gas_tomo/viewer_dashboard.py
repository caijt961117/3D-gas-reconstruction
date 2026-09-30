"""Offline scientific 3-D dashboard. No local server, CDN or telemetry required.

Single-frame HTML is self-contained by default. Sequences use sibling JS frame
assets loaded on demand, keeping only a small number of full volumes in memory.
"""
from __future__ import annotations
import base64
import html
import io
import json
from pathlib import Path
from typing import Any
import numpy as np
from plotly.offline import get_plotlyjs
from .grid import VolumeGrid
from .surfaces import resolve_levels, surface_mesh


def _json(value):
    # HTML script data must not be able to terminate a script element.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def _b64(volume):
    return base64.b64encode(np.asarray(volume, dtype="<f4").tobytes()).decode("ascii")


def load_dashboard_frame(bundle_path: str | Path) -> dict:
    with np.load(bundle_path, allow_pickle=False) as data:
        volume = data["concentration"].copy()
        grid = VolumeGrid(data["bounds_m"], tuple(map(int,data["shape_nx_ny_nz"])))
        meta = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
        cameras = json.loads(str(data["cameras_json"])) if "cameras_json" in data else []
        frame = {"volume": volume, "grid": grid, "metadata": meta, "cameras": cameras}
        frame["measurement_shapes"] = json.loads(str(data["measurement_shapes_json"])) if "measurement_shapes_json" in data else {}
        for name in ("observed","predicted","weights","measurement_shape"):
            frame[name] = data[name].copy() if name in data else None
    return frame


def _fallback_png(meshes: list[dict], grid: VolumeGrid, peak: float, estimate: bool, color_range=None) -> str:
    """Static 3-D fallback when the browser cannot create a WebGL context."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import colormaps
    figure=Figure(figsize=(7.2,5.3),dpi=105)
    FigureCanvasAgg(figure)
    axis=figure.add_subplot(111,projection="3d")
    lo, hi = color_range if color_range is not None else (0.0, peak if peak > 0 else 1.0)
    if estimate:
        for idx,mesh in enumerate(meshes):
            xyz=np.column_stack((mesh["x"],mesh["y"],mesh["z"]))
            triangles=np.column_stack((mesh["i"],mesh["j"],mesh["k"]))
            artist=Poly3DCollection(xyz[triangles],alpha=min(.9,.18+.14*idx),
                facecolor=colormaps["viridis"](np.clip((mesh["level"]-lo)/(hi-lo),0,1)),edgecolor="none")
            axis.add_collection3d(artist)
    axis.set(xlim=grid.bounds[0],ylim=grid.bounds[1],zlim=grid.bounds[2],
        xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)")
    axis.set_box_aspect(grid.bounds[:,1]-grid.bounds[:,0]);axis.view_init(24,-58)
    figure.suptitle("3-D isosurfaces | static fallback" if meshes and estimate else
                    "No valid concentration estimate" if not estimate else "No internal surface | inspect slices")
    buffer=io.BytesIO();figure.savefig(buffer,format="png",bbox_inches="tight")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _frame_payload(frame: dict, levels: list[float], index: int, color_range=None) -> dict:
    volume, grid = np.asarray(frame["volume"],dtype=np.float32), frame["grid"]
    if volume.shape != grid.array_shape or not np.isfinite(volume).all():
        raise ValueError("Dashboard volume must be finite and match its physical grid.")
    meta = frame.get("metadata") or {}
    estimate = meta.get("quality", {}).get("has_estimate", not meta.get("numeric_placeholder", False))
    meshes = []
    if estimate:
        for level in levels:
            xyz, faces = surface_mesh(volume,grid,level)
            if faces.size:
                meshes.append({"level":level,"x":xyz[:,0].tolist(),"y":xyz[:,1].tolist(),"z":xyz[:,2].tolist(),
                               "i":faces[:,0].tolist(),"j":faces[:,1].tolist(),"k":faces[:,2].tolist()})
    result = {"index":index,"values_b64":_b64(volume),"peak":float(volume.max()),
              "minimum":float(volume.min()), "metadata":meta,"has_estimate":estimate,"meshes":meshes,
              "static_png":_fallback_png(meshes,grid,float(volume.max()),estimate,color_range)}
    for name in ("observed","predicted","weights"):
        value = frame.get(name)
        result[name] = None if value is None else np.asarray(value,dtype=float).reshape(-1).tolist()
    return result


def write_dashboard(frames: list[dict], output_path: str | Path,
                    cfg: dict[str,Any] | None = None, *, external_frames: bool = False) -> Path:
    if not frames:
        raise ValueError("The dashboard requires at least one frame.")
    cfg = dict(cfg or {})
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    grid = frames[0]["grid"]
    if any(f["grid"].shape != grid.shape or not np.allclose(f["grid"].bounds,grid.bounds) for f in frames):
        raise ValueError("All sequence frames must use exactly the same physical grid.")
    peak = max(float(np.max(f["volume"])) for f in frames)
    level_cfg = dict(cfg)
    if cfg.get("isosurface_levels") is None:
        level_cfg["isosurface_levels"] = [v*peak for v in cfg.get("isosurface_fractions",cfg.get("interactive_isosurface_fractions",[.1,.25,.5]))]
    levels = resolve_levels(np.array([0,peak]),level_cfg)
    color_range = cfg.get("color_range") or [0.0, peak if peak>0 else 1.0]
    if len(color_range)!=2 or not np.isfinite(color_range).all() or not color_range[0]<color_range[1]:
        raise ValueError("color_range must contain two finite increasing numbers.")
    unit_records=[f.get("metadata",{}).get("units",{}) for f in frames]
    available_units=[u.get("concentration") for u in unit_records if u.get("concentration")]
    if len(set(available_units))>1:
        raise ValueError("Cannot compare sequence frames with different concentration units.")
    if available_units:
        cfg["concentration_unit"]=available_units[0]
    camera_models = [c.metadata() if hasattr(c,"metadata") else c for c in (frames[0].get("cameras") or [])]
    shape2 = frames[0].get("measurement_shape")
    shapes = frames[0].get("measurement_shapes") or ({c["name"]:list(map(int,shape2)) for c in camera_models} if shape2 is not None else {})
    # Camera mappings are global in this viewer: do not silently show old cameras
    # after a moving-camera / scene-switch export.
    for frame in frames[1:]:
        cams = [c.metadata() if hasattr(c,"metadata") else c for c in (frame.get("cameras") or [])]
        if _json(cams) != _json(camera_models):
            raise ValueError("A sequence viewer requires fixed camera geometry; export changed scenes separately.")
        fshapes = frame.get("measurement_shapes") or shapes
        if _json(fshapes) != _json(shapes):
            raise ValueError("A sequence viewer requires fixed observation grids.")
    manifest = []
    initial = None
    asset_dir = path.parent / (path.stem+"_assets")
    if external_frames:
        asset_dir.mkdir(exist_ok=True)
    for i,frame in enumerate(frames):
        payload = _frame_payload(frame,levels,i,color_range)
        meta = payload["metadata"]
        item = {"index":i,"frame_id":meta.get("frame_id") or str(i),"timestamp_s":meta.get("timestamp_s")}
        if external_frames:
            target = asset_dir/f"frame_{i:06d}.js"
            target.write_text(f"window.GAS_TOMO_FRAMES[{i}]={_json(payload)};",encoding="utf-8")
            item["asset"] = f"{asset_dir.name}/{target.name}"
        elif i==0:
            initial=payload
        else:
            item["embedded"]=payload
        manifest.append(item)
    if external_frames:
        # Embed first frame: the page is immediately useful even before loading assets.
        initial = _frame_payload(frames[0],levels,0,color_range)
    config = {"shape":list(grid.shape),"bounds":grid.bounds.tolist(),"spacing":grid.spacing.tolist(),
              "cameras":camera_models,"measurement_shape":None if shape2 is None else list(map(int,shape2)),
              "measurement_shapes":shapes,
              "color_range":list(map(float,color_range)),"levels":levels,
              "unit":cfg.get("concentration_unit","relative response / m"),
              "colorscale":cfg.get("interactive_colorscale","Viridis"),
              "max_voxels":int(cfg.get("interactive_max_voxels",180000)),
              "point_limit":int(cfg.get("interactive_point_limit",20000)),
              "volume_opacity":float(cfg.get("volume_opacity",0.08)),
              "min_fraction":float(cfg.get("volume_min_fraction",0.03)),
              "manifest":manifest, "initial":initial}
    if 0<config["max_voxels"]<8:
        raise ValueError("interactive_max_voxels must be 0 or at least 8.")
    include = str(cfg.get("interactive_include_plotlyjs","inline")).lower()
    if external_frames or include=="directory":
        asset_dir.mkdir(exist_ok=True)
        plotly_path=asset_dir/"plotly.min.js"
        if not plotly_path.exists():
            plotly_path.write_text(get_plotlyjs(),encoding="utf-8")
        plotly_tag = f'<script src="{html.escape(asset_dir.name)}/plotly.min.js"></script>'
    elif include=="cdn":
        from plotly.offline.offline import get_plotlyjs_version
        plotly_tag = f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>'
    else:
        plotly_tag = "<script>"+get_plotlyjs()+"</script>"
    root=Path(__file__).parent/"web_assets"
    template=(root/"dashboard.html").read_text(encoding="utf-8")
    javascript=(root/"dashboard.js").read_text(encoding="utf-8")
    rendered=template.replace("__PLOTLY__",plotly_tag).replace("__DATA__",_json(config)).replace("__SCRIPT__",javascript)
    path.write_text(rendered,encoding="utf-8")
    return path


def write_sequence_dashboard(bundles: list[str | Path], output_path: str | Path,
                             cfg: dict[str,Any] | None = None) -> Path:
    # The browser is lazy-loaded. Export currently reads the numerical bundles once;
    # split very long recordings into blocks when workstation RAM is limited.
    return write_dashboard([load_dashboard_frame(p) for p in bundles], output_path,cfg,external_frames=True)
