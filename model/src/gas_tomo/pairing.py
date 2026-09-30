"""Explicit, fail-closed pairing of two camera streams."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import csv
import re
import math


@dataclass(frozen=True)
class FramePair:
    frame_id: str
    images: dict[str, str]
    timestamp_s: float | None = None
    skew_s: float | None = None
    reset_temporal: bool = False
    pairing_basis: str = "frame_id"
    supports: dict[str,str] = field(default_factory=dict)
    validity: dict[str,str] = field(default_factory=dict)
    backgrounds: dict[str,str] = field(default_factory=dict)
    sequence_id: str = ""


def _numeric_id(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        raise ValueError(f"No numeric frame ID in {path.name}; provide an explicit CSV manifest.")
    return int(matches[-1])


def pair_folders(folders: dict[str, str | Path], camera_names: list[str], pattern: str = "*.tif") -> list[FramePair]:
    """Require identical numeric IDs, never infer synchronization from list length."""
    if len(camera_names)<2 or set(folders)!=set(camera_names):
        raise ValueError("Provide exactly the configured camera folders (at least two).")
    lists={}
    for name in camera_names:
        folder=Path(folders[name])
        if not folder.is_dir():
            raise NotADirectoryError(str(folder))
        ids={}
        for path in folder.glob(pattern):
            if not path.is_file():
                continue
            idx=_numeric_id(path)
            if idx in ids:
                raise ValueError(f"Duplicate frame ID {idx} in {folder}.")
            ids[idx]=str(path.resolve())
        if not ids:
            raise ValueError(f"No frames match {pattern!r} in {folder}.")
        lists[name]=ids
    a=camera_names[0]
    for b in camera_names[1:]:
        if set(lists[a])!=set(lists[b]):
            raise ValueError(f"Mismatched frame IDs between {a} and {b}.")
    output=[]
    previous=None
    for idx in sorted(lists[a]):
        output.append(FramePair(str(idx),{name:lists[name][idx] for name in camera_names},
                                reset_temporal=previous is not None and idx!=previous+1))
        previous=idx
    return output


def load_pair_manifest(path: str | Path, camera_names: list[str],
                       max_time_skew_s: float | None = None) -> list[FramePair]:
    """CSV: frame_id,<cam>_path[,<cam>_timestamp_s],... . Times must share a clock.

    A timestamped manifest requires an explicit skew tolerance chosen for the
    experiment. Uncorrected independent camera clocks are not synchronized here.
    """
    path=Path(path).resolve()
    with path.open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        columns=set(reader.fieldnames or [])
        required={"frame_id",*(name+"_path" for name in camera_names)}
        if not required<=columns:
            raise ValueError(f"Manifest missing columns: {sorted(required-columns)}")
        ts_cols=[name+"_timestamp_s" for name in camera_names]
        has_time=all(c in columns for c in ts_cols)
        if any(c in columns for c in ts_cols) and not has_time:
            raise ValueError("A timestamp is required for ALL cameras, or for none.")
        if has_time and (max_time_skew_s is None or not math.isfinite(max_time_skew_s) or max_time_skew_s<0):
            raise ValueError("Timestamped manifests require explicit --max-time-skew-s >=0.")
        output=[];seen=set();previous_time=None;previous_id=None;used_paths={name:set() for name in camera_names}
        previous_sequence = None
        for line,row in enumerate(reader,start=2):
            sequence_id = (row.get("sequence_id") or "").strip()
            explicit_reset = (row.get("reset_temporal") or "").strip().lower() in {"1","true","yes"}
            if sequence_id != previous_sequence or explicit_reset:
                previous_time, previous_id = None, None
            scene_reset = previous_sequence is not None and sequence_id != previous_sequence
            previous_sequence = sequence_id
            frame_id=str(row["frame_id"]).strip()
            frame_key = (sequence_id,frame_id)
            if not frame_id or frame_key in seen:
                raise ValueError(f"Line {line}: empty or duplicate frame ID.")
            seen.add(frame_key);images={}
            for name in camera_names:
                raw=(row.get(name+"_path") or "").strip()
                if not raw:
                    raise ValueError(f"Line {line}: missing {name} path.")
                image=Path(raw).expanduser()
                if not image.is_absolute():
                    image=path.parent/image
                image=image.resolve()
                if not image.is_file():
                    raise FileNotFoundError(str(image))
                if str(image) in used_paths[name]:
                    raise ValueError(f"Line {line}: repeated image for {name}.")
                used_paths[name].add(str(image));images[name]=str(image)
            timestamp=skew=None
            if has_time:
                times=[float(row[c]) for c in ts_cols]
                if not all(math.isfinite(t) for t in times):
                    raise ValueError(f"Line {line}: timestamps must be finite.")
                skew=max(times)-min(times)
                if skew>max_time_skew_s+1e-12:
                    raise ValueError(f"Line {line}: camera skew {skew:.9g}s exceeds tolerance {max_time_skew_s}s.")
                timestamp=sum(times)/len(times)
                if previous_time is not None and timestamp<=previous_time:
                    raise ValueError(f"Line {line}: timestamps must increase strictly.")
                previous_time=timestamp
            reset=explicit_reset or scene_reset
            if frame_id.isdigit() and previous_id is not None:
                if int(frame_id)<=previous_id:
                    raise ValueError(f"Line {line}: numeric IDs must increase strictly.")
                reset=reset or int(frame_id)!=previous_id+1
            previous_id=int(frame_id) if frame_id.isdigit() else None
            auxiliary = {}
            for suffix in ("support", "validity", "background"):
                values = {}
                for name in camera_names:
                    raw = (row.get(name+"_"+suffix) or "").strip()
                    if raw:
                        file = Path(raw).expanduser()
                        file = (path.parent/file).resolve() if not file.is_absolute() else file.resolve()
                        if not file.is_file():
                            raise FileNotFoundError(file)
                        values[name] = str(file)
                auxiliary[suffix] = values
            output.append(FramePair(frame_id,images,timestamp,skew,reset,
                "explicit_manifest_timestamps" if has_time else "explicit_manifest_ids",
                auxiliary["support"],auxiliary["validity"],auxiliary["background"],sequence_id))
    if not output:
        raise ValueError("Manifest contains no frames.")
    return output
