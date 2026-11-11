"""
Writing and reading processed sessions.

One directory per session, containing:

    <stem>_traces.parquet    raw per-frame ROI means, frame time, LED state
    <stem>_dff.parquet       processed dF/F on the signal timebase
    <stem>_events.parquet    one row per stimulation: time, frame, ROI, power,
                             gate structure
    <stem>_manifest.json     every input hash, every parameter, every QC
                             number, and the package version

The manifest is the point.  A processed trace with no record of which
calibration produced its ROIs, which channel map read its DAQ log, or which
baseline timescale fitted its F0 cannot be audited later, and those are
exactly the choices that silently change results.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

__all__ = [
    "MANIFEST_SUFFIX",
    "session_output_dir",
    "write_session",
    "read_manifest",
    "is_up_to_date",
    "summarise_outputs",
]

MANIFEST_SUFFIX = "_manifest.json"


def _jsonable(obj: Any) -> Any:
    """Coerce numpy types, dataclasses and paths into JSON-serialisable form.

    Note the ordering: ``np.float64`` is a subclass of ``float`` and
    ``np.bool_`` is not, so the numpy cases must be tested first or a numpy
    NaN slips through as a bare ``NaN`` literal — which ``json`` will happily
    write and no strict JSON reader will accept.  Non-finite values become
    null, since JSON has no representation for them.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(x) for x in obj]
    return str(obj)


def session_output_dir(root: str | Path, session_id: str) -> Path:
    return Path(root) / session_id


def write_session(
    out_root: str | Path,
    session_id: str,
    manifest: dict,
    traces=None,
    dff=None,
    events=None,
    compression: str = "snappy",
) -> dict[str, Path]:
    """Write a session's tables and manifest, returning the paths written."""
    out_dir = session_output_dir(out_root, session_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for name, df in (("traces", traces), ("dff", dff), ("events", events)):
        if df is None:
            continue
        p = out_dir / f"{session_id}_{name}.parquet"
        df.to_parquet(p, compression=compression, index=False)
        written[name] = p

    manifest = dict(manifest)
    manifest["outputs"] = {k: str(v) for k, v in written.items()}
    manifest["written_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    p = out_dir / f"{session_id}{MANIFEST_SUFFIX}"
    with open(p, "w", encoding="utf-8") as fh:
        # allow_nan=False turns any escapee into an error here rather than an
        # unreadable file discovered later.
        json.dump(_jsonable(manifest), fh, indent=2, sort_keys=False,
                  allow_nan=False)
    written["manifest"] = p
    return written


def read_manifest(out_root: str | Path, session_id: str) -> Optional[dict]:
    p = session_output_dir(out_root, session_id) / f"{session_id}{MANIFEST_SUFFIX}"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def is_up_to_date(
    out_root: str | Path,
    session_id: str,
    input_hashes: dict[str, str],
    config_hash: str,
) -> bool:
    """True if this session was processed from identical inputs and settings.

    Both halves matter: re-running after a config change must reprocess even
    though the raw files are untouched, and re-running after a file is
    replaced must reprocess even though the settings are unchanged.
    """
    man = read_manifest(out_root, session_id)
    if man is None or man.get("status") != "ok":
        return False
    if man.get("config_hash") != config_hash:
        return False
    prior = man.get("inputs", {}).get("sha256", {})
    return all(prior.get(k) == v for k, v in input_hashes.items())


def summarise_outputs(out_root: str | Path):
    """One row per processed session, for a quick look across a batch.

    An ``animal_id`` ending in ``*`` was assumed rather than matched to a
    record; its region labels are unverified.
    """
    import pandas as pd

    rows = []
    for p in sorted(Path(out_root).glob(f"*/*{MANIFEST_SUFFIX}")):
        with open(p, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        qc = m.get("qc", {})
        rows.append({
            "session_id": m.get("session_id"),
            "animal_id": (f"{m.get('animal_id')}*" if m.get("animal_assumed")
                          else m.get("animal_id")),
            "status": m.get("status"),
            "n_frames": qc.get("n_video_frames"),
            "n_roi": qc.get("n_roi"),
            "n_events": qc.get("n_laser_events"),
            "fps": qc.get("effective_fps"),
            "anchor": qc.get("frame_anchor_confidence"),
            "parity_breaks": qc.get("led_parity_breaks"),
            "led_separation": qc.get("led_separation"),
            "error": m.get("error"),
            "n_warnings": len(m.get("warnings", [])),
        })
    return pd.DataFrame(rows)
