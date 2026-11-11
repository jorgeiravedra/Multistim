"""
Reading MATLAB files produced by the stim-record rig.

Two formats are in circulation: MATLAB <= v7.2 (readable by scipy.io.loadmat)
and v7.3 (HDF5, requires h5py).  ``load_mat`` hides the difference and always
returns a plain dict of numpy arrays / lists, so nothing downstream needs to
know which it got.

The bcam log is a MATLAB cell array named ``bcl``.  Its element ordering is NOT
self-describing, so the mapping below is taken from the usage in
``roi_analysis_OCstim_pascal_FASTER_101926`` and must be confirmed against the
rig's acquisition code before these outputs are trusted.  Every assumption is
surfaced through :class:`BcamLog.provenance` rather than buried.

MATLAB cell indices are 1-based; the map uses the same convention as the
original script so it can be diffed against it directly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "load_mat",
    "STIM_LOG_VARNAMES",
    "file_sha256",
    "file_fingerprint",
    "LARGE_FILE_BYTES",
    "BCAM_CHANNEL_MAP",
    "BcamLog",
    "load_bcam",
    "load_stim_log",
    "load_calibration",
]


# --------------------------------------------------------------------------- #
# Generic .mat loading
# --------------------------------------------------------------------------- #

def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, for provenance records."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


#: Files at or above this size are fingerprinted rather than fully hashed.
LARGE_FILE_BYTES: int = 64 << 20


def file_fingerprint(path: str | Path, chunk: int = 8 << 20) -> str:
    """Cheap identity check for a large file.

    Hashing a multi-gigabyte video reads the whole file, which on a slow or
    networked drive costs more than everything else in a reprocessing run that
    reuses the extracted traces.  This reads only the first and last ``chunk``
    bytes and combines them with the size and modification time.

    It is weaker than a full hash: a change confined to the middle of a file
    whose size and mtime are preserved would go unnoticed.  That does not
    happen to acquisition output, which is written once and not edited.  It is
    used only to decide whether extraction can be skipped; a full SHA-256 is
    still recorded whenever extraction actually runs, so the provenance of
    every processed trace is complete.
    """
    path = Path(path)
    st = path.stat()
    h = hashlib.sha256()
    h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    with open(path, "rb") as fh:
        h.update(fh.read(chunk))
        if st.st_size > chunk:
            fh.seek(max(st.st_size - chunk, chunk))
            h.update(fh.read(chunk))
    return "fp:" + h.hexdigest()


def _is_hdf5(path: Path) -> bool:
    """True for a MATLAB v7.3 (HDF5-backed) .mat file.

    A v7.3 file is not a bare HDF5 file: MATLAB writes its own 128-byte
    descriptive header first, so the HDF5 superblock signature appears at
    offset 512, not 0.  (The superblock may in principle sit at any power-of-2
    offset from 512, so those are probed too.)  Three independent checks, any
    of which is sufficient:

    * the signature at offset 0 — a plain HDF5 file;
    * the version field at bytes 124:126 of the MATLAB header, which is
      0x0200 for v7.3 and 0x0100 for v7 and earlier;
    * the signature at one of the candidate superblock offsets.
    """
    with open(path, "rb") as fh:
        head = fh.read(132)
        if head[:8] == _HDF5_MAGIC:
            return True
        # MATLAB header: 116 bytes text, 8 bytes subsys offset, 2 version,
        # 2 endian indicator ('IM' or 'MI').
        if len(head) >= 128:
            endian = head[126:128]
            if endian in (b"IM", b"MI"):
                ver = (int.from_bytes(head[124:126], "little")
                       if endian == b"IM"
                       else int.from_bytes(head[124:126], "big"))
                if ver == 0x0200:
                    return True
                if ver == 0x0100:
                    return False
        for off in (512, 1024, 2048, 4096):
            fh.seek(off)
            if fh.read(8) == _HDF5_MAGIC:
                return True
    return False


def _squeeze_scipy(obj: Any) -> Any:
    """Recursively unwrap the nested object arrays scipy.io produces."""
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        # A MATLAB cell array.  Flatten the singleton wrapper dimensions that
        # scipy adds, but keep the cell structure itself as a Python list.
        flat = obj.ravel(order="F")
        return [_squeeze_scipy(x) for x in flat]
    if isinstance(obj, np.ndarray) and obj.dtype.names:  # struct
        return {name: _squeeze_scipy(obj[name].ravel(order="F")[0])
                for name in obj.dtype.names}
    if isinstance(obj, np.ndarray):
        return np.squeeze(obj)
    return obj


UNREADABLE = "<unreadable MATLAB object>"


def _squeeze_h5(h5file, obj) -> Any:
    import h5py  # local import: only needed for v7.3 files

    if isinstance(obj, h5py.Dataset):
        if obj.dtype == h5py.ref_dtype:  # cell array of references
            refs = np.asarray(obj).ravel(order="F")
            out = []
            for r in refs:
                try:
                    out.append(_squeeze_h5(h5file, h5file[r]))
                except Exception:
                    # A MATLAB object (datetime, table, classdef) or a null
                    # reference.  Keep the cell's position so that indices into
                    # the cell array stay correct.
                    out.append(UNREADABLE)
            return out
        try:
            arr = np.asarray(obj)
        except Exception:
            return UNREADABLE
        # HDF5 stores MATLAB arrays transposed.
        if arr.ndim >= 2:
            arr = arr.T
        return np.squeeze(arr)
    if isinstance(obj, h5py.Group):
        # A MATLAB object is stored as a group carrying a MATLAB_class
        # attribute of a type we cannot reconstruct; a struct is a plain group.
        cls = obj.attrs.get("MATLAB_class", b"")
        if isinstance(cls, bytes):
            cls = cls.decode("utf-8", "replace")
        if cls in ("datetime", "duration", "table", "timetable", "string"):
            return UNREADABLE
        out_d = {}
        for k in obj.keys():
            try:
                out_d[k] = _squeeze_h5(h5file, obj[k])
            except Exception:
                out_d[k] = UNREADABLE
        return out_d
    return obj


def load_mat(path: str | Path) -> dict[str, Any]:
    """Load a .mat file of either vintage into a dict of numpy arrays/lists.

    MATLAB objects that have no numpy equivalent (datetime, table, classdef
    instances) cannot be reconstructed here.  They are returned as the string
    ``'<unreadable MATLAB object>'`` rather than raising, so that one
    unreadable cell — typically the datetimes in ``bcl{1:2}`` — does not make
    the numeric channels beside it unreachable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if _is_hdf5(path):
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ImportError(
                f"{path.name} is a MATLAB v7.3 file and needs h5py: "
                "conda install -c conda-forge h5py"
            ) from exc

        out: dict[str, Any] = {}
        with h5py.File(path, "r") as fh:
            for key in fh.keys():
                if key.startswith("#"):  # #refs#, #subsystem#
                    continue
                out[key] = _squeeze_h5(fh, fh[key])
        return out

    from scipy.io import loadmat

    try:
        raw = loadmat(path, squeeze_me=False, struct_as_record=False)
    except NotImplementedError as exc:  # v7.3 that _is_hdf5 failed to spot
        raise NotImplementedError(
            f"{path.name} is a MATLAB v7.3 file but was not detected as one. "
            "Report the first 132 bytes of the file "
            f"(open(r'{path}','rb').read(132)) so the sniffer can be fixed; "
            "as a workaround, re-save it from MATLAB with -v7."
        ) from exc
    return {k: _squeeze_scipy(v) for k, v in raw.items()
            if not k.startswith("__")}


# --------------------------------------------------------------------------- #
# bcam log
# --------------------------------------------------------------------------- #

#: MATLAB 1-based cell index -> channel name.  Derived from the usage in the
#: legacy preprocessing script:
#:   bcl{1}, bcl{2}  -> datetimes (used as ``second(bcl{1}) - second(bcl{2})``)
#:   bcl{3}          -> DAQ timestamps, seconds
#:   bcl{4}          -> Hamamatsu camera frame trigger
#:   bcl{5}          -> FLIR (behavior) camera frame trigger
#:   bcl{6}          -> LED identity / state line
#:   bcl{7}          -> laser analog command voltage
#:   bcl{8}          -> laser digital gate
#:
#: Confirmed empirically on derek_1000ms_testpfc (2026-10-19, 2 kHz, 112.2 s)
#: with :func:`load_bcam_raw_profile`:
#:
#:   cell4  0.003-3.34 V, 79.9 Hz, 8846 rising edges   -> camera exposure trigger
#:   cell5  2.064-2.099 V (34 mV span), no periodicity -> DC line, NOT a sync
#:   cell6  0.003-0.009 V (6 mV span), no periodicity  -> dead, NOT the LED state
#:   cell7  -0.003 to 0.117 V, 10 rising edges, 8.9%   -> laser analog command
#:   cell8  binary 0/1, 802 edges at 79.9 Hz, 2.3%     -> laser gate, chopped
#:                                                        at the camera rate
#:
#: The laser occupies two channels with different meanings.  ``laser_cmd`` is
#: the analog amplitude command: its rising edge is the commanded onset and its
#: level encodes power (cf. ``intensity_mat``, 0.12-0.175 V for 1.6 mW).
#: ``laser_gate`` is the digital line that actually fires the laser, chopped at
#: the camera frame rate when "LASER BETWEEN CAMERA FRAMES ONLY" is enabled --
#: 802 gate edges for 10 commanded 1 s pulses is ~80 chops per pulse.  An
#: analysis that counts gate edges counts chops, not stimulations.
BCAM_CHANNEL_MAP: dict[int, str] = {
    3: "daq_time",
    4: "ham_cam",
    5: "aux_analog",   # labelled FLIRCam by the acquisition code; see below
    6: "led",
    7: "laser_cmd",
    8: "laser_gate",
}

#: Minimum peak-to-peak span (V) for a channel to be considered live.
LIVE_CHANNEL_MIN_SPAN: float = 0.05


@dataclass
class BcamLog:
    """Parsed DAQ log.

    Attributes
    ----------
    time
        DAQ timestamps in seconds, one per DAQ sample.
    channels
        Analog/digital traces, same length as ``time``.
    rate_hz
        Median sampling rate estimated from ``time``.  Not read from the file:
        the file does not record it.
    extras
        Cells that fell outside the channel map (e.g. the leading datetimes),
        retained verbatim.
    provenance
        Everything needed to audit how this object was built.
    """

    time: np.ndarray
    channels: dict[str, np.ndarray]
    rate_hz: float
    extras: dict[int, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.time.size)

    @property
    def duration_s(self) -> float:
        return float(self.time[-1] - self.time[0]) if self.n_samples else 0.0

    def describe(self) -> str:
        lines = [
            f"bcam log: {self.n_samples} samples, {self.duration_s:.2f} s, "
            f"~{self.rate_hz:.1f} Hz",
        ]
        for name, trace in self.channels.items():
            lines.append(
                f"  {name:<10s} min={np.nanmin(trace):+.4g} "
                f"max={np.nanmax(trace):+.4g} "
                f"unique={np.unique(np.round(trace, 4)).size}"
            )
        if self.provenance.get("warnings"):
            for w in self.provenance["warnings"]:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


def load_bcam(
    path: str | Path,
    channel_map: Mapping[int, str] = BCAM_CHANNEL_MAP,
    varname: str = "bcl",
) -> BcamLog:
    """Parse a ``*_bcam.mat`` DAQ log.

    The sampling rate is estimated from the timestamp channel, since the file
    does not store it.  A non-uniform timebase (jitter > 5% of the median
    interval) is reported as a warning rather than silently resampled; for a
    hardware-clocked DAQ it indicates dropped scans.
    """
    path = Path(path)
    raw = load_mat(path)
    warnings: list[str] = []

    if varname not in raw:
        raise KeyError(
            f"{path.name}: no variable '{varname}' "
            f"(found: {sorted(raw)}). Pass varname= to override."
        )
    cells = raw[varname]
    if not isinstance(cells, list):
        cells = [cells]

    n_cells = len(cells)
    expected = max(channel_map)
    if n_cells != expected:
        warnings.append(
            f"cell array has {n_cells} elements, channel map expects "
            f"{expected}; mapping may be wrong for this acquisition version"
        )

    channels: dict[str, np.ndarray] = {}
    extras: dict[int, Any] = {}
    for idx1, cell in enumerate(cells, start=1):
        name = channel_map.get(idx1)
        if name is None:
            extras[idx1] = cell
            continue
        try:
            arr = np.asarray(cell, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{path.name}: cell {idx1} is mapped to channel {name!r} but "
                f"is not numeric (got {type(cell).__name__}: {cell!r:.60}). "
                "The channel map is probably wrong for this acquisition "
                "version — check load_bcam(..., channel_map=...)."
            ) from exc
        channels[name] = arr

    if "daq_time" not in channels:
        raise KeyError(f"{path.name}: channel map produced no 'daq_time'")
    time = channels.pop("daq_time")

    lengths = {name: a.size for name, a in channels.items()}
    bad = {n: L for n, L in lengths.items() if L != time.size}
    if bad:
        warnings.append(
            f"channel length mismatch vs daq_time ({time.size}): {bad}; "
            "truncating to the shortest"
        )
        n = min([time.size, *lengths.values()])
        time = time[:n]
        channels = {k: v[:n] for k, v in channels.items()}

    dt = np.diff(time)
    if dt.size == 0:
        raise ValueError(f"{path.name}: timestamp channel has < 2 samples")
    med_dt = float(np.median(dt))
    if med_dt <= 0:
        raise ValueError(f"{path.name}: non-increasing timestamps")
    jitter = float(np.max(np.abs(dt - med_dt)) / med_dt)
    if jitter > 0.05:
        warnings.append(
            f"timebase jitter {jitter:.1%} of the median interval "
            f"(max gap {np.max(dt) * 1e3:.2f} ms) — likely dropped DAQ scans"
        )

    return BcamLog(
        time=time,
        channels=channels,
        rate_hz=1.0 / med_dt,
        extras=extras,
        provenance={
            "path": str(path),
            "sha256": file_sha256(path),
            "varname": varname,
            "channel_map": dict(channel_map),
            "n_cells": n_cells,
            "rate_estimated_from": "median diff of daq_time",
            "timebase_jitter_frac": jitter,
            "warnings": warnings,
        },
    )


# --------------------------------------------------------------------------- #
# Stim log and calibration
# --------------------------------------------------------------------------- #

#: Variable names the stim GUI has used for its pulse log, newest first.
STIM_LOG_VARNAMES: tuple[str, ...] = ("pulse_log", "pl", "laser_log",
                                      "stim_log")


def _stim_log_blocks(cells, first_block: int) -> list:
    """Numeric 2-D blocks from a cell array or a plain matrix."""
    if not isinstance(cells, list):
        arr = np.atleast_2d(np.asarray(cells, dtype=float))
        return [arr] if arr.size else []
    blocks = []
    for cell in cells[first_block - 1:]:
        if isinstance(cell, str):
            continue
        try:
            arr = np.atleast_2d(np.asarray(cell, dtype=float))
        except (TypeError, ValueError):
            continue
        if arr.size and arr.ndim == 2 and arr.shape[1] >= 2:
            blocks.append(arr)
    return blocks


def _infer_stim_log_columns(arr: np.ndarray, n_fibers_max: int = 64) -> tuple:
    """Work out which columns hold the timestamp and the ROI id.

    The log format is not fixed -- the variable has been called ``pl`` and
    ``pulse_log``, and the column order is not guaranteed across versions --
    so columns are identified by content:

    * an **ROI id** column holds integers within the fiber count, with FEWER
      distinct values than rows (several events per site);
    * the **timestamp** column is whatever remains with the most distinct
      values, typically one per event.

    Monotonicity is deliberately not used.  A log concatenated from
    per-sweep blocks is not in time order, and a sorted ROI column would be
    mistaken for a clock.

    Returns ``(id_col, time_col, evidence)``, and raises rather than guessing:
    a wrong ROI column assigns every stimulation to the wrong site and
    permutes the connectivity matrix with no error raised.
    """
    n_rows, n_cols = arr.shape
    if n_rows < 2:
        raise ValueError("stim log has fewer than 2 rows")

    stats = []
    for j in range(n_cols):
        v = arr[:, j]
        finite = np.all(np.isfinite(v))
        n_uniq = int(np.unique(v).size) if finite else 0
        integral = finite and np.allclose(v, np.round(v))
        in_range = finite and v.min() >= 1 and v.max() <= n_fibers_max
        stats.append({"col": j, "n_uniq": n_uniq, "span": float(np.ptp(v))
                      if finite else 0.0, "integral": integral,
                      "in_range": in_range, "finite": finite})

    id_cands = [c for c in stats
                if c["integral"] and c["in_range"] and 1 < c["n_uniq"] < n_rows]
    if not id_cands:
        id_cands = [c for c in stats
                    if c["integral"] and c["in_range"] and c["n_uniq"] >= 1]
    if not id_cands:
        raise ValueError(
            f"no column of integers within 1..{n_fibers_max} in a "
            f"{n_rows}x{n_cols} stim log; cannot identify the ROI id. "
            "Pass id_col explicitly."
        )
    id_col = max(id_cands, key=lambda c: (c["n_uniq"], -c["col"]))["col"]

    time_cands = [c for c in stats
                  if c["col"] != id_col and c["finite"] and c["n_uniq"] > 1]
    if not time_cands:
        raise ValueError(
            f"no column with more than one distinct value besides the ROI id "
            f"in a {n_rows}x{n_cols} stim log; cannot identify the "
            "timestamp. Pass time_col explicitly."
        )
    time_col = max(time_cands, key=lambda c: (c["n_uniq"], c["span"]))["col"]

    return id_col, time_col, {
        "n_rows": int(n_rows), "n_cols": int(n_cols),
        "id_col": int(id_col), "time_col": int(time_col),
        "n_distinct_ids": int(np.unique(arr[:, id_col]).size),
        "ids": sorted(int(x) for x in np.unique(arr[:, id_col])),
        "time_span_s": float(np.ptp(arr[:, time_col])),
        "n_distinct_times": int(np.unique(arr[:, time_col]).size),
    }


def load_stim_log(
    path: str | Path,
    varname: Optional[str] = None,
    first_block: int = 3,
    id_col: Optional[int] = None,
    time_col: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Load the GUI stim log into an (n_pulses, 2) array of [roi_id, time_s].

    Neither the variable name nor the column order is stable across versions
    of the acquisition GUI -- the log has been called ``pl`` and
    ``pulse_log`` -- so both are discovered rather than assumed.  Pass
    ``varname``, ``id_col`` or ``time_col`` to override.

    Rows are returned sorted by time.  The legacy MATLAB called
    ``sort(pulse_ids, 2)``, which sorts *within* each row across the two
    columns; it preserved column order only because ROI ids happen to be
    numerically smaller than timestamps, and it never ordered the events
    chronologically, so concatenating sweeps that were not already in time
    order mis-assigned ROI identity.
    """
    path = Path(path)
    raw = load_mat(path)
    candidates = ([varname] if varname else
                  [v for v in STIM_LOG_VARNAMES if v in raw])
    if not candidates:
        numeric = [k for k, v in raw.items()
                   if isinstance(v, (list, np.ndarray))]
        if len(numeric) == 1:
            candidates = numeric
    if not candidates or candidates[0] not in raw:
        raise KeyError(
            f"{path.name}: no stim log variable found. Tried "
            f"{STIM_LOG_VARNAMES}; file contains {sorted(raw)}. "
            "Pass varname= to override."
        )
    name = candidates[0]
    blocks = _stim_log_blocks(raw[name], first_block)
    if not blocks:
        # a newer format may store the table directly rather than in blocks
        blocks = _stim_log_blocks(raw[name], 1)
    if not blocks:
        return np.empty((0, 2), dtype=float)

    arr = np.vstack(blocks)
    if id_col is None or time_col is None:
        i, t, ev = _infer_stim_log_columns(arr)
        id_col = i if id_col is None else id_col
        time_col = t if time_col is None else time_col
        if verbose:
            print(f"{path.name}: variable {name!r}, {ev}")
    if arr.shape[1] <= max(id_col, time_col):
        raise ValueError(
            f"{path.name}: stim log has {arr.shape[1]} columns, need > "
            f"{max(id_col, time_col)}"
        )
    out = arr[:, [id_col, time_col]]
    return out[np.argsort(out[:, 1], kind="stable")]


def load_calibration(
    translated_coords_path: str | Path,
    mids_path: str | Path,
) -> dict[str, Any]:
    """Load fiber ROI calibration (centres + radii, and the image midpoints).

    ``translated_coords`` is (n_fibers, 3): [x, y, radius] in pixels of the
    full camera frame; ``mids`` is the (1, 2) offset subtracted to place ROIs
    in the cropped image.
    """
    tc = np.atleast_2d(np.asarray(load_mat(translated_coords_path)["translated_coords"],
                                  dtype=float))
    mids = np.asarray(load_mat(mids_path)["mids"], dtype=float).ravel()
    if tc.shape[1] < 3:
        raise ValueError(
            f"translated_coords has shape {tc.shape}; expected (n_fibers, >=3) "
            "of [x, y, radius]"
        )
    if mids.size < 2:
        raise ValueError(f"mids has {mids.size} elements; expected >= 2")
    return {
        "coords": tc[:, :3],
        "mids": mids[:2],
        "n_fibers": int(tc.shape[0]),
        "provenance": {
            "translated_coords": {
                "path": str(translated_coords_path),
                "sha256": file_sha256(translated_coords_path),
            },
            "mids": {
                "path": str(mids_path),
                "sha256": file_sha256(mids_path),
            },
        },
    }


# --------------------------------------------------------------------------- #
# Channel profiling
# --------------------------------------------------------------------------- #

def _otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    """Two-class threshold maximising between-class variance."""
    finite = x[np.isfinite(x)]
    if finite.size == 0 or np.ptp(finite) == 0:
        return float("nan")
    counts, edges = np.histogram(finite, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(counts).astype(float)
    total = w[-1]
    if total == 0:
        return float("nan")
    mu = np.cumsum(counts * centres)
    w0 = w / total
    w1 = 1.0 - w0
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = mu / w
        m1 = (mu[-1] - mu) / (total * w1)
        between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = -np.inf
    return float(centres[int(np.argmax(between))])


def _dominant_frequency(x: np.ndarray, rate_hz: float,
                        fmin: float = 0.5) -> tuple[float, float]:
    """Peak frequency of the mean-subtracted trace, and its share of power.

    A square pulse train has power at its fundamental and odd harmonics, so the
    fundamental is the lowest strong peak; the returned share indicates how
    periodic the channel is at all (a DC line or white noise gives a low share).
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 16 or np.ptp(v) == 0:
        return float("nan"), 0.0
    v = v - v.mean()
    n = int(2 ** np.floor(np.log2(v.size)))  # power of two, cheap FFT
    v = v[:n]
    spec = np.abs(np.fft.rfft(v * np.hanning(n))) ** 2
    freq = np.fft.rfftfreq(n, d=1.0 / rate_hz)
    keep = freq >= fmin
    if not keep.any() or spec[keep].sum() == 0:
        return float("nan"), 0.0
    k = int(np.argmax(spec[keep]))
    return float(freq[keep][k]), float(spec[keep][k] / spec[keep].sum())


def profile_channels(log: "BcamLog", edge_threshold: str = "otsu"):
    """Characterise every channel of a DAQ log, to identify what it carries.

    Returns a DataFrame, one row per channel, with:

    ``min/max/p01/p99``   amplitude range, robust and raw
    ``n_levels``          distinct values after rounding to 3 dp; a digital
                          line has few, an analog line has many
    ``threshold``         the two-class threshold used for edge counting
    ``frac_high``         duty cycle above that threshold
    ``n_rising``          rising edges over the whole recording
    ``edge_rate_hz``      rising edges per second
    ``dom_freq_hz``       dominant frequency of the mean-subtracted trace
    ``freq_power_frac``   that frequency's share of total power; high means
                          genuinely periodic, low means DC or noise
    ``guess``             a plain-language reading of the above

    The ``guess`` column is a description of the signal, not an assignment of
    meaning: it will tell you a channel is an ~80 Hz pulse train, which is what
    identifies it as the camera trigger given a known frame rate, but it cannot
    tell a camera trigger from any other 80 Hz line.
    """
    import pandas as pd

    rows = []
    for name, x in log.channels.items():
        x = np.asarray(x, dtype=float)
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        p01, p99 = (float(v) for v in np.nanpercentile(x, [1, 99]))
        n_levels = int(np.unique(np.round(x, 3)).size)
        thr = (_otsu_threshold(x) if edge_threshold == "otsu"
               else 0.5 * (lo + hi))
        if np.isfinite(thr):
            high = x > thr
            n_rising = int(np.count_nonzero(np.diff(high.astype(np.int8)) == 1))
            frac_high = float(high.mean())
        else:
            n_rising, frac_high = 0, float("nan")
        dur = log.duration_s or float("nan")
        fdom, fshare = _dominant_frequency(x, log.rate_hz)

        span = hi - lo
        if span < 1e-3:
            guess = "constant / unused"
        elif fshare > 0.02 and np.isfinite(fdom):
            guess = f"periodic ~{fdom:.1f} Hz ({n_rising} edges)"
        elif n_levels <= 8:
            guess = f"digital-like, {frac_high:.1%} high, {n_rising} edges"
        elif n_rising < 10:
            guess = "analog, few transitions (DC or slow)"
        else:
            guess = f"analog, {n_rising} threshold crossings"

        rows.append({
            "channel": name,
            "min": lo, "max": hi, "p01": p01, "p99": p99,
            "n_levels": n_levels,
            "threshold": thr,
            "frac_high": frac_high,
            "n_rising": n_rising,
            "edge_rate_hz": n_rising / dur if dur else float("nan"),
            "dom_freq_hz": fdom,
            "freq_power_frac": fshare,
            "guess": guess,
        })
    return pd.DataFrame(rows).set_index("channel")


def load_bcam_raw_profile(path: str | Path, varname: str = "bcl"):
    """Profile a bcam file's cells *without* applying a channel map.

    Use this when the channel map is unknown or suspected wrong: it loads every
    cell, reports which are numeric traces of full length, and profiles those,
    labelling them by their 1-based MATLAB cell index.
    """
    raw = load_mat(path)
    if varname not in raw:
        raise KeyError(f"{Path(path).name}: no variable {varname!r} "
                       f"(found: {sorted(raw)})")
    cells = raw[varname]
    if not isinstance(cells, list):
        cells = [cells]

    numeric: dict[int, np.ndarray] = {}
    skipped: dict[int, str] = {}
    for i, cell in enumerate(cells, start=1):
        if isinstance(cell, str):
            skipped[i] = cell
            continue
        try:
            arr = np.asarray(cell, dtype=float).ravel()
        except (TypeError, ValueError):
            skipped[i] = f"non-numeric ({type(cell).__name__})"
            continue
        numeric[i] = arr

    if not numeric:
        raise ValueError(f"{Path(path).name}: no numeric cells found")

    n = max(a.size for a in numeric.values())
    full = {i: a for i, a in numeric.items() if a.size == n}
    for i, a in numeric.items():
        if a.size != n:
            skipped[i] = f"length {a.size} (not {n})"

    # The timestamp channel is the one that is strictly increasing.
    time_idx = next(
        (i for i, a in sorted(full.items()) if a.size > 1 and np.all(np.diff(a) > 0)),
        None,
    )
    if time_idx is None:
        time = np.arange(n, dtype=float)
        rate = float("nan")
    else:
        time = full.pop(time_idx)
        rate = 1.0 / float(np.median(np.diff(time)))

    stub = BcamLog(
        time=time,
        channels={f"cell{i}": a for i, a in sorted(full.items())},
        rate_hz=rate,
        provenance={"path": str(path), "warnings": []},
    )
    prof = profile_channels(stub)
    prof.attrs["time_cell"] = time_idx
    prof.attrs["rate_hz"] = rate
    prof.attrs["skipped"] = skipped
    prof.attrs["n_samples"] = n
    return prof


def check_channels(
    log: "BcamLog",
    min_span: float = LIVE_CHANNEL_MIN_SPAN,
    expect_live: Sequence[str] = ("ham_cam", "laser_cmd"),
) -> list[str]:
    """Flag channels that carry no signal.

    A dead channel is not an error — on this rig several logged lines are
    genuinely unconnected — but it is a silent one: an unconnected input sits
    at a few millivolts of noise, and any threshold applied to it produces
    thousands of meaningless "edges".  (``cell6``, the nominal LED state line,
    yields ~52,000 such crossings across a 6 mV span.)  Call this before
    trusting any channel.
    """
    msgs: list[str] = []
    for name, x in log.channels.items():
        span = float(np.nanmax(x) - np.nanmin(x))
        if span < min_span:
            level = float(np.nanmedian(x))
            msgs.append(
                f"{name}: span {span * 1e3:.1f} mV about {level:.3f} V — no "
                "signal; thresholding this channel yields noise crossings, not "
                "events"
                + (" [EXPECTED LIVE]" if name in expect_live else "")
            )
    return msgs
