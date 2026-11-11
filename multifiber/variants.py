"""
Recompute preprocessing variants from cached traces, for comparison.

The pipeline writes one processed result per session.  Choosing between
preprocessing options -- a baseline timescale, an isosbestic regressor, a
blanking rule -- means seeing several results side by side on the same data,
which is what this module makes cheap: it reads ``_traces.parquet`` and the
DAQ log once, then recomputes from memory.

Everything here is derived, never written back.  A variant is an exploration;
the pipeline's own output remains the record, and anything decided here has
to be applied through ``run_pipeline`` or ``reprocess`` to become one.

The comparison is honest only if the same data feed every variant, so the
session is loaded once and shared.  Note also that the circular-shift null is
recomputed per variant: a null built under one baseline does not describe
another.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import events as ev
from . import matio
from . import preprocess as pp

__all__ = ["VariantSpec", "SessionData", "load_session", "run_variant",
           "peri_event", "shift_null", "VariantResult"]


@dataclass(frozen=True)
class VariantSpec:
    """One preprocessing configuration."""

    # baseline
    baseline_tau_s: float = 60.0
    differences: int = 2
    baseline_mode: str = "continuous"       # continuous | epochs
    baseline_method: str = "airpls"         # airpls | symmetric
    lambda_override: Optional[float] = None  # bypass the timescale conversion

    # motion correction
    regress: bool = True
    predictors: str = "common"              # same | all | common
    common_method: str = "median"           # median | pc1
    alpha: "float | str" = "cv"

    # artifact
    blank_laser: bool = True
    exposure_s: Optional[float] = 0.0085
    blank_pad: int = 1

    # peri-event
    pre_s: float = 2.0
    post_s: float = 4.0
    baseline_window_s: float = 0.5
    response_start_s: float = 0.0
    response_window_s: float = 1.5
    detrend: str = "mean"                   # mean | linear
    statistic: str = "mean"                 # mean | median
    lowpass_hz: Optional[float] = None

    label: str = ""

    def key(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k != "label"}
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def describe(self) -> str:
        reg = ("none" if not self.regress
               else f"{self.predictors}"
                    + (f"/{self.common_method}" if self.predictors == "common"
                       else ""))
        lam = ("auto" if self.lambda_override is None
               else f"{self.lambda_override:.2e}")
        return (f"tau={self.baseline_tau_s:g}s d={self.differences} "
                f"{self.baseline_method} lam={lam} | iso={reg} "
                f"alpha={self.alpha} | blank={'on' if self.blank_laser else 'off'}"
                f" | base={self.baseline_window_s:g}s "
                f"resp={self.response_start_s:g}-"
                f"{self.response_start_s + self.response_window_s:g}s "
                f"{self.statistic}")


@dataclass
class SessionData:
    """Everything a variant needs, loaded once."""

    session_id: str
    traces: np.ndarray          # (n_frames, n_roi) raw ROI means
    frame_t: np.ndarray         # (n_frames,) DAQ time
    led_state: np.ndarray       # (n_frames,) 0/1
    labels: list[str]
    events: "object"            # DataFrame of laser events
    gate_on_t: Optional[np.ndarray]
    manifest: dict = field(default_factory=dict)

    @property
    def n_roi(self) -> int:
        return int(self.traces.shape[1])


def load_session(session_dir: str | Path,
                 load_gate: bool = True) -> SessionData:
    """Read the cached traces, events and (optionally) the gate channel."""
    import pandas as pd

    d = Path(session_dir)
    sid = d.name
    man_p = d / f"{sid}_manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}

    tdf = pd.read_parquet(d / f"{sid}_traces.parquet")
    labels = [c for c in tdf.columns
              if c not in ("frame", "t", "led_state", "roi_mean")]
    ev_p = d / f"{sid}_events.parquet"
    events = pd.read_parquet(ev_p) if ev_p.exists() else pd.DataFrame()

    gate = None
    if load_gate:
        bcam = man.get("inputs", {}).get("paths", {}).get("bcam")
        if bcam and Path(bcam).exists():
            try:
                log = matio.load_bcam(bcam)
                g = log.channels.get("laser_gate")
                if g is not None and float(np.nanmax(g) - np.nanmin(g)) >= 0.5:
                    gate = ev.detect_edges(log.time, g, 0.5).rising_t
            except Exception:
                gate = None

    return SessionData(
        session_id=sid,
        traces=tdf[labels].to_numpy(dtype=float),
        frame_t=tdf["t"].to_numpy(dtype=float),
        led_state=tdf["led_state"].to_numpy(dtype=int),
        labels=labels, events=events, gate_on_t=gate, manifest=man)


@dataclass
class VariantResult:
    spec: VariantSpec
    t: np.ndarray
    dff: np.ndarray             # (n_samples, n_roi)
    f0: np.ndarray
    blanked: np.ndarray
    qc: dict
    labels: list[str]

    @property
    def fs(self) -> float:
        p = float(np.median(np.diff(self.t)))
        return 1.0 / p if p > 0 else float("nan")


def run_variant(sd: SessionData, spec: VariantSpec) -> VariantResult:
    """Compute one preprocessing variant. No file is written."""
    onset = offset = None
    if len(sd.events):
        onset = sd.events["onset_t"].to_numpy(dtype=float)
        offset = sd.events["offset_t"].to_numpy(dtype=float)

    out = pp.preprocess_traces(
        sd.traces, sd.led_state, sd.frame_t,
        laser_onset_t=onset, laser_offset_t=offset,
        gate_on_t=(sd.gate_on_t if spec.blank_laser else None),
        exposure_s=spec.exposure_s,
        labels=sd.labels,
        baseline_tau_s=spec.baseline_tau_s,
        differences=spec.differences,
        baseline_mode=spec.baseline_mode,
        baseline_method=spec.baseline_method,
        alpha=spec.alpha,
        predictors=spec.predictors,
        common_method=spec.common_method,
        regress=spec.regress,
        blank_pad=spec.blank_pad,
    )
    dff = out.dff

    if spec.lambda_override is not None:
        # Refit F0 at an explicit penalty, bypassing the timescale conversion.
        dm = pp.demux_channels(sd.traces, sd.led_state, sd.frame_t,
                               labels=sd.labels)
        sig = dm.signal
        f0 = np.empty_like(sig)
        raw = np.empty_like(sig)
        for j in range(sig.shape[1]):
            f0[:, j] = pp.estimate_f0(sig[:, j],
                                      lambda_=float(spec.lambda_override),
                                      differences=spec.differences,
                                      method=spec.baseline_method)
            raw[:, j] = pp.compute_dff(sig[:, j], f0[:, j])
        dff = raw
        out.f0 = f0

    if spec.lowpass_hz:
        from scipy.signal import butter, sosfiltfilt
        period = float(np.median(np.diff(out.t)))
        sos = butter(4, spec.lowpass_hz, "lowpass", fs=1 / period, output="sos")
        for j in range(dff.shape[1]):
            v = dff[:, j]
            ok = np.isfinite(v)
            if ok.sum() > 20:
                w = v.copy()
                w[ok] = sosfiltfilt(sos, v[ok])
                dff[:, j] = w

    return VariantResult(spec=spec, t=out.t, dff=dff, f0=out.f0,
                         blanked=out.blanked, qc=out.qc, labels=sd.labels)


def _windows(y, t, onsets, pre, post, fs):
    npre, npost = int(pre * fs), int(post * fs)
    idx = np.searchsorted(t, onsets)
    idx = idx[(idx - npre >= 0) & (idx + npost < t.size)]
    if idx.size == 0:
        return None, npre, npost
    return np.stack([y[i - npre:i + npost] for i in idx]), npre, npost


def _baseline(W, npre, nbase, mode, tail_col=None):
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", category=RuntimeWarning)
        if mode == "mean" or tail_col is None:
            return W - np.nanmean(W[:, npre - nbase:npre], axis=1,
                                  keepdims=True)
        cx = np.arange(W.shape[1], dtype=float)
        fit = np.concatenate([np.arange(0, npre),
                              np.arange(min(tail_col, W.shape[1]),
                                        W.shape[1])])
        Y = W[:, fit]
        x = cx[fit]
        good = np.isfinite(Y)
        n = good.sum(axis=1)
        xs, ys = np.where(good, x, 0.0), np.where(good, Y, 0.0)
        sx, sy = xs.sum(1), ys.sum(1)
        sxx, sxy = (xs * xs).sum(1), (xs * ys).sum(1)
        den = n * sxx - sx * sx
        sl = np.where(den > 0, (n * sxy - sx * sy) / np.where(den > 0, den, 1), 0)
        it = np.where(n > 0, (sy - sl * sx) / np.where(n > 0, n, 1), 0)
        return W - (sl[:, None] * cx[None, :] + it[:, None])


def peri_event(res: VariantResult, onsets: np.ndarray, roi: int,
               width_s: float = 1.0) -> dict:
    """Trial-averaged peri-event trace and scalar effect for one ROI."""
    import warnings as _w

    s = res.spec
    fs = res.fs
    y = res.dff[:, roi]
    W, npre, npost = _windows(y, res.t, onsets, s.pre_s, s.post_s, fs)
    if W is None:
        return {"n": 0}
    nbase = int(s.baseline_window_s * fs)
    tail = npre + int((width_s + 1.2) * fs)
    Wc = _baseline(W, npre, nbase, s.detrend, tail)
    comb = np.nanmedian if s.statistic == "median" else np.nanmean
    with _w.catch_warnings():
        _w.simplefilter("ignore", category=RuntimeWarning)
        obs = comb(Wc, axis=0)
        r0 = npre + int(s.response_start_s * fs)
        r1 = r0 + int(s.response_window_s * fs)
        eff = float(np.nanmean(obs[r0:r1]))
    return {"t": (np.arange(-npre, npost)) / fs, "mean": obs, "trials": Wc,
            "effect": eff, "n": int(Wc.shape[0]), "r0": r0, "r1": r1}


def shift_null(res: VariantResult, onsets: np.ndarray, roi: int,
               n_shift: int = 400, seed: int = 0,
               width_s: float = 1.0) -> dict:
    """Circular-shift null for one ROI under this variant.

    Recomputed per variant on purpose: a null built under one baseline does
    not describe another, so comparing a variant's effect against another
    variant's null would be meaningless.
    """
    import warnings as _w

    s = res.spec
    fs = res.fs
    rng = np.random.default_rng(seed)
    y = res.dff[:, roi]
    npre, npost = int(s.pre_s * fs), int(s.post_s * fs)
    idx = np.searchsorted(res.t, onsets)
    idx = idx[(idx - npre >= 0) & (idx + npost < res.t.size)]
    if idx.size == 0:
        return {"n": 0}
    lo, hi = npre, res.t.size - npost
    span = hi - lo
    ms = npre + npost
    if span <= 2 * ms:
        return {"n": 0}
    rel = idx - lo
    nbase = int(s.baseline_window_s * fs)
    tail = npre + int((width_s + 1.2) * fs)
    comb = np.nanmedian if s.statistic == "median" else np.nanmean
    r0 = npre + int(s.response_start_s * fs)
    r1 = r0 + int(s.response_window_s * fs)

    effs, traces = [], []
    with _w.catch_warnings():
        _w.simplefilter("ignore", category=RuntimeWarning)
        for sh in rng.integers(ms, span - ms, size=n_shift):
            c = ((rel + sh) % span) + lo
            W = np.stack([y[i - npre:i + npost] for i in c])
            m = comb(_baseline(W, npre, nbase, s.detrend, tail), axis=0)
            traces.append(m)
            effs.append(float(np.nanmean(m[r0:r1])))
    N = np.vstack(traces)
    e = np.asarray(effs, dtype=float)
    return {"effects": e, "traces": N, "mean": float(np.nanmean(e)),
            "sd": float(np.nanstd(e)), "n": int(e.size)}


def compare(res: VariantResult, onsets: np.ndarray, roi: int,
            n_shift: int = 400, width_s: float = 1.0) -> dict:
    """Effect, z and p for one ROI under one variant."""
    pe = peri_event(res, onsets, roi, width_s)
    if not pe.get("n"):
        return {"n": 0}
    nl = shift_null(res, onsets, roi, n_shift=n_shift, width_s=width_s)
    if not nl.get("n"):
        return {**pe, "z": np.nan, "p": np.nan}
    z = ((pe["effect"] - nl["mean"]) / nl["sd"]) if nl["sd"] > 0 else np.nan
    p = (1.0 + int(np.sum(np.abs(nl["effects"] - nl["mean"])
                          >= abs(pe["effect"] - nl["mean"])))) / (nl["n"] + 1.0)
    return {**pe, "null": nl, "z": z, "p": p}
