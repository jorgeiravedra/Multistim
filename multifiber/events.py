"""
Event extraction from the DAQ log.

This module replaces the event-handling in the legacy MATLAB preprocessing
script.  Four things are done differently, each because the original is a
silent-failure path:

1. **Camera frames are reconciled, not trimmed.**  The original did
   ``frame_times = frame_times(3:end)`` with no comment and then used the
   result as a row index into a table that starts at video frame 1.  Here the
   trigger count is compared to the video frame count and any offset is
   reported with its sign and magnitude.

2. **LED state is derived twice** — from the DAQ line and from image
   brightness — and the two are compared.  The original inferred parity from a
   single frame pair and propagated it with ``repmat``, so one dropped frame
   flipped the channel assignment for the remainder of the recording with no
   indication.

3. **Laser pulse detection is explicit about its heuristics.**  Minimum pulse
   width and the double-trigger merge rule are parameters with defaults, not
   magic numbers inline.

4. **Laser ROI identity is matched by timestamp, not by row order.**  The
   original concatenated per-sweep blocks, called ``sort(pulse_ids, 2)`` (which
   sorts across columns within each row, not chronologically), and then paired
   log row *i* with detected onset *i*.  Any sweep ordering other than
   chronological permuted the ROI labels, which transposes the connectivity
   matrix.  Here the two event sequences are aligned on a fitted constant clock
   offset and matched one-to-one, with the residuals reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .matio import BcamLog

__all__ = [
    "EdgeSet",
    "detect_edges",
    "detect_camera_frames",
    "reconcile_frame_count",
    "FrameAlignment",
    "align_triggers_to_frames",
    "LedAssignment",
    "led_state_from_daq",
    "led_state_from_brightness",
    "compare_led_assignments",
    "channel_is_live",
    "detect_interleaving",
    "detect_parity_breaks",
    "gate_phase_within_frame",
    "implied_exposure",
    "LaserPulses",
    "detect_laser_pulses",
    "group_pulses_into_bursts",
    "classify_protocols",
    "infer_protocol",
    "parse_session_name",
    "protocol_matches_name",
    "LaserAssignment",
    "assign_laser_ids",
    "events_to_frames",
]


# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #

@dataclass
class EdgeSet:
    rising_idx: np.ndarray
    falling_idx: np.ndarray
    rising_t: np.ndarray
    falling_t: np.ndarray
    threshold: float
    starts_high: bool = False
    initial_included: bool = False

    @property
    def n(self) -> int:
        return int(self.rising_idx.size)


def detect_edges(
    time: np.ndarray,
    trace: np.ndarray,
    threshold: float,
    include_initial_high: bool = False,
) -> EdgeSet:
    """Rising/falling threshold crossings of a logic-like trace.

    An edge is indexed at the first sample on the new side of the threshold,
    and timestamped with that sample's DAQ time.  No interpolation: the
    quantisation is one DAQ sample, which bounds the timing resolution of every
    downstream alignment.

    A trace that is already high at sample 0 has no detectable rising edge
    there, which silently loses one event.  ``include_initial_high=True`` adds
    sample 0 as a rising edge and sets ``initial_included``; its timestamp is
    the recording start, not an observed transition, so its true onset is
    unknown and only bounded from above.  Appropriate for camera exposures
    (the frame exists regardless); not appropriate for laser pulses, where an
    unknown onset makes the event unusable for peri-stimulus alignment.
    """
    time = np.asarray(time, dtype=float)
    high = np.asarray(trace, dtype=float) > threshold
    d = np.diff(high.astype(np.int8))
    rising = np.flatnonzero(d == 1) + 1
    falling = np.flatnonzero(d == -1) + 1

    starts_high = bool(high[0]) if high.size else False
    initial_included = False
    if starts_high and include_initial_high:
        rising = np.concatenate([[0], rising])
        initial_included = True

    return EdgeSet(
        rising_idx=rising,
        falling_idx=falling,
        rising_t=time[rising],
        falling_t=time[falling],
        threshold=float(threshold),
        starts_high=starts_high,
        initial_included=initial_included,
    )


def detect_camera_frames(
    bcam: BcamLog,
    channel: str = "ham_cam",
    threshold: float = 2.0,
    include_initial_high: bool = True,
) -> EdgeSet:
    """Camera exposure triggers from the DAQ log.

    ``include_initial_high`` defaults to True: if DAQ logging began during an
    exposure, that exposure produced a video frame and must be counted, or
    every subsequent frame index is off by one.
    """
    if channel not in bcam.channels:
        raise KeyError(f"no channel {channel!r}; have {sorted(bcam.channels)}")
    return detect_edges(bcam.time, bcam.channels[channel], threshold,
                        include_initial_high=include_initial_high)


def reconcile_frame_count(
    frames: EdgeSet,
    n_video_frames: int,
    tolerance: int = 0,
) -> dict:
    """Compare DAQ trigger count to video frame count.

    Returns a report rather than modifying anything.  ``offset`` > 0 means more
    triggers than frames, i.e. the DAQ saw exposures the file does not contain
    (acquisition started/stopped asymmetrically, or frames were dropped at
    write time).  The caller decides whether to trim leading or trailing
    triggers; this function will not guess which end.
    """
    offset = frames.n - int(n_video_frames)
    dt = np.diff(frames.rising_t)
    report = {
        "n_triggers": frames.n,
        "n_video_frames": int(n_video_frames),
        "offset": offset,
        "within_tolerance": abs(offset) <= tolerance,
        "trace_starts_high": frames.starts_high,
        "first_trigger_synthesised": frames.initial_included,
        "median_interval_s": float(np.median(dt)) if dt.size else float("nan"),
        "max_interval_s": float(np.max(dt)) if dt.size else float("nan"),
        "estimated_fps": float(1.0 / np.median(dt)) if dt.size else float("nan"),
    }
    if dt.size:
        med = np.median(dt)
        # A dropped exposure shows up as an interval that is an integer
        # multiple of the nominal period.
        gaps = np.flatnonzero(dt > 1.5 * med)
        report["n_long_intervals"] = int(gaps.size)
        report["long_interval_frames"] = gaps.tolist()[:50]
    return report


@dataclass
class FrameAlignment:
    """Mapping from written video frames to DAQ exposure triggers.

    The DAQ commonly logs more triggers than the file contains: the first
    exposures after arming are not steady-state, and logging continues briefly
    after the writer stops.  The legacy code handled this with a bare
    ``frame_times(3:end)``, dropping two leading triggers with no comment and
    no check on the other end.

    The mapping is **anchored at the start**: frame 0 is taken to be the first
    steady-state trigger, found by skipping the standalone exposures that
    precede acquisition (the rig's fiber-detection and laser-centring
    snapshots, which show up as inter-trigger gaps tens of times the frame
    period).  ``trigger_idx`` is then ``lead + k``, so triggers left over at
    the *end* are simply unused and do not move any frame.

    ``n_unused_trailing`` is therefore a leftover count, not a timing error.
    The residual risk is a different one: that the writer opened after the
    train had already begun, which is invisible because those triggers sit in
    the regular train.  ``anchor_risk_ms`` bounds that risk — it is the shift
    that would result if the alignment were instead anchored at the last
    trigger — and ``confidence`` scores the leading evidence that fixes the
    anchor.
    """

    frame_t: np.ndarray
    trigger_idx: np.ndarray
    n_dropped_leading: int
    n_unused_trailing: int
    confidence: str
    anchor_risk_ms: float
    n_writer_drops: int = 0
    evidence: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.frame_t.size)

    def describe(self) -> str:
        drops = (f", {self.n_writer_drops} writer drops"
                 if self.n_writer_drops else "")
        return (f"{self.n} frames from trigger {self.trigger_idx[0]} "
                f"(skip {self.n_dropped_leading} pre-acquisition{drops}; "
                f"{self.n_unused_trailing} unused at end), "
                f"anchor={self.confidence}, risk<={self.anchor_risk_ms:.1f} ms")


def align_triggers_to_frames(
    frames: EdgeSet,
    n_video_frames: int,
    long_interval_factor: float = 1.5,
    max_leading_scan: int = 20,
    drop_after_frames: Optional[Sequence[int]] = None,
) -> FrameAlignment:
    """Decide which exposure triggers correspond to written video frames.

    Leading triggers are dropped while the inter-trigger interval is anomalous
    (> ``long_interval_factor`` x the median) — the camera settling after
    arming.  Any remaining excess is dropped from the end, on the reasoning
    that the DAQ stops logging after the file is closed.

    That second step is an assumption, not a measurement, so it is scored:

    The anchor is scored by how clearly the pre-acquisition exposures separate
    from the acquisition train:

    ``high``    the gap before the first acquisition trigger is >= 10x the
                frame period, so the start of the train is unmistakable.
    ``medium``  a gap of 1.5-10x: anomalous, but less decisive.
    ``low``     no anomalous leading interval at all, so there is nothing to
                anchor on and frame 0 is assumed to be trigger 0.

    ``drop_after_frames`` are frame indices after which an exposure is missing
    from the file, from :func:`detect_parity_breaks`.  Pass them: a frame lost
    at the writer is invisible in the trigger intervals, and without this the
    missing exposure is charged to the end of the recording and every frame
    after the drop is mapped to the wrong trigger.  Supplying them also
    removes that count from the trailing excess, which is what lets the
    trailing drop be corroborated.

    Raises if the video has more frames than the DAQ saw triggers, which means
    the trigger channel or its threshold is wrong rather than that frames
    were gained.
    """
    t = np.asarray(frames.rising_t, dtype=float)
    n_video = int(n_video_frames)
    if t.size < 2:
        raise ValueError("need at least 2 triggers to align")
    dt = np.diff(t)
    med = float(np.median(dt))

    # The leading triggers are standalone exposures taken before acquisition
    # (the getsnapshot calls in the rig's fiber-detection and laser-centring
    # steps).  Scan a window rather than requiring a contiguous run: a
    # snapshot separated from the next by a normal interval would otherwise
    # truncate the scan and leave an uncounted trigger.
    lead = 0
    for i in range(min(max_leading_scan, dt.size)):
        if dt[i] > long_interval_factor * med:
            lead = i + 1

    drops = np.asarray(sorted(set(int(d) for d in (drop_after_frames or []))),
                       dtype=int)
    if drops.size > 0.05 * max(n_video, 1):
        raise ValueError(
            f"{drops.size} writer-side drops were reported for {n_video} "
            "video frames. A drop rate this high is not a drop rate: it means "
            "the LED parity test was run on a recording that does not "
            "alternate. Check events.detect_interleaving() before using "
            "detect_parity_breaks()."
        )
    remaining = t.size - lead - drops.size
    if remaining < n_video:
        raise ValueError(
            f"{n_video} video frames but only {remaining} usable triggers "
            f"({t.size} total, {lead} dropped as settling, {drops.size} as "
            "writer-side drops). The trigger channel or its threshold is "
            "wrong — frames cannot be gained."
        )
    trail = remaining - n_video

    # Score the anchor by the size of the gap that identifies the start of the
    # acquisition train, not by the trailing leftover (which moves nothing).
    lead_gap_ratio = float(dt[lead - 1] / med) if lead > 0 else 0.0
    if lead_gap_ratio >= 10.0:
        confidence = "high"
    elif lead_gap_ratio >= long_interval_factor:
        confidence = "medium"
    else:
        confidence = "low"
    anchor_risk = trail * med * 1e3

    # Frame k maps to trigger lead + k + (number of drops at or before k).
    idx = lead + np.arange(n_video)
    if drops.size:
        # 'drop_after_frames' names the frame the gap follows, so frame d
        # itself is unshifted and frame d+1 is the first to skip a trigger;
        # side='left' is what gives that boundary.
        idx = idx + np.searchsorted(drops, np.arange(n_video), side="left")
    return FrameAlignment(
        frame_t=t[idx],
        trigger_idx=idx,
        n_dropped_leading=lead,
        n_unused_trailing=trail,
        confidence=confidence,
        anchor_risk_ms=anchor_risk,
        n_writer_drops=int(drops.size),
        evidence={
            "n_triggers": int(t.size),
            "n_video_frames": n_video,
            "median_interval_s": med,
            "leading_intervals_s": dt[:max(lead + 2, 3)].tolist(),
            "trailing_intervals_s": dt[-3:].tolist(),
            "lead_gap_ratio": lead_gap_ratio,
            "long_interval_factor": long_interval_factor,
            "n_writer_drops": int(drops.size),
            "drop_after_frames": drops.tolist()[:50],
        },
    )


# --------------------------------------------------------------------------- #
# LED state
# --------------------------------------------------------------------------- #

@dataclass
class LedAssignment:
    """Per-frame LED/channel assignment from one source."""

    source: str
    state: np.ndarray            # integer cluster id per frame
    values: np.ndarray           # the underlying measurement per frame
    n_states: int
    separation: float            # d-prime: state gap / within-state noise
    details: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.state.size)


def _cluster_1d(values: np.ndarray, k: int, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 1-D k-means seeded at evenly spaced quantiles.

    Deterministic seeding matters: a random-seeded clusterer would relabel
    channels between runs of the same data.
    """
    v = np.asarray(values, dtype=float)
    centres = np.quantile(v[np.isfinite(v)],
                          np.linspace(0.5 / k, 1 - 0.5 / k, k))
    centres = np.sort(centres)
    labels = np.zeros(v.size, dtype=int)
    for _ in range(iters):
        labels = np.argmin(np.abs(v[:, None] - centres[None, :]), axis=1)
        new = np.array([
            v[labels == j].mean() if np.any(labels == j) else centres[j]
            for j in range(k)
        ])
        new = np.sort(new)
        if np.allclose(new, centres):
            break
        centres = new
    labels = np.argmin(np.abs(v[:, None] - centres[None, :]), axis=1)
    return labels, centres


def _separation(values: np.ndarray, labels: np.ndarray, centres: np.ndarray) -> float:
    """Smallest between-centre gap divided by the largest within-cluster SD."""
    if centres.size < 2:
        return float("inf")
    gap = float(np.min(np.diff(np.sort(centres))))
    sds = [float(np.std(values[labels == j])) for j in range(centres.size)
           if np.any(labels == j)]
    worst = max(sds) if sds else 0.0
    return gap / worst if worst > 0 else float("inf")


def led_state_from_daq(
    bcam: BcamLog,
    frames: EdgeSet,
    channel: str = "led",
    n_states: int = 2,
    sample: str = "at_edge",
    trigger_idx: Optional[np.ndarray] = None,
) -> LedAssignment:
    """Per-frame LED state read from the DAQ LED line.

    ``trigger_idx`` selects which triggers correspond to written video frames,
    from :class:`FrameAlignment`.  Pass it whenever the result will be
    compared against an image-derived assignment: the DAQ sees every exposure,
    including the pre-acquisition snapshots, so comparing trigger *k* against
    video frame *k* offsets the two sequences and the comparison is
    meaningless.

    ``sample='at_edge'`` takes the LED value at the exposure trigger;
    ``sample='exposure_mean'`` averages over each inter-trigger interval, which
    is more robust if the LED line settles after the trigger but blurs the
    boundary if it settles late.
    """
    if channel not in bcam.channels:
        raise KeyError(f"no channel {channel!r}; have {sorted(bcam.channels)}")
    led = bcam.channels[channel]

    sel = (np.arange(frames.n) if trigger_idx is None
           else np.asarray(trigger_idx, dtype=int))
    if sample == "at_edge":
        vals = led[frames.rising_idx[sel]]
    elif sample == "exposure_mean":
        edges = np.append(frames.rising_idx, led.size)
        vals = np.array([led[edges[i]:edges[i + 1]].mean() for i in sel])
    else:
        raise ValueError(f"unknown sample mode {sample!r}")

    labels, centres = _cluster_1d(vals, n_states)
    return LedAssignment(
        source=f"daq:{channel}:{sample}",
        state=labels,
        values=vals,
        n_states=n_states,
        separation=_separation(vals, labels, centres),
        details={"centres": centres.tolist()},
    )


def led_state_from_brightness(
    frame_intensity: np.ndarray,
    n_states: int = 2,
    mode: str = "relative",
) -> LedAssignment:
    """Per-frame LED state inferred from image brightness.

    ``mode='relative'`` classifies each frame against the mean of its immediate
    neighbours, which tracks the interleave through photobleaching and does not
    break at a dropped frame (it produces a single misclassified frame rather
    than flipping every subsequent frame, which is what a fixed parity
    assignment does).

    ``mode='absolute'`` clusters raw intensity, which is only safe over short
    recordings with little bleaching.

    ``frame_intensity`` is one scalar per video frame, e.g. the mean over all
    ROI pixels.  Use the same pixel set for every frame.
    """
    x = np.asarray(frame_intensity, dtype=float)
    if x.ndim != 1:
        raise ValueError("frame_intensity must be 1-D, one value per frame")

    if mode == "absolute":
        labels, centres = _cluster_1d(x, n_states)
        return LedAssignment(
            source="image:absolute",
            state=labels,
            values=x,
            n_states=n_states,
            separation=_separation(x, labels, centres),
            details={"centres": centres.tolist()},
        )

    if mode != "relative":
        raise ValueError(f"unknown mode {mode!r}")
    if n_states != 2:
        raise ValueError("relative mode is defined for 2 interleaved states only")

    neigh = np.empty_like(x)
    neigh[1:-1] = 0.5 * (x[:-2] + x[2:])
    neigh[0] = x[1] if x.size > 1 else x[0]
    neigh[-1] = x[-2] if x.size > 1 else x[-1]
    contrast = x - neigh
    labels = (contrast > 0).astype(int)  # 1 = brighter than neighbours

    sep = _state_separation(contrast, labels)
    return LedAssignment(
        source="image:relative",
        state=labels,
        values=contrast,
        n_states=2,
        separation=sep,
        details={"note": "state 1 = brighter than neighbouring frames"},
    )


def channel_is_live(
    x: np.ndarray,
    min_span: float = 0.05,
    percentiles: tuple[float, float] = (0.5, 99.5),
) -> bool:
    """Whether a channel carries signal, judged on a robust span.

    Peak-to-peak is outlier-sensitive: a single spike lifts a dead channel
    above any fixed threshold, after which clustering its noise into two
    states produces labels that agree with anything at chance.  A percentile
    span ignores the spike.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return False
    lo, hi = np.percentile(x, percentiles)
    return bool(hi - lo >= min_span)


def _state_separation(values: np.ndarray, labels: np.ndarray) -> float:
    """Separation between two labelled states, in within-state noise units.

    ``|median(state 1) - median(state 0)| / pooled within-state MAD``, scaled
    so that the denominator estimates a standard deviation for Gaussian noise.
    This is a d-prime: it grows without bound as the states get cleaner.

    The obvious alternative — dividing by the MAD of the *pooled* values — is
    wrong and was what this function did first.  For a clean alternation the
    pooled MAD equals the half-separation, so the ratio pins to exactly 2 no
    matter how good the data are, and rises above 2 only when frames with
    near-zero contrast drag the pooled MAD down.  It was therefore largest
    for the worst recordings.
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    a, b = values[labels == 0], values[labels == 1]
    if a.size < 2 or b.size < 2:
        return float("nan")
    gap = float(abs(np.median(b) - np.median(a)))
    mads = [float(np.median(np.abs(g - np.median(g)))) for g in (a, b)]
    scale = 1.4826 * float(np.sqrt(np.mean(np.square(mads))))
    if scale <= 0:
        return float("inf")
    return gap / scale


def _cohen_kappa(a: np.ndarray, b: np.ndarray, k: int) -> float:
    n = a.size
    if n == 0:
        return float("nan")
    cm = np.zeros((k, k), dtype=float)
    np.add.at(cm, (a, b), 1.0)
    po = np.trace(cm) / n
    pe = float((cm.sum(0) / n) @ (cm.sum(1) / n))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def compare_led_assignments(
    primary: LedAssignment,
    secondary: LedAssignment,
    align_window: int = 500,
    flip_threshold: float = 0.98,
) -> dict:
    """Agreement between two LED assignments, after resolving label identity.

    The two sources label their clusters independently (a DAQ voltage level has
    no intrinsic colour), so the secondary labels must be permuted before
    anything is scored.  The permutation is resolved on the first
    ``align_window`` frames, **not** globally: a source that flips parity
    mid-recording sits at ~50% global agreement, where a global vote is a coin
    toss and can hide the flip by relabelling it onto the first half instead.
    The start of the recording is the part most likely to be correct.

    A single parity flip is then tested explicitly.  For each candidate
    changepoint *k*, the "one flip at k" model predicts agreement before k and
    disagreement after; the best-fitting *k* and its accuracy are returned.  An
    accuracy above ``flip_threshold`` with ``k`` inside the recording means the
    two sources diverge at a point, which is the dropped-frame signature.
    Scattered disagreement produces a low flip-model accuracy instead.
    """
    a = np.asarray(primary.state, dtype=int)
    b = np.asarray(secondary.state, dtype=int)
    n = min(a.size, b.size)
    trimmed = abs(a.size - b.size)
    a, b = a[:n], b[:n]

    k_states = max(primary.n_states, secondary.n_states)
    if k_states != 2:
        raise NotImplementedError(
            "label matching is implemented for 2 states; extend with the "
            "Hungarian algorithm for >2 interleaved LEDs"
        )
    if n == 0:
        raise ValueError("no frames to compare")

    w = int(min(max(align_window, 1), n))
    if float(np.mean(a[:w] == b[:w])) >= 0.5:
        b_aligned, perm = b, {0: 0, 1: 1}
    else:
        b_aligned, perm = 1 - b, {0: 1, 1: 0}

    mismatch = a != b_aligned
    idx = np.flatnonzero(mismatch)

    # Single-changepoint flip model: agree on [0, k), disagree on [k, n).
    # correct(k) = (k - mismatches before k) + (mismatches at or after k)
    cum = np.concatenate([[0], np.cumsum(mismatch)])        # cum[k] = # mismatch < k
    total = int(cum[-1])
    kk = np.arange(n + 1)
    correct = (kk - cum) + (total - cum)
    best_k = int(np.argmax(correct))
    flip_acc = float(correct[best_k] / n)

    return {
        "n_compared": int(n),
        "n_trimmed": int(trimmed),
        "align_window": w,
        "label_permutation": perm,
        "agreement": float(np.mean(~mismatch)),
        "cohen_kappa": _cohen_kappa(a, b_aligned, 2),
        "n_mismatch": int(idx.size),
        "first_mismatch_frame": int(idx[0]) if idx.size else None,
        "flip_model_accuracy": flip_acc,
        "flip_model_changepoint": best_k if best_k < n else None,
        "suspected_parity_flip": bool(best_k < n and flip_acc >= flip_threshold),
        "primary": primary.source,
        "secondary": secondary.source,
        "primary_separation": primary.separation,
        "secondary_separation": secondary.separation,
    }


def gate_phase_within_frame(
    frame_t: np.ndarray,
    gate_on_t: np.ndarray,
    exposure_s: Optional[float] = None,
) -> dict:
    """Where in each frame period the laser gate fires.

    With the camera-synchronous chop, the gate is meant to fire in the dead
    time after the sensor stops integrating.  This measures whether it
    actually does, rather than taking the GUI's interlock checkbox on trust:
    the phase of every gate pulse relative to the preceding exposure trigger
    should cluster tightly above ``exposure_s``.

    A tight cluster inside the gap confirms the interlock.  A phase spread
    uniformly across the frame period means the gate is not synchronised to
    exposures at all, and the stimulus frames are genuinely contaminated.

    ``margin_ms`` is how far the earliest gate pulse sits after the end of
    integration; it is the quantity to read, because a gate that hugs the
    boundary is correct but leaves no room for error in ``exposure_s``.
    """
    frame_t = np.asarray(frame_t, dtype=float).ravel()
    gate_on_t = np.asarray(gate_on_t, dtype=float).ravel()
    if frame_t.size < 2 or gate_on_t.size == 0:
        return {"n_pulses": int(gate_on_t.size), "period_s": float("nan")}
    period = float(np.median(np.diff(frame_t)))

    idx = np.searchsorted(frame_t, gate_on_t, side="right") - 1
    ok = (idx >= 0) & (idx < frame_t.size)
    phase = gate_on_t[ok] - frame_t[idx[ok]]
    phase = phase[phase <= period * 1.5]
    if phase.size == 0:
        return {"n_pulses": 0, "period_s": period}

    q = np.percentile(phase, [0, 1, 50, 99, 100])
    out = {
        "n_pulses": int(phase.size),
        "period_s": period,
        "phase_ms": {"min": q[0] * 1e3, "p1": q[1] * 1e3, "median": q[2] * 1e3,
                     "p99": q[3] * 1e3, "max": q[4] * 1e3},
        "spread_ms": float((q[3] - q[1]) * 1e3),
        "uniform_fraction": float((q[3] - q[1]) / period),
    }
    if exposure_s is not None:
        exposure_s = float(exposure_s)
        # A pulse starting exactly at the end of integration does not overlap
        # it, so the comparison is strict.  exposure_s is itself a measured
        # figure with its own uncertainty, so a tolerance of one DAQ sample at
        # 2 kHz keeps a boundary-hugging gate from being called an overlap.
        tol = 0.0005
        inside = float(np.mean(phase < exposure_s - tol))
        margin_ms = float((np.percentile(phase, 1) - exposure_s) * 1e3)
        out["exposure_s"] = exposure_s
        out["tolerance_s"] = tol
        out["gap_ms"] = (period - exposure_s) * 1e3
        out["margin_ms"] = margin_ms          # earliest pulse minus exposure end
        out["fraction_during_exposure"] = inside
        out["interlock_holds"] = bool(inside < 0.01)
    return out


def detect_interleaving(
    frame_intensity: np.ndarray,
    threshold: float = -0.3,
) -> dict:
    """Whether successive frames alternate between two excitation wavelengths.

    Not every recording is interleaved: the rig runs some sessions at a single
    wavelength, with no isosbestic channel at all (the legacy MATLAB sets
    ``led_rep = [1 1]`` for those).  Everything downstream that assumes an
    alternation — the LED state, the parity-break test, the isosbestic
    regression — produces nonsense on such a file rather than failing, so the
    question has to be asked before any of it runs.

    The test is the lag-1 autocorrelation of the detrended per-frame
    intensity.  A strict alternation drives it toward -1; a single-wavelength
    recording has slow fluorescence dynamics and gives a value at or above
    zero.  It is a one-number summary of exactly the structure that
    interleaving creates.
    """
    x = np.asarray(frame_intensity, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 10:
        return {"is_interleaved": False, "lag1_autocorr": float("nan"),
                "n_frames": int(x.size), "reason": "too few frames"}

    # Remove slow structure (bleaching) so the test sees frame-to-frame
    # alternation and not the decay.
    k = min(101, (x.size // 4) * 2 + 1)
    pad = k // 2
    trend = np.convolve(np.pad(x, pad, mode="edge"), np.ones(k) / k,
                        mode="valid")
    v = x - trend[:x.size]
    denom = float(np.dot(v, v))
    r1 = float(np.dot(v[:-1], v[1:]) / denom) if denom > 0 else float("nan")

    alt_frac = float(np.mean(np.sign(v[:-1]) != np.sign(v[1:])))
    return {
        "is_interleaved": bool(np.isfinite(r1) and r1 < threshold),
        "lag1_autocorr": r1,
        "sign_alternation_fraction": alt_frac,
        "threshold": threshold,
        "n_frames": int(x.size),
    }


def implied_exposure(
    frame_t: np.ndarray,
    gate_on_t: np.ndarray,
    gate_off_t: Optional[np.ndarray] = None,
) -> dict:
    """Infer the exposure window from the gate, rather than assuming it.

    If the between-frames interlock works, the gate fires in the dead time and
    nowhere else.  The earliest gate onset within a frame period is then the
    end of integration, and the latest gate offset is the start of the next
    exposure.  That gives an exposure time measured from the data, which can
    be compared against the configured one.

    This also exposes a failure that ``exposure_s`` cannot: ``frame_t`` is the
    camera channel's rising edge, which this rig may configure as a
    *programmable output* pulse with its own delay, not as the start of
    integration.  If so, every phase is offset by that delay.  The signature
    is an implied exposure that disagrees with the configured one by a
    constant, or a gate phase that does not sit flush against a boundary.
    """
    frame_t = np.asarray(frame_t, dtype=float).ravel()
    gate_on_t = np.asarray(gate_on_t, dtype=float).ravel()
    if frame_t.size < 2 or gate_on_t.size == 0:
        return {"n_pulses": int(gate_on_t.size)}
    period = float(np.median(np.diff(frame_t)))

    idx = np.searchsorted(frame_t, gate_on_t, side="right") - 1
    ok = (idx >= 0) & (idx < frame_t.size)
    phase = gate_on_t[ok] - frame_t[idx[ok]]
    phase = phase[(phase >= 0) & (phase <= period)]
    if phase.size < 5:
        return {"n_pulses": int(phase.size), "period_s": period}

    p1 = float(np.percentile(phase, 1))
    p99 = float(np.percentile(phase, 99))
    out = {
        "n_pulses": int(phase.size),
        "period_s": period,
        "earliest_onset_s": p1,
        "latest_onset_s": p99,
        "implied_exposure_s": p1,
        "implied_gap_s": period - p1,
        "onset_spread_s": p99 - p1,
    }
    if gate_off_t is not None and len(gate_off_t):
        off = np.asarray(gate_off_t, dtype=float).ravel()
        j = np.searchsorted(frame_t, off, side="right") - 1
        okj = (j >= 0) & (j < frame_t.size)
        ph_off = off[okj] - frame_t[j[okj]]
        ph_off = ph_off[(ph_off >= 0) & (ph_off <= period)]
        if ph_off.size:
            out["latest_offset_s"] = float(np.percentile(ph_off, 99))
            out["gate_width_s"] = float(np.median(ph_off) - np.median(phase))
    return out


def detect_parity_breaks(
    led: "LedAssignment",
    min_run: int = 2,
) -> dict:
    """Find breaks in the LED alternation, which mark frames lost at the writer.

    With two LEDs interleaved one per exposure, the recorded sequence must
    alternate strictly.  Two consecutive frames in the same state means an odd
    number of exposures between them is missing from the *file* — the camera
    fired (the DAQ logged the trigger) but the frame was never written.

    This is the only way to see such a drop.  The inter-trigger interval stays
    a clean 1/fps because nothing went wrong on the DAQ side, so interval
    analysis cannot detect it, and a count-based alignment will silently
    attribute the missing frame to the end of the recording and shift every
    frame after the drop by one.

    Returns the frame indices *after* which a drop occurred, plus the run
    structure for inspection.  An even number of consecutive dropped frames is
    invisible to this test, so the count is a lower bound.
    """
    state = np.asarray(led.state, dtype=int)
    if state.size < 2:
        return {"break_after": [], "n_breaks": 0, "runs": [], "lower_bound": True}

    same = state[1:] == state[:-1]
    idx = np.flatnonzero(same)

    runs, start = [], None
    for i in range(state.size):
        if start is None:
            start = i
        elif state[i] != state[i - 1]:
            if i - start >= min_run:
                runs.append((start, i - start))
            start = i
    if start is not None and state.size - start >= min_run:
        runs.append((start, state.size - start))

    return {
        "break_after": idx.tolist(),
        "n_breaks": int(idx.size),
        "runs": runs,
        "lower_bound": True,
        "note": "an even number of consecutive dropped frames is invisible here",
    }


# --------------------------------------------------------------------------- #
# Laser pulses
# --------------------------------------------------------------------------- #

@dataclass
class LaserPulses:
    """Commanded stimulation events, and how each was actually delivered.

    One row per *stimulation*, defined by the analog command envelope — not
    per laser-on transition.  With the camera-synchronous chop enabled, a
    single 1 s command produces ~80 gate pulses; counting gate edges counts
    chops, and would turn 10 stimulations into 800.
    """

    onset_t: np.ndarray
    offset_t: np.ndarray
    width_s: np.ndarray
    amplitude_v: np.ndarray          # mean command voltage within the envelope
    n_gate_pulses: np.ndarray        # gate transitions inside each envelope
    gate_duty: np.ndarray            # fraction of the envelope the gate was on
    gate_on_s: np.ndarray            # total gate-on time per envelope
    dropped_short: int = 0
    dropped_doubles: int = 0
    source: str = "envelope"
    params: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.onset_t.size)

    def describe(self) -> str:
        if self.n == 0:
            return "no laser events detected"
        lines = [f"{self.n} stimulation events ({self.source})",
                 f"  width   {np.median(self.width_s) * 1e3:.1f} ms "
                 f"[{self.width_s.min() * 1e3:.1f}-{self.width_s.max() * 1e3:.1f}]",
                 f"  command {np.median(self.amplitude_v):.4f} V "
                 f"[{self.amplitude_v.min():.4f}-{self.amplitude_v.max():.4f}]"]
        if np.any(self.n_gate_pulses > 0):
            lines.append(
                f"  gate    {np.median(self.n_gate_pulses):.0f} pulses/event, "
                f"duty {np.median(self.gate_duty):.1%} "
                f"({np.median(self.gate_on_s) * 1e3:.0f} ms on)")
        if self.dropped_short or self.dropped_doubles:
            lines.append(f"  dropped {self.dropped_short} short, "
                         f"{self.dropped_doubles} double-triggers")
        return "\n".join(lines)


def detect_laser_pulses(
    bcam: BcamLog,
    command_channel: str = "laser_cmd",
    gate_channel: Optional[str] = "laser_gate",
    threshold: float = 0.07,
    gate_threshold: float = 0.5,
    min_width_s: float = 0.0015,
    double_gap_s: float = 0.0,
    resolve_doubles: str = "keep_longer",
) -> LaserPulses:
    """Detect stimulation events from the laser analog command.

    Parameters
    ----------
    command_channel
        The analog command voltage. Its rising edge is the commanded onset and
        its level encodes power, so it defines the stimulation event. The
        default threshold of 0.07 V sits above the ~3 mV noise floor and below
        the lowest per-region command in ``intensity_mat`` (0.12 V at 1.6 mW),
        which is why the legacy code used it.
    gate_channel
        The digital line that fires the laser, chopped at the camera rate when
        "LASER BETWEEN CAMERA FRAMES ONLY" is on. Used to *characterise* each
        event (chop count, duty cycle, total on-time), never to define events.
        Pass None if the channel is unconnected.
    min_width_s
        Commands shorter than this are discarded as driver glitches. The
        legacy 1.5 ms is 30% of a nominal 5 ms pulse; check
        ``describe()``'s width range before trusting it on short protocols.
    double_gap_s
        Merge rule for a command that double-triggers. Defaults to 0 (off):
        with the gate channel carrying the chopping, the command envelope is
        clean, and a non-zero value here would merge genuine closely spaced
        stimulations. Set it only if ``describe()`` shows spurious pairs.
    """
    if command_channel not in bcam.channels:
        raise KeyError(f"no channel {command_channel!r}; "
                       f"have {sorted(bcam.channels)}")
    cmd = np.asarray(bcam.channels[command_channel], dtype=float)
    span = float(np.nanmax(cmd) - np.nanmin(cmd))
    if span < threshold:
        raise ValueError(
            f"{command_channel}: full span is {span * 1e3:.1f} mV, below the "
            f"{threshold} V threshold — the channel is unconnected or the "
            "channel map is wrong. Run matio.load_bcam_raw_profile() to check."
        )

    edges = detect_edges(bcam.time, cmd, threshold)
    on, off = edges.rising_t, edges.falling_t
    if on.size and off.size and off[0] < on[0]:
        off = off[1:]
    n = min(on.size, off.size)
    on, off = on[:n], off[:n]
    width = off - on

    keep = width >= min_width_s
    dropped_short = int((~keep).sum())
    on, off, width = on[keep], off[keep], width[keep]

    dropped_doubles = 0
    if double_gap_s > 0 and on.size > 1:
        gaps = on[1:] - off[:-1]
        pair_idx = np.flatnonzero(gaps < double_gap_s)
        drop = np.zeros(on.size, dtype=bool)
        for i in pair_idx:
            if drop[i] or drop[i + 1]:
                continue
            if resolve_doubles == "keep_longer":
                loser = i if width[i] < width[i + 1] else i + 1
            elif resolve_doubles == "keep_first":
                loser = i + 1
            elif resolve_doubles == "keep_last":
                loser = i
            else:
                raise ValueError(f"unknown resolve_doubles {resolve_doubles!r}")
            drop[loser] = True
        dropped_doubles = int(drop.sum())
        on, off, width = on[~drop], off[~drop], width[~drop]

    # ---- per-event command amplitude and gate structure ------------------- #
    t = bcam.time
    i0 = np.searchsorted(t, on, side="left")
    # 'off' is the timestamp of the first LOW sample, so the envelope is the
    # half-open interval [i0, i1); 'right' here would include that low sample
    # and bias the mean command voltage downward by one sample's worth.
    i1 = np.searchsorted(t, off, side="left")
    amplitude = np.array([float(np.mean(cmd[a:b])) if b > a else np.nan
                          for a, b in zip(i0, i1)])

    n_gate = np.zeros(on.size, dtype=int)
    duty = np.full(on.size, np.nan)
    gate_on = np.full(on.size, np.nan)
    gate_used = None
    if gate_channel and gate_channel in bcam.channels:
        g = np.asarray(bcam.channels[gate_channel], dtype=float)
        if float(np.nanmax(g) - np.nanmin(g)) >= gate_threshold:
            gate_used = gate_channel
            high = g > gate_threshold
            rise = np.flatnonzero(np.diff(high.astype(np.int8)) == 1) + 1
            dt = 1.0 / bcam.rate_hz
            for k, (a, b) in enumerate(zip(i0, i1)):
                if b <= a:
                    continue
                n_gate[k] = int(np.count_nonzero((rise >= a) & (rise < b)))
                seg = high[a:b]
                duty[k] = float(seg.mean())
                gate_on[k] = float(seg.sum() * dt)

    return LaserPulses(
        onset_t=on,
        offset_t=off,
        width_s=width,
        amplitude_v=amplitude,
        n_gate_pulses=n_gate,
        gate_duty=duty,
        gate_on_s=gate_on,
        dropped_short=dropped_short,
        dropped_doubles=dropped_doubles,
        source=f"envelope:{command_channel}",
        params={
            "command_channel": command_channel,
            "gate_channel": gate_used,
            "threshold": threshold,
            "gate_threshold": gate_threshold,
            "min_width_s": min_width_s,
            "double_gap_s": double_gap_s,
            "resolve_doubles": resolve_doubles if double_gap_s > 0 else None,
            "daq_rate_hz": bcam.rate_hz,
            "timing_resolution_s": 1.0 / bcam.rate_hz,
        },
    )


# --------------------------------------------------------------------------- #
# Laser -> ROI assignment
# --------------------------------------------------------------------------- #

def group_pulses_into_bursts(
    pulses: "LaserPulses",
    max_gap_s: Optional[float] = None,
    gap_factor: float = 3.0,
) -> "LaserPulses":
    """Collapse the pulses of a burst protocol into one event per burst.

    A 10 Hz burst delivers pulses 100 ms apart.  Treated as separate events
    they are not independent trials: aligning to each one puts the
    neighbouring pulses of the same burst inside the peri-event window, so the
    "baseline" contains stimulation and the trial mean carries a comb at the
    burst rate.  Counting them as n also inflates the trial count by the
    number of pulses per burst, which is pseudoreplication -- the unit that
    was randomised is the burst.

    ``max_gap_s`` is the longest inter-pulse interval still considered part of
    the same burst.  Left as None it is inferred: the inter-pulse intervals of
    a burst protocol are strongly bimodal (within-burst vs between-burst), and
    the threshold is placed at ``gap_factor`` times the median within-burst
    interval.  A protocol with no bimodality is returned unchanged.

    The returned events span first onset to last offset of each burst;
    ``n_gate_pulses`` becomes the pulse count, and ``amplitude_v`` the mean
    command voltage over the burst.
    """
    on, off = np.asarray(pulses.onset_t), np.asarray(pulses.offset_t)
    if on.size < 3:
        return pulses

    gaps = on[1:] - off[:-1]
    if max_gap_s is None:
        med = float(np.median(gaps))
        big = gaps[gaps > gap_factor * med]
        if big.size == 0 or med <= 0:
            return pulses               # not a burst protocol
        max_gap_s = gap_factor * med

    starts = np.concatenate([[0], np.flatnonzero(gaps > max_gap_s) + 1])
    stops = np.concatenate([starts[1:], [on.size]])
    if starts.size == on.size:
        return pulses                   # every pulse already its own burst

    n_b = starts.size
    b_on = on[starts]
    b_off = off[stops - 1]
    amp = np.array([float(np.nanmean(pulses.amplitude_v[a:b]))
                    for a, b in zip(starts, stops)])
    npulse = (stops - starts).astype(int)
    gate = np.array([int(np.nansum(pulses.n_gate_pulses[a:b]))
                     for a, b in zip(starts, stops)])
    gon = np.array([float(np.nansum(pulses.gate_on_s[a:b]))
                    for a, b in zip(starts, stops)])
    width = b_off - b_on

    out = LaserPulses(
        onset_t=b_on, offset_t=b_off, width_s=width, amplitude_v=amp,
        n_gate_pulses=gate, gate_duty=np.where(width > 0, gon / width, np.nan),
        gate_on_s=gon,
        dropped_short=pulses.dropped_short,
        dropped_doubles=pulses.dropped_doubles,
        source=f"burst({pulses.source})",
        params=dict(pulses.params),
    )
    out.params.update({
        "grouped_into_bursts": True,
        "max_gap_s": float(max_gap_s),
        "n_pulses_before": int(on.size),
        "n_bursts": int(n_b),
        "pulses_per_burst": {"min": int(npulse.min()),
                             "median": float(np.median(npulse)),
                             "max": int(npulse.max())},
        "within_burst_rate_hz": (float(np.median(npulse - 1)
                                       / np.median(width))
                                 if np.median(width) > 0 else None),
    })
    return out


def classify_protocols(
    events: "LaserPulses",
    roi_id: Optional[np.ndarray] = None,
    tol: float = 0.15,
    min_events: int = 3,
    split_on_power: bool = True,
) -> np.ndarray:
    """Label each event by which protocol delivered it.

    A session may interleave several protocols -- different train durations,
    rates or powers in one recording, which is the design that removes
    between-session confounds from a dose-response.  Summarising such a
    session with one set of protocol parameters averages across conditions
    and describes none of them.

    Events are grouped on commanded duration, and then on command voltage
    **within a stimulated site**.  The voltage is not comparable across sites:
    it encodes which fiber is being driven, taken from ``intensity_mat``,
    where each region has its own value for the same delivered power.
    Splitting on raw voltage therefore separates sites rather than
    conditions, turning three interleaved durations across twelve fibers into
    dozens of spurious classes.  ``roi_id`` is required for a power split;
    without it, only duration is used.

    Durations within ``tol`` (fractional) of each other are one class, and
    classes with fewer than ``min_events`` members are merged into the
    nearest larger class, since a handful of stray events is more likely to
    be detection noise than a condition.

    Returns an integer label per event, ordered by increasing duration.
    """
    w = np.asarray(events.width_s, dtype=float)
    v = np.asarray(events.amplitude_v, dtype=float)
    n = w.size
    if n == 0:
        return np.zeros(0, dtype=int)

    order = np.argsort(w)
    labels = np.full(n, -1, dtype=int)
    cur, ref = 0, w[order[0]]
    for i in order:
        if ref > 0 and abs(w[i] - ref) / ref > tol:
            cur += 1
            ref = w[i]
        labels[i] = cur

    # Split further on power. The raw command voltage is not comparable
    # across sites -- it encodes which fiber is driven, from intensity_mat,
    # where each region has its own value for the same delivered power -- so
    # it is first normalised by that site's own median. A relative voltage
    # near 1 is "this site's usual power"; a distinct cluster away from 1 is
    # a genuine power condition, and is the same condition at every site.
    out = labels.copy()
    if split_on_power and roi_id is not None and np.isfinite(v).any():
        rid = np.asarray(roi_id, dtype=int)
        v_rel = np.full(n, np.nan)
        for r in np.unique(rid):
            m = rid == r
            # Reference on a LOW quantile, not the median: where a site
            # received two power levels the median falls between them, and
            # every event then gets a ratio away from 1, splitting one
            # condition into two.
            ref = np.nanpercentile(v[m], 10) if np.isfinite(v[m]).any() else np.nan
            if np.isfinite(ref) and ref > 0:
                v_rel[m] = v[m] / ref
        if np.isfinite(v_rel).any():
            spread = np.nanmax(v_rel) - np.nanmin(v_rel)
            if spread > tol:
                nxt = labels.max() + 1
                for c in np.unique(labels):
                    m = labels == c
                    vv = v_rel[m]
                    if not np.isfinite(vv).any():
                        continue
                    order_v = np.argsort(np.where(np.isfinite(vv), vv, np.inf))
                    idxs = np.flatnonzero(m)[order_v]
                    ref, grp = v_rel[idxs[0]], 0
                    sub = np.zeros(idxs.size, dtype=int)
                    for k, i in enumerate(idxs):
                        if np.isfinite(ref) and ref > 0 and \
                                abs(v_rel[i] - ref) / ref > tol:
                            grp += 1
                            ref = v_rel[i]
                        sub[k] = grp
                    for g in range(1, sub.max() + 1):
                        out[idxs[sub == g]] = nxt
                        nxt += 1

    # merge classes too small to be a condition
    for c in np.unique(out):
        m = out == c
        if m.sum() >= min_events:
            continue
        others = [k for k in np.unique(out) if k != c and (out == k).sum() >= min_events]
        if not others:
            continue
        tgt = min(others,
                  key=lambda k: abs(np.median(w[out == k]) - np.median(w[m])))
        out[m] = tgt

    # relabel in order of increasing duration
    uniq = sorted(np.unique(out), key=lambda c: np.median(w[out == c]))
    remap = {c: i for i, c in enumerate(uniq)}
    return np.array([remap[c] for c in out], dtype=int)


def infer_protocol(
    pulses: "LaserPulses",
    bursts: Optional["LaserPulses"] = None,
    roi_id: Optional[np.ndarray] = None,
    intensity_mat: Optional[np.ndarray] = None,
    calibrated_mw: float = 1.6,
) -> dict:
    """Derive the stimulation protocol from the recorded laser channel.

    Everything here is measured, not declared.  Scripts that read a burst
    frequency out of the session name are trusting a filename: a session
    called ``10Hzburst`` whose protocol was changed at the rig, or renamed by
    hand, will silently drive a frequency-dependent analysis with the wrong
    number.  The inferred values are compared against the name by
    :func:`protocol_matches_name`.

    ``intensity_mat`` is the ``lsr_ints`` calibration, (n_fiber, 2) of
    [roi_id, volts] for ``calibrated_mw``.  Supplied, the commanded voltage of
    each event is converted to power, which is the quantity a reader needs and
    the one that is never recorded anywhere else.  The conversion assumes the
    laser's voltage-to-power curve is linear through the origin over this
    range; it is a cross-check against the calibration, not a substitute for
    a power meter at the fiber tip.
    """
    out: dict = {"n_pulses": int(pulses.n)}
    if pulses.n == 0:
        return out

    b_for_class = bursts if bursts is not None else pulses
    classes = classify_protocols(b_for_class, roi_id=roi_id)
    n_classes = int(np.unique(classes).size) if classes.size else 0
    out["n_protocol_classes"] = n_classes
    if n_classes > 1:
        # Report each condition separately; a single summary across
        # interleaved protocols describes none of them.
        per = []
        for c in np.unique(classes):
            m = classes == c
            per.append({
                "class": int(c),
                "n_events": int(m.sum()),
                "duration_s": float(np.median(b_for_class.width_s[m])),
                "command_v": float(np.nanmedian(b_for_class.amplitude_v[m])),
                "gate_duty": float(np.nanmedian(b_for_class.gate_duty[m]))
                if np.isfinite(b_for_class.gate_duty[m]).any() else None,
            })
        out["protocol_classes"] = per
        out["interleaved"] = True

    w = np.asarray(pulses.width_s, dtype=float)
    out["pulse_width_s"] = {"median": float(np.median(w)),
                            "min": float(w.min()), "max": float(w.max())}

    b = bursts if bursts is not None else pulses
    grouped = bool(b.params.get("grouped_into_bursts"))
    out["grouped_into_bursts"] = grouped
    out["n_events"] = int(b.n)

    if grouped:
        ppb = b.params["pulses_per_burst"]
        out["pulses_per_burst"] = ppb
        bw = np.asarray(b.width_s, dtype=float)
        out["burst_duration_s"] = float(np.median(bw))
        # within-burst rate from the pulse train itself, not from the name
        gaps = np.asarray(pulses.onset_t[1:]) - np.asarray(pulses.onset_t[:-1])
        within = gaps[gaps < 0.5 * np.median(np.diff(b.onset_t))] if b.n > 1 \
            else gaps[gaps < 1.0]
        if within.size:
            out["within_burst_rate_hz"] = float(1.0 / np.median(within))
            out["duty_cycle"] = float(np.median(w) * out["within_burst_rate_hz"])

    if b.n > 1:
        out["inter_event_interval_s"] = float(np.median(np.diff(
            np.sort(np.asarray(b.onset_t, dtype=float)))))

    if roi_id is not None and len(roi_id) == b.n:
        rid = np.asarray(roi_id, dtype=int)
        valid = rid[rid > 0]
        out["n_sites"] = int(np.unique(valid).size)
        if valid.size:
            counts = np.unique(valid, return_counts=True)[1]
            out["events_per_site"] = {"min": int(counts.min()),
                                      "median": float(np.median(counts)),
                                      "max": int(counts.max())}
            # interval between repeats of ONE site, which is what bounds how
            # long a response may last and still be measurable
            per = []
            for r in np.unique(valid):
                tt = np.sort(np.asarray(b.onset_t)[rid == r])
                if tt.size > 1:
                    per.append(float(np.median(np.diff(tt))))
            if per:
                out["same_site_interval_s"] = float(np.median(per))

    amp = np.asarray(b.amplitude_v, dtype=float)
    out["command_v"] = {"median": float(np.nanmedian(amp)),
                        "min": float(np.nanmin(amp)),
                        "max": float(np.nanmax(amp))}
    if intensity_mat is not None and roi_id is not None:
        im = np.atleast_2d(np.asarray(intensity_mat, dtype=float))
        cal = {int(r): v for r, v in im[:, :2]}
        rid = np.asarray(roi_id, dtype=int)
        mw, used = [], 0
        for a, r in zip(amp, rid):
            v = cal.get(int(r))
            if v and np.isfinite(a) and v > 0:
                mw.append(calibrated_mw * a / v)
                used += 1
        if mw:
            out["estimated_mw"] = {"median": float(np.median(mw)),
                                   "min": float(np.min(mw)),
                                   "max": float(np.max(mw)),
                                   "n_events": used,
                                   "calibrated_mw": calibrated_mw,
                                   "assumes": "linear V->mW through origin"}
    return out


#: Patterns for the parameters a session name may encode.  Names are a HINT
#: only: every one of these is also measured from the laser channel, and the
#: measurement wins.  A session renamed by hand, or whose protocol changed at
#: the rig, would otherwise drive an analysis from a string.
NAME_PATTERNS = {
    "rate_hz": re.compile(r"(\d+(?:\.\d+)?)\s*Hz", re.I),
    "duration_ms": re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*ms", re.I),
    "power_mw": re.compile(r"(\d+(?:\.\d+)?)\s*mW", re.I),
    "n_protocols": re.compile(r"(\d+)\s*prots?", re.I),
    "date": re.compile(r"(?<!\d)(\d{6})(?!\d)"),
}


def parse_session_name(session_id: str) -> dict:
    """Extract whatever protocol hints a session name carries.

    Returns only what is present; absence of a key means the name says
    nothing about it.  ``n_protocols`` comes from forms such as
    ``3protsin1``, which announce that conditions were interleaved without
    saying what they were -- useful as a check on the number of classes the
    data yield, and nothing more.
    """
    out: dict = {}
    for key, pat in NAME_PATTERNS.items():
        hits = pat.findall(session_id)
        if not hits:
            continue
        vals = [float(h) for h in hits]
        if key == "duration_ms":
            out[key] = sorted(set(vals))
        elif key in ("n_protocols",):
            out[key] = int(vals[0])
        elif key == "date":
            out[key] = hits[0]
        else:
            out[key] = vals[0]
    return out


def protocol_matches_name(protocol: dict, session_id: str) -> dict:
    """Check an inferred protocol against the numbers in the session name.

    A mismatch does not mean the data are wrong; it means the name is, and
    any analysis that parsed the name instead of the data is using the wrong
    parameter.
    """
    out = {"session_id": session_id, "checks": [], "ok": True}
    hints = parse_session_name(session_id)
    out["name_hints"] = hints

    # number of interleaved conditions, when the name announces one
    if "n_protocols" in hints and "n_protocol_classes" in protocol:
        named, meas = hints["n_protocols"], protocol["n_protocol_classes"]
        ok = named == meas
        out["checks"].append({"field": "n_protocols", "named": named,
                              "measured": meas, "ok": ok})
        out["ok"] &= ok

    # commanded power, if the name states one and the calibration resolves it
    if "power_mw" in hints and "estimated_mw" in protocol:
        named = hints["power_mw"]
        meas = protocol["estimated_mw"]["median"]
        ok = abs(named - meas) <= max(0.25 * named, 0.2)
        out["checks"].append({"field": "power_mw", "named": named,
                              "measured": meas, "ok": ok})
        out["ok"] &= ok

    # durations: with interleaved classes, compare the SET
    if "duration_ms" in hints and protocol.get("protocol_classes"):
        named = sorted(hints["duration_ms"])
        meas = sorted(round(c["duration_s"] * 1e3) for c in
                      protocol["protocol_classes"])
        ok = (len(named) == len(meas)
              and all(abs(a - b) <= max(0.2 * a, 10) for a, b in
                      zip(named, meas)))
        out["checks"].append({"field": "durations_ms", "named": named,
                              "measured": meas, "ok": ok})
        out["ok"] &= ok

    m = NAME_PATTERNS["rate_hz"].search(session_id)
    if m and "within_burst_rate_hz" in protocol:
        named, meas = float(m.group(1)), protocol["within_burst_rate_hz"]
        ok = abs(named - meas) <= max(0.1 * named, 0.5)
        out["checks"].append({"field": "rate_hz", "named": named,
                              "measured": meas, "ok": ok})
        out["ok"] &= ok
    m = (NAME_PATTERNS["duration_ms"].search(session_id)
         if not protocol.get("protocol_classes") else None)
    if m:
        named = float(m.group(1)) / 1e3
        meas = (protocol.get("burst_duration_s")
                or protocol.get("pulse_width_s", {}).get("median"))
        if meas:
            ok = abs(named - meas) <= max(0.2 * named, 0.01)
            out["checks"].append({"field": "duration_s", "named": named,
                                  "measured": meas, "ok": ok})
            out["ok"] &= ok
    return out


@dataclass
class LaserAssignment:
    onset_t: np.ndarray          # DAQ time of each matched pulse
    roi_id: np.ndarray           # 1-based fiber index, -1 where unmatched
    residual_s: np.ndarray       # matched log time + offset - onset time
    clock_offset_s: float
    n_unmatched_pulses: int
    n_unmatched_log: int
    method: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.onset_t.size)

    @property
    def ok(self) -> bool:
        return (self.n_unmatched_pulses == 0
                and self.n_unmatched_log == 0
                and np.all(self.roi_id > 0))


def _best_offset(
    pulses: np.ndarray,
    log_t: np.ndarray,
    tol_s: float,
    n_probe: int = 25,
) -> tuple[float, int]:
    """Estimate the constant clock offset between the two event sequences.

    Candidate offsets are taken from the first ``n_probe`` events of each
    sequence (so that missing events at the start do not defeat the search) and
    scored by how many pulses fall within ``tol_s`` of a log entry.  The offset
    is then refined to the median residual of the matched subset.
    """
    if pulses.size == 0 or log_t.size == 0:
        return 0.0, 0
    cands = np.unique(
        (pulses[:n_probe, None] - log_t[None, :n_probe]).ravel()
    )
    best_off, best_score = 0.0, -1
    log_sorted = np.sort(log_t)
    for off in cands:
        shifted = log_sorted + off
        pos = np.searchsorted(shifted, pulses)
        pos = np.clip(pos, 1, shifted.size - 1) if shifted.size > 1 else pos * 0
        left = shifted[np.clip(pos - 1, 0, shifted.size - 1)]
        right = shifted[np.clip(pos, 0, shifted.size - 1)]
        d = np.minimum(np.abs(pulses - left), np.abs(pulses - right))
        score = int(np.sum(d <= tol_s))
        if score > best_score:
            best_off, best_score = float(off), score
    # refine
    shifted = log_sorted + best_off
    pos = np.clip(np.searchsorted(shifted, pulses), 1, max(shifted.size - 1, 1))
    left = shifted[np.clip(pos - 1, 0, shifted.size - 1)]
    right = shifted[np.clip(pos, 0, shifted.size - 1)]
    nearest = np.where(np.abs(pulses - left) <= np.abs(pulses - right), left, right)
    resid = pulses - nearest
    inliers = np.abs(resid) <= tol_s
    if inliers.any():
        best_off += float(np.median(resid[inliers]))
    return best_off, best_score


def assign_laser_ids(
    pulses: LaserPulses,
    stim_log: np.ndarray,
    tol_s: float = 0.05,
    fallback_to_order: bool = False,
) -> LaserAssignment:
    """Match detected laser pulses to stim-log entries and attach ROI identity.

    ``stim_log`` is the (n, 2) ``[roi_id, time_s]`` array from
    :func:`multifiber.matio.load_stim_log`, already sorted by time.

    Matching is one-to-one and greedy on absolute residual after a fitted
    constant clock offset.  Unmatched events on either side are reported, not
    filled: an unmatched pulse is a stimulation whose target is unknown, and an
    unmatched log entry is a commanded stimulation that the DAQ did not see.
    Either invalidates the session's weight matrix for that site.

    ``fallback_to_order=True`` reproduces the legacy index-order pairing when
    the counts match exactly.  It is off by default because it cannot detect
    the permutation it is vulnerable to.
    """
    onsets = np.asarray(pulses.onset_t, dtype=float)
    stim_log = np.atleast_2d(np.asarray(stim_log, dtype=float))
    if stim_log.size == 0:
        return LaserAssignment(
            onset_t=onsets,
            roi_id=np.full(onsets.size, -1, dtype=int),
            residual_s=np.full(onsets.size, np.nan),
            clock_offset_s=0.0,
            n_unmatched_pulses=int(onsets.size),
            n_unmatched_log=0,
            method="none:empty_stim_log",
            diagnostics={"note": "no stim log supplied"},
        )

    log_ids = stim_log[:, 0].astype(int)
    log_t = stim_log[:, 1]
    order = np.argsort(log_t, kind="stable")
    log_ids, log_t = log_ids[order], log_t[order]

    offset, score = _best_offset(onsets, log_t, tol_s)
    shifted = log_t + offset

    # Greedy one-to-one matching on |residual|.
    pairs = []
    for i, t in enumerate(onsets):
        j = int(np.argmin(np.abs(shifted - t)))
        pairs.append((abs(shifted[j] - t), i, j))
    pairs.sort()
    used_pulse = np.zeros(onsets.size, dtype=bool)
    used_log = np.zeros(shifted.size, dtype=bool)
    roi = np.full(onsets.size, -1, dtype=int)
    resid = np.full(onsets.size, np.nan)
    for d, i, j in pairs:
        if used_pulse[i] or used_log[j] or d > tol_s:
            continue
        used_pulse[i] = used_log[j] = True
        roi[i] = log_ids[j]
        resid[i] = shifted[j] - onsets[i]

    n_unmatched_pulses = int((~used_pulse).sum())
    n_unmatched_log = int((~used_log).sum())
    method = "timestamp"

    if (n_unmatched_pulses or n_unmatched_log) and fallback_to_order:
        if onsets.size == log_t.size:
            roi = log_ids.copy()
            resid = shifted - onsets
            n_unmatched_pulses = n_unmatched_log = 0
            method = "index_order_fallback"

    finite = np.isfinite(resid)
    return LaserAssignment(
        onset_t=onsets,
        roi_id=roi,
        residual_s=resid,
        clock_offset_s=offset,
        n_unmatched_pulses=n_unmatched_pulses,
        n_unmatched_log=n_unmatched_log,
        method=method,
        diagnostics={
            "tol_s": tol_s,
            "n_pulses": int(onsets.size),
            "n_log_entries": int(log_t.size),
            "offset_inlier_count": int(score),
            "residual_median_s": float(np.median(resid[finite])) if finite.any() else None,
            "residual_max_abs_s": float(np.max(np.abs(resid[finite]))) if finite.any() else None,
            "roi_counts": {int(k): int(v) for k, v in
                           zip(*np.unique(roi[roi > 0], return_counts=True))},
        },
    )


# --------------------------------------------------------------------------- #
# Events -> frames
# --------------------------------------------------------------------------- #

def events_to_frames(
    event_t: np.ndarray,
    frame_t: np.ndarray,
    side: str = "preceding",
) -> np.ndarray:
    """Map event times to video frame indices.

    ``side='preceding'`` returns the last frame whose exposure began at or
    before the event (what the legacy code did via ``find(frame_times < lt)``);
    ``side='nearest'`` returns the closest exposure.  Events before the first
    frame return -1.

    For an event that falls inside an exposure, 'preceding' attributes it to
    the frame it contaminates, which is what artifact blanking needs; 'nearest'
    is appropriate for defining t=0 of a peri-event window.
    """
    event_t = np.asarray(event_t, dtype=float)
    frame_t = np.asarray(frame_t, dtype=float)
    idx = np.searchsorted(frame_t, event_t, side="right") - 1
    if side == "preceding":
        return np.where(idx < 0, -1, idx)
    if side != "nearest":
        raise ValueError(f"unknown side {side!r}")
    nxt = np.clip(idx + 1, 0, frame_t.size - 1)
    cur = np.clip(idx, 0, frame_t.size - 1)
    take_next = np.abs(frame_t[nxt] - event_t) < np.abs(frame_t[cur] - event_t)
    out = np.where(take_next, nxt, cur)
    return np.where(idx < -1, -1, out)
