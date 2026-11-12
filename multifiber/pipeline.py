"""
End-to-end pipeline: a directory in, a processed photometry dataset out.

    discover -> bind animal + calibration -> extract ROI traces -> recover LED
    state -> align triggers to frames -> detect laser events -> assign ROI
    identity -> preprocess -> write

Three properties this is built for:

* **One bad session does not stop the batch.**  Every session is processed
  inside its own error boundary and a failure is written to that session's
  manifest with the stage that failed, rather than raising out of the run.
* **Nothing is silently assumed.**  Each stage contributes to a QC record, and
  checks that cannot be satisfied become warnings on the session, visible in
  the summary table.
* **Re-running is cheap and correct.**  A session is skipped only if both its
  input hashes and the configuration hash are unchanged; either a replaced
  file or a changed parameter forces reprocessing.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from . import events as ev
from . import extract as ex
from . import io as mio
from . import matio, metadata
from . import preprocess as pp
from .discovery import SessionCandidate, discover_sessions
from .progress import Stage

__all__ = ["PipelineConfig", "SessionResult", "process_session",
           "process_directory"]


@dataclass
class PipelineConfig:
    """Every analysis choice, in one hashable place.

    The hash goes into each manifest, so a change here forces reprocessing and
    an old output can be told apart from a new one.
    """

    # extraction
    roi_fraction: float = 0.75
    half: str = "left"
    video_plane: int = 0

    # events
    camera_channel: str = "ham_cam"
    camera_threshold: float = 2.0
    laser_command_channel: str = "laser_cmd"
    laser_gate_channel: Optional[str] = "laser_gate"
    laser_threshold: float = 0.07
    laser_min_width_s: float = 0.0015
    laser_match_tol_s: float = 0.05
    #: Collapse the pulses of a burst protocol into one event per burst.
    #: "auto" infers the grouping from the inter-pulse interval distribution,
    #: a float sets the maximum within-burst gap in seconds, None keeps every
    #: pulse as its own event (which is pseudoreplication for a burst
    #: protocol, and puts neighbouring pulses inside the peri-event baseline).
    group_bursts: "str | float | None" = "auto"

    # LED
    led_mode: str = "relative"
    led_daq_channel: Optional[str] = "led"     # cross-check when live

    # preprocessing
    baseline_tau_s: float = 60.0
    airpls_differences: int = 2
    #: "continuous" fits F0 in one piece (right when the animal is doing the
    #: same thing throughout, as in a stim-record session); "epochs" fits
    #: baseline/stim/post separately (right when the regime changes partway,
    #: as in a resident-intruder assay).
    baseline_mode: str = "continuous"
    #: "airpls" fits the lower envelope (its reweighting discards samples
    #: above the baseline); "symmetric" fits the centre of the data. Both give
    #: the same peri-event contrast; symmetric centres the dF/F scale.
    baseline_method: str = "airpls"
    guard_s: float = 5.0
    regression_alpha: "float | str" = "cv"
    regression_predictors: str = "common"
    #: how the common isosbestic component is built: "median" (robust) or
    #: "pc1" (optimal if the artifact is one scaled waveform).
    isosbestic_common_method: str = "median"
    #: Estimator for the motion regression: lasso | ols | ridge | hierarchical.
    regress_method: str = "lasso"
    #: Fit stimulus-locked square and kinetic regressors, subtracting the
    #: square (an optical artifact has no kinetics) and keeping the kinetic
    #: term as the response estimate.
    stim_artifact_regression: bool = False
    #: Skip the isosbestic regression entirely.  Worth setting when the 415
    #: channel is near-dark: its dF/F is then dominated by its own small
    #: denominator and subtracting it injects noise.
    regress_isosbestic: bool = True
    #: Remove the component shared across all fibers ("median" or "pc1"), the
    #: signature of bundle motion.  None leaves it in.  The stimulated fiber
    #: is excluded from the estimate automatically.
    common_mode: Optional[str] = None
    blank_pad: int = 1
    #: Sensor integration time per frame, seconds.  Leave None to treat the
    #: whole frame period as exposed -- conservative, but with a per-frame
    #: gate chop that flags every stimulus frame.  The rig's camera config
    #: sets ExposureTime (0.0125 s in the pipeline document).
    exposure_s: Optional[float] = None
    blank_laser: bool = True
    smooth: Optional[int] = None

    # bookkeeping
    #: Minimum d-prime between the two LED states (see events._state_separation).
    #: Below ~4 the 415/470 assignment starts to be unreliable.
    min_led_separation: float = 4.0
    write_raw_traces: bool = True
    #: Draw a progress bar for long stages (extraction, preprocessing).
    show_progress: bool = True

    #: Fields that affect ROI extraction from the video.  Everything else
    #: operates on the extracted traces, so changing it does not require
    #: decoding the video again.
    EXTRACTION_FIELDS = ("roi_fraction", "half", "video_plane")

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def extraction_hash(self) -> str:
        """Hash of only the settings that change the extracted traces."""
        d = {k: getattr(self, k) for k in self.EXTRACTION_FIELDS}
        payload = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class SessionResult:
    session_id: str
    status: str                      # ok | failed | skipped
    manifest: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    error: Optional[str] = None
    failed_stage: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _bind_roi_labels(cand: SessionCandidate, n_roi: int) -> list[str]:
    """Fiber index -> region label, checked against the detected ROI count.

    A mismatch is fatal: the calibration's ROI ordering is what the stim GUI's
    laser IDs refer to, so a record describing a different number of fibers
    cannot be mapped onto these traces without guessing, and guessing permutes
    the connectivity matrix.
    """
    labels = cand.animal.implant.ordered_labels()
    if len(labels) != n_roi:
        raise ValueError(
            f"animal record describes {len(labels)} fibers but the "
            f"calibration detected {n_roi} ROIs. These must agree: the ROI "
            "order is what laser IDs index."
        )
    return labels


def process_session(
    cand: SessionCandidate,
    out_root: str | Path,
    config: Optional[PipelineConfig] = None,
    force: bool = False,
    force_extract: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> SessionResult:
    """Run one session end to end and write its outputs.

    ``force`` reprocesses a session that is already up to date.  Extraction is
    still skipped if the video, the calibration and the extraction settings
    are unchanged, since decoding takes minutes and no preprocessing
    parameter affects it; pass ``force_extract`` to decode again as well.
    """
    import pandas as pd

    from . import __version__

    cfg = config or PipelineConfig()
    sid = cand.session_id
    say = progress or (lambda _m: None)
    show = progress is not None and cfg.show_progress
    warnings: list[str] = list(cand.warnings)
    qc: dict = {}
    stage = "setup"

    manifest: dict = {
        "session_id": sid,
        "animal_id": cand.animal.animal_id if cand.animal else None,
        "animal_assumed": cand.animal_assumed,
        "package_version": __version__,
        "config": asdict(cfg),
        "config_hash": cfg.hash(),
        "extraction_hash": cfg.extraction_hash(),
    }

    try:
        if not cand.ok:
            raise ValueError("session has unresolved problems: "
                             + "; ".join(cand.problems))

        # Cheap identity check first: a full hash of the video is only worth
        # paying for if extraction is going to run anyway.
        input_hashes = cand.file_hashes()
        # Always recorded, and always the quantity the reuse check compares.
        # The full hash replaces sha256 when extraction runs, so comparing a
        # fingerprint against a stored full hash would never match and the
        # cache would never be used.
        fingerprints = dict(input_hashes)
        manifest["inputs"] = {
            "paths": {k: str(v) for k, v in [
                ("video", cand.video), ("bcam", cand.bcam),
                ("stim_log", cand.stim_log)] if v},
            "sha256": input_hashes,
            "fingerprint": fingerprints,
            "calibration": (cand.calibration.provenance()
                            if cand.calibration else None),
            "animal_record": {
                "path": cand.animal.source_path,
                "sha256": cand.animal.source_sha256,
                "region_map": cand.animal.implant.region_map(),
            },
        }

        if not force and mio.is_up_to_date(out_root, sid, input_hashes,
                                           cfg.hash()):
            say(f"{sid}: up to date, skipping")
            return SessionResult(sid, "skipped",
                                 manifest=mio.read_manifest(out_root, sid) or {})

        # ---- cheap preflight, before the video is opened ----------------- #
        # Decoding costs minutes. Anything that can fail on metadata alone --
        # a missing DAQ log, an unreadable one, a wrong channel map, a region
        # map that does not match the calibration -- is checked first, so a
        # session that cannot be processed costs seconds rather than a full
        # pass over its video.
        stage = "preflight"
        say(f"{sid}: [1/7] preflight -- DAQ log, triggers, region map")
        log = matio.load_bcam(cand.bcam)
        warnings.extend(log.provenance.get("warnings", []))
        dead = matio.check_channels(log)
        warnings.extend(dead)
        frames = ev.detect_camera_frames(
            log, channel=cfg.camera_channel, threshold=cfg.camera_threshold)
        if frames.n < 2:
            raise ValueError(
                f"only {frames.n} camera triggers on channel "
                f"{cfg.camera_channel!r}; the channel map or threshold is "
                "wrong for this session"
            )
        cal_n = matio.load_calibration(
            cand.calibration.path("translated_coords"),
            cand.calibration.path("mids"))["n_fibers"]
        _bind_roi_labels(cand, cal_n)

        # ---- video + ROI extraction ------------------------------------- #
        stage = "extract"
        say(f"{sid}: [2/7] extraction")

        # Reuse previously extracted traces when nothing that affects them has
        # changed: the video, the calibration, and the extraction settings.
        # Decoding is minutes per session and depends on no preprocessing
        # parameter, so this check happens before the video is touched at all.
        reuse = None
        if not force_extract:
            prior = mio.read_manifest(out_root, sid)
            tp = (mio.session_output_dir(out_root, sid)
                  / f"{sid}_traces.parquet")
            if prior and tp.exists():
                same_ext = prior.get("extraction_hash") == cfg.extraction_hash()
                prior_fp = (prior.get("inputs", {}).get("fingerprint")
                            or prior.get("inputs", {}).get("sha256", {}))
                same_video = prior_fp.get("video") == fingerprints.get("video")
                same_cal = all(
                    prior_fp.get(k) == v for k, v in fingerprints.items()
                    if k.startswith("calibration/"))
                if same_ext and same_video and same_cal:
                    reuse = tp
                elif prior.get("extraction_hash"):
                    why = ("extraction settings changed" if not same_ext
                           else "video changed" if not same_video
                           else "calibration changed")
                    say(f"{sid}: re-extracting ({why})")

        if reuse is not None:
            say(f"{sid}: reusing extracted traces ({reuse.name})")
            tdf = pd.read_parquet(reuse)
            labels = [c for c in tdf.columns
                      if c not in ("frame", "t", "led_state", "roi_mean")]
            _bind_roi_labels(cand, len(labels))
            res = ex.ExtractionResult(
                traces=tdf[labels].to_numpy(dtype=float),
                frame_intensity=tdf["roi_mean"].to_numpy(dtype=float),
                n_frames=len(tdf), labels=labels,
                provenance={"reused_from": str(reuse)})
            info = ex.VideoInfo(
                path=Path(cand.video), width=0, height=0,
                n_frames=len(tdf), n_frames_reported=None, fps=None,
                pix_fmt=None, counted=True)
            masks = None
        else:
            # Extraction is about to read the whole video, so a full hash is
            # negligible by comparison and worth having in the manifest.
            # Extraction is about to read the whole video, so a full hash is
            # negligible by comparison. It goes to sha256 for provenance; the
            # fingerprint stays as written, since that is what the next run's
            # reuse check compares against.
            manifest["inputs"]["sha256"] = cand.file_hashes(full=True)

            # Probe WITHOUT counting frames. Counting demuxes the entire file,
            # and extraction is about to walk every frame anyway and return
            # the authoritative count -- so counting first doubles the passes
            # over the video for nothing.
            info = ex.probe_video(cand.video, count_frames=False,
                                  require_count=False)
            cal = matio.load_calibration(
                cand.calibration.path("translated_coords"),
                cand.calibration.path("mids"))
            labels = _bind_roi_labels(cand, cal["n_fibers"])
            masks = ex.build_roi_masks(
                cal["coords"], cal["mids"],
                image_shape=(info.height, info.width),
                roi_fraction=cfg.roi_fraction, half=cfg.half, labels=labels,
                provenance=cal["provenance"])

            n_hint = info.n_frames_reported
            say(f"{sid}: extracting {masks.n_roi} ROIs from "
                f"{n_hint or 'an unknown number of'} frames")
            with Stage(f"{sid} extract", total=n_hint, enabled=show) as st:
                res = ex.extract_traces(
                    ex.iter_video_frames(cand.video, plane=cfg.video_plane),
                    masks, n_frames=None, progress=st.update_to)
            info.n_frames = res.n_frames
            info.counted = True
            if (info.n_frames_reported is not None
                    and info.n_frames_reported != res.n_frames):
                warnings.append(
                    f"video header claims {info.n_frames_reported} frames, "
                    f"{res.n_frames} were decoded; the decoded count is used"
                )

        # ---- LED state ---------------------------------------------------- #
        stage = "led"
        say(f"{sid}: [3/7] LED state from image brightness")
        led = ev.led_state_from_brightness(res.frame_intensity,
                                           mode=cfg.led_mode)
        if led.separation < cfg.min_led_separation:
            warnings.append(
                f"LED states separate by only {led.separation:.1f} SD of "
                "within-state noise; the 415/470 assignment may be unreliable"
            )
        breaks = ev.detect_parity_breaks(led)
        if breaks["n_breaks"]:
            warnings.append(
                f"{breaks['n_breaks']} break(s) in the LED alternation: "
                "frames are missing from the video file at "
                f"{breaks['break_after'][:5]}"
            )

        led_agreement = None

        # ---- frame alignment ---------------------------------------------- #
        stage = "align"
        say(f"{sid}: [4/7] aligning triggers to frames")
        align = ev.align_triggers_to_frames(
            frames, info.n_frames, drop_after_frames=breaks["break_after"])
        if align.confidence == "low":
            warnings.append(
                "no pre-acquisition exposures to anchor on; frame 0 is "
                f"assumed to be trigger 0 (risk <= {align.anchor_risk_ms:.0f} ms)"
            )

        if cfg.led_daq_channel and cfg.led_daq_channel in log.channels:
            ch = log.channels[cfg.led_daq_channel]
            if ev.channel_is_live(ch, matio.LIVE_CHANNEL_MIN_SPAN):
                daq_led = ev.led_state_from_daq(
                    log, frames, channel=cfg.led_daq_channel,
                    trigger_idx=align.trigger_idx)
                if daq_led.separation >= cfg.min_led_separation:
                    led_agreement = ev.compare_led_assignments(led, daq_led)
                    if led_agreement["agreement"] < 0.99:
                        warnings.append(
                            "LED state from image brightness and from the DAQ "
                            f"agree on only {led_agreement['agreement']:.1%} "
                            "of frames; the two sources disagree about the "
                            "415/470 assignment"
                        )
                else:
                    warnings.append(
                        f"DAQ LED channel is live but its states separate by "
                        f"only {daq_led.separation:.1f} SD, so it was not used "
                        "as a cross-check"
                    )

        # ---- laser events --------------------------------------------------- #
        stage = "laser"
        say(f"{sid}: [5/7] laser events and protocol")
        pulses = ev.detect_laser_pulses(
            log, command_channel=cfg.laser_command_channel,
            gate_channel=cfg.laser_gate_channel,
            threshold=cfg.laser_threshold,
            min_width_s=cfg.laser_min_width_s)

        pulses_raw = pulses
        if cfg.group_bursts is not None:
            n_before = pulses.n
            pulses = ev.group_pulses_into_bursts(
                pulses,
                max_gap_s=(None if cfg.group_bursts == "auto"
                           else float(cfg.group_bursts)))
            gp = pulses.params
            if gp.get("grouped_into_bursts"):
                # A "burst" lasting tens of seconds at a fraction of a hertz
                # is not a burst; it is several trials that the gap-finder
                # merged. The within-train rate is the tell.
                rate = gp.get("within_burst_rate_hz")
                wid = (float(np.median(pulses.width_s)) if pulses.n else 0.0)
                if rate is not None and rate < 1.0 and wid > 2.0:
                    warnings.append(
                        f"burst grouping produced {pulses.n} events of "
                        f"{wid:.1f} s each at {rate:.2f} Hz within the "
                        "train. A train that slow and that long is more "
                        "likely to be separate trials merged by the gap "
                        "finder than one burst. Set group_bursts to None "
                        "(--group-bursts none) if single pulses are the "
                        "trials, or give an explicit gap in seconds."
                    )
            if gp.get("grouped_into_bursts"):
                ppb = pulses.params["pulses_per_burst"]
                warnings.append(
                    f"grouped {n_before} laser pulses into {pulses.n} bursts "
                    f"(median {ppb['median']:.0f} pulses each); the burst is "
                    "the randomised unit, so it is the trial"
                )

        assign = None
        if cand.stim_log is not None and pulses.n:
            stim_rows = matio.load_stim_log(cand.stim_log, verbose=True)
            assign = ev.assign_laser_ids(pulses, stim_rows,
                                         tol_s=cfg.laser_match_tol_s)
            if not assign.ok:
                warnings.append(
                    f"laser/stim-log matching incomplete: "
                    f"{assign.n_unmatched_pulses} pulses and "
                    f"{assign.n_unmatched_log} log entries unmatched — the "
                    "target of those stimulations is unknown"
                )
        elif pulses.n:
            warnings.append(
                "no stim log: laser events have no ROI identity. For a "
                "fixed-site protocol, record the site in the session sidecar."
            )

        # ---- preprocessing -------------------------------------------------- #
        stage = "preprocess"
        say(f"{sid}: [6/7] preprocessing")
        gate_on_t = None
        if cfg.laser_gate_channel and cfg.laser_gate_channel in log.channels:
            g = log.channels[cfg.laser_gate_channel]
            if float(np.nanmax(g) - np.nanmin(g)) >= 0.5:
                gedges = ev.detect_edges(log.time, g, 0.5)
                gate_on_t = gedges.rising_t

        # What protocol was actually delivered, measured from the laser
        # channel rather than read from the session name.
        protocol = name_check = None
        if pulses.n:
            im = None
            lp = cand.calibration.path("laser_intensities") if cand.calibration else None
            if lp is not None and Path(lp).exists():
                try:
                    im = matio.load_mat(lp).get("intensity_mat")
                except Exception:
                    im = None
            protocol = ev.infer_protocol(
                pulses_raw, pulses,
                roi_id=(assign.roi_id if assign is not None else None),
                intensity_mat=im)
            name_check = ev.protocol_matches_name(protocol, sid)
            if protocol.get("interleaved"):
                cls = protocol["protocol_classes"]
                warnings.append(
                    f"{len(cls)} protocols interleaved in this session: "
                    + "; ".join(f"{c['n_events']} events of "
                                f"{c['duration_s'] * 1e3:.0f} ms at "
                                f"{c['command_v']:.3f} V" for c in cls)
                    + ". Each is analysed separately; a single summary "
                      "describes none of them."
                )
            for c in name_check["checks"]:
                if not c["ok"]:
                    warnings.append(
                        f"session name says {c['field']} = {c['named']:g} but "
                        f"the laser channel measures {c['measured']:g}. Any "
                        "analysis that parses the filename is using the wrong "
                        "parameter."
                    )

        stim_roi_idx = None
        if assign is not None and cfg.common_mode:
            ids = {int(i) for i in np.unique(assign.roi_id) if i > 0}
            stim_roi_idx = sorted(i - 1 for i in ids)

        gate_phase = None
        if gate_on_t is not None and len(gate_on_t):
            gate_phase = ev.gate_phase_within_frame(
                align.frame_t, gate_on_t, exposure_s=cfg.exposure_s)
            gedges_off = ev.detect_edges(
                log.time, log.channels[cfg.laser_gate_channel], 0.5).falling_t
            implied = ev.implied_exposure(align.frame_t, gate_on_t,
                                          gedges_off)
            gate_phase["implied"] = implied
            if cfg.exposure_s and "implied_exposure_s" in implied:
                diff = implied["implied_exposure_s"] - cfg.exposure_s
                if abs(diff) > 0.001:
                    warnings.append(
                        f"the gate's earliest onset sits "
                        f"{implied['implied_exposure_s']*1e3:.2f} ms after the "
                        f"camera trigger, but exposure_s is "
                        f"{cfg.exposure_s*1e3:.2f} ms ({diff*1e3:+.2f} ms). "
                        "If the interlock is known to hold, the earliest "
                        "onset IS the end of integration, so either "
                        "exposure_s is wrong or the camera channel records "
                        "an output trigger offset from the start of "
                        "exposure rather than the exposure itself."
                    )
            if cfg.exposure_s is not None and not gate_phase.get("interlock_holds",
                                                                 True):
                warnings.append(
                    f"{gate_phase['fraction_during_exposure']:.1%} of laser "
                    "gate pulses fall inside the exposure window "
                    f"({cfg.exposure_s * 1e3:.1f} ms of a "
                    f"{gate_phase['period_s'] * 1e3:.1f} ms period) — the "
                    "between-frames interlock is not holding"
                )

        say(f"{sid}: preprocessing {len(labels)} fibers "
            f"(baseline tau {cfg.baseline_tau_s:.0f} s, "
            f"{cfg.regression_predictors} isosbestic)")
        _pst = Stage(f"{sid} baseline fit", total=len(labels), enabled=show)
        _pst.__enter__()
        proc = pp.preprocess_traces(
            res.traces, led.state, align.frame_t,
            laser_onset_t=pulses.onset_t if pulses.n else None,
            laser_offset_t=pulses.offset_t if pulses.n else None,
            gate_on_t=gate_on_t if cfg.blank_laser else None,
            exposure_s=cfg.exposure_s,
            labels=labels,
            baseline_tau_s=cfg.baseline_tau_s,
            differences=cfg.airpls_differences,
            baseline_mode=cfg.baseline_mode,
            baseline_method=cfg.baseline_method,
            guard_s=cfg.guard_s,
            alpha=cfg.regression_alpha,
            predictors=cfg.regression_predictors,
            common_method=cfg.isosbestic_common_method,
            regress_method=cfg.regress_method,
            stim_artifact=cfg.stim_artifact_regression,
            regress=cfg.regress_isosbestic,
            common_mode=cfg.common_mode,
            common_mode_exclude=stim_roi_idx,
            blank_pad=cfg.blank_pad,
            smooth=cfg.smooth,
            progress=_pst,
        )
        _pst.__exit__(None, None, None)
        warnings.extend(proc.qc.get("warnings", []))

        # ---- tables ----------------------------------------------------------- #
        stage = "write"
        say(f"{sid}: [7/7] writing outputs")
        traces_df = None
        if cfg.write_raw_traces:
            traces_df = res.to_frame()
            traces_df.insert(1, "t", align.frame_t)
            traces_df.insert(2, "led_state", led.state)
        dff_df = proc.to_frame()

        events_df = None
        if pulses.n:
            ev_frames = ev.events_to_frames(pulses.onset_t, align.frame_t,
                                            side="nearest")
            roi = (assign.roi_id if assign is not None
                   else np.full(pulses.n, -1, dtype=int))
            prot_class = ev.classify_protocols(
                pulses,
                roi_id=(assign.roi_id if assign is not None else None))
            events_df = pd.DataFrame({
                "event": np.arange(pulses.n),
                "protocol_class": prot_class,
                "onset_t": pulses.onset_t,
                "offset_t": pulses.offset_t,
                "width_s": pulses.width_s,
                "onset_frame": ev_frames,
                "roi_id": roi,
                "region": [labels[i - 1] if 1 <= i <= len(labels) else None
                           for i in roi],
                "command_v": pulses.amplitude_v,
                "n_gate_pulses": pulses.n_gate_pulses,
                "gate_duty": pulses.gate_duty,
                "gate_on_s": pulses.gate_on_s,
                "match_residual_s": (assign.residual_s if assign is not None
                                     else np.full(pulses.n, np.nan)),
            })

        qc = {
            "n_video_frames": info.n_frames,
            "n_video_frames_reported": info.n_frames_reported,
            "n_roi": len(labels),
            "roi_sizes": (masks.roi_sizes.tolist() if masks is not None
                          else None),
            "traces_reused": reuse is not None,
            "daq_rate_hz": log.rate_hz,
            "n_camera_triggers": frames.n,
            "frame_anchor_confidence": align.confidence,
            "frame_anchor_risk_ms": align.anchor_risk_ms,
            "n_dropped_leading": align.n_dropped_leading,
            "n_unused_trailing": align.n_unused_trailing,
            "n_writer_drops": align.n_writer_drops,
            "led_separation": led.separation,
            "led_parity_breaks": breaks["n_breaks"],
            "led_daq_agreement": led_agreement,
            "dead_channels": dead,
            "n_laser_events": pulses.n,
            "laser_params": pulses.params,
            "protocol": protocol,
            "protocol_name_check": name_check,
            "laser_assignment": (assign.diagnostics if assign is not None
                                 else None),
            "effective_fps": proc.qc["effective_fps"],
            "epochs": proc.epochs,
            "regression_r2": proc.qc["regression_r2"],
            "dff_sd_before_regression": proc.qc["dff_sd_before_regression"],
            "dff_sd_after_regression": proc.qc["dff_sd_after_regression"],
            "isosbestic_f0_median": proc.qc["isosbestic_f0_median"],
            "n_roi_high_noise": proc.qc["n_roi_high_noise"],
            "common_mode": proc.qc["common_mode"],
            "regression_alpha": proc.qc["regression_alpha"],
            "baseline_lambda": proc.qc["lambda"],
            "baseline_mode": proc.qc["baseline_mode"],
            "baseline_method": proc.qc["baseline_method"],
            "f0_boundary_jumps": proc.qc["f0_boundary_jumps"],
            "n_blanked": int(proc.blanked.sum()),
            "exposure_s": cfg.exposure_s,
            "blank_laser": cfg.blank_laser,
            "gate_phase": gate_phase,
        }
        manifest.update({"status": "ok", "qc": qc, "warnings": warnings})
        paths = mio.write_session(out_root, sid, manifest,
                                  traces=traces_df, dff=dff_df,
                                  events=events_df)
        say(f"{sid}: done ({len(warnings)} warnings)")
        return SessionResult(sid, "ok", manifest=manifest,
                             paths={k: str(v) for k, v in paths.items()})

    except Exception as exc:  # noqa: BLE001 - boundary is deliberate
        manifest.update({
            "status": "failed",
            "failed_stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "qc": qc,
            "warnings": warnings,
        })
        try:
            mio.write_session(out_root, sid, manifest)
        except Exception:  # pragma: no cover - never mask the original error
            pass
        say(f"{sid}: FAILED at {stage}: {exc}")
        return SessionResult(sid, "failed", manifest=manifest,
                             error=str(exc), failed_stage=stage)


def process_directory(
    data_root: str | Path,
    out_root: str | Path,
    registry_root: Optional[str | Path] = None,
    calibration_dir: Optional[str | Path] = None,
    config: Optional[PipelineConfig] = None,
    sessions: Optional[Sequence[str]] = None,
    force: bool = False,
    force_extract: bool = False,
    default_animal: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = print,
):
    """Process every session under ``data_root``.

    Returns ``(results, summary)``: the per-session results and a DataFrame
    with one row per session, including those that failed.  Failures are
    recorded, not raised, so a batch completes and the summary shows which
    sessions need attention and at which stage.
    """
    cfg = config or PipelineConfig()
    registry = (metadata.load_registry(registry_root)
                if registry_root is not None else {})
    cands = discover_sessions(data_root, registry,
                              calibration_dir=calibration_dir,
                              default_animal=default_animal)
    if sessions is not None:
        wanted = set(sessions)
        cands = [c for c in cands if c.session_id in wanted]

    results = [process_session(c, out_root, cfg, force=force,
                               force_extract=force_extract, progress=progress)
               for c in cands]
    return results, mio.summarise_outputs(out_root)
