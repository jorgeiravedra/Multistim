"""
Isolate a native crash in the preprocessing stage.

A heap-corruption exit (Windows 0xC0000374 / 3228369023) kills the process
without a Python traceback, so the usual debugging tools show nothing.  This
runs the pipeline stages one at a time with flushed output: the LAST LINE
PRINTED is the step that died.

Run from a terminal, not a notebook -- a dying kernel discards buffered output:

    conda activate multifiber
    python scripts/diagnose_crash.py "F:\\Polina\\...\\101926" ^
        --registry "C:\\Users\\FalknerLab\\code\\multifiber-metadata" ^
        --calibration "F:\\Jorge\\Tests\\multistim"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def say(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--session", default=None, help="stem; default = first OK")
    ap.add_argument("--roi", type=int, default=None,
                    help="only fit this ROI (1-based)")
    args = ap.parse_args()

    say("=== environment ===")
    import numpy as np
    import scipy
    import sklearn
    say(f"python     {sys.version.split()[0]}")
    say(f"numpy      {np.__version__}")
    say(f"scipy      {scipy.__version__}")
    say(f"sklearn    {sklearn.__version__}")
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        say(f"numpy blas {blas.get('name')} {blas.get('version')}")
    except Exception as exc:
        say(f"numpy blas <unavailable: {exc}>")

    import multifiber
    from multifiber import discovery, events as ev, extract as ex
    from multifiber import matio, metadata, preprocess as pp
    say(f"multifiber {multifiber.__version__}")

    say("\n=== discovery ===")
    registry = metadata.load_registry(args.registry) if args.registry else {}
    cands = discovery.discover_sessions(args.data_root, registry,
                                        calibration_dir=args.calibration)
    pick = [c for c in cands if c.ok]
    if args.session:
        pick = [c for c in cands if c.stem == args.session]
    if not pick:
        say("no usable session found")
        return 1
    cand = pick[0]
    say(f"session {cand.stem}")

    say("\n=== extract ===")
    info = ex.probe_video(cand.video, count_frames=True)
    say(f"video {info.n_frames} frames {info.width}x{info.height}")
    cal = matio.load_calibration(cand.calibration.path("translated_coords"),
                                 cand.calibration.path("mids"))
    labels = cand.animal.implant.ordered_labels()
    masks = ex.build_roi_masks(cal["coords"], cal["mids"],
                               image_shape=(info.height, info.width),
                               labels=labels)
    res = ex.extract_traces(ex.iter_video_frames(cand.video), masks,
                            n_frames=info.n_frames)
    say(f"traces {res.traces.shape} "
        f"range {np.nanmin(res.traces):.1f}-{np.nanmax(res.traces):.1f}")
    say(f"non-finite in traces: {int((~np.isfinite(res.traces)).sum())}")

    say("\n=== events ===")
    log = matio.load_bcam(cand.bcam)
    frames = ev.detect_camera_frames(log)
    led = ev.led_state_from_brightness(res.frame_intensity)
    breaks = ev.detect_parity_breaks(led)
    align = ev.align_triggers_to_frames(frames, info.n_frames,
                                        drop_after_frames=breaks["break_after"])
    pulses = ev.detect_laser_pulses(log)
    say(f"{align.describe()}; {pulses.n} laser events")

    say("\n=== demux ===")
    dm = pp.demux_channels(res.traces, led.state, align.frame_t, labels=labels)
    say(f"signal {dm.signal.shape} isosbestic {dm.isosbestic.shape}")
    say(f"signal non-finite: {int((~np.isfinite(dm.signal)).sum())}")
    say(f"signal min {np.nanmin(dm.signal):.3f} max {np.nanmax(dm.signal):.3f}")
    zero_cols = [labels[j] for j in range(dm.signal.shape[1])
                 if np.nanmax(np.abs(dm.signal[:, j])) < 1e-9]
    if zero_cols:
        say(f"!! all-zero ROIs: {zero_cols}")

    say("\n=== baseline fit, one ROI at a time ===")
    fs = 1.0 / float(np.median(np.diff(dm.signal_t)))
    lam = pp.lambda_from_timescale(60.0, fs, 2)
    say(f"effective fps {fs:.3f}, lambda {lam:.3e}, n {dm.signal.shape[0]}")
    rois = ([args.roi - 1] if args.roi
            else range(dm.signal.shape[1]))
    for j in rois:
        col = dm.signal[:, j]
        say(f"  ROI {j+1:2d} {labels[j]:<10s} "
            f"min={np.nanmin(col):9.2f} max={np.nanmax(col):9.2f} "
            f"nonfinite={int((~np.isfinite(col)).sum())}")
        for solver in ("banded", "sparse"):
            say(f"      {solver} solve ...")
            z = pp.airpls(col, lambda_=lam, differences=2, solver=solver)
            say(f"      {solver} ok, median F0 {np.median(z):.2f}")

    say("\n=== regression ===")
    iso = np.column_stack([np.interp(dm.signal_t, dm.isos_t, dm.isosbestic[:, j])
                           for j in range(dm.n_roi)])
    for j in rois:
        say(f"  ROI {j+1} LassoCV ...")
        f0 = pp.airpls(dm.signal[:, j], lambda_=lam, differences=2)
        dff = pp.compute_dff(dm.signal[:, j], f0)
        iso_f0 = pp.airpls(iso[:, j], lambda_=lam, differences=2)
        iso_dff = pp.compute_dff(iso[:, j], iso_f0)
        _, infod = pp.regress_isosbestic(dff, iso_dff.reshape(-1, 1), alpha="cv")
        say(f"  ROI {j+1} alpha={infod['alpha']:.3e} r2={infod['r2_in_sample']:.3f}")

    say("\n=== full preprocess_traces ===")
    out = pp.preprocess_traces(
        res.traces, led.state, align.frame_t,
        laser_onset_t=pulses.onset_t if pulses.n else None,
        laser_offset_t=pulses.offset_t if pulses.n else None,
        labels=labels)
    say(out.describe())
    say("\nALL STAGES COMPLETED")
    return 0


if __name__ == "__main__":
    import numpy as np  # noqa: F401  (used above after import inside main)
    raise SystemExit(main())
