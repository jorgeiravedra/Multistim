"""
Compare laser-event alignment against an independent pipeline.

Takes a MATLAB-style ``<sid>_laser_times.csv`` -- columns (time_s, video frame
index, ROI id), one row per PULSE, with time referenced to the first written
video frame -- and checks it against this pipeline's events.

Four things are compared, and they fail in different ways:

* **Event count**, at pulse and at burst level.  A mismatch here means the two
  pipelines disagree about what a stimulation is, not about when it happened.
* **Timing**, after removing the constant offset between the two clocks (one
  is DAQ-absolute, the other is relative to the first video frame).  The
  residual after that offset is the real disagreement; the offset itself is
  only a difference of reference.
* **ROI identity**, which is the one that silently permutes a connectivity
  matrix if it is wrong.
* **Frame index**, which is where a different frame-anchoring convention
  shows up -- a constant integer offset here means one pipeline counted
  pre-acquisition exposures that the other dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("reference_csv",
                    help="the other pipeline's *_laser_times.csv")
    ap.add_argument("--tol-ms", type=float, default=25.0,
                    help="matching tolerance; 25 ms is one camera frame at "
                         "80 fps")
    ap.add_argument("--burst-gap-s", type=float, default=0.5)
    args = ap.parse_args()

    d = Path(args.session_dir)
    sid = d.name
    ref = pd.read_csv(args.reference_csv, header=None,
                      names=["t_rel", "frame", "roi"])
    ours = pd.read_parquet(d / f"{sid}_events.parquet")
    traces = pd.read_parquet(d / f"{sid}_traces.parquet")

    # video frame index -> DAQ time, from our own alignment
    f2t = dict(zip(traces["frame"].to_numpy(), traces["t"].to_numpy()))
    frame0_t = float(traces["t"].iloc[0])

    print(f"=== {sid} ===")
    print(f"reference: {Path(args.reference_csv).name}\n")

    # ---- counts ---------------------------------------------------------- #
    rt = ref["t_rel"].to_numpy(float)
    gaps = np.diff(np.sort(rt))
    ref_starts = np.r_[0, np.flatnonzero(gaps > args.burst_gap_s) + 1]
    ref_bursts = ref.iloc[ref_starts].reset_index(drop=True)
    print("--- event counts ---")
    print(f"  reference: {len(ref)} pulses -> {len(ref_bursts)} bursts "
          f"({len(ref) // max(len(ref_bursts), 1)} pulses each)")
    print(f"  ours:      {len(ours)} events")
    grouped = len(ours) == len(ref_bursts)
    if grouped:
        print("  -> same number of events; ours are burst-level.")
    elif len(ours) == len(ref):
        print("  -> ours are PULSE-level, the reference's are too. Note the "
              "burst is the randomised unit; pulse-level n is "
              "pseudoreplication.")
    else:
        print("  !! counts differ at both levels. The pipelines disagree "
              "about what a stimulation event is.")

    tgt = ref_bursts if grouped else ref
    # ---- timing ---------------------------------------------------------- #
    ours_t = ours["onset_t"].to_numpy(float)
    ref_t = tgt["t_rel"].to_numpy(float)
    # the two clocks differ by a constant: ours is DAQ-absolute, theirs is
    # referenced to the first written video frame
    offset = float(np.median(np.sort(ours_t)[:len(ref_t)] - np.sort(ref_t)))
    print("\n--- timing ---")
    print(f"  clock offset (ours - reference): {offset:.4f} s")
    print(f"  our first video frame sits at DAQ t = {frame0_t:.4f} s "
          f"({'consistent' if abs(offset - frame0_t) < 0.05 else 'DIFFERENT'}"
          " with the reference being frame-0-relative)")

    shifted = np.sort(ref_t) + offset
    o = np.sort(ours_t)
    n = min(o.size, shifted.size)
    resid = (o[:n] - shifted[:n]) * 1e3
    print(f"  residual after offset: median {np.median(resid):+.2f} ms, "
          f"max |{np.abs(resid).max():.2f}| ms over {n} events")
    tol = args.tol_ms
    print(f"  within {tol:.0f} ms: {int(np.sum(np.abs(resid) <= tol))}/{n}")
    if np.abs(resid).max() <= tol:
        print("  -> timing agrees to within one camera frame.")
    else:
        print("  !! some events differ by more than one frame; inspect the "
              "largest residuals below.")
        k = np.argsort(-np.abs(resid))[:5]
        for i in k:
            print(f"     event {i}: ours {o[i]:.4f}s, "
                  f"reference {shifted[i]:.4f}s, {resid[i]:+.1f} ms")

    # ---- ROI identity ----------------------------------------------------- #
    print("\n--- ROI identity ---")
    if "roi_id" not in ours.columns:
        print("  our events carry no roi_id; nothing to compare.")
    else:
        ours_sorted = ours.sort_values("onset_t").reset_index(drop=True)
        ref_sorted = tgt.sort_values("t_rel").reset_index(drop=True)
        m = min(len(ours_sorted), len(ref_sorted))
        a = ours_sorted["roi_id"].to_numpy()[:m]
        b = ref_sorted["roi"].to_numpy()[:m]
        agree = int(np.sum(a == b))
        print(f"  identical for {agree}/{m} events ({agree / m:.1%})")
        if agree < m:
            bad = np.flatnonzero(a != b)[:10]
            print("  !! disagreements (event: ours vs reference):")
            for i in bad:
                print(f"     {i}: {a[i]} vs {b[i]}")
            xt = pd.crosstab(pd.Series(b, name="reference"),
                             pd.Series(a, name="ours"))
            print("\n  cross-tabulation (a permutation here means one "
                  "pipeline's fiber order is shifted,\n  which transposes the "
                  "connectivity matrix without any error being raised):")
            print(xt.to_string())
        else:
            print("  -> ROI assignment is identical; the fiber-to-event map "
                  "agrees.")

    # ---- frame index ------------------------------------------------------ #
    print("\n--- frame index ---")
    if "onset_frame" not in ours.columns:
        print("  our events carry no onset_frame; nothing to compare.")
    else:
        ours_sorted = ours.sort_values("onset_t").reset_index(drop=True)
        ref_sorted = tgt.sort_values("t_rel").reset_index(drop=True)
        m = min(len(ours_sorted), len(ref_sorted))
        fa = ours_sorted["onset_frame"].to_numpy()[:m]
        fb = ref_sorted["frame"].to_numpy()[:m]
        dfm = fa - fb
        print(f"  difference (ours - reference): median {np.median(dfm):+.1f} "
              f"frames, range {dfm.min():+d} to {dfm.max():+d}")

        # Sub-frame phase of each event: how far after the preceding camera
        # trigger it fell, as a fraction of the frame period. A rounding
        # convention shows up here and a real misalignment does not.
        ot = ours_sorted["onset_t"].to_numpy(float)[:m]
        ft = traces["t"].to_numpy(float)
        period = float(np.median(np.diff(ft)))
        prev = np.searchsorted(ft, ot, side="right") - 1
        ok = (prev >= 0) & (prev < ft.size)
        phase = np.full(m, np.nan)
        phase[ok] = (ot[ok] - ft[prev[ok]]) / period

        spread = int(np.ptp(dfm))
        if spread == 0:
            if dfm[0] == 0:
                print("  -> frame indices are identical.")
            else:
                print(f"  -> a CONSTANT offset of {dfm[0]:+d} frames: the two "
                      "pipelines anchor frames differently, one counting "
                      "pre-acquisition exposures the other drops. Timing is "
                      "otherwise identical.")
        elif spread == 1 and set(np.unique(dfm)) <= {0, 1, -1}:
            hi = dfm == dfm.max()
            lo = dfm == dfm.min()
            ph_hi = float(np.nanmean(phase[hi])) if hi.any() else np.nan
            ph_lo = float(np.nanmean(phase[lo])) if lo.any() else np.nan
            print(f"  sub-frame phase: {np.nanmean(phase):.2f} of a frame "
                  f"on average; events differing by {dfm.max():+d} sit at "
                  f"phase {ph_hi:.2f}, those at {dfm.min():+d} at "
                  f"{ph_lo:.2f}")
            if np.isfinite(ph_hi) and np.isfinite(ph_lo) and ph_hi > ph_lo:
                print("  -> a ROUNDING CONVENTION, not a misalignment. Events "
                      "late within their frame period round up in one "
                      "pipeline and down in the other (nearest vs preceding "
                      "frame). The underlying times agree; the quantisation "
                      "differs by at most one frame, which is 12.5 ms at "
                      "80 fps.")
                print("     Which is right depends on use: 'preceding' names "
                      "the frame an event contaminates, 'nearest' the frame "
                      "closest to it. Use preceding for blanking, nearest "
                      "for defining t=0.")
            else:
                print("  !! a one-frame difference not explained by sub-frame "
                      "phase; inspect individual events.")
        else:
            print("  !! differences span more than one frame, so the two "
                  "disagree about more than rounding: suspect dropped frames "
                  "handled differently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
