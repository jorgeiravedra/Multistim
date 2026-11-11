"""
Distinguish a laser bleed-through artifact from an evoked response at the
stimulated site.

They are easy to tell apart in time, and impossible to tell apart from summary
statistics:

* an **artifact** is square, starts and ends within one frame of the command,
  and its amplitude scales with commanded power rather than with biology;
* a **response** rises over hundreds of milliseconds and decays after the
  command ends.

This prints, per event, the pre/during/post means and the 10-90% rise time at
the stimulated ROI, and writes a peri-event figure.

If the stimulus window was blanked during preprocessing there is nothing to
measure, and the script says so rather than producing an empty figure.  Re-run
the pipeline with the real ``--exposure-s`` (read back from the camera), or
with ``--no-blank-laser``, or pass ``--raw`` here to read the unblanked ROI
traces instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", help="a processed session output directory")
    ap.add_argument("--pre", type=float, default=2.0)
    ap.add_argument("--post", type=float, default=4.0)
    ap.add_argument("--raw", action="store_true",
                    help="use the raw ROI traces instead of dF/F; these are "
                         "never blanked, so the artifact is visible even when "
                         "preprocessing excluded the stimulus window")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.session_dir)
    sid = d.name

    man_path = d / f"{sid}_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    qc = man.get("qc", {})

    print(f"=== {sid} ===")
    print(f"blanked samples: {qc.get('n_blanked')}   "
          f"exposure_s: {qc.get('exposure_s')}")
    gp = qc.get("gate_phase")
    if gp and "phase_ms" in gp:
        ph = gp["phase_ms"]
        print(f"gate phase within frame: median {ph['median']:.2f} ms "
              f"(p1 {ph['p1']:.2f} - p99 {ph['p99']:.2f}), "
              f"period {gp['period_s']*1e3:.2f} ms")
        if "fraction_during_exposure" in gp:
            print(f"  {gp['fraction_during_exposure']:.1%} of gate pulses fall "
                  f"inside the {gp['exposure_s']*1e3:.1f} ms exposure; "
                  f"interlock_holds={gp.get('interlock_holds')}")
    elif gp is None:
        print("gate phase: not recorded (older manifest, or no gate channel)")

    events = pd.read_parquet(d / f"{sid}_events.parquet")
    if events.empty:
        print("no laser events")
        return 1

    if args.raw:
        src = pd.read_parquet(d / f"{sid}_traces.parquet")
        note = "raw ROI mean, camera counts (never blanked)"
        ylabel = "ROI mean"
        src = src[src["led_state"] == src["led_state"].mode().iloc[0]]
    else:
        src = pd.read_parquet(d / f"{sid}_dff.parquet")
        note = "dF/F (blanked samples are NaN)"
        ylabel = r"$\Delta F/F$"
    print(f"signal: {note}\n")

    t = src["t"].to_numpy()
    fs = 1.0 / np.median(np.diff(t))
    npre, npost = int(args.pre * fs), int(args.post * fs)

    status = 0
    for region, grp in events.groupby("region"):
        if region is None or region not in src.columns:
            print(f"stimulated region {region!r} has no column; skipping")
            continue
        y = src[region].to_numpy(dtype=float)
        segs, rises, rows = [], [], []
        width = float(grp["width_s"].median())
        dur = max(int(width * fs), 1)

        for _, row in grp.iterrows():
            k = int(np.searchsorted(t, row["onset_t"]))
            if k - npre < 0 or k + npost >= y.size:
                continue
            seg = y[k - npre:k + npost]
            segs.append(seg)
            pre = np.nanmean(seg[:npre]) if np.isfinite(seg[:npre]).any() else np.nan
            dw = y[k:k + dur]
            during = np.nanmean(dw) if np.isfinite(dw).any() else np.nan
            pw = y[k + dur:k + dur + npre]
            post = np.nanmean(pw) if np.isfinite(pw).any() else np.nan
            amp = during - pre
            rise = np.nan
            if np.isfinite(amp) and amp > 0 and int(np.isfinite(dw).sum()) > 3:
                w = dw - pre
                h10 = np.flatnonzero(w >= 0.1 * amp)
                h90 = np.flatnonzero(w >= 0.9 * amp)
                if h10.size and h90.size and h90[0] >= h10[0]:
                    rise = (h90[0] - h10[0]) / fs * 1e3
            rises.append(rise)
            rows.append((int(row["event"]), pre, during, post, rise,
                         int(np.isfinite(dw).sum()), dur))

        if not segs:
            print(f"{region}: no event had a full window; skipping")
            continue

        M = np.vstack(segs)
        n_ok_total = sum(r[5] for r in rows)
        print(f"--- stimulated site: {region} "
              f"({len(rows)} events, {width*1e3:.0f} ms command) ---")

        if n_ok_total == 0:
            print("\n  THE STIMULUS WINDOW IS ENTIRELY BLANKED.")
            print("  Every sample during the command was excluded by laser-")
            print("  artifact blanking, so neither an artifact nor a response")
            print("  can be measured. Re-run the pipeline with the camera's")
            print("  real --exposure-s, or with --no-blank-laser, or re-run")
            print("  this script with --raw.\n")
            status = 2
        else:
            print(f"{'ev':>3} {'pre':>10} {'during':>10} {'post':>10} "
                  f"{'rise10_90_ms':>13} {'n_ok/n':>10}")
            for e, pre, during, post, rise, nok, nd in rows:
                print(f"{e:3d} {pre:10.4f} {during:10.4f} {post:10.4f} "
                      f"{rise:13.1f} {nok:6d}/{nd:<4d}")
            # --- is there an effect at all, before asking what shape it is --
            amps = np.array([r[2] - r[1] for r in rows], dtype=float)
            pre_vals = np.array([r[1] for r in rows], dtype=float)
            n_pos = int(np.sum(amps > 0))
            n_neg = int(np.sum(amps < 0))
            drift = float(np.nanmax(pre_vals) - np.nanmin(pre_vals))
            med_amp = float(np.nanmedian(amps))
            n_rise = int(np.sum(np.isfinite(rises)))

            print(f"\namplitude (during - pre): median {med_amp:+.4f}, "
                  f"{n_pos} positive / {n_neg} negative of {len(rows)} events")
            print(f"pre-period level drifts by {drift:.4f} across the session "
                  f"(|median amplitude| is {abs(med_amp)/drift:.2f} of that)"
                  if drift > 0 else "")

            frame_ms = 1e3 / fs
            inconsistent = min(n_pos, n_neg) > 0.25 * len(rows)
            small = drift > 0 and abs(med_amp) < 0.5 * drift

            if inconsistent or small:
                print("\n  NO CONSISTENT EFFECT. Rise time is not interpretable "
                      "here and is not reported:")
                if inconsistent:
                    print(f"    - the sign is inconsistent ({n_pos} up, "
                          f"{n_neg} down), so events are not doing the same "
                          "thing;")
                if small:
                    print("    - the median amplitude is smaller than the "
                          "session-long drift in the pre-period level, so any "
                          "apparent effect is within the bleaching trend.")
                print("    Rise time was computable for only "
                      f"{n_rise}/{len(rows)} events (it is undefined when the "
                      "amplitude is negative), so a median over them describes "
                      "a hand-picked subset, not the data.")
            else:
                med_rise = float(np.nanmedian(rises))
                print(f"median 10-90% rise: {med_rise:.1f} ms "
                      f"({med_rise / frame_ms:.1f} frames at {fs:.0f} Hz), "
                      f"from {n_rise}/{len(rows)} events")
                if np.isfinite(med_rise) and med_rise <= 2 * frame_ms:
                    print("  -> completes within ~1 frame: consistent with an "
                          "optical artifact, NOT a calcium response")
                elif np.isfinite(med_rise):
                    print("  -> rises over multiple frames: consistent with an "
                          "evoked response")
                print("\n  Note: this is a shape diagnostic on one site in one "
                      "session, not a test. It has no sham condition and no "
                      "correction for the pre-period drift.")

        tt = (np.arange(M.shape[1]) - npre) / fs
        n_finite = np.isfinite(M).sum(axis=0)
        ok = n_finite > 0
        mean_tr = np.full(M.shape[1], np.nan)
        if ok.any():
            mean_tr[ok] = np.nanmean(M[:, ok], axis=0)

        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=200)
        ax.plot(tt, M.T, color="0.7", lw=0.5)
        ax.plot(tt, mean_tr, color="C3", lw=1.5)
        ax.axvspan(0, width, color="C0", alpha=0.15, lw=0)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        if not ok.all():
            gaps = tt[~ok]
            ax.plot(gaps, np.full(gaps.size, ax.get_ylim()[0]), "|",
                    color="C1", ms=4,
                    label=f"blanked ({int((~ok).sum())}/{tt.size} samples)")
            ax.legend(fontsize=6, frameon=False, loc="upper right")
        ax.set_xlabel("time from command onset (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{sid}  {region}  (n={M.shape[0]})", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        out = Path(args.out or d / f"{sid}_{region}_peristim.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"  figure: {out}\n")

    return status


if __name__ == "__main__":
    raise SystemExit(main())
