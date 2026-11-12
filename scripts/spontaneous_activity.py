"""
Is each fiber reporting anything at all?

A null stimulation result has two very different causes, and they are
separated without sacrificing the animal: either the perturbation failed, or
the recording channel cannot report a response of any origin.  Spontaneous
calcium transients distinguish them.  A freely moving mouse produces
transients of roughly 1-10% dF/F in these regions; a fiber that shows them has
a working indicator, sits in tissue, and has adequate collection efficiency,
so a null there is informative about the opsin.  A flat fiber reports nothing
about anything.

Transients are detected on the signal channel and, as a specificity check, the
same detection is run on the isosbestic: a real calcium transient is absent
from 415, while motion and coupling artifacts appear in both.  The isosbestic
rate is therefore the false-positive rate of the detector on that fiber.

Output per fiber: transient rate, median and 90th-percentile amplitude, the
isosbestic rate, and the amplitude expressed in units of that channel's
stimulation detection floor -- which is the number that says whether the
channel could have seen an evoked response of plausible size.
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
    from scipy.signal import find_peaks

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--min-prominence-mad", type=float, default=4.0,
                    help="peak prominence threshold, in robust SD (MAD-based) "
                         "of the high-passed trace")
    ap.add_argument("--min-width-s", type=float, default=0.15,
                    help="minimum peak width; below the indicator's rise+decay "
                         "a peak is noise, not a transient")
    ap.add_argument("--highpass", type=float, default=0.1)
    ap.add_argument("--exclude-stim-s", type=float, default=5.0,
                    help="ignore this much time around every laser command, so "
                         "the measure is of spontaneous activity only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib as mpl
    mpl.rcParams.update({"pdf.fonttype": 42, "font.size": 7,
                         "axes.spines.top": False, "axes.spines.right": False})

    d = Path(args.session_dir)
    sid = d.name
    dff = pd.read_parquet(d / f"{sid}_dff.parquet")
    man_p = d / f"{sid}_manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    rois = [c for c in dff.columns
            if c not in ("t", "frame", "blanked") and not c.endswith("__iso")]
    t = dff["t"].to_numpy()
    fs = 1.0 / np.median(np.diff(t))

    keep = np.ones(t.size, dtype=bool)
    ev_p = d / f"{sid}_events.parquet"
    events = pd.read_parquet(ev_p) if ev_p.exists() else pd.DataFrame()
    for _, e in events.iterrows():
        keep &= ~((t > e.onset_t - args.exclude_stim_s)
                  & (t < e.offset_t + args.exclude_stim_s))
    dur = float(keep.sum() / fs)
    print(f"=== {sid} ===")
    print(f"{len(rois)} fibers, {dur:.0f} s of stimulation-free recording "
          f"at {fs:.0f} Hz\n")
    if dur < 30:
        print("!! under 30 s free of stimulation; rates will be unreliable.")

    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, args.highpass, "highpass", fs=fs, output="sos")

    def detect(y):
        v = np.where(np.isfinite(y), y, np.nanmedian(y[np.isfinite(y)]))
        v = sosfiltfilt(sos, v)[keep]
        mad = float(np.median(np.abs(v - np.median(v))))
        sd = 1.4826 * mad
        if sd <= 0:
            return np.array([]), np.array([]), sd
        pk, props = find_peaks(v, prominence=args.min_prominence_mad * sd,
                               width=args.min_width_s * fs)
        return pk, props["prominences"], sd

    rows = []
    for r in rois:
        pk, amp, sd = detect(dff[r].to_numpy(float))
        ipk, iamp, isd = (detect(dff[f"{r}__iso"].to_numpy(float))
                          if f"{r}__iso" in dff.columns
                          else (np.array([]), np.array([]), np.nan))
        rows.append({
            "fiber": r,
            "rate_per_min": 60.0 * pk.size / dur if dur > 0 else np.nan,
            "iso_rate_per_min": 60.0 * ipk.size / dur if dur > 0 else np.nan,
            "amp_median": float(np.median(amp)) if amp.size else np.nan,
            "amp_p90": float(np.percentile(amp, 90)) if amp.size else np.nan,
            "noise_sd": sd,
        })
    adf = pd.DataFrame(rows)

    # detection floor for stimulation, from the effects table if present
    eff_p = d / f"{sid}_effects.csv"
    if eff_p.exists():
        ef = pd.read_csv(eff_p)
        floors = (ef[ef.stim_site != ef.recorded_site]
                  .groupby("recorded_site")["null_sd"].median() * 3)
        adf["stim_floor"] = adf["fiber"].map(floors)
        adf["amp_over_floor"] = adf["amp_median"] / adf["stim_floor"]

    cols = [c for c in ("fiber", "rate_per_min", "iso_rate_per_min",
                        "amp_median", "amp_p90", "stim_floor",
                        "amp_over_floor") if c in adf.columns]
    print(adf[cols].to_string(index=False,
                              float_format=lambda v: f"{v:9.4f}"))

    silent = adf[adf["rate_per_min"] < 0.5]
    suspect = adf[adf["iso_rate_per_min"] > 0.5 * adf["rate_per_min"]]
    print("\n--- reading this ---")
    if len(silent):
        print(f"  {len(silent)} fiber(s) show essentially no spontaneous "
              f"transients: {', '.join(silent['fiber'])}.")
        print("     These channels report nothing of any origin, so their "
              "null stimulation result says nothing about the opsin. Suspect "
              "indicator expression, fiber placement or coupling.")
    else:
        print("  Every fiber shows spontaneous transients: the indicator, the "
              "placement and the detection chain work throughout, so a null "
              "stimulation result is about the perturbation.")
    if len(suspect):
        print(f"\n  {len(suspect)} fiber(s) detect nearly as many 'transients' "
              f"in the isosbestic as in the signal: "
              f"{', '.join(suspect['fiber'])}.")
        print("     A calcium transient is absent from 415. Equal rates mean "
              "the detector is firing on motion or coupling artifact, and "
              "that fiber's apparent activity is not neural.")
    if "amp_over_floor" in adf.columns:
        weak = adf[adf["amp_over_floor"] < 1]
        print(f"\n  Median spontaneous amplitude is {adf['amp_over_floor'].median():.1f}x "
              "the stimulation detection floor across fibers.")
        if len(weak):
            print(f"  {len(weak)} fiber(s) have spontaneous transients SMALLER "
                  f"than their own stimulation floor: {', '.join(weak['fiber'])}.")
            print("     An evoked response the size of a typical spontaneous "
                  "event would be undetectable there even if it occurred.")

    adf.to_csv(d / f"{sid}_spontaneous.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    o = np.argsort(adf["rate_per_min"].to_numpy())
    axes[0].barh(np.arange(len(adf)), adf["rate_per_min"].to_numpy()[o],
                 color="C0", label="470")
    axes[0].barh(np.arange(len(adf)), adf["iso_rate_per_min"].to_numpy()[o],
                 color="C1", alpha=0.7, height=0.5, label="415 (false +)")
    axes[0].set_yticks(np.arange(len(adf)))
    axes[0].set_yticklabels(adf["fiber"].to_numpy()[o], fontsize=5)
    axes[0].set_xlabel("spontaneous transients / min")
    axes[0].legend(fontsize=5, frameon=False)

    axes[1].barh(np.arange(len(adf)), adf["amp_median"].to_numpy()[o],
                 color="C0")
    if "stim_floor" in adf.columns:
        axes[1].barh(np.arange(len(adf)), adf["stim_floor"].to_numpy()[o],
                     color="C3", alpha=0.6, height=0.5, label="stim floor")
        axes[1].legend(fontsize=5, frameon=False)
    axes[1].set_yticks([])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("median transient amplitude (dF/F)")

    if "amp_over_floor" in adf.columns:
        axes[2].barh(np.arange(len(adf)), adf["amp_over_floor"].to_numpy()[o],
                     color="C2")
        axes[2].axvline(1, color="k", lw=0.6, ls="--")
        axes[2].set_xscale("log")
        axes[2].set_xlabel("transient amplitude / stim floor")
    axes[2].set_yticks([])
    for ax in axes:
        ax.tick_params(labelsize=5)
    fig.suptitle(f"{sid} — spontaneous activity: can each fiber report "
                 "anything at all?", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(args.out or d / f"{sid}_spontaneous.pdf")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\n  {out}\n  {sid}_spontaneous.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
