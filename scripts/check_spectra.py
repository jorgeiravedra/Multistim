"""
Spectral diagnosis of jitter in processed dF/F traces.

Three artifacts have distinct, identifiable frequencies on this rig:

* **stimulation frequency** (e.g. 10 Hz for a 10 Hz burst protocol).  Laser
  bleed into the emission path appears at the burst rate and only while the
  command is on, so the test is a peak at that frequency that is present
  during stimulation and absent outside it.
* **Nyquist (half the demultiplexed rate, 20 Hz here)** -- a sample-to-sample
  alternation in the signal channel: residual 415/470 crosstalk, a partly
  wrong LED assignment, or an alternating gain between successive flashes of
  the same LED.

  This one is reported BOTH as an absolute peak prominence and as a
  during/outside ratio.  The ratio alone is blind to it: an alternation
  present throughout the recording appears equally in both windows and gives
  a ratio of 1, which reads as "absent".
* **broadband** elevation with no peak is just noise, and at these dF/F
  magnitudes is usually quantisation of an 8-bit video.

Compares the amplitude spectrum during the command windows against matched
windows outside them, so a peak that exists in both is a property of the
recording rather than of the stimulation.
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
    from scipy.signal import welch

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--stim-freq", type=float, default=None,
                    help="burst frequency in Hz; inferred from the session "
                         "name (e.g. '10Hzburst') when omitted")
    ap.add_argument("--raw", action="store_true",
                    help="use raw ROI traces instead of dF/F")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.session_dir)
    sid = d.name
    out = Path(args.out or d)

    stim_f = args.stim_freq
    if stim_f is None:
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*Hz", sid, re.I)
        stim_f = float(m.group(1)) if m else None
    print(f"=== {sid} ===")
    print(f"stimulation frequency: {stim_f if stim_f else 'unknown'} Hz")

    events = pd.read_parquet(d / f"{sid}_events.parquet")
    if args.raw:
        src = pd.read_parquet(d / f"{sid}_traces.parquet")
        src = src[src["led_state"] == src["led_state"].mode().iloc[0]]
    else:
        src = pd.read_parquet(d / f"{sid}_dff.parquet")
    rois = [c for c in src.columns if c not in ("t", "frame", "blanked",
                                                "led_state", "roi_mean")]
    t = src["t"].to_numpy()
    fs = 1.0 / np.median(np.diff(t))
    print(f"effective sampling rate {fs:.2f} Hz (Nyquist {fs/2:.1f} Hz)\n")

    on = np.zeros(t.size, bool)
    for _, e in events.iterrows():
        on |= (t >= e.onset_t) & (t <= e.offset_t)
    # matched off-windows: same durations, shifted well clear of any command
    off = np.zeros(t.size, bool)
    for _, e in events.iterrows():
        w = e.offset_t - e.onset_t
        off |= (t >= e.offset_t + 2.0) & (t <= e.offset_t + 2.0 + w)
    off &= ~on
    print(f"{on.sum()} samples during stimulation, {off.sum()} matched outside")
    per_ev = on.sum() / max(len(events), 1)
    if per_ev < 4:
        print(f"\n  !! {per_ev:.1f} samples per event. The 'during' spectrum "
              "would be computed on a concatenation of isolated samples, "
              "which is meaningless — Welch needs contiguous segments.")
        print("  This is what a burst protocol looks like when each PULSE is "
              "treated as an event: the commands are milliseconds long.")
        print("  Re-run the pipeline with burst grouping (PipelineConfig("
              "group_bursts='auto')), or pass --stim-freq and read only the "
              "'outside' spectra.\n")
        if per_ev < 1:
            print("  Refusing to report during/outside ratios.")
            return 2

    nper = int(min(256, max(32, on.sum() // 4)))
    rows = []
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 4.6), sharex=True)
    for ax, r in zip(axes.ravel(), rois):
        y = src[r].to_numpy(dtype=float)

        def spec(mask):
            v = y[mask]
            v = v[np.isfinite(v)]
            if v.size < nper:
                return None, None
            f, p = welch(v - np.nanmean(v), fs=fs, nperseg=nper)
            return f, np.sqrt(p)

        f_on, a_on = spec(on)
        f_off, a_off = spec(off)
        if f_on is None or f_off is None:
            ax.set_title(f"{r}  (too few samples)", fontsize=6)
            continue

        def prominence_at(f_axis, amp, freq):
            """Peak amplitude at ``freq`` over the local background."""
            if freq is None or freq > f_axis[-1]:
                return np.nan
            i = int(np.argmin(np.abs(f_axis - freq)))
            nb = np.abs(f_axis - freq) > 2.0
            bg = np.nanmedian(amp[nb])
            return float(amp[i] / bg) if bg > 0 else np.nan

        def ratio_at(freq):
            if freq is None or freq >= fs / 2:
                return np.nan
            i = int(np.argmin(np.abs(f_on - freq)))
            nb = (np.abs(f_on - freq) > 2.0)      # background, >2 Hz away
            bg_on = np.nanmedian(a_on[nb])
            bg_off = np.nanmedian(a_off[nb])
            return float((a_on[i] / bg_on) / (a_off[i] / bg_off))

        nyq = fs / 2
        rows.append({"roi": r,
                     "stim_freq_ratio": ratio_at(stim_f),
                     "stim_prom_on": prominence_at(f_on, a_on, stim_f),
                     "nyq_prom_on": prominence_at(f_on, a_on, nyq),
                     "nyq_prom_off": prominence_at(f_off, a_off, nyq),
                     "nyq_ratio": ratio_at(nyq - 1e-9),
                     "rms_on": float(np.nanstd(y[on])),
                     "rms_off": float(np.nanstd(y[off]))})

        ax.semilogy(f_off, a_off, color="0.6", lw=0.6, label="outside")
        ax.semilogy(f_on, a_on, color="C3", lw=0.6, label="during")
        if stim_f and stim_f < fs / 2:
            ax.axvline(stim_f, color="C0", ls="--", lw=0.6)
        ax.axvline(fs / 2, color="C2", ls=":", lw=0.6)
        ax.set_title(r, fontsize=6)
        ax.tick_params(labelsize=5)
    axes.ravel()[0].legend(fontsize=5, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("amplitude")
    fig.suptitle(f"{sid}  —  spectra during vs outside stimulation "
                 f"(blue dashed = {stim_f} Hz stim, green dotted = Nyquist)",
                 fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p_out = out / f"{sid}_spectra.pdf"
    fig.savefig(p_out, dpi=200)
    plt.close(fig)

    df = pd.DataFrame(rows)
    print("\nratio = (peak/background during stim) / (peak/background outside).")
    print("A value near 1 means the feature is not stimulation-locked.\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    if stim_f:
        bad = df[df["stim_freq_ratio"] > 2.0]
        if len(bad):
            print(f"\n!! {len(bad)} ROIs show a {stim_f} Hz peak >2x stronger "
                  "during stimulation: consistent with laser bleed.")
            print(bad[["roi", "stim_freq_ratio"]].to_string(index=False))
        else:
            print(f"\nNo ROI shows a stimulation-locked {stim_f} Hz peak.")
    nyq = fs / 2
    alt = df[(df["nyq_prom_on"] > 2.0) | (df["nyq_prom_off"] > 2.0)]
    print(f"\n{len(alt)} of {len(df)} ROIs show a {nyq:.0f} Hz "
          "(sample-to-sample alternation) peak at least 2x above their local "
          "background, in either window.")
    if len(alt):
        print(alt[["roi", "nyq_prom_on", "nyq_prom_off", "nyq_ratio"]]
              .to_string(index=False))
        print("\n  This is an alternation between successive samples of the "
              "SAME LED channel: 415/470 crosstalk, a partly wrong LED "
              "assignment, or an alternating gain between flashes.")
        print("  It is not removed by trial averaging when event onsets share "
              "a parity (see the onset-parity check below), because the "
              "alternation then adds coherently across trials.")
        print("  It does NOT bias z or p: the circular-shift null carries the "
              "same alternation, so the effect is measured against it.")
        print("  To remove it cosmetically, low-pass at the indicator "
              "bandwidth (make_figures.py --lowpass 5).")

    # ---- onset parity ----------------------------------------------------- #
    onset_idx = np.searchsorted(t, events["onset_t"].to_numpy())
    par = onset_idx % 2
    frac = max(float(np.mean(par == 0)), float(np.mean(par == 1)))
    print(f"\nonset parity: {frac:.0%} of events land on the same parity "
          f"({len(onset_idx)} events)")
    if frac > 0.9:
        print("  -> trials share the alternation's phase, so averaging "
              "reinforces it rather than cancelling it. Expected when the "
              "inter-stimulus interval is an even number of samples.")
    else:
        print("  -> parities are mixed, so averaging suppresses the "
              "alternation by roughly sqrt(n).")
    df.to_csv(out / f"{sid}_spectra.csv", index=False)
    print(f"\n  {p_out}\n  {sid}_spectra.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
