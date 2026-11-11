"""
Multi-page QC report for one processed session.

Diagnostics that belong beside the peri-event figures but not in them:

1. **Channel quality.**  Raw intensity, high-frequency noise, isosbestic
   level and R^2, and the resulting detection floor, per fiber.  Sensitivity
   varies by orders of magnitude between fibers in one bundle; a figure that
   does not show this invites an effect in a quiet channel to be compared
   against a noisy channel's null.

2. **Effect versus burst index.**  Photobleaching, opsin desensitisation and
   habituation all produce a monotone trend across repetitions.  A response
   present only in the first burst, or only the last, is not the same claim as
   a response present throughout.

3. **Within-burst frequency content.**  A testable NEGATIVE prediction: a
   10 Hz burst cannot drive a 10 Hz modulation of GCaMP6f, whose decay
   constant (~0.4 s) low-passes far below the burst rate.  Any power at the
   burst frequency in the response window is therefore artifact, not signal,
   and this is the cleanest internal test for stimulation bleed the design
   affords.

4. **Timing and alignment.**  Frame anchoring, gate phase, blanking, LED
   separation and parity, drawn from the manifest.

5. **Baseline quality.**  Per-trial baseline level and slope across the
   session, which is what determines whether a contrast is measuring a
   response or a trend that crossed the stimulus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    import pandas as pd
    from scipy.signal import welch

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--pre", type=float, default=2.0)
    ap.add_argument("--post", type=float, default=4.0)
    ap.add_argument("--baseline-window", type=float, default=0.5)
    ap.add_argument("--response-window", type=float, default=0.5)
    ap.add_argument("--stim-freq", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib as mpl
    mpl.rcParams.update({"pdf.fonttype": 42, "font.size": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "lines.linewidth": 0.8})

    d = Path(args.session_dir)
    sid = d.name
    man = json.loads((d / f"{sid}_manifest.json").read_text(encoding="utf-8"))
    qc = man.get("qc", {})
    dff = pd.read_parquet(d / f"{sid}_dff.parquet")
    events = pd.read_parquet(d / f"{sid}_events.parquet")
    traces = pd.read_parquet(d / f"{sid}_traces.parquet")
    rois = [c for c in dff.columns
            if c not in ("t", "frame", "blanked") and not c.endswith("__iso")]
    t = dff["t"].to_numpy()
    fs = 1.0 / np.median(np.diff(t))

    stim_f = args.stim_freq
    if stim_f is None:
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*Hz", sid, re.I)
        stim_f = float(m.group(1)) if m else None

    npre, npost = int(args.pre * fs), int(args.post * fs)
    nbase, nresp = int(args.baseline_window * fs), int(args.response_window * fs)

    out = Path(args.out or d / f"{sid}_qc.pdf")
    with PdfPages(out) as pdf:

        # ---------- page 1: channel quality ------------------------------- #
        sig_state = traces.groupby("led_state")[rois].mean().mean(axis=1).idxmax()
        R = traces.loc[traces.led_state == sig_state, rois].to_numpy(float)
        mean_c = np.nanmean(R, axis=0)
        hf = np.array([np.nanstd(np.diff(dff[r].to_numpy(float))) / np.sqrt(2)
                       for r in rois])
        iso_lv = qc.get("isosbestic_f0_median") or [np.nan] * len(rois)
        r2 = qc.get("regression_r2") or [np.nan] * len(rois)
        sd_after = qc.get("dff_sd_after_regression") or [np.nan] * len(rois)

        fig, axes = plt.subplots(2, 3, figsize=(9, 5))
        panels = [
            ("mean 470 intensity (counts)", mean_c, False),
            ("mean 415 intensity (counts)", np.asarray(iso_lv, float), False),
            ("high-frequency SD (dF/F)", hf, True),
            ("total dF/F SD", np.asarray(sd_after, float), True),
            ("isosbestic R$^2$", np.asarray(r2, float), False),
        ]
        order = np.argsort(mean_c)
        for ax, (lab, v, logy) in zip(axes.ravel(), panels):
            ax.barh(np.arange(len(rois)), np.asarray(v)[order], color="C0")
            ax.set_yticks(np.arange(len(rois)))
            ax.set_yticklabels([rois[i] for i in order], fontsize=5)
            ax.set_xlabel(lab, fontsize=6)
            if logy:
                ax.set_xscale("log")
            ax.tick_params(labelsize=5)
        axes.ravel()[-1].axis("off")
        axes.ravel()[-1].text(
            0, 0.95,
            "Sensitivity is a property of the recorded channel.\n"
            "A 60x spread in noise across one bundle means an\n"
            "effect must be judged against its own channel's\n"
            "floor, never a session-wide median.\n\n"
            f"n events: {len(events)}\n"
            f"effective rate: {fs:.1f} Hz\n"
            f"blanked samples: {qc.get('n_blanked')}\n"
            f"LED separation: {qc.get('led_separation', float('nan')):.1f}\n"
            f"LED parity breaks: {qc.get('led_parity_breaks')}\n"
            f"frame anchor: {qc.get('frame_anchor_confidence')} "
            f"(risk {qc.get('frame_anchor_risk_ms', float('nan')):.0f} ms)",
            va="top", fontsize=5.5, family="monospace")
        fig.suptitle(f"{sid} — channel quality", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig); plt.close(fig)

        # ---------- page 2: effect vs burst index -------------------------- #
        fig, axes = plt.subplots(3, 4, figsize=(9.5, 6), sharex=True)
        for ax, r in zip(axes.ravel(), rois):
            y = dff[r].to_numpy(float)
            for site, g in events.groupby("region"):
                if site is None:
                    continue
                ons = np.sort(g["onset_t"].to_numpy(float))
                idx = np.searchsorted(t, ons)
                keep = (idx - npre >= 0) & (idx + npost < t.size)
                idx = idx[keep]
                if idx.size < 3:
                    continue
                W = np.stack([y[i - npre:i + npost] for i in idx])
                base = np.nanmean(W[:, npre - nbase:npre], axis=1)
                eff = np.nanmean(W[:, npre:npre + nresp], axis=1) - base
                ax.plot(np.arange(eff.size), eff, lw=0.5, alpha=0.6,
                        marker="o", ms=1.5,
                        color="C3" if site == r else "0.6")
            ax.axhline(0, color="k", lw=0.4, ls=":")
            ax.set_title(r, fontsize=6)
            ax.tick_params(labelsize=5)
        for ax in axes[-1]:
            ax.set_xlabel("burst index within session")
        for ax in axes[:, 0]:
            ax.set_ylabel("effect (dF/F)")
        fig.suptitle(f"{sid} — effect vs repetition "
                     "(red: self-stimulation; a monotone trend indicates "
                     "bleaching, desensitisation or habituation)", fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig); plt.close(fig)

        # ---------- page 3: frequency content in the response -------------- #
        if stim_f:
            fig, axes = plt.subplots(3, 4, figsize=(9.5, 6), sharex=True)
            nper = int(min(64, max(16, nresp)))
            for ax, r in zip(axes.ravel(), rois):
                y = dff[r].to_numpy(float)
                segs_r, segs_b = [], []
                for o in events["onset_t"].to_numpy(float):
                    k = int(np.searchsorted(t, o))
                    if k - npre < 0 or k + npost >= t.size:
                        continue
                    segs_r.append(y[k:k + nresp])
                    segs_b.append(y[k - nresp:k])
                if len(segs_r) < 3:
                    continue
                def sp(S):
                    S = np.vstack(S)
                    S = np.where(np.isfinite(S), S, 0.0)
                    f, P = welch(S - S.mean(axis=1, keepdims=True),
                                 fs=fs, nperseg=min(nper, S.shape[1]), axis=1)
                    return f, np.sqrt(P.mean(axis=0))
                fr, ar = sp(segs_r)
                fb, ab = sp(segs_b)
                ax.semilogy(fb, ab, color="0.6", lw=0.6)
                ax.semilogy(fr, ar, color="C3", lw=0.6)
                if stim_f < fs / 2:
                    ax.axvline(stim_f, color="C0", ls="--", lw=0.6)
                ax.set_title(r, fontsize=6)
                ax.tick_params(labelsize=5)
            for ax in axes[-1]:
                ax.set_xlabel("frequency (Hz)")
            fig.suptitle(
                f"{sid} — spectrum in the response window (red) vs an equal "
                f"pre-stimulus window (grey). GCaMP6f cannot follow {stim_f:g} Hz "
                "(blue dashed): power there is artifact, not signal.",
                fontsize=6.5)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig); plt.close(fig)

        # ---------- page 4: baseline level and slope over the session ------ #
        fig, axes = plt.subplots(3, 4, figsize=(9.5, 6), sharex=True)
        for ax, r in zip(axes.ravel(), rois):
            y = dff[r].to_numpy(float)
            ons = np.sort(events["onset_t"].to_numpy(float))
            idx = np.searchsorted(t, ons)
            keep = (idx - npre >= 0) & (idx + npost < t.size)
            idx, ons = idx[keep], ons[keep]
            base, slope = [], []
            x = np.arange(nbase, dtype=float)
            for i in idx:
                seg = y[i - nbase:i]
                g = np.isfinite(seg)
                base.append(np.nanmean(seg))
                if g.sum() > 2:
                    xm, ym = x[g].mean(), seg[g].mean()
                    den = ((x[g] - xm) ** 2).sum()
                    slope.append(((x[g] - xm) * (seg[g] - ym)).sum() / den * fs
                                 if den > 0 else np.nan)
                else:
                    slope.append(np.nan)
            ax.plot(ons, base, color="C0", lw=0.5, marker="o", ms=1.5)
            ax2 = ax.twinx()
            ax2.plot(ons, slope, color="C1", lw=0.5, marker="s", ms=1.5)
            ax2.tick_params(labelsize=4, colors="C1")
            ax2.spines["right"].set_visible(True)
            ax.axhline(0, color="k", lw=0.4, ls=":")
            ax.set_title(r, fontsize=6)
            ax.tick_params(labelsize=5)
        for ax in axes[-1]:
            ax.set_xlabel("session time (s)")
        fig.suptitle(f"{sid} — per-trial baseline level (blue) and slope "
                     "(orange, dF/F per s). A contrast is only a response if "
                     "the slope does not predict it.", fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig); plt.close(fig)

    print(f"  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
