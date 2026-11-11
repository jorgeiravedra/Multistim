"""
Identify the source of high-frequency noise in multi-fiber photometry.

Three causes have different remedies, and they are distinguishable:

1. **Common-mode artifact** (motion, fiber-bundle flexion, patchcord coupling).
   Moves every fiber together, because the whole bundle moves.  Diagnosed by
   the fraction of high-frequency variance shared across ROIs.  Removable in
   software by projecting out the common component.

2. **Photon shot noise.**  Variance scales linearly with mean intensity across
   ROIs, with a slope set by photons per count.  No software fix: it needs
   more light, longer exposure, or larger ROIs.

3. **Quantisation.**  A floor independent of mean, visible as variance that
   stops falling for the dimmest ROIs.  Fixed by recording more bits.

Also compares the 470 and 415 channels: a fluctuation present in both at the
same relative size is not calcium.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.signal import butter, sosfiltfilt

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--highpass", type=float, default=2.0,
                    help="Hz; above the GCaMP band, so what remains is noise")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.session_dir)
    sid = d.name
    out = Path(args.out or d)

    raw = pd.read_parquet(d / f"{sid}_traces.parquet")
    dff = pd.read_parquet(d / f"{sid}_dff.parquet")
    rois = [c for c in dff.columns if c not in ("t", "frame", "blanked")]

    states = raw["led_state"].to_numpy()
    sig_state = raw.groupby("led_state")[rois].mean().mean(axis=1).idxmax()
    R_sig = raw.loc[states == sig_state, rois].to_numpy(float)
    R_iso = raw.loc[states != sig_state, rois].to_numpy(float)
    n = min(R_sig.shape[0], R_iso.shape[0])
    R_sig, R_iso = R_sig[:n], R_iso[:n]

    t = dff["t"].to_numpy()
    fs = 1.0 / np.median(np.diff(t))
    print(f"=== {sid} ===")
    print(f"{len(rois)} ROIs, {n} samples per channel, {fs:.1f} Hz\n")

    sos = butter(4, args.highpass, "highpass", fs=fs, output="sos")

    def hp(X):
        X = np.asarray(X, float)
        X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0))
        return sosfiltfilt(sos, X, axis=0)

    # ---- 1. shot noise / quantisation: variance vs mean ------------------ #
    mean_c = R_sig.mean(axis=0)
    var_c = hp(R_sig).var(axis=0)
    ok = mean_c > 0
    slope, intercept = np.polyfit(mean_c[ok], var_c[ok], 1)
    r = np.corrcoef(mean_c[ok], var_c[ok])[0, 1]
    print("--- 1. is it photon-limited? ---")
    print(f"variance vs mean across ROIs: slope {slope:.4f} counts, "
          f"intercept {intercept:.4f}, r = {r:.3f}")
    if r > 0.7 and slope > 0:
        print(f"  -> variance scales with intensity: SHOT-NOISE-LIMITED. "
              f"~{1/slope:.2f} photons per count.")
        print("     No software fix. More LED power, longer exposure, or "
              "larger ROIs.")
    else:
        print("  -> variance does NOT track intensity, so the noise is not "
              "photon-limited; look at 2 and 3.")
    if intercept > 0.5 * var_c.mean():
        print(f"  -> a large intensity-independent floor ({intercept:.3f}) "
              "suggests quantisation; the video is 8-bit.")

    # ---- 2. common mode across fibers ------------------------------------ #
    H = hp(R_sig / mean_c)              # fractional, so ROIs are comparable
    C = np.corrcoef(H.T)
    off = C[~np.eye(len(rois), dtype=bool)]
    ev = np.linalg.eigvalsh(np.cov(H.T))[::-1]
    pc1 = float(ev[0] / ev.sum())
    print("\n--- 2. is it common-mode (motion / bundle flexion)? ---")
    print(f"median off-diagonal correlation of high-passed traces: "
          f"{np.median(off):+.3f}")
    print(f"first principal component explains {pc1:.1%} of "
          "high-frequency variance")
    if pc1 > 0.4 or np.median(off) > 0.3:
        print("  -> a large shared component: COMMON-MODE ARTIFACT.")
        print("     Removable: regress out the across-fiber median, or the "
              "first PC, from every ROI.")
        resid = H - np.median(H, axis=1, keepdims=True)
        gain = H.std(axis=0) / resid.std(axis=0)
        print(f"     Removing the across-fiber median would reduce per-ROI "
              f"noise by a median factor of {np.median(gain):.2f}x.")
        print("     CAUTION: genuinely shared neural activity is removed too. "
              "A response at one stimulated site is not common-mode, so this "
              "is comparatively safe here, but it will bias any analysis of "
              "network-wide co-fluctuation.")
    else:
        print("  -> little shared structure: the noise is mostly independent "
              "per fiber, so common-mode removal will not help.")

    # ---- 3. 470 vs 415 ---------------------------------------------------- #
    f_sig = hp(R_sig / R_sig.mean(axis=0)).std(axis=0)
    f_iso = hp(R_iso / np.where(R_iso.mean(axis=0) > 0,
                                R_iso.mean(axis=0), np.nan)).std(axis=0)
    print("\n--- 3. does the isosbestic show the same noise? ---")
    tab = pd.DataFrame({"roi": rois,
                        "mean_470": mean_c,
                        "mean_415": R_iso.mean(axis=0),
                        "hf_sd_470": f_sig,
                        "hf_sd_415": f_iso,
                        "ratio_415_470": f_iso / f_sig})
    print(tab.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    med = float(np.nanmedian(tab["ratio_415_470"]))
    print(f"\nmedian 415/470 high-frequency SD ratio: {med:.2f}")
    if med > 1.5:
        print("  -> the isosbestic is NOISIER than the signal channel, which "
              "is expected if 415 illumination is weak. It is then a poor "
              "motion regressor: it contributes more noise than it explains.")
    elif med > 0.7:
        print("  -> comparable noise in both channels: the fluctuation is NOT "
              "calcium. It is optical or mechanical, and in principle "
              "correctable by the isosbestic.")
    else:
        print("  -> the signal channel is much noisier than the isosbestic, "
              "consistent with the fluctuation being largely biological.")

    tab.to_csv(out / f"{sid}_noise.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    axes[0].scatter(mean_c, var_c, s=12, color="C0")
    xx = np.linspace(mean_c.min(), mean_c.max(), 10)
    axes[0].plot(xx, slope * xx + intercept, color="C3", lw=0.8)
    axes[0].set_xlabel("mean ROI intensity (counts)")
    axes[0].set_ylabel("high-frequency variance")
    axes[0].set_title(f"shot-noise test (r={r:.2f})", fontsize=7)

    im = axes[1].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(rois)))
    axes[1].set_xticklabels(rois, rotation=90, fontsize=5)
    axes[1].set_yticks(range(len(rois)))
    axes[1].set_yticklabels(rois, fontsize=5)
    axes[1].set_title(f"cross-fiber corr (PC1={pc1:.0%})", fontsize=7)
    fig.colorbar(im, ax=axes[1], fraction=0.045)

    axes[2].scatter(f_sig, f_iso, s=12, color="C2")
    lim = max(np.nanmax(f_sig), np.nanmax(f_iso))
    axes[2].plot([0, lim], [0, lim], color="0.5", lw=0.6, ls="--")
    axes[2].set_xlabel("470 high-freq SD")
    axes[2].set_ylabel("415 high-freq SD")
    axes[2].set_title("signal vs isosbestic noise", fontsize=7)
    for ax in axes:
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{sid} — noise diagnosis (high-pass {args.highpass} Hz)",
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / f"{sid}_noise.pdf", dpi=200)
    plt.close(fig)
    print(f"\n  {sid}_noise.pdf\n  {sid}_noise.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
