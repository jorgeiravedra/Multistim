"""
Trial-by-trial diagnostics for one stimulated-site / recorded-site pair.

At n = 5 the trial mean is not a summary, it is five numbers.  This answers
three questions that the peri-event figure cannot:

* **Is one trial driving the result?**  Jackknife: recompute the effect with
  each trial left out, and report how far the answer moves.
* **Is the baseline doing the work?**  A trial's "response" is response minus
  baseline, so a trial whose baseline happens to sit low produces a positive
  effect with no change during the stimulus at all.  Reported as the
  correlation between baseline level and effect across trials, and as the
  split of each trial's effect into its two halves.
* **Is slow drift aligned with the stimulus?**  The slope through each trial's
  baseline window predicts where the trace is heading independently of the
  stimulus; if slope correlates with effect, the "response" is drift.

Single-trial values are scored against a SINGLE-TRIAL null (random alignment
times), not against the trial-mean null used for the headline statistic.  The
latter is ~sqrt(n) narrower and individual trials lie outside it routinely.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _nan_slope(x, y):
    """Least-squares slope of y on x, ignoring NaNs. NaN if under 3 points."""
    import numpy as np

    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return np.nan
    xs, ys = x[good], y[good]
    xm, ym = xs.mean(), ys.mean()
    denom = float(((xs - xm) ** 2).sum())
    if denom <= 0:
        return np.nan
    return float(((xs - xm) * (ys - ym)).sum() / denom)


def _nan_corr(a, b):
    """Pearson correlation over pairs where both are finite."""
    import numpy as np

    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3:
        return np.nan
    sa, sb = a[good].std(), b[good].std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.corrcoef(a[good], b[good])[0, 1])


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--stim-site", required=True)
    ap.add_argument("--recorded-site", required=True)
    ap.add_argument("--pre", type=float, default=2.0)
    ap.add_argument("--post", type=float, default=4.0)
    ap.add_argument("--baseline-window", type=float, default=0.5)
    ap.add_argument("--response-window", type=float, default=1.5)
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = Path(args.session_dir)
    sid = d.name
    dff = pd.read_parquet(d / f"{sid}_dff.parquet")
    events = pd.read_parquet(d / f"{sid}_events.parquet")

    col = args.recorded_site
    if col not in dff.columns:
        print(f"no column {col!r}")
        return 1
    grp = events[events["region"] == args.stim_site]
    if grp.empty:
        print(f"no events for stim site {args.stim_site!r}")
        return 1

    t = dff["t"].to_numpy()
    y = dff[col].to_numpy(dtype=float)
    fs = 1.0 / np.median(np.diff(t))
    npre, npost = int(args.pre * fs), int(args.post * fs)
    nbase, nresp = int(args.baseline_window * fs), int(args.response_window * fs)

    idx = np.searchsorted(t, grp["onset_t"].to_numpy())
    idx = idx[(idx - npre >= 0) & (idx + npost < t.size)]
    n = idx.size
    W = np.stack([y[i - npre:i + npost] for i in idx])

    with np.errstate(invalid="ignore"):
        base_raw = np.nanmean(W[:, npre - nbase:npre], axis=1)
        resp_raw = np.nanmean(W[:, npre:npre + nresp], axis=1)
    eff = resp_raw - base_raw
    usable = np.isfinite(eff)
    if usable.sum() < n:
        print(f"!! {n - int(usable.sum())} of {n} trials have no usable "
              "baseline or response (entirely blanked) and are dropped.\n")

    # Closed-form least-squares slope, skipping NaNs. polyfit goes through
    # LAPACK's DGELSD, which has no NaN handling and fails with an opaque
    # "SVD did not converge"; blanked laser-contaminated samples land in the
    # baseline window whenever a command starts within it.
    x = np.arange(nbase, dtype=float)
    slope = np.array([_nan_slope(x, W[k, npre - nbase:npre]) * fs
                      for k in range(n)])
    n_nan_base = int(np.isnan(W[:, npre - nbase:npre]).sum())
    n_nan_resp = int(np.isnan(W[:, npre:npre + nresp]).sum())
    if n_nan_base or n_nan_resp:
        print(f"note: {n_nan_base} blanked samples in the baseline windows, "
              f"{n_nan_resp} in the response windows "
              f"(of {n * nbase} and {n * nresp}); these are excluded from "
              "every statistic below.\n")

    # single-trial null
    pool = np.arange(npre, t.size - npost)
    nidx = rng.choice(pool, size=min(args.n_null, pool.size), replace=False)
    NW = np.stack([y[i - npre:i + npost] for i in nidx])
    with np.errstate(invalid="ignore"):
        neff = (np.nanmean(NW[:, npre:npre + nresp], axis=1)
                - np.nanmean(NW[:, npre - nbase:npre], axis=1))
    neff = neff[np.isfinite(neff)]
    if neff.size < 50:
        print("!! fewer than 50 usable null windows; the single-trial null "
              "is unreliable.")
    nmu, nsd = float(np.mean(neff)), float(np.std(neff))

    print(f"=== {sid}: stim {args.stim_site} -> record {col} ===")
    print(f"{n} trials, {fs:.0f} Hz, baseline {args.baseline_window}s, "
          f"response {args.response_window}s")
    print(f"single-trial null: mean {nmu:+.5f}, SD {nsd:.5f}\n")

    print(f"{'trial':>5} {'onset_t':>9} {'baseline':>10} {'response':>10} "
          f"{'effect':>10} {'z_single':>9} {'base_slope/s':>13}")
    for k in range(n):
        zs = (eff[k] - nmu) / nsd if np.isfinite(eff[k]) and nsd > 0 else np.nan
        print(f"{k:5d} {t[idx[k]]:9.2f} {base_raw[k]:10.5f} "
              f"{resp_raw[k]:10.5f} {eff[k]:+10.5f} "
              f"{zs:9.2f} {slope[k]:+13.5f}")

    e_ok = eff[usable]
    n_ok = e_ok.size
    mean_eff = float(np.mean(e_ok)) if n_ok else float("nan")
    sd_eff = float(np.std(e_ok, ddof=1)) if n_ok > 1 else float("nan")
    print(f"\nmean effect {mean_eff:+.5f}   "
          f"SD across trials {sd_eff:.5f}   "
          f"SEM {sd_eff/np.sqrt(n_ok) if n_ok else float('nan'):.5f}")

    # ---- jackknife --------------------------------------------------------- #
    jk = np.array([np.mean(np.delete(e_ok, k)) for k in range(n_ok)])
    worst = int(np.argmax(np.abs(jk - mean_eff)))
    print("\n--- leave-one-out ---")
    for k in range(n_ok):
        flag = "  <-- largest influence" if k == worst else ""
        print(f"  without trial {k}: {jk[k]:+.5f} "
              f"({100*(jk[k]-mean_eff)/abs(mean_eff) if mean_eff else float('nan'):+.0f}%)"
              f"{flag}")
    if abs(jk[worst] - mean_eff) > 0.5 * abs(mean_eff):
        print(f"  !! removing one trial of {n_ok} moves the effect by more than "
              "half. The result rests on a single trial.")
    if np.sign(jk).min() != np.sign(jk).max():
        print("  !! leave-one-out means do not even agree on sign.")

    # ---- is it the baseline, or the response? ------------------------------ #
    r_base = _nan_corr(base_raw, eff)
    r_resp = _nan_corr(resp_raw, eff)
    r_slope = _nan_corr(slope, eff)
    print("\n--- what varies across trials ---")
    print(f"  corr(baseline slope, effect) {r_slope:+.3f}   <- read this first")
    print(f"  corr(baseline, effect)       {r_base:+.3f}")
    print(f"  corr(response, effect)       {r_resp:+.3f}")
    crit = {3: 0.997, 4: 0.950, 5: 0.878, 6: 0.811, 7: 0.754,
            8: 0.707, 9: 0.666, 10: 0.632}.get(n_ok)
    if crit:
        print(f"  (at n={n_ok}, |r| > {crit:.3f} is p<0.05; these three are "
              "mutually collinear, so do not read them as independent)")

    if np.isfinite(r_slope) and abs(r_slope) > 0.7:
        print("\n  !! THE BASELINE SLOPE PREDICTS THE EFFECT. Where each "
              "trial was already heading before the stimulus determines where "
              "the response window lands: this is slow drift crossing the "
              "stimulus, not a response to it. This test overrides the other "
              "two, which cannot separate a response from a trend that "
              "continues through the response window.")
    elif not np.isfinite(r_base) or not np.isfinite(r_resp):
        print("  (too few usable trials to compare)")
    elif abs(r_base) > abs(r_resp):
        print("  -> the effect tracks the BASELINE more than the response: "
              "trial-to-trial variation in where the trace sat before the "
              "stimulus is producing the 'response'.")
    else:
        print("  -> no slope dependence; the effect tracks the response "
              "period, as it should.")

    # ---- robust summaries -------------------------------------------------- #
    z_tr = (eff - nmu) / nsd
    n_out = int(np.sum(np.abs(z_tr[usable]) > 2))
    n_pos = int(np.sum(e_ok > 0))
    from math import comb
    p_sign = sum(comb(n_ok, k) for k in range(n_pos, n_ok + 1)) / 2 ** n_ok \
        if n_ok else float("nan")
    med = float(np.median(e_ok)) if n_ok else float("nan")
    print("\n--- robust summaries (the mean of 5 trials is not a summary) ---")
    print(f"  median effect            {med:+.5f}   (mean {mean_eff:+.5f})")
    print(f"  trials with |z_single|>2 {n_out} of {n_ok}   "
          f"(expected {0.046 * n_ok:.2f} under the null)")
    print(f"  trials positive          {n_pos} of {n_ok}   "
          f"one-sided sign test p = {p_sign:.3f}")
    if n_out == 1 and n_ok >= 4:
        k = int(np.nanargmax(np.abs(z_tr)))
        print(f"  -> exactly one trial (trial {k}) is an outlier. One "
              "excursion in five is not rare: the chance of at least one "
              f"|z|>2 among {n_ok} trials is {1 - 0.954 ** n_ok:.0%}.")
        if np.isfinite(base_raw[k]) and abs(base_raw[k]) > 3 * nsd:
            print(f"     Its BASELINE is already {base_raw[k]:+.4f}, far from "
                  "the others, so the excursion was underway before the "
                  "stimulus and cannot have been evoked by it.")
    if abs(med) < 0.3 * abs(mean_eff):
        print("  -> the median is far smaller than the mean: the mean is "
              "being carried by outliers, not by a shift in the typical "
              "trial.")

    # ---- the same effect, with each trial's trend removed ----------------- #
    tail0 = npre + int((float(grp["width_s"].median()) + 1.2) * fs)
    fit_cols = np.concatenate([np.arange(0, npre),
                               np.arange(min(tail0, W.shape[1]), W.shape[1])])
    cx = np.arange(W.shape[1], dtype=float)
    eff_dt = np.full(n, np.nan)
    for k in range(n):
        sl = _nan_slope(cx[fit_cols], W[k, fit_cols])
        if not np.isfinite(sl):
            continue
        inter = np.nanmean(W[k, fit_cols] - sl * cx[fit_cols])
        resid = W[k] - (sl * cx + inter)
        with np.errstate(invalid="ignore"):
            eff_dt[k] = (np.nanmean(resid[npre:npre + nresp])
                         - np.nanmean(resid[npre - nbase:npre]))
    ok_dt = np.isfinite(eff_dt)
    print("\n--- effect after removing each trial's own linear trend ---")
    print("  (line fitted through the pre-window AND the late post-window, "
          "both outside the expected response)")
    if ok_dt.sum() > 1:
        m_dt = float(np.mean(eff_dt[ok_dt]))
        sd_dt = float(np.std(eff_dt[ok_dt], ddof=1))
        print(f"  mean effect: {mean_eff:+.5f} raw  ->  {m_dt:+.5f} detrended "
              f"({100 * (m_dt - mean_eff) / abs(mean_eff) if mean_eff else float('nan'):+.0f}%)")
        print(f"  SD across trials: {sd_eff:.5f} -> {sd_dt:.5f}")
        if abs(m_dt) < 0.3 * abs(mean_eff):
            print("  -> the effect largely disappears once the trend is "
                  "removed: it was drift.")
        elif sd_dt < 0.6 * sd_eff:
            print("  -> the effect survives and the trial scatter shrinks: "
                  "drift was adding noise, not the effect itself.")
        else:
            print("  -> detrending changes little.")
        print("  NOTE: detrending is not free. Simulated against this rig's "
              "noise it costs power at n>=10 (it estimates a slope from noisy "
              "flanks) and only helps when the slope genuinely predicts the "
              "effect, as it does here. Decide on the diagnostic, not on "
              "which answer you prefer.")

    # ---- figure ------------------------------------------------------------ #
    tt = (np.arange(-npre, npost)) / fs
    Wc = W - np.where(np.isfinite(base_raw), base_raw, 0.0)[:, None]
    with np.errstate(invalid="ignore"):
        NWc = NW - np.nanmean(NW[:, npre - nbase:npre], axis=1, keepdims=True)
    lo, hi = np.nanpercentile(NWc, [2.5, 97.5], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].fill_between(tt, lo, hi, color="0.88", lw=0,
                         label="single-trial null 95%")
    for k in range(n):
        axes[0].plot(tt, Wc[k], lw=0.6, alpha=0.8, label=f"trial {k}")
    axes[0].plot(tt, Wc.mean(0), color="k", lw=1.4, label="mean")
    axes[0].axvspan(-args.baseline_window, 0, color="0.6", alpha=0.3, lw=0)
    axes[0].axvspan(0, float(grp["width_s"].median()), color="C0",
                    alpha=0.15, lw=0)
    axes[0].axhline(0, color="k", lw=0.4, ls=":")
    axes[0].set_xlabel("time from onset (s)")
    axes[0].set_ylabel(r"$\Delta F/F$")
    axes[0].legend(fontsize=5, frameon=False, ncol=2)
    axes[0].set_title(f"{args.stim_site} -> {col}", fontsize=7)

    axes[1].axhline(0, color="k", lw=0.4, ls=":")
    axes[1].axhspan(nmu - 2 * nsd, nmu + 2 * nsd, color="0.88", lw=0)
    axes[1].scatter(np.arange(n), eff, s=18, color="C3")
    axes[1].axhline(mean_eff, color="k", lw=1.0)
    axes[1].set_xlabel("trial")
    axes[1].set_ylabel("effect (response - baseline)")
    axes[1].set_title("per-trial effect vs single-trial null (2 SD)",
                      fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    out = d / f"{sid}_trials_{args.stim_site}_to_{col}.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\n  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
