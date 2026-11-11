"""
Choose the baseline timescale from data, not by eye.

Two things trade off as the F0 timescale ``tau`` is shortened:

* **noise falls** -- a baseline that follows slow drift removes it from
  dF/F, lowering the null SD and with it the detection floor;
* **signal is absorbed** -- a baseline fast enough to follow the response
  subtracts it away.

Neither is visible by looking at a trace.  This measures both: for each tau
it reports the resulting null SD (the floor) and the recovery of a synthetic
response of known amplitude and the session's own shape, injected into the
real data and put through the identical pipeline.  Pick the shortest tau that
still recovers the response.

A hard constraint the curve cannot show: tau must stay well above the
interval between repeats of one site.  If F0 can follow a pattern recurring
every ISI seconds, it will track the envelope of the responses themselves.
Four to five times the same-site interval is a safe lower bound, and is
flagged below.

Also reported is the asymmetry of the result.  airPLS fits the LOWER envelope
by design -- it was built for spectroscopy, where peaks point up -- so dF/F
comes out positively skewed and most samples sit above zero.  That does not
bias a peri-event contrast, which re-baselines each trial locally, but it
does mean the absolute dF/F scale is not centred, and for a preparation whose
response is a decrease the convention is worth knowing about.
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

    from multifiber import preprocess as pp

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--taus", default="5,10,20,30,60,120",
                    help="baseline timescales to test, seconds")
    ap.add_argument("--inject-amp", type=float, default=0.01,
                    help="amplitude of the synthetic response, dF/F")
    ap.add_argument("--train-s", type=float, default=None,
                    help="synthetic response duration; defaults to the "
                         "measured burst duration")
    ap.add_argument("--decay-tau", type=float, default=0.4)
    ap.add_argument("--pre", type=float, default=2.0)
    ap.add_argument("--post", type=float, default=4.0)
    ap.add_argument("--baseline-window", type=float, default=0.5)
    ap.add_argument("--response-window", type=float, default=None)
    ap.add_argument("--rois", default=None, help="comma-separated subset")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import json
    import matplotlib as mpl
    mpl.rcParams.update({"pdf.fonttype": 42, "font.size": 7,
                         "axes.spines.top": False, "axes.spines.right": False})

    d = Path(args.session_dir)
    sid = d.name
    man = json.loads((d / f"{sid}_manifest.json").read_text(encoding="utf-8"))
    qc = man.get("qc", {})
    proto = qc.get("protocol") or {}
    traces = pd.read_parquet(d / f"{sid}_traces.parquet")
    events = pd.read_parquet(d / f"{sid}_events.parquet")

    rois = [c for c in traces.columns
            if c not in ("frame", "t", "led_state", "roi_mean")]
    if args.rois:
        want = {x.strip() for x in args.rois.split(",")}
        rois = [r for r in rois if r in want]

    led = traces["led_state"].to_numpy(int)
    sig_state = int(traces.groupby("led_state")[rois].mean()
                    .mean(axis=1).idxmax())
    m = led == sig_state
    t = traces.loc[m, "t"].to_numpy(float)
    fs = 1.0 / np.median(np.diff(t))
    raw = traces.loc[m, rois].to_numpy(float)

    train = args.train_s or proto.get("burst_duration_s") or 0.5
    nresp = int((args.response_window
                 or min(train + 3 * args.decay_tau, 2.0)) * fs)
    npre, npost = int(args.pre * fs), int(args.post * fs)
    nbase = int(args.baseline_window * fs)
    isi = proto.get("same_site_interval_s")

    onsets = events["onset_t"].to_numpy(float)
    idx = np.searchsorted(t, onsets)
    idx = idx[(idx - npre >= 0) & (idx + npost < t.size)]

    print(f"=== {sid} ===")
    print(f"{len(rois)} fibers, {t.size} samples at {fs:.1f} Hz "
          f"({(t[-1] - t[0]):.0f} s), {idx.size} events")
    print(f"synthetic response: {args.inject_amp:.1%} dF/F, {train:.2f} s "
          f"train, {args.decay_tau:.2f} s decay; response window "
          f"{nresp / fs:.2f} s")
    if isi:
        print(f"same-site interval {isi:.1f} s -> tau should exceed "
              f"~{4 * isi:.0f} s to avoid F0 tracking the repeated response")
    print()

    # synthetic response, same shape for every injected trial
    tt = np.arange(-npre, npost) / fs
    s_ = np.arange(0, 6, 1 / fs)
    h = (1 - np.exp(-s_ / 0.07)) * np.exp(-s_ / args.decay_tau)
    kern = np.convolve(((tt >= 0) & (tt < train)).astype(float), h)[:tt.size]
    kern = kern / kern.max()

    taus = [float(x) for x in args.taus.split(",")]
    rows = []
    rng = np.random.default_rng(0)
    for tau in taus:
        lam = pp.lambda_from_timescale(tau, fs, 2)
        floors, recov, skews, abovef = [], [], [], []
        for j, r in enumerate(rois):
            y = raw[:, j]
            # inject into the RAW trace, before any baseline fitting
            y_inj = y.copy()
            f0_rough = pp.airpls(y, lambda_=lam, differences=2)
            for i in idx:
                a, b = i - npre, i + npost
                y_inj[a:b] = y_inj[a:b] + args.inject_amp * kern * f0_rough[a:b]

            def contrast(trace):
                f0 = pp.airpls(trace, lambda_=lam, differences=2)
                dff = (trace - f0) / f0
                W = np.stack([dff[i - npre:i + npost] for i in idx])
                W = W - np.nanmean(W[:, npre - nbase:npre], axis=1,
                                   keepdims=True)
                mean = np.nanmean(W, axis=0)
                return dff, float(np.nanmean(mean[npre:npre + nresp]))

            dff0, e0 = contrast(y)
            _, e1 = contrast(y_inj)
            # expected amplitude of the injected response over the window
            expect = args.inject_amp * float(np.mean(kern[npre:npre + nresp]))
            recov.append((e1 - e0) / expect if expect else np.nan)

            # floor: SD of the same contrast at random alignments
            pool = np.arange(npre, dff0.size - npost)
            draws = rng.choice(pool, size=min(400, pool.size), replace=False)
            N = np.stack([dff0[k - npre:k + npost] for k in draws])
            N = N - np.nanmean(N[:, npre - nbase:npre], axis=1, keepdims=True)
            per = np.nanmean(N[:, npre:npre + nresp], axis=1)
            floors.append(3 * float(np.nanstd(per) / np.sqrt(idx.size)))

            finite = dff0[np.isfinite(dff0)]
            skews.append(float(pd.Series(finite).skew()))
            abovef.append(float(np.mean(finite > 0)))

        rows.append({"tau_s": tau,
                     "median_floor": float(np.median(floors)),
                     "median_recovery": float(np.median(recov)),
                     "worst_recovery": float(np.min(recov)),
                     "median_skew": float(np.median(skews)),
                     "frac_above_F0": float(np.median(abovef))})

    rdf = pd.DataFrame(rows)
    print(rdf.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    ok = rdf[rdf["worst_recovery"] > 0.9]
    if isi:
        ok = ok[ok["tau_s"] >= 4 * isi]
    print("\n--- choosing tau ---")
    if len(ok):
        best = ok.loc[ok["median_floor"].idxmin()]
        print(f"  shortest tau that recovers >90% of the injected response "
              f"in every fiber{' and clears 4x the same-site interval' if isi else ''}: "
              f"{best.tau_s:g} s, floor {best.median_floor:.3%}")
        cur = man.get("config", {}).get("baseline_tau_s")
        if cur:
            row = rdf[rdf.tau_s == cur]
            if len(row):
                print(f"  current setting {cur:g} s gives floor "
                      f"{row.iloc[0].median_floor:.3%} and recovery "
                      f"{row.iloc[0].median_recovery:.0%}")
                gain = row.iloc[0].median_floor / best.median_floor
                if gain > 1.2:
                    print(f"  -> switching would lower the floor {gain:.1f}x "
                          "at no cost in recovery.")
    else:
        print("  No tau both recovers the response and clears the "
              "same-site interval. The response is too close to the drift "
              "timescale to separate by baseline fitting alone.")
    print(f"\n  dF/F is positively skewed (median skew "
          f"{rdf['median_skew'].median():+.2f}; "
          f"{rdf['frac_above_F0'].median():.0%} of samples above F0) because "
          "airPLS fits the lower envelope by design. This does not bias a "
          "peri-event contrast, which re-baselines each trial, but the "
          "absolute dF/F scale is not centred on zero.")

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    axes[0].plot(rdf.tau_s, rdf.median_floor * 100, marker="o", color="C0")
    axes[0].set_xscale("log"); axes[0].set_xlabel("baseline tau (s)")
    axes[0].set_ylabel("detection floor (%)")
    axes[1].plot(rdf.tau_s, rdf.median_recovery, marker="o", color="C2",
                 label="median")
    axes[1].plot(rdf.tau_s, rdf.worst_recovery, marker="s", color="C1",
                 lw=0.8, label="worst fiber")
    axes[1].axhline(1.0, color="k", lw=0.5, ls=":")
    axes[1].axhline(0.9, color="C3", lw=0.5, ls="--")
    if isi:
        for ax in axes:
            ax.axvspan(ax.get_xlim()[0], 4 * isi, color="C3", alpha=0.10, lw=0)
    axes[1].set_xscale("log"); axes[1].set_xlabel("baseline tau (s)")
    axes[1].set_ylabel("recovery of injected response")
    axes[1].legend(fontsize=5, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{sid} — baseline timescale: noise vs signal loss "
                 "(shaded: too fast relative to the same-site interval)",
                 fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(args.out or d / f"{sid}_baseline_tuning.pdf")
    fig.savefig(out, dpi=200); plt.close(fig)
    rdf.to_csv(d / f"{sid}_baseline_tuning.csv", index=False)
    print(f"\n  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
