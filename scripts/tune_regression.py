"""
Grid search the motion regression, per fiber, on an isosbestic criterion.

Which predictor set and which estimator work best is not the same for every
fiber -- a dim or noisy isosbestic behaves differently from a clean one --
and picking by eye across twelve fibers is both slow and unreliable.

The criterion is built from the isosbestic, not from an assumed response
shape.  If the 415 channel is the arbiter of what counts as artifact, then a
good fit leaves no 415-correlated structure in the 470 residual.  Two
quantities are therefore reported per configuration:

* ``r_stim``  -- |corr(residual 470, own 415)| INSIDE the stimulation
  windows.  This is the number to minimise: whatever the laser does to the
  isosbestic, it should not survive in the signal channel.
* ``mse_out`` -- out-of-sample squared error on held-out time blocks,
  scored OUTSIDE the stimulation windows.  This guards against a
  configuration that scrubs the stimulus window by fitting noise
  everywhere else.

A configuration wins by having the lowest ``r_stim`` among those whose
``mse_out`` is within a tolerance of the best.  Ranking on ``r_stim`` alone
would reward overfitting; ranking on ``mse_out`` alone is what the automatic
selector does, and it has no reason to care about the artifact at all.

What this cannot do: tell you whether a stimulus-locked deflection that
survives is neural.  It can only say that the isosbestic no longer explains
it.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def main() -> int:
    import numpy as np
    import pandas as pd

    from multifiber import preprocess as pp
    from multifiber import variants as V

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--predictors", default="same,common,others,all")
    ap.add_argument("--common-methods",
                    default="median,pc1,pc2,pc3,pc5,pc7")
    ap.add_argument("--methods", default="lasso,ols,hierarchical")
    ap.add_argument("--alphas", default="cv,1e-4,1e-6")
    ap.add_argument("--head-methods", default="ols,lasso")
    ap.add_argument("--tail-methods", default="lasso,ols")
    ap.add_argument("--rois", default=None)
    ap.add_argument("--mse-tolerance", type=float, default=0.05,
                    help="a configuration stays in contention if its "
                         "out-of-sample error is within this fraction of the "
                         "best; among those, the lowest r_stim wins")
    ap.add_argument("--baseline-tau-s", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def alpha_of(x):
        return "cv" if x == "cv" else float(x)

    d = Path(args.session_dir)
    sid = d.name
    sd = V.load_session(d)
    if not len(sd.events):
        print("no laser events; nothing to tune against")
        return 1

    rois = (None if not args.rois else
            [sd.labels.index(x.strip()) for x in args.rois.split(",")])
    alphas = tuple(alpha_of(x.strip()) for x in args.alphas.split(","))
    print(f"=== {sid} ===")
    print(f"{sd.n_roi} fibers, {len(sd.events)} events\n")

    def tick(done, total):
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}", end="\r", flush=True)

    df = V.tune_regression(
        sd,
        predictors=tuple(x.strip() for x in args.predictors.split(",")),
        common_methods=tuple(x.strip() for x in
                             args.common_methods.split(",")),
        methods=tuple(x.strip() for x in args.methods.split(",")),
        alphas=alphas,
        head_methods=tuple(x.strip() for x in args.head_methods.split(",")),
        tail_methods=tuple(x.strip() for x in args.tail_methods.split(",")),
        rois=rois, baseline_tau_s=args.baseline_tau_s, progress=tick)
    print()

    out = Path(args.out or d / f"{sid}_regression_grid.csv")
    df.to_csv(out, index=False)

    win = V.pick_winners(df, mse_tolerance=args.mse_tolerance)
    print("--- best configuration per fiber ---")
    print("(lowest residual-isosbestic correlation inside the stimulation "
          "windows,\n among configurations whose out-of-sample error is "
          f"within {args.mse_tolerance:.0%} of the best)\n")
    print(f"{'fiber':10s} {'predictors':>11s} {'common':>7s} {'method':>13s} "
          f"{'alpha':>6s} {'head':>6s} {'r_stim':>7s} {'vs worst':>9s}")
    for _, b in win.iterrows():
        print(f"{b.fiber:10s} {b.predictors:>11s} {b['common']:>7s} "
              f"{b.method:>13s} {str(b.alpha):>6s} {b.head:>6s} "
              f"{b.r_stim:7.3f} "
              f"{b.r_stim_worst / max(b.r_stim, 1e-9):8.1f}x")
    wdf = win
    print("\n--- is one configuration good enough for every fiber? ---")
    combo_cols = ["predictors", "common", "method", "alpha", "head"]
    med = (df.dropna(subset=["r_stim"]).groupby(combo_cols)["r_stim"]
           .median().sort_values())
    print("  best few by MEDIAN r_stim across fibers:")
    for k, v in med.head(5).items():
        print(f"    {str(k):60s} {v:.3f}")
    n_distinct = wdf[combo_cols].drop_duplicates().shape[0]
    print(f"\n  {n_distinct} distinct winner(s) across {len(wdf)} fibers.")
    if n_distinct > 1:
        print("  Fibers disagree. Per-fiber choice is defensible here only "
              "because the criterion\n  never looks at the response: it asks "
              "whether the isosbestic still explains\n  anything in the "
              "residual. Record the chosen configuration per fiber.")
    print(f"\n  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
