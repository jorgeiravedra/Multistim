"""
Why does the processed trace resemble the isosbestic?

Regression removes a predictor's shared variance and ADDS its independent
noise.  Where the isosbestic carries little that the signal channel shares,
that trade goes the wrong way and the output can look worse than the input.
Three distinct failures produce a processed trace that resembles the 415:

* **negative coefficient** -- ``resid = y - beta*x``, so a negative beta adds
  the predictor instead of removing it. Unconstrained OLS permits this; the
  non-negative Lasso does not.
* **over-correction** -- a coefficient larger than the shared component
  warrants leaves an inverted copy in the residual.
* **under-correction** -- shared structure survives, so both traces keep it
  and look alike for an innocent reason.

They are told apart by the SIGN of corr(residual, isosbestic): negative after
over-correction or a negative weight, positive if structure simply survived.

Reported per fiber: the coefficient, that correlation before and after, and
whether the regression made the trace noisier than leaving it alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    import numpy as np
    import pandas as pd

    from multifiber import preprocess as pp
    from multifiber import variants as V

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--predictors", default="common")
    ap.add_argument("--common-method", default="median")
    ap.add_argument("--method", default="hierarchical")
    ap.add_argument("--head-method", default="ols")
    ap.add_argument("--tail-method", default="lasso")
    ap.add_argument("--alpha", default="cv")
    ap.add_argument("--baseline-tau-s", type=float, default=60.0)
    ap.add_argument("--non-negative", action="store_true")
    args = ap.parse_args()

    alpha = "cv" if args.alpha == "cv" else float(args.alpha)
    d = Path(args.session_dir)
    sd = V.load_session(d)
    t, sig, iso = V.isosbestic_dff(sd, baseline_tau_s=args.baseline_tau_s)

    print(f"=== {d.name} ===")
    print(f"{args.predictors}/{args.common_method} · {args.method} "
          f"(head {args.head_method}, tail {args.tail_method}) · "
          f"alpha {args.alpha}"
          + (" · non-negative" if args.non_negative else "") + "\n")
    print(f"{'fiber':10s} {'coef own':>9s} {'r(raw,iso)':>11s} "
          f"{'r(resid,iso)':>13s} {'SD raw':>9s} {'SD out':>9s} "
          f"{'change':>8s}  verdict")

    rows = []
    for j, lab in enumerate(sd.labels):
        X, names = pp.build_isosbestic_predictors(
            iso, j, mode=args.predictors,
            common_method=args.common_method)
        r, info = pp.regress_isosbestic(
            sig[:, j], X, alpha=alpha, method=args.method,
            head_method=args.head_method, tail_method=args.tail_method,
            non_negative=args.non_negative, unpenalized=1)
        ok = np.isfinite(r) & np.isfinite(sig[:, j]) & np.isfinite(iso[:, j])
        r_before = float(np.corrcoef(sig[ok, j], iso[ok, j])[0, 1])
        r_after = float(np.corrcoef(r[ok], iso[ok, j])[0, 1])
        sd_raw, sd_out = float(np.std(sig[ok, j])), float(np.std(r[ok]))
        chg = sd_out / sd_raw - 1.0
        coef0 = info["coefficients"][0] if info["coefficients"] else np.nan

        if coef0 < -1e-9:
            v = "NEGATIVE weight: isosbestic is being added"
        elif r_after < -0.1 and abs(r_after) > abs(r_before) * 0.5:
            v = "over-corrected: inverted isosbestic left behind"
        elif chg > 0.05:
            v = "noisier than no regression at all"
        elif abs(r_after) > 0.5 * abs(r_before) and r_after > 0.1:
            v = "under-corrected: shared structure survives"
        else:
            v = "ok"
        rows.append({"fiber": lab, "coef_own": coef0, "r_before": r_before,
                     "r_after": r_after, "sd_raw": sd_raw, "sd_out": sd_out,
                     "sd_change": chg, "verdict": v,
                     "n_negative": info.get("n_negative_coefficients", 0)})
        print(f"{lab:10s} {coef0:9.3f} {r_before:11.3f} {r_after:13.3f} "
              f"{sd_raw:9.5f} {sd_out:9.5f} {chg:+7.1%}  {v}")

    df = pd.DataFrame(rows)
    bad = df[df["verdict"] != "ok"]
    print("\n--- reading this ---")
    if not len(bad):
        print("  Every fiber: shared structure removed, trace not made "
              "noisier.")
    else:
        print(f"  {len(bad)}/{len(df)} fiber(s) flagged.")
        if (df["coef_own"] < -1e-9).any():
            print("  A negative weight means the isosbestic is ADDED to the "
                  "residual. Use --non-negative,\n  or an estimator that "
                  "constrains the sign (the Lasso branch always does).")
        if (df["sd_change"] > 0.05).any():
            print("  Where the regression makes a fiber noisier, the "
                  "isosbestic is contributing more\n  independent noise than "
                  "shared variance. For those fibers no regression is "
                  "better\n  than this one; check whether the 415 channel is "
                  "bright enough to be a regressor.")
    out = d / f"{d.name}_regression_diagnosis.csv"
    df.to_csv(out, index=False)
    print(f"\n  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
