"""
Does the same stimulation reproduce across sessions?

With one animal and no sham, a correction for multiple comparisons is not the
strongest evidence available -- replication is.  A real effect of stimulating
site X should appear again when X is stimulated in another session, and
should keep its sign when the train length or rate changes.  Noise should not.

For every stimulated site present in two or more sessions, this compares the
vector of effects across recorded sites: the rank correlation between
sessions, the fraction of recorded sites keeping their sign, and a
permutation test in which the recorded-site labels of one session are
shuffled.  The permutation null is the right one here because the effect
vectors are not independent across recorded sites -- a session-wide
fluctuation correlates them -- so a parametric test on the correlation would
be anticonservative.

Reproducibility is reported BEFORE any individual pair is named, so that an
unreplicated pair is not read as a finding.
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
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("--suffix", default="",
                    help="effects-file suffix, e.g. '_full' when figures were "
                         "produced with --windows")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib as mpl
    mpl.rcParams.update({"pdf.fonttype": 42, "font.size": 7,
                         "axes.spines.top": False, "axes.spines.right": False})
    rng = np.random.default_rng(args.seed)

    root = Path(args.out_root)
    frames = []
    for csv in sorted(root.glob(f"*/*_effects{args.suffix}.csv")):
        sid = csv.parent.name
        df = pd.read_csv(csv)
        df["session"] = sid
        man_p = csv.parent / f"{sid}_manifest.json"
        proto = {}
        if man_p.exists():
            proto = (json.loads(man_p.read_text(encoding="utf-8"))
                     .get("qc", {}).get("protocol") or {})
        df["rate_hz"] = proto.get("within_burst_rate_hz", np.nan)
        df["train_s"] = proto.get("burst_duration_s", np.nan)
        frames.append(df)
    if not frames:
        print(f"no *_effects{args.suffix}.csv under {root}")
        return 1
    allf = pd.concat(frames, ignore_index=True)
    off = allf[allf.stim_site != allf.recorded_site]

    sessions_per_site = off.groupby("stim_site")["session"].nunique()
    repeated = sessions_per_site[sessions_per_site >= 2].index.tolist()
    print(f"{allf.session.nunique()} sessions, "
          f"{off.stim_site.nunique()} stimulated sites, "
          f"{len(repeated)} stimulated in 2 or more sessions\n")
    if not repeated:
        print("Nothing to replicate: no stimulated site appears in more than "
              "one session. Replication is the strongest evidence available "
              "with one animal and no sham; repeat a site to get it.")
        return 1

    rows = []
    for site in repeated:
        g = off[off.stim_site == site]
        sess = sorted(g.session.unique())
        for i in range(len(sess)):
            for j in range(i + 1, len(sess)):
                a = g[g.session == sess[i]].set_index("recorded_site")["z"]
                b = g[g.session == sess[j]].set_index("recorded_site")["z"]
                common = a.index.intersection(b.index)
                if common.size < 4:
                    continue
                av, bv = a[common].to_numpy(), b[common].to_numpy()
                rho, _ = spearmanr(av, bv)
                null = np.array([spearmanr(av, rng.permutation(bv))[0]
                                 for _ in range(args.n_perm)])
                p = (1.0 + int(np.sum(np.abs(null) >= abs(rho)))) \
                    / (args.n_perm + 1.0)
                same = int(np.sum(np.sign(av) == np.sign(bv)))
                rows.append({
                    "stim_site": site,
                    "session_a": sess[i], "session_b": sess[j],
                    "n_recorded": int(common.size),
                    "rho": float(rho), "p_perm": float(p),
                    "same_sign": same,
                    "frac_same_sign": same / common.size,
                })

    # ---- concordance across THREE OR MORE sessions ---------------------- #
    # With several repeats, pairwise correlations are neither independent nor
    # a complete summary. Kendall's W asks the right question directly: do
    # the sessions rank the recorded sites consistently? W = 1 is perfect
    # agreement, W = 0 none. The null is again by permutation, shuffling each
    # session's labels independently.
    conc = []
    for site in repeated:
        g = off[off.stim_site == site]
        piv = g.pivot_table(index="recorded_site", columns="session",
                            values="z")
        piv = piv.dropna()
        if piv.shape[1] < 3 or piv.shape[0] < 4:
            continue
        M = piv.to_numpy()

        def kendall_w(A):
            R = np.apply_along_axis(
                lambda c: pd.Series(c).rank().to_numpy(), 0, A)
            n_obj, n_raters = R.shape
            S = np.sum((R.sum(axis=1) - R.sum(axis=1).mean()) ** 2)
            return 12 * S / (n_raters ** 2 * (n_obj ** 3 - n_obj))

        w = kendall_w(M)
        null = np.array([
            kendall_w(np.column_stack([rng.permutation(M[:, k])
                                       for k in range(M.shape[1])]))
            for _ in range(min(args.n_perm, 2000))])
        pw = (1.0 + int(np.sum(null >= w))) / (null.size + 1.0)
        conc.append({"stim_site": site, "n_sessions": int(piv.shape[1]),
                     "n_recorded": int(piv.shape[0]),
                     "kendall_w": float(w), "p_perm": float(pw),
                     "sessions": ", ".join(piv.columns)})
    if conc:
        cdf = pd.DataFrame(conc)
        print("--- concordance across all sessions of a site ---")
        print(cdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        for _, r in cdf.iterrows():
            if r.p_perm < 0.05:
                print(f"  {r.stim_site}: the {r.n_sessions} sessions rank the "
                      f"recorded sites consistently (W={r.kendall_w:.2f}, "
                      f"p={r.p_perm:.3f}).")
            else:
                print(f"  {r.stim_site}: no consistent ranking across "
                      f"{r.n_sessions} sessions (W={r.kendall_w:.2f}, "
                      f"p={r.p_perm:.3f}). Whatever produced the largest "
                      "effects in one session did not produce them in the "
                      "others.")
        cdf.to_csv(root / f"concordance{args.suffix}.csv", index=False)
        print()

    rdf = pd.DataFrame(rows)
    if rdf.empty:
        print("no comparable session pairs")
        return 1
    print("--- replication across sessions ---")
    print(rdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n--- reading this ---")
    good = rdf[(rdf.p_perm < 0.05) & (rdf.rho > 0)]
    if len(good):
        print(f"  {len(good)} of {len(rdf)} session pair(s) reproduce: the "
              "pattern of effects across recorded sites is correlated between "
              "sessions beyond what shuffling the labels gives.")
        print(good[["stim_site", "session_a", "session_b", "rho",
                    "p_perm"]].to_string(index=False))
    else:
        print("  NO session pair reproduces. The pattern of effects across "
              "recorded sites is no more similar between two stimulations of "
              "the same site than between shuffled labels.")
        print("  Individual pairs with low p-values in a single session are "
              "therefore not findings: whatever produced them did not "
              "produce them again.")
    med_sign = float(rdf.frac_same_sign.median())
    print(f"\n  Median fraction of recorded sites keeping their sign across "
          f"sessions: {med_sign:.0%} (chance is 50%).")
    if med_sign < 0.65:
        print("  A real effect does not reverse when the train length or "
              "rate changes. At this level the signs are consistent with "
              "noise.")

    # per-pair detail for the best-replicating site
    best = rdf.loc[rdf.rho.idxmax()]
    g = off[off.stim_site == best.stim_site]
    a = g[g.session == best.session_a].set_index("recorded_site")
    b = g[g.session == best.session_b].set_index("recorded_site")
    common = a.index.intersection(b.index)
    det = pd.DataFrame({
        "recorded": common,
        f"z[{best.session_a}]": a.loc[common, "z"].to_numpy(),
        f"z[{best.session_b}]": b.loc[common, "z"].to_numpy(),
    })
    det["same_sign"] = np.sign(det.iloc[:, 1]) == np.sign(det.iloc[:, 2])
    print(f"\n--- detail: stimulating {best.stim_site} "
          f"(rho {best.rho:+.3f}, p {best.p_perm:.4f}) ---")
    print(det.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    n = len(rdf)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.4 * nrow),
                             squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), rdf.iterrows()):
        g = off[off.stim_site == r.stim_site]
        a = g[g.session == r.session_a].set_index("recorded_site")["z"]
        b = g[g.session == r.session_b].set_index("recorded_site")["z"]
        common = a.index.intersection(b.index)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.axvline(0, color="k", lw=0.4, ls=":")
        ax.scatter(a[common], b[common], s=14, color="C0")
        for lbl in common:
            ax.annotate(lbl, (a[lbl], b[lbl]), fontsize=4,
                        xytext=(2, 2), textcoords="offset points")
        lim = max(np.abs(np.r_[a[common], b[common]]).max(), 1) * 1.15
        ax.plot([-lim, lim], [-lim, lim], color="0.7", lw=0.6, ls="--")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"z, {r.session_a}", fontsize=5)
        ax.set_ylabel(f"z, {r.session_b}", fontsize=5)
        ax.set_title(f"stim {r.stim_site}\nrho={r.rho:+.2f} "
                     f"p={r.p_perm:.3f}, {r.frac_same_sign:.0%} same sign",
                     fontsize=6)
        ax.tick_params(labelsize=5)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Replication: the same stimulated site in two sessions. "
                 "Points on the diagonal reproduce.", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(args.out or root / f"replication{args.suffix}.pdf")
    fig.savefig(out, dpi=200); plt.close(fig)
    rdf.to_csv(root / f"replication{args.suffix}.csv", index=False)
    print(f"\n  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
