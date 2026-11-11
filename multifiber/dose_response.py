"""
Cross-session dose-response: effect versus stimulation parameter.

This is the strongest positive control the frequency/duration design affords,
and it is stronger than any single session's p-value.  A response driven by
opsin activation must grow with delivered light; drift, spontaneous
transients and non-specific responses to the laser have no reason to.  A
monotone relation across three or four parameter levels is evidence that
survives small n, because the ordering is predicted in advance rather than
selected after the fact.

The parameter is read from the session name: ``10Hzburst`` gives a burst
frequency, ``500ms`` a pulse duration.  Sessions sharing a stimulated site
are compared.

    python scripts/dose_response.py "F:\\Jorge\\Multistim_Analysis" ^
        --parameter frequency

Caveats worth keeping in view.  Sessions differ in more than the parameter --
time since surgery, bleaching, the animal's state, coupling after reseating
the patchcord -- so an ordering across sessions is weaker than one within a
session.  Interleaving parameter levels within a single session removes that
confound entirely and is worth the protocol change.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERNS = {
    "frequency": (re.compile(r"(\d+(?:\.\d+)?)\s*Hz", re.I), "Hz"),
    "duration": (re.compile(r"(\d+(?:\.\d+)?)\s*ms", re.I), "ms"),
}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("path",
                    help="a single SESSION directory (compare the conditions "
                         "interleaved within that one recording) or a root "
                         "containing several (compare across sessions). The "
                         "within-session comparison is the stronger one: "
                         "across sessions the level is confounded with time, "
                         "cumulative bleaching and patchcord reseating.")
    ap.add_argument("--within-session", action="store_true",
                    help="never combine levels from different sessions, even "
                         "when given a root; each session is analysed on its "
                         "own.")
    ap.add_argument("--parameter", choices=tuple(PATTERNS), default="frequency")
    ap.add_argument("--pre", type=float, default=2.0)
    ap.add_argument("--post", type=float, default=4.0)
    ap.add_argument("--baseline-window", type=float, default=0.5)
    ap.add_argument("--response-window", type=float, default=0.5)
    ap.add_argument("--self-only", action="store_true",
                    help="only the stimulated site's own fiber (the positive "
                         "control); otherwise every recorded site")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib as mpl
    mpl.rcParams.update({"pdf.fonttype": 42, "font.size": 7,
                         "axes.spines.top": False, "axes.spines.right": False})

    pat, unit = PATTERNS[args.parameter]
    root = Path(args.path)
    # A session directory contains its own manifest; a root contains several.
    own = list(root.glob(f"{root.name}_manifest.json"))
    if own:
        manifests = own
        root = root.parent
        print(f"single session: {own[0].parent.name}\n")
    else:
        manifests = sorted(root.glob("*/*_manifest.json"))
    rows = []

    # A session may interleave conditions, in which case the level is a
    # property of the EVENTS, not of the session name. That is the stronger
    # design -- one animal, one session, no between-session confound -- so it
    # is handled first and the name is used only as a fallback.
    for man_p in manifests:
        d = man_p.parent
        sid = d.name
        ev_p = d / f"{sid}_events.parquet"
        interleaved = False
        if ev_p.exists():
            try:
                _e = pd.read_parquet(ev_p)
                interleaved = ("protocol_class" in _e.columns
                               and _e["protocol_class"].nunique() > 1)
            except Exception:
                interleaved = False
        m = pat.search(sid)
        if not interleaved and not m:
            continue
        level = float(m.group(1)) if m else float("nan")
        try:
            dff = pd.read_parquet(d / f"{sid}_dff.parquet")
            events = pd.read_parquet(d / f"{sid}_events.parquet")
        except Exception:
            continue
        if events.empty:
            continue
        man = json.loads(man_p.read_text(encoding="utf-8"))
        rois = [c for c in dff.columns
                if c not in ("t", "frame", "blanked")
                and not c.endswith("__iso")]
        t = dff["t"].to_numpy()
        fs = 1.0 / np.median(np.diff(t))
        npre, npost = int(args.pre * fs), int(args.post * fs)
        nbase = int(args.baseline_window * fs)
        nresp = int(args.response_window * fs)

        groupers = ["region"]
        if interleaved and "protocol_class" in events.columns:
            groupers = ["protocol_class", "region"]
        for key, g in events.groupby(groupers):
            if interleaved:
                pcls, site = key
                # the level IS the measured parameter of this class
                lvl = (float(g["width_s"].median()) * 1e3
                       if args.parameter == "duration"
                       else float(g["command_v"].median()))
            else:
                site = key if not isinstance(key, tuple) else key[-1]
                pcls, lvl = None, level
            if site is None:
                continue
            ons = g["onset_t"].to_numpy(float)
            idx = np.searchsorted(t, ons)
            idx = idx[(idx - npre >= 0) & (idx + npost < t.size)]
            if idx.size == 0:
                continue
            targets = [site] if args.self_only else rois
            for rec in targets:
                if rec not in dff.columns:
                    continue
                y = dff[rec].to_numpy(float)
                W = np.stack([y[i - npre:i + npost] for i in idx])
                W = W - np.nanmean(W[:, npre - nbase:npre], axis=1,
                                   keepdims=True)
                eff = np.nanmean(W[:, npre:npre + nresp], axis=1)
                rows.append({
                    "session": sid, "level": lvl, "stim_site": site,
                    "protocol_class": pcls,
                    "recorded_site": rec, "n": int(idx.size),
                    "effect": float(np.nanmean(eff)),
                    "sem": float(np.nanstd(eff, ddof=1) / np.sqrt(idx.size))
                    if idx.size > 1 else np.nan,
                    "power_mw": man.get("qc", {}).get("laser_params", {})
                    .get("command_v"),
                })

    if not rows:
        print(f"nothing to compare under {args.path}: no session has a "
              f"{args.parameter} in its name and none interleaves conditions")
        return 1
    df = pd.DataFrame(rows)
    if args.within_session or len(manifests) == 1:
        # Compare only levels delivered in the same recording.
        keys = ["session", "stim_site", "recorded_site"]
        df["_grp"] = df["session"]
    else:
        keys = ["stim_site", "recorded_site"]
        df["_grp"] = ""
    pairs = (df.groupby(keys)["level"].nunique().sort_values(ascending=False))
    usable = pairs[pairs >= 2]
    if args.within_session or len(manifests) == 1:
        print("comparing only levels delivered within the same recording")
    within = int(df.groupby("session")["level"].nunique().gt(1).sum())
    print(f"{df['session'].nunique()} sessions, levels: "
          f"{sorted(round(x, 3) for x in df['level'].unique())} {unit}")
    if within:
        print(f"  {within} session(s) vary the level WITHIN the recording. "
              "Those comparisons are free of the between-session confounds "
              "(time, bleaching, patchcord reseating) that weaken a "
              "cross-session dose-response.")
    print(f"{len(usable)} site pairs appear at 2 or more levels\n")
    if usable.empty:
        print("Nothing to compare: no stimulated site appears at more than "
              "one level. A dose-response needs the same site at several "
              "parameter values.")
        return 1

    # rank correlation of effect with level, per pair
    res = []
    for key in usable.index:
        if len(keys) == 3:
            sess, ss, rs = key
            g = df[(df.session == sess) & (df.stim_site == ss)
                   & (df.recorded_site == rs)]
        else:
            ss, rs = key
            sess = "(across sessions)"
            g = df[(df.stim_site == ss) & (df.recorded_site == rs)]
        g = g.groupby("level")["effect"].mean().sort_index()
        if g.size < 2:
            continue
        from scipy.stats import spearmanr
        rho, p = spearmanr(g.index.to_numpy(), g.to_numpy())
        res.append({"session": sess, "stim_site": ss,
                    "recorded_site": rs, "n_levels": g.size,
                    "rho": float(rho), "p": float(p),
                    "effect_lo": float(g.iloc[0]), "effect_hi": float(g.iloc[-1])})
    rdf = pd.DataFrame(res).sort_values("rho", key=abs, ascending=False)
    print(rdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nA monotone relation across levels is predicted in advance for a "
          "light-driven response and\nfor nothing else, so it carries more "
          "weight than a p-value at one level.")
    print("With few levels, Spearman p cannot reach significance "
          "(3 levels: min p = 0.333); read rho and the\nspread, not p.")

    sites = sorted(set(rdf["stim_site"]))
    ncol = min(4, max(1, len(sites)))
    nrow = int(np.ceil(len(sites) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.1 * nrow),
                             squeeze=False)
    for ax, ss in zip(axes.ravel(), sites):
        g = df[df.stim_site == ss]
        if len(keys) == 3:
            g = g[g.session == rdf.loc[rdf.stim_site == ss, "session"].iloc[0]]
        for rs, gg in g.groupby("recorded_site"):
            agg = gg.groupby("level").agg(effect=("effect", "mean"),
                                          sem=("sem", "mean")).sort_index()
            if agg.shape[0] < 2:
                continue
            is_self = (rs == ss)
            ax.errorbar(agg.index, agg["effect"], yerr=agg["sem"],
                        marker="o", ms=3, lw=1.2 if is_self else 0.5,
                        color="C3" if is_self else "0.7",
                        label=rs if is_self else None, zorder=3 if is_self else 1)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_title(f"stim {ss}", fontsize=7)
        ax.set_xlabel(f"{args.parameter} ({unit})")
        ax.set_ylabel("effect (dF/F)")
        ax.tick_params(labelsize=5)
    for ax in axes.ravel()[len(sites):]:
        ax.axis("off")
    fig.suptitle(f"Dose-response across sessions — red: the stimulated "
                 f"site's own fiber", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    stem = (manifests[0].parent.name if len(manifests) == 1
            else f"dose_response_{args.parameter}")
    out = Path(args.out or (manifests[0].parent if len(manifests) == 1 else root)
               / f"{stem}_dose_{args.parameter}.pdf")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    rdf.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\n  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
