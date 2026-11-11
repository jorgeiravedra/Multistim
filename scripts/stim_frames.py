"""
Look at the raw video and raw ROI signal around stimulation.

Preprocessing can hide an optical artifact as easily as it can remove one, so
this bypasses it: raw camera frames and raw ROI means at the full 80 Hz frame
rate, with the laser gate drawn on top.  If the laser contaminates the image,
it is visible here and nowhere else.

Three outputs:

  <sid>_stimframes_<site>.png   a montage of raw frames spanning one
                                stimulation, with the gate state marked
  <sid>_stimraw_<site>.pdf      raw ROI means at the camera rate, both LED
                                channels, with gate pulses overlaid
  <sid>_stimframes_<site>.gif   the same frames as an animation (optional)

The raw trace panel is the quantitative one.  At 80 Hz each LED channel is
sampled alternately, so a laser artifact confined to one channel implicates
the optics; one present in both implicates something slower.
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

    from multifiber import events as ev
    from multifiber import extract as ex
    from multifiber import matio

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--site", default=None, help="stimulated region")
    ap.add_argument("--event", type=int, default=0, help="which event")
    ap.add_argument("--pre", type=float, default=0.25)
    ap.add_argument("--post", type=float, default=0.25)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib as mpl
    mpl.rcParams.update({"pdf.fonttype": 42, "font.size": 7,
                         "axes.spines.top": False, "axes.spines.right": False})

    d = Path(args.session_dir)
    sid = d.name
    man = json.loads((d / f"{sid}_manifest.json").read_text(encoding="utf-8"))
    traces = pd.read_parquet(d / f"{sid}_traces.parquet")
    events = pd.read_parquet(d / f"{sid}_events.parquet")
    if args.site and "region" in events.columns:
        events = events[events["region"] == args.site]
    if events.empty:
        print("no events")
        return 1
    e = events.iloc[min(args.event, len(events) - 1)]
    site = e.get("region", "?")
    t0, t1 = float(e["onset_t"]), float(e["offset_t"])

    # ---- raw ROI means at the camera rate -------------------------------- #
    rois = [c for c in traces.columns
            if c not in ("frame", "t", "led_state", "roi_mean")]
    t = traces["t"].to_numpy(float)
    m = (t >= t0 - args.pre) & (t <= t1 + args.post)
    sub = traces.loc[m]
    if sub.empty:
        print("event outside the trace")
        return 1

    bcam = man.get("inputs", {}).get("paths", {}).get("bcam")
    gate_on = gate_off = None
    if bcam and Path(bcam).exists():
        log = matio.load_bcam(bcam)
        g = log.channels.get("laser_gate")
        if g is not None and float(np.nanmax(g) - np.nanmin(g)) >= 0.5:
            edges = ev.detect_edges(log.time, g, 0.5)
            gate_on, gate_off = edges.rising_t, edges.falling_t

    n_show = min(6, len(rois))
    fig, axes = plt.subplots(n_show, 1, figsize=(8, 1.15 * n_show),
                             sharex=True)
    for ax, r in zip(np.atleast_1d(axes), rois[:n_show]):
        for state, colour, lbl in ((sub["led_state"].mode().iloc[0], "C2",
                                    "signal"),
                                   (1 - sub["led_state"].mode().iloc[0], "C4",
                                    "isosbestic")):
            k = sub["led_state"] == state
            ax.plot(sub.loc[k, "t"], sub.loc[k, r], ".-", ms=2.5, lw=0.6,
                    color=colour, label=lbl if r == rois[0] else None)
        if gate_on is not None:
            for a, b in zip(gate_on, gate_off):
                if t0 - args.pre <= a <= t1 + args.post:
                    ax.axvspan(a, b, color="C3", alpha=0.25, lw=0)
        ax.axvline(t0, color="k", ls="--", lw=0.7)
        ax.axvline(t1, color="k", ls=":", lw=0.7)
        ax.set_ylabel(r, rotation=0, ha="right", va="center", fontsize=6)
        ax.tick_params(labelsize=5)
    np.atleast_1d(axes)[0].legend(fontsize=5, frameon=False, ncol=2)
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    np.atleast_1d(axes)[0].set_title(
        f"{sid} — raw ROI means at the camera rate, stimulating {site} "
        f"(red: laser gate on; dashed: command on/off)", fontsize=7)
    fig.tight_layout()
    out = Path(args.out or d / f"{sid}_stimraw_{site}.pdf")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  {out}")

    # quantify: does the signal step when the gate fires?
    if gate_on is not None:
        print("\n--- raw signal during gate-on vs gate-off frames ---")
        print("A step here is an optical artifact; preprocessing would hide "
              "it, so this is the place to look.\n")
        print(f"{'fiber':10s} {'gate-on':>10s} {'gate-off':>10s} "
              f"{'diff':>9s} {'diff/SD':>8s}")
        period = float(np.median(np.diff(t)))
        on_mask = np.zeros(len(sub), dtype=bool)
        tt = sub["t"].to_numpy(float)
        for a, b in zip(gate_on, gate_off):
            on_mask |= (tt >= a - period) & (tt <= b + period)
        sig_state = sub["led_state"].mode().iloc[0]
        s_only = (sub["led_state"] == sig_state).to_numpy()
        for r in rois:
            v = sub[r].to_numpy(float)
            a = v[on_mask & s_only]
            b = v[(~on_mask) & s_only]
            if a.size < 3 or b.size < 3:
                continue
            diff = float(a.mean() - b.mean())
            sdv = float(np.std(np.r_[a, b]))
            print(f"{r:10s} {a.mean():10.3f} {b.mean():10.3f} "
                  f"{diff:+9.3f} {diff / sdv if sdv else float('nan'):8.2f}")

    # ---- raw frames ------------------------------------------------------ #
    video = man.get("inputs", {}).get("paths", {}).get("video")
    if not video or not Path(video).exists():
        print(f"\nvideo not found at {video!r}; skipping the frame montage")
        return 0

    i0 = int(sub["frame"].iloc[0])
    i1 = int(sub["frame"].iloc[-1])
    n = min(args.max_frames, i1 - i0 + 1)
    step = max(1, (i1 - i0 + 1) // n)
    want = set(range(i0, i1 + 1, step))
    frames, kept = [], []
    for k, fr in enumerate(ex.iter_video_frames(video)):
        if k in want:
            frames.append(fr)
            kept.append(k)
        if k > i1:
            break
    if not frames:
        print("no frames read")
        return 0

    half = frames[0].shape[1] // 2
    gate_state = []
    for k in kept:
        row = traces[traces["frame"] == k]
        tk = float(row["t"].iloc[0]) if len(row) else np.nan
        on = bool(gate_on is not None and np.any(
            (gate_on <= tk) & (gate_off >= tk))) if np.isfinite(tk) else False
        gate_state.append(on)

    ncol = 10
    nrow = int(np.ceil(len(frames) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.0 * ncol, 1.0 * nrow))
    vmin = float(np.percentile(frames[0][:, :half], 1))
    vmax = float(np.percentile(frames[0][:, :half], 99.9))
    for ax, fr, k, on in zip(np.atleast_1d(axes).ravel(), frames, kept,
                             gate_state):
        ax.imshow(fr[:, :half], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{k}{' *' if on else ''}", fontsize=4,
                     color=("C3" if on else "k"))
        for sp in ax.spines.values():
            sp.set_color("C3" if on else "0.8")
            sp.set_linewidth(1.2 if on else 0.4)
    for ax in np.atleast_1d(axes).ravel()[len(frames):]:
        ax.axis("off")
    fig.suptitle(f"{sid} — raw frames around one {site} stimulation "
                 "(* and red border: laser gate on; same grey scale "
                 "throughout)", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pout = d / f"{sid}_stimframes_{site}.png"
    fig.savefig(pout, dpi=220)
    plt.close(fig)
    print(f"  {pout}")

    if args.gif:
        try:
            import imageio.v2 as imageio

            arr = [np.clip((f[:, :half] - vmin) / max(vmax - vmin, 1) * 255,
                           0, 255).astype(np.uint8) for f in frames]
            gout = d / f"{sid}_stimframes_{site}.gif"
            imageio.mimsave(gout, arr, duration=0.08)
            print(f"  {gout}")
        except ImportError:
            print("  (install imageio for --gif)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
