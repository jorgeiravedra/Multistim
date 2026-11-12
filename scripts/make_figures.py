"""
Exploratory figures for a processed stim-record session.

Produces, per session:

  <sid>_overview.pdf      all ROI dF/F traces with stimulation times marked
  <sid>_peristim_<site>.pdf   per-recorded-site peri-stimulus mean, with a
                              null band from time-shuffled pseudo-events
  <sid>_matrix.pdf        stimulated site x recorded site effect, in units of
                          the null distribution's SD

The null matters more than the signal here.  A pre-versus-during contrast on
photometry is confounded by photobleaching, by slow state drift, and by the
fact that the stimulated site is the one you were looking at.  So the reported
effect is "how far outside what this recording does on its own", not "how far
from zero".

The null is built by **circularly shifting the whole event train** by a random
lag and wrapping around, then running the identical analysis.  Two properties
make this the right null rather than scattering pseudo-events at random:

* it preserves the autocorrelation of the signal, which at the seconds
  timescale is strong -- independent draws understate how much a structured
  set of events can differ from baseline by chance;
* it preserves the *spacing* of the real events.  That matters most for
  closed-loop sessions, where stimulation times are behaviourally triggered
  and therefore clustered, so a uniform null is not comparable to the data.

Each shift yields a trial **mean** over the same number of events as the real
data, so the null distribution is of the same quantity being measured.  An
earlier version compared a trial mean against a distribution of single trials,
which deflated every z by roughly sqrt(n_events).

Choice of estimator
-------------------
Simulated with the real noise structure (correlated, as photometry is), with a
response built from the command envelope and a double-exponential indicator
kernel, scoring each estimator against the same shift null:

    1 s command, kinetics as assumed   boxcar 0-1.5 s  z=13.3   kernel z=12.4
    1 s command, decay 1.0 s not 0.4   boxcar 0.2-2.5  z=14.9   kernel z=11.1
    0.1 s command                      kernel  z=9.6   boxcar 0-1.5 z=7.5

Two things follow, neither of them the intuitive one.  A matched filter is
optimal for *white* noise; photometry noise is strongly correlated, so the
match buys little and a kernel built on the wrong decay constant actively
loses power -- it weights samples where the real signal has already decayed.
For long commands the response occupies most of any sensible window anyway, so
the boxcar is as good and has no parameters to get wrong.  For brief commands
the response is a small part of the window and matching pays.

Hence: boxcar by default, kernel for short bursts, and in either case the
window should be set from the data (where the trial mean actually rises and
decays) rather than inherited.  Do not try several and keep the best -- that
is a free parameter chosen on the outcome.

This is a look at the data.  It is not a test: there is no sham condition, one
animal, and no correction for the number of site pairs examined.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path


#: Indicator timescales (GCaMP6f, bulk population), colour-coded on every
#: peri-event panel.  Colourblind-safe (Okabe-Ito / seaborn "colorblind").
TIMESCALES = {
    "sampling":  {"span": (0.000, 0.025), "color": "#999999",
                  "label": "25 ms: one sample"},
    "rise":      {"span": (0.000, 0.100), "color": "#0173B2",
                  "label": "0-100 ms: rise to peak"},
    "transient": {"span": (0.100, 0.500), "color": "#029E73",
                  "label": "100-500 ms: decay"},
    "slow":      {"span": (1.000, 4.000), "color": "#CC78BC",
                  "label": ">1 s: not indicator kinetics"},
}


def _shade_timescales(ax, show_slow=True, labels=False):
    """Mark the indicator's timescales on a peri-event axis."""
    for key, d in TIMESCALES.items():
        if key == "slow" and not show_slow:
            continue
        if key == "sampling":
            continue
        a, b = d["span"]
        ax.axvspan(a, b, color=d["color"], alpha=0.13, lw=0,
                   label=d["label"] if labels else None)
    a, b = TIMESCALES["sampling"]["span"]
    ax.axvspan(a, b, facecolor="none", edgecolor=TIMESCALES["sampling"]["color"],
               hatch="////", lw=0.0, alpha=0.9,
               label=TIMESCALES["sampling"]["label"] if labels else None)


def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,          # editable text in Illustrator
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def gcamp_kernel(tt, width_s, rise_tau, decay_tau):
    """Predicted response: command envelope convolved with a double exponential.

    The envelope is 1 while the laser is commanded and 0 otherwise; the
    indicator kernel is ``(1 - exp(-s/rise)) * exp(-s/decay)``.  Projecting
    onto this is a matched filter: it weights the samples where signal is
    expected and downweights the rest, which a boxcar cannot do.  It is
    returned zero-mean and unit-norm so the projection has the units of the
    trace and is comparable across windows.
    """
    import numpy as np

    dt = float(np.median(np.diff(tt)))
    env = ((tt >= 0) & (tt < width_s)).astype(float)
    s = np.arange(0, max(8 * decay_tau, width_s + 4 * decay_tau), dt)
    h = (1.0 - np.exp(-s / rise_tau)) * np.exp(-s / decay_tau)
    k = np.convolve(env, h)[:tt.size]
    k = k - k.mean()
    n = np.linalg.norm(k)
    return k / n if n > 0 else k


def df_diag_report(rows):
    """Summarise the stimulated site's own response.

    This is the positive control, and the first thing to read.  It tests the
    whole chain -- opsin expression, light delivery, indicator expression,
    readout -- in one number per site.  A null diagonal means an off-diagonal
    null says nothing about connectivity, because nothing has been shown to
    have been perturbed.

    **The expected sign depends on the preparation.**  With an excitatory
    opsin in Vgat+ cells and the indicator in the complementary Cre-negative
    population, driving the inhibitory cells should DECREASE the recorded
    signal at that site.  A negative diagonal is then the predicted result,
    not a failure -- but it is also what bleaching or quenching of the
    indicator by the stimulation light produces, so the two must be separated
    on the isosbestic channel before either is believed.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return ""
    d = df[df["stim_site"] == df["recorded_site"]]
    if d.empty:
        return "\nNo stimulated site was also a recorded site."
    d = d.copy()
    d["floor_z3"] = 3 * d["null_sd"]
    d["above_floor"] = d["effect"].abs() > d["floor_z3"]
    lines = ["\n=== positive control: response at the stimulated site ===",
             "Each site is judged against ITS OWN channel's floor; null SD "
             "varies by orders\nof magnitude between fibers, so a "
             "session-wide floor is not a valid threshold."]
    lines.append(d[["stim_site", "effect", "floor_z3", "above_floor",
                    "z", "p_shift", "n_events"]]
                 .to_string(index=False,
                            float_format=lambda v: f"{v:.5f}"))
    if (d["z"].abs() < 2).all():
        lines.append(
            "\n  No stimulated site shows a response at its own fiber. Until "
            "that works, an absence of off-diagonal effects is uninformative: "
            "it is consistent with no connectivity AND with no perturbation. "
            "Check opsin expression, light delivery and fiber placement "
            "before interpreting the matrix."
        )
    else:
        hit = d[d["z"].abs() >= 2]
        lines.append(
            f"\n  {len(hit)} of {len(d)} sites respond at their own fiber "
            f"({', '.join(hit['stim_site'])}). Sign: "
            f"{int((hit['effect'] < 0).sum())} negative, "
            f"{int((hit['effect'] > 0).sum())} positive. With an excitatory "
            "opsin in Vgat+ cells and the indicator in the Cre-negative "
            "population, negative is the predicted direction -- and is also "
            "what stimulation-light bleaching produces. See the bleaching "
            "control below before interpreting."
        )
    return "\n".join(lines)


def _linear_baseline(W, fit_cols, col_x):
    """Per-row linear fit through ``fit_cols``, evaluated at every column."""
    import numpy as np

    Y = W[:, fit_cols]
    x = col_x[fit_cols]
    good = np.isfinite(Y)
    n = good.sum(axis=1)
    xs = np.where(good, x, 0.0)
    ys = np.where(good, Y, 0.0)
    sx = xs.sum(axis=1)
    sy = ys.sum(axis=1)
    sxx = (xs * xs).sum(axis=1)
    sxy = (xs * ys).sum(axis=1)
    denom = n * sxx - sx * sx
    slope = np.where(denom > 0, (n * sxy - sx * sy) / np.where(denom > 0, denom, 1), 0.0)
    inter = np.where(n > 0, (sy - slope * sx) / np.where(n > 0, n, 1), 0.0)
    return slope[:, None] * col_x[None, :] + inter[:, None]


def _epochs(t, events, pre, post, fs):
    """(n_events, n_samples) windows around each onset, dropping incomplete."""
    import numpy as np

    npre, npost = int(pre * fs), int(post * fs)
    idx = np.searchsorted(t, events)
    keep = (idx - npre >= 0) & (idx + npost < t.size)
    return idx[keep], npre, npost


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--pre", type=float, default=2.0)
    ap.add_argument("--post", type=float, default=4.0)
    ap.add_argument("--estimator", choices=("kernel", "boxcar"),
                    default="boxcar",
                    help="boxcar: mean over a fixed window (default). "
                         "kernel: project the trial mean onto the predicted "
                         "GCaMP response. Simulation under correlated noise "
                         "(see docstring) gives the kernel NO advantage for "
                         "long commands and a clear one only for short ones, "
                         "so the boxcar is the default and the kernel is "
                         "worth using for brief bursts.")
    ap.add_argument("--rise-tau", type=float, default=0.07,
                    help="GCaMP rise time constant, s (6f bulk default)")
    ap.add_argument("--decay-tau", type=float, default=0.40,
                    help="GCaMP decay time constant, s (6f bulk default)")
    ap.add_argument("--response-window", type=float, default=0.5,
                    help="boxcar length after onset, seconds. Default 0.5 s "
                         "covers the full GCaMP6f transient including decay. "
                         "A window extending to 1-1.5 s mixes indicator "
                         "kinetics with network reverberation, behavioural "
                         "consequences of the stimulus and drift; those are "
                         "reported separately as the slow window.")
    ap.add_argument("--slow-window", type=float, nargs=2,
                    default=(1.0, 4.0), metavar=("START", "STOP"),
                    help="a second window, past the indicator's kinetics, "
                         "scored and reported separately. Set to 0 0 to skip.")
    ap.add_argument("--response-start", type=float, default=0.0,
                    help="boxcar start relative to onset, seconds")
    ap.add_argument("--baseline-window", type=float, default=0.5,
                    help="seconds before onset used to re-baseline each "
                         "trial; 0.5 s matches the transient's own scale")
    ap.add_argument("--detrend", choices=("mean", "linear"), default="mean",
                    help="per-trial baseline correction. 'mean' subtracts the "
                         "pre-window mean, which removes an offset but leaves "
                         "any slope. 'linear' fits a line through the "
                         "pre-window AND a late post-window, both outside the "
                         "expected response, and subtracts it. Simulated "
                         "against drift-dominated noise matching this rig, "
                         "'linear' scores 0.79x the z of 'mean': the drift is "
                         "slow relative to a 6 s window, so it is nearly an "
                         "offset already, and fitting a slope from noisy "
                         "flanks adds more variance than it removes. 'mean' "
                         "is therefore the default.")
    ap.add_argument("--tail-start", type=float, default=None,
                    help="seconds after onset where the late window used for "
                         "linear detrending begins; defaults to the command "
                         "duration plus 3 decay constants")
    ap.add_argument("--lowpass", type=float, default=None,
                    help="Hz; band-limit to the indicator's bandwidth. GCaMP6f "
                         "cannot follow above ~5 Hz, so anything faster is "
                         "noise by construction. Applied identically to the "
                         "null.")
    ap.add_argument("--n-null", type=int, default=1000,
                    help="circular shifts used to build the null distribution")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-responsive", action="store_true",
                    help="restrict the matrix and the FDR correction to stim "
                         "sites whose OWN fiber responded (|z| >= "
                         "--diag-z). Sites where the perturbation was not "
                         "demonstrated produce uninterpretable rows, and "
                         "including them inflates the correction. The "
                         "selection is made on the diagonal only, so it does "
                         "not condition on the off-diagonal outcomes.")
    ap.add_argument("--diag-z", type=float, default=2.0)
    ap.add_argument("--statistic", choices=("mean", "median"), default="mean",
                    help="how trials are combined. At small n the mean is "
                         "carried by single spontaneous transients that "
                         "happen to land near a stimulus; the median is "
                         "robust to them. Applied identically to the "
                         "observed data and to every circular shift, so the "
                         "null remains valid either way.")
    ap.add_argument("--split-by-protocol", choices=("auto", "always", "never"),
                    default="auto",
                    help="when a session interleaves conditions (the design "
                         "that removes between-session confounds from a "
                         "dose-response), analyse each separately and label "
                         "the outputs by duration and power. 'auto' splits "
                         "whenever more than one protocol class is present.")
    ap.add_argument("--protocol-class", type=int, default=None,
                    help="analyse only this class (set by the dispatcher)")
    ap.add_argument("--protocol", default=None,
                    help="manual protocol, overriding the manifest and the "
                         "session name. Format 'key=value,...' with keys "
                         "rate_hz, train_s, pulse_ms, isi_s. Needed when the "
                         "name does not reveal the protocol -- e.g. "
                         "'500ms_testvmh' gives the train length but not the "
                         "80 Hz rate. Sets the response and slow windows and "
                         "the frequency used by the spectral checks.")
    ap.add_argument("--show-trials", choices=("auto", "always", "never"),
                    default="auto",
                    help="overlay individual trials. 'auto' draws them when "
                         "n <= --max-trials-shown. With few trials the mean "
                         "hides how much of it rests on one trial.")
    ap.add_argument("--max-trials-shown", type=int, default=25)
    ap.add_argument("--target-effect", type=float, default=0.003,
                    help="effect size, as dF/F, for the power table: how "
                         "many trials each channel would need to resolve it. "
                         "Default 0.3%%.")
    ap.add_argument("--target-z", type=float, default=3.0,
                    help="detection threshold for the power table")
    ap.add_argument("--windows", default=None,
                    help="run several response windows in one go, writing a "
                         "labelled set of outputs per window. Format "
                         "'name:start:stop' in seconds, comma-separated, e.g. "
                         "'onset:0:0.5,full:0:1.5'. Overrides "
                         "--response-start / --response-window. Both are "
                         "reported; choosing between them after seeing the "
                         "result is a free parameter selected on the outcome.")
    args = ap.parse_args()

    _style()

    # Interleaved conditions: one run per class, each with its own labelled
    # outputs. A single analysis across classes averages conditions that were
    # deliberately separated at acquisition.
    if args.protocol_class is None and args.split_by_protocol != "never":
        import pandas as _pd

        _d = Path(args.session_dir)
        _ev = _d / f"{_d.name}_events.parquet"
        if _ev.exists():
            _e = _pd.read_parquet(_ev)
            if "protocol_class" in _e.columns:
                _cls = sorted(int(c) for c in _e["protocol_class"].unique())
                if len(_cls) > 1 or args.split_by_protocol == "always":
                    _hint = ""
                    try:
                        from multifiber.events import parse_session_name
                        h = parse_session_name(_d.name)
                        if "n_protocols" in h:
                            _hint = (f"  (the name says {h['n_protocols']}; "
                                     f"the data give {len(_cls)}"
                                     + ("" if h["n_protocols"] == len(_cls)
                                        else " -- MISMATCH") + ")")
                    except Exception:
                        pass
                    print(f"\n{len(_cls)} interleaved protocol classes"
                          f"{_hint}. Running each separately.\n")
                    rc = 0
                    for c in _cls:
                        sub = argparse.Namespace(**vars(args))
                        sub.protocol_class = c
                        g = _e[_e["protocol_class"] == c]
                        # Name a condition by what differs: a session can
                        # interleave rates inside one envelope, and three
                        # files all called "1000ms" would overwrite nothing
                        # but confuse everything.
                        from multifiber.events import format_duration
                        _d = float(g["width_s"].median())
                        lab = format_duration(_d).replace(" ", "")
                        if ("n_gate_pulses" in g.columns
                                and g["n_gate_pulses"].notna().any()
                                and _d > 0):
                            _r = float(g["n_gate_pulses"].median()) / _d
                            if _r > 1:
                                lab += f"_{_r:.0f}Hz"
                        if "command_v" in g.columns:
                            lab += f"_{g['command_v'].median():.3f}V"
                        sub.class_label = lab
                        print(f"{'=' * 70}\n### protocol class {c}: {lab}, "
                              f"{len(g)} events\n{'=' * 70}")
                        rc |= _run(sub)
                    print("\nEach class is reported separately. The "
                          "dose-response across them is the comparison the "
                          "interleaved design was built for; pooling them "
                          "is not.")
                    return rc

    # Several response windows in one run, each writing its own labelled
    # outputs. Aligning to the FIRST pulse of a burst means a short window
    # measures the rise of the train, not its peak; a window spanning the
    # whole burst plus the indicator's decay measures the accumulated
    # response. They answer different questions and both are reported.
    if args.windows:
        specs = []
        for part in args.windows.split(","):
            bits = part.split(":")
            if len(bits) != 3:
                raise SystemExit(f"bad --windows entry {part!r}; "
                                 "expected name:start:stop")
            specs.append((bits[0], float(bits[1]), float(bits[2])))
        rc = 0
        for name, w0, w1 in specs:
            sub = argparse.Namespace(**vars(args))
            sub.windows = None
            sub.response_start = w0
            sub.response_window = w1 - w0
            sub.window_label = name
            print(f"\n{'=' * 70}\n### response window '{name}': "
                  f"{w0:+.2f} to {w1:+.2f} s from burst onset\n{'=' * 70}")
            rc |= _run(sub)
        print("\nBoth windows are reported. Picking the one with the larger "
              "effect after the fact\nis a free parameter chosen on the "
              "outcome; state in advance which is primary.")
        return rc
    return _run(args)


def _run(args) -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(args.seed)

    d = Path(args.session_dir)
    sid = d.name
    out = Path(args.outdir or d)
    tag = getattr(args, "window_label", None)
    clab = getattr(args, "class_label", None)
    suf = ("_" + clab if clab else "") + (f"_{tag}" if tag else "")
    out.mkdir(parents=True, exist_ok=True)

    man_p = d / f"{sid}_manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    qc = man.get("qc", {})

    # ---- protocol, measured from the laser channel ----------------------- #
    # Read from the manifest, never parsed out of the session name: a renamed
    # or changed session would otherwise set every window and frequency from
    # a string.
    proto = dict(qc.get("protocol") or {})
    name_chk = qc.get("protocol_name_check") or {}

    if args.protocol:
        manual = {}
        for part in args.protocol.split(","):
            if "=" not in part:
                raise SystemExit(f"bad --protocol entry {part!r}; "
                                 "expected key=value")
            k, v = part.split("=", 1)
            manual[k.strip()] = float(v)
        if "rate_hz" in manual:
            proto["within_burst_rate_hz"] = manual["rate_hz"]
        if "train_s" in manual:
            proto["burst_duration_s"] = manual["train_s"]
        if "pulse_ms" in manual:
            proto["pulse_width_s"] = {"median": manual["pulse_ms"] / 1e3}
        if "isi_s" in manual:
            proto["same_site_interval_s"] = manual["isi_s"]
        if ("rate_hz" in manual and "pulse_ms" in manual):
            proto["duty_cycle"] = manual["rate_hz"] * manual["pulse_ms"] / 1e3
        measured = qc.get("protocol") or {}
        key_map = {"rate_hz": "within_burst_rate_hz",
                   "train_s": "burst_duration_s",
                   "isi_s": "same_site_interval_s"}
        clashes = []
        for k, mk in key_map.items():
            if k in manual and mk in measured:
                mv = float(measured[mk])
                if abs(manual[k] - mv) > max(0.1 * mv, 1e-6):
                    clashes.append((k, manual[k], mv))
        proto["source"] = "manual (--protocol)"
        print(f"=== protocol overridden manually: {args.protocol} ===")
        for k, given, mv in clashes:
            print(f"  !! {k}: you gave {given:g}, the laser channel measures "
                  f"{mv:g}. The manual value is used for the windows, but "
                  "the measured one is what was delivered -- check which is "
                  "wrong before trusting the windows.")
    try:
        from multifiber.events import parse_session_name
        hints = parse_session_name(sid)
    except Exception:
        hints = {}
    if hints:
        print("=== session name says ===")
        print(f"  {hints}")
        print("  These are HINTS. Every one is also measured from the laser "
              "channel below, and the measurement wins.")
    if proto:
        print("\n=== protocol (measured, not from the filename) ===")
        rate = proto.get("within_burst_rate_hz")
        bd = proto.get("burst_duration_s")
        pw = (proto.get("pulse_width_s") or {}).get("median")
        print(f"  {proto.get('n_events')} events"
              + (f", {proto['pulses_per_burst']['median']:.0f} pulses each"
                 if "pulses_per_burst" in proto else "")
              + (f" at {rate:.1f} Hz" if rate else "")
              + (f", burst {bd:.3g} s" if bd else "")
              + (f", pulse {pw*1e3:.0f} ms" if pw else ""))
        if "duty_cycle" in proto:
            print(f"  duty cycle within burst: {proto['duty_cycle']:.1%}")
        if "same_site_interval_s" in proto:
            print(f"  same-site interval: {proto['same_site_interval_s']:.1f} s"
                  f"  ({proto.get('n_sites', '?')} sites, "
                  f"{(proto.get('events_per_site') or {}).get('median', '?')}"
                  " events each)")
        if "estimated_mw" in proto:
            e = proto["estimated_mw"]
            print(f"  commanded power: {e['median']:.2f} mW "
                  f"({e['min']:.2f}-{e['max']:.2f})")
        for c in name_chk.get("checks", []):
            if not c["ok"]:
                print(f"  !! the session NAME says {c['field']} = "
                      f"{c['named']:g} but the laser channel measures "
                      f"{c['measured']:g}. The measured value is used.")
    else:
        print("=== protocol: not in the manifest (older run) ===")
        print("  Windows fall back to the command-line defaults. Re-run the "
              "pipeline to record the measured protocol.")

    dff = pd.read_parquet(d / f"{sid}_dff.parquet")
    events = pd.read_parquet(d / f"{sid}_events.parquet")
    if (args.protocol_class is not None
            and "protocol_class" in events.columns):
        events = events[events["protocol_class"] == args.protocol_class]
        if events.empty:
            print(f"  no events in protocol class {args.protocol_class}")
            return 1
    rois = [c for c in dff.columns
            if c not in ("t", "frame", "blanked") and not c.endswith("__iso")]
    has_iso = any(c.endswith("__iso") for c in dff.columns)
    t = dff["t"].to_numpy()
    fs = 1.0 / np.median(np.diff(t))

    # Default windows follow the measured protocol: a response window that
    # spans the whole train plus the indicator's decay, and a slow window
    # after it, both bounded by half the same-site interval so that a window
    # can never reach into the previous or next stimulation of that site.
    # With interleaved conditions the window must come from the class being
    # analysed, not from a session-level summary that averages across them.
    bd = proto.get("burst_duration_s")
    if args.protocol_class is not None and "width_s" in events.columns:
        bd = float(events["width_s"].median())
        print(f"  class {args.protocol_class}: measured command duration "
              f"{bd*1e3:.0f} ms (window derived from this class)")
    elif ("width_s" in events.columns and not bd
          and events["width_s"].notna().any()):
        bd = float(events["width_s"].median())
    half_isi = 0.5 * proto.get("same_site_interval_s", np.inf)
    if bd and not getattr(args, "window_label", None) and args.windows is None:
        want = min(bd + 3 * args.decay_tau, half_isi)
        if abs(want - args.response_window) > 0.05:
            print(f"\n  response window set from the protocol: 0 to "
                  f"{want:.2f} s (burst {bd:.2f} s + 3 decay constants"
                  + (", capped at half the same-site interval"
                     if want >= half_isi - 1e-9 else "") + ")")
            args.response_window = float(want)
        sw0 = min(want, half_isi)
        sw1 = min(max(sw0 + 1.0, 4.0), half_isi)
        if sw1 > sw0:
            args.slow_window = (float(sw0), float(sw1))

    if args.lowpass:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, args.lowpass, "lowpass", fs=fs, output="sos")
        for r in rois:
            v = dff[r].to_numpy(dtype=float)
            ok = np.isfinite(v)
            if ok.sum() > 20:
                f = v.copy()
                f[ok] = sosfiltfilt(sos, v[ok])
                dff[r] = f
        print(f"low-passed at {args.lowpass} Hz "
              f"(indicator bandwidth; applied before any windowing)")

    print(f"=== {sid} ===")
    print(f"{len(rois)} ROIs, {len(events)} events, {fs:.1f} Hz, "
          f"blanked {qc.get('n_blanked')} samples, "
          f"exposure_s {qc.get('exposure_s')}")
    if events.empty:
        print("no events; nothing to plot")
        return 1

    # Per-channel stimulation floor, if a previous run left an effects table;
    # used to put a scale bar of meaningful length on each overview trace.
    floors = None
    eff_prev = out / f"{sid}_effects{suf}.csv"
    if eff_prev.exists():
        try:
            _e = pd.read_csv(eff_prev)
            floors = (_e[_e.stim_site != _e.recorded_site]
                      .groupby("recorded_site")["null_sd"].median() * 3)
        except Exception:
            floors = None

    # ---------------- overview ------------------------------------------- #
    fig, axes = plt.subplots(len(rois), 1, figsize=(7.6, 0.62 * len(rois)),
                             sharex=True)
    # Each fiber keeps its own y-scale -- they differ by orders of magnitude,
    # so a shared axis would flatten most of them -- but the scale is shown
    # rather than hidden. Without it, a 0.05% trace and a 5% trace look
    # identical, which is how a quiet channel's noise gets read as a response.
    for ax, r in zip(np.atleast_1d(axes), rois):
        y = dff[r].to_numpy(dtype=float)
        ax.plot(t, y, color="0.2", lw=0.4)
        for _, e in events.iterrows():
            ax.axvspan(e.onset_t, e.offset_t, color="C0", alpha=0.18, lw=0)
        ax.axhline(0, color="k", lw=0.3, ls=":")

        finite = y[np.isfinite(y)]
        if finite.size:
            lo, hi = np.percentile(finite, [0.5, 99.5])
            pad = 0.1 * max(hi - lo, 1e-9)
            ax.set_ylim(lo - pad, hi + pad)
            # ticks at zero and the robust extremes, labelled as percent
            ticks = [lo, 0.0, hi] if lo < 0 < hi else [lo, hi]
            ax.set_yticks(sorted(set(np.round(ticks, 6))))
            ax.set_yticklabels([f"{v:.2%}" for v in sorted(set(np.round(ticks, 6)))],
                               fontsize=4.5)
            # a scale bar whose length is this channel's own stimulation floor,
            # when one is known, so the trace can be read against it
            if floors is not None and r in floors.index:
                fl = float(floors[r])
                x0 = t[0] + 0.01 * (t[-1] - t[0])
                ax.plot([x0, x0], [0, fl], color="C3", lw=1.4,
                        solid_capstyle="butt")
                ax.text(x0 + 0.004 * (t[-1] - t[0]), fl / 2,
                        f"floor {fl:.2%}", color="C3", fontsize=4,
                        va="center")
        else:
            ax.set_yticks([])
        ax.set_ylabel(r, rotation=0, ha="right", va="center", fontsize=6)
        ax.tick_params(axis="y", length=1.5, pad=1)
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    np.atleast_1d(axes)[len(rois) // 2].text(
        -0.075, 0.5, r"$\Delta F/F$", transform=np.atleast_1d(axes)[len(rois)//2].transAxes,
        rotation=90, va="center", ha="center", fontsize=7)
    np.atleast_1d(axes)[0].set_title(
        f"{sid}  —  shaded: laser command"
        + (f"  (window set: {tag})" if tag else ""), fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"{sid}_overview{suf}.pdf")
    plt.close(fig)
    print(f"  {sid}_overview.pdf")

    # ---------------- peri-stimulus, per stimulated site ------------------ #
    nbase = int(args.baseline_window * fs)
    rows = []

    for site, grp in events.groupby("region"):
        if site is None:
            continue
        # t = 0 is the onset of the FIRST pulse of each burst. This is the
        # one alignment that is comparable across protocols: it does not move
        # when the train length or the pulse rate changes, whereas aligning
        # to the last pulse or to the burst centre does.
        onsets = grp["onset_t"].to_numpy()
        if "offset_t" in grp.columns:
            spans = grp["offset_t"].to_numpy() - onsets
            if np.nanmedian(spans) < 0:
                raise ValueError(
                    f"{site}: offset precedes onset; events are not "
                    "first-pulse aligned")
        idx, npre, npost = _epochs(t, onsets, args.pre, args.post, fs)
        if idx.size == 0:
            print(f"  {site}: no complete window; skipped")
            continue

        n_ev = idx.size
        lo_i, hi_i = npre, t.size - npost
        span = hi_i - lo_i
        if span < 4 * (npre + npost):
            print(f"  {site}: recording too short for a shift null")
            continue

        # Circular shifts of the whole event train, wrapped into the usable
        # index range.  Shifts smaller than one window are excluded, since
        # those overlap the real events they are meant to be a null for.
        min_shift = npre + npost
        shifts = rng.integers(min_shift, span - min_shift, size=args.n_null)
        rel = idx - lo_i
        null_sets = [((rel + sh) % span) + lo_i for sh in shifts]

        fig, axes = plt.subplots(3, 4, figsize=(9.5, 6.2), sharex=True)
        tt = (np.arange(-npre, npost)) / fs
        width = float(grp["width_s"].median())
        kern = gcamp_kernel(tt, width, args.rise_tau, args.decay_tau)
        col_x = np.arange(tt.size, dtype=float)
        # Applied identically to the observed trials and to every circular
        # shift, so the null distribution is of the same statistic.
        def _combine(A):
            # All-NaN columns are expected wherever every trial was blanked;
            # they are reported as missing data rather than warned about on
            # every call.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return (np.nanmedian(A, axis=0) if args.statistic == "median"
                        else np.nanmean(A, axis=0))
        pool_single = np.flatnonzero(
            np.ones(t.size, dtype=bool)
            & (np.arange(t.size) >= npre) & (np.arange(t.size) < t.size - npost))
        tail0 = (args.tail_start if args.tail_start is not None
                 else width + 3 * args.decay_tau)
        fit_cols = np.flatnonzero((tt < 0) | (tt >= tail0))
        if fit_cols.size < 8:
            fit_cols = np.flatnonzero(tt < 0)
        r0 = npre + int(args.response_start * fs)
        r1 = r0 + int(args.response_window * fs)
        sw0, sw1 = args.slow_window
        s0 = npre + int(sw0 * fs)
        s1 = npre + int(sw1 * fs) if sw1 > sw0 else s0
        # Channels where every trial is blanked for part of the response window.
        blank_frac: dict = {}
        for ax, r in zip(axes.ravel(), rois):
            y = dff[r].to_numpy(dtype=float)

            def trial_mean(centres):
                W = np.stack([y[i - npre:i + npost] for i in centres])
                if args.detrend == "mean":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        W = W - np.nanmean(W[:, npre - nbase:npre], axis=1,
                                           keepdims=True)
                else:
                    # Line through the pre-window and a late post-window, both
                    # outside the expected response, removes within-trial
                    # drift as well as offset.
                    W = W - _linear_baseline(W, fit_cols, col_x)
                return _combine(W)

            obs = trial_mean(idx)
            null_tr = np.stack([trial_mean(c) for c in null_sets])

            if args.estimator == "kernel":
                eff = float(np.nansum(np.where(np.isfinite(obs), obs, 0.0) * kern))
                nt = np.where(np.isfinite(null_tr), null_tr, 0.0)
                null_eff = nt @ kern
            else:
                eff = float(np.nanmean(obs[r0:r1]))
                null_eff = np.nanmean(null_tr[:, r0:r1], axis=1)

            # The slow window is a different phenomenon, not a longer look at
            # the same one: past ~1 s nothing is indicator kinetics.
            eff_slow = z_slow = p_slow = np.nan
            if s1 > s0:
                eff_slow = float(np.nanmean(obs[s0:s1]))
                ns = np.nanmean(null_tr[:, s0:s1], axis=1)
                mus, sds = float(np.nanmean(ns)), float(np.nanstd(ns))
                z_slow = (eff_slow - mus) / sds if sds > 0 else np.nan
                p_slow = (1.0 + int(np.sum(np.abs(ns - mus)
                                           >= abs(eff_slow - mus)))) \
                    / (ns.size + 1.0)
            mu, sd = float(np.nanmean(null_eff)), float(np.nanstd(null_eff))
            z = (eff - mu) / sd if sd > 0 else np.nan
            # two-sided empirical p, with the +1 correction so that p is never
            # reported as 0 from a finite number of shifts
            p = (1.0 + int(np.sum(np.abs(null_eff - mu) >= abs(eff - mu)))) \
                / (null_eff.size + 1.0)
            rows.append({"stim_site": site, "recorded_site": r,
                         "effect": eff, "null_mean": mu, "null_sd": sd,
                         "z": z, "p_shift": p, "n_events": int(n_ev),
                         "n_shifts": int(null_eff.size),
                         "effect_slow": eff_slow, "z_slow": z_slow,
                         "p_slow": p_slow})

            lo, hi = np.nanpercentile(null_tr, [2.5, 97.5], axis=0)
            ax.fill_between(tt, lo, hi, color="0.85", lw=0)
            show_trials = (args.show_trials == "always"
                           or (args.show_trials == "auto"
                               and n_ev <= args.max_trials_shown))
            if show_trials:
                # With few trials the mean hides how variable they are; show
                # them -- but against a band for SINGLE trials, not the band
                # for means. The grey band is ~sqrt(n) narrower, so individual
                # trials lie outside it routinely and comparing them to it is
                # a category error.
                W = np.stack([y[i - npre:i + npost] for i in idx])
                W = W - np.nanmean(W[:, npre - nbase:npre], axis=1,
                                   keepdims=True)
                single = np.stack([y[i - npre:i + npost]
                                   for i in rng.choice(pool_single,
                                                       size=400,
                                                       replace=False)])
                single = single - np.nanmean(single[:, npre - nbase:npre],
                                             axis=1, keepdims=True)
                slo, shi = np.nanpercentile(single, [2.5, 97.5], axis=0)
                ax.plot(tt, slo, color="0.55", lw=0.4, ls=":")
                ax.plot(tt, shi, color="0.55", lw=0.4, ls=":")
                ax.plot(tt, W.T, color="C3", lw=0.3, alpha=0.35)
            ax.axvspan(-args.baseline_window, 0, color="0.6", alpha=0.25, lw=0)
            ax.plot(tt, obs, color="C3", lw=1.0)
            # Where every trial is blanked the mean does not exist. Drawn
            # explicitly: a gap in a line is easily read as "no response"
            # rather than "no data".
            n_ok_t = np.isfinite(np.stack([y[i - npre:i + npost]
                                           for i in idx])).sum(axis=0)
            gap = n_ok_t == 0

            # Scale the axis to the MEAN and its null band, not to the
            # trials. With few trials one excursion otherwise sets the range
            # and flattens the quantity being measured; trials that leave the
            # axis are counted in the title rather than silently cropped.
            ref = np.concatenate([obs[np.isfinite(obs)],
                                  lo[np.isfinite(lo)], hi[np.isfinite(hi)]])
            if ref.size:
                span = float(np.nanmax(ref) - np.nanmin(ref)) or 1e-6
                ylo = float(np.nanmin(ref)) - 0.18 * span
                yhi = float(np.nanmax(ref)) + 0.18 * span
                ax.set_ylim(ylo, yhi)
            else:
                ylo, yhi = ax.get_ylim()
            n_clipped = 0
            if show_trials:
                n_clipped = int(np.sum(np.any((W < ylo) | (W > yhi), axis=1)))

            if gap.any():
                ax.fill_between(tt, ylo, yhi, where=gap, color="C1",
                                alpha=0.30, lw=0, step="mid", zorder=0)
                blank_frac[r] = float(gap[r0:r1].mean())
            ax.axvspan(0, width, color="C0", alpha=0.10, lw=0)
            _shade_timescales(ax)
            ax.axhline(0, color="k", lw=0.4, ls=":")

            # Inset on the indicator's own timescale. The main axis spans
            # -2 to +4 s, within which the entire GCaMP6f transient occupies
            # about one twelfth of the width and cannot be read.
            axi = ax.inset_axes([0.56, 0.60, 0.42, 0.38])
            m = (tt >= -0.2) & (tt <= 0.6)
            axi.fill_between(tt[m], lo[m], hi[m], color="0.85", lw=0)
            axi.plot(tt[m], obs[m], color="C3", lw=0.9)
            _shade_timescales(axi, show_slow=False)
            if gap[m].any():
                # The gap is a fraction of a second: invisible on the main
                # axis, and exactly where the response would be.
                ilo, ihi = axi.get_ylim()
                axi.fill_between(tt[m], ilo, ihi, where=gap[m], color="C1",
                                 alpha=0.35, lw=0, step="mid", zorder=0)
                axi.set_ylim(ilo, ihi)
            axi.axhline(0, color="k", lw=0.3, ls=":")
            axi.axvline(0, color="k", lw=0.4, ls="--")
            axi.set_xticks([0, 0.1, 0.5])
            axi.set_xticklabels(["0", ".1", ".5"], fontsize=4)
            axi.tick_params(axis="y", labelsize=4, length=1.5)
            axi.set_xlim(-0.2, 0.6)
            for sp in axi.spines.values():
                sp.set_linewidth(0.4)

            if args.estimator == "kernel":
                sc = np.nanmax(np.abs(obs)) or 1.0
                ax.plot(tt, kern / (np.max(np.abs(kern)) or 1.0) * sc * 0.6,
                        color="C0", lw=0.5, ls="--", alpha=0.7)
            ttl = f"{r}  z={z:+.1f} p={p:.3f}"
            if np.isfinite(z_slow):
                ttl += f"   slow z={z_slow:+.1f}"
            if n_clipped:
                ttl += f"   [{n_clipped}/{n_ev} trials off-axis]"
            if gap.any():
                ttl += f"   [{gap.sum()} blanked]"
            ax.set_title(ttl, fontsize=5.5)
            ax.tick_params(labelsize=5)
        for ax in axes[-1]:
            ax.set_xlabel("time from onset (s)")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$\Delta F/F$")
        win = (f"{args.response_start:+.2f} to "
               f"{args.response_start + args.response_window:+.2f} s"
               + (f" [{tag}]" if tag else ""))
        est = (f"kernel (rise {args.rise_tau*1e3:.0f} ms, "
               f"decay {args.decay_tau*1e3:.0f} ms)"
               if args.estimator == "kernel"
               else f"boxcar {args.response_start:+.1f} to "
                    f"{args.response_start + args.response_window:+.1f} s")
        fig.suptitle(f"{sid}  —  stimulating {site}  (n={n_ev} events; "
                     f"window {win}; t=0 is the FIRST pulse of the burst; "
                     f"grey = 95% of {args.n_null} circular-shift trial means; "
                     f"{est}; combined by {args.statistic})", fontsize=6.5)
        if blank_frac:
            worst = max(blank_frac.values())
            n_bad = sum(1 for v in blank_frac.values() if v > 0.5)
            print(f"  stimulating {site}: up to {worst:.0%} of the response "
                  f"window has no data in {len(blank_frac)} channel(s) "
                  f"({n_bad} above 50%).")
            if worst > 0.5:
                print("     !! most of the window is missing. At a "
                      "stimulation rate near the camera frame rate the gate "
                      "fires every frame, so the train is flagged as "
                      "contaminated throughout. Either the interlock is not "
                      "holding or exposure_s is wrong; check "
                      "qc.gate_phase.interlock_holds. If it genuinely does "
                      "not hold, the response cannot be observed during the "
                      "train and the window must start after it.")
            elif worst > 0.15:
                print("     A sizeable minority of the window is missing; "
                      "the mean is built from fewer trials there than "
                      "elsewhere.")
            else:
                print("     A small fraction, consistent with the interlock "
                      "holding and only occasional frames being clipped.")

        handles_ax = axes.ravel()[0]
        _shade_timescales(handles_ax, labels=True)
        h, lbl = handles_ax.get_legend_handles_labels()
        seen, hh, ll = set(), [], []
        for a, b in zip(h, lbl):
            if b not in seen:
                seen.add(b); hh.append(a); ll.append(b)
        fig.legend(hh, ll, loc="lower center", ncol=4, frameon=False,
                   fontsize=5.5, bbox_to_anchor=(0.5, -0.005))
        fig.tight_layout(rect=(0, 0.035, 1, 0.95))
        fig.savefig(out / f"{sid}_peristim_{site}{suf}.pdf")
        plt.close(fig)
        print(f"  {sid}_peristim_{site}.pdf")

    df_all = pd.DataFrame(rows)
    if not df_all.empty:
        print("\n=== what this session can resolve ===")
        # Sensitivity is a property of the RECORDED channel, not of the site
        # being stimulated: the null SD is the variability of that fiber's
        # trace. Averaging over recorded sites hides a range that can span
        # two orders of magnitude, understating the floor in quiet channels
        # and overstating it in noisy ones.
        off = df_all[df_all.stim_site != df_all.recorded_site]
        per_rec = (off.groupby("recorded_site")["null_sd"].median()
                   .sort_values())
        n_ev_med = int(df_all["n_events"].median())
        print(f"  n = {n_ev_med} bursts per site. Floor is per RECORDED "
              "channel:")
        print(f"  {'recorded':10s} {'null SD':>10s} {'floor (z=3)':>12s}")
        for rec, sd in per_rec.items():
            print(f"  {rec:10s} {sd:10.6f} {3 * sd:11.3%}")
        lo, hi = float(per_rec.min()), float(per_rec.max())
        print(f"\n  Floor ranges {3*lo:.3%} to {3*hi:.2%} across channels "
              f"({hi/lo:.0f}x). A single session-wide floor is not "
              "meaningful; quote the floor of the channel an effect was seen "
              "in.")
        print("  The floor scales as 1/sqrt(n), so 4x more bursts halves it.")

        # ---- how many trials would each channel need? ------------------- #
        # The null SD above is the SD of the trial MEAN at this session's n.
        # Single-trial variability follows as sd1 = null_sd * sqrt(n), and
        # the n needed to resolve an effect of a given size is then
        # n_req = (z * sd1 / effect)^2. Required n differs across channels by
        # the square of their noise ratio, so a session-wide trial count is
        # adequate for some fibers and hopeless for others.
        tgt, tz = float(args.target_effect), float(args.target_z)
        print(f"\n  --- trials needed to resolve a {tgt:.2%} effect at "
              f"z={tz:g} ---")
        print("  (and the effect each channel CAN resolve at this session's "
              f"n={n_ev_med})")
        print(f"  {'recorded':10s} {'single-trial SD':>16s} "
              f"{'resolvable now':>15s} {'n needed':>9s} "
              f"{'session at 10s ISI':>19s}")
        rows_pw = []
        for rec, sdm in per_rec.items():
            sd1 = float(sdm) * np.sqrt(max(n_ev_med, 1))
            # At least one trial, and never fewer than it takes to have a
            # trial mean at all. A fractional requirement means the effect is
            # already far above this channel's noise.
            n_req = max(1.0, (tz * sd1 / tgt) ** 2) if tgt > 0 else np.inf
            resolvable = tz * sdm
            mins = n_req * 10.0 / 60.0
            rows_pw.append({"recorded_site": rec, "single_trial_sd": sd1,
                            "resolvable_at_current_n": resolvable,
                            "n_required": n_req,
                            "target_effect": tgt, "target_z": tz})
            ok = n_req <= n_ev_med
            print(f"  {rec:10s} {sd1:16.4%} {resolvable:14.3%} "
                  f"{n_req:9.0f} "
                  + ("     already enough" if ok
                     else f"{mins:16.0f} min"))
        pw = pd.DataFrame(rows_pw).sort_values("n_required")
        pw.to_csv(out / f"{sid}_power{suf}.csv", index=False)

        n_ok = int((pw["n_required"] <= n_ev_med).sum())
        print(f"\n  {n_ok}/{len(pw)} channel(s) can already resolve "
              f"{tgt:.2%} at n={n_ev_med}.")
        if n_ok == len(pw):
            print(f"  Every channel could have detected {tgt:.2%}. Its "
                  "absence is therefore evidence about the perturbation, "
                  "not about the sensitivity of the measurement.")
        med_req = float(pw["n_required"].median())
        print(f"  Median channel needs n={med_req:.0f}; the quietest needs "
              f"n={pw['n_required'].min():.0f}, the noisiest "
              f"n={pw['n_required'].max():.0f} "
              f"({pw['n_required'].max() / max(pw['n_required'].min(), 1):.0f}x "
              "spread, the square of the noise ratio).")
        print("  Required n scales with the SQUARE of the effect you want to "
              "resolve: halving the target quadruples the trials.")
        if med_req > 4 * n_ev_med:
            print(f"  !! At n={n_ev_med} this session cannot resolve "
                  f"{tgt:.2%} in most channels. A null result here is a "
                  "statement about the measurement, not about the biology.")
        print(f"  {sid}_power{suf}.csv")

    diag = df_diag_report(rows)
    if diag:
        print(diag)

    if has_iso and not df_all.empty:
        print("\n=== bleaching control: same contrast on the 415 channel ===")
        print("A decrease at the stimulated site is produced BOTH by "
              "inhibition of the\nrecorded population and by the stimulation "
              "light bleaching or quenching the\nindicator. The 470 channel "
              "cannot tell them apart; the isosbestic can, because\n"
              "bleaching removes fluorophore and calcium changes do not.\n")
        print("Only sites whose own fiber responded ABOVE that channel's "
              "floor are\ninformative here; the rest have no effect to "
              "attribute to anything.\n")
        print(f"{'site':10s} {'470 effect':>12s} {'415 effect':>12s} "
              f"{'415/470':>9s} {'floor':>8s}  verdict")
        for site, g in df_all.groupby("stim_site"):
            if site not in rois or f"{site}__iso" not in dff.columns:
                continue
            onsets = events.loc[events["region"] == site, "onset_t"].to_numpy()
            idx, npre, npost = _epochs(t, onsets, args.pre, args.post, fs)
            if idx.size == 0:
                continue
            r0 = npre + int(args.response_start * fs)
            r1 = r0 + int(args.response_window * fs)

            # Against the RECORDED channel's own floor, not a session-wide
            # median: null SD spans two orders of magnitude across channels,
            # so a median threshold calls quiet channels significant and
            # noisy ones null regardless of the data.
            own = df_all[(df_all.stim_site == site)
                         & (df_all.recorded_site == site)]
            sd_own = (float(own["null_sd"].iloc[0]) if len(own)
                      else float(np.nanmedian(df_all["null_sd"])))

            def contrast(col):
                """Same contrast as the headline analysis, applied to one
                column. Kept for the 415 channel only: the 470 value is taken
                from the computed results so the two tables cannot disagree.
                """
                y = dff[col].to_numpy(dtype=float)
                W = np.stack([y[i - npre:i + npost] for i in idx])
                if args.detrend == "mean":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        W = W - np.nanmean(W[:, npre - nbase:npre], axis=1,
                                           keepdims=True)
                else:
                    cx = np.arange(W.shape[1], dtype=float)
                    tail = npre + int((float(
                        events.loc[events["region"] == site, "width_s"].median()
                    ) + 1.2) * fs)
                    fc = np.concatenate([np.arange(0, npre),
                                         np.arange(min(tail, W.shape[1]),
                                                   W.shape[1])])
                    W = W - _linear_baseline(W, fc, cx)
                combined = (np.nanmedian(W, axis=0)
                            if args.statistic == "median"
                            else np.nanmean(W, axis=0))
                return float(np.nanmean(combined[r0:r1]))

            # 470 comes from the headline computation, not a second pass.
            e470 = float(own["effect"].iloc[0]) if len(own) else contrast(site)
            e415 = contrast(f"{site}__iso")
            ratio = e415 / e470 if abs(e470) > 1e-9 else np.nan
            if abs(e470) < 3 * sd_own:
                verdict = f"below this channel's floor ({3*sd_own:.2%})"
            elif np.isfinite(ratio) and ratio > 0.5:
                verdict = "TRACKS 415 -> bleaching/quenching, not calcium"
            elif np.isfinite(ratio) and ratio < 0.2:
                verdict = "absent from 415 -> consistent with a real response"
            else:
                verdict = "partial; inconclusive"
            print(f"{site:10s} {e470:12.5f} {e415:12.5f} {ratio:9.2f} "
                  f"{3*sd_own:7.2%}  {verdict}")

    if not rows:
        return 0

    # ---------------- effect matrix --------------------------------------- #
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{sid}_effects{suf}.csv", index=False)
    mat = df.pivot(index="stim_site", columns="recorded_site", values="z")
    mat = mat.reindex(columns=[r for r in rois if r in mat.columns])

    fig, ax = plt.subplots(figsize=(0.42 * mat.shape[1] + 2.2,
                                    0.42 * mat.shape[0] + 1.8))
    vmax = float(np.nanmax(np.abs(mat.to_numpy()))) if mat.size else 1.0
    vmax = max(vmax, 1.0)
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=90)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_xlabel("recorded site")
    ax.set_ylabel("stimulated site")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=5,
                        color="w" if abs(v) > 0.6 * vmax else "k")
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("effect (SD of time-shuffled null)")
    ax.set_title(f"{sid}  —  window {win}\n"
                 "NOT a test: no sham, one animal, no correction for "
                 "multiple site pairs", fontsize=6)
    fig.tight_layout()
    fig.savefig(out / f"{sid}_matrix{suf}.pdf")
    plt.close(fig)
    print(f"  {sid}_matrix.pdf\n  {sid}_effects.csv")

    big = df.reindex(df["z"].abs().sort_values(ascending=False).index).head(10)
    print("\nlargest |z| (exploratory, uncorrected):")
    print(big[["stim_site", "recorded_site", "effect", "z", "p_shift",
               "n_events"]].to_string(index=False))

    # ---- non-specific, stimulation-locked response ----------------------- #
    # A deflection that follows EVERY stimulation regardless of which site was
    # stimulated is not connectivity: it is a response to the act of
    # stimulating (galvo motion, electrical pickup, light leak, startle). It
    # masquerades as convergent connectivity, and it surfaces first in the
    # quietest recorded channel, because that is where a small common
    # deflection clears significance.
    n_stim_sites = int(df_all["stim_site"].nunique()) if not df_all.empty else 0
    if not df_all.empty and n_stim_sites < 4:
        print("\n=== non-specific response to stimulation ===")
        print(f"  Not assessed: {n_stim_sites} stimulated site(s) in this "
              "session. Distinguishing a response to a particular site from "
              "a response to the act of stimulating requires several "
              "stimulated sites; with one, 'the same sign after every site' "
              "is true by construction.")
    elif not df_all.empty:
        print("\n=== non-specific response to stimulation ===")
        print(f"{'recorded':10s} {'mean effect':>12s} {'n stim sites':>12s} "
              f"{'same sign':>10s} {'median null SD':>15s} {'mean/SD':>8s}")
        nonspec = []
        for rec, g in df_all[df_all.stim_site != df_all.recorded_site] \
                .groupby("recorded_site"):
            e = g["effect"].to_numpy(dtype=float)
            same = max(int((e > 0).sum()), int((e < 0).sum()))
            sd = float(np.nanmedian(g["null_sd"]))
            ratio = float(np.mean(e)) / sd if sd > 0 else np.nan
            nonspec.append({"recorded_site": rec, "mean_effect": float(np.mean(e)),
                            "n": len(e), "same_sign": same, "null_sd": sd,
                            "ratio": ratio})
            print(f"{rec:10s} {np.mean(e):12.6f} {len(e):12d} "
                  f"{same:>7d}/{len(e)} {sd:15.6f} {ratio:8.2f}")
        nd = pd.DataFrame(nonspec)
        flagged = nd[(nd["same_sign"] >= 0.8 * nd["n"]) & (nd["ratio"].abs() > 1)]
        if len(flagged):
            print(f"\n  !! {len(flagged)} recorded site(s) show a consistent "
                  "deflection after stimulation of ALMOST EVERY site: "
                  f"{', '.join(flagged['recorded_site'])}.")
            print("     Convergence of many unrelated sources on one target "
                  "with one sign is not connectivity. Treat these rows as a "
                  "non-specific response to stimulation until a sham "
                  "condition separates them.")
        quiet = nd.nsmallest(1, "null_sd")
        if len(quiet):
            q = quiet.iloc[0]
            print(f"\n  Quietest recorded channel: {q.recorded_site} "
                  f"(null SD {q.null_sd:.6f}, "
                  f"{nd.null_sd.max()/q.null_sd:.1f}x below the noisiest). "
                  "Sensitivity differs that much between channels, so the "
                  "per-stim-site floor above understates what can be detected "
                  "there and overstates it elsewhere.")

    # ---- window contamination ------------------------------------------- #
    # The peri-event contrast is response minus a local pre-window. It
    # recovers a response faithfully ONLY while that pre-window is free of
    # stimulation: simulated, a dip shorter than the interval recovers at
    # ~100%, while one that outlasts it recovers at a few percent, because
    # the baseline is already depressed and the contrast cancels. With many
    # sites interleaved, the gap between ANY two events is much shorter than
    # the gap between repeats of one site.
    all_on = events["onset_t"].to_numpy(dtype=float)
    all_off = events["offset_t"].to_numpy(dtype=float)
    print("\n=== peri-event window contamination ===")
    gaps = np.diff(np.sort(all_on))
    print(f"  interval between ANY two events: median {np.median(gaps):.2f} s "
          f"(min {gaps.min():.2f} s)")
    print(f"  baseline window {args.baseline_window:.1f} s, "
          f"pre-window {args.pre:.1f} s, "
          f"response window {args.response_window:.1f} s")
    rows_c = []
    for site, g in events.groupby("region"):
        if site is None:
            continue
        ons = g["onset_t"].to_numpy(dtype=float)
        own = np.diff(np.sort(ons))
        n_base = n_resp = 0
        for o in ons:
            other = (all_on != o)
            in_base = np.any((all_off[other] > o - args.baseline_window)
                             & (all_on[other] < o))
            in_resp = np.any((all_on[other] < o + args.response_window)
                             & (all_off[other] > o))
            n_base += int(in_base)
            n_resp += int(in_resp)
        rows_c.append({"site": site, "n": len(ons),
                       "own_isi_s": float(np.median(own)) if own.size else np.nan,
                       "base_contaminated": n_base,
                       "resp_contaminated": n_resp})
    cdf = pd.DataFrame(rows_c)
    print(cdf.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
    tot_b = int(cdf["base_contaminated"].sum())
    tot_r = int(cdf["resp_contaminated"].sum())
    if tot_b or tot_r:
        print(f"\n  !! {tot_b} trials have ANOTHER site's stimulation inside "
              f"their baseline window and {tot_r} inside their response "
              "window. Where that other site drives the recorded fiber, the "
              "contrast is attenuated (baseline contamination) or "
              "misattributed (response contamination).")
        print("     Shorten the windows, or interleave sites with a longer "
              "gap at acquisition.")
    else:
        print("\n  No trial has another site's stimulation inside either "
              "window.")
    med_isi = float(np.nanmedian(cdf["own_isi_s"]))
    if np.isfinite(med_isi):
        print(f"\n  Median interval between repeats of one site: "
              f"{med_isi:.1f} s. A response outlasting roughly half of this "
              "cannot be measured by a pre-window contrast at all, whatever "
              "the baseline estimator.")

    # ---- optical crosstalk check ---------------------------------------- #
    cal = man.get("inputs", {}).get("calibration", {})
    coords_path = (cal.get("files", {}).get("translated_coords", {})
                   .get("path"))
    if coords_path and Path(coords_path).exists():
        try:
            from multifiber.matio import load_mat

            tc = np.atleast_2d(np.asarray(
                load_mat(coords_path)["translated_coords"], dtype=float))
            order = rois
            dist = {}
            for i, a in enumerate(order):
                for j, b in enumerate(order):
                    dist[(a, b)] = float(np.hypot(tc[i, 0] - tc[j, 0],
                                                  tc[i, 1] - tc[j, 1]))
            df["bundle_distance_px"] = [dist.get((r.stim_site,
                                                  r.recorded_site), np.nan)
                                        for r in df.itertuples()]
            offd = df[df["stim_site"] != df["recorded_site"]].dropna(
                subset=["bundle_distance_px"])
            if len(offd) > 5:
                rho = float(np.corrcoef(offd["bundle_distance_px"],
                                        offd["z"].abs())[0, 1])
                print("\n=== optical crosstalk check ===")
                print("Light from a stimulated fiber can scatter into "
                      "physically adjacent fibers at the bundle face and "
                      "appear as a downstream effect. Bundle geometry and "
                      "anatomy are unrelated, so a dependence of effect size "
                      "on distance IN THE BUNDLE is a signature of crosstalk "
                      "rather than connectivity.")
                print(f"  corr(|z|, bundle distance) over {len(offd)} "
                      f"off-diagonal pairs: {rho:+.3f}")
                if rho < -0.3:
                    print("  -> effects are larger for nearby fibers: "
                          "OPTICAL CROSSTALK IS A LIVE CONCERN.")
                else:
                    print("  -> no distance dependence; crosstalk is not "
                          "driving the pattern.")
        except Exception as exc:
            print(f"\n(crosstalk check skipped: {exc})")

    # ---- restrict to sites with a demonstrated perturbation -------------- #
    if args.only_responsive:
        d = df[df["stim_site"] == df["recorded_site"]]
        good = set(d.loc[d["z"].abs() >= args.diag_z, "stim_site"])
        if good:
            print(f"\nRestricting to {len(good)} site(s) with a demonstrated "
                  f"perturbation: {', '.join(sorted(good))}")
            df = df[df["stim_site"].isin(good)]
        else:
            print("\n--only-responsive requested but no site passed the "
                  "diagonal threshold; keeping all sites.")

    df = df[df["stim_site"] != df["recorded_site"]]
    n_pairs = len(df)
    from statsmodels.stats.multitest import multipletests
    rej, p_adj, _, _ = multipletests(df["p_shift"].to_numpy(), method="fdr_bh")
    df["p_fdr"] = p_adj
    df.to_csv(out / f"{sid}_effects{suf}.csv", index=False)
    if "z_slow" in df.columns and df["z_slow"].notna().any():
        print("\n=== slow window "
              f"({args.slow_window[0]:.1f}-{args.slow_window[1]:.1f} s) ===")
        print("Past ~1 s nothing is GCaMP6f kinetics. Anything here is "
              "network reverberation, a behavioural consequence of the "
              "stimulus, or drift. It is reported separately and should be "
              "described as a slower phenomenon, not as the response.")
        sl = df.reindex(df["z_slow"].abs().sort_values(ascending=False).index)
        print(sl[["stim_site", "recorded_site", "effect_slow", "z_slow",
                  "p_slow"]].head(8).to_string(index=False))
        agree = df.dropna(subset=["z", "z_slow"])
        if len(agree) > 3:
            rho = float(np.corrcoef(agree["z"], agree["z_slow"])[0, 1])
            print(f"\n  corr(z fast, z slow) over {len(agree)} pairs: "
                  f"{rho:+.3f}")
            if abs(rho) > 0.6:
                print("  -> the two windows are measuring the same thing, "
                      "which at this separation means drift spanning both "
                      "rather than a transient plus its aftermath.")

    print(f"\n{n_pairs} off-diagonal site pairs examined. "
          f"Uncorrected p<0.05 in "
          f"{int((df['p_shift'] < 0.05).sum())} "
          f"({0.05 * n_pairs:.1f} expected by chance); "
          f"Benjamini-Hochberg FDR q<0.05 in {int(rej.sum())}.")
    print("Shift p-values have a floor of "
          f"{1.0 / (args.n_null + 1):.4f} with {args.n_null} shifts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
