"""
Interactive exploration of one processed session.

Two tabs over the same data:

**preprocessing** — two configurations side by side. Change a baseline
timescale, an isosbestic regressor or a window and see what it does to the
trace, to the peri-event average, to the effect in each condition, and to the
statistics.

**playback** — the session as it ran. Raw camera output on top, processed
dF/F below, on a shared clock, with laser commands shaded and blanked samples
marked. Play, scrub, change the window, pick fibers. Preprocessing can remove
an artifact or manufacture one; watching both panels as the protocol runs is
how you tell which, and no summary figure shows it.

    conda activate multifiber
    pip install dash plotly
    python scripts/compare_app.py "F:\\Jorge\\Multistim_Analysis\\<session>"

then open http://127.0.0.1:8050

What this is for, and what it is not.  Comparing options on the data is how
you understand what a choice does; choosing the option that gives the biggest
effect is a free parameter selected on the outcome, and with twelve fibers
and a dozen knobs it will always find one.  Decide on a diagnostic -- does
the baseline track the bleaching, does the regression lower the noise, does
the injected response come back -- and not on the z-score.

The null is recomputed for every variant.  A null built under one baseline
does not describe another, so comparing variant A's effect against variant
B's null would be meaningless, and the app never does it.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_controls(dash, prefix: str, defaults: dict):
    """One column of controls for a variant."""
    from dash import dcc, html

    def row(label, ctrl, hint=""):
        return html.Div([
            html.Label(label, style={"fontSize": "11px", "fontWeight": 600}),
            ctrl,
            html.Div(hint, style={"fontSize": "9px", "color": "#777"}),
        ], style={"marginBottom": "8px"})

    return html.Div([
        html.H4(prefix, style={"marginBottom": "6px"}),
        row("baseline tau (s)",
            dcc.Slider(id=f"{prefix}-tau", min=2, max=180, step=1,
                       value=defaults["tau"],
                       marks={2: "2", 20: "20", 60: "60", 120: "120",
                              180: "180"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            "lambda is derived from this and the measured rate"),
        row("lambda override",
            dcc.Input(id=f"{prefix}-lam", type="text", value="",
                      placeholder="blank = from tau",
                      style={"width": "100%"}),
            "set a raw penalty to bypass the timescale conversion"),
        row("difference order",
            dcc.RadioItems(id=f"{prefix}-diff",
                           options=[{"label": " 1", "value": 1},
                                    {"label": " 2", "value": 2}],
                           value=defaults["diff"], inline=True),
            "1 drives F0 toward a constant and cannot follow bleaching"),
        row("baseline method",
            dcc.RadioItems(id=f"{prefix}-method",
                           options=[{"label": " airPLS", "value": "airpls"},
                                    {"label": " symmetric",
                                     "value": "symmetric"}],
                           value=defaults["method"], inline=True),
            "airPLS follows the lower envelope; symmetric the centre"),
        row("isosbestic regressors",
            dcc.RadioItems(id=f"{prefix}-pred",
                           options=[{"label": " none", "value": "none"},
                                    {"label": " same fiber", "value": "same"},
                                    {"label": " all fibers", "value": "all"},
                                    {"label": " own + common",
                                     "value": "common"}],
                           value=defaults["pred"]),
            "common averages the others' noise down; same is conservative"),
        row("common component",
            dcc.RadioItems(id=f"{prefix}-common",
                           options=[{"label": " median", "value": "median"},
                                    {"label": " PC1", "value": "pc1"}],
                           value=defaults["common"], inline=True)),
        row("Lasso alpha",
            dcc.Input(id=f"{prefix}-alpha", type="text",
                      value=defaults["alpha"], style={"width": "100%"}),
            "'cv' for blocked cross-validation, or a number"),
        row("blank laser frames",
            dcc.RadioItems(id=f"{prefix}-blank",
                           options=[{"label": " on", "value": 1},
                                    {"label": " off", "value": 0}],
                           value=defaults["blank"], inline=True)),
        row("exposure (ms)",
            dcc.Input(id=f"{prefix}-exp", type="number", value=8.5, step=0.1,
                      style={"width": "100%"}),
            "a physical property; the blanking test depends on it"),
        html.Hr(),
        row("peri-event baseline (s)",
            dcc.Slider(id=f"{prefix}-base", min=0.1, max=4, step=0.1,
                       value=defaults["base"],
                       marks={0.5: "0.5", 1: "1", 2: "2", 4: "4"},
                       tooltip={"placement": "bottom",
                                "always_visible": True}),
            "short windows are noisy: autocorrelation makes 20 samples "
            "worth about 1.5 independent ones"),
        row("response window (s)",
            dcc.RangeSlider(id=f"{prefix}-resp", min=-1, max=5, step=0.1,
                            value=defaults["resp"],
                            marks={0: "0", 0.5: ".5", 1: "1", 2: "2",
                                   4: "4"},
                            tooltip={"placement": "bottom",
                                     "always_visible": True})),
        row("per-trial baseline",
            dcc.RadioItems(id=f"{prefix}-detrend",
                           options=[{"label": " mean", "value": "mean"},
                                    {"label": " flanking line",
                                     "value": "linear"}],
                           value=defaults["detrend"], inline=True)),
        row("combine trials",
            dcc.RadioItems(id=f"{prefix}-stat",
                           options=[{"label": " mean", "value": "mean"},
                                    {"label": " median", "value": "median"}],
                           value=defaults["stat"], inline=True)),
        row("low-pass (Hz, 0 = off)",
            dcc.Input(id=f"{prefix}-lp", type="number", value=0, step=0.5,
                      style={"width": "100%"}),
            "GCaMP6f cannot follow above ~5 Hz"),
    ], style={"width": "300px", "padding": "10px",
              "borderRight": "1px solid #ddd"})


def main() -> int:
    import numpy as np
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, State, dcc, html

    from multifiber import variants as V

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir",
                    help="a processed session folder containing "
                         "<sid>_traces.parquet, <sid>_dff.parquet, "
                         "<sid>_events.parquet and <sid>_manifest.json")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--n-shift", type=int, default=300,
                    help="shifts per null; lower is faster, noisier p")
    args = ap.parse_args()

    from multifiber.events import parse_session_name

    sd = V.load_session(args.session_dir)
    print(f"{sd.session_id}: {sd.n_roi} fibers, {sd.traces.shape[0]} frames, "
          f"{len(sd.events)} events")
    sites = sorted(x for x in sd.events.get("region", []).unique()
                   if isinstance(x, str)) if len(sd.events) else []

    hints = parse_session_name(sd.session_id)
    classes = (sorted(int(c) for c in sd.events["protocol_class"].unique())
               if "protocol_class" in sd.events.columns else [])

    def class_label(c):
        g = sd.events[sd.events["protocol_class"] == c]
        lab = f"{g['width_s'].median() * 1e3:.0f} ms"
        if "command_v" in g.columns and g["command_v"].notna().any():
            lab += f" / {g['command_v'].median():.3f} V"
        return lab

    labels_by_class = {c: class_label(c) for c in classes}
    hint_txt = ""
    if hints:
        hint_txt = f"name says {hints}"
        if "n_protocols" in hints and classes:
            ok = hints["n_protocols"] == len(classes)
            hint_txt += (f" — data give {len(classes)} condition(s)"
                         + ("" if ok else "  ** MISMATCH **"))
    if classes:
        print(f"{len(classes)} condition(s): "
              + "; ".join(f"{c}={labels_by_class[c]}" for c in classes))
    if hint_txt:
        print(hint_txt)

    _cache: dict = {}

    def cached(spec):
        """Variants are recomputed on control changes, not on playback ticks."""
        k = spec.key()
        if k not in _cache:
            if len(_cache) > 8:
                _cache.clear()
            _cache[k] = V.run_variant(sd, spec)
        return _cache[k]

    DEF = dict(tau=60, diff=2, method="airpls", pred="common",
               common="median", alpha="cv", blank=1, base=0.5,
               resp=[0, 1.5], detrend="mean", stat="mean")
    ALT = dict(DEF, tau=20, pred="same", base=1.0)

    app = Dash(__name__)
    app.title = f"multifiber — {sd.session_id}"
    app.layout = html.Div([
        html.Div([
            html.H3(sd.session_id, style={"margin": "4px 0"}),
            html.Div(hint_txt, style={"fontSize": "10px", "color": "#555"}),
        ], style={"padding": "6px", "borderBottom": "1px solid #ddd"}),
        dcc.Tabs(id="tabs", value="compare", children=[

            # ---------------- preprocessing comparison ------------------- #
            dcc.Tab(label="preprocessing", value="compare", children=[
                html.Div([
                    html.Div("Variants are exploratory. Choose on a "
                             "diagnostic, not on the z-score: with twelve "
                             "fibers and a dozen knobs, selecting the largest "
                             "effect will always find one.",
                             style={"fontSize": "11px", "color": "#a33",
                                    "padding": "4px"}),
                    html.Div([
                        html.Label("fiber", style={"fontSize": "11px"}),
                        dcc.Dropdown(id="roi", value=0, clearable=False,
                                     options=[{"label": l, "value": i}
                                              for i, l in
                                              enumerate(sd.labels)],
                                     style={"width": "180px"}),
                        html.Label("stimulated site",
                                   style={"fontSize": "11px"}),
                        dcc.Dropdown(id="site",
                                     value=(sites[0] if sites else None),
                                     clearable=False,
                                     options=[{"label": x, "value": x}
                                              for x in sites],
                                     style={"width": "180px"}),
                        html.Label("conditions", style={"fontSize": "11px"}),
                        dcc.Dropdown(id="pmode", value="separate",
                                     clearable=False,
                                     options=([{"label": "each, overlaid",
                                                "value": "separate"},
                                               {"label": "POOLED (averages "
                                                         "conditions "
                                                         "together)",
                                                "value": "pooled"}]
                                              + [{"label": f"only {c}: "
                                                           f"{labels_by_class[c]}",
                                                  "value": f"only:{c}"}
                                                 for c in classes]),
                                     style={"width": "280px"}),
                    ], style={"display": "flex", "gap": "12px",
                              "alignItems": "center", "padding": "4px"}),
                    html.Div([
                        build_controls(None, "A", DEF),
                        build_controls(None, "B", ALT),
                        html.Div([
                            dcc.Loading(dcc.Graph(id="fig-trace",
                                                  style={"height": "280px"})),
                            dcc.Loading(dcc.Graph(id="fig-peth",
                                                  style={"height": "340px"})),
                            dcc.Loading(dcc.Graph(id="fig-dose",
                                                  style={"height": "230px"})),
                            html.Pre(id="stats",
                                     style={"fontSize": "11px",
                                            "background": "#f7f7f7",
                                            "padding": "8px"}),
                        ], style={"flex": 1, "padding": "6px"}),
                    ], style={"display": "flex"}),
                ]),
            ]),

            # ---------------- playback ----------------------------------- #
            dcc.Tab(label="playback", value="play", children=[
                html.Div([
                    html.Div("Raw (top) and processed (bottom) on a shared "
                             "clock, with the laser marked. The whole session "
                             "is drawn once and only the view window moves, "
                             "so panning is done in the browser and stays "
                             "smooth.",
                             style={"fontSize": "11px", "color": "#555",
                                    "padding": "4px"}),
                    html.Div([
                        html.Button("play", id="play-btn", n_clicks=0,
                                    style={"width": "70px"}),
                        html.Label("fibers", style={"fontSize": "11px"}),
                        dcc.Dropdown(id="pb-rois", multi=True,
                                     value=[0, 1, 2],
                                     options=[{"label": l, "value": i}
                                              for i, l in
                                              enumerate(sd.labels)],
                                     style={"width": "340px"}),
                        html.Label("window (s)", style={"fontSize": "11px"}),
                        dcc.Input(id="pb-win", type="number", value=20,
                                  min=2, step=1, style={"width": "70px"}),
                        html.Label("speed x", style={"fontSize": "11px"}),
                        dcc.Input(id="pb-speed", type="number", value=2.0,
                                  min=0.1, step=0.1, style={"width": "70px"}),
                        html.Label("settings", style={"fontSize": "11px"}),
                        dcc.RadioItems(id="pb-variant",
                                       options=[{"label": " A", "value": "A"},
                                                {"label": " B", "value": "B"},
                                                {"label": " own",
                                                 "value": "own"}],
                                       value="own", inline=True),
                        dcc.Checklist(id="pb-opts",
                                      options=[{"label": " isosbestic",
                                                "value": "iso"},
                                               {"label": " label every event",
                                                "value": "labels"}],
                                      value=["labels"], inline=True,
                                      style={"fontSize": "11px"}),
                        html.Span(id="pb-clock",
                                  style={"fontSize": "12px",
                                         "fontFamily": "monospace"}),
                    ], style={"display": "flex", "gap": "10px",
                              "alignItems": "center", "padding": "4px",
                              "flexWrap": "wrap"}),
                    html.Div([
                        html.Div([
                            html.Label("baseline tau (s)",
                                       style={"fontSize": "10px"}),
                            dcc.Slider(id="pb-tau", min=2, max=180, step=1,
                                       value=60,
                                       marks={2: "2", 20: "20", 60: "60",
                                              120: "120", 180: "180"},
                                       tooltip={"placement": "bottom",
                                                "always_visible": True}),
                        ], style={"width": "300px"}),
                        html.Div([
                            html.Label("low-pass (Hz, 0 = off)",
                                       style={"fontSize": "10px"}),
                            dcc.Slider(id="pb-lp", min=0, max=20, step=0.5,
                                       value=0,
                                       marks={0: "off", 5: "5", 10: "10",
                                              20: "20"},
                                       tooltip={"placement": "bottom",
                                                "always_visible": True}),
                        ], style={"width": "240px"}),
                        html.Div([
                            html.Label("isosbestic",
                                       style={"fontSize": "10px"}),
                            dcc.RadioItems(id="pb-pred",
                                           options=[{"label": " none",
                                                     "value": "none"},
                                                    {"label": " same",
                                                     "value": "same"},
                                                    {"label": " all",
                                                     "value": "all"},
                                                    {"label": " own+common",
                                                     "value": "common"}],
                                           value="common", inline=True,
                                           style={"fontSize": "10px"}),
                        ]),
                        html.Div([
                            html.Label("baseline", style={"fontSize": "10px"}),
                            dcc.RadioItems(id="pb-method",
                                           options=[{"label": " airPLS",
                                                     "value": "airpls"},
                                                    {"label": " symmetric",
                                                     "value": "symmetric"}],
                                           value="airpls", inline=True,
                                           style={"fontSize": "10px"}),
                        ]),
                        html.Div([
                            html.Label("blank laser frames",
                                       style={"fontSize": "10px"}),
                            dcc.RadioItems(id="pb-blank",
                                           options=[{"label": " on",
                                                     "value": 1},
                                                    {"label": " off",
                                                     "value": 0}],
                                           value=1, inline=True,
                                           style={"fontSize": "10px"}),
                        ]),
                    ], style={"display": "flex", "gap": "18px",
                              "alignItems": "flex-end", "padding": "4px 8px",
                              "flexWrap": "wrap",
                              "borderTop": "1px solid #eee"}),
                    dcc.Slider(id="pb-time", min=float(sd.frame_t[0]),
                               max=float(sd.frame_t[-1]),
                               value=float(sd.frame_t[0]), step=0.05,
                               marks=None, updatemode="drag",
                               tooltip={"placement": "bottom",
                                        "always_visible": True}),
                    # 40 ms ticks: the pan is a browser relayout, not a
                    # figure transfer, so this is cheap.
                    dcc.Interval(id="pb-tick", interval=40, disabled=True),
                    dcc.Loading(dcc.Graph(
                        id="fig-play", style={"height": "620px"},
                        config={"scrollZoom": True,
                                "displayModeBar": True})),
                    html.Div(id="pb-sink", style={"display": "none"}),
                ]),
            ]),
        ]),
    ])

    def spec_from(vals, label):
        (tau, lam, diff, method, pred, common, alpha, blank, exp,
         base, resp, detrend, stat, lp) = vals
        try:
            lam_v = float(lam) if lam not in (None, "") else None
        except ValueError:
            lam_v = None
        try:
            alpha_v = "cv" if str(alpha).strip() == "cv" else float(alpha)
        except ValueError:
            alpha_v = "cv"
        return V.VariantSpec(
            baseline_tau_s=float(tau), lambda_override=lam_v,
            differences=int(diff), baseline_method=method,
            regress=(pred != "none"),
            predictors=("same" if pred == "none" else pred),
            common_method=common, alpha=alpha_v,
            blank_laser=bool(blank),
            exposure_s=(float(exp) / 1e3 if exp else None),
            baseline_window_s=float(base),
            response_start_s=float(resp[0]),
            response_window_s=float(resp[1] - resp[0]),
            detrend=detrend, statistic=stat,
            lowpass_hz=(float(lp) if lp else None), label=label)

    ids = lambda p: [Input(f"{p}-{k}", "value") for k in
                     ("tau", "lam", "diff", "method", "pred", "common",
                      "alpha", "blank", "exp", "base", "resp", "detrend",
                      "stat", "lp")]

    @app.callback(
        Output("fig-trace", "figure"), Output("fig-peth", "figure"),
        Output("fig-dose", "figure"), Output("stats", "children"),
        Input("roi", "value"), Input("site", "value"), Input("pmode", "value"),
        *ids("A"), *ids("B"))
    def update(roi, site, pmode, *vals):
        a = spec_from(vals[:14], "A")
        b = spec_from(vals[14:], "B")
        roi = int(roi)

        base = sd.events
        if len(base) and site is not None and "region" in base.columns:
            base = base[base["region"] == site]

        # Which conditions to analyse. Separate is the default: an
        # interleaved session varies the condition deliberately, and a single
        # average across them describes none of them.
        conds = []
        if len(base) and "protocol_class" in base.columns and classes:
            if pmode == "separate":
                for c in sorted(base["protocol_class"].unique()):
                    g = base[base["protocol_class"] == c]
                    conds.append((labels_by_class.get(int(c), str(c)), g,
                                  int(c)))
            elif pmode.startswith("only:"):
                c = int(pmode.split(":")[1])
                g = base[base["protocol_class"] == c]
                conds.append((labels_by_class.get(c, str(c)), g, c))
            else:
                conds.append(("POOLED", base, None))
        else:
            conds.append(("all events", base, None))

        DASH = ["solid", "dash", "dot", "dashdot", "longdash"]
        results = {}
        for spec, colour in ((a, "#0173B2"), (b, "#D55E00")):
            results[spec.label] = (cached(spec), colour, spec)

        txt, peth_items, dose = [], [], {}
        for lbl, (res, colour, spec) in results.items():
            lam = res.qc.get("lambda", float("nan"))
            r2 = res.qc.get("regression_r2") or []
            txt.append(
                f"{lbl}  {spec.describe()}\n"
                f"   lambda {lam:.3e} | dF/F SD {np.nanstd(res.dff[:, roi]):.5f}"
                f" | blanked {int(res.blanked.sum())}"
                f" | iso R^2 "
                f"{(r2[roi] if len(r2) > roi else float('nan')):.3f}")
            dose[lbl] = []
            for k, (cname, g, cid) in enumerate(conds):
                onsets = g["onset_t"].to_numpy(dtype=float)
                width = float(g["width_s"].median()) if len(g) else 1.0
                if onsets.size == 0:
                    continue
                c = V.compare(res, onsets, roi, n_shift=args.n_shift,
                              width_s=width)
                if not c.get("n"):
                    txt.append(f"   {cname}: no complete windows")
                    continue
                peth_items.append((lbl, cname, colour, DASH[k % len(DASH)],
                                   c, width))
                lvl = (float(g["width_s"].median()) * 1e3 if cid is not None
                       else np.nan)
                dose[lbl].append((cname, lvl, c["effect"], c.get("z"),
                                  c.get("p"),
                                  c["null"]["sd"] if "null" in c else np.nan))
                txt.append(f"   {cname:>18}: n={c['n']:3d} "
                           f"effect {c['effect']:+.5f} "
                           f"z {c.get('z', float('nan')):+.2f} "
                           f"p {c.get('p', float('nan')):.4f}")
            txt.append("")

        if pmode == "pooled" and len(classes) > 1:
            txt.insert(0, "** POOLED: conditions averaged together. This "
                          "session varies the condition on purpose; a single "
                          "average describes none of them. **\n")

        # ---- full trace -------------------------------------------------- #
        f1 = go.Figure()
        for lbl, (res, colour, spec) in results.items():
            f1.add_trace(go.Scatter(x=res.t, y=res.dff[:, roi], name=lbl,
                                    line=dict(width=1, color=colour)))
        for k, (cname, g, cid) in enumerate(conds):
            for o, w in zip(g["onset_t"].to_numpy(dtype=float)[:300],
                            g["width_s"].to_numpy(dtype=float)[:300]):
                f1.add_vrect(x0=o, x1=o + w, line_width=0, opacity=0.16,
                             fillcolor=["#555", "#2a9d8f", "#e9c46a",
                                        "#9b5de5"][k % 4])
        f1.update_layout(margin=dict(l=40, r=10, t=24, b=30),
                         title=f"{sd.labels[roi]} — full trace "
                               f"(shading: conditions)",
                         yaxis_title="dF/F", xaxis_title="time (s)",
                         legend=dict(orientation="h"))

        # ---- peri-event, one trace per variant x condition --------------- #
        f2 = go.Figure()
        for lbl, cname, colour, dashst, c, width in peth_items:
            if "null" in c:
                lo = np.nanpercentile(c["null"]["traces"], 2.5, axis=0)
                hi = np.nanpercentile(c["null"]["traces"], 97.5, axis=0)
                f2.add_trace(go.Scatter(
                    x=np.r_[c["t"], c["t"][::-1]], y=np.r_[hi, lo[::-1]],
                    fill="toself", fillcolor=colour, opacity=0.08,
                    line=dict(width=0), showlegend=False, hoverinfo="skip"))
            f2.add_trace(go.Scatter(
                x=c["t"], y=c["mean"], name=f"{lbl} · {cname}",
                line=dict(width=2, color=colour, dash=dashst)))
        if peth_items:
            f2.add_vrect(x0=0, x1=max(w for *_, w in peth_items),
                         fillcolor="#0173B2", opacity=0.08, line_width=0)
        f2.add_hline(y=0, line=dict(width=1, dash="dot", color="#000"))
        f2.update_layout(margin=dict(l=40, r=10, t=24, b=30),
                         title=f"peri-event — stimulating {site} "
                               f"(bands: each variant's own null)",
                         yaxis_title="dF/F",
                         xaxis_title="time from onset (s)",
                         legend=dict(orientation="h"))

        # ---- effect across conditions ------------------------------------ #
        f3 = go.Figure()
        for lbl, rows in dose.items():
            if len(rows) < 1:
                continue
            colour = results[lbl][1]
            rows = sorted(rows, key=lambda r: (r[1] if np.isfinite(r[1])
                                               else 0))
            x = [r[0] for r in rows]
            y = [r[2] for r in rows]
            e = [3 * r[5] if np.isfinite(r[5]) else 0 for r in rows]
            f3.add_trace(go.Scatter(x=x, y=y, name=lbl, mode="lines+markers",
                                    error_y=dict(array=e, visible=True),
                                    line=dict(color=colour, width=2)))
        f3.add_hline(y=0, line=dict(width=1, dash="dot", color="#000"))
        f3.update_layout(margin=dict(l=40, r=10, t=28, b=40),
                         title="effect by condition (bars: 3x that "
                               "condition's null SD — the detection floor)",
                         yaxis_title="dF/F", xaxis_title="condition",
                         legend=dict(orientation="h"))
        return f1, f2, f3, "\n".join(txt)

    # ---------------- playback -------------------------------------------- #
    # The whole session is drawn once; playback moves only the x-range, and
    # that move happens in the browser. Re-sending a figure every tick is
    # what made the earlier version stutter: each frame was a server
    # round-trip carrying all the data again.

    @app.callback(Output("pb-tick", "disabled"), Output("play-btn", "children"),
                  Input("play-btn", "n_clicks"))
    def toggle(n):
        playing = bool((n or 0) % 2)
        return (not playing), ("pause" if playing else "play")

    @app.callback(Output("fig-play", "figure"),
                  Input("pb-rois", "value"), Input("pb-variant", "value"),
                  Input("pb-opts", "value"), Input("pb-win", "value"),
                  Input("pb-tau", "value"), Input("pb-lp", "value"),
                  Input("pb-pred", "value"), Input("pb-method", "value"),
                  Input("pb-blank", "value"),
                  *ids("A"), *ids("B"))
    def draw_session(rois, which, opts, win, pb_tau, pb_lp, pb_pred,
                     pb_method, pb_blank, *vals):
        from plotly.subplots import make_subplots

        if which == "A":
            spec = spec_from(vals[:14], "A")
        elif which == "B":
            spec = spec_from(vals[14:], "B")
        else:
            spec = V.VariantSpec(
                baseline_tau_s=float(pb_tau or 60),
                baseline_method=pb_method,
                regress=(pb_pred != "none"),
                predictors=("same" if pb_pred == "none" else pb_pred),
                blank_laser=bool(pb_blank),
                lowpass_hz=(float(pb_lp) if pb_lp else None),
                label="playback")
        res = cached(spec)
        rois = [int(r) for r in (rois or [0])]
        win = float(win or 20)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.07,
                            subplot_titles=("raw ROI means (camera rate)",
                                            f"processed dF/F — "
                                            f"{spec.describe()}"))
        pal = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161",
               "#949494", "#ECE133", "#56B4E9", "#D55E00", "#F0E442",
               "#009E73", "#000000"]
        sig_state = int(np.bincount(sd.led_state).argmax())
        sig = sd.led_state == sig_state

        # Scattergl: WebGL rendering, so tens of thousands of points pan
        # without the canvas redraw becoming the bottleneck.
        for j in rois:
            colour = pal[j % len(pal)]
            fig.add_trace(go.Scattergl(
                x=sd.frame_t[sig], y=sd.traces[sig, j], name=sd.labels[j],
                legendgroup=str(j), line=dict(width=1, color=colour)),
                row=1, col=1)
            if "iso" in (opts or []):
                fig.add_trace(go.Scattergl(
                    x=sd.frame_t[~sig], y=sd.traces[~sig, j],
                    name=f"{sd.labels[j]} 415", legendgroup=str(j),
                    showlegend=False,
                    line=dict(width=0.7, color=colour, dash="dot")),
                    row=1, col=1)
            fig.add_trace(go.Scattergl(
                x=res.t, y=res.dff[:, j], name=sd.labels[j],
                legendgroup=str(j), showlegend=False,
                line=dict(width=1.2, color=colour)), row=2, col=1)

        if res.blanked.any():
            for j in rois:
                b = res.blanked
                fig.add_trace(go.Scattergl(
                    x=res.t[b], y=res.dff[b, j], mode="markers",
                    marker=dict(size=4, color="#D55E00", symbol="x"),
                    name="blanked", legendgroup="blank",
                    showlegend=(j == rois[0]), hoverinfo="skip"),
                    row=2, col=1)

        # Laser commands as shapes, panned with the axis, each labelled with
        # what it was and where it went. Without that, an interleaved session
        # is a row of identical orange bars.
        shapes, annos, htxt, hx = [], [], [], []
        if len(sd.events):
            cls_col = ("protocol_class" if "protocol_class" in sd.events.columns
                       else None)
            label_all = "labels" in (opts or [])
            for _, r in sd.events.iterrows():
                a, b = float(r["onset_t"]), float(r["offset_t"])
                for ref, dom in (("y", "x"), ("y2", "x2")):
                    shapes.append(dict(type="rect", xref=dom,
                                       yref=f"{ref} domain", x0=a, x1=b,
                                       y0=0, y1=1, line_width=0,
                                       fillcolor="#D55E00", opacity=0.18,
                                       layer="below"))
                site = r.get("region") or f"roi {int(r.get('roi_id', -1))}"
                dur = (b - a) * 1e3
                volt = r.get("command_v")
                cid = int(r[cls_col]) if cls_col is not None else None
                txt = f"{site} · {dur:.0f} ms"
                if volt is not None and np.isfinite(volt):
                    txt += f" · {volt:.3f} V"
                if cid is not None:
                    txt += f" · class {cid}"
                hx.append(0.5 * (a + b))
                htxt.append(txt)
                if label_all:
                    annos.append(dict(
                        x=0.5 * (a + b), y=1.02, xref="x", yref="y domain",
                        text=f"{site}<br>{dur:.0f} ms", showarrow=False,
                        font=dict(size=7, color="#9c4a00"),
                        align="center", yanchor="bottom"))
            if hx:
                # Hover works at any zoom, including where labels are too
                # dense to draw.
                fig.add_trace(go.Scattergl(
                    x=hx, y=np.full(len(hx), np.nan), mode="markers",
                    marker=dict(size=1, opacity=0), hovertext=htxt,
                    hoverinfo="text", name="stimulation", showlegend=False),
                    row=1, col=1)
        t0 = float(sd.frame_t[0])
        if len(annos) > 300:
            # Too many to read; hover still carries every one.
            annos = []
        fig.update_layout(
            shapes=shapes, annotations=list(fig.layout.annotations) + annos,
            margin=dict(l=55, r=10, t=48, b=36),
            uirevision="static", dragmode="pan",
            legend=dict(orientation="h", y=1.14))
        fig.update_xaxes(range=[t0, t0 + win], row=1, col=1)
        fig.update_xaxes(range=[t0, t0 + win], title_text="time (s)",
                         row=2, col=1)
        fig.update_yaxes(title_text="counts", row=1, col=1)
        fig.update_yaxes(title_text="dF/F", row=2, col=1)
        return fig

    # Advance the clock in the browser, and pan by relayout. No server round
    # trip, no data transferred: this is what keeps it smooth.
    app.clientside_callback(
        """
        function(nTick, tSlider, win, speed, tMin, tMax) {
            const trig = dash_clientside.callback_context.triggered;
            const byTick = trig.length && trig[0].prop_id.indexOf("pb-tick") === 0;
            let t = (tSlider === null || tSlider === undefined) ? tMin : tSlider;
            if (byTick) {
                t = t + (speed || 1) * 0.04;     // interval is 40 ms
                if (t > tMax) { t = tMin; }
            }
            const w = win || 20;
            const lo = t - 0.75 * w, hi = t + 0.25 * w;
            const gd = document.querySelector("#fig-play .js-plotly-plot");
            if (gd && window.Plotly) {
                Plotly.relayout(gd, {"xaxis.range": [lo, hi],
                                     "xaxis2.range": [lo, hi]});
            }
            const clock = t.toFixed(2) + " s  /  " + tMax.toFixed(0) + " s";
            return [byTick ? t : dash_clientside.no_update, clock];
        }
        """,
        [Output("pb-time", "value"), Output("pb-clock", "children")],
        [Input("pb-tick", "n_intervals"), Input("pb-time", "value"),
         Input("pb-win", "value"), Input("pb-speed", "value")],
        [State("pb-time", "min"), State("pb-time", "max")],
    )

    print(f"open http://127.0.0.1:{args.port}")
    app.run(debug=False, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
