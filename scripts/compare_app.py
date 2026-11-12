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
                                    {"label": " others only",
                                     "value": "others"},
                                    {"label": " own + common",
                                     "value": "common"}],
                           value=defaults["pred"]),
            "common averages the others' noise down; same is conservative"),
        row("common component",
            dcc.RadioItems(id=f"{prefix}-common",
                           options=([{"label": " median",
                                      "value": "median"}]
                                    + [{"label": f" PC1-{k}" if k > 1
                                        else " PC1", "value": f"pc{k}"}
                                       for k in range(1, 8)]),
                           value=defaults["common"]),
            "more components model several independent artifact sources; "
            "judge them on the detection floor, not on R^2"),
        row("Lasso alpha",
            dcc.Input(id=f"{prefix}-alpha", type="text",
                      value=defaults["alpha"], style={"width": "100%"}),
            "'cv' for blocked cross-validation, or a number"),
        row("regression estimator",
            dcc.RadioItems(id=f"{prefix}-rmethod",
                           options=[{"label": " lasso", "value": "lasso"},
                                    {"label": " OLS", "value": "ols"},
                                    {"label": " ridge", "value": "ridge"},
                                    {"label": " hierarchical",
                                     "value": "hierarchical"},
                                    {"label": " auto (per fiber)",
                                     "value": "auto"}],
                           value=defaults.get("rmethod", "lasso")),
            "hierarchical fits the own fiber unpenalized first, so a "
            "penalty chosen for the others cannot shrink it away"),
        row("hierarchical: step 1",
            html.Div([
                dcc.RadioItems(id=f"{prefix}-head",
                               options=[{"label": " OLS", "value": "ols"},
                                        {"label": " lasso",
                                         "value": "lasso"},
                                        {"label": " ridge",
                                         "value": "ridge"}],
                               value=defaults.get("head", "ols"), inline=True),
                dcc.Input(id=f"{prefix}-halpha", type="text",
                          value=defaults.get("halpha", "cv"),
                          placeholder="step-1 alpha",
                          style={"width": "70px"}),
                dcc.Input(id=f"{prefix}-unpen", type="number",
                          value=defaults.get("unpen", 1), min=1, max=8,
                          step=1, style={"width": "55px"}),
            ], style={"display": "flex", "gap": "6px",
                      "alignItems": "center"}),
            "estimator, penalty and number of leading columns for the first "
            "stage; step 2 uses the estimator and alpha above. Only applies "
            "when the estimator is hierarchical."),
        row("stimulus-locked artifact",
            dcc.RadioItems(id=f"{prefix}-stimart",
                           options=[{"label": " off", "value": 0},
                                    {"label": " on", "value": 1}],
                           value=defaults.get("stimart", 0), inline=True),
            "fits a square term (an optical artifact has no kinetics) and "
            "keeps a kinetic term as the response; the isosbestic alone "
            "cannot do this, because stray light is additive and motion is "
            "multiplicative"),
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
    from dash import Dash, Input, Output, State, dcc, html, no_update as dash_no_update

    from multifiber import variants as V

    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="?", default=None,
                    help="a processed session folder containing "
                         "<sid>_traces.parquet, <sid>_dff.parquet, "
                         "<sid>_events.parquet and <sid>_manifest.json. "
                         "Omit it and pick one in the browser instead.")
    ap.add_argument("--root", default=None,
                    help="where to look for sessions when none is given; "
                         "defaults to the current directory")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--n-shift", type=int, default=300,
                    help="shifts per null; lower is faster, noisier p")
    args = ap.parse_args()

    from multifiber.events import parse_session_name

    # The session is loaded on demand: the app opens whether or not one was
    # named on the command line, and a different one can be chosen without
    # restarting it.
    ST: dict = {"sd": None, "sites": [], "classes": [], "labels": {},
                "hint": "", "cache": {}}

    def find_sessions(root):
        """Folders under ``root`` that look like processed sessions."""
        root = Path(root).expanduser()
        if not root.is_dir():
            return []
        out = []
        if list(root.glob(f"{root.name}_manifest.json")):
            out.append(root)                       # root IS a session
        out += sorted({m.parent for m in root.glob("*/*_manifest.json")})
        out += sorted({m.parent for m in root.glob("*/*/*_manifest.json")})
        seen, uniq = set(), []
        for d in out:
            if str(d) not in seen:
                seen.add(str(d))
                uniq.append(d)
        return uniq

    def load(path) -> str:
        sd = V.load_session(path)
        sites = sorted(x for x in sd.events.get("region", []).unique()
                       if isinstance(x, str)) if len(sd.events) else []
        classes = (sorted(int(c) for c in sd.events["protocol_class"].unique())
                   if "protocol_class" in sd.events.columns else [])

        def class_label(c):
            g = sd.events[sd.events["protocol_class"] == c]
            lab = f"{g['width_s'].median() * 1e3:.0f} ms"
            if "command_v" in g.columns and g["command_v"].notna().any():
                lab += f" / {g['command_v'].median():.3f} V"
            return lab

        labels_by_class = {c: class_label(c) for c in classes}
        hints = parse_session_name(sd.session_id)
        hint = ""
        if hints:
            hint = f"name says {hints}"
            if "n_protocols" in hints and classes:
                ok = hints["n_protocols"] == len(classes)
                hint += (f" — data give {len(classes)} condition(s)"
                         + ("" if ok else "  ** MISMATCH **"))
        ST.update(sd=sd, sites=sites, classes=classes,
                  labels=labels_by_class, hint=hint, cache={})
        return (f"{sd.session_id}: {sd.n_roi} fibers, "
                f"{sd.traces.shape[0]} frames, {len(sd.events)} events"
                + (f" | {len(classes)} condition(s)" if classes else ""))

    if args.session_dir:
        print(load(args.session_dir))

    def cached(spec):
        """Variants are recomputed on control changes, not on playback ticks."""
        c = ST["cache"]
        k = spec.key()
        if k not in c:
            if len(c) > 8:
                c.clear()
            c[k] = V.run_variant(ST["sd"], spec)
        return c[k]

    DEF = dict(tau=60, diff=2, method="airpls", pred="common",
               common="median", alpha="cv", blank=1, base=0.5,
               resp=[0, 1.5], detrend="mean", stat="mean",
               rmethod="lasso", stimart=0, head="ols", halpha="cv", unpen=1)
    ALT = dict(DEF, tau=20, pred="same", base=1.0,
               rmethod="hierarchical", stimart=0, head="lasso", halpha="cv")

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "multifiber"

    def session_ui():
        sd = ST["sd"]
        sites, classes = ST["sites"], ST["classes"]
        labels_by_class, hint_txt = ST["labels"], ST["hint"]
        return html.Div([
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

            # ---------------- all fibers at once ------------------------- #
            dcc.Tab(label="all PETHs", value="grid", children=[
                html.Div([
                    html.Div("Every recorded fiber under one set of "
                             "preprocessing choices. Panels share the time "
                             "axis and keep their own y-scale, because the "
                             "fibers differ by tens of times in noise and a "
                             "shared y-axis would flatten the quiet ones.",
                             style={"fontSize": "11px", "color": "#555",
                                    "padding": "4px"}),
                    html.Div([
                        html.Label("stimulated site",
                                   style={"fontSize": "11px"}),
                        dcc.Dropdown(id="g-site",
                                     value=(sites[0] if sites else None),
                                     clearable=False,
                                     options=[{"label": x, "value": x}
                                              for x in sites],
                                     style={"width": "170px"}),
                        html.Label("conditions", style={"fontSize": "11px"}),
                        dcc.Dropdown(id="g-cond", value="separate",
                                     clearable=False,
                                     options=([{"label": "each, overlaid",
                                                "value": "separate"},
                                               {"label": "POOLED",
                                                "value": "pooled"}]
                                              + [{"label": f"only "
                                                           f"{labels_by_class[c]}",
                                                  "value": f"only:{c}"}
                                                 for c in classes]),
                                     style={"width": "220px"}),
                        html.Label("settings", style={"fontSize": "11px"}),
                        dcc.RadioItems(id="g-variant",
                                       options=[{"label": " A", "value": "A"},
                                                {"label": " B", "value": "B"},
                                                {"label": " own",
                                                 "value": "own"}],
                                       value="own", inline=True),
                        html.Label("combine", style={"fontSize": "11px"}),
                        dcc.RadioItems(id="g-stat",
                                       options=[{"label": " mean",
                                                 "value": "mean"},
                                                {"label": " median",
                                                 "value": "median"}],
                                       value="mean", inline=True),
                        html.Label("cols", style={"fontSize": "11px"}),
                        dcc.Input(id="g-cols", type="number", value=3, min=1,
                                  max=6, step=1, style={"width": "60px"}),
                        html.Label("panel w x h (px)",
                                   style={"fontSize": "11px"}),
                        dcc.Input(id="g-pw", type="number", value=320,
                                  min=120, max=900, step=10,
                                  style={"width": "75px"}),
                        dcc.Input(id="g-ph", type="number", value=220,
                                  min=100, max=700, step=10,
                                  style={"width": "75px"}),
                        dcc.Checklist(id="g-opts",
                                      options=[{"label": " null bands",
                                                "value": "null"},
                                               {"label": " single trials",
                                                "value": "trials"}],
                                      value=["null"], inline=True,
                                      style={"fontSize": "11px"}),
                        html.Label("shifts", style={"fontSize": "11px"}),
                        dcc.Input(id="g-nshift", type="number", value=200,
                                  min=20, max=2000, step=20,
                                  style={"width": "80px"}),
                    ], style={"display": "flex", "gap": "10px",
                              "alignItems": "center", "padding": "4px",
                              "flexWrap": "wrap"}),
                    html.Div([
                        html.Div([
                            html.Label("time axis (s from onset)",
                                       style={"fontSize": "10px"}),
                            dcc.RangeSlider(id="g-xrange", min=-10, max=20,
                                            step=0.5, value=[-2, 4],
                                            marks={-10: "-10", -5: "-5",
                                                   0: "0", 5: "5", 10: "10",
                                                   20: "20"},
                                            tooltip={"placement": "bottom",
                                                     "always_visible": True}),
                        ], style={"width": "420px"}),
                        html.Div([
                            html.Label("response window (s)",
                                       style={"fontSize": "10px"}),
                            dcc.RangeSlider(id="g-resp", min=-1, max=8,
                                            step=0.1, value=[0, 1.5],
                                            marks={0: "0", 1: "1", 2: "2",
                                                   4: "4", 8: "8"},
                                            tooltip={"placement": "bottom",
                                                     "always_visible": True}),
                        ], style={"width": "320px"}),
                        html.Div([
                            html.Label("peri-event baseline (s)",
                                       style={"fontSize": "10px"}),
                            dcc.Slider(id="g-base", min=0.1, max=4, step=0.1,
                                       value=0.5,
                                       marks={0.5: "0.5", 1: "1", 2: "2",
                                              4: "4"},
                                       tooltip={"placement": "bottom",
                                                "always_visible": True}),
                        ], style={"width": "240px"}),
                        html.Div([
                            html.Label("baseline tau (s)",
                                       style={"fontSize": "10px"}),
                            dcc.Slider(id="g-tau", min=2, max=180, step=1,
                                       value=60,
                                       marks={2: "2", 20: "20", 60: "60",
                                              120: "120", 180: "180"},
                                       tooltip={"placement": "bottom",
                                                "always_visible": True}),
                        ], style={"width": "260px"}),
                        html.Div([
                            html.Label("low-pass (Hz, 0 = off)",
                                       style={"fontSize": "10px"}),
                            dcc.Slider(id="g-lp", min=0, max=20, step=0.5,
                                       value=0,
                                       marks={0: "off", 5: "5", 10: "10",
                                              20: "20"},
                                       tooltip={"placement": "bottom",
                                                "always_visible": True}),
                        ], style={"width": "220px"}),
                        html.Div([
                            html.Label("isosbestic",
                                       style={"fontSize": "10px"}),
                            dcc.RadioItems(id="g-pred",
                                           options=[{"label": " none",
                                                     "value": "none"},
                                                    {"label": " same fiber",
                                                     "value": "same"},
                                                    {"label": " all fibers",
                                                     "value": "all"},
                                                    {"label": " others only",
                                                     "value": "others"},
                                                    {"label": " own+common",
                                                     "value": "common"}],
                                           value="common",
                                           style={"fontSize": "10px"}),
                        ]),
                        html.Div([
                            html.Label("common component",
                                       style={"fontSize": "10px"}),
                            dcc.Dropdown(id="g-common", value="median",
                                         clearable=False,
                                         options=([
                                             {"label": "median of others",
                                              "value": "median"},
                                             {"label": "PC1",
                                              "value": "pc1"}]
                                             + [{"label": f"PC1-{k}",
                                                 "value": f"pc{k}"}
                                                for k in range(2, 8)]),
                                         style={"width": "165px",
                                                "fontSize": "10px"}),
                            html.Div("applies to own+common",
                                     style={"fontSize": "9px",
                                            "color": "#777"}),
                        ]),
                        html.Div([
                            html.Label("baseline", style={"fontSize": "10px"}),
                            dcc.RadioItems(id="g-method",
                                           options=[{"label": " airPLS",
                                                     "value": "airpls"},
                                                    {"label": " symmetric",
                                                     "value": "symmetric"}],
                                           value="airpls", inline=True,
                                           style={"fontSize": "10px"}),
                        ]),
                        html.Div([
                            html.Label("estimator / stim artifact",
                                       style={"fontSize": "10px"}),
                            dcc.RadioItems(id="g-rmethod",
                                           options=[{"label": " lasso",
                                                     "value": "lasso"},
                                                    {"label": " OLS",
                                                     "value": "ols"},
                                                    {"label": " hier",
                                                     "value": "hierarchical"},
                                                    {"label": " auto",
                                                     "value": "auto"}],
                                           value="lasso", inline=True,
                                           style={"fontSize": "10px"}),
                            dcc.RadioItems(id="g-stimart",
                                           options=[{"label": " artifact off",
                                                     "value": 0},
                                                    {"label": " artifact on",
                                                     "value": 1}],
                                           value=0, inline=True,
                                           style={"fontSize": "10px"}),
                        ]),
                        html.Div([
                            html.Label("hier step 1 (est / alpha)",
                                       style={"fontSize": "10px"}),
                            dcc.RadioItems(id="g-head",
                                           options=[{"label": " OLS",
                                                     "value": "ols"},
                                                    {"label": " lasso",
                                                     "value": "lasso"},
                                                    {"label": " ridge",
                                                     "value": "ridge"}],
                                           value="ols", inline=True,
                                           style={"fontSize": "10px"}),
                            dcc.Input(id="g-halpha", type="text", value="cv",
                                      style={"width": "70px"}),
                        ]),
                        html.Div([
                            html.Label("Lasso alpha",
                                       style={"fontSize": "10px"}),
                            dcc.Input(id="g-alpha", type="text", value="cv",
                                      style={"width": "80px"}),
                            html.Div("'cv' tunes on overall drift and can "
                                     "discard a brief artifact; use 1e-6 to "
                                     "remove one",
                                     style={"fontSize": "9px",
                                            "color": "#777",
                                            "maxWidth": "150px"}),
                        ]),
                        html.Div([
                            html.Label("blank", style={"fontSize": "10px"}),
                            dcc.RadioItems(id="g-blank",
                                           options=[{"label": " on",
                                                     "value": 1},
                                                    {"label": " off",
                                                     "value": 0}],
                                           value=1, inline=True,
                                           style={"fontSize": "10px"}),
                        ]),
                        html.Div([
                            html.Label("palette", style={"fontSize": "10px"}),
                            dcc.Dropdown(id="g-pal", value="okabe",
                                         clearable=False,
                                         options=[
                                             {"label": "Okabe-Ito "
                                                       "(colourblind-safe)",
                                              "value": "okabe"},
                                             {"label": "blue to red",
                                              "value": "bwr"},
                                             {"label": "viridis",
                                              "value": "viridis"},
                                             {"label": "greys",
                                              "value": "grey"}],
                                         style={"width": "210px",
                                                "fontSize": "10px"}),
                        ]),
                        html.Div([
                            html.Label("colour by",
                                       style={"fontSize": "10px"}),
                            dcc.RadioItems(id="g-colorby",
                                           options=[{"label": " condition",
                                                     "value": "cond"},
                                                    {"label": " fiber",
                                                     "value": "fiber"}],
                                           value="cond", inline=True,
                                           style={"fontSize": "10px"}),
                        ]),
                        html.Div([
                            html.Label("line style",
                                       style={"fontSize": "10px"}),
                            dcc.RadioItems(id="g-dash",
                                           options=[{"label": " vary by "
                                                              "condition",
                                                     "value": "vary"},
                                                    {"label": " all solid",
                                                     "value": "solid"}],
                                           value="vary", inline=True,
                                           style={"fontSize": "10px"}),
                        ]),
                        html.Div([
                            html.Label("line width",
                                       style={"fontSize": "10px"}),
                            dcc.Slider(id="g-lw", min=0.5, max=4, step=0.25,
                                       value=1.8,
                                       marks={1: "1", 2: "2", 4: "4"},
                                       tooltip={"placement": "bottom",
                                                "always_visible": True}),
                        ], style={"width": "180px"}),
                    ], style={"display": "flex", "gap": "18px",
                              "alignItems": "flex-end", "padding": "4px 8px",
                              "flexWrap": "wrap",
                              "borderTop": "1px solid #eee"}),
                    dcc.Loading(dcc.Graph(
                        id="fig-grid",
                        style={"overflowX": "auto"},
                        config={"toImageButtonOptions": {
                            "format": "svg", "filename": "peth_grid"}})),
                    html.Pre(id="grid-stats",
                             style={"fontSize": "11px",
                                    "background": "#f7f7f7",
                                    "padding": "8px"}),
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
                        html.Label("rows", style={"fontSize": "11px"}),
                        dcc.Checklist(id="pb-rows-sel",
                                      options=[{"label": " 470 raw",
                                                "value": "470"},
                                               {"label": " 415 raw",
                                                "value": "415"},
                                               {"label": " processed",
                                                "value": "dff"}],
                                      value=["470", "415", "dff"],
                                      inline=True,
                                      style={"fontSize": "11px"}),
                        html.Label("470 scale", style={"fontSize": "11px"}),
                        dcc.RadioItems(id="pb-scale",
                                       options=[
                                           {"label": " % of median",
                                            "value": "frac"},
                                           {"label": " stacked",
                                            "value": "stack"},
                                           {"label": " counts",
                                            "value": "counts"}],
                                       value="frac", inline=True,
                                       style={"fontSize": "11px"}),
                        html.Label("415 scale", style={"fontSize": "11px"}),
                        dcc.RadioItems(id="pb-scale415",
                                       options=[
                                           {"label": " % of median",
                                            "value": "frac"},
                                           {"label": " stacked",
                                            "value": "stack"},
                                           {"label": " counts",
                                            "value": "counts"}],
                                       value="frac", inline=True,
                                       style={"fontSize": "11px"}),
                        dcc.Checklist(id="pb-opts",
                                      options=[{"label": " label every event",
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
                                                    {"label": " others",
                                                     "value": "others"},
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
                            html.Label("estimator / stim artifact",
                                       style={"fontSize": "10px"}),
                            dcc.RadioItems(id="pb-rmethod",
                                           options=[{"label": " lasso",
                                                     "value": "lasso"},
                                                    {"label": " OLS",
                                                     "value": "ols"},
                                                    {"label": " hier",
                                                     "value": "hierarchical"},
                                                    {"label": " auto",
                                                     "value": "auto"}],
                                           value="lasso", inline=True,
                                           style={"fontSize": "10px"}),
                            dcc.RadioItems(id="pb-stimart",
                                           options=[{"label": " artifact off",
                                                     "value": 0},
                                                    {"label": " artifact on",
                                                     "value": 1}],
                                           value=0, inline=True,
                                           style={"fontSize": "10px"}),
                        ]),
                        html.Div([
                            html.Label("hier step 1 (est / alpha)",
                                       style={"fontSize": "10px"}),
                            dcc.RadioItems(id="pb-head",
                                           options=[{"label": " OLS",
                                                     "value": "ols"},
                                                    {"label": " lasso",
                                                     "value": "lasso"},
                                                    {"label": " ridge",
                                                     "value": "ridge"}],
                                           value="ols", inline=True,
                                           style={"fontSize": "10px"}),
                            dcc.Input(id="pb-halpha", type="text", value="cv",
                                      style={"width": "70px"}),
                        ]),
                        html.Div([
                            html.Label("Lasso alpha",
                                       style={"fontSize": "10px"}),
                            dcc.Input(id="pb-alpha", type="text", value="cv",
                                      style={"width": "80px"}),
                            html.Div("'cv' tunes on overall drift and can "
                                     "discard a brief artifact; use 1e-6 to "
                                     "remove one",
                                     style={"fontSize": "9px",
                                            "color": "#777",
                                            "maxWidth": "150px"}),
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
                    dcc.Store(id="pb-nrows", data=3),
                    dcc.Loading(dcc.Graph(
                        id="fig-play", style={"height": "620px"},
                        config={"scrollZoom": True,
                                "displayModeBar": True})),
                    html.Div(id="pb-sink", style={"display": "none"}),
                ]),
            ]),
        ]),
        ])

    app.layout = html.Div([
        html.Div([
            html.H3("multifiber", style={"margin": "4px 8px"}),
            html.Div([
                dcc.Input(id="root-path", type="text",
                          value=str(Path(args.root or
                                         (args.session_dir or ".")).resolve()),
                          placeholder="folder to search for sessions",
                          style={"width": "520px"}),
                html.Button("scan", id="scan-btn", n_clicks=0),
                html.Button("browse…", id="browse-btn", n_clicks=0,
                            title="open a folder picker on this machine"),
                dcc.Dropdown(id="session-pick", placeholder="session",
                             style={"width": "420px"}, clearable=False),
                html.Button("open", id="open-btn", n_clicks=0),
                html.Span(id="pick-msg", style={"fontSize": "11px",
                                                "color": "#555"}),
            ], style={"display": "flex", "gap": "8px", "alignItems": "center",
                      "flexWrap": "wrap", "padding": "4px 8px"}),
        ], style={"borderBottom": "1px solid #ddd"}),
        dcc.Loading(html.Div(id="session-ui")),
    ])

    @app.callback(Output("root-path", "value"),
                  Input("browse-btn", "n_clicks"), prevent_initial_call=True)
    def browse(_n):
        """Native folder picker.

        Run in a subprocess rather than in this thread: tkinter wants the
        main thread, and Dash callbacks do not run there.
        """
        import subprocess
        import sys as _sys

        code = ("import tkinter as tk;from tkinter import filedialog;"
                "r=tk.Tk();r.withdraw();r.attributes('-topmost',True);"
                "print(filedialog.askdirectory() or '')")
        try:
            out = subprocess.run([_sys.executable, "-c", code],
                                 capture_output=True, text=True, timeout=300)
            chosen = (out.stdout or "").strip().splitlines()
            return chosen[-1] if chosen and chosen[-1] else dash_no_update
        except Exception:
            return dash_no_update

    @app.callback(Output("session-pick", "options"),
                  Output("session-pick", "value"),
                  Output("pick-msg", "children"),
                  Input("scan-btn", "n_clicks"), Input("root-path", "value"))
    def scan(_n, root):
        found = find_sessions(root or ".")
        if not found:
            return [], None, (f"no processed sessions under {root!r} "
                              "(looking for *_manifest.json)")
        opts = [{"label": d.name, "value": str(d)} for d in found]
        return opts, str(found[0]), f"{len(found)} session(s) found"

    @app.callback(Output("session-ui", "children"),
                  Input("open-btn", "n_clicks"),
                  State("session-pick", "value"))
    def open_session(_n, path):
        target = path or args.session_dir
        if not target:
            return html.Div("Pick a folder above, scan, and press open.",
                            style={"padding": "20px", "color": "#777"})
        try:
            msg = load(target)
        except Exception as exc:
            return html.Div(f"could not load {target}: "
                            f"{type(exc).__name__}: {exc}",
                            style={"padding": "20px", "color": "#a33"})
        print(msg)
        return session_ui()

    def _alpha(v):
        """'cv' or a number; anything else falls back to 'cv'."""
        try:
            return "cv" if str(v).strip() == "cv" else float(v)
        except (TypeError, ValueError):
            return "cv"

    def spec_from(vals, label):
        (tau, lam, diff, method, pred, common, alpha, blank, exp,
         base, resp, detrend, stat, lp, rmethod, stimart,
         head, halpha, unpen) = vals
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
            regress_method=rmethod, stim_artifact=bool(stimart),
            head_method=head, head_alpha=_alpha(halpha),
            unpenalized=int(unpen or 1),
            lowpass_hz=(float(lp) if lp else None), label=label)

    ids = lambda p: [Input(f"{p}-{k}", "value") for k in
                     ("tau", "lam", "diff", "method", "pred", "common",
                      "alpha", "blank", "exp", "base", "resp", "detrend",
                      "stat", "lp", "rmethod", "stimart", "head", "halpha",
                      "unpen")]

    @app.callback(
        Output("fig-trace", "figure"), Output("fig-peth", "figure"),
        Output("fig-dose", "figure"), Output("stats", "children"),
        Input("roi", "value"), Input("site", "value"), Input("pmode", "value"),
        *ids("A"), *ids("B"))
    def update(roi, site, pmode, *vals):
        sd = ST["sd"]
        if sd is None:
            return go.Figure(), go.Figure(), go.Figure(), "no session loaded"
        classes, labels_by_class = ST["classes"], ST["labels"]
        a = spec_from(vals[:19], "A")
        b = spec_from(vals[19:], "B")
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

    # ---------------- all fibers at once ----------------------------------- #

    @app.callback(Output("fig-grid", "figure"), Output("grid-stats", "children"),
                  Input("g-site", "value"), Input("g-cond", "value"),
                  Input("g-variant", "value"), Input("g-stat", "value"),
                  Input("g-cols", "value"), Input("g-opts", "value"),
                  Input("g-nshift", "value"), Input("g-xrange", "value"),
                  Input("g-resp", "value"), Input("g-base", "value"),
                  Input("g-tau", "value"), Input("g-lp", "value"),
                  Input("g-pred", "value"), Input("g-method", "value"),
                  Input("g-blank", "value"), Input("g-alpha", "value"),
                  Input("g-rmethod", "value"), Input("g-stimart", "value"),
                  Input("g-head", "value"), Input("g-halpha", "value"),
                  Input("g-common", "value"),
                  Input("g-pw", "value"),
                  Input("g-ph", "value"), Input("g-pal", "value"),
                  Input("g-colorby", "value"), Input("g-dash", "value"),
                  Input("g-lw", "value"),
                  *ids("A"), *ids("B"))
    def grid(site, cond, which, stat, ncol, opts, nshift, xr, resp, base,
             g_tau, g_lp, g_pred, g_method, g_blank, g_alpha, g_rmethod,
             g_stimart, g_head, g_halpha, g_common, pw, ph, palname,
             colorby, dashmode, lw, *vals):
        from plotly.subplots import make_subplots

        sd = ST["sd"]
        if sd is None:
            return go.Figure(), "no session loaded"
        classes, labels_by_class = ST["classes"], ST["labels"]

        pre, post = float(-min(xr[0], -0.1)), float(max(xr[1], 0.1))
        if which == "A":
            base_spec = spec_from(vals[:19], "A")
        elif which == "B":
            base_spec = spec_from(vals[19:], "B")
        else:
            base_spec = V.VariantSpec(
                baseline_tau_s=float(g_tau or 60), baseline_method=g_method,
                regress=(g_pred != "none"),
                predictors=("same" if g_pred == "none" else g_pred),
                common_method=g_common,
                blank_laser=bool(g_blank), alpha=_alpha(g_alpha),
                regress_method=g_rmethod, stim_artifact=bool(g_stimart),
                head_method=g_head, head_alpha=_alpha(g_halpha),
                lowpass_hz=(float(g_lp) if g_lp else None))
        # The window controls belong to this tab even when the preprocessing
        # is inherited, so they are applied on top.
        import dataclasses as _dc
        spec = _dc.replace(base_spec, pre_s=pre, post_s=post,
                           baseline_window_s=float(base or 0.5),
                           response_start_s=float(resp[0]),
                           response_window_s=float(resp[1] - resp[0]),
                           statistic=stat, label="grid")
        res = cached(spec)

        ev_df = sd.events
        if len(ev_df) and site is not None and "region" in ev_df.columns:
            ev_df = ev_df[ev_df["region"] == site]
        conds = []
        if len(ev_df) and "protocol_class" in ev_df.columns and classes:
            if cond == "separate":
                for c in sorted(ev_df["protocol_class"].unique()):
                    conds.append((labels_by_class.get(int(c), str(c)),
                                  ev_df[ev_df["protocol_class"] == c]))
            elif str(cond).startswith("only:"):
                c = int(str(cond).split(":")[1])
                conds.append((labels_by_class.get(c, str(c)),
                              ev_df[ev_df["protocol_class"] == c]))
            else:
                conds.append(("POOLED", ev_df))
        else:
            conds.append(("all", ev_df))

        labels = sd.labels
        ncol = int(ncol or 3)
        nrow = int(np.ceil(len(labels) / ncol))
        fig = make_subplots(rows=nrow, cols=ncol, shared_xaxes=False,
                            subplot_titles=[l for l in labels],
                            vertical_spacing=max(0.02, 0.10 / nrow),
                            horizontal_spacing=0.05)
        PALETTES = {
            # Okabe-Ito: distinguishable under every common form of colour
            # blindness, which matters for a figure anyone else will read.
            "okabe": ["#0173B2", "#D55E00", "#029E73", "#CC78BC", "#CA9161",
                      "#ECE133", "#56B4E9", "#949494"],
            "bwr": ["#2166AC", "#67A9CF", "#B2182B", "#EF8A62", "#404040"],
            "viridis": ["#440154", "#3B528B", "#21918C", "#5EC962",
                        "#FDE725"],
            "grey": ["#000000", "#555555", "#888888", "#BBBBBB"],
        }
        pal = PALETTES.get(palname, PALETTES["okabe"])
        DASH = (["solid", "dash", "dot", "dashdot"]
                if dashmode != "solid" else ["solid"])
        lw = float(lw or 1.8)
        want_null = "null" in (opts or [])
        want_trials = "trials" in (opts or [])
        lines = []

        for i, lab in enumerate(labels):
            r, c = divmod(i, ncol)
            r, c = r + 1, c + 1
            for k, (cname, g) in enumerate(conds):
                onsets = (g["onset_t"].to_numpy(dtype=float) if len(g)
                          else np.array([]))
                if onsets.size == 0:
                    continue
                width = float(g["width_s"].median())
                if want_null:
                    out = V.compare(res, onsets, i,
                                    n_shift=int(nshift or 200),
                                    width_s=width)
                else:
                    out = V.peri_event(res, onsets, i, width_s=width)
                if not out.get("n"):
                    continue
                colour = (pal[k % len(pal)] if colorby == "cond"
                          else pal[i % len(pal)])
                if want_null and "null" in out:
                    lo = np.nanpercentile(out["null"]["traces"], 2.5, axis=0)
                    hi = np.nanpercentile(out["null"]["traces"], 97.5, axis=0)
                    fig.add_trace(go.Scatter(
                        x=np.r_[out["t"], out["t"][::-1]],
                        y=np.r_[hi, lo[::-1]], fill="toself",
                        fillcolor=colour, opacity=0.10, line=dict(width=0),
                        showlegend=False, hoverinfo="skip"), row=r, col=c)
                if want_trials:
                    for tr in out["trials"][:12]:
                        fig.add_trace(go.Scatter(
                            x=out["t"], y=tr,
                            line=dict(width=max(0.3, lw * 0.25),
                                      color=colour),
                            opacity=0.45, showlegend=False,
                            hoverinfo="skip"), row=r, col=c)
                fig.add_trace(go.Scatter(
                    x=out["t"], y=out["mean"], name=cname,
                    legendgroup=cname, showlegend=(i == 0),
                    line=dict(width=lw, color=colour,
                              dash=DASH[k % len(DASH)])), row=r, col=c)
                lines.append((lab, cname, out))
            fig.add_vrect(x0=float(resp[0]), x1=float(resp[1]),
                          fillcolor="#0173B2", opacity=0.07, line_width=0,
                          row=r, col=c)
            fig.add_hline(y=0, line=dict(width=0.8, dash="dot",
                                         color="#000"), row=r, col=c)
            fig.update_xaxes(range=[-pre, post], row=r, col=c)

        # Titles carry the statistic, so the grid can be read without the
        # table below it.
        for i, lab in enumerate(labels):
            ann = [x for x in (lab, ) if x]
            best = [o for (L, cn, o) in lines if L == lab]
            if best and "z" in best[0]:
                zs = ", ".join(f"{o.get('z', float('nan')):+.1f}"
                               for o in best)
                fig.layout.annotations[i].text = f"{lab}  z={zs}"
            fig.layout.annotations[i].font.size = 9

        # Explicit panel size rather than stretching to the container: a
        # PETH read at the wrong aspect ratio is genuinely harder to parse,
        # and the window is not the right thing to be setting it.
        pw, ph = int(pw or 320), int(ph or 220)
        fig.update_layout(
            width=ncol * pw + 90, height=nrow * ph + 110, autosize=False,
            margin=dict(l=55, r=20, t=56, b=46),
            legend=dict(orientation="h", y=1.02),
            title=(f"stimulating {site} — {spec.describe()}"),
            plot_bgcolor="white", uirevision="grid")
        fig.update_xaxes(showgrid=False, zeroline=False, ticks="outside",
                         linecolor="#444", linewidth=0.8)
        fig.update_yaxes(showgrid=False, zeroline=False, ticks="outside",
                         linecolor="#444", linewidth=0.8)
        fig.update_xaxes(title_text="time from onset (s)", row=nrow)

        txt = [f"stimulating {site} | {spec.describe()}",
               f"{'fiber':12s} {'condition':>18s} {'n':>4s} {'effect':>10s} "
               f"{'z':>7s} {'p':>8s} {'floor z3':>10s}"]
        for lab, cname, o in lines:
            fl = (3 * o["null"]["sd"] if "null" in o else float("nan"))
            txt.append(f"{lab:12s} {cname:>18s} {o['n']:4d} "
                       f"{o['effect']:+10.5f} "
                       f"{o.get('z', float('nan')):+7.2f} "
                       f"{o.get('p', float('nan')):8.4f} {fl:10.4%}")
        if not want_null:
            txt.append("\n(null bands off: z and p are not computed, and the "
                       "effects below cannot be judged without them)")
        return fig, "\n".join(txt)

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

    @app.callback(Output("fig-play", "figure"), Output("pb-nrows", "data"),
                  Input("pb-rois", "value"), Input("pb-variant", "value"),
                  Input("pb-opts", "value"), Input("pb-win", "value"),
                  Input("pb-tau", "value"), Input("pb-lp", "value"),
                  Input("pb-pred", "value"), Input("pb-method", "value"),
                  Input("pb-blank", "value"), Input("pb-scale", "value"),
                  Input("pb-scale415", "value"), Input("pb-rows-sel", "value"),
                  Input("pb-alpha", "value"), Input("pb-rmethod", "value"),
                  Input("pb-stimart", "value"), Input("pb-head", "value"),
                  Input("pb-halpha", "value"),
                  State("pb-time", "value"),
                  *ids("A"), *ids("B"))
    def draw_session(rois, which, opts, win, pb_tau, pb_lp, pb_pred,
                     pb_method, pb_blank, scale470, scale415, rows_sel,
                     pb_alpha, pb_rmethod, pb_stimart, pb_head, pb_halpha,
                     t_now, *vals):
        from plotly.subplots import make_subplots

        sd = ST["sd"]
        if sd is None:
            return go.Figure(), 1

        if which == "A":
            spec = spec_from(vals[:19], "A")
        elif which == "B":
            spec = spec_from(vals[19:], "B")
        else:
            spec = V.VariantSpec(
                baseline_tau_s=float(pb_tau or 60),
                baseline_method=pb_method,
                regress=(pb_pred != "none"),
                predictors=("same" if pb_pred == "none" else pb_pred),
                blank_laser=bool(pb_blank),
                alpha=_alpha(pb_alpha), regress_method=pb_rmethod,
                head_method=pb_head, head_alpha=_alpha(pb_halpha),
                stim_artifact=bool(pb_stimart),
                lowpass_hz=(float(pb_lp) if pb_lp else None),
                label="playback")
        res = cached(spec)
        rois = [int(r) for r in (rois or [0])]
        win = float(win or 20)

        rows_sel = [r for r in ("470", "415", "dff") if r in (rows_sel or [])]
        if not rows_sel:
            rows_sel = ["470"]
        titles = {"470": "470 nm raw (signal channel)",
                  "415": "415 nm raw (isosbestic)",
                  "dff": f"processed dF/F — {spec.describe()}"}
        fig = make_subplots(rows=len(rows_sel), cols=1, shared_xaxes=True,
                            vertical_spacing=0.06,
                            subplot_titles=[titles[r] for r in rows_sel])
        row_of = {name: i + 1 for i, name in enumerate(rows_sel)}

        pal = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161",
               "#949494", "#ECE133", "#56B4E9", "#D55E00", "#F0E442",
               "#009E73", "#000000"]
        sig_state = int(np.bincount(sd.led_state).argmax())
        sig = sd.led_state == sig_state
        masks = {"470": sig, "415": ~sig}
        scales = {"470": scale470, "415": scale415}

        def scaled(channel, j):
            """Each channel referenced to ITS OWN median.

            The isosbestic is dimmer than the signal, so dividing it by the
            470 median would compress it and make the two incomparable --
            which is the whole point of putting them on separate rows.
            """
            m = masks[channel]
            v = sd.traces[m, j]
            if scales[channel] == "counts":
                return v, "counts"
            med = float(np.nanmedian(v)) or 1.0
            return v / med - 1.0, "fraction of own median"

        def offset(channel, k):
            if scales[channel] != "stack":
                return 0.0
            m = masks[channel]
            sp = []
            for i in rois:
                v = sd.traces[m, i]
                med = float(np.nanmedian(v)) or 1.0
                sp.append(np.nanpercentile(np.abs(v / med - 1.0), 99))
            return k * 2.5 * float(np.nanmax(sp)) if sp else 0.0

        units = {}
        for channel in ("470", "415"):
            if channel not in row_of:
                continue
            r = row_of[channel]
            for k, j in enumerate(rois):
                y, unit = scaled(channel, j)
                units[channel] = unit
                fig.add_trace(go.Scattergl(
                    x=sd.frame_t[masks[channel]], y=y + offset(channel, k),
                    name=sd.labels[j], legendgroup=str(j),
                    showlegend=(r == row_of[rows_sel[0]]),
                    line=dict(width=1, color=pal[j % len(pal)]),
                    hovertemplate=f"{sd.labels[j]} {channel}: "
                                  f"%{{y:.4f}}<extra></extra>"),
                    row=r, col=1)
                if scales[channel] == "stack":
                    fig.add_annotation(
                        x=float(sd.frame_t[0]), y=offset(channel, k),
                        xref=f"x{'' if r == 1 else r}",
                        yref=f"y{'' if r == 1 else r}",
                        text=sd.labels[j], showarrow=False, xanchor="left",
                        yanchor="bottom",
                        font=dict(size=8, color=pal[j % len(pal)]))

        if "dff" in row_of:
            r = row_of["dff"]
            spread = float(np.nanpercentile(np.abs(res.dff[:, rois]), 99))
            for k, j in enumerate(rois):
                doff = k * 2.5 * spread if scale470 == "stack" else 0.0
                fig.add_trace(go.Scattergl(
                    x=res.t, y=res.dff[:, j] + doff, name=sd.labels[j],
                    legendgroup=str(j),
                    showlegend=(r == row_of[rows_sel[0]]),
                    line=dict(width=1.2, color=pal[j % len(pal)])),
                    row=r, col=1)
                if res.blanked.any():
                    b = res.blanked
                    fig.add_trace(go.Scattergl(
                        x=res.t[b], y=res.dff[b, j] + doff, mode="markers",
                        marker=dict(size=4, color="#D55E00", symbol="x"),
                        name="blanked", legendgroup="blank",
                        showlegend=(j == rois[0]), hoverinfo="skip"),
                        row=r, col=1)

        # Laser commands, labelled, spanning every row.
        shapes, annos, htxt, hx = [], [], [], []
        if len(sd.events):
            cls_col = ("protocol_class" if "protocol_class" in sd.events.columns
                       else None)
            label_all = "labels" in (opts or [])
            for _, rr in sd.events.iterrows():
                a, b = float(rr["onset_t"]), float(rr["offset_t"])
                for i in range(len(rows_sel)):
                    ax = "" if i == 0 else str(i + 1)
                    shapes.append(dict(type="rect", xref=f"x{ax}",
                                       yref=f"y{ax} domain", x0=a, x1=b,
                                       y0=0, y1=1, line_width=0,
                                       fillcolor="#D55E00", opacity=0.18,
                                       layer="below"))
                site = rr.get("region") or f"roi {int(rr.get('roi_id', -1))}"
                dur = (b - a) * 1e3
                volt = rr.get("command_v")
                cid = int(rr[cls_col]) if cls_col is not None else None
                txt = f"{site} · {dur:.0f} ms"
                if volt is not None and np.isfinite(volt):
                    txt += f" · {volt:.3f} V"
                if cid is not None:
                    txt += f" · class {cid}"
                hx.append(0.5 * (a + b))
                htxt.append(txt)
                if label_all:
                    annos.append(dict(x=0.5 * (a + b), y=1.02, xref="x",
                                      yref="y domain",
                                      text=f"{site}<br>{dur:.0f} ms",
                                      showarrow=False, align="center",
                                      yanchor="bottom",
                                      font=dict(size=7, color="#9c4a00")))
            if hx:
                fig.add_trace(go.Scattergl(
                    x=hx, y=np.full(len(hx), np.nan), mode="markers",
                    marker=dict(size=1, opacity=0), hovertext=htxt,
                    hoverinfo="text", name="stimulation", showlegend=False),
                    row=1, col=1)

        if len(annos) > 300:
            annos = []
        # Keep the view where playback already is, rather than jumping back
        # to the start whenever a control changes.
        t_cur = float(t_now if t_now is not None else sd.frame_t[0])
        lo, hi = t_cur - 0.75 * win, t_cur + 0.25 * win

        # uirevision preserves axes across figure updates -- which is what
        # keeps panning smooth, but it also PINS the y-axis. Changing a scale
        # produces data orders of magnitude different and it would be drawn
        # against the old range. Keying the revision to everything that
        # changes the data means those axes reset, while a playback tick,
        # which changes none of it, does not disturb the view.
        rev = "|".join(str(x) for x in
                       (rows_sel, scale470, scale415, sorted(rois), which,
                        spec.key(), bool("iso" in (opts or []))))
        fig.update_layout(
            shapes=shapes, annotations=list(fig.layout.annotations) + annos,
            margin=dict(l=60, r=10, t=48, b=36), uirevision=rev,
            dragmode="pan", legend=dict(orientation="h", y=1.14),
            height=max(300, 210 * len(rows_sel)))
        for i, name in enumerate(rows_sel):
            r = i + 1
            fig.update_xaxes(range=[lo, hi], row=r, col=1, autorange=False,
                             title_text=("time (s)"
                                         if r == len(rows_sel) else None))
            # y rescales to the new data; the x window is what playback owns
            fig.update_yaxes(autorange=True, row=r, col=1)
            lbl = ("dF/F" if name == "dff"
                   else units.get(name, "counts"))
            fig.update_yaxes(title_text=lbl, row=r, col=1,
                             showticklabels=(scales.get(name) != "stack"))
        return fig, len(rows_sel)

    # Advance the clock in the browser, and pan by relayout. No server round
    # trip, no data transferred: this is what keeps it smooth.
    app.clientside_callback(
        """
        function(nTick, tSlider, win, speed, nRows, tMin, tMax) {
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
                // One axis per visible row; the row count is passed in so
                // relayout never names an axis that is not there.
                const upd = {};
                const n = nRows || 1;
                for (let i = 1; i <= n; i++) {
                    upd["xaxis" + (i === 1 ? "" : i) + ".range"] = [lo, hi];
                }
                Plotly.relayout(gd, upd);
            }
            const clock = t.toFixed(2) + " s  /  " + tMax.toFixed(0) + " s";
            return [byTick ? t : dash_clientside.no_update, clock];
        }
        """,
        [Output("pb-time", "value"), Output("pb-clock", "children")],
        [Input("pb-tick", "n_intervals"), Input("pb-time", "value"),
         Input("pb-win", "value"), Input("pb-speed", "value"),
         Input("pb-nrows", "data")],
        [State("pb-time", "min"), State("pb-time", "max")],
    )

    if not args.session_dir:
        print("no session given — pick one in the browser "
              "(scan a folder, or use browse…)")
    print(f"open http://127.0.0.1:{args.port}")
    app.run(debug=False, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
