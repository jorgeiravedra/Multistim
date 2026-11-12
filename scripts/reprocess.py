"""
Re-run preprocessing on already-extracted sessions.

Reads ``<sid>_traces.parquet`` (raw per-frame ROI means, frame times and LED
state, written by the pipeline) and rewrites ``<sid>_dff.parquet`` with new
preprocessing parameters.  No video is opened, no directory is searched, and
no file is fingerprinted: the only inputs are the processed output directory
and, when laser-artifact blanking is on, the session's DAQ log.

Use this to sweep baseline, regression or blanking settings.  Anything that
changes the *extracted traces* -- the ROI fraction, the sensor half, the
calibration -- is not reachable from here and needs the full pipeline.

    python scripts/reprocess.py "F:\\Jorge\\Multistim_Analysis\\102626" ^
        --baseline-mode continuous --baseline-tau-s 60

The manifest is updated in place: the new config and QC are written, the
extraction provenance is carried through unchanged, and ``reprocessed_from``
records that the traces were not re-extracted.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path


def main() -> int:
    import numpy as np
    import pandas as pd

    from multifiber import __version__, events as ev, io as mio, matio
    from multifiber import preprocess as pp
    from multifiber.pipeline import PipelineConfig

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_root", help="directory of processed sessions")
    ap.add_argument("--session", action="append", default=None,
                    help="session id; repeatable. Default: all.")
    ap.add_argument("--baseline-mode", choices=("continuous", "epochs"),
                    default=None)
    ap.add_argument("--baseline-tau-s", type=float, default=None)
    ap.add_argument("--airpls-differences", type=int, default=None)
    ap.add_argument("--guard-s", type=float, default=None)
    ap.add_argument("--predictors", choices=("same", "all", "common"), default=None)
    ap.add_argument("--alpha", default=None,
                    help="'cv' or a float")
    ap.add_argument("--no-regress-isosbestic", action="store_true")
    ap.add_argument("--common-mode", choices=("median", "pc1"), default=None)
    ap.add_argument("--no-common-mode", action="store_true")
    ap.add_argument("--exposure-s", type=float, default=None)
    ap.add_argument("--no-blank-laser", action="store_true")
    ap.add_argument("--blank-pad", type=int, default=None)
    ap.add_argument("--smooth", type=int, default=None)
    ap.add_argument("--regress-method",
                    choices=("lasso", "ols", "ridge", "hierarchical",
                             "auto"),
                    default=None,
                    help="estimator for the motion regression. lasso is "
                         "non-negative and sparse; ols fits every predictor "
                         "fully, which a brief artifact needs; hierarchical "
                         "fits the own fiber unpenalized first; auto "
                         "picks per fiber by out-of-sample error on held-out "
                         "blocks, scored outside the stimulation windows so "
                         "the choice cannot be influenced by the response.")
    ap.add_argument("--stim-artifact", action="store_true",
                    help="add stimulus-locked square and kinetic regressors. "
                         "The square term absorbs an optical artifact, which "
                         "has no kinetics; the kinetic term is estimated and "
                         "kept as the response.")
    ap.add_argument("--common-method", choices=("median", "pc1", "pc2", "pc3", "pc4", "pc5",
                             "pc6", "pc7"),
                    default=None,
                    help="common isosbestic component: median, or pcN "
                         "for the first N principal components")
    ap.add_argument("--baseline-method",
                    choices=("airpls", "symmetric"), default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    root = Path(args.out_root)
    sessions = sorted(p.parent for p in root.glob("*/*_manifest.json"))
    if args.session:
        want = set(args.session)
        sessions = [d for d in sessions if d.name in want]
    if not sessions:
        print(f"no processed sessions under {root}")
        return 1

    n_ok = n_skip = n_fail = 0
    for d in sessions:
        sid = d.name
        man_p = d / f"{sid}_manifest.json"
        traces_p = d / f"{sid}_traces.parquet"
        man = json.loads(man_p.read_text(encoding="utf-8"))

        if not traces_p.exists():
            print(f"{sid}: no traces parquet (run the pipeline with "
                  "write_raw_traces=True); skipping")
            n_skip += 1
            continue
        if man.get("status") != "ok":
            print(f"{sid}: prior run status {man.get('status')!r}; skipping")
            n_skip += 1
            continue

        # --- config: start from what was used, apply overrides ------------- #
        prior_cfg = man.get("config", {})
        fields = set(PipelineConfig.__dataclass_fields__)
        cfg = PipelineConfig(**{k: v for k, v in prior_cfg.items()
                                if k in fields})
        over = {
            "baseline_mode": args.baseline_mode,
            "baseline_method": args.baseline_method,
            "isosbestic_common_method": args.common_method,
            "regress_method": args.regress_method,
            "stim_artifact_regression": (True if args.stim_artifact else None),
            "baseline_tau_s": args.baseline_tau_s,
            "airpls_differences": args.airpls_differences,
            "guard_s": args.guard_s,
            "regression_predictors": args.predictors,
            "exposure_s": args.exposure_s,
            "blank_pad": args.blank_pad,
            "smooth": args.smooth,
        }
        for k, v in over.items():
            if v is not None:
                setattr(cfg, k, v)
        if args.alpha is not None:
            cfg.regression_alpha = ("cv" if args.alpha == "cv"
                                    else float(args.alpha))
        if args.no_regress_isosbestic:
            cfg.regress_isosbestic = False
        if args.common_mode:
            cfg.common_mode = args.common_mode
        if args.no_common_mode:
            cfg.common_mode = None
        if args.no_blank_laser:
            cfg.blank_laser = False

        if cfg.hash() == man.get("config_hash") and not args.dry_run:
            print(f"{sid}: config unchanged; skipping")
            n_skip += 1
            continue

        try:
            tdf = pd.read_parquet(traces_p)
            labels = [c for c in tdf.columns
                      if c not in ("frame", "t", "led_state", "roi_mean")]
            traces = tdf[labels].to_numpy(dtype=float)
            led_state = tdf["led_state"].to_numpy(dtype=int)
            frame_t = tdf["t"].to_numpy(dtype=float)

            ev_p = d / f"{sid}_events.parquet"
            onset = offset = None
            stim_roi_idx = None
            if ev_p.exists():
                edf = pd.read_parquet(ev_p)
                if len(edf):
                    onset = edf["onset_t"].to_numpy(dtype=float)
                    offset = edf["offset_t"].to_numpy(dtype=float)
                    if cfg.common_mode:
                        ids = {int(i) for i in edf["roi_id"].unique() if i > 0}
                        stim_roi_idx = sorted(i - 1 for i in ids)

            # Gate times are not stored in the outputs; they come from the DAQ
            # log, which is small. Only needed when blanking is on.
            gate_on_t = None
            if cfg.blank_laser and cfg.laser_gate_channel:
                bcam = man.get("inputs", {}).get("paths", {}).get("bcam")
                if bcam and Path(bcam).exists():
                    log = matio.load_bcam(bcam)
                    g = log.channels.get(cfg.laser_gate_channel)
                    if g is not None and float(np.nanmax(g) - np.nanmin(g)) >= 0.5:
                        gate_on_t = ev.detect_edges(log.time, g, 0.5).rising_t
                else:
                    print(f"{sid}: DAQ log not found at {bcam!r}; "
                          "proceeding without laser blanking")

            proc = pp.preprocess_traces(
                traces, led_state, frame_t,
                laser_onset_t=onset, laser_offset_t=offset,
                gate_on_t=gate_on_t, exposure_s=cfg.exposure_s,
                labels=labels,
                baseline_tau_s=cfg.baseline_tau_s,
                differences=cfg.airpls_differences,
                baseline_mode=cfg.baseline_mode,
                baseline_method=cfg.baseline_method,
                common_method=cfg.isosbestic_common_method,
                guard_s=cfg.guard_s,
                alpha=cfg.regression_alpha,
                predictors=cfg.regression_predictors,
                regress=cfg.regress_isosbestic,
                common_mode=cfg.common_mode,
                common_mode_exclude=stim_roi_idx,
                blank_pad=cfg.blank_pad,
                smooth=cfg.smooth,
            )

            old = man.get("qc", {})
            print(f"{sid}: blanked {old.get('n_blanked')} -> "
                  f"{int(proc.blanked.sum())}   "
                  f"median |dF/F| SD {np.nanmedian(old.get('dff_sd_after_regression') or [np.nan]):.4f}"
                  f" -> {np.nanmedian(proc.qc['dff_sd_after_regression']):.4f}")
            for w in proc.qc.get("warnings", []):
                print(f"    ! {w}")
            if args.dry_run:
                n_ok += 1
                continue

            man["config"] = {k: getattr(cfg, k) for k in fields}
            man["config_hash"] = cfg.hash()
            man["package_version"] = __version__
            man["reprocessed_from"] = {
                "traces": str(traces_p),
                "at": _dt.datetime.now().isoformat(timespec="seconds"),
                "previous_config_hash": prior_cfg and man.get("config_hash"),
            }
            qc = dict(man.get("qc", {}))
            qc.update({
                "effective_fps": proc.qc["effective_fps"],
                "epochs": proc.epochs,
                "regression_r2": proc.qc["regression_r2"],
                "regression_alpha": proc.qc["regression_alpha"],
                "baseline_lambda": proc.qc["lambda"],
                "baseline_mode": proc.qc["baseline_mode"],
                "f0_boundary_jumps": proc.qc["f0_boundary_jumps"],
                "dff_sd_before_regression": proc.qc["dff_sd_before_regression"],
                "dff_sd_after_regression": proc.qc["dff_sd_after_regression"],
                "isosbestic_f0_median": proc.qc["isosbestic_f0_median"],
                "n_roi_high_noise": proc.qc["n_roi_high_noise"],
                "common_mode": proc.qc["common_mode"],
                "n_blanked": int(proc.blanked.sum()),
                "exposure_s": cfg.exposure_s,
                "blank_laser": cfg.blank_laser,
            })
            man["qc"] = qc
            man["warnings"] = (
                [w for w in man.get("warnings", [])
                 if not _is_preprocessing_warning(w)]
                + list(proc.qc.get("warnings", [])))
            mio.write_session(root, sid, man, dff=proc.to_frame())
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"{sid}: FAILED: {type(exc).__name__}: {exc}")
            n_fail += 1

    print(f"\n{n_ok} reprocessed, {n_skip} skipped, {n_fail} failed")
    return 1 if n_fail else 0


def _is_preprocessing_warning(w: str) -> bool:
    """Warnings that the preprocessing stage regenerates, so stale copies are
    dropped rather than accumulated across runs."""
    marks = ("isosbestic", "residual dF/F noise", "common-mode",
             "blanking removed", "laser-gate pulse", "piecewise F0",
             "regression disabled")
    return any(m in w for m in marks)


if __name__ == "__main__":
    raise SystemExit(main())
