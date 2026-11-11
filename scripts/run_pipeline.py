"""
Run the multifiber pipeline from a terminal.

Prefer this over a notebook.  scikit-learn and numpy can link different OpenMP
runtimes on Windows; in a Jupyter kernel that combination has been observed to
kill the process with a heap-corruption exit (0xC0000374 / 3228369023) and no
traceback, while the identical code completes in a terminal.  This script sets
the threading environment before importing anything that would load a runtime.

    conda activate multifiber
    python scripts/run_pipeline.py "F:\\Polina\\...\\101926" "F:\\Polina\\processed" ^
        --registry "C:\\Users\\FalknerLab\\code\\multifiber-metadata" ^
        --calibration "F:\\Jorge\\Tests\\multistim"

To use it from a notebook anyway, set the same variables in the FIRST cell,
before importing numpy, and restart the kernel:

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
"""

from __future__ import annotations

import argparse
import os


def _set_threading(threads: int, allow_duplicate_omp: bool) -> None:
    """Must run before numpy / scipy / sklearn are imported."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    if allow_duplicate_omp:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_root")
    ap.add_argument("out_root")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--session", action="append", default=None,
                    help="process only this stem; repeatable")
    ap.add_argument("--force", action="store_true",
                    help="reprocess even if inputs and config are unchanged. "
                         "ROI extraction is still reused when the video, the "
                         "calibration and the extraction settings are "
                         "unchanged, since no preprocessing parameter affects "
                         "it.")
    ap.add_argument("--force-extract", action="store_true",
                    help="decode the video again as well")
    ap.add_argument("--threads", type=int, default=1,
                    help="BLAS/OpenMP threads (1 is the safe default here; "
                         "the solves are small and thread contention has been "
                         "the larger risk)")
    ap.add_argument("--allow-duplicate-omp", action="store_true", default=True)
    ap.add_argument("--baseline-tau-s", type=float, default=60.0)
    ap.add_argument("--predictors", choices=("same", "all", "common"), default="common",
                    help="motion regressors: 'same' uses only that fiber's "
                         "isosbestic, 'all' every fiber's separately, "
                         "'common' the fiber's own plus a noise-averaged "
                         "component from the others -- which is what lets a "
                         "dim isosbestic still be corrected.")
    ap.add_argument("--exposure-s", type=float, default=None,
                    help="sensor integration time per frame, seconds. A "
                         "physical property of the acquisition, not a tuning "
                         "knob: read it back from the camera "
                         "(src.ExposureTime), do not pick a value that makes "
                         "the contamination warning go away. Omit to treat "
                         "the whole frame period as exposed.")
    ap.add_argument("--no-blank-laser", action="store_true",
                    help="do not blank laser-contaminated frames; use to "
                         "inspect the artifact before deciding what to drop")
    ap.add_argument("--animal", default=None,
                    help="animal_id to assume for sessions whose filename "
                         "matches no record. The fiber->region map is "
                         "animal-specific: if the guess is wrong, every "
                         "region label is wrong and nothing errors. For rig "
                         "tests only.")
    ap.add_argument("--no-regress-isosbestic", action="store_true",
                    help="skip the isosbestic regression. Use when the 415 "
                         "channel is near-dark: its dF/F is then dominated by "
                         "its own small denominator and subtracting it adds "
                         "noise instead of removing motion.")
    ap.add_argument("--common-mode", choices=("median", "pc1"), default=None,
                    help="remove the across-fiber shared component, the "
                         "signature of bundle motion. Stimulated fibers are "
                         "excluded from the estimate. Removes shared neural "
                         "activity too, so do not use it for analyses of "
                         "network-wide co-fluctuation.")
    ap.add_argument("--baseline-mode", choices=("continuous", "epochs"),
                    default="continuous",
                    help="fit F0 in one piece (default; correct when the "
                         "animal's regime does not change during the "
                         "session) or separately per baseline/stim/post "
                         "epoch (for assays with a genuine regime change, "
                         "e.g. intruder introduction).")
    ap.add_argument("--baseline-method",
                    choices=("airpls", "symmetric"), default="airpls",
                    help="airpls fits the lower envelope, so dF/F is "
                         "positively skewed; symmetric fits the centre of the "
                         "data and centres the scale. The peri-event contrast "
                         "is identical either way.")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress stage announcements and progress bars")
    ap.add_argument("--summary-csv", default=None)
    args = ap.parse_args()

    _set_threading(args.threads, args.allow_duplicate_omp)

    from multifiber import PipelineConfig, process_directory

    cfg = PipelineConfig(baseline_tau_s=args.baseline_tau_s,
                         regression_predictors=args.predictors,
                         exposure_s=args.exposure_s,
                         blank_laser=not args.no_blank_laser,
                         regress_isosbestic=not args.no_regress_isosbestic,
                         show_progress=not args.quiet,
                         common_mode=args.common_mode,
                         baseline_mode=args.baseline_mode,
                         baseline_method=args.baseline_method)
    results, summary = process_directory(
        data_root=args.data_root,
        out_root=args.out_root,
        registry_root=args.registry,
        calibration_dir=args.calibration,
        config=cfg,
        sessions=args.session,
        force=args.force,
        force_extract=args.force_extract,
        default_animal=args.animal,
        progress=(None if args.quiet else (lambda m: print(m, flush=True))),
    )

    print("\n=== summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    if args.summary_csv:
        summary.to_csv(args.summary_csv, index=False)
        print(f"\nsummary written to {args.summary_csv}", flush=True)

    failed = [r for r in results if r.status == "failed"]
    for r in failed:
        print(f"\nFAILED {r.session_id} at {r.failed_stage}: {r.error}",
              flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
