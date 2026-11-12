# multifiber

Extraction and preprocessing for the multi-site stim-record photometry rig.
Stage 1 of the pipeline: file discovery, metadata, and DAQ event extraction.

## Install (Windows, conda)

    conda env create -f environment.yml
    conda activate multifiber
    pip install -e .
    pytest -q tests/

## What exists now

| module | does |
|---|---|
| `matio` | reads `.mat` v7 and v7.3; parses `*_bcam.mat` (`bcl` cell array) and the stim GUI log, whose variable name and column order are discovered rather than assumed; SHA-256 provenance on every file |
| `metadata` | `<animal_id>.animal.yaml` implant registry — the fiber→region map, validated; session/epoch/LED schedule schema |
| `calibration` | resolves the shared `TranslatedCoords_*`/`mids_*`/`cal_mtx_*`/`lsr_ints_*` per session by date stamp, recording the choice and its evidence |
| `discovery` | walks a directory, groups files into sessions, binds each to its animal record and the calibration files that record names |
| `events` | camera triggers, trigger→frame alignment, LED state from DAQ **and** from image brightness with agreement scoring, laser event detection, laser→ROI assignment by timestamp |
| `extract` | ROI masks at the calibrated fiber positions; per-ROI traces and per-frame intensity from the video |
| `preprocess` | demux 415/470, laser-artifact blanking, airPLS or symmetric F0, dF/F, isosbestic regression (own / all / others / own+common, with median or principal components) |
| `pipeline` | directory in, processed dataset out; per-session error isolation, resume |
| `io` | parquet tables + a provenance manifest per session |
| `variants` | recomputes preprocessing variants from cached traces, for comparison; peri-event statistics and the circular-shift null per variant |
| `progress` | stage announcements and progress bars for the long stages |

## Run the whole thing

```python
from multifiber import PipelineConfig, process_directory

cfg = PipelineConfig(baseline_tau_s=60.0,
                     regression_predictors="common",   # own + the others
                     isosbestic_common_method="pc2",
                     regress_method="hierarchical")
results, summary = process_directory(
    data_root=r"F:\Polina\female_obs_exp\stimcohort\testing\101926",
    out_root=r"F:\Polina\processed",
    registry_root=r"C:\Users\FalknerLab\Downloads\multifiber",
    calibration_dir=r"F:\Jorge\Tests\multistim",
    config=cfg,
)
print(summary)
```

Per session this writes `<stem>_traces.parquet` (raw ROI means, frame time,
LED state), `<stem>_dff.parquet` (processed), `<stem>_events.parquet` (one row
per stimulation: time, frame, ROI, region, command voltage, gate structure,
match residual) and `<stem>_manifest.json`.

The manifest holds every input hash, every parameter, and every QC number.
Re-running skips a session only if **both** its input hashes and the config
hash are unchanged, so a replaced file or a changed parameter forces
reprocessing. A session that fails is recorded with the stage that failed and
the batch continues; `summary` has one row per session including failures.

## The app

Three tabs over one processed session: compare preprocessing choices, watch
the session play back, and see every fiber's PETH at once.

```
conda install -c conda-forge dash plotly
python scripts/compare_app.py                                  # pick a session in the browser
python scripts/compare_app.py --root "F:\Jorge\Multistim_Analysis"
python scripts/compare_app.py "F:\Jorge\Multistim_Analysis\<session>"
```

Then open <http://127.0.0.1:8050>. With no path given, use the bar at the top:
type or **browse…** to a folder, **scan** to list every session under it
(looking for `*_manifest.json`, at the root and one or two levels down), pick
one and **open**. Sessions can be switched without restarting.

Nothing the app does is written back. A variant is an exploration; the
pipeline's output remains the record, and a setting you decide on has to be
applied through `run_pipeline` or `reprocess` to become one.

### preprocessing

Two configurations, A and B, side by side on the same data. Each column sets
the baseline (timescale, difference order, airPLS or symmetric, or a raw
lambda), the motion regression (predictors, common component, estimator,
penalty, the two hierarchical stages), blanking, and the peri-event windows.
Three panels: the full trace, the peri-event average with each variant's own
null band, and the effect by condition with the detection floor as error
bars.

An interleaved session is split by condition automatically — each analysed
separately and drawn together. Pooling them is available and labelled as
such; it averages conditions that were deliberately separated at
acquisition.

**The null is recomputed for every variant**, because a null built under one
baseline does not describe another. The app never compares A's effect against
B's null.

With twelve fibers and a dozen controls, choosing the setting that gives the
biggest z will always find one. Choose on a diagnostic — does F0 track the
bleaching, does the regression lower the detection floor, does the artifact
leave the isosbestic — and not on the statistic.

### playback

Three selectable rows on one clock: 470 raw, 415 raw, processed dF/F, with
laser commands shaded and labelled by site and duration, and blanked samples
marked. Play, pause, scrub, change the window and speed, pick fibers.

Each raw row has its own scale: **% of median** (each channel against its own
median, so fibers that differ tens of times in brightness become comparable),
**stacked**, or **counts**. The processed row can take variant A's or B's
settings, or its own.

The whole session is drawn once and playback moves only the view window, in
the browser, so it stays smooth. Changing a setting redraws at the current
position rather than jumping back to the start.

This is the view for asking whether preprocessing removed an artifact or
created one: anything visible in a raw row and absent below was removed,
anything below but not above was created.

### all PETHs

Every fiber in one grid under a single configuration, with the circular-shift
null per panel and a table of n, effect, z, p and floor beneath. Panel titles
carry the z per condition.

The time axis is a range slider that re-extracts the windows, so widening it
gives real data out to ±20 s rather than rescaling. Panel size is set in
pixels rather than stretched to the window — a PETH at the wrong aspect ratio
is harder to read — and the palette, colour-by, line style and width are
adjustable. The camera button exports SVG.

## Stage-by-stage usage

```python
from multifiber import metadata, discovery, matio, events

registry = metadata.load_registry(r"C:\Users\FalknerLab\Downloads\multifiber")
sessions = discovery.discover_sessions(
    r"F:\Polina\female_obs_exp\stimcohort\testing\101926",
    registry,
    calibration_dir=r"F:\Jorge\Tests\multistim",
)
print(sessions[0].calibration.summary())
for s in sessions:
    print(s.summary())      # one line per session + its problems/warnings

s = next(x for x in sessions if x.ok)
log    = matio.load_bcam(s.bcam)
frames = events.detect_camera_frames(log)
pulses = events.detect_laser_pulses(log)                   # ('laser_a',) non-burst
assign = events.assign_laser_ids(pulses, matio.load_stim_log(s.stim_log))
assert assign.ok, assign.diagnostics

info = extract.probe_video(s.video)                 # frame count is MEASURED
align = events.align_triggers_to_frames(frames, info.n_frames)
print(align.describe())

res, masks, info = extract.extract_session(
    s.video,
    s.calibration.path("translated_coords"),
    s.calibration.path("mids"),
    labels=s.animal.implant.ordered_labels(),
)
led = events.led_state_from_brightness(res.frame_intensity)
```

## DAQ channels, as measured

Confirmed on `derek_1000ms_testpfc` (2026-10-19) with
`matio.load_bcam_raw_profile`. DAQ rate **2 kHz**, timebase jitter 2.8e-11 —
hardware-clocked, no dropped scans.

| cell | name | measured | role |
|---|---|---|---|
| 3 | `daq_time` | strictly increasing, 2 kHz | timebase |
| 4 | `ham_cam` | 0.003–3.34 V, 79.9 Hz, 8846 edges | camera exposure trigger |
| 5 | `aux_analog` | 2.064–2.098 V, 34 mV span, aperiodic | **dead** — labelled FLIRCam by the acquisition code, but carries no sync |
| 6 | `led` | 0.003–0.009 V, 6 mV span, aperiodic | **dead** — LED state must come from image brightness |
| 7 | `laser_cmd` | −0.003–0.117 V, 10 edges, 8.9% high | laser analog command; level encodes power |
| 8 | `laser_gate` | binary, 802 edges at 79.9 Hz, 2.3% high | laser digital gate, chopped at the camera rate |

Three consequences:

- **The laser occupies two channels with different meanings.** `laser_cmd` is
  the commanded envelope — its rising edge is the stimulation onset, its level
  (0.12–0.175 V, cf. `intensity_mat`) encodes power. `laser_gate` is what
  actually fires, chopped per camera frame when "LASER BETWEEN CAMERA FRAMES
  ONLY" is on: 802 gate edges for 10 commanded 1 s pulses. `detect_laser_pulses`
  returns one row per *stimulation* and reports chop count, duty cycle and
  on-time per event. Counting gate edges would turn 10 stimulations into 800.
- **Two channels are unconnected.** An unconnected input sits at a few mV of
  noise, and any threshold applied to it produces thousands of meaningless
  edges — `led` yields ~52,000 across a 6 mV span. Call `matio.check_channels`
  before trusting a channel; `detect_laser_pulses` raises rather than
  thresholding a flat line.
- **No behaviour-camera sync in this log.** Post-hoc behavioural alignment has
  nothing to key on unless `aux_analog` is live in other sessions.

Still assumed rather than measured:

- `min_width_s = 0.0015` — the legacy value, 30% of a nominal 5 ms pulse.
  Check `LaserPulses.describe()`'s width range on short protocols.
- `double_gap_s` now defaults to 0 (off). The command envelope is clean once
  the gate is handled separately, and a non-zero value would merge genuine
  closely spaced stimulations.
- Camera trigger threshold 2.0 V against a 3.34 V peak. Adequate, but not
  generous; check `reconcile_frame_count` against the video frame count.

## Design decisions worth knowing

- **The animal record is the region map.** Fiber index (1-based, as produced by
  the calibration GUI and used as the stim GUI's laser ID) is the only join
  between the laser log and the photometry channels. Non-contiguous indices are
  rejected; `histology: fail|pending` excludes a fiber by default.
- **Calibration is rig state, resolved per session by date**, not named in each
  animal record: the most recent file at or before the session date, where the
  date comes from the filename's `MMDDYY` stamp (mtime is reset by copying) and
  the session date from a `MMDDYY` path component. This is a heuristic, so the
  chosen file, its SHA-256, the runner-up, the gap in days and any newer
  candidate are all recorded on the session; `calibration_overrides` in the
  animal record pins a file and always wins. `TranslatedCoords_*` and `mids_*`
  are written as a pair by the GUI, so a split pair is an error.
- **Laser ROI identity is matched on timestamp**, after fitting a constant clock
  offset between the DAQ and the stim log, with residuals reported. Index-order
  pairing is available but off by default (`fallback_to_order`), because it
  cannot detect the permutation it is vulnerable to.
- **LED state is derived twice and compared.** `led_state_from_brightness(mode=
  'relative')` classifies each frame against its neighbours, so a dropped frame
  costs one misclassified frame rather than inverting everything after it. The
  comparison resolves label identity on a leading window and runs an explicit
  single-changepoint test, which is what distinguishes "a frame was dropped"
  from "the two sources disagree at random".
- **Nothing is silently repaired.** Frame-count offsets, channel-length
  mismatches, timebase jitter and unmatched laser events are reported; the
  caller decides.

## Diagnostics

| script | answers |
|---|---|
| `make_figures.py` | peri-event panels, the connectivity matrix, per-channel detection floors, and how many trials each channel would need to resolve a given effect |
| `make_qc_report.py` | per-session QC: channel quality, effect vs burst index, response-window spectrum, per-trial baseline level and slope |
| `spontaneous_activity.py` | whether each fiber reports anything at all — a channel with no spontaneous transients has a null result that says nothing about the perturbation |
| `trial_diagnostics.py` | whether one trial carries an effect; leave-one-out and the correlation between per-trial baseline slope and effect |
| `replication.py` | whether the same stimulated site reproduces across sessions, with Kendall's W across three or more |
| `dose_response.py` | effect against duration or power, within one session or across them |
| `diagnose_regression.py` | per fiber: the coefficient, residual-isosbestic correlation before and after, and whether the regression made the trace noisier than leaving it alone |
| `tune_regression.py` | grid search over predictors and estimators, scored on how much 415-correlated structure survives inside the stimulation windows |
| `check_phase_locking.py` | whether the stimulation lands on one LED phase — a rate whose interval is an even number of frame periods contaminates the same channel every time |
| `stim_frames.py` | raw camera frames and raw ROI means around one stimulation, with the gate overlaid |
| `tune_baseline.py` | the baseline timescale, by injecting a known response and measuring both the floor and what comes back |
| `compare_alignment.py` | this pipeline's event alignment against another's |

## Not yet built

`align` (peri-event epoching and the weight matrix), `qc`, `io`, GUI.

## Preprocessing notes

- **airPLS uses a second-difference penalty.** A first-difference penalty
  drives F0 toward a constant as λ grows, so it cannot follow photobleaching:
  on a synthetic exponential decay it under-estimates F0 by ~25% at λ ≥ 1e5,
  inflating every dF/F by the same factor. Second-difference is free to follow
  a linear trend, and an exponential is locally linear — median error −0.3%,
  worst case 1–4%. The notebook's `porder=9` was inert: its `WhittakerSmooth`
  ignored the argument and always built a first-difference matrix.
- **λ is set from a timescale**, via `lambda_from_timescale(tau_s, fs)`. The
  raw penalty is not scale-free, so a literal like 25000 means different things
  at 40 and 80 Hz and in a 2-minute versus a 30-minute recording.
- **dF/F = (F − F0)/F0 on the raw trace.** The notebook subtracted the baseline
  and then divided by the mean of the *result* — a near-zero residual of
  arbitrary sign — and the subsequent z-score erased whatever scale that
  produced.
- **Lasso alpha is chosen by blocked CV.** Scoring alphas by training MSE
  always returns the smallest, since training error falls monotonically with
  the penalty. Blocked (contiguous) folds matter because photometry residuals
  are autocorrelated and a shuffled split leaks across fold boundaries. Note
  that alpha is absolute while dF/F is O(0.01): a literal inherited from
  raw-intensity work silently zeroes the correction.
- **Isosbestic predictors default to the same fiber.** Using all fibers
  captures shared motion better, but any real signal bleeding into the
  isosbestic is then subtracted too — which for cross-region work removes
  exactly the shared variance of interest.
- **Smoothing is off by default.** The notebook's 17–29 sample Savitzky-Golay
  window is 425–725 ms at 40 Hz: harmless for bout-scale social signals,
  distorting for onset latency within a 1.5 s peri-stimulus window.
