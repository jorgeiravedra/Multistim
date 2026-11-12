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
| `matio` | reads `.mat` v7 and v7.3; parses `*_bcam.mat` (`bcl` cell array) and the stim GUI log (`pl`); SHA-256 provenance on every file |
| `metadata` | `<animal_id>.animal.yaml` implant registry — the fiber→region map, validated; session/epoch/LED schedule schema |
| `discovery` | walks a directory, groups files into sessions, binds each to its animal record and the calibration files that record names |
| `events` | camera triggers, LED state from DAQ **and** from image brightness with agreement scoring, laser pulse detection, laser→ROI assignment by timestamp |

## Usage sketch

```python
from multifiber import metadata, discovery, matio, events

registry = metadata.load_registry(r"C:\...\Stim Rig Controls\animals")
sessions = discovery.discover_sessions(r"C:\...\Stim Rig Controls", registry)
for s in sessions:
    print(s.summary())

s = next(x for x in sessions if x.ok)
log    = matio.load_bcam(s.bcam)
frames = events.detect_camera_frames(log)
pulses = events.detect_laser_pulses(log)                   # ('laser_a',) non-burst
assign = events.assign_laser_ids(pulses, matio.load_stim_log(s.stim_log))
assert assign.ok, assign.diagnostics
```

## Assumptions that must be confirmed against the rig

These are inferred from the legacy MATLAB script, not from documentation.
They are all surfaced at runtime (`BcamLog.provenance`, `LaserPulses.params`)
rather than hidden.

- `BCAM_CHANNEL_MAP`: `bcl{3..8}` = daq_time, ham_cam, flir_cam, led, laser_a,
  laser_b. `{7}`/`{8}` roles are a guess; the burst branch used `{7}.*{8}`,
  the non-burst branch `{7}` alone — and applied the *same* 0.07 threshold to
  both, which cannot be right for two differently scaled traces.
- DAQ sample rate is estimated from `daq_time`, not read from the file. It
  bounds the timing resolution of every event (`params['timing_resolution_s']`).
- `min_width_s = 0.0015` is the legacy value: 30% of a nominal 5 ms pulse.
  Check the width histogram before trusting it.
- Camera trigger threshold 2.0 V, laser threshold 0.07, LED 2 clusters.

## Design decisions worth knowing

- **The animal record is the region map.** Fiber index (1-based, as produced by
  the calibration GUI and used as the stim GUI's laser ID) is the only join
  between the laser log and the photometry channels. Non-contiguous indices are
  rejected; `histology: fail|pending` excludes a fiber by default.
- **Calibration is bound by name, never by date proximity.** A session uses the
  `TranslatedCoords_*`/`mids_*` its animal record names; a date mismatch between
  the two is an error, since the calibration GUI writes them as a pair.
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

## Not yet built

`extract` (ROI traces from `.avi`), `preprocess` (blanking → airPLS F0 → ΔF/F →
isosbestic regression), `align` (peri-event epoching), `qc`, `io`, GUI.
