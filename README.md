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
