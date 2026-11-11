# multifiber

Extraction and preprocessing for the multi-site stim-record photometry rig.

## Install (Windows, conda)

    conda env create -f environment.yml
    conda activate multifiber
    pip install -e .
    pytest -q tests/

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
