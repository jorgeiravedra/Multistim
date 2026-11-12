"""
Tests for multifiber.events / matio / metadata / discovery.

Fixtures are synthetic DAQ logs with known ground truth, constructed to
reproduce the specific failure modes the legacy pipeline could not detect:
a mid-recording LED parity flip, a non-chronological stim log, a double-
triggered laser command, and a dropped camera frame.

Run:  pytest -q tests/test_events.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import datetime as dt
import numpy as np
import pytest
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multifiber import events as ev  # noqa: E402
from multifiber import matio, metadata, discovery, calibration  # noqa: E402

RNG = np.random.default_rng(20261023)
DAQ_RATE = 2000.0  # Hz, placeholder until the rig value is confirmed


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def make_daq(
    duration_s: float = 20.0,
    camera_fps: float = 80.0,
    rate_hz: float = DAQ_RATE,
    laser_times: np.ndarray | None = None,
    laser_width_s: float = 0.005,
    laser_amplitude: float = 0.12,
):
    """Build a synthetic bcam cell array with known frame and laser timing."""
    n = int(duration_s * rate_hz)
    t = np.arange(n) / rate_hz

    # Camera: 1-sample-wide rising pulses at the frame period.
    ham = np.zeros(n)
    frame_t = np.arange(0, duration_s, 1.0 / camera_fps)
    frame_idx = np.round(frame_t * rate_hz).astype(int)
    frame_idx = frame_idx[frame_idx < n - 2]
    for i in frame_idx:
        ham[i:i + 2] = 5.0

    # LED line: alternates 1 V / 3 V per frame, held until the next frame.
    led = np.zeros(n)
    for k, i in enumerate(frame_idx):
        stop = frame_idx[k + 1] if k + 1 < frame_idx.size else n
        led[i:stop] = 1.0 if k % 2 == 0 else 3.0

    # laser_cmd: analog command voltage, high for the whole commanded pulse.
    # laser_gate: digital line, chopped at the camera rate (the rig's
    # "LASER BETWEEN CAMERA FRAMES ONLY" mode) so that one command yields many
    # gate transitions.
    laser_a = np.zeros(n)
    laser_b = np.zeros(n)
    if laser_times is not None:
        chop = int(round(rate_hz / camera_fps))
        for lt in laser_times:
            i0 = int(round(lt * rate_hz))
            i1 = int(round((lt + laser_width_s) * rate_hz))
            laser_a[i0:i1] = laser_amplitude
            for c in range(i0, i1, chop):
                laser_b[c:min(c + max(chop // 4, 1), i1)] = 1.0

    flir = np.zeros(n)
    bcl = np.empty((1, 8), dtype=object)
    bcl[0, 0] = np.array([[0.0]])
    bcl[0, 1] = np.array([[0.0]])
    bcl[0, 2] = t.reshape(-1, 1)
    bcl[0, 3] = ham.reshape(-1, 1)
    bcl[0, 4] = flir.reshape(-1, 1)
    bcl[0, 5] = led.reshape(-1, 1)
    bcl[0, 6] = laser_a.reshape(-1, 1)
    bcl[0, 7] = laser_b.reshape(-1, 1)
    return bcl, frame_idx, t


def write_bcam(tmp_path: Path, bcl, stem="sess01") -> Path:
    p = tmp_path / f"{stem}_bcam.mat"
    savemat(p, {"bcl": bcl})
    return p


# --------------------------------------------------------------------------- #
# matio
# --------------------------------------------------------------------------- #

def test_load_bcam_roundtrip(tmp_path):
    bcl, frame_idx, t = make_daq()
    log = matio.load_bcam(write_bcam(tmp_path, bcl))
    assert log.n_samples == t.size
    assert log.rate_hz == pytest.approx(DAQ_RATE, rel=1e-6)
    assert set(log.channels) == {"ham_cam", "aux_analog", "led",
                                 "laser_cmd", "laser_gate"}
    assert log.provenance["sha256"]
    assert not log.provenance["warnings"]


def test_load_bcam_reports_wrong_cell_count(tmp_path):
    bcl, _, _ = make_daq()
    short = bcl[:, :6]
    log = matio.load_bcam(write_bcam(tmp_path, short, stem="short"))
    assert any("cell array has 6" in w for w in log.provenance["warnings"])


def test_stim_log_is_sorted_chronologically(tmp_path):
    """The legacy sort() preserved input order; blocks out of time order
    permuted ROI identity. load_stim_log must return time-sorted rows."""
    # two sweeps written out of chronological order
    sweep_a = np.column_stack([np.array([3, 3]), np.zeros((2, 3)),
                               np.array([10.0, 20.0])])
    sweep_b = np.column_stack([np.array([7, 7]), np.zeros((2, 3)),
                               np.array([5.0, 15.0])])
    pl = np.empty((1, 4), dtype=object)
    pl[0, 0] = np.array([[0.0]])
    pl[0, 1] = np.array([[0.0]])
    pl[0, 2] = sweep_a
    pl[0, 3] = sweep_b
    p = tmp_path / "sess01.mat"
    savemat(p, {"pl": pl})

    out = matio.load_stim_log(p)
    assert np.all(np.diff(out[:, 1]) > 0)
    np.testing.assert_array_equal(out[:, 0].astype(int), [7, 3, 7, 3])


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #

def test_camera_frame_detection_and_rate(tmp_path):
    bcl, frame_idx, _ = make_daq(camera_fps=80.0)
    log = matio.load_bcam(write_bcam(tmp_path, bcl))
    frames = ev.detect_camera_frames(log)
    assert frames.n == frame_idx.size
    rep = ev.reconcile_frame_count(frames, frame_idx.size)
    assert rep["offset"] == 0
    assert rep["estimated_fps"] == pytest.approx(80.0, rel=1e-3)


def test_initial_high_exposure_is_counted(tmp_path):
    """The fixture's first exposure begins at DAQ sample 0, where diff() sees
    no rising edge. Missing it shifts every frame index by one."""
    bcl, frame_idx, _ = make_daq()
    log = matio.load_bcam(write_bcam(tmp_path, bcl, stem="high0"))
    with_initial = ev.detect_camera_frames(log, include_initial_high=True)
    without = ev.detect_camera_frames(log, include_initial_high=False)
    assert with_initial.starts_high is True
    assert with_initial.initial_included is True
    assert with_initial.n == without.n + 1
    assert with_initial.n == frame_idx.size


def test_reconcile_reports_offset_without_trimming(tmp_path):
    bcl, frame_idx, _ = make_daq()
    log = matio.load_bcam(write_bcam(tmp_path, bcl))
    frames = ev.detect_camera_frames(log)
    rep = ev.reconcile_frame_count(frames, frame_idx.size - 2)
    assert rep["offset"] == 2
    assert rep["within_tolerance"] is False


# --------------------------------------------------------------------------- #
# LED
# --------------------------------------------------------------------------- #

def test_led_daq_and_brightness_agree_on_clean_data(tmp_path):
    bcl, frame_idx, _ = make_daq()
    log = matio.load_bcam(write_bcam(tmp_path, bcl))
    frames = ev.detect_camera_frames(log)

    daq = ev.led_state_from_daq(log, frames)
    # Image brightness: alternating, with bleaching and noise.
    n = frames.n
    base = 1000 * np.exp(-np.arange(n) / 5000)
    inten = base + np.where(np.arange(n) % 2 == 0, 0.0, 200.0)
    inten += RNG.normal(0, 5, n)
    img = ev.led_state_from_brightness(inten)

    rep = ev.compare_led_assignments(daq, img)
    assert rep["agreement"] == 1.0
    assert rep["cohen_kappa"] == pytest.approx(1.0)
    assert rep["suspected_parity_flip"] is False


def test_brightness_relative_mode_survives_a_dropped_frame():
    """A dropped frame flips a fixed-parity assignment for the remainder of the
    recording; relative classification must localise the damage instead."""
    n = 400
    inten = np.where(np.arange(n) % 2 == 0, 100.0, 300.0)
    inten = np.delete(inten, 200)  # drop one frame -> parity inverts after it
    img = ev.led_state_from_brightness(inten)

    fixed_parity = (np.arange(inten.size) % 2 == 1).astype(int)
    n_bad_fixed = int(np.sum(img.state != fixed_parity))
    assert n_bad_fixed > 0.4 * inten.size  # fixed parity is wrong post-drop

    # relative labelling: bright frames are still bright
    truth = (inten > 200).astype(int)
    assert float(np.mean(img.state == truth)) > 0.99


def test_compare_flags_parity_flip(tmp_path):
    bcl, frame_idx, _ = make_daq()
    log = matio.load_bcam(write_bcam(tmp_path, bcl))
    frames = ev.detect_camera_frames(log)
    daq = ev.led_state_from_daq(log, frames)

    flipped = daq.state.copy()
    half = flipped.size // 2
    flipped[half:] = 1 - flipped[half:]
    other = ev.LedAssignment("synthetic", flipped, daq.values, 2, 10.0)

    rep = ev.compare_led_assignments(daq, other)
    assert rep["suspected_parity_flip"] is True
    assert rep["flip_model_changepoint"] == half
    assert rep["flip_model_accuracy"] == pytest.approx(1.0)
    assert rep["first_mismatch_frame"] == half
    assert rep["agreement"] == pytest.approx(0.5, abs=0.02)


# --------------------------------------------------------------------------- #
# Laser
# --------------------------------------------------------------------------- #

def test_laser_pulse_detection(tmp_path):
    true_t = np.array([2.0, 4.0, 6.0, 8.0])
    bcl, _, _ = make_daq(laser_times=true_t)
    log = matio.load_bcam(write_bcam(tmp_path, bcl, stem="laser"))
    p = ev.detect_laser_pulses(log)
    assert p.n == 4
    np.testing.assert_allclose(p.onset_t, true_t, atol=2 / DAQ_RATE)
    assert np.allclose(p.width_s, 0.005, atol=2 / DAQ_RATE)
    # command amplitude recovered, so it can be cross-checked against the
    # per-region voltages in intensity_mat
    assert np.allclose(p.amplitude_v, 0.12, atol=1e-9)


def test_laser_double_trigger_resolution(tmp_path):
    """A 1 ms glitch 3 ms before a real 5 ms pulse must not become an event.
    double_gap_s is off by default, so it is enabled explicitly here."""
    bcl, _, t = make_daq(duration_s=10.0)
    rate = DAQ_RATE
    la = np.zeros(t.size)  # replaces laser_cmd below

    def put(t0, width):
        la[int(t0 * rate):int((t0 + width) * rate)] = 1.0

    put(2.000, 0.001)   # glitch, below min_width
    put(2.004, 0.005)   # real
    put(5.000, 0.004)   # real
    put(5.006, 0.002)   # double, shorter -> dropped by keep_longer
    bcl[0, 6] = la.reshape(-1, 1)

    log = matio.load_bcam(write_bcam(tmp_path, bcl, stem="dbl"))
    p = ev.detect_laser_pulses(log, gate_channel=None, threshold=0.5,
                               double_gap_s=0.01)
    assert p.dropped_short == 1
    assert p.dropped_doubles == 1
    assert p.n == 2
    np.testing.assert_allclose(p.onset_t, [2.004, 5.000], atol=2 / rate)


def test_assign_laser_ids_is_robust_to_unsorted_log(tmp_path):
    """The failure the legacy code could not detect: stim-log blocks appended
    out of chronological order, paired with onsets by row index."""
    true_t = np.array([2.0, 4.0, 6.0, 8.0])
    true_ids = np.array([5, 11, 2, 9])
    bcl, _, _ = make_daq(laser_times=true_t)
    log = matio.load_bcam(write_bcam(tmp_path, bcl, stem="ids"))
    pulses = ev.detect_laser_pulses(log)

    # log in the wrong order, on a clock offset by +100 s
    perm = np.array([2, 0, 3, 1])
    stim_log = np.column_stack([true_ids[perm], true_t[perm] + 100.0])

    a = ev.assign_laser_ids(pulses, stim_log)
    assert a.ok
    np.testing.assert_array_equal(a.roi_id, true_ids)
    assert a.clock_offset_s == pytest.approx(-100.0, abs=0.01)
    assert abs(a.diagnostics["residual_max_abs_s"]) < 0.01


def test_assign_laser_ids_reports_missing_events(tmp_path):
    true_t = np.array([2.0, 4.0, 6.0, 8.0])
    bcl, _, _ = make_daq(laser_times=true_t)
    log = matio.load_bcam(write_bcam(tmp_path, bcl, stem="miss"))
    pulses = ev.detect_laser_pulses(log)

    # one commanded pulse the DAQ never saw
    stim_log = np.column_stack([[1, 2, 3, 4, 5],
                                [2.0, 4.0, 6.0, 8.0, 9.5]])
    a = ev.assign_laser_ids(pulses, stim_log)
    assert a.n_unmatched_log == 1
    assert a.n_unmatched_pulses == 0
    assert not a.ok


def test_events_to_frames(tmp_path):
    frame_t = np.arange(0, 1.0, 0.0125)  # 80 Hz
    evt = np.array([0.0126, 0.0249, -0.5])
    pre = ev.events_to_frames(evt, frame_t, side="preceding")
    assert pre[0] == 1 and pre[1] == 1 and pre[2] == -1
    near = ev.events_to_frames(evt[:2], frame_t, side="nearest")
    assert near[0] == 1 and near[1] == 2


# --------------------------------------------------------------------------- #
# Metadata / discovery
# --------------------------------------------------------------------------- #

ANIMAL_YAML = """
animal_id: derek
genotype: Vgat-Cre;CD1
sex: M
fibers:
  - {index: 1, region: VMHvl, hemisphere: R, histology: pass}
  - {index: 2, region: AH, hemisphere: R, histology: fail}
  - {index: 3, region: PrL, hemisphere: R, histology: pass}
"""


def _write_animal(tmp_path: Path, text: str = ANIMAL_YAML) -> Path:
    p = tmp_path / "derek.animal.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_animal_record_and_region_map(tmp_path):
    a = metadata.load_animal(_write_animal(tmp_path))
    assert a.implant.n_fibers == 3
    assert a.implant.ordered_labels() == ["VMHvl_R", "AH_R", "PrL_R"]
    # histology: fail -> excluded unless overridden
    assert a.implant.region_map(included_only=True) == {1: "VMHvl_R", 3: "PrL_R"}
    assert a.source_sha256


def test_non_contiguous_fiber_indices_rejected(tmp_path):
    bad = ANIMAL_YAML.replace("{index: 3,", "{index: 5,")
    with pytest.raises(ValueError, match="contiguous"):
        metadata.load_animal(_write_animal(tmp_path, bad))


def _make_session(session_dir, stem="derek_s1"):
    session_dir.mkdir(parents=True, exist_ok=True)
    bcl, _, _ = make_daq(duration_s=2.0)
    savemat(session_dir / f"{stem}_bcam.mat", {"bcl": bcl})
    (session_dir / f"{stem}.avi").write_bytes(b"\x00")


def _make_cal(cal_dir, stamps):
    cal_dir.mkdir(parents=True, exist_ok=True)
    for st in stamps:
        savemat(cal_dir / f"TranslatedCoords_{st}.mat",
                {"translated_coords": np.ones((3, 3))})
        savemat(cal_dir / f"mids_{st}.mat", {"mids": np.zeros((1, 2))})
        savemat(cal_dir / f"cal_mtx_{st}.mat", {"all_points": np.ones((4, 4))})
        savemat(cal_dir / f"lsr_ints_{st}.mat", {"intensity_mat": np.ones((12, 2))})


def test_discovery_binds_animal_and_calibration(tmp_path):
    _write_animal(tmp_path)
    reg = metadata.load_registry(tmp_path)
    sess = tmp_path / "testing" / "101926"
    cal = tmp_path / "multistim"
    _make_session(sess, "derek_10Hzburst_testmpoa")
    _make_cal(cal, ["100726", "090625"])

    sessions = discovery.discover_sessions(sess, reg, calibration_dir=cal)
    hit = [s for s in sessions if s.stem == "derek_10Hzburst_testmpoa"]
    assert len(hit) == 1
    s = hit[0]
    assert s.ok, s.problems
    assert s.animal.animal_id == "derek"
    # most recent stamp at or before 2026-10-19
    assert s.calibration.path("translated_coords").name == "TranslatedCoords_100726.mat"
    assert s.calibration.session_date_source.startswith("path MMDDYY")
    assert s.file_hashes()["calibration/translated_coords"]
    assert "video" in s.file_hashes()


def test_calibration_picks_latest_before_session_not_nearest(tmp_path):
    """A calibration generated after the recording cannot have been used, even
    if it is closer in time."""
    cal = tmp_path / "multistim"
    _make_cal(cal, ["100726", "102026"])   # 7 Oct and 20 Oct
    cs = calibration.resolve_calibration(
        cal, session_date=__import__("datetime").date(2026, 10, 19))
    assert cs.choices["translated_coords"].stamp == "100726"
    assert cs.choices["translated_coords"].newer_available == "102026"
    assert cs.choices["translated_coords"].age_days == 12


def test_calibration_records_runner_up_and_hash(tmp_path):
    cal = tmp_path / "multistim"
    _make_cal(cal, ["090625", "100726"])
    cs = calibration.resolve_calibration(
        cal, session_date=__import__("datetime").date(2026, 10, 19))
    c = cs.choices["translated_coords"]
    assert c.runner_up == "090625"
    assert c.source == "resolved"
    assert len(c.sha256) == 64
    prov = cs.provenance()
    assert prov["files"]["mids"]["stamp"] == "100726"


def test_calibration_override_wins_and_is_not_date_checked(tmp_path):
    cal = tmp_path / "multistim"
    _make_cal(cal, ["090625", "100726"])
    cs = calibration.resolve_calibration(
        cal,
        session_date=__import__("datetime").date(2026, 10, 19),
        overrides={"translated_coords": "TranslatedCoords_090625.mat",
                   "mids": "mids_090625.mat"},
    )
    assert cs.choices["translated_coords"].stamp == "090625"
    assert cs.choices["translated_coords"].source == "override"
    assert cs.ok, cs.problems


def test_calibration_split_pair_is_an_error(tmp_path):
    cal = tmp_path / "multistim"
    _make_cal(cal, ["090625"])
    # a newer translated_coords with no matching mids
    savemat(cal / "TranslatedCoords_100726.mat",
            {"translated_coords": np.ones((3, 3))})
    cs = calibration.resolve_calibration(
        cal, session_date=__import__("datetime").date(2026, 10, 19))
    assert any("different dates" in p for p in cs.problems)


def test_calibration_all_candidates_postdate_session(tmp_path):
    cal = tmp_path / "multistim"
    _make_cal(cal, ["110126"])
    cs = calibration.resolve_calibration(
        cal, session_date=__import__("datetime").date(2026, 10, 19))
    assert not cs.ok
    assert any("post-date" in p for p in cs.problems)


def test_session_date_from_dirname_prefers_deepest(tmp_path):
    from pathlib import Path as _P
    d = calibration.session_date_from_dirname(
        _P(r"F:/Polina/010122/stimcohort/testing/101926/derek_s1.avi"))
    assert d == __import__("datetime").date(2026, 10, 19)


def test_discovery_flags_unknown_animal(tmp_path):
    bcl, _, _ = make_daq(duration_s=2.0)
    savemat(tmp_path / "unknownmouse_s1_bcam.mat", {"bcl": bcl})
    (tmp_path / "unknownmouse_s1.avi").write_bytes(b"\x00")
    s = [x for x in discovery.discover_sessions(tmp_path, {})
         if x.stem == "unknownmouse_s1"][0]
    assert not s.ok
    assert any("no animal record" in p for p in s.problems)




# --------------------------------------------------------------------------- #
# .mat version sniffing
# --------------------------------------------------------------------------- #

def _matlab_header(version: int, endian: bytes = b"IM") -> bytes:
    """Synthesise the 128-byte MATLAB header: 116 text + 8 subsys + 2 ver + 2 endian."""
    text = b"MATLAB 5.0 MAT-file, Platform: PCWIN64".ljust(116, b" ")
    subsys = b"\x00" * 8
    ver = version.to_bytes(2, "little" if endian == b"IM" else "big")
    return text + subsys + ver + endian


def test_v73_detected_despite_offset_512_signature(tmp_path):
    """A v7.3 file is not a bare HDF5 file: MATLAB writes a 128-byte header
    first and the HDF5 superblock signature lands at offset 512."""
    p = tmp_path / "v73.mat"
    body = bytearray(b"\x00" * 512)
    body[:128] = _matlab_header(0x0200)
    p.write_bytes(bytes(body) + b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    assert matio._is_hdf5(p) is True


def test_v73_detected_from_version_field_alone(tmp_path):
    p = tmp_path / "v73b.mat"
    p.write_bytes(_matlab_header(0x0200) + b"\x00" * 64)
    assert matio._is_hdf5(p) is True


def test_v7_not_mistaken_for_hdf5(tmp_path):
    p = tmp_path / "v7.mat"
    p.write_bytes(_matlab_header(0x0100) + b"\x00" * 64)
    assert matio._is_hdf5(p) is False


def test_real_v7_file_roundtrips_as_v7(tmp_path):
    bcl, _, _ = make_daq(duration_s=1.0)
    p = write_bcam(tmp_path, bcl, stem="sniff")
    assert matio._is_hdf5(p) is False
    assert matio.load_bcam(p).n_samples > 0


def test_bare_hdf5_signature_at_zero(tmp_path):
    p = tmp_path / "bare.h5"
    p.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 128)
    assert matio._is_hdf5(p) is True


def test_non_numeric_mapped_channel_raises_clearly(tmp_path):
    """If the channel map points at a cell that is not numeric -- e.g. the
    datetimes in bcl{1:2} -- say so, rather than failing inside numpy."""
    bcl, _, _ = make_daq(duration_s=1.0)
    p = write_bcam(tmp_path, bcl, stem="badmap")
    bad_map = dict(matio.BCAM_CHANNEL_MAP)
    bad_map[1] = "daq_time"     # cell 1 is a datetime placeholder, not a trace
    bad_map.pop(3)
    try:
        matio.load_bcam(p, channel_map=bad_map)
    except (TypeError, ValueError, KeyError):
        pass
    else:
        # cell 1 in the fixture is a 1x1 double, so this map is loadable;
        # assert only that it did not silently produce a bogus timebase
        log = matio.load_bcam(p, channel_map=bad_map)
        assert log.n_samples <= 1


# --------------------------------------------------------------------------- #
# Channel profiling
# --------------------------------------------------------------------------- #

def test_profile_identifies_pulse_trains(tmp_path):
    bcl, _, _ = make_daq(duration_s=20.0, camera_fps=80.0,
                         laser_times=np.array([2.0, 4.0, 6.0]))
    log = matio.load_bcam(write_bcam(tmp_path, bcl, stem="prof"))
    prof = matio.profile_channels(log)

    assert prof.loc["ham_cam", "dom_freq_hz"] == pytest.approx(80.0, abs=1.0)
    assert prof.loc["ham_cam", "edge_rate_hz"] == pytest.approx(80.0, abs=1.0)
    # LED alternates per frame -> half the camera rate
    assert prof.loc["led", "dom_freq_hz"] == pytest.approx(40.0, abs=1.0)
    assert prof.loc["laser_cmd", "n_rising"] == 3
    assert "constant" in prof.loc["aux_analog", "guess"]


def test_raw_profile_finds_timebase_without_a_channel_map(tmp_path):
    """When the channel map is unknown, the timestamp cell is identifiable as
    the only strictly increasing full-length cell."""
    bcl, _, _ = make_daq(duration_s=20.0, camera_fps=80.0)
    p = write_bcam(tmp_path, bcl, stem="rawprof")
    prof = matio.load_bcam_raw_profile(p)

    assert prof.attrs["time_cell"] == 3
    assert prof.attrs["rate_hz"] == pytest.approx(2000.0, rel=1e-6)
    assert prof.loc["cell4", "dom_freq_hz"] == pytest.approx(80.0, abs=1.0)
    # the datetime cells are length-1, not full traces
    assert set(prof.attrs["skipped"]) == {1, 2}


def test_gate_chopping_does_not_inflate_event_count(tmp_path):
    """One 1 s command chopped at the camera rate is ONE stimulation, not 80.
    Counting gate edges is what turns 10 stimulations into 800."""
    true_t = np.array([2.0, 5.0])
    bcl, _, _ = make_daq(duration_s=10.0, camera_fps=80.0,
                         laser_times=true_t, laser_width_s=1.0,
                         laser_amplitude=0.12)
    log = matio.load_bcam(write_bcam(tmp_path, bcl, stem="chop"))

    p = ev.detect_laser_pulses(log)
    assert p.n == 2
    np.testing.assert_allclose(p.onset_t, true_t, atol=2 / DAQ_RATE)
    assert np.allclose(p.width_s, 1.0, atol=0.01)
    # ~80 chops per 1 s command at 80 fps
    assert np.all(p.n_gate_pulses >= 78) and np.all(p.n_gate_pulses <= 81)
    assert np.allclose(p.gate_duty, 0.25, atol=0.05)
    assert p.params["gate_channel"] == "laser_gate"

    # counting gate transitions instead would give ~160 "events"
    gate_edges = ev.detect_edges(log.time, log.channels["laser_gate"], 0.5)
    assert gate_edges.n > 100


def test_dead_command_channel_raises(tmp_path):
    """An unconnected input sits at a few mV of noise; thresholding it yields
    thousands of meaningless crossings. Fail loudly instead."""
    bcl, _, _ = make_daq(duration_s=5.0)       # no laser_times -> cmd is flat
    log = matio.load_bcam(write_bcam(tmp_path, bcl, stem="dead"))
    with pytest.raises(ValueError, match="unconnected or the channel map"):
        ev.detect_laser_pulses(log)


def test_check_channels_flags_dead_lines(tmp_path):
    bcl, _, _ = make_daq(duration_s=5.0, laser_times=np.array([1.0]))
    log = matio.load_bcam(write_bcam(tmp_path, bcl, stem="chk"))
    msgs = matio.check_channels(log)
    # aux_analog is flat in the fixture
    assert any(m.startswith("aux_analog") for m in msgs)
    assert not any(m.startswith("ham_cam") for m in msgs)


# --------------------------------------------------------------------------- #
# Trigger -> frame alignment
# --------------------------------------------------------------------------- #

def _edgeset_from_times(t):
    return ev.EdgeSet(rising_idx=np.arange(t.size), falling_idx=np.arange(t.size),
                      rising_t=np.asarray(t, float), falling_t=np.asarray(t, float),
                      threshold=2.0)


def test_alignment_drops_leading_settling_triggers():
    """Two anomalous intervals at the start are the camera settling after
    arming -- the thing the legacy frame_times(3:end) was compensating for."""
    t = np.concatenate([[0.0, 0.05, 0.10], 0.10 + np.arange(1, 101) * 0.0125])
    al = ev.align_triggers_to_frames(_edgeset_from_times(t), n_video_frames=t.size - 2)
    assert al.n_dropped_leading == 2
    assert al.n_unused_trailing == 0
    assert al.confidence == "medium"        # 4x the frame period: anomalous,
                                            # but not the decisive >=10x gap
    assert al.anchor_risk_ms == 0.0
    assert al.frame_t[0] == pytest.approx(0.10)


def test_no_leading_gap_means_no_anchor():
    """With no pre-acquisition exposures there is nothing to anchor on, so
    frame 0 is merely assumed to be trigger 0."""
    t = np.arange(105) * 0.0125
    al = ev.align_triggers_to_frames(_edgeset_from_times(t), n_video_frames=100)
    assert al.n_dropped_leading == 0
    assert al.n_unused_trailing == 5
    assert al.confidence == "low"
    assert al.anchor_risk_ms == pytest.approx(62.5, abs=0.1)


def test_unused_trailing_triggers_do_not_move_any_frame():
    """The mapping is anchored at the start, so leftover triggers at the end
    change the leftover count and nothing else."""
    t = np.concatenate([[0.0, 0.40], 1.30 + np.arange(200) * 0.0125])
    es = _edgeset_from_times(t)
    a2 = ev.align_triggers_to_frames(es, n_video_frames=t.size - 4)
    a5 = ev.align_triggers_to_frames(es, n_video_frames=t.size - 7)
    assert a2.n_unused_trailing == 2 and a5.n_unused_trailing == 5
    n = a5.n
    np.testing.assert_array_equal(a2.trigger_idx[:n], a5.trigger_idx[:n])
    np.testing.assert_allclose(a2.frame_t[:n], a5.frame_t[:n])


def test_alignment_rejects_more_frames_than_triggers():
    t = np.arange(50) * 0.0125
    with pytest.raises(ValueError, match="cannot be gained"):
        ev.align_triggers_to_frames(_edgeset_from_times(t), n_video_frames=60)


def test_alignment_matches_observed_session_shape():
    """derek_1000ms_testpfc: 8846 triggers, 2 long intervals at index 0 and 1,
    5 more triggers than frames."""
    n_trig = 8846
    t = np.arange(n_trig) * 0.0125
    t[1:] += 0.32          # the two pre-acquisition snapshots: gaps of
    t[2:] += 0.865         # 0.334 s and 0.878 s, i.e. 27x and 70x the period
    al = ev.align_triggers_to_frames(_edgeset_from_times(t),
                                     n_video_frames=n_trig - 5)
    assert al.n_dropped_leading == 2
    assert al.n_unused_trailing == 3
    assert al.n == n_trig - 5
    assert al.confidence == "high"          # snapshot gaps are decisive
    assert al.anchor_risk_ms == pytest.approx(37.5, abs=1.0)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

from multifiber import extract as ex  # noqa: E402


def _fake_calibration(n_roi=4, radius=20.0, spacing=60.0, mids=(10.0, 10.0)):
    """Fiber centres in full-frame coordinates; mids is subtracted to place
    them in the half image, as the acquisition convention does."""
    cx = mids[0] + 40 + spacing * np.arange(n_roi)
    cy = np.full(n_roi, mids[1] + 100.0)
    return np.column_stack([cx, cy, np.full(n_roi, radius)]), np.asarray(mids)


def test_masks_are_disjoint_and_average_correctly():
    tc, mids = _fake_calibration()
    m = ex.build_roi_masks(tc, mids, image_shape=(238, 1152), roi_fraction=0.75)
    assert m.n_roi == 4
    assert m.shape == (238, 576)
    # disjoint: every ROI pixel belongs to exactly one ROI
    assert m.pix_idx.size == int(m.roi_sizes.sum())
    assert np.unique(m.pix_idx).size == m.pix_idx.size
    # 0.75 of a radius-20 fiber -> area pi*15^2 ~ 707 px
    assert np.all(np.abs(m.roi_sizes - np.pi * 15 ** 2) < 60)

    # a frame where each ROI sits on a known constant
    frame = np.zeros((238, 1152), dtype=np.uint16)
    want = [10, 20, 30, 40]
    L = m.label_image()
    for i, v in enumerate(want, start=1):
        frame[:, :576][L == i] = v
    res = ex.extract_traces([frame], m)
    np.testing.assert_allclose(res.traces[0], want)


def test_overlapping_rois_are_rejected():
    """Shared pixels put the same photons in two regions, manufacturing
    between-region correlation that would read as a connectivity edge."""
    tc, mids = _fake_calibration(n_roi=2, radius=40.0, spacing=30.0)
    with pytest.raises(ValueError, match="shared between ROIs"):
        ex.build_roi_masks(tc, mids, image_shape=(238, 1152))


def test_roi_outside_half_is_rejected():
    tc, mids = _fake_calibration(n_roi=1)
    tc[0, 0] += 2000.0
    with pytest.raises(ValueError, match="no pixels"):
        ex.build_roi_masks(tc, mids, image_shape=(238, 1152))


def test_frame_intensity_tracks_led_alternation():
    """The per-frame ROI mean is what recovers LED state from brightness --
    the only route available, since the DAQ LED line is unconnected."""
    tc, mids = _fake_calibration(n_roi=3)
    m = ex.build_roi_masks(tc, mids, image_shape=(238, 1152))
    L = m.label_image()
    base = np.zeros((238, 1152), dtype=np.uint16)
    base[:, :576][L > 0] = 100

    frames = []
    for k in range(40):
        f = base.copy()
        if k % 2 == 1:                      # bright (470) frames
            f[:, :576][L > 0] = 300
        frames.append(f)
    res = ex.extract_traces(frames, m)

    led = ev.led_state_from_brightness(res.frame_intensity)
    truth = (np.arange(40) % 2 == 1).astype(int)
    assert float(np.mean(led.state == truth)) == 1.0


def test_extract_rejects_frame_count_mismatch():
    tc, mids = _fake_calibration(n_roi=2)
    m = ex.build_roi_masks(tc, mids, image_shape=(238, 1152))
    frames = [np.zeros((238, 1152), dtype=np.uint16) for _ in range(5)]
    with pytest.raises(ValueError, match="must match the frames actually"):
        ex.extract_traces(frames, m, n_frames=6)


def test_extract_rejects_wrong_frame_shape():
    tc, mids = _fake_calibration(n_roi=2)
    m = ex.build_roi_masks(tc, mids, image_shape=(238, 1152))
    with pytest.raises(ValueError, match="incompatible with masks"):
        ex.extract_traces([np.zeros((100, 1152))], m)


def test_right_half_offsets_centres():
    tc, mids = _fake_calibration(n_roi=2)
    left = ex.build_roi_masks(tc, mids, image_shape=(238, 1152), half="left")
    right = ex.build_roi_masks(tc, mids, image_shape=(238, 1152), half="right")
    np.testing.assert_allclose(right.centres[:, 0] - left.centres[:, 0], 576)
    np.testing.assert_allclose(right.roi_sizes, left.roi_sizes)


# --------------------------------------------------------------------------- #
# Writer-side frame drops
# --------------------------------------------------------------------------- #

def test_parity_break_locates_a_writer_side_drop():
    """A frame lost at the writer leaves two consecutive frames in the same
    LED state. The DAQ interval stays clean, so this is the only detector."""
    n = 400
    inten = np.where(np.arange(n) % 2 == 0, 100.0, 300.0)
    inten = np.delete(inten, 200)
    led = ev.led_state_from_brightness(inten)
    br = ev.detect_parity_breaks(led)
    assert br["n_breaks"] == 1
    # frames 199 and 200 (post-deletion) share a state, so the gap follows 199
    assert br["break_after"] == [199]


def test_clean_alternation_has_no_parity_breaks():
    inten = np.where(np.arange(500) % 2 == 0, 100.0, 300.0)
    led = ev.led_state_from_brightness(inten)
    assert ev.detect_parity_breaks(led)["n_breaks"] == 0


def test_alignment_with_drop_shifts_only_after_the_drop():
    """Without the drop, every frame after it maps to the wrong trigger."""
    n_trig = 1000
    t = np.arange(n_trig) * 0.0125
    t[1:] += 0.32
    t[2:] += 0.865                                 # two leading snapshots
    es = _edgeset_from_times(t)
    n_video = n_trig - 2 - 2 - 1                   # lead 2, trail 2, 1 drop

    al = ev.align_triggers_to_frames(es, n_video, drop_after_frames=[400])
    assert al.n_dropped_leading == 2
    assert al.n_unused_trailing == 2
    assert al.n_writer_drops == 1
    # before the drop: contiguous from trigger 2
    assert al.trigger_idx[0] == 2
    assert al.trigger_idx[400] == 402
    # after it: one trigger skipped
    assert al.trigger_idx[401] == 404
    assert np.all(np.diff(al.trigger_idx) >= 1)

    # ignoring the drop charges it to the end and misplaces later frames
    naive = ev.align_triggers_to_frames(es, n_video)
    assert naive.n_unused_trailing == 3
    assert naive.trigger_idx[401] == 403           # off by one vs the truth


def test_leading_scan_spans_non_contiguous_snapshots():
    """Two snapshots separated by a normal interval: a contiguous-run scan
    would stop at the first and leave an uncounted trigger."""
    t = np.concatenate([[0.0, 0.40], [0.4125], [1.30],
                        1.30 + np.arange(1, 101) * 0.0125])
    al = ev.align_triggers_to_frames(_edgeset_from_times(t),
                                     n_video_frames=t.size - 3)
    assert al.n_dropped_leading == 3


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

from multifiber import preprocess as pp  # noqa: E402


def _synth_session(n_frames=4000, fps=80.0, n_roi=3, f_base=1000.0,
                   bleach_tau=6000.0, seed=0):
    """Interleaved 415/470 frames with bleaching, a transient, and motion."""
    rng = np.random.default_rng(seed)
    frame_t = np.arange(n_frames) / fps
    led = (np.arange(n_frames) % 2).astype(int)        # 1 = signal (brighter)
    bleach = np.exp(-np.arange(n_frames) / bleach_tau)
    motion = 20.0 * np.sin(2 * np.pi * 0.3 * frame_t)

    traces = np.empty((n_frames, n_roi))
    for j in range(n_roi):
        base = f_base * (1 + 0.1 * j) * bleach
        resp = np.zeros(n_frames)
        for onset in (1500, 2500):
            k = np.arange(n_frames) - onset
            resp += np.where(k >= 0, 300.0 * np.exp(-k / 80.0), 0.0)
        sig = base + motion + resp
        iso = 0.4 * f_base * (1 + 0.1 * j) * bleach + motion
        traces[:, j] = np.where(led == 1, sig, iso)
    traces += rng.normal(0, 2.0, traces.shape)
    return traces, led, frame_t


def test_whittaker_honours_difference_order():
    """The notebook's version ignored this argument, so every call documented
    as higher order was in fact first order."""
    x = np.cumsum(np.random.default_rng(0).normal(0, 1, 300))
    w = np.ones_like(x)
    d1 = pp.whittaker_smooth(x, w, 100.0, differences=1)
    d2 = pp.whittaker_smooth(x, w, 100.0, differences=2)
    assert not np.allclose(d1, d2)


def _bleaching_with_transients(n=4000, tau_bleach=6000.0, amp=300.0, seed=0):
    rng = np.random.default_rng(seed)
    f0_true = 1000.0 * np.exp(-np.arange(n) / tau_bleach)
    resp = np.zeros(n)
    for onset in (1500, 2500):
        k = np.arange(n) - onset
        resp += np.where(k >= 0, amp * np.exp(-k / 80.0), 0.0)
    return f0_true + resp + rng.normal(0, 2.0, n), f0_true, resp


def test_dff_uses_f0_not_a_detrended_residual():
    """dF/F must divide by a real fluorescence level, so that amplitudes are
    comparable across fibers. Dividing a detrended trace by its own mean --
    a near-zero residual -- divides by noise."""
    x, f0_true, resp = _bleaching_with_transients()
    f0 = pp.estimate_f0(x, lambda_=pp.lambda_from_timescale(30.0, 40.0))
    dff = pp.compute_dff(x, f0)

    # the denominator is a fluorescence level, not ~0
    assert np.nanmin(f0) > 500.0
    # baseline periods sit at zero, transient peaks at amp / F0
    quiet = np.arange(200, 1400)
    assert abs(np.nanmedian(dff[quiet])) < 0.01
    peak = np.nanmax(dff[1500:1700])
    assert peak == pytest.approx(resp.max() / f0_true[1500], rel=0.15)


def test_second_difference_penalty_tracks_bleaching():
    """A first-difference penalty drives F0 toward a constant as lambda grows,
    so it cannot follow photobleaching and under-estimates F0 -- which
    inflates every dF/F by the same factor."""
    x, f0_true, _ = _bleaching_with_transients()
    lam = pp.lambda_from_timescale(30.0, 40.0, differences=2)
    err2 = np.abs(pp.airpls(x, lambda_=lam, differences=2) - f0_true) / f0_true
    err1 = np.abs(pp.airpls(x, lambda_=1e5, differences=1) - f0_true) / f0_true
    assert np.median(err2) < 0.01
    assert np.median(err1) > 0.10
    assert np.max(err2) < 0.05


def test_lambda_from_timescale_scales_with_rate():
    """The raw penalty is not scale-free: the same literal means different
    things at 40 and 80 Hz."""
    a = pp.lambda_from_timescale(60.0, 40.0, differences=2)
    b = pp.lambda_from_timescale(60.0, 80.0, differences=2)
    assert b / a == pytest.approx(2.0 ** 4)
    assert pp.lambda_from_timescale(30.0, 40.0) < a


def test_demux_splits_by_led_state():
    traces, led, frame_t = _synth_session()
    dm = pp.demux_channels(traces, led, frame_t)
    assert dm.signal.shape[0] + dm.isosbestic.shape[0] == traces.shape[0]
    # signal channel is the brighter one
    assert dm.signal.mean() > dm.isosbestic.mean()
    np.testing.assert_array_equal(dm.signal_frames % 2, 1)


def test_demux_rejects_mismatched_lengths():
    traces, led, frame_t = _synth_session(n_frames=100)
    with pytest.raises(ValueError, match="must be per-frame"):
        pp.demux_channels(traces, led[:-1], frame_t)


def test_epochs_derived_from_laser_times():
    ep = pp.derive_epochs_from_laser(
        onset_t=np.array([30.0, 40.0, 50.0]),
        offset_t=np.array([31.0, 41.0, 51.0]),
        t_start=0.0, t_end=112.0, guard_s=5.0)
    assert ep["baseline"] == (0.0, 25.0)
    assert ep["stim"] == (30.0, 51.0)
    assert ep["post"] == (56.0, 112.0)


def test_epochs_omit_empty_periods():
    """A session with no pre-stimulus period has no baseline key, rather than
    a zero-length one."""
    ep = pp.derive_epochs_from_laser(np.array([1.0]), np.array([2.0]),
                                     t_start=0.0, t_end=112.0, guard_s=5.0)
    assert "baseline" not in ep
    assert ep["stim"] == (1.0, 2.0)


def test_blanking_marks_and_pads():
    x = np.ones((100, 2))
    mask = np.zeros(100, dtype=bool)
    mask[50] = True
    out, used = pp.blank_frames(x, mask, pad=1)
    assert used[49] and used[50] and used[51]
    assert np.isnan(out[49:52]).all()
    assert not np.isnan(out[48]).any()


def test_downward_artifact_biases_f0_unless_blanked():
    """airPLS down-weights samples ABOVE the baseline, so it is robust to
    upward transients by construction -- but a downward excursion pulls F0
    directly, and only blanking removes it."""
    n = 2000
    x = np.full(n, 1000.0)
    x[1000:1060] = 200.0                          # a dropout
    f0_contaminated = pp.estimate_f0(x, lambda_=1e7)
    blanked, _ = pp.blank_frames(
        x.reshape(-1, 1), np.isin(np.arange(n), np.arange(1000, 1060)))
    f0_clean = pp.estimate_f0(blanked[:, 0], lambda_=1e7)

    assert np.min(f0_contaminated) < 950.0        # F0 dragged down
    assert np.min(f0_clean) == pytest.approx(1000.0, rel=0.02)


def test_blanking_keeps_the_artifact_out_of_dff():
    """Blanking protects the data, not just the baseline: a contaminated
    sample must not appear as a transient."""
    n = 2000
    x = np.full(n, 1000.0)
    x[1000:1010] = 50000.0
    mask = np.isin(np.arange(n), np.arange(1000, 1010))
    f0 = pp.estimate_f0(x, lambda_=1e7)
    assert np.nanmax(pp.compute_dff(x, f0)) > 10.0          # artifact present
    blanked, _ = pp.blank_frames(x.reshape(-1, 1), mask)
    assert np.nanmax(pp.compute_dff(blanked[:, 0], f0)) < 0.1


def test_fixed_alpha_can_silently_zero_the_regression():
    """Lasso's alpha is absolute, but dF/F is O(0.01), so a literal inherited
    from raw-intensity work shrinks every coefficient to zero and the
    correction silently does nothing. Hence 'cv' as the default."""
    rng = np.random.default_rng(3)
    n = 2000
    motion = 0.02 * np.sin(np.arange(n) / 40.0)
    y = motion + rng.normal(0, 0.002, n)
    _, big = pp.regress_isosbestic(y, motion.reshape(-1, 1), alpha=0.001)
    _, cv = pp.regress_isosbestic(y, motion.reshape(-1, 1), alpha="cv")
    assert big["r2_in_sample"] == pytest.approx(0.0, abs=1e-6)
    assert cv["r2_in_sample"] > 0.8


def test_regression_alpha_is_not_chosen_by_training_error():
    """Training MSE falls monotonically with the penalty, so scoring alphas on
    it always returns the smallest. Blocked CV must be able to do otherwise."""
    rng = np.random.default_rng(1)
    n = 2000
    iso = np.sin(np.arange(n) / 50.0)
    y = 0.8 * iso + rng.normal(0, 0.5, n)
    _, info = pp.regress_isosbestic(y, iso.reshape(-1, 1), alpha="cv")
    assert info["alpha_source"] == "blocked CV"
    assert info["alpha"] > 1e-4          # not pinned to the smallest candidate
    assert 0.0 < info["r2_in_sample"] < 1.0


def test_regression_removes_shared_motion():
    rng = np.random.default_rng(2)
    n = 3000
    motion = np.sin(np.arange(n) / 40.0)
    y = motion + rng.normal(0, 0.05, n)
    resid, info = pp.regress_isosbestic(y, motion.reshape(-1, 1), alpha=0.001)
    assert np.nanstd(resid) < 0.3 * np.nanstd(y)


def test_full_pipeline_recovers_transients():
    traces, led, frame_t = _synth_session()
    out = pp.preprocess_traces(
        traces, led, frame_t,
        laser_onset_t=np.array([frame_t[1500], frame_t[2500]]),
        laser_offset_t=np.array([frame_t[1540], frame_t[2540]]),
        labels=["A", "B", "C"], alpha="cv", baseline_tau_s=10.0)

    assert out.dff.shape[1] == 3
    assert set(out.epochs) == {"baseline", "stim", "post"}
    assert np.isfinite(out.dff).mean() > 0.95
    # the transient is recoverable above the pre-stimulus level
    onset = int(np.searchsorted(out.t, frame_t[1500]))
    pre = np.nanstd(out.dff[:onset - 50, 0])
    peak = np.nanmax(out.dff[onset:onset + 100, 0])
    assert peak > 5 * pre


def test_smoothing_is_off_by_default():
    traces, led, frame_t = _synth_session(n_frames=1200)
    out = pp.preprocess_traces(traces, led, frame_t, alpha=0.001,
                               baseline_tau_s=10.0)
    assert out.qc["smooth_window"] is None


# --------------------------------------------------------------------------- #
# Pipeline and IO
# --------------------------------------------------------------------------- #

from multifiber import io as mio          # noqa: E402
from multifiber import pipeline as pl     # noqa: E402


def test_config_hash_changes_with_any_setting():
    a = pl.PipelineConfig()
    b = pl.PipelineConfig(baseline_tau_s=30.0)
    assert a.hash() != b.hash()
    assert a.hash() == pl.PipelineConfig().hash()      # deterministic


def test_up_to_date_requires_both_inputs_and_config(tmp_path):
    """A changed parameter must force reprocessing even when the raw files are
    untouched, and vice versa."""
    man = {"session_id": "s1", "status": "ok", "config_hash": "abc",
           "inputs": {"sha256": {"video": "v1", "bcam": "b1"}}}
    mio.write_session(tmp_path, "s1", man)

    assert mio.is_up_to_date(tmp_path, "s1", {"video": "v1", "bcam": "b1"}, "abc")
    assert not mio.is_up_to_date(tmp_path, "s1", {"video": "v1", "bcam": "b1"}, "xyz")
    assert not mio.is_up_to_date(tmp_path, "s1", {"video": "v2", "bcam": "b1"}, "abc")
    assert not mio.is_up_to_date(tmp_path, "s1", {"video": "v1"}, "zzz")


def test_failed_session_is_not_treated_as_up_to_date(tmp_path):
    man = {"session_id": "s2", "status": "failed", "config_hash": "abc",
           "inputs": {"sha256": {"video": "v1"}}}
    mio.write_session(tmp_path, "s2", man)
    assert not mio.is_up_to_date(tmp_path, "s2", {"video": "v1"}, "abc")


def test_manifest_serialises_numpy_and_dataclasses(tmp_path):
    man = {
        "session_id": "s3", "status": "ok",
        "qc": {"fps": np.float64(39.9), "n": np.int64(8841),
               "arr": np.arange(3), "bad": np.float64("nan"),
               "flag": np.bool_(True)},
    }
    mio.write_session(tmp_path, "s3", man)
    back = mio.read_manifest(tmp_path, "s3")
    assert back["qc"]["n"] == 8841
    assert back["qc"]["arr"] == [0, 1, 2]
    assert back["qc"]["bad"] is None          # NaN is not valid JSON
    assert back["qc"]["flag"] is True


def test_failed_session_records_the_stage(tmp_path):
    """A failure must be isolated to its session and name the stage, so one
    bad file does not stop a batch and the cause is visible afterwards."""
    cand = discovery.SessionCandidate(stem="bad", root=tmp_path)
    cand.problems.append("no animal record matched this filename")
    r = pl.process_session(cand, tmp_path / "out")
    assert r.status == "failed"
    assert r.failed_stage == "setup"
    man = mio.read_manifest(tmp_path / "out", "bad")
    assert man["status"] == "failed"
    assert "unresolved problems" in man["error"]


def test_roi_count_mismatch_is_fatal(tmp_path):
    """The animal record's fiber count and the calibration's ROI count must
    agree: ROI order is what laser IDs index, so guessing permutes the
    connectivity matrix."""
    _write_animal(tmp_path)                      # 3 fibers
    reg = metadata.load_registry(tmp_path)
    cand = discovery.SessionCandidate(stem="x", root=tmp_path,
                                      animal=reg["derek"])
    with pytest.raises(ValueError, match="ROI order is what laser IDs index"):
        pl._bind_roi_labels(cand, n_roi=12)
    assert pl._bind_roi_labels(cand, n_roi=3) == ["VMHvl_R", "AH_R", "PrL_R"]


def test_summarise_outputs_lists_every_session(tmp_path):
    for sid, status in (("a", "ok"), ("b", "failed")):
        mio.write_session(tmp_path, sid, {
            "session_id": sid, "animal_id": "derek", "status": status,
            "qc": {"n_roi": 12, "n_laser_events": 10},
            "warnings": ["w1"],
        })
    df = mio.summarise_outputs(tmp_path)
    assert set(df["session_id"]) == {"a", "b"}
    assert set(df["status"]) == {"ok", "failed"}
    assert df["n_warnings"].tolist() == [1, 1]


def test_alpha_grid_kwarg_adapts_to_sklearn_version():
    """n_alphas was deprecated in scikit-learn 1.7 in favour of alphas taking
    an int; resolve by introspection rather than pinning a version."""
    from sklearn.linear_model import LassoCV

    kw = pp._alpha_grid_kwarg(LassoCV, 20)
    assert set(kw) <= {"n_alphas", "alphas"}
    assert list(kw.values()) in ([20], [])
    # whichever it is, it must actually be accepted
    LassoCV(**kw)


def test_cv_regression_runs_on_this_sklearn():
    rng = np.random.default_rng(7)
    n = 800
    iso = np.sin(np.arange(n) / 30.0)
    y = 0.5 * iso + rng.normal(0, 0.1, n)
    resid, info = pp.regress_isosbestic(y, iso.reshape(-1, 1), alpha="cv")
    assert np.isfinite(resid).all()
    assert info["alpha_source"] == "blocked CV"


def test_whittaker_survives_all_zero_weights():
    """airPLS drives the weight of every sample above the baseline to zero,
    and on a smooth monotone trace that can be all of them. A = W + lam*D'D
    is then exactly singular, and spsolve on a singular matrix is undefined
    behaviour in SuperLU -- on Windows it corrupts the heap and kills the
    process rather than raising."""
    n = 2000
    x = 1000.0 * np.exp(-np.arange(n) / 20000.0)
    lam = pp.lambda_from_timescale(60.0, 40.0, differences=2)
    z = pp.whittaker_smooth(x, np.zeros(n), lam, differences=2)
    assert np.isfinite(z).all()


def test_airpls_never_emits_an_all_zero_weight_vector():
    """Guard the invariant at its source as well as at the solver."""
    n = 1500
    x = 1000.0 * np.exp(-np.arange(n) / 9000.0)
    z = pp.airpls(x, lambda_=pp.lambda_from_timescale(60.0, 40.0), differences=2)
    assert np.isfinite(z).all()
    assert pp.MIN_WEIGHT > 0.0


def test_stiff_baseline_still_tracks_bleaching():
    """The stiff lambda that triggered the crash must also give a good fit."""
    n = 4420
    x = 1000.0 * np.exp(-np.arange(n) / 20000.0)
    lam = pp.lambda_from_timescale(60.0, 40.0, differences=2)
    z = pp.airpls(x, lambda_=lam, differences=2)
    err = np.abs(z - x) / x
    assert np.median(err) < 0.005
    assert np.max(err) < 0.02


def test_whittaker_rejects_non_finite_input():
    x = np.ones(100); x[50] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pp.whittaker_smooth(x, np.ones(100), 1e6, differences=2)


def test_whittaker_handles_degenerate_lengths():
    for n in (0, 1, 2, 3):
        z = pp.whittaker_smooth(np.ones(n), np.ones(n), 1e6, differences=2)
        assert z.size == n and np.isfinite(z).all()
    assert np.isfinite(pp.airpls(np.ones(3), differences=2)).all()
    assert np.isfinite(pp.airpls(np.zeros(500), differences=2)).all()


def test_banded_and_sparse_solvers_agree():
    """The Whittaker system is symmetric positive definite and banded, so a
    banded Cholesky (LAPACK pbsv) is the right solver; SuperLU is kept only
    as an escape hatch."""
    rng = np.random.default_rng(11)
    n = 3000
    x = 1000.0 * np.exp(-np.arange(n) / 20000.0) + rng.normal(0, 2.0, n)
    lam = pp.lambda_from_timescale(60.0, 40.0, differences=2)
    zb = pp.airpls(x, lambda_=lam, differences=2, solver="banded")
    zs = pp.airpls(x, lambda_=lam, differences=2, solver="sparse")
    assert np.max(np.abs(zb - zs)) / np.median(zb) < 1e-5


def test_banded_storage_matches_dense():
    from scipy.sparse import csc_matrix, diags, eye

    n, d = 50, 2
    rng = np.random.default_rng(5)
    w = rng.uniform(0.5, 1.5, n)
    E = eye(n, format="csc")
    D = E[1:] - E[:-1]
    D = D[1:] - D[:-1]
    A = csc_matrix(diags(w, 0, shape=(n, n)) + 10.0 * (D.T @ D))
    ab = pp._banded_from_sparse(A, d)
    dense = A.toarray()
    for i in range(n):
        for j in range(i, min(i + d + 1, n)):
            assert ab[d + i - j, j] == pytest.approx(dense[i, j])


def test_unknown_solver_rejected():
    with pytest.raises(ValueError, match="unknown solver"):
        pp.whittaker_smooth(np.ones(50), np.ones(50), 1e6, 2, solver="nope")


def test_exposure_window_controls_contamination():
    """With a gate pulse once per frame period, treating the whole period as
    exposed flags every frame by construction; a real exposure window does
    not."""
    fs = 40.0
    frame_t = np.arange(400) / fs
    gate_on_t = frame_t + 0.020          # 20 ms after each trigger
    all_flagged = pp.frames_contaminated_by_laser(frame_t, gate_on_t)
    assert all_flagged.all()
    realistic = pp.frames_contaminated_by_laser(frame_t, gate_on_t,
                                                exposure_s=0.0125)
    assert not realistic.any()


def test_warns_when_blanking_erases_the_stim_epoch(tmp_path):
    traces, led, frame_t = _synth_session(n_frames=4000)
    fs_sig = 40.0
    gate = np.arange(20.0, 30.0, 1.0 / fs_sig)     # a pulse every signal frame
    out = pp.preprocess_traces(
        traces, led, frame_t,
        laser_onset_t=np.array([20.0]), laser_offset_t=np.array([30.0]),
        gate_on_t=gate, alpha=0.001, baseline_tau_s=10.0)
    assert any("cannot be measured" in w for w in out.qc["warnings"])


def test_led_separation_grows_with_cleanliness():
    """The metric must be monotone in data quality. An earlier version divided
    by the POOLED MAD, which for a clean alternation equals the half-gap, so
    it pinned to exactly 2 however good the data were -- and exceeded 2 only
    when near-zero-contrast frames dragged the pooled MAD down, i.e. it was
    largest for the worst recordings."""
    n = 2000
    rng = np.random.default_rng(4)
    seps = []
    for noise in (50.0, 10.0, 1.0):
        inten = np.where(np.arange(n) % 2 == 0, 100.0, 300.0)
        inten = inten + rng.normal(0, noise, n)
        seps.append(ev.led_state_from_brightness(inten).separation)
    assert seps[0] < seps[1] < seps[2]
    assert seps[2] > 20.0          # near-noiseless data scores high, not 2


def test_led_separation_detects_a_bad_alternation():
    rng = np.random.default_rng(6)
    n = 2000
    inten = np.where(np.arange(n) % 2 == 0, 100.0, 105.0) + rng.normal(0, 40, n)
    assert ev.led_state_from_brightness(inten).separation < 4.0


def test_daq_led_uses_aligned_triggers():
    """The DAQ sees every exposure including pre-acquisition snapshots, so
    comparing trigger k against video frame k offsets the two sequences."""
    n = 600
    rate = 2000.0
    t = np.arange(int(2.0 * rate)) / rate
    led_trace = np.zeros(t.size)
    frame_idx = np.arange(0, t.size - 2, 25)
    for k, i in enumerate(frame_idx):
        stop = frame_idx[k + 1] if k + 1 < frame_idx.size else t.size
        led_trace[i:stop] = 1.0 if k % 2 == 0 else 3.0
    log = matio.BcamLog(time=t, channels={"led": led_trace}, rate_hz=rate)
    es = ev.EdgeSet(frame_idx, frame_idx, t[frame_idx], t[frame_idx], 0.5)

    full = ev.led_state_from_daq(log, es)
    # drop 3 leading triggers, as a real alignment does
    idx = np.arange(3, es.n)
    sub = ev.led_state_from_daq(log, es, trigger_idx=idx)
    assert sub.n_frames == es.n - 3
    np.testing.assert_array_equal(sub.state, full.state[3:])


def test_channel_is_live_ignores_a_single_spike():
    """Peak-to-peak lets one spike declare a dead channel live; its noise then
    clusters into two states that agree with anything at chance."""
    rng = np.random.default_rng(9)
    dead = 0.005 + rng.normal(0, 0.001, 5000)
    dead[1234] = 2.0                        # one spike
    assert float(np.max(dead) - np.min(dead)) > 0.05     # ptp says live
    assert not ev.channel_is_live(dead)                  # robust span says not
    real = np.where(np.arange(5000) % 2 == 0, 1.0, 3.0)
    assert ev.channel_is_live(real)


def test_gate_phase_confirms_the_interlock():
    """8.5 ms exposure in a 12.5 ms period leaves a 4 ms gap; a gate firing in
    that gap should register as not contaminating any frame."""
    fs = 80.0
    frame_t = np.arange(400) / fs
    gate_on_t = frame_t + 0.0095                 # 9.5 ms after each trigger
    g = ev.gate_phase_within_frame(frame_t, gate_on_t, exposure_s=0.0085)
    assert g["phase_ms"]["median"] == pytest.approx(9.5, abs=0.1)
    assert g["gap_ms"] == pytest.approx(4.0, abs=0.1)
    assert g["fraction_during_exposure"] == 0.0
    assert g["interlock_holds"] is True


def test_gate_phase_detects_an_unsynchronised_gate():
    rng = np.random.default_rng(12)
    fs = 80.0
    frame_t = np.arange(400) / fs
    gate_on_t = frame_t + rng.uniform(0, 1 / fs, frame_t.size)
    g = ev.gate_phase_within_frame(frame_t, gate_on_t, exposure_s=0.0085)
    assert g["uniform_fraction"] > 0.8           # spread across the period
    assert g["fraction_during_exposure"] > 0.5
    assert g["interlock_holds"] is False


def test_gate_starting_exactly_at_exposure_end_is_not_an_overlap():
    """A pulse beginning as integration ends does not overlap it. An
    inclusive comparison called a correctly interlocked gate contaminated."""
    fs = 80.0
    frame_t = np.arange(400) / fs
    gate_on_t = frame_t + 0.0085                 # exactly at the boundary
    g = ev.gate_phase_within_frame(frame_t, gate_on_t, exposure_s=0.0085)
    assert g["fraction_during_exposure"] == 0.0
    assert g["interlock_holds"] is True
    assert g["margin_ms"] == pytest.approx(0.0, abs=0.05)


def test_gate_margin_reports_boundary_hugging():
    fs = 80.0
    frame_t = np.arange(400) / fs
    tight = ev.gate_phase_within_frame(frame_t, frame_t + 0.0086,
                                       exposure_s=0.0085)
    roomy = ev.gate_phase_within_frame(frame_t, frame_t + 0.0105,
                                       exposure_s=0.0085)
    assert tight["margin_ms"] < 0.5
    assert roomy["margin_ms"] > 1.5
    assert tight["interlock_holds"] and roomy["interlock_holds"]


def test_gate_genuinely_inside_exposure_is_caught():
    fs = 80.0
    frame_t = np.arange(400) / fs
    g = ev.gate_phase_within_frame(frame_t, frame_t + 0.004, exposure_s=0.0085)
    assert g["fraction_during_exposure"] == pytest.approx(1.0)
    assert g["interlock_holds"] is False


def test_timestamp_exports_are_not_sessions(tmp_path):
    """A camera timestamp dump looks like a session stem with every companion
    file missing, and clutters the listing with unfixable problems."""
    _write_animal(tmp_path)
    reg = metadata.load_registry(tmp_path)
    bcl, _, _ = make_daq(duration_s=2.0)
    savemat(tmp_path / "20261026_154611_0-479999999_test7_timestamps.mat",
            {"ts": np.arange(10.0)})
    savemat(tmp_path / "derek_s1_bcam.mat", {"bcl": bcl})
    (tmp_path / "derek_s1.avi").write_bytes(b"\x00")

    stems = {s.stem for s in discovery.discover_sessions(tmp_path, reg)}
    assert "derek_s1" in stems
    assert not any("timestamps" in s for s in stems)


def test_default_animal_is_recorded_as_an_assumption(tmp_path):
    """Binding by assumption must be visible downstream: a wrong guess
    relabels every region with no error at any stage."""
    _write_animal(tmp_path)
    reg = metadata.load_registry(tmp_path)
    bcl, _, _ = make_daq(duration_s=2.0)
    savemat(tmp_path / "cl_test_withmice7_bcam.mat", {"bcl": bcl})
    (tmp_path / "cl_test_withmice7.avi").write_bytes(b"\x00")
    savemat(tmp_path / "TranslatedCoords_100726.mat",
            {"translated_coords": np.ones((3, 3))})
    savemat(tmp_path / "mids_100726.mat", {"mids": np.zeros((1, 2))})

    plain = [s for s in discovery.discover_sessions(tmp_path, reg)
             if s.stem == "cl_test_withmice7"][0]
    assert not plain.ok and not plain.animal_assumed

    assumed = [s for s in discovery.discover_sessions(
        tmp_path, reg, default_animal="derek") if s.stem == "cl_test_withmice7"][0]
    assert assumed.animal_assumed
    assert assumed.animal.animal_id == "derek"
    assert any("ASSUMING" in w for w in assumed.warnings)
    assert "ASSUMED" in assumed.summary()


def test_unknown_default_animal_is_an_error(tmp_path):
    bcl, _, _ = make_daq(duration_s=2.0)
    savemat(tmp_path / "x_s1_bcam.mat", {"bcl": bcl})
    (tmp_path / "x_s1.avi").write_bytes(b"\x00")
    s = [c for c in discovery.discover_sessions(tmp_path, {},
                                                default_animal="nobody")][0]
    assert any("not in the registry" in p for p in s.problems)


def test_least_squares_cannot_increase_in_sample_variance():
    """A regression fit by least squares always reduces in-sample residual
    variance, positivity constraint or not. So the isosbestic correction can
    never be the SOURCE of excess noise -- only a failure to remove it."""
    rng = np.random.default_rng(31)
    n = 3000
    y = rng.normal(0, 1.0, n)
    for X in (rng.normal(0, 5.0, (n, 1)),          # uncorrelated, huge
              (y + rng.normal(0, 3.0, n)).reshape(-1, 1)):
        for a in (1e-9, 1e-3, "cv"):
            resid, _ = pp.regress_isosbestic(y, X, alpha=a)
            assert np.nanstd(resid) <= np.nanstd(y) * (1 + 1e-9)


def test_warns_when_residual_noise_is_implausibly_large():
    """Photon noise on a bulk fiber signal is well under 1% dF/F. Anything
    near 10% is motion or coupling artifact the regression did not remove."""
    rng = np.random.default_rng(21)
    n_frames, n_roi = 4000, 2
    frame_t = np.arange(n_frames) / 80.0
    led = (np.arange(n_frames) % 2).astype(int)
    traces = np.empty((n_frames, n_roi))
    for j in range(n_roi):
        sig = 1000.0 + rng.normal(0, 2.0, n_frames)       # clean signal
        iso = 3.0 + rng.normal(0, 1.5, n_frames)          # near-dark, noisy
        traces[:, j] = np.where(led == 1, sig, np.abs(iso))
    out = pp.preprocess_traces(traces, led, frame_t, alpha=1e-6,
                               baseline_tau_s=10.0, large_sd_dff=0.05)
    sd_a = np.array(out.qc["dff_sd_after_regression"])
    sd_b = np.array(out.qc["dff_sd_before_regression"])
    assert np.all(sd_a <= sd_b * (1 + 1e-9))      # never worse, in sample
    assert out.qc["n_roi_high_noise"] == 0        # this fixture is clean

    # now a trace with motion-scale excursions
    big = traces.copy()
    big[led == 1, 0] += 200.0 * np.sin(np.arange((led == 1).sum()) / 30.0)
    out2 = pp.preprocess_traces(big, led, frame_t, alpha=1e-6,
                                baseline_tau_s=10.0, large_sd_dff=0.05)
    assert out2.qc["n_roi_high_noise"] >= 1
    assert any("residual dF/F noise above" in w for w in out2.qc["warnings"])


def test_regression_can_be_disabled():
    traces, led, frame_t = _synth_session(n_frames=1200)
    out = pp.preprocess_traces(traces, led, frame_t, regress=False,
                               baseline_tau_s=10.0)
    np.testing.assert_allclose(out.dff, out.dff_raw, equal_nan=True)
    assert any("disabled" in w for w in out.qc["warnings"])


def test_common_mode_removal_recovers_a_buried_signal():
    """Bundle motion moves every fiber together; an evoked response at one
    site does not. Subtracting the shared component should leave the response
    and remove the artifact."""
    rng = np.random.default_rng(41)
    n_t, n_roi = 4000, 12
    motion = 0.3 * np.sin(np.arange(n_t) / 7.0) + rng.normal(0, 0.1, n_t)
    X = np.tile(motion[:, None], (1, n_roi)) + rng.normal(0, 0.01, (n_t, n_roi))
    resp = np.zeros(n_t)
    resp[1000:1200] = 0.05
    X[:, 3] += resp

    out, info = pp.remove_common_mode(X, method="median", exclude=[3])
    assert np.median(info["noise_reduction"]) > 5
    # the response survives
    rec = out[1000:1200, 3].mean() - out[:900, 3].mean()
    assert rec == pytest.approx(0.05, rel=0.25)


def test_common_mode_excludes_the_stimulated_fiber():
    """If the stimulated fiber contributes to the common component, part of
    its own response is subtracted from it."""
    n_t, n_roi = 2000, 4
    X = np.zeros((n_t, n_roi))
    big = np.zeros(n_t)
    big[500:1500] = 1.0
    X[:, 0] = big
    kept, _ = pp.remove_common_mode(X, method="median", exclude=[0])
    assert kept[500:1500, 0].mean() == pytest.approx(1.0, abs=1e-6)


def test_common_mode_needs_enough_fibers():
    with pytest.raises(ValueError, match="at least 3"):
        pp.remove_common_mode(np.zeros((100, 4)), exclude=[0, 1])


def _burst_log(n_bursts=10, pulses=10, rate=10.0, isi=5.0, width=0.005,
               rate_hz=DAQ_RATE):
    """A 10 Hz burst protocol: pulses 100 ms apart, bursts 5 s apart."""
    times = []
    for b in range(n_bursts):
        t0 = 2.0 + b * isi
        times += [t0 + k / rate for k in range(pulses)]
    return np.array(times)


def test_bursts_are_grouped_into_one_event_each(tmp_path):
    """Each pulse of a 10 Hz burst is not a trial: the burst is what was
    randomised, and pulse-aligned windows contain neighbouring pulses."""
    times = _burst_log()
    bcl, _, _ = make_daq(duration_s=60.0, laser_times=times,
                         laser_width_s=0.005)
    log = matio.load_bcam(write_bcam(tmp_path, bcl, stem="burst"))
    p = ev.detect_laser_pulses(log, gate_channel=None)
    assert p.n == 100                       # every pulse

    g = ev.group_pulses_into_bursts(p)
    assert g.n == 10
    assert g.params["grouped_into_bursts"] is True
    assert g.params["pulses_per_burst"]["median"] == 10
    # burst spans first onset to last offset: 9 intervals of 100 ms + width
    assert np.allclose(g.width_s, 0.9 + 0.005, atol=0.01)
    np.testing.assert_allclose(g.onset_t[:3], times[[0, 10, 20]], atol=0.01)


def test_non_burst_protocol_is_returned_unchanged(tmp_path):
    times = np.arange(2.0, 40.0, 5.0)       # evenly spaced single pulses
    bcl, _, _ = make_daq(duration_s=50.0, laser_times=times)
    log = matio.load_bcam(write_bcam(tmp_path, bcl, stem="single"))
    p = ev.detect_laser_pulses(log, gate_channel=None)
    g = ev.group_pulses_into_bursts(p)
    assert g.n == p.n
    assert not g.params.get("grouped_into_bursts")


def test_explicit_gap_overrides_inference(tmp_path):
    times = _burst_log()
    bcl, _, _ = make_daq(duration_s=60.0, laser_times=times)
    log = matio.load_bcam(write_bcam(tmp_path, bcl, stem="burst2"))
    p = ev.detect_laser_pulses(log, gate_channel=None)
    # a gap larger than the between-burst interval merges everything
    g = ev.group_pulses_into_bursts(p, max_gap_s=10.0)
    assert g.n == 1


def test_continuous_and_piecewise_f0_differ_only_at_boundaries():
    """Piecewise fitting introduces a discontinuity at each epoch edge. With
    no regime change there is nothing for the boundary to respect."""
    traces, led, frame_t = _synth_session(n_frames=6000)
    kw = dict(laser_onset_t=np.array([30.0, 40.0]),
              laser_offset_t=np.array([31.0, 41.0]),
              alpha=0.001, baseline_tau_s=10.0)
    cont = pp.preprocess_traces(traces, led, frame_t,
                                baseline_mode="continuous", **kw)
    piece = pp.preprocess_traces(traces, led, frame_t,
                                 baseline_mode="epochs", **kw)
    assert cont.qc["baseline_mode"] == "continuous"
    assert cont.qc["f0_boundary_jumps"] == []
    assert len(piece.qc["f0_boundary_jumps"]) >= 1
    # the two F0s agree away from the seams but not everywhere
    assert not np.allclose(cont.f0, piece.f0)


def test_short_epochs_make_piecewise_f0_underdetermined():
    """A segment shorter than baseline_tau_s does not constrain the fit, so
    piecewise F0 there differs from a continuous one in fast structure, not
    just at the seams. With long segments the two agree closely."""
    traces, led, frame_t = _synth_session(n_frames=6000)
    kw = dict(laser_onset_t=np.array([30.0, 40.0]),
              laser_offset_t=np.array([31.0, 41.0]),
              alpha=0.001, baseline_tau_s=60.0)
    a = pp.preprocess_traces(traces, led, frame_t,
                             baseline_mode="continuous", **kw)
    b = pp.preprocess_traces(traces, led, frame_t,
                             baseline_mode="epochs", **kw)
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, 0.5, "highpass", fs=a.qc["effective_fps"], output="sos")
    ha = sosfiltfilt(sos, np.nan_to_num(a.dff[:, 0]))
    hb = sosfiltfilt(sos, np.nan_to_num(b.dff[:, 0]))
    # close, but not identical: the 11 s stim epoch is far shorter than the
    # 60 s baseline timescale
    assert 0.98 < np.corrcoef(ha, hb)[0, 1] < 0.999
    assert any("shorter than baseline_tau_s" in w for w in b.qc["warnings"])


def test_unknown_baseline_mode_rejected():
    traces, led, frame_t = _synth_session(n_frames=1200)
    with pytest.raises(ValueError, match="unknown baseline_mode"):
        pp.preprocess_traces(traces, led, frame_t, baseline_mode="nope",
                             baseline_tau_s=10.0)


def test_extraction_hash_ignores_preprocessing_settings():
    """Changing a baseline or regression parameter must not force the video
    to be decoded again: extraction depends on none of them."""
    base = pl.PipelineConfig()
    for kw in ({"baseline_mode": "epochs"}, {"baseline_tau_s": 30.0},
               {"regression_predictors": "all"}, {"common_mode": "median"},
               {"exposure_s": 0.0085}, {"group_bursts": None}):
        other = pl.PipelineConfig(**kw)
        assert base.hash() != other.hash(), kw
        assert base.extraction_hash() == other.extraction_hash(), kw

    for kw in ({"roi_fraction": 0.6}, {"half": "right"}, {"video_plane": 1}):
        other = pl.PipelineConfig(**kw)
        assert base.extraction_hash() != other.extraction_hash(), kw


def test_large_files_are_fingerprinted_not_hashed(tmp_path):
    """Hashing a multi-gigabyte video reads the whole file and would dominate
    a run that otherwise reuses the extracted traces."""
    _write_animal(tmp_path)
    reg = metadata.load_registry(tmp_path)
    bcl, _, _ = make_daq(duration_s=2.0)
    savemat(tmp_path / "derek_s1_bcam.mat", {"bcl": bcl})
    (tmp_path / "derek_s1.avi").write_bytes(b"\x00" * (4 << 20))
    savemat(tmp_path / "TranslatedCoords_100726.mat",
            {"translated_coords": np.ones((3, 3))})
    savemat(tmp_path / "mids_100726.mat", {"mids": np.zeros((1, 2))})

    saved = discovery.LARGE_FILE_BYTES
    discovery.LARGE_FILE_BYTES = 2 << 20      # above the .mat, below the .avi
    try:
        s = [c for c in discovery.discover_sessions(tmp_path, reg)
             if c.stem == "derek_s1"][0]
        quick = s.file_hashes()
        full = s.file_hashes(full=True)
    finally:
        discovery.LARGE_FILE_BYTES = saved
    assert quick["video"].startswith("fp:")
    assert not full["video"].startswith("fp:")
    # the DAQ log is well under the threshold and is hashed either way
    assert not quick["bcam"].startswith("fp:")
    assert quick["bcam"] == full["bcam"]


def test_fingerprint_detects_edits_and_is_stable(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"a" * (1 << 20))
    a = matio.file_fingerprint(p)
    assert a == matio.file_fingerprint(p)          # stable
    p.write_bytes(b"b" + b"a" * ((1 << 20) - 1))   # head change
    assert matio.file_fingerprint(p) != a
    p.write_bytes(b"a" * (1 << 20) + b"z")         # size change
    assert matio.file_fingerprint(p) != a


def test_failures_are_detected_before_the_video_is_opened(tmp_path):
    """Decoding costs minutes. A session that cannot be processed must fail on
    metadata alone, not after a full pass over its video."""
    _write_animal(tmp_path)                       # 3 fibers
    reg = metadata.load_registry(tmp_path)
    bcl, _, _ = make_daq(duration_s=2.0)
    savemat(tmp_path / "derek_s1_bcam.mat", {"bcl": bcl})
    # calibration describing 12 fibers, animal record describing 3
    savemat(tmp_path / "TranslatedCoords_100726.mat",
            {"translated_coords": np.ones((12, 3))})
    savemat(tmp_path / "mids_100726.mat", {"mids": np.zeros((1, 2))})
    # a video path that would raise if anything tried to open it
    (tmp_path / "derek_s1.avi").write_bytes(b"not a video")

    cand = [c for c in discovery.discover_sessions(tmp_path, reg)
            if c.stem == "derek_s1"][0]
    r = pl.process_session(cand, tmp_path / "out")
    assert r.status == "failed"
    assert r.failed_stage == "preflight"
    assert "ROI order is what laser IDs index" in r.error


def test_preflight_catches_a_wrong_camera_channel(tmp_path):
    _write_animal(tmp_path)
    reg = metadata.load_registry(tmp_path)
    bcl, _, _ = make_daq(duration_s=2.0)
    savemat(tmp_path / "derek_s1_bcam.mat", {"bcl": bcl})
    savemat(tmp_path / "TranslatedCoords_100726.mat",
            {"translated_coords": np.ones((3, 3))})
    savemat(tmp_path / "mids_100726.mat", {"mids": np.zeros((1, 2))})
    (tmp_path / "derek_s1.avi").write_bytes(b"not a video")

    cand = [c for c in discovery.discover_sessions(tmp_path, reg)
            if c.stem == "derek_s1"][0]
    cfg = pl.PipelineConfig(camera_channel="laser_gate")   # flat, no triggers
    r = pl.process_session(cand, tmp_path / "out", cfg)
    assert r.status == "failed"
    assert r.failed_stage == "preflight"
    assert "camera triggers" in r.error


def test_protocol_is_inferred_from_the_laser_channel(tmp_path):
    """The protocol must be measured, not read from the filename: a renamed
    or changed session would otherwise drive frequency-dependent analyses
    with the wrong parameter."""
    times = _burst_log(n_bursts=6, pulses=20, rate=10.0, isi=10.0)
    bcl, _, _ = make_daq(duration_s=90.0, laser_times=times,
                         laser_width_s=0.005, laser_amplitude=0.12)
    log = matio.load_bcam(write_bcam(tmp_path, bcl, stem="proto"))
    raw = ev.detect_laser_pulses(log, gate_channel=None)
    bursts = ev.group_pulses_into_bursts(raw)
    roi = np.full(bursts.n, 3, dtype=int)
    im = np.array([[3, 0.12]])

    pr = ev.infer_protocol(raw, bursts, roi_id=roi, intensity_mat=im)
    assert pr["n_pulses"] == 120 and pr["n_events"] == 6
    assert pr["pulses_per_burst"]["median"] == 20
    assert pr["within_burst_rate_hz"] == pytest.approx(10.0, abs=0.2)
    assert pr["burst_duration_s"] == pytest.approx(1.9, abs=0.05)
    assert pr["inter_event_interval_s"] == pytest.approx(10.0, abs=0.1)
    assert pr["duty_cycle"] == pytest.approx(0.05, abs=0.01)
    assert pr["n_sites"] == 1
    assert pr["estimated_mw"]["median"] == pytest.approx(1.6, rel=0.05)


def test_protocol_name_check_catches_a_misnamed_session():
    pr = {"within_burst_rate_hz": 10.0, "burst_duration_s": 1.9}
    assert ev.protocol_matches_name(pr, "derek_10Hzburst_all12")["ok"]
    bad = ev.protocol_matches_name(pr, "derek_02Hzburst_all12")
    assert not bad["ok"]
    assert bad["checks"][0]["named"] == 2.0
    assert bad["checks"][0]["measured"] == 10.0


def test_common_isosbestic_rescues_a_dim_fiber():
    """The point of twelve isosbestics: each carries the same bundle motion
    plus its own photon noise, so a component built from the others corrects
    a fiber whose own 415 channel is too dark to serve as a regressor."""
    rng = np.random.default_rng(0)
    n, n_roi = 4000, 12
    motion = np.convolve(rng.normal(0, 1, n + 400), np.ones(60) / 60,
                         "same")[:n]
    motion = motion / motion.std() * 0.02
    iso = np.empty((n, n_roi))
    lvl = np.full(n_roi, 0.004)
    lvl[0] = 0.016                                   # fiber 0 is dim
    for j in range(n_roi):
        iso[:, j] = motion + rng.normal(0, lvl[j], n)
    y = motion + rng.normal(0, 0.003, n)

    out = {}
    for mode in ("same", "common"):
        X, names = pp.build_isosbestic_predictors(iso, 0, mode=mode)
        resid, info = pp.regress_isosbestic(y, X, alpha=1e-4)
        out[mode] = (float(np.std(resid)),
                     abs(float(np.corrcoef(resid, motion)[0, 1])))
    assert out["common"][0] < 0.6 * out["same"][0]
    # Residual SD is the quantity that matters; the motion-correlation
    # ratio moved when the own term was reparameterised as fiber-specific.
    assert out["common"][0] < 0.5 * out["same"][0]

    X, names = pp.build_isosbestic_predictors(iso, 0, mode="common")
    assert names == ["own_specific", "common_median"]
    assert X.shape == (n, 2)


def test_common_component_excludes_the_fiber_itself():
    """Otherwise the two predictors are the same quantity twice."""
    X = np.zeros((500, 4))
    X[:, 0] = 10.0                                   # fiber 0 is an outlier
    P, _ = pp.build_isosbestic_predictors(X, 0, mode="common")
    assert np.allclose(P[:, 0], 10.0)                # own kept
    assert np.allclose(P[:, 1], 0.0)                 # common has no trace of it


def test_predictor_modes_have_expected_widths():
    X = np.zeros((300, 12))
    assert pp.build_isosbestic_predictors(X, 3, mode="same")[0].shape[1] == 1
    assert pp.build_isosbestic_predictors(X, 3, mode="all")[0].shape[1] == 12
    assert pp.build_isosbestic_predictors(X, 3, mode="common")[0].shape[1] == 2
    with pytest.raises(ValueError, match="unknown predictor mode"):
        pp.build_isosbestic_predictors(X, 3, mode="nope")


def test_interleaved_protocols_are_separated():
    """A session that interleaves conditions must not be summarised by one set
    of parameters: that describes none of them."""
    rng = np.random.default_rng(0)
    w = np.concatenate([np.full(10, 0.1), np.full(10, 0.5), np.full(10, 1.0)])
    w = w * (1 + rng.normal(0, 0.02, w.size))
    v = np.full(w.size, 0.12)
    idx = rng.permutation(w.size)
    w, v = w[idx], v[idx]
    p = ev.LaserPulses(np.cumsum(np.full(w.size, 5.0)), np.zeros_like(w), w, v,
                       np.zeros(w.size, int), np.full(w.size, 0.26),
                       np.zeros_like(w))
    c = ev.classify_protocols(p)
    assert np.unique(c).size == 3
    for k, want in zip(sorted(np.unique(c)), (0.1, 0.5, 1.0)):
        assert np.median(w[c == k]) == pytest.approx(want, rel=0.05)

    pr = ev.infer_protocol(p, p)
    assert pr["interleaved"] is True
    assert pr["n_protocol_classes"] == 3
    assert [x["n_events"] for x in pr["protocol_classes"]] == [10, 10, 10]


def test_power_levels_split_within_one_duration():
    """Two powers at one site is two conditions -- but only once voltages are
    referenced to the site, which requires roi_id."""
    w = np.full(20, 0.5)
    v = np.r_[np.full(10, 0.12), np.full(10, 0.24)]
    roi = np.ones(20, dtype=int)
    p = ev.LaserPulses(np.arange(20) * 5.0, np.zeros(20), w, v,
                       np.zeros(20, int), np.full(20, 0.26), np.zeros(20))
    assert np.unique(ev.classify_protocols(p, roi_id=roi)).size == 2
    # without roi_id the voltage cannot be interpreted, so no power split
    assert np.unique(ev.classify_protocols(p)).size == 1


def test_per_site_voltages_do_not_create_spurious_classes():
    """intensity_mat gives every region its own command voltage for the same
    delivered power. Splitting on raw voltage would turn three interleaved
    durations across twelve fibers into dozens of classes."""
    rng = np.random.default_rng(0)
    dur = np.tile(np.repeat([0.1, 0.5, 1.0], 12), 5)
    roi = np.tile(np.arange(1, 13), 15)
    volt = 0.12 + 0.01 * roi                 # per-site, one power level
    i = rng.permutation(dur.size)
    dur, roi, volt = dur[i], roi[i], volt[i]
    p = ev.LaserPulses(np.arange(dur.size) * 5.0, np.zeros_like(dur), dur,
                       volt, np.zeros(dur.size, int),
                       np.full(dur.size, 0.26), np.zeros_like(dur))
    c = ev.classify_protocols(p, roi_id=roi)
    assert np.unique(c).size == 3
    for k in np.unique(c):
        assert np.unique(roi[c == k]).size == 12   # every site in every class


def test_two_powers_across_many_sites_give_one_split_each():
    rng = np.random.default_rng(1)
    dur = np.tile(np.repeat([0.1, 0.5, 1.0], 12), 5)
    roi = np.tile(np.arange(1, 13), 15)
    base = 0.12 + 0.01 * roi
    volt = base * np.where(rng.random(dur.size) < 0.5, 2.0, 1.0)
    p = ev.LaserPulses(np.arange(dur.size) * 5.0, np.zeros_like(dur), dur,
                       volt, np.zeros(dur.size, int),
                       np.full(dur.size, 0.26), np.zeros_like(dur))
    assert np.unique(ev.classify_protocols(p, roi_id=roi)).size == 6


def test_single_protocol_is_one_class():
    w = np.full(20, 0.5) * (1 + np.random.default_rng(1).normal(0, 0.01, 20))
    p = ev.LaserPulses(np.arange(20) * 5.0, np.zeros(20), w,
                       np.full(20, 0.12), np.zeros(20, int),
                       np.full(20, 0.26), np.zeros(20))
    assert np.unique(ev.classify_protocols(p)).size == 1
    pr = ev.infer_protocol(p, p)
    assert not pr.get("interleaved")


def _write_log(tmp_path, name, var, arr, cell_blocks=False):
    if cell_blocks:
        pl = np.empty((1, 3), dtype=object)
        pl[0, 0] = np.array([[0.0]])
        pl[0, 1] = np.array([[0.0]])
        pl[0, 2] = arr
        savemat(tmp_path / name, {var: pl})
    else:
        savemat(tmp_path / name, {var: arr})
    return tmp_path / name


def test_stim_log_variable_name_is_discovered(tmp_path):
    """The acquisition GUI has called this variable 'pl' and 'pulse_log';
    hardcoding either breaks on the other."""
    m = np.column_stack([np.array([5.0, 10.0, 15.0, 20.0]),
                         np.array([7, 3, 7, 3]), np.full(4, 0.23)])
    for var in ("pulse_log", "pl", "laser_log"):
        p = _write_log(tmp_path, f"{var}.mat", var, m)
        out = matio.load_stim_log(p)
        np.testing.assert_array_equal(out[:, 0].astype(int), [7, 3, 7, 3])
        assert np.all(np.diff(out[:, 1]) >= 0)


def test_stim_log_columns_inferred_in_either_order(tmp_path):
    """Column order is not guaranteed across GUI versions."""
    ids = np.array([3, 3, 7, 7])
    t = np.array([10.0, 20.0, 5.0, 15.0])
    legacy = np.column_stack([ids, np.zeros((4, 3)), t])   # id first
    modern = np.column_stack([t, ids, np.full(4, 0.23)])   # time first
    a = matio.load_stim_log(_write_log(tmp_path, "a.mat", "pl", legacy,
                                       cell_blocks=True))
    b = matio.load_stim_log(_write_log(tmp_path, "b.mat", "pulse_log", modern))
    np.testing.assert_allclose(a, b)
    np.testing.assert_array_equal(a[:, 0].astype(int), [7, 3, 7, 3])


def test_sorted_roi_column_is_not_mistaken_for_time(tmp_path):
    """A log whose ROI column happens to be sorted must not be read as a
    clock: monotonicity is not evidence of a timestamp, and a concatenated
    per-sweep log is not in time order anyway."""
    ids = np.array([3, 3, 7, 7])                    # sorted
    t = np.array([10.0, 20.0, 5.0, 15.0])           # not sorted
    arr = np.column_stack([ids, np.zeros((4, 3)), t])
    i, tc, ev = matio._infer_stim_log_columns(arr)
    assert i == 0 and tc == 4
    assert ev["ids"] == [3, 7]


def test_stim_log_refuses_an_unreadable_layout(tmp_path):
    arr = np.zeros((5, 3))                          # nothing distinguishable
    p = _write_log(tmp_path, "flat.mat", "pulse_log", arr)
    with pytest.raises(ValueError):
        matio.load_stim_log(p)


def test_stim_log_names_what_it_tried(tmp_path):
    """A file with several unrecognised variables must say which names were
    tried and what the file contains, not just fail."""
    savemat(tmp_path / "none.mat", {"alpha": np.zeros((3, 3)),
                                    "beta": np.ones((3, 3))})
    with pytest.raises(KeyError, match="no stim log variable found"):
        matio.load_stim_log(tmp_path / "none.mat")


def test_single_unnamed_numeric_variable_is_used(tmp_path):
    """One numeric variable under an unrecognised name is unambiguous."""
    m = np.column_stack([np.array([5.0, 10.0, 15.0, 20.0]),
                         np.array([7, 3, 7, 3])])
    savemat(tmp_path / "odd.mat", {"laserEvents": m})
    out = matio.load_stim_log(tmp_path / "odd.mat")
    np.testing.assert_array_equal(out[:, 0].astype(int), [7, 3, 7, 3])


def test_progress_stage_counts_and_tolerates_no_total():
    from multifiber.progress import Stage
    import io

    buf = io.StringIO()
    with Stage("x", total=10, enabled=True, stream=buf,
               min_interval_s=0.0) as st:
        for _ in range(10):
            st.update(1)
    assert st.n == 10
    assert "done" in buf.getvalue()

    buf2 = io.StringIO()
    with Stage("y", enabled=True, stream=buf2, min_interval_s=0.0) as st2:
        st2.update(3)
    assert st2.n == 3


def test_progress_update_to_is_absolute():
    """Extraction reports a running total, not an increment."""
    from multifiber.progress import Stage
    import io

    with Stage("z", total=100, enabled=True, stream=io.StringIO(),
               min_interval_s=0.0) as st:
        st.update_to(50)
        st.update_to(50)      # no movement
        st.update_to(75)
    assert st.n == 75


def test_progress_disabled_is_silent():
    from multifiber.progress import Stage
    import io

    buf = io.StringIO()
    with Stage("q", total=5, enabled=False, stream=buf) as st:
        for _ in range(5):
            st.update(1)
    assert buf.getvalue() == ""
    assert st.n == 5


def test_gate_on_the_exposure_boundary_is_not_contaminating():
    """An interlock puts the gate flush against the end of integration. An
    inclusive comparison calls that an overlap and blanks a correctly
    configured rig."""
    fs = 80.0
    ft = np.arange(400) / fs
    assert not pp.frames_contaminated_by_laser(
        ft, ft + 0.0085, exposure_s=0.0085).any()
    assert not pp.frames_contaminated_by_laser(
        ft, ft + 0.0090, exposure_s=0.0085).any()
    # a genuine overlap is still caught
    assert pp.frames_contaminated_by_laser(
        ft, ft + 0.0040, exposure_s=0.0085).all()


def test_implied_exposure_recovers_the_boundary():
    """With a working interlock the earliest gate onset is the end of
    integration, so the exposure can be measured instead of assumed."""
    fs = 80.0
    ft = np.arange(400) / fs
    for true_exp in (0.0065, 0.0085, 0.0100):
        ie = ev.implied_exposure(ft, ft + true_exp, ft + true_exp + 0.0032)
        assert ie["implied_exposure_s"] == pytest.approx(true_exp, abs=1e-4)
        assert ie["implied_gap_s"] == pytest.approx(1 / fs - true_exp,
                                                    abs=1e-4)


def test_implied_exposure_flags_a_trigger_offset():
    """If the camera channel records an output trigger offset from the start
    of exposure, the implied exposure disagrees with the configured one by
    that constant."""
    fs = 80.0
    ft = np.arange(400) / fs
    offset = 0.0065                      # output trigger delay
    ie = ev.implied_exposure(ft, ft + 0.0085 - offset)
    assert abs(ie["implied_exposure_s"] - 0.0085) > 0.001


def test_extraction_cache_matches_on_the_fingerprint(tmp_path):
    """The reuse check must compare like with like. Storing the full hash
    after extraction and comparing a cheap fingerprint on the next run means
    the cache never hits and every run re-decodes the video."""
    fp = {"video": "fp:abc", "calibration/mids": "deadbeef"}
    full = {"video": "0123456789abcdef", "calibration/mids": "deadbeef"}
    mio.write_session(tmp_path, "s1", {
        "session_id": "s1", "status": "ok", "extraction_hash": "EXT",
        "config_hash": "CFG",
        "inputs": {"sha256": full, "fingerprint": fp}})

    prior = mio.read_manifest(tmp_path, "s1")
    prior_fp = prior["inputs"].get("fingerprint") or prior["inputs"]["sha256"]
    assert prior_fp["video"] == fp["video"]          # cache hits
    assert prior["inputs"]["sha256"]["video"] != fp["video"]   # and why

    # an older manifest without a fingerprint block falls back to sha256
    mio.write_session(tmp_path, "s2", {
        "session_id": "s2", "status": "ok", "extraction_hash": "EXT",
        "inputs": {"sha256": fp}})
    prior2 = mio.read_manifest(tmp_path, "s2")
    fb = prior2["inputs"].get("fingerprint") or prior2["inputs"]["sha256"]
    assert fb["video"] == fp["video"]


# --------------------------------------------------------------------------- #
# Variant engine
# --------------------------------------------------------------------------- #

from multifiber import variants as va  # noqa: E402


def _synth_session_data(n_frames=6000, n_roi=4):
    import pandas as pd

    tr, led, ft = _synth_session(n_frames=n_frames, n_roi=n_roi)
    onsets = np.arange(20.0, 60.0, 10.0)
    ev_df = pd.DataFrame({"onset_t": onsets, "offset_t": onsets + 1.0,
                          "width_s": 1.0, "region": "A"})
    return va.SessionData("synth", tr, ft, led,
                          [chr(65 + i) for i in range(n_roi)], ev_df,
                          None), onsets


def test_variants_with_different_specs_differ():
    """If two configurations gave identical output the comparison would be
    pointless; if they gave wildly different statistics with no diagnostic to
    choose between them, picking the best would be selection on the outcome."""
    sd, onsets = _synth_session_data()
    a = va.run_variant(sd, va.VariantSpec(baseline_tau_s=60))
    b = va.run_variant(sd, va.VariantSpec(baseline_tau_s=5))
    assert not np.allclose(np.nan_to_num(a.dff), np.nan_to_num(b.dff))
    assert np.nanstd(b.dff) < np.nanstd(a.dff)     # faster F0 removes drift


def test_variant_spec_key_is_stable_and_label_free():
    s1 = va.VariantSpec(label="A")
    s2 = va.VariantSpec(label="B")
    assert s1.key() == s2.key()                     # label is not a setting
    assert s1.key() != va.VariantSpec(baseline_tau_s=10).key()


def test_lambda_override_bypasses_the_timescale():
    sd, _ = _synth_session_data()
    auto = va.run_variant(sd, va.VariantSpec(baseline_tau_s=60))
    over = va.run_variant(sd, va.VariantSpec(lambda_override=1e5))
    assert not np.allclose(np.nan_to_num(auto.f0), np.nan_to_num(over.f0))


def test_null_is_recomputed_per_variant():
    """A null built under one baseline does not describe another."""
    sd, onsets = _synth_session_data()
    a = va.run_variant(sd, va.VariantSpec(baseline_tau_s=60))
    b = va.run_variant(sd, va.VariantSpec(baseline_tau_s=5))
    na = va.shift_null(a, onsets, 0, n_shift=60)
    nb = va.shift_null(b, onsets, 0, n_shift=60)
    assert na["sd"] != nb["sd"]


def test_compare_returns_effect_z_and_p():
    sd, onsets = _synth_session_data()
    r = va.run_variant(sd, va.VariantSpec())
    c = va.compare(r, onsets, 0, n_shift=60)
    assert c["n"] == onsets.size
    assert np.isfinite(c["effect"]) and np.isfinite(c["z"])
    assert 0 < c["p"] <= 1
    assert c["mean"].size == c["t"].size


def test_several_principal_components_remove_several_artifacts():
    """One component assumes a single artifact waveform. Where two
    independent things move, the second needs its own regressor."""
    rng = np.random.default_rng(0)
    n, nroi = 4000, 12
    a1 = np.convolve(rng.normal(0, 1, n + 200), np.ones(60) / 60, "same")[:n]
    a2 = np.convolve(rng.normal(0, 1, n + 200), np.ones(20) / 20, "same")[:n]
    a1, a2 = a1 / a1.std(), a2 / a2.std()
    w1, w2 = rng.uniform(.5, 1.5, nroi), rng.uniform(-1, 1, nroi)
    iso = np.stack([0.02 * (w1[j] * a1 + w2[j] * a2)
                    + rng.normal(0, 0.004, n) for j in range(nroi)], 1)
    y = 0.02 * (w1[0] * a1 + w2[0] * a2) + rng.normal(0, 0.003, n)

    out = {}
    for cm in ("pc1", "pc2"):
        X, _ = pp.build_isosbestic_predictors(iso, 0, mode="common",
                                              common_method=cm)
        r, _ = pp.regress_isosbestic(y, X, alpha=1e-6)
        out[cm] = abs(float(np.corrcoef(r, a2)[0, 1]))
    assert out["pc2"] < 0.2 * out["pc1"]


def test_components_are_supplied_in_both_signs():
    """A principal component's sign is arbitrary, but the regression
    constrains coefficients to be positive: without both signs a component
    needing a negative weight is silently zeroed."""
    X = np.random.default_rng(0).normal(size=(500, 6))
    P, names = pp.build_isosbestic_predictors(X, 0, mode="common",
                                              common_method="pc2")
    assert names == ["own_specific", "common_pc1+", "common_pc1-",
                     "common_pc2+", "common_pc2-"]
    np.testing.assert_allclose(P[:, 1], -P[:, 2])
    np.testing.assert_allclose(P[:, 3], -P[:, 4])


def test_more_components_than_sources_is_harmless():
    rng = np.random.default_rng(1)
    n, nroi = 3000, 12
    a = np.convolve(rng.normal(0, 1, n + 200), np.ones(60) / 60, "same")[:n]
    a /= a.std()
    w = rng.uniform(.5, 1.5, nroi)
    iso = np.stack([0.02 * w[j] * a + rng.normal(0, 0.004, n)
                    for j in range(nroi)], 1)
    y = 0.02 * w[0] * a + rng.normal(0, 0.003, n)
    sds = []
    for cm in ("pc1", "pc3"):
        X, _ = pp.build_isosbestic_predictors(iso, 0, mode="common",
                                              common_method=cm)
        r, _ = pp.regress_isosbestic(y, X, alpha=1e-6)
        sds.append(float(np.std(r)))
    assert abs(sds[1] - sds[0]) / sds[0] < 0.05


def test_unknown_common_method_is_rejected():
    X = np.zeros((300, 5))
    with pytest.raises(ValueError, match="unknown common_method"):
        pp.build_isosbestic_predictors(X, 0, mode="common",
                                       common_method="pca")


def test_common_does_not_steal_the_own_fiber_coefficient():
    """An artifact carried by a MINORITY of fibers -- stray light reaching
    the fibers near a stimulation -- is suppressed by the median across
    fibers, so only the own-fiber isosbestic carries it. If the common
    component competes for the fit, the L1 penalty shrinks the own
    coefficient (to exactly zero, before this was fixed) and the artifact
    goes uncorrected."""
    rng = np.random.default_rng(0)
    n, nroi = 6000, 12
    drift = np.convolve(rng.normal(0, 1, n + 400), np.ones(120) / 120,
                        "same")[:n]
    drift /= drift.std()
    art = np.zeros(n)
    for k in range(8):
        art[600 + 700 * k:600 + 700 * k + 40] = 1.0
    near = {0, 1, 2}
    iso = np.stack([0.02 * drift + (0.03 * art if j in near else 0.0)
                    + rng.normal(0, 0.004, n) for j in range(nroi)], 1)
    y = 0.02 * drift + 0.012 * art + rng.normal(0, 0.003, n)

    # A brief artifact contributes little to overall prediction error, so a
    # cross-validated L1 penalty -- tuned on the drift, which dominates --
    # shrinks the predictor carrying it. With the penalty small the
    # fiber-specific term does its job.
    X, _ = pp.build_isosbestic_predictors(iso, 0, mode="common")
    r, info = pp.regress_isosbestic(y, X, alpha=1e-6)
    assert info["coefficients"][0] > 0.1
    left = float(np.mean(r[art > 0]) - np.mean(r[art == 0]))
    assert abs(left) < 0.3 * 0.012


def test_common_still_rescues_a_dim_fiber_after_orthogonalisation():
    """The fix must not cost the case the common component exists for."""
    rng = np.random.default_rng(1)
    n, nroi = 5000, 12
    mot = np.convolve(rng.normal(0, 1, n + 200), np.ones(60) / 60, "same")[:n]
    mot /= mot.std()
    lvl = np.full(nroi, 0.004)
    lvl[0] = 0.016
    iso = np.stack([0.02 * mot + rng.normal(0, lvl[j], n)
                    for j in range(nroi)], 1)
    y = 0.02 * mot + rng.normal(0, 0.003, n)
    out = {}
    for mode in ("same", "common"):
        X, _ = pp.build_isosbestic_predictors(iso, 0, mode=mode)
        r, _ = pp.regress_isosbestic(y, X, alpha=1e-6)
        out[mode] = float(np.std(r))
    assert out["common"] < 0.5 * out["same"]


def test_common_component_is_orthogonal_to_own():
    X = np.random.default_rng(2).normal(size=(2000, 8))
    P, _ = pp.build_isosbestic_predictors(X, 0, mode="common")
    own, com = P[:, 0], P[:, 1]
    r = abs(float(np.corrcoef(own - own.mean(), com - com.mean())[0, 1]))
    assert r < 1e-8


def test_stim_regressors_separate_by_kinetics():
    """The square term stops dead at offset; the kinetic term decays through
    it. That difference is what separates an optical artifact from a calcium
    response, since both are time-locked to the laser."""
    fs = 40.0
    t = np.arange(0, 12, 1 / fs)
    on = np.array([2.0, 6.0])
    X, names = pp.stim_regressors(t, on, on + 1.0, decay_tau=0.4)
    assert names == ["stim_square", "stim_kinetic"]
    i_on, i_off = int(2.0 * fs), int(3.0 * fs)
    assert X[i_on, 0] == 1.0 and X[i_on, 1] < 0.05      # square on, no rise
    assert X[i_off + 2, 0] == 0.0                        # square stops dead
    assert X[i_off + 2, 1] > 0.5                         # kinetic persists
    assert X[int(4.5 * fs), 1] < 0.1                     # and then decays


def test_stim_artifact_is_removed_and_the_response_recovered():
    """An additive artifact is a larger FRACTION of the dimmer isosbestic
    than of the signal, while motion is the same fraction of both. One
    coefficient on the isosbestic cannot serve both ratios, which is why the
    stimulus-locked terms are needed."""
    rng = np.random.default_rng(0)
    fs, n = 40.0, 12000
    t = np.arange(n) / fs
    on = np.arange(20, 280, 10.0)
    S, _ = pp.stim_regressors(t, on, on + 1.0)
    square, kin = S[:, 0], S[:, 1]
    mot = np.convolve(rng.normal(0, 1, n + 200), np.ones(60) / 60, "same")[:n]
    mot /= mot.std()
    F470, F415, stray, true_amp = 179.0, 46.0, 0.35, 0.004
    d470 = (0.02 * mot + stray * square / F470 + true_amp * kin
            + rng.normal(0, 0.002, n))
    d415 = 0.02 * mot + stray * square / F415 + rng.normal(0, 0.003, n)
    iso = np.stack([d415 + rng.normal(0, 0.003, n) for _ in range(12)], 1)

    X0, _ = pp.build_isosbestic_predictors(iso, 0, mode="common")
    r0, _ = pp.regress_isosbestic(d470, X0, method="ols")
    left0 = float(np.mean(r0[square > 0]) - np.mean(r0[square == 0]))

    X1 = np.column_stack([X0, S])
    r1, info = pp.regress_isosbestic(d470, X1, method="ols",
                                     keep=[X1.shape[1] - 1])
    est = info["coefficients"][X1.shape[1] - 1]
    assert abs(est - true_amp) < 0.25 * true_amp     # response recovered
    assert abs(left0) > 0.5 * stray / F470           # isosbestic alone fails
    assert info["kept_columns"] == [X1.shape[1] - 1]


def test_keep_columns_are_estimated_but_not_subtracted():
    rng = np.random.default_rng(1)
    n = 3000
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = 2.0 * a + 3.0 * b + rng.normal(0, 0.01, n)
    X = np.column_stack([a, b])
    r, info = pp.regress_isosbestic(y, X, method="ols", keep=[1])
    assert info["coefficients"][1] == pytest.approx(3.0, rel=0.05)
    # b's contribution is still in the residual
    assert abs(float(np.corrcoef(r, b)[0, 1])) > 0.9


def test_regression_methods_all_run():
    rng = np.random.default_rng(2)
    n = 2000
    x1 = np.sin(np.arange(n) / 40.0)
    x2 = np.cos(np.arange(n) / 70.0)
    y = 0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.05, n)
    X = np.column_stack([x1, x2])
    for m in ("lasso", "ols", "ridge", "hierarchical"):
        r, info = pp.regress_isosbestic(y, X, method=m, alpha=1e-6,
                                        unpenalized=1)
        assert info["method"] == m
        assert np.std(r) < np.std(y)
        assert len(info["coefficients"]) == 2


def test_auto_picks_per_fiber_without_seeing_the_stimulation():
    """Choosing an estimator by the result it produces is selection on the
    outcome. Scoring out-of-sample prediction outside the stimulation
    windows makes the choice about motion correction only."""
    rng = np.random.default_rng(0)
    n, nroi = 9000, 12
    mot = np.convolve(rng.normal(0, 1, n + 300), np.ones(80) / 80, "same")[:n]
    mot /= mot.std()
    stim = np.zeros(n, dtype=bool)
    for k in range(12):
        stim[500 + 700 * k:500 + 700 * k + 40] = True
    iso = np.stack([0.02 * mot + rng.normal(0, 0.004, n)
                    for _ in range(nroi)], 1)
    y = 0.02 * mot + rng.normal(0, 0.003, n)
    X, _ = pp.build_isosbestic_predictors(iso, 0, mode="common")
    r, info = pp.regress_isosbestic(y, X, method="auto", alpha="cv",
                                    score_mask=~stim)
    assert info["auto_chosen"] in ("lasso", "ols", "hierarchical")
    assert set(info["auto_scores"]) <= {"lasso", "ols", "hierarchical"}
    assert np.std(r) < np.std(y)


def test_score_mask_governs_scoring_not_fitting():
    """score_mask decides where the automatic selection is JUDGED;
    fit_mask decides where coefficients are estimated. Keeping them separate
    matters: a stimulus-locked artifact lives inside the stimulation
    windows, so a fit that excluded them could never remove it."""
    rng = np.random.default_rng(1)
    n = 6000
    mot = np.convolve(rng.normal(0, 1, n + 200), np.ones(60) / 60, "same")[:n]
    mot /= mot.std()
    iso = np.stack([0.02 * mot + rng.normal(0, 0.004, n) for _ in range(8)], 1)
    y = 0.02 * mot + rng.normal(0, 0.003, n)
    X, _ = pp.build_isosbestic_predictors(iso, 0, mode="common")
    mask = np.ones(n, dtype=bool)
    mask[2000:2400] = False

    # score_mask alone leaves the fit over every sample
    a = pp.regress_isosbestic(y, X, method="ols", score_mask=mask)[1]
    assert a["n_fit"] == n

    # fit_mask removes them from the fit
    b = pp.regress_isosbestic(y, X, method="ols", fit_mask=mask)[1]
    assert b["n_fit"] == int(mask.sum())
    assert b["coefficients"] != a["coefficients"]


def test_fitting_outside_a_window_cannot_remove_what_is_inside_it():
    """The reason fitting defaults to every sample."""
    rng = np.random.default_rng(0)
    fs, n = 40.0, 8000
    t = np.arange(n) / fs
    on = np.arange(20, 190, 10.0)
    S, _ = pp.stim_regressors(t, on, on + 1.0)
    square = S[:, 0]
    iso = np.stack([0.02 * square + rng.normal(0, 0.002, n)
                    for _ in range(8)], 1)
    y = 0.02 * square + rng.normal(0, 0.002, n)
    X, _ = pp.build_isosbestic_predictors(iso, 0, mode="common")
    outside = ~(square > 0)

    r_all, _ = pp.regress_isosbestic(y, X, method="ols")
    r_out, _ = pp.regress_isosbestic(y, X, method="ols", fit_mask=outside)
    left_all = abs(float(np.mean(r_all[square > 0])
                         - np.mean(r_all[square == 0])))
    left_out = abs(float(np.mean(r_out[square > 0])
                         - np.mean(r_out[square == 0])))
    assert left_all < 0.3 * left_out


def test_others_mode_excludes_the_fiber_itself():
    X = np.arange(500 * 6, dtype=float).reshape(500, 6)
    P, names = pp.build_isosbestic_predictors(X, 3, mode="others")
    assert P.shape[1] == 5
    assert "iso3" not in names
    np.testing.assert_allclose(P[:, 3], X[:, 4])     # order preserved


def test_predictor_modes_have_expected_widths_including_others():
    X = np.zeros((300, 12))
    widths = {m: pp.build_isosbestic_predictors(X, 3, mode=m)[0].shape[1]
              for m in ("same", "all", "others", "common")}
    assert widths == {"same": 1, "all": 12, "others": 11, "common": 2}


def test_components_are_capped_at_the_number_available():
    """Asking for more components than there are other fibers is harmless:
    the request is capped rather than erroring or padding with noise."""
    X = np.random.default_rng(0).normal(size=(1500, 12))
    widths = {}
    for k in (1, 3, 7, 11, 20):
        P, names = pp.build_isosbestic_predictors(X, 0, mode="common",
                                                  common_method=f"pc{k}")
        widths[k] = P.shape[1]
        # own + two columns per component (both signs)
        assert (P.shape[1] - 1) % 2 == 0
    assert widths[7] == 15                   # own + 7 components x 2 signs
    assert widths[11] == widths[20] == 23    # capped at 11 other fibers


def test_more_components_do_not_break_the_fit():
    rng = np.random.default_rng(1)
    n, nroi = 4000, 12
    mot = np.convolve(rng.normal(0, 1, n + 200), np.ones(60) / 60, "same")[:n]
    mot /= mot.std()
    iso = np.stack([0.02 * mot + rng.normal(0, 0.004, n)
                    for _ in range(nroi)], 1)
    y = 0.02 * mot + rng.normal(0, 0.003, n)
    for cm in ("pc1", "pc7"):
        X, _ = pp.build_isosbestic_predictors(iso, 0, mode="common",
                                              common_method=cm)
        r, info = pp.regress_isosbestic(y, X, alpha=1e-6)
        assert np.isfinite(r).all()
        assert np.std(r) < np.std(y)


def test_hierarchical_tail_estimator_is_configurable():
    """The second stage is a different question from the first: a sparse
    tail keeps only components that earn their place, while OLS uses them
    all, which suits principal components that are orthogonal anyway."""
    rng = np.random.default_rng(0)
    n, nroi = 6000, 12
    a1 = np.convolve(rng.normal(0, 1, n + 300), np.ones(80) / 80, "same")[:n]
    a2 = np.convolve(rng.normal(0, 1, n + 300), np.ones(25) / 25, "same")[:n]
    a1, a2 = a1 / a1.std(), a2 / a2.std()
    w1, w2 = rng.uniform(.5, 1.5, nroi), rng.uniform(-1, 1, nroi)
    iso = np.stack([0.02 * (w1[j] * a1 + w2[j] * a2)
                    + rng.normal(0, 0.004, n) for j in range(nroi)], 1)
    y = 0.02 * (w1[0] * a1 + w2[0] * a2) + rng.normal(0, 0.003, n)
    X, _ = pp.build_isosbestic_predictors(iso, 0, mode="common",
                                          common_method="pc3")
    out = {}
    for tm in ("lasso", "ols", "ridge"):
        r, info = pp.regress_isosbestic(y, X, method="hierarchical",
                                        head_method="ols", tail_method=tm,
                                        alpha="cv", unpenalized=1)
        out[tm] = abs(float(np.corrcoef(r, a2)[0, 1]))
        assert info["tail_method"] == tm
        assert tm in info["alpha_source"]
    # an unpenalised tail uses every component, so it removes at least as
    # much of the second source as the sparse one
    assert out["ols"] <= out["lasso"] + 1e-6


def test_signal_channel_is_the_brighter_state_not_the_commoner_one():
    """The two LED states alternate, so their frame counts differ by at most
    one. Identifying the signal channel by which state is more COMMON is
    therefore a coin flip, and gets the 470 and 415 the wrong way round half
    the time. The rule is brightness: GCaMP emits more under 470 than at the
    isosbestic."""
    tr, led, ft = _synth_session(n_frames=8001, n_roi=4)
    counts = np.bincount(led)
    assert abs(int(counts[0]) - int(counts[1])) <= 1   # nothing to choose on

    means = {int(s): float(tr[led == s].mean()) for s in np.unique(led)}
    by_bright = max(means, key=means.get)
    dm = pp.demux_channels(tr, led, ft)
    assert dm.signal.mean() == pytest.approx(
        tr[led == by_bright].mean(), rel=1e-9)
    assert dm.isosbestic.mean() == pytest.approx(
        tr[led != by_bright].mean(), rel=1e-9)
    assert dm.signal.mean() > dm.isosbestic.mean()


def test_processed_timebase_matches_the_signal_frames():
    """The playback draws raw and processed on a shared clock; they line up
    only if the processed samples carry the signal frames' own times."""
    tr, led, ft = _synth_session(n_frames=6000, n_roi=3)
    dm = pp.demux_channels(tr, led, ft)
    means = {int(s): float(tr[led == s].mean()) for s in np.unique(led)}
    sig = led == max(means, key=means.get)
    np.testing.assert_allclose(dm.signal_t, ft[sig])


def test_session_date_prefers_the_directory_over_the_filename():
    """A filename is typed by hand and is deeper in the path than the folder,
    so searching every component together lets one transposed digit in a
    stem override a correct directory -- and a date a decade off makes every
    calibration file post-date the session."""
    from multifiber import calibration as cal

    p = Path(r"F:/data/testing/111226/ezra_3mWtest_3protsin1_111216.avi")
    assert cal.session_date_from_dirname(p) == dt.date(2026, 11, 12)

    # with no directory stamp, the filename is still used
    q = Path(r"F:/data/testing/ezra_3mWtest_111226.avi")
    assert cal.session_date_from_dirname(q) == dt.date(2026, 11, 12)


def test_implausible_years_are_rejected():
    """MMDDYY with a two-digit year is ambiguous; a session in 2016 is not a
    date this rig ran on, and returning it silently leaves no calibration
    eligible."""
    from multifiber import calibration as cal

    assert cal.session_date_from_dirname(
        Path(r"F:/data/testing/111216/x_111216.avi")) is None
    assert cal.session_date_from_dirname(
        Path(r"F:/data/testing/102626/x.avi")) == dt.date(2026, 10, 26)


def test_processed_traces_to_frame_has_what_the_readers_expect():
    """The dff table is the input to every downstream script, and they find
    ROI columns by excluding the known ones -- so a missing or extra column
    silently changes which fibers get analysed."""
    tr, led, ft = _synth_session(n_frames=4000, n_roi=3)
    proc = pp.preprocess_traces(tr, led, ft, labels=["A", "B", "C"])
    df = proc.to_frame()

    assert len(df) == proc.n_samples
    assert list(df.columns)[:2] == ["t", "frame"]
    np.testing.assert_allclose(df["t"].to_numpy(), proc.t)
    assert df["blanked"].dtype == bool
    for c in ("A", "B", "C"):
        assert c in df.columns and f"{c}__iso" in df.columns

    rois = [c for c in df.columns
            if c not in ("t", "frame", "blanked") and not c.endswith("__iso")]
    assert rois == ["A", "B", "C"]


def test_rates_inside_one_envelope_are_separate_conditions():
    """Three rates in a one-second train share a duration, so duration alone
    reports one condition and the interleaving is invisible. The gate
    channel records the rate, so it does not have to be."""
    n = 18
    w = np.full(n, 1.0)
    gp = np.r_[np.full(6, 10), np.full(6, 20), np.full(6, 40)]
    duty = np.r_[np.full(6, 0.05), np.full(6, 0.10), np.full(6, 0.20)]
    p = ev.LaserPulses(np.arange(n) * 5.0, np.arange(n) * 5.0 + w, w,
                       np.full(n, 0.23), gp, duty, w * duty)
    c = ev.classify_protocols(p, roi_id=np.ones(n, dtype=int))
    assert np.unique(c).size == 3
    for k in np.unique(c):
        assert np.unique(gp[c == k]).size == 1      # one rate per class


def test_pulse_width_at_one_rate_is_a_separate_condition():
    n = 12
    w = np.full(n, 1.0)
    gp = np.full(n, 20)
    duty = np.r_[np.full(6, 0.08), np.full(6, 0.16)]
    p = ev.LaserPulses(np.arange(n) * 5.0, np.arange(n) * 5.0 + w, w,
                       np.full(n, 0.23), gp, duty, w * duty)
    assert np.unique(ev.classify_protocols(
        p, roi_id=np.ones(n, dtype=int))).size == 2


def test_one_protocol_stays_one_class_with_gate_features():
    n = 12
    w = np.full(n, 1.0)
    p = ev.LaserPulses(np.arange(n) * 5.0, np.arange(n) * 5.0 + w, w,
                       np.full(n, 0.23), np.full(n, 20), np.full(n, 0.08),
                       w * 0.08)
    assert np.unique(ev.classify_protocols(
        p, roi_id=np.ones(n, dtype=int))).size == 1


def test_protocol_classes_report_the_gate_rate():
    n = 12
    w = np.full(n, 1.0)
    gp = np.r_[np.full(6, 10), np.full(6, 40)]
    duty = np.r_[np.full(6, 0.05), np.full(6, 0.20)]
    p = ev.LaserPulses(np.arange(n) * 5.0, np.arange(n) * 5.0 + w, w,
                       np.full(n, 0.23), gp, duty, w * duty)
    pr = ev.infer_protocol(p, p, roi_id=np.ones(n, dtype=int))
    assert pr["n_protocol_classes"] == 2
    rates = sorted(c["gate_rate_hz"] for c in pr["protocol_classes"])
    assert rates == pytest.approx([10.0, 40.0], rel=0.05)


def test_classification_splits_at_gaps_not_along_a_spread():
    """Comparing each value with the first of its group makes the test
    chain: a continuous spread wider than the tolerance gets cut wherever
    the running total happens to cross it, turning measurement noise in the
    gate duty into spurious conditions."""
    rng = np.random.default_rng(0)
    n = 40
    w = np.full(n, 0.2)
    duty = rng.normal(0.26, 0.04, n)          # one condition, measured
    p = ev.LaserPulses(np.arange(n) * 5.0, np.arange(n) * 5.0 + w, w,
                       np.full(n, 0.149), np.full(n, 16), duty, w * duty)
    assert np.unique(ev.classify_protocols(
        p, roi_id=np.ones(n, dtype=int))).size == 1


def test_a_sham_and_two_durations_give_three_conditions():
    """The design this was failing on: a 0.5 ms sham interleaved with 100
    and 200 ms, one site, one power."""
    rng = np.random.default_rng(0)
    w = np.r_[np.full(45, 0.0005), np.full(45, 0.1), np.full(45, 0.2)]
    gp = np.r_[np.full(45, 1), np.full(45, 8), np.full(45, 16)]
    duty = np.r_[rng.normal(0.04, 0.006, 45), rng.normal(0.26, 0.03, 90)]
    v = rng.normal(0.149, 0.002, 135)
    p = ev.LaserPulses(np.arange(135) * 5.0, np.arange(135) * 5.0 + w, w, v,
                       gp, duty, w * duty)
    c = ev.classify_protocols(p, roi_id=np.ones(135, dtype=int))
    assert np.unique(c).size == 3
    for k in np.unique(c):
        assert np.unique(np.round(w[c == k], 6)).size == 1


def test_short_durations_are_not_rounded_away():
    """A 0.5 ms sham labelled '0 ms' is both wrong and the condition whose
    absence is hardest to notice."""
    assert ev.format_duration(0.0005) == "0.50 ms"
    assert ev.format_duration(0.0085) == "8.5 ms"
    assert ev.format_duration(0.1) == "100 ms"
    assert ev.format_duration(1.0) == "1 s"
    assert ev.format_duration(0.0) == "?"


def test_power_is_not_split_without_usable_roi_ids():
    """Command voltage is per site; without a site it cannot be normalised,
    and splitting on the raw value separates fibers rather than powers."""
    n = 20
    w = np.full(n, 0.2)
    v = np.r_[np.full(10, 0.149), np.full(10, 0.224)]
    p = ev.LaserPulses(np.arange(n) * 5.0, np.arange(n) * 5.0 + w, w, v,
                       np.full(n, 16), np.full(n, 0.26), w * 0.26)
    assert np.unique(ev.classify_protocols(p, roi_id=None)).size == 1
    bad = np.zeros(n, dtype=int)           # unassigned
    assert np.unique(ev.classify_protocols(p, roi_id=bad)).size == 1
    good = np.ones(n, dtype=int)
    assert np.unique(ev.classify_protocols(p, roi_id=good)).size == 2
