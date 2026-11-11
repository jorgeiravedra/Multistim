"""
Animal, implant and session metadata.

The central problem this module solves: in the legacy pipeline, fiber -> region
identity lived in a hardcoded ``roi_labels`` cell array inside the analysis
script, with six superseded orderings commented out above it.  Region identity
was positional and unversioned, so a fiber-to-region mis-assignment produced a
transposed connectivity matrix with no error and no audit trail.

Here the implant map is a per-animal YAML record, loaded and validated, and
carried into every output together with the SHA-256 of the file it came from.

Nothing in this module infers metadata it cannot verify.  Fields that must be
supplied by the experimenter are required; fields that can be read from the
data (frame counts, sampling rate) are left empty here and filled by the
extraction stage.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import yaml

from .matio import file_sha256

__all__ = [
    "Fiber",
    "Implant",
    "Animal",
    "load_animal",
    "load_registry",
    "SessionMeta",
    "ANIMAL_TEMPLATE",
]

Histology = Literal["pass", "fail", "pending"]


@dataclass(frozen=True)
class Fiber:
    """One fiber in the implanted array.

    index
        1-based fiber index, matching the ROI numbering produced by the
        calibration GUI and used by the stim GUI's laser IDs.  This is the
        join key between the laser log and the photometry channels.
    region
        Anatomical target, written exactly as it should appear in figures.
    hemisphere
        'L', 'R' or 'M' (midline).
    histology
        Post hoc fiber-tip verification.  'fail' or 'pending' both set
        ``include`` to False unless explicitly overridden.
    include
        Whether this fiber contributes to analysis.  Resolved in __post_init__
        from ``histology`` unless given.
    notes
        Free text; never parsed.
    """

    index: int
    region: str
    hemisphere: str = "R"
    histology: Histology = "pending"
    include: Optional[bool] = None
    opsin: Optional[str] = None
    indicator: Optional[str] = None
    fiber_length_mm: Optional[float] = None
    notes: str = ""

    def __post_init__(self):
        if self.index < 1:
            raise ValueError(f"fiber index must be 1-based, got {self.index}")
        if self.hemisphere not in ("L", "R", "M"):
            raise ValueError(
                f"fiber {self.index}: hemisphere must be L/R/M, "
                f"got {self.hemisphere!r}"
            )
        if self.histology not in ("pass", "fail", "pending"):
            raise ValueError(
                f"fiber {self.index}: histology must be pass/fail/pending, "
                f"got {self.histology!r}"
            )
        if self.include is None:
            object.__setattr__(self, "include", self.histology == "pass")

    @property
    def label(self) -> str:
        return f"{self.region}_{self.hemisphere}"


@dataclass
class Implant:
    """The physical array: which fiber index sits in which region.

    Calibration files are *not* named here.  They describe the rig, not the
    animal — they are regenerated when the patchcord is reseated and reused
    across animals until the next recalibration — so they are resolved per
    session by :mod:`multifiber.calibration`.  ``calibration_overrides`` pins a
    specific file for this animal when the date-based resolution is wrong.
    """

    fibers: list[Fiber]
    array_id: Optional[str] = None
    surgery_date: Optional[_dt.date] = None
    calibration_overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        idx = [f.index for f in self.fibers]
        if not idx:
            raise ValueError("implant has no fibers")
        if len(set(idx)) != len(idx):
            dupes = sorted({i for i in idx if idx.count(i) > 1})
            raise ValueError(f"{self.array_id}: duplicate fiber indices {dupes}")
        expected = list(range(1, len(idx) + 1))
        if sorted(idx) != expected:
            raise ValueError(
                f"{self.array_id}: fiber indices {sorted(idx)} are not "
                f"contiguous 1..{len(idx)}; the ROI ordering from the "
                "calibration GUI is positional, so gaps are unresolvable"
            )
        self.fibers = sorted(self.fibers, key=lambda f: f.index)

    @property
    def n_fibers(self) -> int:
        return len(self.fibers)

    def region_map(self, included_only: bool = False) -> dict[int, str]:
        """1-based fiber index -> region label."""
        return {
            f.index: f.label
            for f in self.fibers
            if (f.include or not included_only)
        }

    def ordered_labels(self) -> list[str]:
        """Region labels in fiber-index order, i.e. Region0G..Region{n-1}G."""
        return [f.label for f in self.fibers]


@dataclass
class Animal:
    animal_id: str
    implant: Implant
    genotype: Optional[str] = None
    sex: Optional[str] = None
    dob: Optional[_dt.date] = None
    cohort: Optional[str] = None
    condition: Optional[str] = None  # EXP / NON / OBS
    notes: str = ""
    source_path: Optional[str] = None
    source_sha256: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_date(value) -> Optional[_dt.date]:
    if value in (None, ""):
        return None
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value))


def load_animal(path: str | Path) -> Animal:
    """Load and validate one ``*.animal.yaml`` record."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")

    try:
        # 'fibers' may sit at the top level or under 'implant'; the former
        # keeps a minimal record to two keys.
        imp_doc = doc.get("implant") or doc
        fibers = [Fiber(**f) for f in imp_doc["fibers"]]
        implant = Implant(
            fibers=fibers,
            array_id=(str(imp_doc["array_id"])
                      if imp_doc.get("array_id") else None),
            surgery_date=_parse_date(imp_doc.get("surgery_date")),
            calibration_overrides=dict(imp_doc.get("calibration_overrides") or {}),
        )
        animal = Animal(
            animal_id=str(doc["animal_id"]),
            implant=implant,
            genotype=doc.get("genotype"),
            sex=doc.get("sex"),
            dob=_parse_date(doc.get("dob")),
            cohort=doc.get("cohort"),
            condition=doc.get("condition"),
            notes=doc.get("notes", ""),
            source_path=str(path),
            source_sha256=file_sha256(path),
        )
    except KeyError as exc:
        raise KeyError(f"{path}: missing required field {exc}") from exc
    except TypeError as exc:
        raise TypeError(f"{path}: {exc}") from exc
    return animal


def load_registry(root: str | Path, pattern: str = "*.animal.yaml") -> dict[str, Animal]:
    """Load every animal record under ``root`` into an ``animal_id`` -> Animal map."""
    root = Path(root)
    out: dict[str, Animal] = {}
    for p in sorted(root.rglob(pattern)):
        a = load_animal(p)
        if a.animal_id in out:
            raise ValueError(
                f"duplicate animal_id {a.animal_id!r}: "
                f"{out[a.animal_id].source_path} and {p}"
            )
        out[a.animal_id] = a
    return out


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #

@dataclass
class Epoch:
    """A named, contiguous period within a session, in seconds from recording
    start.  Drives the segment-wise baseline fit, which in the legacy code was
    a pair of hardcoded literals (``seg_start=2200``, ``decay_remove=600``)."""

    name: str
    start_s: float
    stop_s: Optional[float] = None  # None = to end of recording


@dataclass
class LedSchedule:
    """Which LEDs are interleaved and at what camera rate.

    camera_fps
        Camera frame rate (all LEDs interleaved), e.g. 80.
    sequence
        Per-frame LED cycle, e.g. ``['isos', 'green']`` for 415/470 alternation
        giving 40 Hz effective per channel.
    wavelengths_nm
        Mapping from channel name to excitation wavelength, for the record.
    """

    camera_fps: float
    sequence: list[str]
    wavelengths_nm: dict[str, int] = field(
        default_factory=lambda: {"isos": 415, "green": 470, "red": 560}
    )

    @property
    def period(self) -> int:
        return len(self.sequence)

    @property
    def effective_fps(self) -> float:
        return self.camera_fps / self.period

    def channel_of_frame(self, frame_index: Sequence[int] | int):
        """Nominal channel for a 0-based frame index, assuming no dropped
        frames.  Used only as a prior; the authoritative assignment comes from
        :mod:`multifiber.events`."""
        import numpy as np

        idx = np.asarray(frame_index)
        return np.asarray(self.sequence, dtype=object)[idx % self.period]


@dataclass
class SessionMeta:
    """Everything known about one recording before the video is opened."""

    session_id: str
    animal_id: str
    date: Optional[_dt.date] = None
    paths: dict[str, str] = field(default_factory=dict)
    led: Optional[LedSchedule] = None
    epochs: list[Epoch] = field(default_factory=list)
    protocol: dict[str, Any] = field(default_factory=dict)
    sha256: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def epoch_bounds(self, name: str, fps: float, n_samples: int) -> tuple[int, int]:
        """Resolve an epoch to sample indices at ``fps``."""
        for e in self.epochs:
            if e.name == name:
                i0 = int(round(e.start_s * fps))
                i1 = n_samples if e.stop_s is None else int(round(e.stop_s * fps))
                return max(0, i0), min(n_samples, i1)
        raise KeyError(f"{self.session_id}: no epoch named {name!r} "
                       f"(have: {[e.name for e in self.epochs]})")


ANIMAL_TEMPLATE = """\
# multifiber animal record: <animal_id>.animal.yaml
#
# Minimal form is two keys: animal_id and fibers.  'index' is the 1-based ROI
# number from the calibration GUI, which is also the laser ID in the stim log;
# it is the only join between laser events and photometry channels.  Get it
# wrong and the connectivity matrix is permuted with no error raised.
#
# Calibration files are NOT listed here -- they describe the rig, are shared
# across animals, and are resolved per session by date.  Pin one only to
# override that (see the commented block at the bottom).
animal_id: REQUIRED
fibers:
  - {index: 1,  region: VMHvl, hemisphere: R}
  - {index: 2,  region: AH,    hemisphere: R}
  - {index: 3,  region: PrL,   hemisphere: R}
  - {index: 4,  region: PMv,   hemisphere: R}
  - {index: 5,  region: MeA,   hemisphere: R}
  - {index: 6,  region: LHb,   hemisphere: R}
  - {index: 7,  region: lPAG,  hemisphere: R}
  - {index: 8,  region: PA,    hemisphere: R}
  - {index: 9,  region: NAc,   hemisphere: R}
  - {index: 10, region: vLS,   hemisphere: R}
  - {index: 11, region: mPOA,  hemisphere: M}
  - {index: 12, region: BNST,  hemisphere: R}

# Everything below is optional.
# genotype: Vgat-Cre;CD1
# sex: M
# cohort:
# condition:          # EXP / NON / OBS
#
# Per-fiber extras, added as histology comes in.  histology defaults to
# 'pending', which leaves a fiber out of region_map(included_only=True):
#   - {index: 1, region: VMHvl, hemisphere: R, histology: pass}
#
# calibration_overrides:
#   translated_coords: TranslatedCoords_100726.mat
#   mids: mids_100726.mat
"""
