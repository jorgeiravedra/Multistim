"""
Session discovery.

Input is a directory.  Output is a list of :class:`SessionCandidate`, each a
stem (e.g. ``derek_10Hzburst_testmpoa``) with its companion files resolved and
its completeness checked.  Nothing is processed here and nothing is guessed
silently: every file a session needs is either found, or the session carries an
explicit problem describing what is missing.

File conventions, from the rig code:

    <stem>.avi                      Hamamatsu video (two-colour, side by side)
    <stem>_bcam.mat                 DAQ log (cell array ``bcl``)
    <stem>.mat                      stim GUI log (cell array ``pl``)
    <stem>_daqlog.csv               CSV export of the DAQ log (derived)
    <stem>_extracted_signal.csv     NPM-format traces (derived)
    <stem>_laser_times.csv          laser events (derived)
    TranslatedCoords_MMDDYY.mat     fiber ROI centres + radii
    mids_MMDDYY.mat                 ROI offset
    cal_mtx_MMDDYY.mat              galvo voltage calibration
    lsr_ints_MMDDYY.mat             per-fiber laser power calibration

Calibration files are shared rig state, not per-animal, so they are resolved by
date (most recent at or before the session) rather than named in each animal
record.  That resolution is a heuristic; :mod:`multifiber.calibration` records
the chosen file, its SHA-256, the runner-up and the gap in days on every
session so the choice is auditable, and an explicit override always wins.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .calibration import (
    CALIBRATION_PATTERNS,
    CalibrationSet,
    resolve_calibration,
)
from .matio import LARGE_FILE_BYTES, file_fingerprint, file_sha256
from .metadata import Animal, SessionMeta

__all__ = [
    "SessionCandidate",
    "discover_sessions",
]

#: Derived/companion suffixes that must not be mistaken for a session stem.
_DERIVED_SUFFIXES = (
    "_bcam",
    "_daqlog",
    "_extracted_signal",
    "_laser_times",
    "_organized",
    "_timestamps",
)

#: Filename fragments that mark a file as an export rather than a recording.
#: A camera timestamp dump such as
#: ``20261026_154611_0-479999999_test7_timestamps.mat`` otherwise looks like a
#: session stem with every companion file missing.
_NOT_A_SESSION = ("_timestamps",)


@dataclass
class SessionCandidate:
    """One recording and its resolved companion files."""

    stem: str
    root: Path
    animal_assumed: bool = False
    video: Optional[Path] = None
    bcam: Optional[Path] = None
    stim_log: Optional[Path] = None
    sidecar: Optional[Path] = None
    calibration: Optional[CalibrationSet] = None
    derived: dict[str, Path] = field(default_factory=dict)
    animal: Optional[Animal] = None
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def session_id(self) -> str:
        return self.stem

    def file_hashes(self, full: bool = False) -> dict[str, str]:
        """Identity of every input file.

        Files at or above ``matio.LARGE_FILE_BYTES`` get a fingerprint
        (size, mtime and the first and last 8 MB) rather than a full hash,
        because hashing a multi-gigabyte video reads the entire file and would
        dominate a run that otherwise reuses the extracted traces.  Pass
        ``full=True`` to hash everything; the pipeline does so whenever
        extraction actually runs, so a processed trace always carries a
        complete hash of the video it came from.
        """
        out: dict[str, str] = {}
        for name, p in [
            ("video", self.video),
            ("bcam", self.bcam),
            ("stim_log", self.stim_log),
            ("sidecar", self.sidecar),
        ]:
            if p is not None and p.exists():
                big = p.stat().st_size >= globals()["LARGE_FILE_BYTES"]
                out[name] = (file_fingerprint(p) if big and not full
                             else file_sha256(p))
        if self.calibration is not None:
            for fam, choice in self.calibration.choices.items():
                if choice.sha256:
                    out[f"calibration/{fam}"] = choice.sha256
        return out

    def to_meta(self) -> SessionMeta:
        if self.animal is None:
            raise ValueError(f"{self.stem}: no animal record bound")
        paths = {
            k: str(v)
            for k, v in [
                ("video", self.video),
                ("bcam", self.bcam),
                ("stim_log", self.stim_log),
                ("sidecar", self.sidecar),
            ]
            if v is not None
        }
        if self.calibration is not None:
            paths.update({f"calibration/{fam}": str(c.path)
                          for fam, c in self.calibration.choices.items()
                          if c.path is not None})
        return SessionMeta(
            session_id=self.stem,
            animal_id=self.animal.animal_id,
            date=_date_from_mtime(self.video or self.bcam),
            paths=paths,
            sha256=self.file_hashes(),
        )

    def summary(self) -> str:
        status = "OK " if self.ok else "BAD"
        cal = ""
        if self.calibration is not None:
            c = self.calibration.choices.get("translated_coords")
            if c is not None and c.path is not None:
                cal = f"  cal={c.stamp}({c.source})"
        aid = self.animal.animal_id if self.animal else "?"
        if self.animal_assumed:
            aid += " (ASSUMED)"
        lines = [f"[{status}] {self.stem}  animal={aid}{cal}"]
        for p in self.problems:
            lines.append(f"       x {p}")
        for w in self.warnings:
            lines.append(f"       ! {w}")
        return "\n".join(lines)


def _date_from_mtime(path: Optional[Path]) -> Optional[_dt.date]:
    if path is None or not path.exists():
        return None
    return _dt.date.fromtimestamp(path.stat().st_mtime)


def _strip_derived(stem: str) -> Optional[str]:
    """Return the session stem a derived filename belongs to, else None."""
    for suf in _DERIVED_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return None


def _match_animal(stem: str, registry: dict[str, Animal]) -> Optional[Animal]:
    """Bind a session to an animal by longest matching id in the filename.

    Longest-match avoids the failure mode where animal ids are prefixes of each
    other ('mark' vs 'markii').  Matching is on underscore-delimited tokens so
    that an id cannot match inside an unrelated word.
    """
    tokens = set(re.split(r"[_\-\.]+", stem.lower()))
    hits = [a for aid, a in registry.items() if aid.lower() in tokens]
    if not hits:
        return None
    return max(hits, key=lambda a: len(a.animal_id))


def discover_sessions(
    root: str | Path,
    registry: dict[str, Animal] | None = None,
    video_ext: Iterable[str] = (".avi",),
    require_video: bool = True,
    calibration_dir: str | Path | None = None,
    session_date: Optional["_dt.date"] = None,
    default_animal: Optional[str] = None,
) -> list[SessionCandidate]:
    """Walk ``root`` and return one candidate per recording, validated.

    Parameters
    ----------
    root
        Directory to scan, recursively.
    registry
        ``animal_id -> Animal`` from :func:`multifiber.metadata.load_registry`.
        If None, sessions are returned unbound with a problem recorded; the
        region map is mandatory for any downstream stage.
    require_video
        If False, a session with a ``_bcam.mat`` but no video is still usable
        (e.g. re-deriving events from an already-extracted CSV).
    calibration_dir
        Directory holding the shared calibration files.  The session's own
        directory is always searched FIRST and wins: the rig's MATLAB does
        ``cd(rawDataDir)`` before loading, so a calibration copied beside the
        data is the one that was actually applied.  The shared directory
        (``TranslatedCoords_*``, ``mids_*``, ``cal_mtx_*``, ``lsr_ints_*``),
        e.g. ``F:\\Jorge\\Tests\\multistim``.  For each session, the most
        recent file at or before the session date is used; see
        :func:`multifiber.calibration.resolve_calibration` for the rule and the
        evidence it records.  Defaults to ``root``.
    default_animal
        ``animal_id`` to fall back on when a filename matches no record.  The
        implant map is per-animal: applying one animal's fiber->region mapping
        to another silently relabels every row and column of the connectivity
        matrix, with no error at any stage.  A session bound this way is
        therefore marked ``animal_assumed`` and carries a warning that is
        copied into its manifest, so an output built on the assumption can be
        told from one built on a record.  Use it for rig tests, not for data
        that will be analysed.
    session_date
        Override the inferred session date for every session.  Normally left
        None, in which case the date comes from a ``MMDDYY`` component of the
        session path (rig convention), falling back to file mtime with a
        warning.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    registry = registry or {}

    cal_dir = Path(calibration_dir) if calibration_dir is not None else root
    video_ext = tuple(e.lower() for e in video_ext)

    # ---- collect stems ---------------------------------------------------- #
    stems: dict[str, Path] = {}  # stem -> containing directory
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        stem = p.stem
        if any(frag in stem for frag in _NOT_A_SESSION):
            continue
        if suffix in video_ext:
            stems.setdefault(stem, p.parent)
        elif suffix == ".mat":
            if any(pat.match(p.name) for pat in CALIBRATION_PATTERNS.values()):
                continue
            base = _strip_derived(stem)
            if base is not None:
                stems.setdefault(base, p.parent)
            else:
                stems.setdefault(stem, p.parent)

    candidates: list[SessionCandidate] = []
    for stem in sorted(stems):
        d = stems[stem]
        cand = SessionCandidate(stem=stem, root=d)

        for ext in video_ext:
            cands = sorted(d.glob(f"{stem}{ext}"))
            if cands:
                cand.video = cands[0]
                break

        bcam = d / f"{stem}_bcam.mat"
        cand.bcam = bcam if bcam.exists() else None

        stim = d / f"{stem}.mat"
        cand.stim_log = stim if stim.exists() else None

        for sc in (d / f"{stem}.session.yaml", d / f"{stem}.session.yml"):
            if sc.exists():
                cand.sidecar = sc
                break

        for name, suf in [
            ("daqlog_csv", "_daqlog.csv"),
            ("signal_csv", "_extracted_signal.csv"),
            ("laser_times_csv", "_laser_times.csv"),
        ]:
            p = d / f"{stem}{suf}"
            if p.exists():
                cand.derived[name] = p

        # ---- bind animal and calibration ---------------------------------- #
        animal = _match_animal(stem, registry)
        if animal is None and default_animal is not None:
            animal = registry.get(default_animal)
            if animal is None:
                cand.problems.append(
                    f"default_animal={default_animal!r} is not in the registry"
                )
            else:
                cand.animal_assumed = True
                cand.warnings.append(
                    f"no animal record matched this filename; ASSUMING "
                    f"{default_animal!r}. The fiber->region map is "
                    "animal-specific, so if this is a different animal every "
                    "region label is wrong and the connectivity matrix is "
                    "permuted."
                )
        cand.animal = animal
        if animal is None:
            cand.problems.append(
                "no animal record matched this filename; create "
                "<animal_id>.animal.yaml (see metadata.ANIMAL_TEMPLATE), or "
                "pass default_animal= for a rig test"
            )
        else:
            cal = resolve_calibration(
                [d, cal_dir] if cal_dir != d else [d],
                session_date=session_date,
                session_path=(cand.video or cand.bcam or d),
                fallback_mtime_path=cand.bcam or cand.video,
                overrides=animal.implant.calibration_overrides,
            )
            cand.calibration = cal
            cand.problems.extend(cal.problems)
            cand.warnings.extend(cal.warnings)

        # ---- completeness -------------------------------------------------- #
        if cand.video is None and require_video:
            cand.problems.append("no video file found")
        if cand.bcam is None:
            cand.problems.append(f"no {stem}_bcam.mat (DAQ log) found")
        if cand.stim_log is None:
            cand.warnings.append(
                f"no {stem}.mat stim log; laser events will have no ROI "
                "identity unless the session is a fixed-site protocol"
            )

        candidates.append(cand)

    return candidates
