"""
Calibration file resolution.

The calibration files (``TranslatedCoords_*``, ``mids_*``, ``cal_mtx_*``,
``lsr_ints_*``) describe the *rig*, not the animal: they are regenerated when
the patchcord is reseated or the galvos drift, and are reused across animals
until the next recalibration.  They therefore live in one directory and are
resolved per session rather than being named in each animal record.

Resolution rule: for each family, take the most recent file whose date stamp is
at or before the session date.

Date stamps come from the **filename** (``MMDDYY``), not the file's mtime.
Mtime is reset by copying, and these files routinely move between the rig PC
and the data drive, so mtime would silently re-rank the candidates after an
archival copy.

This is a heuristic, and heuristics that pick files are exactly how ROIs end up
drawn in the wrong place.  Three things make it auditable rather than implicit:

* the resolved path and its SHA-256 are recorded on the session, so a later
  reader can verify what was used;
* the runner-up (the next-older candidate) and the gap in days are reported,
  so a suspicious resolution is visible without re-deriving it;
* an explicit override in the animal record or the session sidecar always wins.

``TranslatedCoords_*`` and ``mids_*`` are written as a pair by the calibration
GUI and must resolve to the same stamp; a split pair is an error.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .matio import file_sha256

__all__ = [
    "CALIBRATION_PATTERNS",
    "PAIRED_FAMILIES",
    "parse_stamp",
    "session_date_from_dirname",
    "CalibrationChoice",
    "CalibrationSet",
    "index_calibration_dir",
    "resolve_calibration",
]

#: family -> regex capturing the MMDDYY stamp.
CALIBRATION_PATTERNS: dict[str, re.Pattern] = {
    "translated_coords": re.compile(r"^TranslatedCoords_(\d{6})\.mat$", re.I),
    "mids": re.compile(r"^mids_(\d{6})\.mat$", re.I),
    "cal_mtx": re.compile(r"^cal_mtx_(\d{6})\.mat$", re.I),
    "laser_intensities": re.compile(r"^lsr_ints_(\d{6})\.mat$", re.I),
}

#: Families written together by one GUI run; their stamps must agree.
PAIRED_FAMILIES: tuple[tuple[str, str], ...] = (("translated_coords", "mids"),)

#: Families without which ROI geometry is undefined.
REQUIRED_FAMILIES: tuple[str, ...] = ("translated_coords", "mids")

_DIRNAME_STAMP = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def parse_stamp(stamp: str, century: int = 2000) -> _dt.date:
    """``MMDDYY`` -> date.  Two-digit years are assumed to be 20YY."""
    if not re.fullmatch(r"\d{6}", stamp):
        raise ValueError(f"not a MMDDYY stamp: {stamp!r}")
    mm, dd, yy = int(stamp[:2]), int(stamp[2:4]), int(stamp[4:])
    return _dt.date(century + yy, mm, dd)


def session_date_from_dirname(path: Path) -> Optional[_dt.date]:
    """Infer a session date from a ``MMDDYY`` component of the path.

    Rig convention puts the acquisition date in the directory name
    (``.../testing/101926``).  The *deepest* matching component wins, since a
    date further down the tree is more specific to the session.
    """
    for part in reversed(path.parts):
        m = _DIRNAME_STAMP.search(part)
        if m:
            try:
                return parse_stamp(m.group(1))
            except ValueError:
                continue
    return None


@dataclass
class CalibrationChoice:
    """One resolved calibration file and the evidence for the choice."""

    family: str
    path: Optional[Path]
    stamp: Optional[str] = None
    date: Optional[_dt.date] = None
    age_days: Optional[int] = None
    runner_up: Optional[str] = None       # next-older candidate's stamp
    newer_available: Optional[str] = None  # nearest stamp *after* the session
    source: str = "resolved"               # resolved | override | missing
    sha256: Optional[str] = None

    def __str__(self) -> str:
        if self.path is None:
            return f"{self.family}: MISSING"
        return (f"{self.family}: {self.path.name} "
                f"({self.source}, {self.age_days}d before session)")


@dataclass
class CalibrationSet:
    choices: dict[str, CalibrationChoice] = field(default_factory=dict)
    session_date: Optional[_dt.date] = None
    session_date_source: str = "unknown"
    directory: Optional[Path] = None
    searched: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def path(self, family: str) -> Optional[Path]:
        c = self.choices.get(family)
        return c.path if c else None

    @property
    def ok(self) -> bool:
        return not self.problems

    def provenance(self) -> dict:
        return {
            "session_date": self.session_date.isoformat() if self.session_date else None,
            "session_date_source": self.session_date_source,
            "directory": str(self.directory) if self.directory else None,
            "searched": list(self.searched),
            "files": {
                fam: {
                    "path": str(c.path) if c.path else None,
                    "stamp": c.stamp,
                    "age_days": c.age_days,
                    "source": c.source,
                    "sha256": c.sha256,
                    "runner_up": c.runner_up,
                    "newer_available": c.newer_available,
                }
                for fam, c in self.choices.items()
            },
        }

    def summary(self) -> str:
        head = (f"calibration @ {self.session_date} "
                f"(date from {self.session_date_source})")
        body = [f"  {c}" for c in self.choices.values()]
        body += [f"  x {p}" for p in self.problems]
        body += [f"  ! {w}" for w in self.warnings]
        return "\n".join([head, *body])


def index_calibration_dir(
    directory: "str | Path | Sequence[str | Path]",
    families: Iterable[str] = tuple(CALIBRATION_PATTERNS),
) -> dict[str, dict[_dt.date, Path]]:
    """family -> {date: path} for calibration files under one or more roots.

    Earlier roots win on a date collision, so the session's own directory
    takes precedence over the shared calibration directory.  The rig's MATLAB
    ``cd(rawDataDir)`` before loading means a session's calibration is often
    copied in beside its data, and that copy is the one that was actually
    applied.
    """
    roots = ([Path(directory)] if isinstance(directory, (str, Path))
             else [Path(d) for d in directory])
    index: dict[str, dict[_dt.date, Path]] = {f: {} for f in families}
    seen_any = False
    for directory in roots:
        if not directory.is_dir():
            continue
        seen_any = True
        for p in directory.rglob("*.mat"):
            for family in index:
                m = CALIBRATION_PATTERNS[family].match(p.name)
                if not m:
                    continue
                try:
                    d = parse_stamp(m.group(1))
                except ValueError:
                    continue
                index[family].setdefault(d, p)
    if not seen_any:
        raise NotADirectoryError(f"no searchable directory in {roots}")
    return index


def resolve_calibration(
    directory: "str | Path | Sequence[str | Path]",
    session_date: Optional[_dt.date] = None,
    session_path: Optional[Path] = None,
    fallback_mtime_path: Optional[Path] = None,
    overrides: Optional[dict[str, str]] = None,
    max_age_days: int = 45,
    require_pairs: bool = True,
) -> CalibrationSet:
    """Resolve the calibration set in force for one session.

    Parameters
    ----------
    directory
        Where calibration files live.  May be several directories, searched in
        order with earlier ones winning: pass the session's own directory
        first and the shared calibration directory second, because a
        calibration copied in beside the data is the one that was applied.
    session_date
        Explicit session date.  If None, inferred from a ``MMDDYY`` component
        of ``session_path``, and failing that from the mtime of
        ``fallback_mtime_path`` (a last resort: mtime is reset by copying, so
        this is reported as a low-confidence source).
    overrides
        ``family -> filename`` pinned in the animal record or session sidecar.
        An override always wins and is never date-checked, which is the escape
        hatch for a session whose calibration was regenerated mid-day.
    max_age_days
        A resolved file older than this raises a warning.  It does not fail:
        a long gap is normal when the rig has not been recalibrated, but it is
        also what a missing calibration file looks like.
    """
    roots = ([Path(directory)] if isinstance(directory, (str, Path))
             else [Path(d) for d in directory])
    cs = CalibrationSet(directory=roots[0])
    cs.searched = [str(r) for r in roots]
    overrides = overrides or {}

    # ---- session date --------------------------------------------------- #
    if session_date is not None:
        cs.session_date, cs.session_date_source = session_date, "explicit"
    elif session_path is not None and (d := session_date_from_dirname(Path(session_path))):
        cs.session_date, cs.session_date_source = d, "path MMDDYY"
    elif fallback_mtime_path is not None and Path(fallback_mtime_path).exists():
        cs.session_date = _dt.date.fromtimestamp(
            Path(fallback_mtime_path).stat().st_mtime)
        cs.session_date_source = "file mtime (low confidence)"
        cs.warnings.append(
            "session date taken from file mtime, which is reset by copying; "
            "pin it in the session sidecar if this session has been archived"
        )
    else:
        cs.problems.append(
            "no session date could be determined; calibration cannot be "
            "resolved by date — pass session_date or pin the files explicitly"
        )

    index = index_calibration_dir(roots)

    for family in CALIBRATION_PATTERNS:
        if family in overrides and overrides[family]:
            name = overrides[family]
            hits = [h for r in roots for h in r.rglob(name)]
            if hits:
                m = CALIBRATION_PATTERNS[family].match(hits[0].name)
                stamp = m.group(1) if m else None
                cs.choices[family] = CalibrationChoice(
                    family=family, path=hits[0], stamp=stamp,
                    date=parse_stamp(stamp) if stamp else None,
                    source="override", sha256=file_sha256(hits[0]),
                    age_days=((cs.session_date - parse_stamp(stamp)).days
                              if stamp and cs.session_date else None),
                )
            else:
                cs.choices[family] = CalibrationChoice(
                    family=family, path=None, source="missing")
                cs.problems.append(
                    f"{family}: pinned file {name!r} not found under {roots}")
            continue

        cands = index.get(family, {})
        if not cands:
            cs.choices[family] = CalibrationChoice(
                family=family, path=None, source="missing")
            msg = (f"{family}: no files matching the naming convention in "
                   f"{', '.join(str(r) for r in roots)}")
            (cs.problems if family in REQUIRED_FAMILIES else cs.warnings).append(msg)
            continue

        if cs.session_date is None:
            cs.choices[family] = CalibrationChoice(
                family=family, path=None, source="missing")
            continue

        dates = sorted(cands)
        eligible = [d for d in dates if d <= cs.session_date]
        newer = [d for d in dates if d > cs.session_date]
        if not eligible:
            cs.choices[family] = CalibrationChoice(
                family=family, path=None, source="missing",
                newer_available=min(newer).strftime("%m%d%y") if newer else None)
            msg = (f"{family}: every candidate post-dates the session "
                   f"({cs.session_date}); earliest is {min(dates)}")
            (cs.problems if family in REQUIRED_FAMILIES else cs.warnings).append(msg)
            continue

        chosen = max(eligible)
        prior = [d for d in eligible if d < chosen]
        path = cands[chosen]
        age = (cs.session_date - chosen).days
        cs.choices[family] = CalibrationChoice(
            family=family,
            path=path,
            stamp=chosen.strftime("%m%d%y"),
            date=chosen,
            age_days=age,
            runner_up=max(prior).strftime("%m%d%y") if prior else None,
            newer_available=min(newer).strftime("%m%d%y") if newer else None,
            source="resolved",
            sha256=file_sha256(path),
        )
        if age > max_age_days:
            cs.warnings.append(
                f"{family}: resolved file is {age} days older than the session "
                f"({path.name}); confirm the rig was not recalibrated since"
            )
        if newer and (min(newer) - cs.session_date).days <= 1:
            cs.warnings.append(
                f"{family}: a newer calibration ({min(newer)}) exists within a "
                "day of the session; if it was generated before the recording, "
                "pin it explicitly"
            )

    # ---- paired families must agree -------------------------------------- #
    if require_pairs:
        for a, b in PAIRED_FAMILIES:
            ca, cb = cs.choices.get(a), cs.choices.get(b)
            if ca and cb and ca.stamp and cb.stamp and ca.stamp != cb.stamp:
                cs.problems.append(
                    f"{a} ({ca.stamp}) and {b} ({cb.stamp}) resolved to "
                    "different dates; the calibration GUI writes them together, "
                    "so a split pair means one is missing from the directory"
                )

    return cs
