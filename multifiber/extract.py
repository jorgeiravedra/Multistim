"""
ROI trace extraction from the raw video.

Port of the MATLAB extraction loop, with the same arithmetic (one sparse
matrix-vector product per frame over ROI pixels only) and three differences:

* the frame count is *measured*, not taken from the container header, because
  an AVI header's ``nb_frames`` is frequently absent or wrong and the whole
  trigger-to-frame alignment depends on it;
* the per-frame mean over all ROI pixels is returned alongside the per-ROI
  traces, so LED state can be recovered from image brightness (the DAQ LED
  line is unconnected on this rig);
* the ROI geometry is built once and kept, so the identical mask can be
  re-applied without redrawing.

Sensor layout
-------------
The camera writes two emission channels side by side on one sensor.  In this
cohort only the **left** half carries a recorded indicator (Cre-off GCaMP6f);
the right half carries opsin fluorescence (rsChRmine-oScarlet) and is used, if
at all, as a check on expression rather than as a signal.  ``half='left'`` is
therefore the default.  The acquisition crop is ``[0 320 1152 476]``, i.e. two
576-px halves; ``translated_coords - mids`` places ROI centres inside the half
that is used, which is the same convention the MATLAB figure draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from .matio import file_sha256, load_calibration

__all__ = [
    "VideoInfo",
    "probe_video",
    "iter_video_frames",
    "RoiMasks",
    "build_roi_masks",
    "ExtractionResult",
    "extract_traces",
    "extract_session",
]


# --------------------------------------------------------------------------- #
# Video probing
# --------------------------------------------------------------------------- #

@dataclass
class VideoInfo:
    """Video metadata.  ``n_frames`` is authoritative only when ``counted``."""

    path: Path
    width: int
    height: int
    n_frames: int
    n_frames_reported: Optional[int]
    fps: Optional[float]
    pix_fmt: Optional[str]
    counted: bool
    sha256: Optional[str] = None

    def describe(self) -> str:
        src = "decoded" if self.counted else "container header"
        line = (f"{self.path.name}: {self.width}x{self.height}, "
                f"{self.n_frames} frames ({src})")
        if (self.n_frames_reported is not None
                and self.n_frames_reported != self.n_frames):
            line += f" — header claimed {self.n_frames_reported}"
        return line


def probe_video(
    path: str | Path,
    count_frames: bool = False,
    hash_file: bool = False,
    require_count: bool = True,
) -> VideoInfo:
    """Inspect a video, measuring the frame count rather than trusting it.

    ``count_frames=True`` demuxes the whole file to count packets.  This costs
    a pass over the file but no decoding, and it is the number the
    trigger-to-frame alignment depends on: an AVI written by the acquisition
    GUI often reports ``nb_frames = 0``, and a header count that disagrees with
    the real one shifts every peri-event window.

    ``count_frames`` defaults to False because counting demuxes the entire
    file, and :func:`extract_traces` walks every frame anyway and returns the
    authoritative count -- counting first simply doubles the passes over the
    video.  Set it True when the count is needed without extracting.

    With ``require_count=False`` a missing header count is tolerated and
    ``n_frames`` is left at 0 with ``counted=False``; such a result must not
    be fed to :func:`multifiber.events.align_triggers_to_frames` until the
    real count is known.
    """
    import av

    path = Path(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        reported = stream.frames or None
        width, height = stream.codec_context.width, stream.codec_context.height
        pix_fmt = stream.codec_context.pix_fmt
        fps = float(stream.average_rate) if stream.average_rate else None
        n = reported
        counted = False
        if count_frames:
            stream.thread_type = "AUTO"
            n = sum(1 for _ in container.demux(stream) if _.size)
            counted = True

    if n is None:
        if require_count:
            raise ValueError(
                f"{path.name}: container reports no frame count; "
                "re-probe with count_frames=True"
            )
        n = 0
    return VideoInfo(
        path=path, width=width, height=height, n_frames=int(n),
        n_frames_reported=reported, fps=fps, pix_fmt=pix_fmt, counted=counted,
        sha256=file_sha256(path) if hash_file else None,
    )


def iter_video_frames(
    path: str | Path,
    plane: int = 0,
) -> Iterator[np.ndarray]:
    """Yield successive frames as 2-D arrays.

    Frames are decoded sequentially.  The MATLAB used ``read(v, n)``, which
    re-seeks and re-decodes from the preceding keyframe on every call; for a
    30-minute recording that is the dominant cost of extraction.

    ``plane`` selects the colour plane for an RGB-coded greyscale video, which
    is what the acquisition writes (all three planes carry the same data).
    """
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")
            yield arr[:, :, plane] if arr.ndim == 3 else arr


# --------------------------------------------------------------------------- #
# ROI geometry
# --------------------------------------------------------------------------- #

@dataclass
class RoiMasks:
    """Per-ROI pixel masks, stored as a sparse averaging operator.

    ``W @ pixels[pix_idx]`` gives the mean pixel value within each ROI, which
    is one sparse matvec per frame over ROI pixels only.
    """

    W: "object"                   # scipy.sparse matrix, (n_roi, n_pix)
    pix_idx: np.ndarray           # flat indices of every ROI pixel
    roi_sizes: np.ndarray         # pixel count per ROI
    shape: tuple[int, int]        # (height, width) of the image half
    centres: np.ndarray           # (n_roi, 2) x, y within the half
    radii: np.ndarray             # (n_roi,) in pixels
    roi_fraction: float
    half: str
    labels: Optional[list[str]] = None
    provenance: dict = field(default_factory=dict)

    @property
    def n_roi(self) -> int:
        return int(self.roi_sizes.size)

    def label_image(self) -> np.ndarray:
        """ROI id per pixel (0 = background, 1-based ids), for QC figures."""
        L = np.zeros(self.shape, dtype=np.int16)
        flat = L.ravel()
        roi_of_pixel = np.repeat(np.arange(1, self.n_roi + 1),
                                 self.roi_sizes.astype(int))
        flat[self.pix_idx] = roi_of_pixel
        return L


def build_roi_masks(
    translated_coords: np.ndarray,
    mids: np.ndarray,
    image_shape: tuple[int, int],
    roi_fraction: float = 0.75,
    half: str = "left",
    half_width: Optional[int] = None,
    labels: Optional[Sequence[str]] = None,
    provenance: Optional[dict] = None,
) -> RoiMasks:
    """Build circular ROI masks at the calibrated fiber positions.

    ``roi_fraction`` is the fraction of the detected fiber-face radius used,
    0.75 as in the acquisition and MATLAB analysis code — it keeps the ROI
    inside the fiber face so that cladding and inter-fiber epoxy do not enter
    the average.

    Overlapping ROIs are an error, not a warning: a pixel shared between two
    fibers makes both traces partly the same signal, which manufactures
    correlation between regions and would appear as a connectivity edge.
    """
    from scipy.sparse import csr_matrix

    tc = np.atleast_2d(np.asarray(translated_coords, dtype=float))
    mids = np.asarray(mids, dtype=float).ravel()
    if tc.shape[1] < 3:
        raise ValueError(f"translated_coords has shape {tc.shape}; "
                         "expected (n_fibers, >=3) of [x, y, radius]")
    h, w = int(image_shape[0]), int(image_shape[1])
    hw = int(half_width if half_width is not None else w // 2)

    cx = tc[:, 0] - mids[0]
    cy = tc[:, 1] - mids[1]
    if half == "right":
        cx = cx + hw
    elif half != "left":
        raise ValueError(f"half must be 'left' or 'right', got {half!r}")
    radii = tc[:, 2] * float(roi_fraction)

    n_roi = tc.shape[0]
    x0, x1 = (0, hw) if half == "left" else (hw, w)
    yy, xx = np.mgrid[0:h, x0:x1]

    masks = []
    for i in range(n_roi):
        d2 = (xx - cx[i]) ** 2 + (yy - cy[i]) ** 2
        masks.append(d2 <= radii[i] ** 2)
    stack = np.stack(masks, axis=0)

    counts = stack.sum(axis=0)
    if np.any(counts > 1):
        overlapping = sorted({
            (i + 1, j + 1)
            for i in range(n_roi) for j in range(i + 1, n_roi)
            if np.any(stack[i] & stack[j])
        })
        raise ValueError(
            f"{int(np.sum(counts > 1))} pixels are shared between ROIs "
            f"{overlapping}. Shared pixels put the same photons in two "
            "regions, manufacturing between-region correlation; reduce "
            "roi_fraction or recheck the fiber detection."
        )
    empty = [i + 1 for i in range(n_roi) if not stack[i].any()]
    if empty:
        raise ValueError(
            f"ROIs {empty} contain no pixels — centres fall outside the "
            f"{half} half ({h}x{hw}). Check 'mids' and the half selection."
        )

    # Flat indices into the *half* image, ROI-major so that roi_sizes repeats
    # cleanly in label_image().
    pix_idx_list, roi_sizes = [], np.zeros(n_roi, dtype=np.int64)
    for i in range(n_roi):
        idx = np.flatnonzero(stack[i].ravel())
        pix_idx_list.append(idx)
        roi_sizes[i] = idx.size
    pix_idx = np.concatenate(pix_idx_list)

    rows = np.repeat(np.arange(n_roi), roi_sizes)
    cols = np.arange(pix_idx.size)
    vals = 1.0 / roi_sizes[rows]
    W = csr_matrix((vals, (rows, cols)), shape=(n_roi, pix_idx.size))

    return RoiMasks(
        W=W, pix_idx=pix_idx, roi_sizes=roi_sizes,
        shape=(h, hw), centres=np.column_stack([cx, cy]), radii=radii,
        roi_fraction=float(roi_fraction), half=half,
        labels=list(labels) if labels is not None else None,
        provenance=dict(provenance or {}),
    )


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

@dataclass
class ExtractionResult:
    traces: np.ndarray            # (n_frames, n_roi) mean pixel value per ROI
    frame_intensity: np.ndarray   # (n_frames,) mean over all ROI pixels
    n_frames: int
    labels: Optional[list[str]] = None
    provenance: dict = field(default_factory=dict)

    def to_frame(self):
        import pandas as pd

        cols = self.labels or [f"roi{i + 1}" for i in range(self.traces.shape[1])]
        df = pd.DataFrame(self.traces, columns=cols)
        df.insert(0, "frame", np.arange(self.n_frames))
        df["roi_mean"] = self.frame_intensity
        return df

    def describe(self) -> str:
        return (f"{self.n_frames} frames x {self.traces.shape[1]} ROIs; "
                f"intensity {np.nanmin(self.traces):.1f}-"
                f"{np.nanmax(self.traces):.1f}")


def extract_traces(
    frames: Iterable[np.ndarray],
    masks: RoiMasks,
    n_frames: Optional[int] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> ExtractionResult:
    """Extract per-ROI mean intensity from an iterable of frames.

    ``frames`` is any iterable of 2-D arrays — :func:`iter_video_frames`, or a
    list, which is what makes this testable without a video file.

    Each frame is cropped to the half the masks were built for and reduced by
    one sparse matvec over ROI pixels only.  Frames whose shape does not match
    the mask geometry raise rather than being silently reshaped.
    """
    h, hw = masks.shape
    out = []
    inten = []
    for k, frame in enumerate(frames):
        arr = np.asarray(frame)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.shape[0] != h or arr.shape[1] < hw:
            raise ValueError(
                f"frame {k} has shape {arr.shape}, incompatible with masks "
                f"built for a {h}x{hw} half"
            )
        sub = arr[:, :hw] if masks.half == "left" else arr[:, -hw:]
        pix = sub.ravel()[masks.pix_idx].astype(np.float64)
        out.append(masks.W @ pix)
        inten.append(pix.mean())
        if progress is not None and (k % 50 == 0):
            progress(k)

    if not out:
        raise ValueError("no frames were read")
    traces = np.vstack(out)
    n_read = traces.shape[0]
    if n_frames is not None and n_read != n_frames:
        raise ValueError(
            f"read {n_read} frames but {n_frames} were expected; the frame "
            "count used for trigger alignment must match the frames actually "
            "decoded"
        )
    return ExtractionResult(
        traces=traces,
        frame_intensity=np.asarray(inten, dtype=float),
        n_frames=n_read,
        labels=masks.labels,
        provenance={
            "roi_fraction": masks.roi_fraction,
            "half": masks.half,
            "roi_sizes": masks.roi_sizes.tolist(),
            "mask_shape": list(masks.shape),
            **masks.provenance,
        },
    )


def extract_session(
    video_path: str | Path,
    translated_coords_path: str | Path,
    mids_path: str | Path,
    roi_fraction: float = 0.75,
    half: str = "left",
    labels: Optional[Sequence[str]] = None,
    video_info: Optional[VideoInfo] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> tuple[ExtractionResult, RoiMasks, VideoInfo]:
    """Probe, build masks and extract, in one call.

    Returns the traces, the masks (keep these to re-apply identical geometry or
    to draw the QC overlay) and the measured video info.
    """
    info = video_info or probe_video(video_path, count_frames=True)
    cal = load_calibration(translated_coords_path, mids_path)
    masks = build_roi_masks(
        cal["coords"], cal["mids"],
        image_shape=(info.height, info.width),
        roi_fraction=roi_fraction, half=half, labels=labels,
        provenance={"video": str(info.path), **cal["provenance"]},
    )
    res = extract_traces(iter_video_frames(video_path), masks,
                         n_frames=info.n_frames, progress=progress)
    return res, masks, info
