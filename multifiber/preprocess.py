"""
Photometry preprocessing.

Pipeline, in order:

    demux 415/470  ->  blank laser-contaminated frames  ->  F0 by airPLS
    ->  dF/F = (F - F0) / F0  ->  isosbestic regression  ->  (optional) smoothing

Five things differ from the legacy notebook, each because the original is
either inert or wrong rather than merely different:

1. **airPLS actually honours ``porder``.**  The notebook's ``WhittakerSmooth``
   built a first-difference penalty unconditionally and ignored the argument,
   so ``porder=9`` and ``porder=1`` produced identical output.

2. **dF/F is computed against F0, on the raw trace.**  The notebook subtracted
   the airPLS baseline and *then* divided by the mean of the result — a
   near-zero number of arbitrary sign, since the trace had just been
   detrended.  That is a division by noise; a subsequent z-score then erased
   whatever scale it produced.  Here the airPLS fit *is* F0, which also makes
   amplitudes comparable across fibers with different coupling efficiency.
   That matters more for weight mapping than it did for correlational work,
   because the matrix entries are amplitudes.

3. **Epoch boundaries come from the data**, not from ``seg_start=2200`` and
   ``decay_remove=600`` hardcoded for a particular session shape.

4. **Laser-contaminated frames are blanked before the baseline fit**, so the
   artifact cannot bias F0.

5. **The Lasso penalty is chosen by blocked cross-validation, or fixed.**  The
   notebook scored seven alphas by *training* MSE, which is monotonically
   non-increasing in the penalty, so it selected the smallest alpha every
   time regardless of the data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.linalg import solveh_banded
from scipy.sparse import csc_matrix, diags, eye

__all__ = [
    "remove_common_mode",
    "whittaker_smooth",
    "airpls",
    "lambda_from_timescale",
    "demux_channels",
    "derive_epochs_from_laser",
    "frames_contaminated_by_laser",
    "blank_frames",
    "estimate_f0",
    "compute_dff",
    "regress_isosbestic",
    "stim_regressors",
    "ProcessedTraces",
    "preprocess_traces",
]


# --------------------------------------------------------------------------- #
# Baseline estimation
# --------------------------------------------------------------------------- #

#: Smallest weight any sample is allowed to carry in the Whittaker solve.
#: See :func:`whittaker_smooth` for why this cannot be zero.
MIN_WEIGHT: float = 1e-9


def _banded_from_sparse(A, d: int) -> np.ndarray:
    """Upper-banded storage of a symmetric matrix with bandwidth ``d``.

    Layout is what ``solveh_banded`` expects for ``lower=False``:
    ``ab[d + i - j, j] == A[i, j]`` for the ``d`` superdiagonals and the
    main diagonal.
    """
    n = A.shape[0]
    ab = np.zeros((d + 1, n), dtype=float)
    A = A.tocsr()
    for k in range(d + 1):                      # k-th superdiagonal
        diag = A.diagonal(k)
        ab[d - k, k:] = diag
    return ab


def whittaker_smooth(
    x: np.ndarray,
    w: np.ndarray,
    lambda_: float,
    differences: int = 1,
    min_weight: float = MIN_WEIGHT,
    solver: str = "banded",
) -> np.ndarray:
    """Penalised least-squares smoother.

    ``differences`` is the order of the difference penalty and is honoured
    here.  The notebook's version accepted the argument and built a
    first-difference matrix regardless, so every call documented as higher
    order was in fact first order.

    Two numerical guards, both load-bearing:

    * **Weights are floored at ``min_weight``.**  ``airpls`` drives the weight
      of every sample above the current baseline to zero, and on a smooth
      monotone trace that can be *all* of them.  ``A = W + lambda * D'D`` is
      then exactly singular — ``D'D`` has a null space of dimension
      ``differences`` (constants, then linear trends) — and ``spsolve`` on a
      singular matrix is undefined behaviour in SuperLU, which on Windows
      surfaces as a heap-corruption crash of the whole process rather than an
      exception.  A tiny floor keeps ``A`` positive definite without
      measurably changing the fit.
    * **The system is solved as a banded Cholesky, not a sparse LU.**
      ``W + lambda * D'D`` is symmetric positive definite with bandwidth
      ``differences``, so ``scipy.linalg.solveh_banded`` is the right solver:
      it is O(n * d^2), it exploits the symmetry, and it runs in LAPACK
      (``pbsv``) rather than SuperLU.  ``spsolve`` is available via
      ``solver='sparse'`` but is not the default — SuperLU on Windows has been
      observed to take the whole process down with a heap-corruption exit code
      instead of raising.
    """
    x = np.asarray(x, dtype=float).ravel()
    m = x.size
    w = np.asarray(w, dtype=float).ravel()
    if w.size != m:
        raise ValueError(f"weights have {w.size} entries for {m} samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("input contains non-finite values; interpolate or "
                         "blank them before fitting")
    d = int(differences)
    if m <= d + 1:
        return np.full(m, float(np.mean(x)))

    w = np.maximum(w, float(min_weight))

    E = eye(m, format="csc")
    D = E[1:] - E[:-1]
    for _ in range(d - 1):
        D = D[1:] - D[:-1]
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (float(lambda_) * (D.T @ D)))
    rhs = w * x

    if solver == "banded":
        ab = _banded_from_sparse(A, d)
        try:
            z = solveh_banded(ab, rhs, lower=False, check_finite=True)
        except np.linalg.LinAlgError as exc:
            raise FloatingPointError(
                f"baseline solve failed (lambda={lambda_:.3e}, n={m}, "
                f"differences={d}): {exc}. The system is too ill-conditioned; "
                "reduce baseline_tau_s or the difference order."
            ) from exc
    elif solver == "sparse":
        from scipy.sparse.linalg import spsolve

        z = spsolve(A, rhs)
    else:
        raise ValueError(f"unknown solver {solver!r}")

    z = np.asarray(z, dtype=float).ravel()
    if not np.all(np.isfinite(z)):
        raise FloatingPointError(
            f"baseline solve returned non-finite values (lambda={lambda_:.3e}, "
            f"n={m}). The system is too ill-conditioned; reduce the baseline "
            "timescale or the difference order."
        )
    return z


def lambda_from_timescale(
    tau_s: float,
    fs: float,
    differences: int = 2,
) -> float:
    """Convert a baseline timescale in seconds to a Whittaker penalty.

    The penalty ``lambda_`` is not scale-free: its meaning depends on the
    sampling rate and on the difference order, so a literal like 25000 means
    different things at 40 Hz and 80 Hz, and in a 2-minute versus a 30-minute
    recording.  This expresses it as the slowest fluctuation the baseline is
    allowed to follow.

    From the smoother's frequency response, half-power occurs where
    ``lambda_ * (2 sin(pi f / fs))**(2d) == 1``; for ``f << fs`` that gives
    ``lambda_ = (fs / (2 pi f_c))**(2d)`` with ``f_c = 1 / tau_s``.

    A baseline timescale should be slower than the transients of interest and
    faster than the photobleaching decay.
    """
    if tau_s <= 0 or fs <= 0:
        raise ValueError("tau_s and fs must be positive")
    return float((fs * tau_s / (2.0 * np.pi)) ** (2 * int(differences)))


def airpls(
    x: np.ndarray,
    lambda_: float = 1e7,
    differences: int = 2,
    itermax: int = 15,
    solver: str = "banded",
) -> np.ndarray:
    """Adaptive iteratively reweighted penalised least squares baseline.

    Returns the fitted baseline, which is used directly as F0.  Samples lying
    above the current baseline are down-weighted each iteration, so transients
    do not drag the baseline upward.

    ``differences=2`` is the default, not the notebook's 1.  A first-difference
    penalty drives the baseline toward a *constant* as ``lambda_`` grows, so it
    cannot follow photobleaching: on a synthetic trace with an exponential
    decay it under-estimates F0 by ~25% at lambda >= 1e5, which inflates every
    dF/F by the same factor.  A second-difference penalty is free to follow a
    linear trend, and an exponential decay is locally linear; the same test
    gives a median error of -0.3% and a worst case of 1-4%.

    Set ``lambda_`` with :func:`lambda_from_timescale` rather than by literal.
    """
    x = np.asarray(x, dtype=float).ravel()
    m = x.size
    w = np.ones(m)
    z = np.zeros(m)
    total = float(np.abs(x).sum())
    if total == 0.0 or m < 4:
        return np.full(m, float(np.mean(x)) if m else 0.0)
    for i in range(1, itermax + 1):
        z = whittaker_smooth(x, w, lambda_, differences, solver=solver)
        d = x - z
        neg = d[d < 0]
        dssn = float(np.abs(neg.sum()))
        if dssn < 0.001 * total or i == itermax:
            break
        # Down-weight, but never to exactly zero: an all-zero weight vector
        # makes the Whittaker system singular (see whittaker_smooth).
        w[d >= 0] = MIN_WEIGHT
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        if neg.size:
            w[0] = float(np.exp(i * np.abs(neg).max() / dssn))
            w[-1] = w[0]
    return z


# --------------------------------------------------------------------------- #
# Demultiplexing and epochs
# --------------------------------------------------------------------------- #

@dataclass
class Demuxed:
    """One interleaved recording split into its excitation channels."""

    signal: np.ndarray            # (n_sig, n_roi)
    isosbestic: np.ndarray        # (n_iso, n_roi)
    signal_frames: np.ndarray     # frame indices of each signal sample
    isos_frames: np.ndarray
    signal_t: np.ndarray          # DAQ time of each signal sample
    isos_t: np.ndarray
    labels: Optional[list[str]] = None

    @property
    def n_roi(self) -> int:
        return int(self.signal.shape[1])


def demux_channels(
    traces: np.ndarray,
    led_state: np.ndarray,
    frame_t: np.ndarray,
    signal_state: Optional[int] = None,
    labels: Optional[Sequence[str]] = None,
) -> Demuxed:
    """Split interleaved frames into signal (470) and isosbestic (415).

    ``signal_state`` defaults to the brighter state, since GCaMP emission
    under 470 nm excitation exceeds that under 415 nm.  Pass it explicitly if
    that assumption does not hold for a given fiber set.

    The two channels are *not* resampled onto a common grid and are *not*
    forced to equal length — each keeps its own timestamps.  The notebook
    reconciled lengths by dropping a sample or zero-padding, which hides a
    demultiplexing error rather than surfacing it.
    """
    traces = np.atleast_2d(np.asarray(traces, dtype=float))
    led_state = np.asarray(led_state, dtype=int).ravel()
    frame_t = np.asarray(frame_t, dtype=float).ravel()
    n = traces.shape[0]
    if led_state.size != n or frame_t.size != n:
        raise ValueError(
            f"traces has {n} frames but led_state has {led_state.size} and "
            f"frame_t has {frame_t.size}; all three must be per-frame"
        )
    states = np.unique(led_state)
    if states.size == 1:
        # Single-wavelength recording: every frame is signal and there is no
        # isosbestic channel.  Returned explicitly rather than faked, so the
        # absence propagates to the motion correction instead of being filled
        # with a copy of the signal.
        return Demuxed(
            signal=traces, isosbestic=np.empty((0, traces.shape[1])),
            signal_frames=np.arange(n), isos_frames=np.empty(0, dtype=int),
            signal_t=frame_t, isos_t=np.empty(0),
            labels=list(labels) if labels is not None else None,
        )
    if states.size != 2:
        raise ValueError(f"expected 1 or 2 LED states, found {states.tolist()}")

    if signal_state is None:
        means = {int(s): float(traces[led_state == s].mean()) for s in states}
        signal_state = max(means, key=means.get)

    sig = led_state == signal_state
    iso = ~sig
    return Demuxed(
        signal=traces[sig], isosbestic=traces[iso],
        signal_frames=np.flatnonzero(sig), isos_frames=np.flatnonzero(iso),
        signal_t=frame_t[sig], isos_t=frame_t[iso],
        labels=list(labels) if labels is not None else None,
    )


def derive_epochs_from_laser(
    onset_t: np.ndarray,
    offset_t: np.ndarray,
    t_start: float,
    t_end: float,
    guard_s: float = 5.0,
) -> dict[str, tuple[float, float]]:
    """Baseline / stim / post epochs, measured from the laser events.

    ``baseline`` runs from the start of the recording to ``guard_s`` before the
    first command, ``stim`` spans first onset to last offset, and ``post``
    starts ``guard_s`` after the last offset.  The guard keeps any
    pre-stimulus anticipation or post-stimulus decay out of the windows used
    to fit the baseline.

    Returns only the epochs that are non-empty, so a session with no
    pre-stimulus period simply has no ``baseline`` key rather than an
    zero-length one.
    """
    onset_t = np.asarray(onset_t, dtype=float).ravel()
    offset_t = np.asarray(offset_t, dtype=float).ravel()
    if onset_t.size == 0:
        return {"baseline": (t_start, t_end)}

    first, last = float(onset_t.min()), float(offset_t.max())
    out: dict[str, tuple[float, float]] = {}
    if first - guard_s > t_start:
        out["baseline"] = (t_start, first - guard_s)
    out["stim"] = (first, last)
    if last + guard_s < t_end:
        out["post"] = (last + guard_s, t_end)
    return out


# --------------------------------------------------------------------------- #
# Artifact blanking
# --------------------------------------------------------------------------- #

def frames_contaminated_by_laser(
    frame_t: np.ndarray,
    gate_on_t: np.ndarray,
    exposure_s: Optional[float] = None,
    tol_s: float = 0.0005,
) -> np.ndarray:
    """Boolean mask of frames whose exposure overlaps any laser-gate pulse.

    With "LASER BETWEEN CAMERA FRAMES ONLY" enabled the gate fires in the
    inter-exposure interval and this should return all-False; a non-empty
    result means the interlock was off or mistimed, which is worth knowing
    before the traces are interpreted.

    Note what ``frame_t`` is assumed to mark: the START of integration. It is
    taken from the camera channel's rising edge, which is the trigger, not
    necessarily the exposure.  If the DAQ records the camera's *output*
    trigger -- which this rig configures as a programmable pulse with its own
    delay and width -- that edge is offset from the start of integration by
    that delay, and every phase measured against it is offset with it.  A gate
    that appears to sit at 8.5 ms after the trigger may be elsewhere in the
    exposure entirely.  :func:`multifiber.events.implied_exposure` checks this
    against the data.

    ``exposure_s`` is the time after each trigger during which the sensor is
    actually integrating.  It defaults to the full inter-frame interval, which
    treats the whole period as exposed — the conservative choice, but note
    that if the gate fires once per frame period (as the camera-synchronous
    chop does), the conservative default flags *every* stimulus frame by
    construction.  Supply the real exposure time to make this test
    informative; the rig's camera configuration sets ``ExposureTime``.
    """
    frame_t = np.asarray(frame_t, dtype=float).ravel()
    gate_on_t = np.asarray(gate_on_t, dtype=float).ravel()
    if frame_t.size < 2 or gate_on_t.size == 0:
        return np.zeros(frame_t.size, dtype=bool)
    period = float(np.median(np.diff(frame_t)))
    exp = period if exposure_s is None else float(exposure_s)

    idx = np.searchsorted(frame_t, gate_on_t, side="right") - 1
    ok = (idx >= 0) & (idx < frame_t.size)
    idx = idx[ok]
    # Strict, with a tolerance. A pulse beginning exactly as integration ends
    # does not overlap it, and ``exposure_s`` is itself a measured figure: an
    # inclusive test flags a correctly interlocked gate whose pulses sit on
    # the boundary, which is where an interlock puts them by design.
    within = gate_on_t[ok] - frame_t[idx] < (exp - tol_s)
    mask = np.zeros(frame_t.size, dtype=bool)
    mask[idx[within]] = True
    return mask


def blank_frames(
    traces: np.ndarray,
    mask: np.ndarray,
    method: str = "nan",
    pad: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove contaminated samples.

    ``method='nan'`` marks them missing, which is what the baseline fit and
    every summary statistic should see.  ``method='interp'`` additionally
    fills them by linear interpolation, for steps that cannot take NaN; the
    returned mask still says which samples were invented, and they must not be
    counted as data.
    """
    traces = np.array(traces, dtype=float, copy=True)
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size != traces.shape[0]:
        raise ValueError(f"mask has {mask.size} entries for "
                         f"{traces.shape[0]} samples")
    if pad > 0 and mask.any():
        k = np.ones(2 * pad + 1, dtype=int)
        mask = np.convolve(mask.astype(int), k, mode="same") > 0

    traces[mask] = np.nan
    if method == "nan":
        return traces, mask
    if method != "interp":
        raise ValueError(f"unknown method {method!r}")

    idx = np.arange(traces.shape[0])
    for j in range(traces.shape[1]):
        col = traces[:, j]
        good = ~np.isnan(col)
        if good.sum() < 2:
            continue
        traces[:, j] = np.interp(idx, idx[good], col[good])
    return traces, mask


# --------------------------------------------------------------------------- #
# dF/F
# --------------------------------------------------------------------------- #

def estimate_f0(
    x: np.ndarray,
    epochs: Optional[Sequence[tuple[int, int]]] = None,
    lambda_: float = 1e7,
    differences: int = 2,
    min_segment: int = 50,
    method: str = "airpls",
) -> np.ndarray:
    """F0 for one trace, fitted piecewise over ``epochs``.

    ``epochs`` are (start, stop) sample indices; samples outside every epoch,
    and epochs too short or too thoroughly blanked to fit, are filled by
    interpolation from the fitted segments.  Fitting separately per epoch
    lets the baseline follow a fast initial bleaching decay without the
    smoothing that suits a long steady period, which is what the notebook's
    two-lambda split was approximating with hardcoded indices.

    NaNs (blanked samples) are interpolated for the fit and restored to the
    fitted curve afterwards, so a blanked artifact cannot pull F0.

    ``method='airpls'`` fits the LOWER ENVELOPE: its reweighting discards
    every sample above the current baseline, which is right for spectroscopy,
    where peaks point up and belong to the signal.  Applied to photometry it
    puts F0 below the bulk of the data, so dF/F is positively skewed and most
    samples sit above zero -- measured at skew +3.2 with 71% of samples above
    F0 on a real session.

    ``method='symmetric'`` is a plain Whittaker smoother with uniform weights,
    so F0 follows the CENTRE of the data.  dF/F is then centred on zero and
    its sign is interpretable, which matters when the expected response is a
    decrease.  Both give the same peri-event contrast, because that
    re-baselines every trial locally and F0's absolute level cancels; the
    choice affects how the trace reads, not what is measured.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    f0 = np.full(n, np.nan)
    segments = list(epochs) if epochs else [(0, n)]

    for a, b in segments:
        a, b = max(0, int(a)), min(n, int(b))
        if b - a < min_segment:
            continue
        seg = x[a:b].copy()
        bad = np.isnan(seg)
        if bad.all():
            continue
        if bad.any():
            idx = np.arange(seg.size)
            seg[bad] = np.interp(idx[bad], idx[~bad], seg[~bad])
        if method == "airpls":
            f0[a:b] = airpls(seg, lambda_=lambda_, differences=differences)
        elif method == "symmetric":
            f0[a:b] = whittaker_smooth(seg, np.ones_like(seg), lambda_,
                                       differences=differences)
        else:
            raise ValueError(f"unknown baseline method {method!r}")

    if np.isnan(f0).any():
        good = ~np.isnan(f0)
        if not good.any():
            raise ValueError(
                "no epoch could be fitted: every segment is shorter than "
                f"min_segment={min_segment} or entirely non-finite. If the "
                "stimulus epoch was fully blanked, supply exposure_s or set "
                "blank_laser=False."
            )
        idx = np.arange(n)
        f0 = np.interp(idx, idx[good], f0[good])
    return f0


def compute_dff(
    x: np.ndarray,
    f0: np.ndarray,
    min_f0: float = 1e-6,
) -> np.ndarray:
    """(F - F0) / F0, with a guard against a near-zero denominator.

    F0 here is the fitted baseline of the *raw* trace, so the denominator is a
    real fluorescence level and dF/F is comparable across fibers.  Dividing a
    detrended residual by its own mean, as the notebook did, is a division by
    noise.
    """
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    denom = np.where(np.abs(f0) < min_f0, np.nan, f0)
    if np.isnan(denom).any():
        n_bad = int(np.isnan(denom).sum()) - int(np.isnan(f0).sum())
        if n_bad > 0:
            import warnings
            warnings.warn(
                f"{n_bad} samples have |F0| < {min_f0}; dF/F is undefined "
                "there and is returned as NaN",
                RuntimeWarning, stacklevel=2,
            )
    return (x - f0) / denom


# --------------------------------------------------------------------------- #
# Motion correction
# --------------------------------------------------------------------------- #

def _alpha_grid_kwarg(cls, n_alphas: int) -> dict:
    """Name of the alpha-grid argument, which differs across scikit-learn.

    ``n_alphas`` was deprecated in 1.7 in favour of ``alphas`` accepting an
    integer, and removed later.  Before 1.7, ``alphas`` existed but expected
    an array, so passing an integer there would be wrong.  The version is
    therefore checked rather than only the signature: on >= 1.7 use
    ``alphas`` (which also silences the deprecation warning), below it use
    ``n_alphas``.
    """
    import inspect

    params = inspect.signature(cls.__init__).parameters
    try:
        import sklearn

        major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
        modern = (major, minor) >= (1, 7)
    except Exception:
        modern = False

    if modern and "alphas" in params:
        return {"alphas": n_alphas}
    if "n_alphas" in params:
        return {"n_alphas": n_alphas}
    if "alphas" in params:
        return {"alphas": n_alphas}
    return {}


def remove_common_mode(
    dff: np.ndarray,
    method: str = "median",
    exclude: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, dict]:
    """Remove the component shared across all fibers.

    Motion of the head, flexion of the fiber bundle and changes in patchcord
    coupling move every fiber together, because they act on the bundle rather
    than on one brain region.  Photon noise does not.  So the component common
    to all fibers is mostly artifact, and subtracting it can cut per-fiber
    noise substantially.

    ``method='median'`` subtracts the across-fiber median at each timepoint,
    which is robust to one or two fibers carrying a large real signal.
    ``method='pc1'`` regresses out the first principal component, which is
    optimal if the artifact is a single scaled waveform but is dragged by any
    strong shared biology.

    **This removes real shared neural activity too.** For stimulation at one
    site the evoked response is confined to a few fibers and is not
    common-mode, so the bias is small; but any analysis of network-wide
    co-fluctuation, global state, or arousal is invalidated by it.  Use
    ``exclude`` to keep the stimulated fiber (and any site expected to respond)
    out of the estimate of the common component, so the thing being measured
    does not contribute to what is subtracted from it.
    """
    X = np.asarray(dff, dtype=float)
    n_t, n_roi = X.shape
    keep = np.ones(n_roi, dtype=bool)
    if exclude is not None:
        keep[np.asarray(list(exclude), dtype=int)] = False
    if keep.sum() < 3:
        raise ValueError(
            f"only {int(keep.sum())} fibers left to estimate the common mode; "
            "at least 3 are needed for it to mean anything"
        )

    src = X[:, keep]
    if method == "median":
        common = np.nanmedian(src, axis=1)
    elif method == "pc1":
        M = np.where(np.isfinite(src), src, 0.0)
        M = M - M.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(M, full_matrices=False)
        common = M @ vt[0]
    else:
        raise ValueError(f"unknown method {method!r}")

    c = common - np.nanmean(common)
    denom = float(np.nansum(c * c))
    out = np.empty_like(X)
    gains = np.zeros(n_roi)
    for j in range(n_roi):
        y = X[:, j]
        good = np.isfinite(y) & np.isfinite(c)
        if denom <= 0 or good.sum() < 10:
            out[:, j] = y
            continue
        beta = float(np.nansum(y[good] * c[good]) / np.nansum(c[good] ** 2))
        out[:, j] = y - beta * c
        gains[j] = beta

    sd_before = np.nanstd(X, axis=0)
    sd_after = np.nanstd(out, axis=0)
    return out, {
        "method": method,
        "excluded": [] if exclude is None else list(map(int, exclude)),
        "coefficients": gains.tolist(),
        "sd_before": sd_before.tolist(),
        "sd_after": sd_after.tolist(),
        "noise_reduction": (sd_before / np.where(sd_after > 0, sd_after, np.nan)
                            ).tolist(),
    }


def build_isosbestic_predictors(
    iso_dff: np.ndarray,
    j: int,
    mode: str = "common",
    common_method: str = "median",
    standardize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Predictor matrix for one fiber's motion regression.

    ``'same'``    that fiber's own isosbestic only.
    ``'all'``     every fiber's isosbestic as a separate predictor.
    ``'others'``  every fiber's EXCEPT this one, separately.
    ``'common'``  the fiber's own isosbestic PLUS a common component built
                  from the OTHER fibers.

    ``'common'`` is the one to use when the isosbestic is dim.  Each 415
    channel carries the same bundle motion plus its own photon noise; a
    median (or first principal component) across fibers averages the noise
    down while keeping the shared artifact, so a fiber whose own isosbestic is
    too dark to be a useful regressor can still be corrected using the others.
    That is the point of having twelve of them.

    Unlike common-mode removal on the signal channel, this cannot subtract
    neural activity: 415 nm is the calcium-insensitive wavelength, so its
    shared component is motion and coupling, not physiology.

    The own-fiber predictor is kept alongside the common one because motion
    does not couple identically into every fiber -- bending affects each
    differently -- and the regression can weight the two.  The common
    component excludes fiber ``j`` so the two predictors are not the same
    quantity twice.

    ``common_method`` is ``'median'`` or ``'pcN'``.  ``'pc1'`` assumes one
    artifact waveform scaled differently in each fiber; ``'pc2'`` and
    ``'pc3'`` add further components as separate regressors, which is the
    right model if several independent things move -- bundle bending,
    connector play, an LED drifting -- since those project onto different
    directions.  Each extra component is another coefficient fitted from the
    same data, so the gain has to be checked rather than assumed: compare the
    detection floor, not the regression R^2, which rises with any added
    predictor.

    A caution on the penalty.  ``alpha='cv'`` selects the L1 strength by
    cross-validated prediction error, which is dominated by slow drift.  A
    brief artifact -- stray light during a one-second stimulation, a few
    percent of the samples -- contributes almost nothing to that error, so
    the cross-validated penalty can shrink the predictor carrying it, in
    testing to exactly zero.  Where the artifact of interest is brief, set a
    small fixed ``alpha`` (1e-6) rather than ``'cv'``, and check the result
    on the artifact itself rather than on R^2.

    ``standardize`` scales each isosbestic to unit variance before combining.
    Without it a dim fiber, whose dF/F is large because its denominator is
    small, would dominate the common component through noise alone.
    """
    X = np.asarray(iso_dff, dtype=float)
    n_t, n_roi = X.shape
    if mode == "same":
        return X[:, [j]], ["own"]
    if mode == "all":
        return X, [f"iso{k}" for k in range(n_roi)]
    if mode == "others":
        # Every other fiber's isosbestic as its own predictor, with this
        # fiber's excluded. Where an artifact reaches several fibers but not
        # uniformly, the others carry it in a pattern that a single summary
        # cannot represent -- and leaving the own fiber out keeps the fit
        # from simply reproducing it.
        other = [k for k in range(n_roi) if k != j]
        return X[:, other], [f"iso{k}" for k in other]
    if mode != "common":
        raise ValueError(
            f"unknown predictor mode {mode!r}; expected 'same', 'all', "
            "'others' or 'common'")

    others = [k for k in range(n_roi) if k != j]
    if not others:
        return X[:, [j]], ["own"]
    O = X[:, others]
    if standardize:
        sd = np.nanstd(O, axis=0, keepdims=True)
        sd = np.where(sd > 0, sd, 1.0)
        O = (O - np.nanmean(O, axis=0, keepdims=True)) / sd

    if common_method == "median":
        common = np.nanmedian(O, axis=1)[:, None]
        names = ["common_median"]
    elif re.fullmatch(r"pc\d+", str(common_method)):
        # pcN keeps the first N components as SEPARATE regressors. One
        # component assumes the artifact is a single waveform scaled per
        # fiber; several allow for independent sources -- bundle bending,
        # connector movement, a drifting LED -- that project differently.
        n_pc = int(str(common_method)[2:])
        M = np.where(np.isfinite(O), O, 0.0)
        M = M - M.mean(axis=0, keepdims=True)
        _, sv, vt = np.linalg.svd(M, full_matrices=False)
        n_pc = max(1, min(n_pc, vt.shape[0]))
        comp = M @ vt[:n_pc].T
        # The sign of a principal component is arbitrary, but the regression
        # constrains coefficients to be positive (an artifact adds to the
        # signal channel, it does not subtract from it). A component that
        # needs a negative weight would simply be zeroed, which is why extra
        # components appeared to do nothing. Each is therefore supplied in
        # both signs, and the fit keeps whichever it can use.
        common = np.empty((comp.shape[0], 2 * n_pc))
        names = []
        for k in range(n_pc):
            common[:, 2 * k] = comp[:, k]
            common[:, 2 * k + 1] = -comp[:, k]
            names += [f"common_pc{k + 1}+", f"common_pc{k + 1}-"]
    else:
        raise ValueError(
            f"unknown common_method {common_method!r}; expected 'median' or "
            "'pcN' (pc1, pc2, pc3, ...)")

    # Split the two predictors so they do not compete, and in the direction
    # that lets each do its job: the OWN term is residualised against the
    # common component, not the other way round.
    #
    # The common component is an average over fibers, so it has the better
    # signal-to-noise for anything shared -- bundle motion -- and should
    # carry it. What is left of the own isosbestic after removing that is
    # what this fiber alone sees: stray light from a nearby stimulation
    # fiber, its own coupling change. That remainder is the part no other
    # fiber can supply.
    #
    # Left collinear, the L1 penalty gives the weight to whichever predictor
    # fits the shared drift better and shrinks the other -- to exactly zero
    # in testing -- so a fiber-specific artifact went uncorrected.
    own = X[:, j].astype(float).copy()
    ok = np.isfinite(own) & np.all(np.isfinite(common), axis=1)
    if ok.sum() > 10 and common.shape[1]:
        C = common[ok] - common[ok].mean(axis=0, keepdims=True)
        o = own[ok] - own[ok].mean()
        try:
            beta, *_ = np.linalg.lstsq(C, o, rcond=None)
            own = own - common @ beta
        except np.linalg.LinAlgError:
            pass
    return np.column_stack([own, common]), ["own_specific"] + names


@dataclass
class ProcessedTraces:
    """Output of :func:`preprocess_traces`.

    ``dff`` is after motion regression, ``dff_raw`` before it, so the two can
    be compared and the regression judged rather than taken on trust.  ``f0``
    is the fitted baseline on the signal channel, ``isos_dff`` the isosbestic
    put through the identical treatment, and ``blanked`` marks samples
    removed as laser-contaminated (NaN in the traces).
    """

    dff: np.ndarray
    dff_raw: np.ndarray
    f0: np.ndarray
    isos_dff: np.ndarray
    t: np.ndarray
    frames: np.ndarray
    blanked: np.ndarray
    labels: list
    epochs: Optional[list] = None
    qc: dict = field(default_factory=dict)

    @property
    def n_roi(self) -> int:
        return int(self.dff.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.dff.shape[0])

    def to_frame(self):
        """Long-form table for ``<sid>_dff.parquet``.

        One row per signal sample. ``t`` is DAQ time and ``frame`` the video
        frame index, so this joins to the events table and to the raw
        traces without re-deriving anything. Each ROI appears as its own
        column, with the isosbestic alongside as ``<label>__iso``, and
        ``blanked`` marks samples dropped as laser-contaminated -- those are
        NaN in the ROI columns, and the flag records that they were removed
        rather than merely missing.
        """
        import pandas as pd

        cols = self.labels or [f"roi{i + 1}"
                               for i in range(self.dff.shape[1])]
        df = pd.DataFrame(self.dff, columns=list(cols))
        df.insert(0, "t", self.t)
        df.insert(1, "frame", self.frames)
        df["blanked"] = self.blanked
        if self.isos_dff is not None and self.isos_dff.size:
            for i, c in enumerate(cols):
                df[f"{c}__iso"] = self.isos_dff[:, i]
        return df


def stim_regressors(
    t: np.ndarray,
    onset_t: np.ndarray,
    offset_t: np.ndarray,
    decay_tau: float = 0.4,
    rise_tau: float = 0.07,
) -> tuple[np.ndarray, list[str]]:
    """Two stimulus-locked regressors that differ in kinetics.

    An optical artifact follows the command exactly: on within a sample, off
    within a sample, no tail.  A calcium response cannot -- GCaMP6f takes
    tens of milliseconds to rise and hundreds to decay, so it outlasts the
    command.  The two are therefore separable by SHAPE even though both are
    time-locked to the laser:

    * ``stim_square`` is the command envelope itself;
    * ``stim_kinetic`` is that envelope convolved with the indicator's
      impulse response.

    Fitting both and reading the kinetic coefficient is the measurement;
    the square coefficient absorbs the artifact.  This is the only way to
    separate them within a session, because the isosbestic cannot do it: the
    artifact is additive (a fixed number of stray photons, so a larger
    FRACTION of the dimmer 415 channel) while motion is multiplicative (the
    same fraction of both), and one coefficient on the isosbestic has to
    serve both ratios at once.

    A caveat that cannot be designed away: a response that genuinely has no
    kinetics -- an instantaneous, square physiological change -- would be
    absorbed by the square term. No such calcium response exists, which is
    what makes the separation safe here, but the assumption is doing work
    and should be stated.
    """
    t = np.asarray(t, dtype=float).ravel()
    on = np.atleast_1d(np.asarray(onset_t, dtype=float))
    off = np.atleast_1d(np.asarray(offset_t, dtype=float))
    square = np.zeros(t.size, dtype=float)
    for a, b in zip(on, off):
        square[(t >= a) & (t < b)] = 1.0

    period = float(np.median(np.diff(t))) if t.size > 1 else 0.025
    s_ = np.arange(0.0, max(8 * decay_tau, 4 * period), period)
    h = (1.0 - np.exp(-s_ / max(rise_tau, 1e-6))) * np.exp(-s_ / decay_tau)
    if h.max() > 0:
        h = h / h.max()
    kinetic = np.convolve(square, h)[:t.size]
    if kinetic.max() > 0:
        kinetic = kinetic / kinetic.max()
    return np.column_stack([square, kinetic]), ["stim_square", "stim_kinetic"]


def regress_isosbestic(
    y: np.ndarray,
    X: np.ndarray,
    alpha: "float | str" = "cv",
    method: str = "lasso",
    unpenalized: int = 0,
    head_method: str = "ols",
    head_alpha: "float | str" = "cv",
    tail_method: str = "lasso",
    non_negative: bool = False,
    keep: Optional[Sequence[int]] = None,
    score_mask: Optional[np.ndarray] = None,
    fit_mask: Optional[np.ndarray] = None,
    candidates: Sequence[str] = ("lasso", "ols", "hierarchical"),
    n_alphas: int = 30,
) -> tuple[np.ndarray, dict]:
    """Remove from ``y`` what the predictors ``X`` can explain.

    ``method`` selects the estimator, because no single one suits every
    case:

    ``'lasso'``
        Non-negative Lasso (the default).  The non-negativity encodes that
        an artifact ADDS to the signal channel; the L1 term drops predictors
        that earn nothing.  Its weakness is that a cross-validated penalty
        is tuned on total prediction error, which slow drift dominates, so
        it can discard a predictor carrying a brief artifact.
    ``'ols'``
        Least squares with no penalty.  Fits every predictor fully, which is
        what a brief artifact needs, at the cost of fitting noise when
        predictors are many or collinear.  ``non_negative`` keeps
        coefficients at or above zero: a negative weight would ADD the
        predictor to the residual, and where the isosbestic carries little
        shared variance an unconstrained fit can do exactly that, putting
        415 noise into the signal channel.  Set it False only to see what
        the unconstrained fit wants.
    ``'ridge'``
        L2, shrinking all coefficients smoothly rather than zeroing any.
        Sensible with several correlated predictors.
    ``'hierarchical'``
        Two stages. The first ``unpenalized`` columns are fitted with
        ``head_method`` at ``head_alpha``; the rest are fitted to the
        residual with ``tail_method`` at ``alpha``. The own-fiber
        isosbestic is therefore never shrunk away by a penalty chosen for
        the others, which is the failure the plain Lasso showed on a
        minority artifact.

        Both stages are configurable because neither choice is obviously
        right. An unpenalized OLS head fits the own fiber fully, including
        its noise; a penalized head fits it more conservatively and can
        work better where the isosbestic carries the artifact strongly. The
        tail is a different question again: a sparse Lasso keeps only the
        components that earn their place, while OLS or ridge use them all,
        which suits a tail of principal components that are orthogonal by
        construction and therefore not competing. Which wins is a property
        of the recording, not something to settle by argument.

    ``unpenalized`` is the number of leading columns in the first stage.

    ``'auto'``
        Fit every estimator in ``candidates`` and keep the one with the
        lowest out-of-sample error on held-out time blocks, scored only
        where ``score_mask`` is True.  Pass a mask that EXCLUDES the
        stimulation windows: the choice is then made on how well each
        estimator predicts motion, and cannot be influenced by the response
        being measured.  Which estimator wins differs by fiber -- a noisy
        isosbestic benefits from shrinkage, a clean one from fitting fully
        -- and this picks per fiber without selecting on the outcome.

    ``keep`` names columns to ESTIMATE but not subtract.  A stimulus-locked
    response regressor belongs here: it must be in the model, or the
    nuisance terms absorb the response and their coefficients are wrong,
    but subtracting it would remove the very thing being measured.  Its
    fitted coefficient is reported in ``info['coefficients']`` and is the
    amplitude estimate.

    ``r2_in_sample`` is reported for diagnosis only.  It rises with any
    added predictor, so it cannot be used to choose between these; judge on
    the artifact or on the detection floor.
    """
    import warnings as _w

    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    # Which samples the coefficients are FITTED on. This defaults to all of
    # them, and that default matters: a stimulus-locked artifact lives
    # INSIDE the stimulation windows, so a fit that excludes them never sees
    # it and cannot remove it. Excluding them protects the response from
    # being partly absorbed, but the two goals are in direct conflict and no
    # choice of mask serves both -- separating artifact from response needs
    # the shape difference (see stim_regressors), not a mask.
    #
    # ``score_mask`` therefore governs SCORING only, which is what the
    # automatic selection needs. ``fit_mask`` is a separate, explicit
    # opt-in for the rarer case where the response must be protected and
    # the artifact is not an issue.
    fit_ok = ok if fit_mask is None else (ok & np.asarray(fit_mask,
                                                          dtype=bool))
    ok_fit = fit_ok if fit_ok.sum() >= 50 else ok
    info: dict = {"method": method, "n_used": int(ok.sum()),
                  "alpha": float("nan"), "alpha_source": "none",
                  "coefficients": [], "r2_in_sample": float("nan"),
                  "noise_reduction": float("nan")}
    if ok.sum() < 20:
        info["warning"] = "too few finite samples; no regression applied"
        return y.copy(), info

    Xo, yo = X[ok_fit], y[ok_fit]
    Xmu = Xo.mean(axis=0, keepdims=True)
    ymu = yo.mean()
    Xc = Xo - Xmu
    yc = yo - ymu
    info["n_fit"] = int(ok_fit.sum())

    def _ols(A, b):
        """Least squares, optionally constrained to non-negative weights.

        The constraint is not cosmetic. ``resid = y - beta * x``, so a
        NEGATIVE beta adds the predictor to the residual rather than
        removing it: where the isosbestic carries little shared variance,
        an unconstrained fit can land on a small negative weight and inject
        415 noise into the signal channel. It shows up as a processed trace
        that resembles the isosbestic instead of being cleaned of it.

        An artifact adds light to both channels, so the physically
        meaningful weight is non-negative, and the Lasso branch has always
        enforced it.
        """
        if non_negative:
            try:
                from scipy.optimize import nnls

                coef, _ = nnls(A, b)
                return coef
            except Exception:
                pass
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        return coef

    if method == "auto":
        from sklearn.model_selection import TimeSeriesSplit

        sm = (np.ones(y.size, dtype=bool) if score_mask is None
              else np.asarray(score_mask, dtype=bool))
        sm_o = sm[ok_fit]
        best, best_err, scores = None, np.inf, {}
        for cand in candidates:
            errs = []
            for tr, te in TimeSeriesSplit(n_splits=4).split(Xc):
                te_s = te[sm_o[te]]
                if te_s.size < 20 or tr.size < 50:
                    continue
                try:
                    _, sub = regress_isosbestic(
                        yc[tr], Xc[tr], alpha=alpha, method=cand,
                        unpenalized=unpenalized, head_method=head_method,
                        head_alpha=head_alpha)
                except Exception:
                    continue
                c = np.asarray(sub["coefficients"], dtype=float).ravel()
                if c.size != Xc.shape[1]:
                    continue
                errs.append(float(np.mean((yc[te_s] - Xc[te_s] @ c) ** 2)))
            if errs:
                scores[cand] = float(np.mean(errs))
                if scores[cand] < best_err:
                    best, best_err = cand, scores[cand]
        method = best or "lasso"
        info["method"] = method
        info["auto_scores"] = scores
        info["auto_chosen"] = method

    if method == "ols":
        coef = _ols(Xc, yc)
        info["alpha"] = 0.0
        info["alpha_source"] = "none (OLS)"
    elif method == "ridge":
        from sklearn.linear_model import Ridge, RidgeCV


        if alpha == "cv":
            m = RidgeCV(alphas=np.logspace(-8, 2, n_alphas))
        else:
            m = Ridge(alpha=float(alpha))
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            m.fit(Xc, yc)
        coef = np.asarray(m.coef_, dtype=float).ravel()
        info["alpha"] = float(getattr(m, "alpha_", alpha if alpha != "cv"
                                      else np.nan))
        info["alpha_source"] = ("CV" if alpha == "cv" else "fixed")
    elif method == "hierarchical":
        k = max(1, min(int(unpenalized or 1), Xc.shape[1]))
        if head_method == "ols":
            head = _ols(Xc[:, :k], yc)
            info["alpha_source"] = "hierarchical: OLS head"
        else:
            _, hsub = regress_isosbestic(yc, Xc[:, :k], alpha=head_alpha,
                                         method=head_method)
            head = np.asarray(hsub["coefficients"], dtype=float).ravel()
            info["alpha_source"] = (f"hierarchical: {head_method} head "
                                    f"({hsub['alpha_source']})")
        info["head_method"] = head_method
        info["head_alpha"] = (0.0 if head_method == "ols"
                              else float(hsub["alpha"]))
        resid = yc - Xc[:, :k] @ head
        if Xc.shape[1] > k:
            tail, sub = regress_isosbestic(resid, Xc[:, k:], alpha=alpha,
                                           method=tail_method)
            coef = np.r_[head, np.asarray(sub["coefficients"],
                                          dtype=float).ravel()]
            info["alpha"] = sub["alpha"]
            info["alpha_source"] += (f" + {tail_method} tail "
                                     f"({sub['alpha_source']})")
            info["tail_method"] = tail_method
        else:
            coef = head
            info["alpha"] = 0.0
    else:
        from sklearn.linear_model import Lasso, LassoCV

        if alpha == "cv":
            from sklearn.model_selection import TimeSeriesSplit

            kw = _alpha_grid_kwarg(LassoCV, n_alphas)
            m = LassoCV(positive=True, fit_intercept=False,
                        cv=TimeSeriesSplit(n_splits=5), **kw)
        else:
            m = Lasso(alpha=float(alpha), positive=True, fit_intercept=False)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            m.fit(Xc, yc)
        coef = np.asarray(m.coef_, dtype=float).ravel()
        info["alpha"] = float(getattr(m, "alpha_", alpha if alpha != "cv"
                                      else np.nan))
        info["alpha_source"] = ("blocked CV" if alpha == "cv" else "fixed")

    # Everything is estimated jointly; only the nuisance columns are
    # subtracted. A response regressor left out of the fit would have its
    # variance absorbed by the nuisance terms; one subtracted along with
    # them would be removed from the data.
    sub = np.ones(X.shape[1], dtype=bool)
    if keep is not None:
        for c in keep:
            if 0 <= int(c) < sub.size:
                sub[int(c)] = False
    info["kept_columns"] = [int(c) for c in np.flatnonzero(~sub)]
    pred_all = (X - Xmu)[:, sub] @ coef[sub]
    resid = y - np.where(np.isfinite(pred_all), pred_all, 0.0)
    fitted = Xc @ coef
    denom = float(np.sum(yc ** 2))
    info["coefficients"] = [float(c) for c in coef]
    n_neg = int(np.sum(np.asarray(coef) < -1e-12))
    info["n_negative_coefficients"] = n_neg
    if n_neg:
        info["warning"] = (
            f"{n_neg} negative coefficient(s): those predictors are being "
            "ADDED to the residual, not removed. Where a predictor carries "
            "little shared variance this injects its noise into the signal.")
    info["r2_in_sample"] = (float(1.0 - np.sum((yc - fitted) ** 2) / denom)
                            if denom > 0 else float("nan"))
    sd_before = float(np.std(y[ok]))
    sd_after = float(np.std(resid[ok]))
    info["noise_reduction"] = (1.0 - sd_after / sd_before
                               if sd_before > 0 else float("nan"))
    return resid, info


def preprocess_traces(
    traces: np.ndarray,
    led_state: np.ndarray,
    frame_t: np.ndarray,
    laser_onset_t: Optional[np.ndarray] = None,
    laser_offset_t: Optional[np.ndarray] = None,
    gate_on_t: Optional[np.ndarray] = None,
    exposure_s: Optional[float] = None,
    labels: Optional[Sequence[str]] = None,
    baseline_tau_s: float = 60.0,
    differences: int = 2,
    guard_s: float = 5.0,
    alpha: "float | str" = "cv",
    predictors: str = "common",
    common_method: str = "median",
    regress: bool = True,
    regress_method: str = "lasso",
    head_method: str = "ols",
    head_alpha: "float | str" = "cv",
    tail_method: str = "lasso",
    unpenalized: int = 1,
    stim_artifact: bool = False,
    stim_decay_tau: float = 0.4,
    blank_pad: int = 1,
    smooth: Optional[int] = None,
    large_sd_dff: float = 0.05,
    baseline_mode: str = "continuous",
    baseline_method: str = "airpls",
    common_mode: Optional[str] = None,
    common_mode_exclude: Optional[Sequence[int]] = None,
    progress: Optional[object] = None,
) -> ProcessedTraces:
    """Run the full chain on one session's extracted ROI traces.

    ``predictors`` selects the motion regressors: ``'same'`` (own isosbestic
    only), ``'all'`` (every fiber's, as separate predictors) or ``'common'``
    (own plus a noise-averaged common component from the others).  See
    :func:`build_isosbestic_predictors`; ``'common'`` is the default and is
    what makes a dim isosbestic usable.

    ``baseline_mode`` controls whether F0 is fitted in one piece.

    ``'continuous'`` (default) fits the whole recording at once.  This is
    right for a stim-record session: the animal is doing the same thing
    throughout, so there is no regime change for a boundary to respect, and
    independent fits introduce a discontinuity at each epoch edge for no
    benefit.

    ``'epochs'`` fits baseline / stim / post separately.  That is appropriate
    when the signal regime genuinely changes partway through -- a
    resident-intruder assay, where the pre-intruder and social periods differ
    in mean, variance and bleaching rate -- and it is where this option came
    from.  Using it here would be inherited habit rather than a reason.

    Note that neither choice affects fast structure: with ``baseline_tau_s``
    at 60 s, F0 cannot follow a two-second excursion either way.  Trial-to-
    trial spread at the edges of a peri-event window is drift in the data, not
    an artifact of how F0 was segmented.

    ``baseline_tau_s`` is the slowest fluctuation F0 is allowed to follow,
    converted to a Whittaker penalty against the measured effective sampling
    rate by :func:`lambda_from_timescale`.  It should sit between the
    transients of interest (faster) and the photobleaching decay (slower).

    ``smooth`` is a Savitzky-Golay window in samples, **off by default**.  The
    notebook smoothed with a 17- or 29-sample window (425-725 ms at 40 Hz),
    which is harmless for bout-scale social signals but distorts onset latency
    and attenuates amplitude non-uniformly across regions within a 1.5 s
    peri-stimulus window.
    """
    warnings_out: list[str] = []
    dm = demux_channels(traces, led_state, frame_t, labels=labels)

    n_sig, n_roi = dm.signal.shape
    single_channel = dm.isosbestic.shape[0] == 0
    if single_channel:
        warnings_out.append(
            "single-wavelength recording: there is no isosbestic channel, so "
            "no motion correction was applied. Movement artefacts remain in "
            "dF/F and cannot be distinguished from signal."
        )
    elif dm.isosbestic.shape[0] < n_sig:
        pad = n_sig - dm.isosbestic.shape[0]
        warnings_out.append(
            f"isosbestic has {pad} fewer samples than signal; "
            "they are kept on separate timebases and the isosbestic is "
            "interpolated onto the signal times for regression"
        )

    # isosbestic onto the signal timebase (interpolation, not truncation)
    if single_channel:
        iso_on_sig = np.full((n_sig, n_roi), np.nan)
    else:
        iso_on_sig = np.column_stack([
            np.interp(dm.signal_t, dm.isos_t, dm.isosbestic[:, j])
            for j in range(n_roi)
        ])

    # ---- blanking ---------------------------------------------------- #
    if gate_on_t is not None and len(gate_on_t):
        contaminated = frames_contaminated_by_laser(
            dm.signal_t, gate_on_t, exposure_s=exposure_s)
        if contaminated.any():
            warnings_out.append(
                f"{int(contaminated.sum())} signal frames overlap a laser-gate "
                "pulse"
                + ("" if exposure_s is not None else
                   " (exposure_s not supplied, so the whole frame period was "
                   "treated as exposed; with a per-frame gate chop this flags "
                   "every stimulus frame by construction)")
            )
    else:
        contaminated = np.zeros(n_sig, dtype=bool)
    sig, blanked = blank_frames(dm.signal, contaminated, method="nan",
                                pad=blank_pad)

    # ---- epochs ------------------------------------------------------- #
    t0, t1 = float(dm.signal_t[0]), float(dm.signal_t[-1])
    if laser_onset_t is not None and len(laser_onset_t):
        ep_t = derive_epochs_from_laser(
            laser_onset_t,
            laser_offset_t if laser_offset_t is not None else laser_onset_t,
            t0, t1, guard_s=guard_s)
    else:
        ep_t = {"baseline": (t0, t1)}
    ep_idx = {
        name: (int(np.searchsorted(dm.signal_t, a, "left")),
               int(np.searchsorted(dm.signal_t, b, "right")))
        for name, (a, b) in ep_t.items()
    }
    period = float(np.median(np.diff(dm.signal_t)))
    fs_eff = 1.0 / period if period > 0 else float("nan")
    lam_value = lambda_from_timescale(baseline_tau_s, fs_eff, differences)
    lam = {name: lam_value for name in ep_idx}

    if "stim" in ep_idx:
        a, b = ep_idx["stim"]
        if b > a:
            frac = float(blanked[a:b].mean())
            if frac > 0.5:
                warnings_out.append(
                    f"blanking removed {frac:.0%} of the stimulus epoch, so "
                    "the evoked response cannot be measured there. Supply "
                    "exposure_s, or set blank_laser=False and inspect the "
                    "artifact directly before deciding what to exclude."
                )

    # ---- F0 and dF/F --------------------------------------------------- #
    f0 = np.empty_like(sig)
    dff_raw = np.empty_like(sig)
    iso_dff = np.empty_like(iso_on_sig)
    iso_f0_by_roi: list[np.ndarray] = []
    _tick = (progress.update if progress is not None else (lambda *_a: None))
    for j in range(n_roi):
        if baseline_mode == "continuous":
            merged = estimate_f0(sig[:, j], [(0, n_sig)], lambda_=lam_value,
                                 differences=differences,
                                 method=baseline_method)
        elif baseline_mode == "epochs":
            # Segments are fitted independently; one that is entirely blanked
            # is filled from its fitted neighbours rather than left without a
            # baseline.
            merged = estimate_f0(sig[:, j], list(ep_idx.values()),
                                 lambda_=lam_value, differences=differences,
                                 method=baseline_method)
        else:
            raise ValueError(f"unknown baseline_mode {baseline_mode!r}")
        f0[:, j] = merged
        dff_raw[:, j] = compute_dff(sig[:, j], merged)

        if single_channel:
            iso_dff[:, j] = np.nan
            iso_f0_by_roi.append(np.full(n_sig, np.nan))
        else:
            iso_f0 = estimate_f0(iso_on_sig[:, j], lambda_=lam_value,
                                 differences=differences,
                                 method=baseline_method)
            iso_f0_by_roi.append(iso_f0)
            iso_dff[:, j] = compute_dff(iso_on_sig[:, j], iso_f0)
        _tick(1)

    # ---- isosbestic regression ------------------------------------------ #
    dff = np.empty_like(dff_raw)
    r2, alphas = [], []
    stim_amp = []
    chosen = []
    if single_channel or not regress:
        dff = dff_raw.copy()
        r2 = [float("nan")] * n_roi
        alphas = [float("nan")] * n_roi
        if not regress and not single_channel:
            warnings_out.append("isosbestic regression disabled (regress=False)")
    else:
        pred_names = None
        coefs = []
        # A stimulus-locked artifact cannot be removed by the isosbestic
        # alone: stray light is additive (a larger FRACTION of the dimmer
        # 415) while motion is multiplicative (the same fraction of both),
        # and one coefficient on the isosbestic serves both ratios at once.
        # Two stimulus-locked regressors separate them by kinetics: the
        # square command envelope has none, a calcium response must have
        # some. The square term is subtracted; the kinetic term is estimated
        # and kept, because subtracting it would remove the measurement.
        stim_X = stim_names = None
        if stim_artifact and laser_onset_t is not None and len(laser_onset_t):
            stim_X, stim_names = stim_regressors(
                dm.signal_t, laser_onset_t,
                (laser_offset_t if laser_offset_t is not None
                 else np.asarray(laser_onset_t) + 1.0),
                decay_tau=stim_decay_tau)
        # For automatic per-fiber selection, SCORE only where no laser was
        # on, so the choice is about motion correction and cannot be
        # influenced by the response. Coefficients are still fitted on every
        # sample: a stimulus-locked artifact lives inside these windows.
        score_mask = np.ones(n_sig, dtype=bool)
        if laser_onset_t is not None and len(laser_onset_t):
            offs = (laser_offset_t if laser_offset_t is not None
                    else np.asarray(laser_onset_t) + 1.0)
            for a, b in zip(np.atleast_1d(laser_onset_t), np.atleast_1d(offs)):
                score_mask &= ~((dm.signal_t >= a - 1.0)
                                & (dm.signal_t <= b + 3.0))
        stim_amp = []
        chosen = []
        for j in range(n_roi):
            X, pred_names = build_isosbestic_predictors(
                iso_dff, j, mode=predictors, common_method=common_method)
            keep = None
            if stim_X is not None:
                n_iso = X.shape[1]
                X = np.column_stack([X, stim_X])
                keep = [n_iso + 1]            # the kinetic column
                pred_names = list(pred_names) + list(stim_names)
            resid, info = regress_isosbestic(
                dff_raw[:, j], X, alpha=alpha,
                method=("ols" if stim_X is not None and
                        regress_method == "lasso" else regress_method),
                head_method=head_method, head_alpha=head_alpha,
                tail_method=tail_method, unpenalized=unpenalized, keep=keep,
                score_mask=score_mask)
            if keep is not None:
                stim_amp.append(info["coefficients"][keep[0]])
            if "auto_chosen" in info:
                chosen.append(info["auto_chosen"])
            dff[:, j] = resid
            r2.append(info["r2_in_sample"])
            alphas.append(info["alpha"])
            coefs.append(info["coefficients"])

    # Report any discontinuity the segmentation introduced.
    f0_jumps = []
    if baseline_mode == "epochs":
        short = {name: (b - a) / fs_eff for name, (a, b) in ep_idx.items()
                 if (b - a) / fs_eff < baseline_tau_s}
        if short:
            warnings_out.append(
                "piecewise F0: "
                + ", ".join(f"the '{k}' epoch is {v:.0f} s" for k, v in short.items())
                + f", shorter than baseline_tau_s={baseline_tau_s:.0f} s. A "
                "segment shorter than the baseline timescale does not "
                "constrain the fit, so F0 there is poorly determined and the "
                "resulting dF/F differs from a continuous fit in its fast "
                "structure, not just at the seams."
            )
        edges = sorted({b for _, b in ep_idx.values()} | {a for a, _ in ep_idx.values()})
        for e in edges:
            if 1 <= e < n_sig - 1:
                jump = np.abs(f0[e, :] - f0[e - 1, :]) / np.abs(f0[e - 1, :])
                f0_jumps.append(float(np.nanmax(jump)))
        if f0_jumps and max(f0_jumps) > 0.01:
            warnings_out.append(
                f"piecewise F0 has a discontinuity of up to "
                f"{max(f0_jumps):.1%} at an epoch boundary. This session has "
                "no behavioural regime change, so baseline_mode='continuous' "
                "avoids it."
            )

    cm_info = None
    if common_mode:
        dff, cm_info = remove_common_mode(dff, method=common_mode,
                                          exclude=common_mode_exclude)
        warnings_out.append(
            f"common-mode removal applied ({common_mode}); shared neural "
            "activity is removed along with shared artifact, so network-wide "
            "co-fluctuation cannot be interpreted from these traces"
        )

    # --- did the correction actually help? ------------------------------- #
    sd_before = np.array([float(np.nanstd(dff_raw[:, j])) for j in range(n_roi)])
    sd_after = np.array([float(np.nanstd(dff[:, j])) for j in range(n_roi)])
    iso_level = np.array([
        float(np.nanmedian(iso_f0_by_roi[j]))
        if j < len(iso_f0_by_roi) and np.isfinite(iso_f0_by_roi[j]).any()
        else float("nan")
        for j in range(n_roi)])
    # A least-squares fit cannot increase in-sample variance, so the
    # regression is never the source of excess noise; what it can do is fail
    # to remove it.  The informative quantity is therefore how much residual
    # noise is left, in absolute terms.
    noisy = [j for j in range(n_roi) if sd_after[j] > large_sd_dff]
    if noisy:
        names = [(labels[j] if labels is not None else f"roi{j+1}")
                 for j in noisy]
        worst = float(np.nanmax(sd_after))
        warnings_out.append(
            f"{len(noisy)} ROIs have residual dF/F noise above "
            f"{large_sd_dff:.0%} (worst {worst:.0%}: "
            f"{', '.join(names[:6])}{'...' if len(names) > 6 else ''}). "
            "Photon noise on a bulk fiber signal is well under 1%, so this is "
            "motion or fiber-coupling artifact that the isosbestic regression "
            "did not remove -- check its R^2 and the 415 channel's level "
            "before interpreting any peri-event effect against this noise."
        )

    if smooth:
        from scipy.signal import savgol_filter
        if smooth % 2 == 0:
            smooth += 1
        for j in range(n_roi):
            col = dff[:, j]
            good = np.isfinite(col)
            if good.sum() > smooth:
                col[good] = savgol_filter(col[good], smooth, 3)
                dff[:, j] = col

    return ProcessedTraces(
        dff=dff, dff_raw=dff_raw, f0=f0, isos_dff=iso_dff,
        t=dm.signal_t, frames=dm.signal_frames, blanked=blanked,
        labels=dm.labels, epochs=ep_t,
        qc={
            "effective_fps": fs_eff,
            "exposure_s": exposure_s,
            "n_blanked": int(blanked.sum()),
            "n_signal": int(n_sig),
            "n_isosbestic": int(dm.isosbestic.shape[0]),
            "regression_r2": r2,
            "regression_alpha": alphas,
            "dff_sd_before_regression": sd_before.tolist(),
            "dff_sd_after_regression": sd_after.tolist(),
            "isosbestic_f0_median": iso_level.tolist(),
            "n_roi_high_noise": len(noisy),
            "common_mode": cm_info,
            "baseline_mode": baseline_mode,
            "baseline_method": baseline_method,
            "f0_boundary_jumps": f0_jumps,
            "large_sd_dff_threshold": large_sd_dff,
            "predictors": None if single_channel else predictors,
            "regression_method": (None if single_channel or not regress
                                  else regress_method),
            "regression_method_chosen": (chosen if chosen else None),
            "stim_artifact_regression": bool(stim_artifact),
            # Per-fiber amplitude of the kinetic (indicator-shaped) term.
            # This is the response estimate when the stimulus-locked
            # regressors are fitted: the square term has absorbed whatever
            # had no kinetics, which an optical artifact does not.
            "stim_kinetic_amplitude": (stim_amp if stim_amp else None),
            "single_channel": single_channel,
            "baseline_tau_s": baseline_tau_s,
            "lambda": lam_value,
            "differences": differences,
            "smooth_window": smooth,
            "epoch_samples": ep_idx,
            "warnings": warnings_out,
        },
    )
