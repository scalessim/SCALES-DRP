"""
Brandt-style detector linearity calibration.
Coefficient convention (ascending powers):
    corrected = c0 + c1*M + c2*M**2 + ...
with c0=0 and c1=1 fixed, while the requested nonlinear powers are fitted.
"""
from __future__ import annotations

import time
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from astropy.io import fits

BRANDT_DQ = {
    "INPUT_BPM": np.uint32(1 << 0),
    "NONFINITE_INPUT": np.uint32(1 << 1),
    "INVALID_SATURATION": np.uint32(1 << 2),
    "TOO_FEW_SAMPLES": np.uint32(1 << 3),
    "FIT_FAILED": np.uint32(1 << 4),
    "NONFINITE_COEFFICIENT": np.uint32(1 << 5),
    "NONMONOTONIC": np.uint32(1 << 6),
    "HIGH_CHISQ": np.uint32(1 << 7),
    "IDENTITY_FALLBACK": np.uint32(1 << 8),
    "INVALID_GAIN": np.uint32(1 << 9),
    "INVALID_READNOISE": np.uint32(1 << 10),
    "EXCESSIVE_CORRECTION": np.uint32(1 << 11),
    "EXCESSIVE_DERIVATIVE": np.uint32(1 << 12),
}


def validate_polynomial_terms(polynomial_terms: Iterable[int]) -> tuple[int, ...]:
    """Validate and normalize the nonlinear powers to fit."""
    try:
        terms = tuple(int(term) for term in polynomial_terms)
    except TypeError as exc:
        raise TypeError("polynomial_terms must be an iterable of integers.") from exc

    if not terms:
        raise ValueError("polynomial_terms must contain at least one power >= 2.")
    if any(term < 2 for term in terms):
        raise ValueError("All polynomial_terms must be integers >= 2.")
    if len(set(terms)) != len(terms):
        raise ValueError("polynomial_terms must not contain duplicates.")

    return tuple(sorted(terms))


def coefficient_count(polynomial_terms: Iterable[int]) -> int:
    """Return the full coefficient-vector length, including zero-filled gaps."""
    terms = validate_polynomial_terms(polynomial_terms)
    return max(terms) + 1


def identity_coefficients(
    polynomial_terms: Iterable[int],
    *,
    dtype=np.float64,
) -> np.ndarray:
    """Return identity coefficients with enough space for the requested terms."""
    coeff = np.zeros(coefficient_count(polynomial_terms), dtype=dtype)
    coeff[1] = 1.0
    return coeff


def evaluate_polynomial(measured, coefficients) -> np.ndarray:
    """Evaluate one ascending-order polynomial."""
    measured = np.asarray(measured)
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 1 or coefficients.size < 2:
        raise ValueError("coefficients must be a 1-D array with at least 2 values.")
    return np.polynomial.polynomial.polyval(measured, coefficients)


def evaluate_pixel_polynomials(measured, coefficients) -> np.ndarray:
    """Evaluate one polynomial per pixel along the last data axis."""
    measured = np.asarray(measured)
    coefficients = np.asarray(coefficients)
    if measured.ndim < 1:
        raise ValueError("measured must have at least one dimension.")
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape (n_coefficients, n_pixels).")
    if coefficients.shape[1] != measured.shape[-1]:
        raise ValueError("Coefficient pixel count must match measured.shape[-1].")

    out = np.zeros(measured.shape, dtype=np.result_type(measured, coefficients, np.float32))
    for order in range(coefficients.shape[0] - 1, -1, -1):
        out *= measured
        out += coefficients[order]
    return out


def polynomial_derivative_coefficients(coefficients) -> np.ndarray:
    """Return ascending-order derivative coefficients."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 1 or coefficients.size < 2:
        raise ValueError("coefficients must be a 1-D array with at least 2 values.")
    return coefficients[1:] * np.arange(1, coefficients.size, dtype=np.float64)


def is_polynomial_monotonic(
    coefficients,
    minimum_value,
    maximum_value,
    *,
    n_test_points=256,
    derivative_tolerance=-1.0e-6,
) -> tuple[bool, float]:
    """Test whether the correction is monotonically increasing over a range."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    minimum_value = float(minimum_value)
    maximum_value = float(maximum_value)
    n_test_points = int(n_test_points)

    if (
        not np.all(np.isfinite(coefficients))
        or not np.isfinite(minimum_value)
        or not np.isfinite(maximum_value)
        or maximum_value <= minimum_value
    ):
        return False, np.nan
    if n_test_points < 2:
        raise ValueError("n_test_points must be at least 2.")

    values = np.linspace(minimum_value, maximum_value, n_test_points)
    derivative = np.polynomial.polynomial.polyval(
        values, polynomial_derivative_coefficients(coefficients)
    )
    if not np.all(np.isfinite(derivative)):
        return False, np.nan

    minimum_derivative = float(np.min(derivative))
    return minimum_derivative >= float(derivative_tolerance), minimum_derivative


def compute_chisq(residuals, variance, *, n_parameters=None):
    """Return chi-square, reduced chi-square, valid count, and dof."""
    residuals, variance = np.broadcast_arrays(
        np.asarray(residuals, dtype=np.float64),
        np.asarray(variance, dtype=np.float64),
    )
    valid = np.isfinite(residuals) & np.isfinite(variance) & (variance > 0)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return np.nan, np.nan, 0, None

    chisq = float(np.sum(residuals[valid] ** 2 / variance[valid]))
    if n_parameters is None:
        return chisq, np.nan, n_valid, None

    dof = n_valid - int(n_parameters)
    redchisq = chisq / dof if dof > 0 else np.nan
    return chisq, redchisq, n_valid, dof


def estimate_ramp_pedestals(
    pixel_ramps,
    *,
    n_pedestal_reads=2,
    pedestal_start_read=0,
):
    """Estimate one median pedestal for each ramp."""
    pixel_ramps = np.asarray(pixel_ramps, dtype=np.float64)
    if pixel_ramps.ndim != 2:
        raise ValueError("pixel_ramps must have shape (n_ramps, n_reads).")

    n_pedestal_reads = int(n_pedestal_reads)
    pedestal_start_read = int(pedestal_start_read)
    if n_pedestal_reads < 1 or pedestal_start_read < 0:
        raise ValueError("Pedestal read settings must be non-negative and non-empty.")

    stop = min(pedestal_start_read + n_pedestal_reads, pixel_ramps.shape[1])
    if pedestal_start_read >= stop:
        raise ValueError("No reads are available for pedestal estimation.")

    region = pixel_ramps[:, pedestal_start_read:stop]
    finite_count = np.sum(np.isfinite(region), axis=1, dtype=np.uint16)
    with np.errstate(invalid="ignore"):
        pedestal = np.nanmedian(region, axis=1)
    pedestal[finite_count == 0] = np.nan
    return pedestal, finite_count


def build_variance_model(
    signal,
    *,
    gain,
    readnoise,
    minimum_variance=1.0e-12,
):
    """Construct variance in DN^2 from Poisson noise plus read noise."""
    signal = np.asarray(signal, dtype=np.float64)
    gain = float(gain)
    readnoise = float(readnoise)
    minimum_variance = float(minimum_variance)

    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be finite and > 0.")
    if not np.isfinite(readnoise) or readnoise < 0:
        raise ValueError("readnoise must be finite and >= 0.")
    if not np.isfinite(minimum_variance) or minimum_variance <= 0:
        raise ValueError("minimum_variance must be finite and > 0.")

    variance = (readnoise / gain) ** 2 + np.maximum(signal, 0.0) / gain
    variance = np.maximum(variance, minimum_variance)
    variance[~np.isfinite(signal)] = np.nan
    return variance


def prepare_pixel_data(
    pixel_ramps,
    saturation,
    *,
    gain,
    readnoise,
    skip_initial_reads=2,
    n_pedestal_reads=2,
    saturation_fraction=0.95,
    minimum_reads_per_ramp=4,
    minimum_total_samples=None,
    minimum_valid_ramps=2,
    require_positive_signal=True,
    debug=False,
):
    """Prepare one pixel's ramp samples, mask, pedestal, and variance."""

    ramps = np.asarray(pixel_ramps, dtype=np.float64)

    if ramps.ndim != 2:
        raise ValueError(
            "pixel_ramps must have shape (n_ramps, n_reads)."
        )

    nramps, _ = ramps.shape
    quality = np.uint32(0)

    skip_initial_reads = int(skip_initial_reads)
    minimum_reads_per_ramp = int(minimum_reads_per_ramp)
    minimum_valid_ramps = int(minimum_valid_ramps)

    if minimum_total_samples is None:
        minimum_total_samples = (
            minimum_reads_per_ramp * minimum_valid_ramps
        )

    minimum_total_samples = int(minimum_total_samples)

    def failed_result(flag):
        return {
            "SIGNAL": np.full_like(ramps, np.nan),
            "VARIANCE": np.full_like(ramps, np.nan),
            "VALID_MASK": np.zeros_like(ramps, dtype=bool),
            "PEDESTAL": np.full(nramps, np.nan),
            "NVALID_PER_RAMP": np.zeros(
                nramps,
                dtype=np.uint16,
            ),
            "NVALID_TOTAL": 0,
            "NVALID_RAMPS": 0,
            "SUCCESS": False,
            "QUALITY": np.uint32(quality | flag),
        }

    # ------------------------------------------------------------
    # Validate scalar calibration inputs
    # ------------------------------------------------------------

    if not np.isfinite(saturation) or saturation <= 0:
        return failed_result(
            BRANDT_DQ["INVALID_SATURATION"]
        )

    if not np.isfinite(gain) or gain <= 0:
        return failed_result(
            BRANDT_DQ["INVALID_GAIN"]
        )

    if not np.isfinite(readnoise) or readnoise < 0:
        return failed_result(
            BRANDT_DQ["INVALID_READNOISE"]
        )

    if not np.all(np.isfinite(ramps)):
        quality |= BRANDT_DQ["NONFINITE_INPUT"]

    # ------------------------------------------------------------
    # Estimate and subtract one pedestal per ramp
    # ------------------------------------------------------------

    pedestal, _ = estimate_ramp_pedestals(
        ramps,
        n_pedestal_reads=n_pedestal_reads,
        pedestal_start_read=0,
    )

    signal = ramps - pedestal[:, None]

    # ------------------------------------------------------------
    # Convert absolute saturation DN to pedestal-subtracted signal
    #
    # saturation                : absolute detector DN
    # saturation - pedestal     : pedestal-subtracted saturation
    # saturation_fraction       : safety fraction of usable range
    # ------------------------------------------------------------

    usable_saturation_signal = (
        float(saturation) - pedestal
    ) * float(saturation_fraction)

    valid_saturation_range = (
        np.isfinite(usable_saturation_signal)
        & (usable_saturation_signal > 0.0)
    )

    # ------------------------------------------------------------
    # Construct initial valid-read mask
    # ------------------------------------------------------------

    valid = (
        np.isfinite(ramps)
        & np.isfinite(signal)
    )

    valid &= valid_saturation_range[:, None]

    #valid &= (
    #    signal
    #    < usable_saturation_signal[:, None])

    crossed_saturation = (
        ~np.isfinite(signal)
        | (signal >= usable_saturation_signal[:, None]))

    # Once a ramp crosses saturation, all later reads are invalid.
    after_saturation = np.maximum.accumulate(
        crossed_saturation,
        axis=1)

    valid &= ~after_saturation

    if skip_initial_reads > 0:
        valid[:, :skip_initial_reads] = False

    if require_positive_signal:
        valid &= signal >= 0.0

    # ------------------------------------------------------------
    # Optional single-pixel diagnostics
    # ------------------------------------------------------------

    if debug:
        print("\n" + "=" * 70)
        print("prepare_pixel_data diagnostics")
        print("-" * 70)
        print(f"Absolute saturation DN : {float(saturation):.6f}")
        print(f"Saturation fraction    : {float(saturation_fraction):.6f}")
        print("Pedestal per ramp      :", pedestal)
        print("Signal fit limit       :", usable_saturation_signal)

        print(
            "Valid reads before minimum-read test:",
            np.sum(valid, axis=1),
        )

        for ramp_index in range(nramps):
            ramp_valid = np.flatnonzero(valid[ramp_index])

            if ramp_valid.size:
                last_read = int(ramp_valid[-1])
                maximum_valid_signal = float(
                    np.nanmax(
                        signal[
                            ramp_index,
                            ramp_valid,
                        ]
                    )
                )

                print(
                    f"Ramp {ramp_index}: "
                    f"last valid read={last_read}, "
                    f"maximum valid signal="
                    f"{maximum_valid_signal:.6f}"
                )
            else:
                print(
                    f"Ramp {ramp_index}: "
                    "no valid reads"
                )

        print("=" * 70)

    # ------------------------------------------------------------
    # Reject complete ramps with too few usable reads
    # ------------------------------------------------------------

    initial_counts = np.sum(valid, axis=1)

    valid &= (
        initial_counts >= minimum_reads_per_ramp
    )[:, None]

    # ------------------------------------------------------------
    # Build variance model and reject invalid variances
    # ------------------------------------------------------------

    variance = build_variance_model(
        signal,
        gain=gain,
        readnoise=readnoise,
    )

    valid &= (
        np.isfinite(variance)
        & (variance > 0)
    )

    # ------------------------------------------------------------
    # Final sample accounting
    # ------------------------------------------------------------

    per_ramp = np.sum(
        valid,
        axis=1,
        dtype=np.uint16,
    )

    total = int(np.sum(per_ramp))
    n_valid_ramps = int(np.count_nonzero(per_ramp))

    success = (
        total >= minimum_total_samples
        and n_valid_ramps >= minimum_valid_ramps
    )

    if not success:
        quality |= BRANDT_DQ["TOO_FEW_SAMPLES"]

    return {
        "SIGNAL": signal,
        "VARIANCE": variance,
        "VALID_MASK": valid,
        "PEDESTAL": pedestal,
        "USABLE_SATURATION_SIGNAL": usable_saturation_signal,
        "NVALID_PER_RAMP": per_ramp,
        "NVALID_TOTAL": total,
        "NVALID_RAMPS": n_valid_ramps,
        "SUCCESS": bool(success),
        "QUALITY": quality,
    }


def _build_design_matrix(signal, valid_mask, polynomial_terms, read_times=None):
    """Build the shared-nonlinearity, per-ramp line-fit system."""
    signal = np.asarray(signal, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    terms = validate_polynomial_terms(polynomial_terms)
    if signal.ndim != 2 or valid_mask.shape != signal.shape:
        raise ValueError("signal and valid_mask must share shape (n_ramps, n_reads).")

    nramps, nreads = signal.shape
    if read_times is None:
        times = np.broadcast_to(np.arange(nreads, dtype=np.float64), signal.shape)
    else:
        read_times = np.asarray(read_times, dtype=np.float64)
        if read_times.shape == (nreads,):
            times = np.broadcast_to(read_times, signal.shape)
        elif read_times.shape == signal.shape:
            times = read_times
        else:
            raise ValueError("read_times must have shape (n_reads,) or (n_ramps,n_reads).")

    fit_mask = valid_mask & np.isfinite(signal) & np.isfinite(times)
    ramp_idx, read_idx = np.nonzero(fit_mask)
    measured = signal[ramp_idx, read_idx]
    sample_times = times[ramp_idx, read_idx]

    n_terms = len(terms)
    intercept_start = n_terms
    slope_start = intercept_start + nramps
    n_parameters = n_terms + 2 * nramps

    design = np.zeros((measured.size, n_parameters), dtype=np.float64)
    for column, power in enumerate(terms):
        design[:, column] = measured**power

    rows = np.arange(measured.size)
    design[rows, intercept_start + ramp_idx] = -1.0
    design[rows, slope_start + ramp_idx] = -sample_times

    layout = {
        "TERMS": terms,
        "NONLINEAR": slice(0, n_terms),
        "INTERCEPT": slice(intercept_start, slope_start),
        "SLOPE": slice(slope_start, n_parameters),
        "NPARAM": n_parameters,
    }
    return design, -measured, ramp_idx, read_idx, layout


def _solve_weighted(design, target, variance, *, rcond=None):

    valid = (
        np.all(np.isfinite(design), axis=1)
        & np.isfinite(target)
        & np.isfinite(variance)
        & (variance > 0)
    )

    if not np.any(valid):
        raise np.linalg.LinAlgError("No valid samples remain for fitting.")

    weights = 1.0 / np.sqrt(variance[valid])

    A = design[valid] * weights[:, None]
    b = target[valid] * weights

    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1.0

    A_scaled = A / scale

    parameters_scaled, _, rank, singular_values = np.linalg.lstsq(
        A_scaled,
        b,
        rcond=rcond,
    )

    parameters = parameters_scaled / scale

    return parameters, int(rank), singular_values


def _empty_pixel_result(polynomial_terms, nramps, quality=0):
    return {
        "COEFF": identity_coefficients(polynomial_terms),
        "QUALITY": np.uint32(quality),
        "CHISQ": np.nan,
        "REDCHISQ": np.nan,
        "NVALID": 0,
        "DOF": 0,
        "PEDESTAL": np.full(nramps, np.nan),
        "SLOPE": np.full(nramps, np.nan),
        "INTERCEPT": np.full(nramps, np.nan),
        "RESIDUAL_RMS": np.nan,
        "MIN_DERIVATIVE": np.nan,
        "NITER": 0,
        "SUCCESS": False,
        "VALID_MASK": None,
    }


def detect_sustained_plateau(
    signal,
    *,
    skip_initial_reads=2,
    reference_diff_reads=20,
    plateau_fraction=0.60,
    small_difference_fraction=0.70,
    window=15,
    required_fraction=0.70,
):
    """
    Fast detection of the first sustained plateau read in one ramp.

    This version avoids looping over every candidate read window.
    """

    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")

    differences = np.diff(signal)
    n_differences = differences.size

    if n_differences < window:
        return None

    early_start = max(int(skip_initial_reads), 0)
    early_stop = min(
        early_start + int(reference_diff_reads),
        n_differences,
    )

    early = differences[early_start:early_stop]
    early = early[np.isfinite(early)]

    if early.size == 0:
        return None

    reference_difference = np.nanmedian(early)

    if (
        not np.isfinite(reference_difference)
        or reference_difference <= 0
    ):
        return None

    signed_threshold = (
        float(plateau_fraction) * reference_difference
    )

    small_threshold = (
        float(small_difference_fraction)
        * reference_difference
    )

    finite = np.isfinite(differences)

    # Replace nonfinite values only for convolution.
    clean_difference = np.where(
        finite,
        differences,
        0.0,
    )

    kernel = np.ones(
        int(window),
        dtype=np.float64,
    )

    # Number of finite samples in each forward window.
    finite_count = np.convolve(
        finite.astype(np.float64),
        kernel,
        mode="valid",
    )

    # Moving signed mean.
    #
    # This is used instead of a moving signed median because it is much
    # faster. Positive and negative plateau noise still approximately
    # cancels.
    moving_signed_mean = np.convolve(
        clean_difference,
        kernel,
        mode="valid",
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        moving_signed_mean = (
            moving_signed_mean / finite_count
        )

    small_difference = (
        finite
        & (np.abs(differences) <= small_threshold)
    )

    small_count = np.convolve(
        small_difference.astype(np.float64),
        kernel,
        mode="valid",
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        moving_small_fraction = (
            small_count / finite_count
        )

    minimum_finite = int(
        np.ceil(required_fraction * window)
    )

    candidate = (
        (finite_count >= minimum_finite)
        & np.isfinite(moving_signed_mean)
        & (
            np.abs(moving_signed_mean)
            <= signed_threshold
        )
        & (
            moving_small_fraction
            >= required_fraction
        )
    )

    # Do not search inside the early reference region.
    candidate[:early_stop] = False

    candidate_indices = np.flatnonzero(candidate)

    if candidate_indices.size == 0:
        return None

    # The moving window begins at this difference index.
    # Difference i lies between read i and read i+1.
    return int(candidate_indices[0] + 1)


def exclude_plateau_reads(
    signal,
    valid_mask,
    *,
    skip_initial_reads=2,
    reference_diff_reads=20,
    plateau_fraction=0.60,
    small_difference_fraction=0.70,
    plateau_window=15,
    required_fraction=0.70,
):
    """
    Remove plateau reads from a Brandt fitting mask.

    Parameters
    ----------
    signal : ndarray
        Shape (n_ramps, n_reads).

    valid_mask : ndarray
        Existing Boolean mask with the same shape.

    Returns
    -------
    updated_mask : ndarray
        Valid mask after plateau reads are excluded.

    plateau_start : ndarray
        First plateau read for each ramp. -1 means none detected.
    """

    signal = np.asarray(signal, dtype=np.float64)
    updated_mask = np.asarray(valid_mask, dtype=bool).copy()

    if signal.ndim != 2:
        raise ValueError(
            "signal must have shape (n_ramps, n_reads)."
        )

    if updated_mask.shape != signal.shape:
        raise ValueError(
            "valid_mask must have the same shape as signal."
        )

    n_ramps, _ = signal.shape

    plateau_start = np.full(
        n_ramps,
        -1,
        dtype=np.int32,
    )

    for ramp_index in range(n_ramps):
        first_plateau_read = detect_sustained_plateau(
            signal[ramp_index],
            skip_initial_reads=skip_initial_reads,
            reference_diff_reads=reference_diff_reads,
            plateau_fraction=plateau_fraction,
            small_difference_fraction=small_difference_fraction,
            window=plateau_window,
            required_fraction=required_fraction,
        )

        if first_plateau_read is not None:
            plateau_start[ramp_index] = first_plateau_read

            updated_mask[
                ramp_index,
                first_plateau_read:
            ] = False

        if skip_initial_reads > 0:
            updated_mask[
                ramp_index,
                :skip_initial_reads
            ] = False

    return updated_mask, plateau_start


def fit_brandt_pixel(
    pixel_ramps,
    saturation,
    *,
    gain,
    readnoise,
    polynomial_terms=(2, 3),
    read_times=None,
    skip_initial_reads=2,
    n_pedestal_reads=2,
    saturation_fraction=0.95,
    minimum_reads_per_ramp=4,
    minimum_total_samples=None,
    minimum_valid_ramps=2,
    require_positive_signal=True,
    sigma_clip=5.0,
    maximum_iterations=3,
    maximum_reduced_chisq=None,
    check_monotonic=True,
    derivative_tolerance=-1.0e-6,
    n_monotonic_test_points=256,
    rcond=None,
    maximum_correction_fraction=None,
    correction_floor_dn=100.0,
    maximum_derivative=None,
):
    """Fit one pixel's requested nonlinear polynomial terms."""
    ramps = np.asarray(pixel_ramps, dtype=np.float64)
    if ramps.ndim != 2:
        raise ValueError("pixel_ramps must have shape (n_ramps, n_reads).")
    nramps = ramps.shape[0]
    terms = validate_polynomial_terms(polynomial_terms)
    maximum_iterations = int(maximum_iterations)
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be at least 1.")

    prepared = prepare_pixel_data(
        ramps,
        saturation,
        gain=gain,
        readnoise=readnoise,
        skip_initial_reads=skip_initial_reads,
        n_pedestal_reads=n_pedestal_reads,
        saturation_fraction=saturation_fraction,
        minimum_reads_per_ramp=minimum_reads_per_ramp,
        minimum_total_samples=minimum_total_samples,
        minimum_valid_ramps=minimum_valid_ramps,
        require_positive_signal=require_positive_signal,
    )

    quality = np.uint32(prepared["QUALITY"])
    if not prepared["SUCCESS"]:
        result = _empty_pixel_result(terms, nramps, quality | BRANDT_DQ["IDENTITY_FALLBACK"])
        result["PEDESTAL"] = prepared["PEDESTAL"]
        result["VALID_MASK"] = prepared["VALID_MASK"]
        return result

    signal = prepared["SIGNAL"]
    variance = prepared["VARIANCE"]
    working_mask = prepared["VALID_MASK"].copy()

    # ------------------------------------------------------------
    # Remove plateau reads before constructing the Brandt fit
    # ------------------------------------------------------------
    #print("Signal max:", np.nanmax(signal, axis=1))
    #print("Signal last:", signal[:, -5:])

    working_mask, plateau_start = exclude_plateau_reads(
        signal,
        working_mask,
        skip_initial_reads=skip_initial_reads,
        reference_diff_reads=20,
        plateau_fraction=0.60,
        small_difference_fraction=0.70,
        plateau_window=15,
        required_fraction=0.70,
    )

    #for i in range(working_mask.shape[0]):
    #    valid = np.flatnonzero(working_mask[i])
    #    if valid.size:
    #        print(f"Ramp {i}: last valid read = {valid[-1]}")
    #    else:
    #        print(f"Ramp {i}: no valid reads")

    niter_completed = 0
    def fallback(flag, mask=working_mask):
        result = _empty_pixel_result(terms, nramps, quality | flag | BRANDT_DQ["IDENTITY_FALLBACK"])
        result.update(PEDESTAL=prepared["PEDESTAL"], VALID_MASK=mask.copy(), NITER=niter_completed,
                      PLATEAU_START=plateau_start)
        return result

    def solve(mask):
        design, target, ri, ti, layout = _build_design_matrix(signal, mask, terms, read_times)
        variances = variance[ri, ti]
        if target.size <= layout["NPARAM"] or np.count_nonzero(np.sum(mask, axis=1) >= minimum_reads_per_ramp) < minimum_valid_ramps:
            return None
        try:
            pars, rank, _ = _solve_weighted(design, target, variances, rcond=rcond)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return None
        if rank < layout["NPARAM"] or not np.all(np.isfinite(pars)):
            return None
        return pars, design, target, ri, ti, layout, variances

    fitted = None
    for iteration in range(maximum_iterations):
        niter_completed = iteration + 1
        fitted = solve(working_mask)
        if fitted is None:
            return fallback(BRANDT_DQ["FIT_FAILED"])
        parameters, design, target, sample_ramps, sample_reads, layout, sample_variance = fitted
        if sigma_clip is None:
            break
        if not np.isfinite(sigma_clip) or float(sigma_clip) <= 0:
            raise ValueError("sigma_clip must be finite and positive or None")
        residuals = target - design @ parameters
        keep = np.isfinite(residuals) & (np.abs(residuals / np.sqrt(sample_variance)) <= float(sigma_clip))
        if np.all(keep):
            break
        new_mask = working_mask.copy()
        new_mask[sample_ramps[~keep], sample_reads[~keep]] = False
        if np.array_equal(new_mask, working_mask):
            break
        working_mask = new_mask
        fitted = None  # Mask changed: coefficients must be refitted.

    if fitted is None:
        fitted = solve(working_mask)
        if fitted is None:
            return fallback(BRANDT_DQ["FIT_FAILED"] | BRANDT_DQ["TOO_FEW_SAMPLES"])
    parameters, design, target, sample_ramps, sample_reads, layout, sample_variance = fitted
    coefficients = identity_coefficients(terms)
    for value, power in zip(parameters[layout["NONLINEAR"]], terms):
        coefficients[power] = value
    intercepts = parameters[layout["INTERCEPT"]].copy()
    slopes = parameters[layout["SLOPE"]].copy()
    residuals = target - design @ parameters
    chisq, redchisq, n_valid, dof = compute_chisq(residuals, sample_variance, n_parameters=layout["NPARAM"])
    residual_rms = float(np.sqrt(np.mean(residuals**2))) if n_valid else np.nan
    success = True
    if maximum_reduced_chisq is not None and (not np.isfinite(redchisq) or redchisq > maximum_reduced_chisq):
        quality |= BRANDT_DQ["HIGH_CHISQ"]
        success = False
    valid_signal = signal[working_mask]
    minimum_derivative = np.nan
    if valid_signal.size < 2:
        success = False
        quality |= BRANDT_DQ["TOO_FEW_SAMPLES"]
    else:
        xmin = min(0.0, float(np.min(valid_signal)))
        xmax = float(np.max(valid_signal))
        grid = np.linspace(xmin, xmax, max(2, int(n_monotonic_test_points)))
        derivative = np.polynomial.polynomial.polyval(grid, polynomial_derivative_coefficients(coefficients))
        if not np.all(np.isfinite(derivative)):
            success = False
            quality |= BRANDT_DQ["NONFINITE_COEFFICIENT"]
        else:
            minimum_derivative = float(np.min(derivative))
            if check_monotonic and minimum_derivative < derivative_tolerance:
                quality |= BRANDT_DQ["NONMONOTONIC"]
                success = False
            if maximum_derivative is not None and np.max(derivative) > maximum_derivative:
                quality |= BRANDT_DQ["EXCESSIVE_DERIVATIVE"]
                success = False
        if maximum_correction_fraction is not None:
            mapped = evaluate_polynomial(grid, coefficients)
            fraction = np.abs(mapped - grid) / np.maximum(np.abs(grid), correction_floor_dn)
            if not np.all(np.isfinite(fraction)) or np.max(fraction) > maximum_correction_fraction:
                quality |= BRANDT_DQ["EXCESSIVE_CORRECTION"]
                success = False
    if not success:
        quality |= BRANDT_DQ["IDENTITY_FALLBACK"]
        coefficients = identity_coefficients(terms)
    return {
        "COEFF": coefficients, "QUALITY": quality, "CHISQ": chisq,
        "REDCHISQ": redchisq, "NVALID": n_valid, "DOF": dof,
        "PEDESTAL": prepared["PEDESTAL"], "SLOPE": slopes, "INTERCEPT": intercepts,
        "RESIDUAL_RMS": residual_rms, "MIN_DERIVATIVE": minimum_derivative,
        "NITER": niter_completed, "SUCCESS": success,
        "VALID_MASK": working_mask, "PLATEAU_START": plateau_start,
    }


def _prepare_map(value, shape, *, name, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return np.full(shape, array.item(), dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} must be scalar or have shape {shape}; got {array.shape}.")
    return array


def _allocate_block_outputs(nramps, nrows, nx, polynomial_terms):
    terms = validate_polynomial_terms(polynomial_terms)
    ncoeff = coefficient_count(terms)
    coeff = np.broadcast_to(
        identity_coefficients(terms)[:, None, None], (ncoeff, nrows, nx)
    ).copy()
    return {
        "COEFF": coeff,
        "QUALITY": np.zeros((nrows, nx), dtype=np.uint32),
        "CHISQ": np.full((nrows, nx), np.nan),
        "REDCHISQ": np.full((nrows, nx), np.nan),
        "NVALID": np.zeros((nrows, nx), dtype=np.uint32),
        "DOF": np.zeros((nrows, nx), dtype=np.int32),
        "PEDESTAL": np.full((nramps, nrows, nx), np.nan),
        "SLOPE": np.full((nramps, nrows, nx), np.nan),
        "INTERCEPT": np.full((nramps, nrows, nx), np.nan),
        "RESIDUAL_RMS": np.full((nrows, nx), np.nan),
        "MIN_DERIVATIVE": np.full((nrows, nx), np.nan),
        "NITER": np.zeros((nrows, nx), dtype=np.uint16),
        "SUCCESS": np.zeros((nrows, nx), dtype=bool),
        "POLYNOMIAL_TERMS": terms,
    }


def fit_brandt_block(
    ramps,
    saturation_map,
    *,
    bad_pixel_map=None,
    gain=1.0,
    readnoise=0.0,
    polynomial_terms=(2, 3),
    read_times=None,
    skip_initial_reads=2,
    n_pedestal_reads=2,
    saturation_fraction=0.95,
    minimum_reads_per_ramp=4,
    minimum_total_samples=None,
    minimum_valid_ramps=2,
    require_positive_signal=True,
    sigma_clip=5.0,
    maximum_iterations=3,
    maximum_reduced_chisq=None,
    check_monotonic=True,
    derivative_tolerance=-1.0e-6,
    n_monotonic_test_points=256,
    rcond=None,
    maximum_correction_fraction=None,
    correction_floor_dn=100.0,
    maximum_derivative=None,
    progress=False,
    print_summary=True,
):
    """
    Fit the requested Brandt linearity terms for every pixel in a detector block.

    Parameters
    ----------
    ramps : array-like
        Detector ramps with shape
        ``(n_ramps, n_reads, nrows, nx)``.

    saturation_map : array-like
        Saturation level for each pixel, with shape ``(nrows, nx)``.
        A scalar value may also be accepted if supported by ``_prepare_map``.

    bad_pixel_map : array-like or None
        Boolean bad-pixel mask with shape ``(nrows, nx)``. Masked pixels
        receive identity coefficients and the INPUT_BPM and
        IDENTITY_FALLBACK quality flags.

    gain : float or array-like
        Detector gain in electrons per DN.

    readnoise : float or array-like
        Read noise in electrons.

    polynomial_terms : sequence of int
        Nonlinear polynomial powers to fit, for example ``(2, 3, 4)``.

    read_times : array-like or None
        Read times with shape ``(n_reads,)`` or
        ``(n_ramps, n_reads)``.

    progress : bool
        Print progress periodically while processing the block.

    print_summary : bool
        Print detailed quality-flag and pixel-accounting summaries after
        processing the complete block.

    Returns
    -------
    outputs : dict
        Dictionary containing the block-level calibration products.
    """

    # ------------------------------------------------------------------
    # Validate the ramp cube
    # ------------------------------------------------------------------
    ramps = np.asarray(ramps)

    if ramps.ndim != 4:
        raise ValueError(
            "ramps must have shape "
            "(n_ramps, n_reads, nrows, nx)."
        )

    nramps, nreads, nrows, nx = ramps.shape
    shape = (nrows, nx)
    npixels = nrows * nx

    terms = validate_polynomial_terms(polynomial_terms)

    # ------------------------------------------------------------------
    # Validate scalar or detector-map inputs
    # ------------------------------------------------------------------
    saturation = _prepare_map(
        saturation_map,
        shape,
        name="saturation_map",
    )

    gain_map = _prepare_map(
        gain,
        shape,
        name="gain",
    )

    readnoise_map = _prepare_map(
        readnoise,
        shape,
        name="readnoise",
    )

    if bad_pixel_map is None:
        bpm = np.zeros(shape, dtype=bool)
    else:
        bpm = _prepare_map(
            bad_pixel_map,
            shape,
            name="bad_pixel_map",
            dtype=bool,
        )

    # ------------------------------------------------------------------
    # Validate read times
    # ------------------------------------------------------------------
    if read_times is not None:
        read_times = np.asarray(read_times, dtype=np.float64)

        allowed_shapes = {
            (nreads,),
            (nramps, nreads),
        }

        if read_times.shape not in allowed_shapes:
            raise ValueError(
                "read_times must have shape "
                f"{(nreads,)} or {(nramps, nreads)}; "
                f"received {read_times.shape}."
            )

    # ------------------------------------------------------------------
    # Allocate block outputs
    # ------------------------------------------------------------------
    outputs = _allocate_block_outputs(
        nramps,
        nrows,
        nx,
        terms,
    )

    # ------------------------------------------------------------------
    # Flatten detector dimensions for pixel-wise processing
    # ------------------------------------------------------------------
    ramps_flat = ramps.reshape(
        nramps,
        nreads,
        npixels,
    )

    saturation_flat = saturation.ravel()
    gain_flat = gain_map.ravel()
    readnoise_flat = readnoise_map.ravel()
    bpm_flat = bpm.ravel()

    flat = {
        "COEFF": outputs["COEFF"].reshape(
            outputs["COEFF"].shape[0],
            npixels,
        ),
        "QUALITY": outputs["QUALITY"].ravel(),
        "CHISQ": outputs["CHISQ"].ravel(),
        "REDCHISQ": outputs["REDCHISQ"].ravel(),
        "NVALID": outputs["NVALID"].ravel(),
        "DOF": outputs["DOF"].ravel(),
        "PEDESTAL": outputs["PEDESTAL"].reshape(
            nramps,
            npixels,
        ),
        "SLOPE": outputs["SLOPE"].reshape(
            nramps,
            npixels,
        ),
        "INTERCEPT": outputs["INTERCEPT"].reshape(
            nramps,
            npixels,
        ),
        "RESIDUAL_RMS": outputs["RESIDUAL_RMS"].ravel(),
        "MIN_DERIVATIVE": outputs["MIN_DERIVATIVE"].ravel(),
        "NITER": outputs["NITER"].ravel(),
        "SUCCESS": outputs["SUCCESS"].ravel(),
    }

    vector_output_keys = {
        "COEFF",
        "PEDESTAL",
        "SLOPE",
        "INTERCEPT",
    }

    result_keys = (
        "COEFF",
        "QUALITY",
        "CHISQ",
        "REDCHISQ",
        "NVALID",
        "DOF",
        "PEDESTAL",
        "SLOPE",
        "INTERCEPT",
        "RESIDUAL_RMS",
        "MIN_DERIVATIVE",
        "NITER",
        "SUCCESS",
    )

    # ------------------------------------------------------------------
    # Process pixels
    # ------------------------------------------------------------------
    for pixel in range(npixels):

        # --------------------------------------------------------------
        # Input BPM pixels are skipped
        # --------------------------------------------------------------
        if bpm_flat[pixel]:
            flat["QUALITY"][pixel] = (
                BRANDT_DQ["INPUT_BPM"]
                | BRANDT_DQ["IDENTITY_FALLBACK"]
            )

            flat["SUCCESS"][pixel] = False

            # This assumes _allocate_block_outputs() initializes COEFF
            # to the identity correction. If it does not, explicitly set
            # identity coefficients here.
            continue

        result = fit_brandt_pixel(
            ramps_flat[:, :, pixel],
            saturation_flat[pixel],
            gain=gain_flat[pixel],
            readnoise=readnoise_flat[pixel],
            polynomial_terms=terms,
            read_times=read_times,
            skip_initial_reads=skip_initial_reads,
            n_pedestal_reads=n_pedestal_reads,
            saturation_fraction=saturation_fraction,
            minimum_reads_per_ramp=minimum_reads_per_ramp,
            minimum_total_samples=minimum_total_samples,
            minimum_valid_ramps=minimum_valid_ramps,
            require_positive_signal=require_positive_signal,
            sigma_clip=sigma_clip,
            maximum_iterations=maximum_iterations,
            maximum_reduced_chisq=maximum_reduced_chisq,
            check_monotonic=check_monotonic,
            derivative_tolerance=derivative_tolerance,
            n_monotonic_test_points=n_monotonic_test_points,
            rcond=rcond,
            maximum_correction_fraction=maximum_correction_fraction,
            correction_floor_dn=correction_floor_dn,
            maximum_derivative=maximum_derivative,
        )
        
        # --------------------------------------------------------------
        # Copy pixel result into block outputs
        # --------------------------------------------------------------
        for key in result_keys:
            if key not in result:
                raise KeyError(
                    f"fit_brandt_pixel() did not return required key "
                    f"{key!r}."
                )

            if key in vector_output_keys:
                flat[key][:, pixel] = result[key]
            else:
                flat[key][pixel] = result[key]

        # --------------------------------------------------------------
        # Optional progress message
        # --------------------------------------------------------------
        if progress and (
            pixel == 0
            or pixel + 1 == npixels
            or (pixel + 1) % 1000 == 0
        ):
            fraction = 100.0 * (pixel + 1) / npixels

            print(
                f"Brandt fitting: {pixel + 1}/{npixels} pixels "
                f"({fraction:.1f}%)"
            )

    # ------------------------------------------------------------------
    # End-of-block diagnostics
    # ------------------------------------------------------------------
    if print_summary:
        quality = flat["QUALITY"]
        success = flat["SUCCESS"].astype(bool)

        identity_fallback = (
            quality & BRANDT_DQ["IDENTITY_FALLBACK"]
        ) != 0

        finite_saturation = np.isfinite(saturation_flat)
        invalid_saturation = ~finite_saturation

        print("\nDetailed QUALITY flag summary")
        print("-" * 65)

        for name, bit in BRANDT_DQ.items():
            count = np.count_nonzero(
                (quality & bit) != 0
            )
            fraction = 100.0 * count / quality.size

            print(
                f"{name:<30s}: "
                f"{count:>10d} "
                f"({fraction:>7.3f}%)"
            )

        print("\nBasic pixel accounting")
        print("-" * 65)
        print(
            f"{'Total pixels':<30s}: "
            f"{quality.size:>10d}"
        )
        print(
            f"{'Successful fits':<30s}: "
            f"{np.count_nonzero(success):>10d}"
        )
        print(
            f"{'Failed/skipped pixels':<30s}: "
            f"{np.count_nonzero(~success):>10d}"
        )
        print(
            f"{'Identity fallbacks':<30s}: "
            f"{np.count_nonzero(identity_fallback):>10d}"
        )
        print(
            f"{'Finite saturation':<30s}: "
            f"{np.count_nonzero(finite_saturation):>10d}"
        )
        print(
            f"{'Invalid saturation':<30s}: "
            f"{np.count_nonzero(invalid_saturation):>10d}"
        )
        print(
            f"{'Input BPM':<30s}: "
            f"{np.count_nonzero(bpm_flat):>10d}"
        )

        # --------------------------------------------------------------
        # Check whether all failures received identity fallback
        # --------------------------------------------------------------
        failed_without_identity = (
            (~success) & (~identity_fallback)
        )

        successful_with_identity = (
            success & identity_fallback
        )

        print(
            f"{'Failures without identity':<30s}: "
            f"{np.count_nonzero(failed_without_identity):>10d}"
        )
        print(
            f"{'Success with identity flag':<30s}: "
            f"{np.count_nonzero(successful_with_identity):>10d}"
        )

        # --------------------------------------------------------------
        # Minimum-derivative distribution
        # --------------------------------------------------------------
        minimum_derivative = flat["MIN_DERIVATIVE"]
        finite_derivative = np.isfinite(minimum_derivative)

        print("\nMinimum derivative distribution")
        print("-" * 65)

        derivative_thresholds = (
            0.0,
            -1.0e-6,
            -1.0e-5,
            -1.0e-4,
            -1.0e-3,
            -1.0e-2,
            -1.0e-1,
        )

        for threshold in derivative_thresholds:
            count = np.count_nonzero(
                finite_derivative
                & (minimum_derivative < threshold)
            )

            fraction = 100.0 * count / npixels

            print(
                f"MIN_DERIVATIVE < {threshold:>9.1e}: "
                f"{count:>10d} "
                f"({fraction:>7.3f}%)"
            )

        # --------------------------------------------------------------
        # Reduced chi-square statistics for successful fits
        # --------------------------------------------------------------
        reduced_chisq = flat["REDCHISQ"]

        valid_reduced_chisq = (
            success
            & np.isfinite(reduced_chisq)
        )

        if np.any(valid_reduced_chisq):
            values = reduced_chisq[valid_reduced_chisq]

            print("\nSuccessful-fit reduced chi-square")
            print("-" * 65)
            print(
                f"{'Median':<30s}: "
                f"{np.nanmedian(values):>12.6g}"
            )
            print(
                f"{'16th percentile':<30s}: "
                f"{np.nanpercentile(values, 16):>12.6g}"
            )
            print(
                f"{'84th percentile':<30s}: "
                f"{np.nanpercentile(values, 84):>12.6g}"
            )
    return outputs


def apply_coefficients(data, coefficients):
    """Apply a coefficient cube to 2-D, 3-D, or 4-D detector data."""
    data = np.asarray(data)
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 3:
        raise ValueError("coefficients must have shape (n_coefficients, nrows, nx).")
    if data.shape[-2:] != coefficients.shape[-2:]:
        raise ValueError("The last two data axes must match the coefficient maps.")

    corrected = np.zeros(data.shape, dtype=np.result_type(data, coefficients, np.float32))
    for order in range(coefficients.shape[0] - 1, -1, -1):
        corrected *= data
        corrected += coefficients[order]
    return corrected


def summarize_brandt_block(outputs):
    """Return a compact summary of a block-fitting result."""
    quality = np.asarray(outputs["QUALITY"], dtype=np.uint32)
    success = np.asarray(outputs["SUCCESS"], dtype=bool)
    npixels = quality.size
    return {
        "NPIXELS": npixels,
        "NSUCCESS": int(np.count_nonzero(success)),
        "NFAILED": int(npixels - np.count_nonzero(success)),
        "SUCCESS_FRACTION": float(np.mean(success)),
        "N_INPUT_BPM": int(np.count_nonzero(quality & BRANDT_DQ["INPUT_BPM"])),
        "N_TOO_FEW_SAMPLES": int(np.count_nonzero(quality & BRANDT_DQ["TOO_FEW_SAMPLES"])),
        "N_FIT_FAILED": int(np.count_nonzero(quality & BRANDT_DQ["FIT_FAILED"])),
        "N_NONMONOTONIC": int(np.count_nonzero(quality & BRANDT_DQ["NONMONOTONIC"])),
        "N_HIGH_CHISQ": int(np.count_nonzero(quality & BRANDT_DQ["HIGH_CHISQ"])),
        "N_IDENTITY_FALLBACK": int(np.count_nonzero(quality & BRANDT_DQ["IDENTITY_FALLBACK"])),
        "MEDIAN_REDCHISQ": float(np.nanmedian(outputs["REDCHISQ"])),
        "MEDIAN_NVALID": float(np.nanmedian(outputs["NVALID"])),
        "MAX_NITER": int(np.max(outputs["NITER"])),
    }


def analyze_monotonic_fits(outputs, brandt_dq):
    """
    Analyze monotonicity diagnostics for Brandt polynomial fits.

    Parameters
    ----------
    outputs : dict
        Output dictionary returned by fit_brandt_detector().

    brandt_dq : dict
        Dictionary containing the Brandt DQ bit definitions.
    """

    import numpy as np

    quality = outputs["QUALITY"]
    min_derivative = outputs["MIN_DERIVATIVE"]
    success = outputs["SUCCESS"].astype(bool)

    nonmonotonic = (
        quality & brandt_dq["NONMONOTONIC"]
    ) != 0

    finite = np.isfinite(min_derivative)

    print("\n" + "=" * 65)
    print("Monotonicity analysis")
    print("-" * 65)

    print(f"Total pixels                : {min_derivative.size}")
    print(f"Successful fits             : {np.count_nonzero(success)}")
    print(f"Nonmonotonic fits           : {np.count_nonzero(nonmonotonic)}")
    print(f"Finite minimum derivatives  : {np.count_nonzero(finite)}")

    print("\nCumulative derivative thresholds")
    print("-" * 65)

    thresholds = [
        0.0,
        -1e-8,
        -1e-7,
        -1e-6,
        -1e-5,
        -1e-4,
        -1e-3,
        -1e-2,
        -1e-1,
        -1.0,
    ]

    for t in thresholds:

        mask = (
            finite &
            (min_derivative < t)
        )

        n = np.count_nonzero(mask)

        print(
            f"MIN_DERIVATIVE < {t:10.1e} : "
            f"{n:10d} ({100*n/min_derivative.size:6.2f}%)"
        )

    print("\nExclusive severity bins")
    print("-" * 65)

    bins = [
        (-np.inf, -1.0, "< -1"),
        (-1.0, -1e-1, "-1 to -0.1"),
        (-1e-1, -1e-2, "-0.1 to -0.01"),
        (-1e-2, -1e-3, "-0.01 to -0.001"),
        (-1e-3, -1e-4, "-0.001 to -1e-4"),
        (-1e-4, -1e-5, "-1e-4 to -1e-5"),
        (-1e-5, -1e-6, "-1e-5 to -1e-6"),
        (-1e-6, 0.0, "-1e-6 to 0"),
    ]

    n_nonmono = np.count_nonzero(nonmonotonic)

    for lo, hi, label in bins:

        mask = (
            finite &
            (min_derivative >= lo) &
            (min_derivative < hi)
        )

        n = np.count_nonzero(mask)

        frac = (
            100 * n / n_nonmono
            if n_nonmono > 0 else 0
        )

        print(
            f"{label:20s}: "
            f"{n:10d} ({frac:6.2f}% of nonmonotonic)"
        )

    bad = min_derivative[
        nonmonotonic & finite
    ]

    if bad.size > 0:

        print("\nMinimum derivative percentiles")
        print("-" * 65)

        for p in [
            0,
            1,
            5,
            16,
            25,
            50,
            75,
            84,
            95,
            99,
            100,
        ]:

            print(
                f"{p:3d}th percentile : "
                f"{np.percentile(bad,p):12.5g}"
            )

    print("\nSuccessful fits")

    good = min_derivative[
        success & finite
    ]

    if good.size > 0:

        print(f"Median minimum derivative : {np.median(good):.6g}")
        print(f"Minimum derivative        : {np.min(good):.6g}")
        print(f"Maximum derivative        : {np.max(good):.6g}")

    print("=" * 65)


def print_brandt_block_summary(outputs):
    """Print a compact block-fitting summary."""

    s = summarize_brandt_block(outputs)
    print("\n" + "=" * 58)
    print("Brandt block fitting summary")
    print("-" * 58)
    print(f"Polynomial terms         : {outputs['POLYNOMIAL_TERMS']}")
    print(f"Total pixels             : {s['NPIXELS']}")
    print(f"Successful fits          : {s['NSUCCESS']} ({100*s['SUCCESS_FRACTION']:.2f}%)")
    print(f"Failed/skipped pixels    : {s['NFAILED']}")
    print(f"Input BPM pixels         : {s['N_INPUT_BPM']}")
    print(f"Too few samples          : {s['N_TOO_FEW_SAMPLES']}")
    print(f"Numerical fit failures   : {s['N_FIT_FAILED']}")
    print(f"Nonmonotonic fits        : {s['N_NONMONOTONIC']}")
    print(f"High chi-square fits     : {s['N_HIGH_CHISQ']}")
    print(f"Identity fallbacks       : {s['N_IDENTITY_FALLBACK']}")
    print(f"Median valid samples     : {s['MEDIAN_NVALID']:.1f}")
    print(f"Median reduced chi-square: {s['MEDIAN_REDCHISQ']:.4g}")
    print(f"Maximum iterations       : {s['MAX_NITER']}")
    print("=" * 58)


def _read_fits_array(
    filename: str | Path,
    *,
    extension: int | str = 0,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """
    Read an image from a FITS file.

    Parameters
    ----------
    filename : str or Path
        FITS filename.
    extension : int or str
        HDU number or extension name.
    dtype : numpy dtype, optional
        Cast output array to this dtype.

    Returns
    -------
    ndarray
        Image array.
    """

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"FITS file does not exist: {filename}")

    with fits.open(filename, memmap=True) as hdul:

        try:
            hdu = hdul[extension]
        except (KeyError, IndexError):
            available = [h.name for h in hdul]
            raise ValueError(
                f"Extension {extension!r} not found in {filename}.\n"
                f"Available extensions: {available}"
            )

        if hdu.data is None:
            raise ValueError(
                f"Extension {extension!r} in {filename} contains no image data."
            )

        data = np.asarray(hdu.data)

    if dtype is not None:
        data = data.astype(dtype, copy=False)

    return data


def _load_detector_map(
    value: np.ndarray | str | Path | float | int,
    detector_shape: tuple[int, int],
    *,
    name: str,
    extension: int | str = 0,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Convert a scalar, array, or FITS image into a detector-sized map."""
    if isinstance(value, (str, Path)):
        array = _read_fits_array(value, extension=extension, dtype=dtype)
    else:
        array = np.asarray(value, dtype=dtype)

    if array.ndim == 0:
        return np.full(detector_shape, array.item(), dtype=dtype)

    if array.shape != detector_shape:
        raise ValueError(
            f"{name} has shape {array.shape}; expected scalar or "
            f"detector map with shape {detector_shape}."
        )

    return np.asarray(array, dtype=dtype)


def _normalize_ramp_file_list(
    ramp_files: str | Path | Sequence[str | Path],
) -> list[Path]:
    """Return a validated list of ramp FITS files."""
    if isinstance(ramp_files, (str, Path)):
        path = Path(ramp_files)

        if path.is_dir():
            files = sorted(
                p for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in {".fits", ".fit", ".fts"}
            )
        else:
            files = [path]
    else:
        files = [Path(item) for item in ramp_files]

    if not files:
        raise ValueError("No ramp FITS files were supplied or found.")

    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following ramp files do not exist:\n" + "\n".join(missing)
        )

    return files


def _inspect_ramp_input(
    ramps: np.ndarray | str | Path | Sequence[str | Path],
    *,
    ramp_extension: int | str = 0,
) -> tuple[int, int, int, int]:
    """Return ``(nramps, nreads, nrows, nx)`` without loading all FITS data."""
    if isinstance(ramps, np.ndarray):
        if ramps.ndim != 4:
            raise ValueError(
                "Array input must have shape "
                "(n_ramps, n_reads, nrows, nx)."
            )
        return tuple(int(v) for v in ramps.shape)

    files = _normalize_ramp_file_list(ramps)

    first = _read_fits_array(files[0], extension=ramp_extension)

    if first.ndim != 3:
        raise ValueError(
            f"Ramp file {files[0]} has shape {first.shape}; expected "
            "(n_reads, nrows, nx)."
        )

    nreads, nrows, nx = first.shape

    for filename in files[1:]:
        with fits.open(filename, memmap=True) as hdul:
            shape = hdul[ramp_extension].shape

        if tuple(shape) != (nreads, nrows, nx):
            raise ValueError(
                f"Ramp file {filename} has shape {shape}; expected "
                f"{(nreads, nrows, nx)}."
            )

    return len(files), nreads, nrows, nx


def _read_ramp_block(
    ramps: np.ndarray | str | Path | Sequence[str | Path],
    row_start: int,
    row_stop: int,
    *,
    ramp_extension: int | str = 0,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Read one detector-row block with shape ``(nramps,nreads,nblock,nx)``."""
    if isinstance(ramps, np.ndarray):
        return np.asarray(
            ramps[:, :, row_start:row_stop, :],
            dtype=dtype,
        )

    files = _normalize_ramp_file_list(ramps)
    blocks = []

    for filename in files:
        with fits.open(filename, memmap=True) as hdul:
            data = hdul[ramp_extension].data

            if data is None:
                raise ValueError(
                    f"No image data in extension {ramp_extension!r} "
                    f"of {filename}."
                )

            blocks.append(
                np.asarray(data[:, row_start:row_stop, :], dtype=dtype)
            )

    return np.stack(blocks, axis=0)


def _prepare_read_times(
    read_times: None | np.ndarray | str | Path,
    nramps: int,
    nreads: int,
) -> np.ndarray | None:
    """Validate read times or load them from ``.npy``, text, or FITS."""
    if read_times is None:
        return None

    if isinstance(read_times, (str, Path)):
        filename = Path(read_times)
        suffix = filename.suffix.lower()

        if suffix == ".npy":
            value = np.load(filename)
        elif suffix in {".fits", ".fit", ".fts"}:
            value = fits.getdata(filename)
        else:
            value = np.loadtxt(filename)
    else:
        value = np.asarray(read_times)

    value = np.asarray(value, dtype=np.float64)

    if value.shape not in {(nreads,), (nramps, nreads)}:
        raise ValueError(
            f"read_times has shape {value.shape}; expected "
            f"{(nreads,)} or {(nramps, nreads)}."
        )

    return value


def allocate_detector_outputs(
    nramps: int,
    nrows: int,
    nx: int,
    polynomial_terms: Sequence[int],
) -> dict[str, Any]:
    """Allocate full-detector outputs compatible with ``fit_brandt_block``."""
    terms = validate_polynomial_terms(polynomial_terms)
    max_power = max(terms) if terms else 1

    outputs: dict[str, Any] = {
        "COEFF": np.zeros((max_power + 1, nrows, nx), dtype=np.float64),
        "QUALITY": np.zeros((nrows, nx), dtype=np.uint32),
        "CHISQ": np.full((nrows, nx), np.nan, dtype=np.float64),
        "REDCHISQ": np.full((nrows, nx), np.nan, dtype=np.float64),
        "NVALID": np.zeros((nrows, nx), dtype=np.uint32),
        "DOF": np.zeros((nrows, nx), dtype=np.int32),
        "PEDESTAL": np.full((nramps, nrows, nx), np.nan, dtype=np.float64),
        "SLOPE": np.full((nramps, nrows, nx), np.nan, dtype=np.float64),
        "INTERCEPT": np.full((nramps, nrows, nx), np.nan, dtype=np.float64),
        "RESIDUAL_RMS": np.full((nrows, nx), np.nan, dtype=np.float64),
        "MIN_DERIVATIVE": np.full((nrows, nx), np.nan, dtype=np.float64),
        "NITER": np.zeros((nrows, nx), dtype=np.uint16),
        "SUCCESS": np.zeros((nrows, nx), dtype=bool),
        "POLYNOMIAL_TERMS": tuple(terms),
    }

    identity = identity_coefficients(terms)
    outputs["COEFF"][:] = identity[:, None, None]

    return outputs


def store_detector_block(
    detector_outputs: dict[str, Any],
    block_outputs: dict[str, Any],
    row_start: int,
    row_stop: int,
) -> None:
    """Store one fitted row block into the full-detector output arrays."""
    block_keys = (
        "COEFF",
        "QUALITY",
        "CHISQ",
        "REDCHISQ",
        "NVALID",
        "DOF",
        "PEDESTAL",
        "SLOPE",
        "INTERCEPT",
        "RESIDUAL_RMS",
        "MIN_DERIVATIVE",
        "NITER",
        "SUCCESS",
    )

    for key in block_keys:
        if key not in block_outputs:
            raise KeyError(f"Block outputs do not contain required key {key!r}.")

        if key in {"COEFF", "PEDESTAL", "SLOPE", "INTERCEPT"}:
            detector_outputs[key][:, row_start:row_stop, :] = block_outputs[key]
        else:
            detector_outputs[key][row_start:row_stop, :] = block_outputs[key]


def _image_hdu(
    data: np.ndarray,
    name: str,
    *,
    dtype: np.dtype | type | None = None,
) -> fits.ImageHDU:
    """Create a named FITS image extension."""
    array = np.asarray(data)

    if dtype is not None:
        array = array.astype(dtype, copy=False)

    return fits.ImageHDU(data=array, name=name)


def write_linearity_reference_fits(
    filename: str | Path,
    outputs: dict[str, Any],
    saturation_map: np.ndarray,
    bad_pixel_map: np.ndarray,
    *,
    gain_map: np.ndarray | None = None,
    readnoise_map: np.ndarray | None = None,
    primary_header: fits.Header | None = None,
    overwrite: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write all calibration products to a multi-extension FITS file."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    header = fits.Header() if primary_header is None else primary_header.copy()

    terms = tuple(int(v) for v in outputs["POLYNOMIAL_TERMS"])
    summary = summarize_brandt_block(outputs)

    header["REFTYPE"] = ("LINEARITY", "Reference-file type")
    header["NCOEFF"] = (int(outputs["COEFF"].shape[0]), "Coefficient planes")
    header["PTERMS"] = (
        ",".join(str(v) for v in terms),
        "Fitted nonlinear powers",
    )
    header["NPIXELS"] = (int(summary["NPIXELS"]), "Detector pixels")
    header["NSUCC"] = (int(summary["NSUCCESS"]), "Successful pixel fits")
    header["SUCCFRAC"] = (
        float(summary["SUCCESS_FRACTION"]),
        "Successful-fit fraction",
    )
    header["BUNIT"] = ("DN", "Input detector signal unit")
    header["COEFORD"] = ("ASCENDING", "Coefficient power ordering")

    for name, value in (metadata or {}).items():
        keyword = str(name).upper()[:8]
        try:
            header[keyword] = value
        except Exception:
            header.add_history(f"{name}={value}")

    hdus: list[fits.hdu.base.ExtensionHDU] = [fits.PrimaryHDU(header=header)]

    hdus.extend(
        [
            _image_hdu(outputs["COEFF"], "COEFF", dtype=np.float64),
            _image_hdu(saturation_map, "SATURATION", dtype=np.float32),
            _image_hdu(bad_pixel_map.astype(np.uint8), "BPM", dtype=np.uint8),
            _image_hdu(outputs["QUALITY"], "QUALITY", dtype=np.uint32),
            _image_hdu(outputs["SUCCESS"].astype(np.uint8), "SUCCESS", dtype=np.uint8),
            _image_hdu(outputs["CHISQ"], "CHISQ", dtype=np.float32),
            _image_hdu(outputs["REDCHISQ"], "REDCHISQ", dtype=np.float32),
            _image_hdu(outputs["NVALID"], "NVALID", dtype=np.uint32),
            _image_hdu(outputs["DOF"], "DOF", dtype=np.int32),
            _image_hdu(outputs["PEDESTAL"], "PEDESTAL", dtype=np.float32),
            _image_hdu(outputs["SLOPE"], "SLOPE", dtype=np.float32),
            _image_hdu(outputs["INTERCEPT"], "INTERCEPT", dtype=np.float32),
            _image_hdu(outputs["RESIDUAL_RMS"], "RESIDRMS", dtype=np.float32),
            _image_hdu(outputs["MIN_DERIVATIVE"], "MINDERIV", dtype=np.float32),
            _image_hdu(outputs["NITER"], "NITER", dtype=np.uint16),
        ]
    )

    if gain_map is not None:
        hdus.append(_image_hdu(gain_map, "GAIN", dtype=np.float32))

    if readnoise_map is not None:
        hdus.append(_image_hdu(readnoise_map, "READNOISE", dtype=np.float32))

    dq_names = sorted(BRANDT_DQ.items(), key=lambda item: int(item[1]))
    columns = [
        fits.Column(
            name="NAME",
            format="32A",
            array=np.asarray([name for name, _ in dq_names], dtype="S32"),
        ),
        fits.Column(
            name="VALUE",
            format="J",
            array=np.asarray([int(value) for _, value in dq_names], dtype=np.uint32),
        ),
        fits.Column(
            name="BIT",
            format="J",
            array=np.asarray(
                [int(np.log2(int(value))) for _, value in dq_names],
                dtype=np.int32,
            ),
        ),
    ]
    hdus.append(fits.BinTableHDU.from_columns(columns, name="DQDEF"))

    fits.HDUList(hdus).writeto(filename, overwrite=overwrite, checksum=True)

    return filename



def run_linearity_calibration(
    ramps: np.ndarray,
    saturation_map: np.ndarray,
    *,
    bad_pixel_map: np.ndarray | None = None,
    gain: float | np.ndarray = 1.0,
    readnoise: float | np.ndarray = 0.0,
    polynomial_terms: Sequence[int] = (2, 3,4),
    output_fits: str | Path = "linearity_reference.fits",
    block_rows: int = 32,
    read_times: np.ndarray | None = None,
    skip_initial_reads: int = 2,
    n_pedestal_reads: int = 2,
    saturation_fraction: float = 0.95,
    minimum_reads_per_ramp: int = 4,
    minimum_total_samples: int | None = None,
    minimum_valid_ramps: int = 2,
    require_positive_signal: bool = True,
    sigma_clip: float | None = 5.0,
    maximum_iterations: int = 3,
    maximum_reduced_chisq: float | None = None,
    check_monotonic: bool = True,
    derivative_tolerance: float = -1.0e-6,
    n_monotonic_test_points: int = 256,
    rcond: float | None = None,
    maximum_correction_fraction: float | None = None,
    correction_floor_dn: float = 100.0,
    maximum_derivative: float | None = None,
    progress: bool = True,
    overwrite: bool = True,
    primary_header: fits.Header | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run detector linearity calibration using NumPy arrays.

    Parameters
    ----------
    ramps : ndarray, shape (nramps, nreads, nrows, nx)
        Input detector ramps in DN.

    saturation_map : ndarray, shape (nrows, nx)
        Per-pixel saturation level in DN.

    bad_pixel_map : ndarray, optional
        Input BPM. Nonzero values indicate bad pixels.

    gain : float or ndarray
        Detector gain, scalar or 2D map.

    readnoise : float or ndarray
        Detector read noise, scalar or 2D map.

    polynomial_terms : sequence of int
        Nonlinear polynomial powers to fit.

    output_fits : str or Path
        Output calibration FITS filename.

    block_rows : int
        Number of detector rows processed per block.

    read_times : ndarray, optional
        Read times with shape (nreads,) or (nramps, nreads).

    Returns
    -------
    dict
        Calibration outputs, detector maps, summary, output filename,
        and execution time.
    """

    start_time = time.perf_counter()

    # ------------------------------------------------------------
    # 1. Validate input ramp array
    # ------------------------------------------------------------
    if not isinstance(ramps, np.ndarray):
        raise TypeError(
            "ramps must be a NumPy array with shape "
            "(nramps, nreads, nrows, nx)"
        )

    if ramps.ndim != 4:
        raise ValueError(
            "ramps must have shape "
            "(nramps, nreads, nrows, nx)"
        )

    nramps, nreads, nrows, nx = ramps.shape
    detector_shape = (nrows, nx)

    if nramps < 1 or nreads < 2:
        raise ValueError(
            "At least one ramp and two reads are required."
        )

    terms = validate_polynomial_terms(polynomial_terms)
    if not np.isfinite(correction_floor_dn) or correction_floor_dn <= 0:
        raise ValueError("correction_floor_dn must be finite and > 0")
    if maximum_correction_fraction is not None and (not np.isfinite(maximum_correction_fraction) or maximum_correction_fraction < 0):
        raise ValueError("maximum_correction_fraction must be nonnegative")
    if maximum_derivative is not None and (not np.isfinite(maximum_derivative) or maximum_derivative <= 0):
        raise ValueError("maximum_derivative must be positive")

    block_rows = int(block_rows)

    if block_rows < 1:
        raise ValueError("block_rows must be at least 1.")

    # ------------------------------------------------------------
    # 2. Validate saturation map
    # ------------------------------------------------------------
    if not isinstance(saturation_map, np.ndarray):
        raise TypeError(
            "saturation_map must be a NumPy array."
        )

    saturation = np.asarray(
        saturation_map,
        dtype=np.float64,
    )

    if saturation.shape != detector_shape:
        raise ValueError(
            f"saturation_map shape {saturation.shape} "
            f"does not match detector shape {detector_shape}"
        )

    # ------------------------------------------------------------
    # 3. Validate bad pixel map
    # ------------------------------------------------------------
    if bad_pixel_map is None:
        bpm = np.zeros(detector_shape, dtype=bool)
    else:
        if not isinstance(bad_pixel_map, np.ndarray):
            raise TypeError(
                "bad_pixel_map must be a NumPy array."
            )

        if bad_pixel_map.shape != detector_shape:
            raise ValueError(
                f"bad_pixel_map shape {bad_pixel_map.shape} "
                f"does not match detector shape {detector_shape}"
            )

        bpm = np.asarray(bad_pixel_map) != 0

    # ------------------------------------------------------------
    # 4. Prepare gain and readnoise maps
    # ------------------------------------------------------------
    def prepare_detector_map(value, name):
        if np.isscalar(value):
            return np.full(
                detector_shape,
                float(value),
                dtype=np.float64,
            )

        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"{name} must be a scalar or NumPy array."
            )

        arr = np.asarray(value, dtype=np.float64)

        if arr.shape != detector_shape:
            raise ValueError(
                f"{name} shape {arr.shape} "
                f"does not match detector shape {detector_shape}"
            )

        return arr

    gain_map = prepare_detector_map(gain, "gain")

    readnoise_map = prepare_detector_map(
        readnoise,
        "readnoise",
    )

    # ------------------------------------------------------------
    # 5. Prepare read times
    # ------------------------------------------------------------
    if read_times is None:
        times = _prepare_read_times(
            None,
            nramps,
            nreads,
        )
    else:
        if not isinstance(read_times, np.ndarray):
            raise TypeError(
                "read_times must be a NumPy array."
            )

        if read_times.shape not in {
            (nreads,),
            (nramps, nreads),
        }:
            raise ValueError(
                "read_times must have shape "
                "(nreads,) or (nramps, nreads)"
            )

        times = _prepare_read_times(
            read_times,
            nramps,
            nreads,
        )

    # ------------------------------------------------------------
    # 6. Allocate output arrays
    # ------------------------------------------------------------
    outputs = allocate_detector_outputs(
        nramps,
        nrows,
        nx,
        terms,
    )

    nblocks = (nrows + block_rows - 1) // block_rows

    # ------------------------------------------------------------
    # 7. Print input summary
    # ------------------------------------------------------------
    if progress:
        print("\n" + "=" * 68)
        print("Detector linearity calibration")
        print("-" * 68)
        print(f"Ramps                 : {nramps}")
        print(f"Reads per ramp        : {nreads}")
        print(f"Detector shape        : {detector_shape}")
        print(f"Polynomial terms      : {terms}")
        print(f"Rows per block        : {block_rows}")
        print(f"Number of blocks      : {nblocks}")
        print(f"Bad pixels            : {np.count_nonzero(bpm)}")
        print(f"Output FITS           : {output_fits}")
        print("=" * 68)

    # ------------------------------------------------------------
    # 8. Process detector in row blocks
    # ------------------------------------------------------------
    for block_index, row_start in enumerate(
        range(0, nrows, block_rows),
        start=1,
    ):

        row_stop = min(
            row_start + block_rows,
            nrows,
        )

        if progress:
            print(
                f"\nBlock {block_index}/{nblocks}: "
                f"rows {row_start}:{row_stop}"
            )

        # Extract block directly from input NumPy array.
        # Shape: (nramps, nreads, block_rows, nx)
        ramp_block = np.asarray(
            ramps[:, :, row_start:row_stop, :],
            dtype=np.float32,
        )

        sat_block = saturation[row_start:row_stop, :]
        bpm_block = bpm[row_start:row_stop, :]

        gain_block = gain_map[row_start:row_stop, :]
        readnoise_block = readnoise_map[row_start:row_stop, :]

        # --------------------------------------------------------
        # Fit nonlinear coefficients
        # --------------------------------------------------------
        block_outputs = fit_brandt_block(
            ramp_block,
            sat_block,
            bad_pixel_map=bpm_block,
            gain=gain_block,
            readnoise=readnoise_block,
            polynomial_terms=terms,
            read_times=times,
            skip_initial_reads=skip_initial_reads,
            n_pedestal_reads=n_pedestal_reads,
            saturation_fraction=saturation_fraction,
            minimum_reads_per_ramp=minimum_reads_per_ramp,
            minimum_total_samples=minimum_total_samples,
            minimum_valid_ramps=minimum_valid_ramps,
            require_positive_signal=require_positive_signal,
            sigma_clip=sigma_clip,
            maximum_iterations=maximum_iterations,
            maximum_reduced_chisq=maximum_reduced_chisq,
            check_monotonic=check_monotonic,
            derivative_tolerance=derivative_tolerance,
            n_monotonic_test_points=n_monotonic_test_points,
            rcond=rcond,
            progress=False,
            print_summary=False,
        )

        # --------------------------------------------------------
        # Store block results in full-detector output arrays
        # --------------------------------------------------------
        store_detector_block(
            outputs,
            block_outputs,
            row_start,
            row_stop,
        )

        if progress:
            block_summary = summarize_brandt_block(
                block_outputs
            )

            print(
                f"  successful fits: "
                f"{block_summary['NSUCCESS']}/"
                f"{block_summary['NPIXELS']} "
                f"({100.0 * block_summary['SUCCESS_FRACTION']:.2f}%)"
            )

        del ramp_block
        del block_outputs

    # ------------------------------------------------------------
    # 9. Final detector summary
    # ------------------------------------------------------------
    summary = summarize_brandt_block(outputs)

    # ------------------------------------------------------------
    # 10. Write calibration reference FITS
    # ------------------------------------------------------------
    output_path = write_linearity_reference_fits(
        output_fits,
        outputs,
        saturation,
        bpm,
        gain_map=gain_map,
        readnoise_map=readnoise_map,
        primary_header=primary_header,
        overwrite=overwrite,
        metadata=metadata,
    )

    elapsed = time.perf_counter() - start_time

    # ------------------------------------------------------------
    # 11. Diagnostics
    # ------------------------------------------------------------
    if progress:
        analyze_monotonic_fits(outputs, BRANDT_DQ)

        print_brandt_block_summary(outputs)

        print(f"\nReference FITS written : {output_path}")
        print(f"Elapsed time           : {elapsed:.3f} s")

    # ------------------------------------------------------------
    # 12. Return outputs
    # ------------------------------------------------------------
    return {
        "OUTPUTS": outputs,
        "SATURATION_MAP": saturation,
        "BAD_PIXEL_MAP": bpm,
        "GAIN_MAP": gain_map,
        "READNOISE_MAP": readnoise_map,
        "SUMMARY": summary,
        "OUTPUT_FITS": output_path,
        "ELAPSED_SECONDS": elapsed,
    }




def apply_coefficients_to_ramp_safe(ramp, coefficients, *, pedestal=None, success=None,
                                    saturation_map=None, saturation_fraction=0.95):
    """Apply ascending-power coefficients to absolute-DN reads, preserving invalid reads.

    ramp: (nreads, ny, nx); pedestal: (ny,nx) absolute DN, required.
    Coefficients were calibrated on pedestal-subtracted signal.
    Pixels without successful fits remain unchanged. Saturated reads remain raw.
    """
    raw = np.asarray(ramp)
    coeff = np.asarray(coefficients)
    if raw.ndim != 3 or coeff.ndim != 3 or coeff.shape[1:] != raw.shape[1:]:
        raise ValueError("Expected ramp (N,H,W) and coefficients (K,H,W)")
    if pedestal is None:
        raise ValueError("pedestal is required: coefficients fit pedestal-subtracted signal")
    ped = np.asarray(pedestal, dtype=np.float64)
    if ped.shape != raw.shape[1:]:
        raise ValueError("pedestal must have shape (H,W)")
    if success is None:
        good = np.ones(raw.shape[1:], dtype=bool)
    else:
        good = np.asarray(success, dtype=bool)
        if good.shape != raw.shape[1:]:
            raise ValueError("success shape mismatch")
    signal = raw.astype(np.float64) - ped[None]
    with np.errstate(over="ignore", invalid="ignore"):
        mapped = apply_coefficients(signal, coeff) + ped[None]
    valid = np.isfinite(raw) & np.isfinite(mapped) & good[None] & np.isfinite(ped)[None]
    if saturation_map is not None:
        sat = np.asarray(saturation_map, dtype=np.float64)
        if sat.shape != raw.shape[1:]:
            raise ValueError("saturation_map shape mismatch")
        threshold = ped + saturation_fraction * (sat - ped)
        crossed = ~np.isfinite(raw) | (raw >= threshold[None])
        valid &= ~np.maximum.accumulate(crossed, axis=0)
    output = raw.copy()
    output[valid] = mapped[valid].astype(output.dtype)
    return output


###########################ploting##########################################
DQ = {
    'INPUT_BPM':1<<0,'NONFINITE_INPUT':1<<1,'INVALID_SATURATION':1<<2,
    'TOO_FEW_SAMPLES':1<<3,'FIT_FAILED':1<<4,'NONFINITE_COEFFICIENT':1<<5,
    'NONMONOTONIC':1<<6,'HIGH_CHISQ':1<<7,'IDENTITY_FALLBACK':1<<8,
    'INVALID_GAIN':1<<9,'INVALID_READNOISE':1<<10,
    'EXCESSIVE_CORRECTION':1<<11,'EXCESSIVE_DERIVATIVE':1<<12,
}

def _image(ax, data, title, label, *, cmap='viridis', percentiles=(2,98), mask=None):
    arr=np.asarray(data,dtype=float).copy()
    if mask is not None:
        arr[~mask]=np.nan
    finite=arr[np.isfinite(arr)]
    if finite.size:
        lo,hi=np.percentile(finite,percentiles)
        if lo==hi:
            lo-=max(1,abs(lo))*.01
            hi+=max(1,abs(hi))*.01
    else:
        lo,hi=0,1
    im=ax.imshow(np.ma.masked_invalid(arr),origin='lower',cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
    ax.set_title(title,fontsize=11)
    ax.set_xlabel('Column'); ax.set_ylabel('Row')
    ax.figure.colorbar(im,ax=ax,shrink=.79,label=label)
    return im

def _save(fig, path):
    if path is not None:
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=180,bbox_inches='tight')


def plot_lin_ref_dia(
    result,
    *,
    signal_dn=2500.0,
    output_dir="linearity_plots",
    save_prefix="linearity_diagnostic",
    show=False,
):
    """
    Create and save two 2x2 diagnostic figures for linearity calibration.

    Figure 1: Linearity coefficient diagnostics
        1. Successful-fit map
        2. Quadratic coefficient c2
        3. Cubic coefficient c3
        4. Fractional correction at signal_dn

    Figure 2: Fit reliability diagnostics
        1. Reduced chi-square
        2. Residual RMS
        3. Minimum polynomial derivative
        4. Identity-fallback map

    Parameters
    ----------
    result : dict
        Dictionary returned by run_linearity_calibration(), or
        the OUTPUTS dictionary returned by fit_brandt_block().

    signal_dn : float, optional
        Pedestal-subtracted signal in DN at which the fractional
        linearity correction is evaluated. Default is 2500 DN.

    output_dir : str or Path, optional
        Directory where diagnostic figures will be saved.
        Created automatically if it does not exist.

    save_prefix : str, optional
        Prefix for output PNG filenames.

    show : bool, optional
        Whether to display figures interactively.
        Default is False.

    Returns
    -------
    fig1, fig2 : matplotlib.figure.Figure
        Coefficient and fit-reliability diagnostic figures.
    """

    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------

    out = result.get("OUTPUTS", result)

    coeff = np.asarray(out["COEFF"], dtype=float)
    success = np.asarray(out["SUCCESS"], dtype=bool)
    quality = np.asarray(out["QUALITY"], dtype=np.uint32)

    if coeff.ndim != 3:
        raise ValueError(
            "COEFF must have shape (n_coefficients, nrows, nx)."
        )

    if success.shape != coeff.shape[1:]:
        raise ValueError(
            "SUCCESS shape must match coefficient detector dimensions."
        )

    if quality.shape != success.shape:
        raise ValueError(
            "QUALITY shape must match SUCCESS shape."
        )

    if not np.isfinite(signal_dn) or signal_dn <= 0:
        raise ValueError(
            "signal_dn must be finite and positive."
        )

    # ------------------------------------------------------------
    # Prepare output directory
    # ------------------------------------------------------------

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_prefix is not None:
        save_prefix = Path(str(save_prefix)).stem

    # ------------------------------------------------------------
    # Extract polynomial coefficients
    # ------------------------------------------------------------

    c2 = (
        coeff[2]
        if coeff.shape[0] > 2
        else np.zeros_like(success, dtype=float)
    )

    c3 = (
        coeff[3]
        if coeff.shape[0] > 3
        else np.zeros_like(success, dtype=float)
    )

    # ------------------------------------------------------------
    # Evaluate polynomial at specified signal
    #
    # L(M) = c0 + c1*M + c2*M^2 + ...
    # ------------------------------------------------------------

    corrected = np.zeros(success.shape, dtype=float)

    for k in range(coeff.shape[0] - 1, -1, -1):
        corrected = corrected * signal_dn + coeff[k]

    fractional = (
        100.0 * (corrected - signal_dn) / signal_dn
    )

    # ------------------------------------------------------------
    # FIGURE 1: Coefficient diagnostics
    # ------------------------------------------------------------

    fig1, ax = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        constrained_layout=True,
    )

    _image(
        ax[0, 0],
        success.astype(float),
        "Successful fit (1=yes)",
        "Success",
        cmap="Greys",
        percentiles=(0, 100),
    )

    _image(
        ax[0, 1],
        c2,
        "Quadratic coefficient c₂",
        "DN⁻¹",
        mask=success,
        cmap="coolwarm",
    )

    _image(
        ax[1, 0],
        c3,
        "Cubic coefficient c₃",
        "DN⁻²",
        mask=success,
        cmap="coolwarm",
    )

    _image(
        ax[1, 1],
        fractional,
        f"Correction at {signal_dn:g} DN",
        "(corrected − raw) / raw [%]",
        mask=success,
        cmap="coolwarm",
    )

    fig1.suptitle(
        "Linearity coefficient diagnostics | "
        f"successful {success.mean():.1%}",
        fontsize=14,
    )

    # ------------------------------------------------------------
    # FIGURE 2: Fit reliability diagnostics
    # ------------------------------------------------------------

    fig2, ax = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        constrained_layout=True,
    )

    _image(
        ax[0, 0],
        out["REDCHISQ"],
        "Reduced χ² (successful pixels)",
        "χ² / dof",
        mask=success,
    )

    _image(
        ax[0, 1],
        out["RESIDUAL_RMS"],
        "Residual RMS (successful pixels)",
        "DN",
        mask=success,
    )

    _image(
        ax[1, 0],
        out["MIN_DERIVATIVE"],
        "Minimum correction derivative",
        "dL/dM",
        mask=success,
    )

    fallback = (
        (quality & DQ["IDENTITY_FALLBACK"]) != 0
    )

    _image(
        ax[1, 1],
        fallback.astype(float),
        "Identity fallback (1=yes)",
        "Fallback",
        cmap="Greys",
        percentiles=(0, 100),
    )

    fig2.suptitle(
        "Linearity fit reliability | "
        f"fallback {fallback.mean():.1%}",
        fontsize=14,
    )

    # ------------------------------------------------------------
    # Save figures
    # ------------------------------------------------------------

    if save_prefix is not None:

        coeff_path = (
            output_dir / f"{save_prefix}_coefficients.png"
        )

        reliability_path = (
            output_dir / f"{save_prefix}_reliability.png"
        )

        fig1.savefig(
            coeff_path,
            dpi=200,
            bbox_inches="tight",
        )

        fig2.savefig(
            reliability_path,
            dpi=200,
            bbox_inches="tight",
        )

        print("\nLinearity diagnostic plots saved:")
        print(f"  Coefficients : {coeff_path}")
        print(f"  Reliability  : {reliability_path}")

    # ------------------------------------------------------------
    # Display or close figures
    # ------------------------------------------------------------

    if show:
        plt.show()

    return fig1, fig2


def plot_linearity_ramp_validation(
    ramps,
    result,
    *,
    pixels,
    read_times=None,
    saturation_fraction=0.95,
    skip_initial_reads=2,
    n_pedestal_reads=2,
    output_dir="linearity_plots",
    save_prefix="linearity_diagnostic",
    show=False,
):
    """
    Create a 2x2 diagnostic plot for one detector pixel.

    Panels
    ------
    1. Raw and linearity-corrected ramps
    2. Raw and corrected linear-fit residuals
    3. Residual distributions and RMS comparison
    4. Fractional linearity correction distribution

    Parameters
    ----------
    ramps : ndarray
        Raw ramp data with shape (nramps, nreads, H, W).

    result : dict
        Output returned by run_linearity_calibration().

    pixels : tuple
        Pixel coordinates (row, column).

    read_times : ndarray, optional
        Read times with shape (nreads,) or (nramps, nreads).
        Defaults to read indices.

    saturation_fraction : float
        Fraction of the saturation range used for validation.

    skip_initial_reads : int
        Number of initial reads excluded from the fit.

    n_pedestal_reads : int
        Number of initial reads used for pedestal estimation.

    output_dir : str or Path
        Directory where diagnostic plots will be saved.

    save_prefix : str
        Prefix for the output filename.

    show : bool
        Display the figure interactively.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------

    out = result.get("OUTPUTS", result)

    raw = np.asarray(ramps)

    if raw.ndim != 4:
        raise ValueError(
            "ramps must have shape (nramps, nreads, H, W)"
        )

    nr, nt, H, W = raw.shape

    r, c = map(int, pixels)

    if not (0 <= r < H and 0 <= c < W):
        raise ValueError("Pixel coordinates are out of bounds.")

    if not (0 < saturation_fraction <= 1):
        raise ValueError(
            "saturation_fraction must be between 0 and 1."
        )

    if not (1 <= n_pedestal_reads <= nt):
        raise ValueError(
            "n_pedestal_reads must be between 1 and nreads."
        )

    if not (0 <= skip_initial_reads < nt):
        raise ValueError(
            "skip_initial_reads must be between 0 and nreads-1."
        )

    coeff = np.asarray(
        out["COEFF"][:, r, c],
        dtype=np.float64,
    )

    success = bool(out["SUCCESS"][r, c])

    # ------------------------------------------------------------
    # Saturation map
    # ------------------------------------------------------------

    sat = result.get("SATURATION_MAP", None)

    if sat is not None:
        sat_value = float(np.asarray(sat)[r, c])
    else:
        sat_value = np.inf

    # ------------------------------------------------------------
    # Read times
    # ------------------------------------------------------------

    if read_times is None:

        times = np.broadcast_to(
            np.arange(nt, dtype=np.float64),
            (nr, nt),
        )

    else:

        times = np.asarray(
            read_times,
            dtype=np.float64,
        )

        if times.shape == (nt,):
            times = np.broadcast_to(times, (nr, nt))

        if times.shape != (nr, nt):
            raise ValueError(
                "read_times must have shape "
                "(nreads,) or (nramps,nreads)."
            )

    # ------------------------------------------------------------
    # Extract pixel ramps
    # ------------------------------------------------------------

    pix = raw[:, :, r, c].astype(np.float64)

    # ------------------------------------------------------------
    # Estimate pedestal for each ramp
    # ------------------------------------------------------------

    pedestal_reads = pix[:, :n_pedestal_reads]

    pedestal_reads = np.where(
        np.isfinite(pedestal_reads),
        pedestal_reads,
        np.nan,
    )

    with np.errstate(invalid="ignore"):
        ped = np.nanmedian(
            pedestal_reads,
            axis=1,
        )

    signal = pix - ped[:, None]

    # ------------------------------------------------------------
    # Apply polynomial correction
    #
    # L(M) = c0 + c1*M + c2*M^2 + ...
    # ------------------------------------------------------------

    corrected_signal = np.polynomial.polynomial.polyval(
        signal,
        coeff,
    )

    corrected = ped[:, None] + corrected_signal

    # ------------------------------------------------------------
    # Construct valid-read mask
    # ------------------------------------------------------------

    valid = (
        np.isfinite(pix)
        & np.isfinite(corrected)
        & np.isfinite(times)
        & np.isfinite(signal)
    )

    if np.isfinite(sat_value) and sat_value > 0:

        threshold = (
            ped[:, None]
            + saturation_fraction
            * (sat_value - ped[:, None])
        )

        crossed = (
            ~np.isfinite(pix)
            | (pix >= threshold)
        )

        # Once saturated, all later reads are excluded
        after_saturation = np.maximum.accumulate(
            crossed,
            axis=1,
        )

        valid &= ~after_saturation

    else:
        # Without a valid saturation threshold, retain only
        # finite samples and flag the diagnostic limitation.
        valid &= ~np.maximum.accumulate(
            ~np.isfinite(pix),
            axis=1,
        )

    valid[:, :skip_initial_reads] = False

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        constrained_layout=True,
    )

    raw_residuals = []
    corrected_residuals = []
    correction_amplitudes = []

    # ------------------------------------------------------------
    # Loop through ramps
    # ------------------------------------------------------------

    for j in range(nr):

        t = times[j]
        mask = valid[j]

        # Panel 1: Raw and corrected ramps
        ax[0, 0].plot(
            t,
            np.where(np.isfinite(pix[j]), pix[j], np.nan),
            ".-",
            alpha=0.35,
            color="tab:blue",
            label="Raw" if j == 0 else None,
        )

        ax[0, 0].plot(
            t,
            np.where(mask, corrected[j], np.nan),
            ".-",
            alpha=0.65,
            color="tab:orange",
            label="Corrected (usable)" if j == 0 else None,
        )

        if np.count_nonzero(mask) < 3:
            continue

        x = t[mask]

        # --------------------------------------------------------
        # Fit a straight line independently to raw/corrected ramps
        # --------------------------------------------------------

        raw_fit = np.polyfit(
            x,
            pix[j, mask],
            1,
        )

        corrected_fit = np.polyfit(
            x,
            corrected[j, mask],
            1,
        )

        raw_res = (
            pix[j, mask]
            - np.polyval(raw_fit, x)
        )

        corrected_res = (
            corrected[j, mask]
            - np.polyval(corrected_fit, x)
        )

        raw_residuals.extend(raw_res)
        corrected_residuals.extend(corrected_res)

        # Panel 2: Residuals
        ax[0, 1].plot(
            x,
            raw_res,
            ".",
            alpha=0.45,
            color="tab:blue",
            label="Raw" if j == 0 else None,
        )

        ax[0, 1].plot(
            x,
            corrected_res,
            ".",
            alpha=0.45,
            color="tab:orange",
            label="Corrected" if j == 0 else None,
        )

        # Fractional correction
        amplitude = (
            100.0
            * (corrected[j, mask] - pix[j, mask])
            / np.maximum(
                np.abs(signal[j, mask]),
                100.0,
            )
        )

        correction_amplitudes.extend(amplitude)

    # ------------------------------------------------------------
    # Panel 1: Raw and corrected ramps
    # ------------------------------------------------------------

    ax[0, 0].set(
        title=f"Pixel ({r}, {c}) | fit success={success}",
        xlabel="Read time",
        ylabel="Absolute DN",
    )

    ax[0, 0].legend()
    ax[0, 0].grid(alpha=0.2)

    # ------------------------------------------------------------
    # Panel 2: Linear-fit residuals
    # ------------------------------------------------------------

    ax[0, 1].set(
        title="Raw versus corrected residuals",
        xlabel="Read time",
        ylabel="Residual [DN]",
    )

    ax[0, 1].axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
    )

    handles, labels = ax[0, 1].get_legend_handles_labels()

    if handles:
        ax[0, 1].legend()

    ax[0, 1].grid(alpha=0.2)

    # ------------------------------------------------------------
    # Panel 3: Residual distributions
    # ------------------------------------------------------------

    raw_residuals = np.asarray(
        raw_residuals,
        dtype=np.float64,
    )

    corrected_residuals = np.asarray(
        corrected_residuals,
        dtype=np.float64,
    )

    if raw_residuals.size and corrected_residuals.size:

        raw_rms = np.sqrt(
            np.mean(raw_residuals**2)
        )

        corrected_rms = np.sqrt(
            np.mean(corrected_residuals**2)
        )

        ax[1, 0].hist(
            raw_residuals,
            bins=40,
            alpha=0.5,
            color="tab:blue",
            label=f"Raw RMS = {raw_rms:.3f} DN",
        )

        ax[1, 0].hist(
            corrected_residuals,
            bins=40,
            alpha=0.5,
            color="tab:orange",
            label=f"Corrected RMS = {corrected_rms:.3f} DN",
        )

        ax[1, 0].legend()

    ax[1, 0].set(
        title="Residual distribution",
        xlabel="Residual [DN]",
        ylabel="Number of samples",
    )

    ax[1, 0].grid(alpha=0.2)

    # ------------------------------------------------------------
    # Panel 4: Fractional correction distribution
    # ------------------------------------------------------------

    correction_amplitudes = np.asarray(
        correction_amplitudes,
        dtype=np.float64,
    )

    correction_amplitudes = correction_amplitudes[
        np.isfinite(correction_amplitudes)
    ]

    if correction_amplitudes.size:

        ax[1, 1].hist(
            correction_amplitudes,
            bins=40,
            color="tab:green",
            alpha=0.75,
        )

        median_correction = np.median(
            correction_amplitudes
        )

        ax[1, 1].axvline(
            median_correction,
            color="black",
            linestyle="--",
            label=f"Median = {median_correction:.3f}%",
        )

        ax[1, 1].legend()

    ax[1, 1].set(
        title="Fractional correction on usable reads",
        xlabel="Correction [%]",
        ylabel="Number of samples",
    )

    ax[1, 1].grid(alpha=0.2)

    # ------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------

    fig.suptitle(
        f"Linearity ramp validation | Pixel ({r}, {c}) "
        "| In-sample",
        fontsize=14,
    )

    # ------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------

    if save_prefix is not None:

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prefix = Path(str(save_prefix)).stem

        filename = (
            f"{prefix}_ramp_validation_"
            f"row{r}_col{c}.png"
        )

        savepath = output_dir / filename

        fig.savefig(
            savepath,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"Linearity ramp validation saved: {savepath}"
        )

    if show:
        plt.show()

    return fig, ax







