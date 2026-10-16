"""
Automatic per-pixel saturation-threshold estimation

The detector is processed in row blocks. Input ramps should have shape
(n_ramps, n_reads, nrows, nx). Currently the input is ACN and 1/f, and linearity corrected.
The main output maps have shape (nrows, nx), 2D saturation map
"""
import argparse
import warnings
from pathlib import Path
import numpy as np
from astropy.io import fits
# ======================================================================
# Saturation-quality DQ flags
# ======================================================================

SAT_DQ = {
    "NONFINITE":             np.uint32(1 << 0),
    "LOW_REFERENCE_DIFF":    np.uint32(1 << 1),
    "NO_SATURATION_FOUND":   np.uint32(1 << 2),
    "FALLBACK_USED":         np.uint32(1 << 3),
    "SATURATION_TOO_LOW":    np.uint32(1 << 4),
    "SATURATION_TOO_HIGH":   np.uint32(1 << 5),
    "LOW_CONFIDENCE":        np.uint32(1 << 6),
    "TOO_FEW_VALID_RAMPS":   np.uint32(1 << 7),
    "EARLY_SATURATION":      np.uint32(1 << 8),
    "INCONSISTENT_RAMPS":    np.uint32(1 << 9),
    "NEIGHBOR_FILL_USED":    np.uint32(1 << 10),
    "GLOBAL_FILL_USED":      np.uint32(1 << 11),
}

def first_sustained_true(condition, window=3):
    """
    Find the first index where a condition is True for a sustained window.
    Three consecutive difference two be True
    Parameters
    ----------
    condition : ndarray of bool
        Boolean array with shape

            (..., n_differences)

        The last axis is interpreted as the ramp-difference axis.

    window : int, optional
        Number of consecutive True values required.

    Returns
    -------
    first_index : ndarray of int32
        First index where the sustained condition begins.

        Shape is ``condition.shape[:-1]``.

        A value of -1 means no sustained sequence was found.

    detected : ndarray of bool
        True where a sustained sequence was found.
    """

    condition = np.asarray(condition, dtype=bool)

    if condition.ndim < 1:
        raise ValueError("condition must have at least one dimension.")

    window = int(window)

    if window < 1:
        raise ValueError("window must be at least 1.")

    ndiff = condition.shape[-1]

    if ndiff < window: #if the number of difference in less than the window size,
        #detection is not possible
        output_shape = condition.shape[:-1]

        return (
            np.full(output_shape, -1, dtype=np.int32), #-1 for every first index
            np.zeros(output_shape, dtype=bool),
        )

    #create a array filled with True
    sustained = np.ones(
        condition.shape[:-1] + (ndiff - window + 1,),
        dtype=bool,
    )

    for offset in range(window):
        sustained &= condition[..., offset:offset + ndiff - window + 1]

    detected = np.any(sustained, axis=-1) #return true if one valid sustained window exist

    #return the index of the first maximum value
    #if the entire sequence is False, np.argmax() still returns 0
    first_index = np.argmax(sustained, axis=-1).astype(np.int32)

    #For elements where no sustained window was found,
    #the misleading 0 returned by argmax() is replaced with -1.
    first_index = np.where(detected, first_index, -1).astype(np.int32)

    #returns the difference index and detected=True or False
    return first_index, detected

def estimate_reference_difference(
    differences,
    *,
    start_difference=1,
    n_reference_differences=10,
    minimum_valid=3,
):
    """
    Estimate the normal early-ramp difference for every ramp and pixel.

    Parameters
    ----------
    differences : ndarray
        Ramp differences with shape

            (n_ramps, n_differences, nrows, nx)

    start_difference : int, optional
        First ramp-difference index used for estimating the linear signal.

        Setting this to 1 skips the first difference, which may contain
        reset-related behavior.

    n_reference_differences : int, optional
        Maximum number of early differences used.

    minimum_valid : int, optional
        Minimum number of finite differences required.

    Returns
    -------
    reference_difference : ndarray
        Per-ramp, per-pixel early-ramp reference difference with shape

            (n_ramps, nrows, nx)

    n_valid_reference : ndarray
        Number of finite early differences used, with shape

            (n_ramps, nrows, nx)
    """

    differences = np.asarray(differences)

    if differences.ndim != 4:
        raise ValueError(
            "differences must have shape "
            "(n_ramps, n_differences, nrows, nx)."
        )

    if minimum_valid < 1:
        raise ValueError("minimum_valid must be positive.")
    ndiff = differences.shape[1]

    start_difference = int(start_difference)
    n_reference_differences = int(n_reference_differences)
    minimum_valid = int(minimum_valid)

    if start_difference < 0:
        raise ValueError("start_difference must be non-negative.")

    if n_reference_differences < 1:
        raise ValueError(
            "n_reference_differences must be at least 1."
        )

    stop_difference = min(
        start_difference + n_reference_differences,
        ndiff,
    )

    if start_difference >= stop_difference:
        raise ValueError(
            "The requested reference-difference range is empty."
        )

    reference_region = differences[
        :,
        start_difference:stop_difference,
        :,
        :,
    ]

    #boolean array where True for finite values & False otherwise
    finite = np.isfinite(reference_region)

    #count usable differences
    n_valid_reference = np.sum(
        finite,
        axis=1,
        dtype=np.uint32,
    )

    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        reference_difference = np.nanmedian(
            np.where(finite, reference_region, np.nan),
            axis=1,
        )

    reference_difference[
        n_valid_reference < minimum_valid
    ] = np.nan
    #returns median early ramp difference & number fo finite sample used
    return reference_difference, n_valid_reference

def estimate_saturation_block(
    ramps,
    *,
    drop_fraction=0.85,
    sustained_window=3, 
    start_difference=1,
    n_reference_differences=10,
    minimum_reference_differences=3,
    minimum_reference_signal=1.0,
    minimum_saturation_dn=2000.0,
    maximum_saturation_dn=None,
    fallback_saturation_dn=None,
    minimum_valid_ramps=1,
    minimum_detected_ramps=3,
    search_after_reference=False,
    minimum_detection_fraction=0.5,
    maximum_ramp_scatter_fraction=0.10,
    skip_initial_reads=2,
):
    """Measure one block; no filling or fallback is performed here.

    Input shape: (ramps, reads, rows, columns), with equal read cadence.
    Defaults: 0.85 drop, three sustained differences, early reference indices
    1:11. Detection searches the whole ramp unless search_after_reference=True.
    Early hits are retained and flagged if their onset overlaps the reference
    interval or precedes skip_initial_reads. This does not prove the reference
    is linear; inspect early flagged ramps or shorten the reference interval.

    Absolute input DN at read k+1 is used when difference k first drops.
    Match calibration and science preprocessing, including linearity correction
    and pedestal convention. No reset-pedestal subtraction is performed.
    SAT_READ is the rounded median zero-based read index; SATURATION_RAW is
    independently the median DN. They need not represent the same ramp.
    NVALIDRAMPS counts usable references; NDETECTED counts finite hits.
    SAT_SCATTER is 1.4826*MAD, NaN for fewer than three hits. SAT_FRACTION
    is NaN without usable references. LOW_CONFIDENCE retains the measurement.
    fallback_saturation_dn is retained only to reject legacy block-level filling.
    """

    ramps = np.asarray(ramps)

    if ramps.ndim != 4:
        raise ValueError(
            "ramps must have shape "
            "(n_ramps, n_reads, nrows, nx)."
        )

    nramps, nreads, nrows, nx = ramps.shape

    if min(nrows, nx) < 1:
        raise ValueError("Spatial dimensions must be nonempty.")
    if minimum_valid_ramps < 1 or minimum_detected_ramps < 1:
        raise ValueError("Minimum ramp counts must be positive.")
    if minimum_reference_signal <= 0 or maximum_ramp_scatter_fraction < 0:
        raise ValueError("Reference signal must be positive; scatter limit nonnegative.")
    if skip_initial_reads < 0:
        raise ValueError("skip_initial_reads must be nonnegative.")
    if maximum_saturation_dn is not None and maximum_saturation_dn < minimum_saturation_dn:
        raise ValueError("Maximum saturation must be at least the minimum.")
    if fallback_saturation_dn is not None:
        raise ValueError("Apply fallback in finalize_saturation_map(), after assembling all blocks.")
    if nramps < 1:
        raise ValueError("At least one ramp is required.")

    if nreads < 3:
        raise ValueError(
            "At least three reads are required for saturation detection."
        )

    if not 0.0 < drop_fraction < 1.0:
        raise ValueError(
            "drop_fraction must be between 0 and 1."
        )

    if not 0.0 <= minimum_detection_fraction <= 1.0:
        raise ValueError(
            "minimum_detection_fraction must be between 0 and 1."
        )

    # --------------------------------------------------------------
    # Compute read-to-read differences
    # --------------------------------------------------------------

    differences = np.diff(
        ramps.astype(np.float32, copy=False),
        axis=1,
    )

    # --------------------------------------------------------------
    # Estimate normal early-ramp difference
    # --------------------------------------------------------------
    #for ramp and pixel:
    #reference_per_ramp contains the median early difference (n_ramps,nrows,nx),
    #n_reference_valid (n_ramps,nrows,nx) contains the number of finite early differences used.
    reference_per_ramp, n_reference_valid = (
        estimate_reference_difference(
            differences,
            start_difference=start_difference,
            n_reference_differences=n_reference_differences,
            minimum_valid=minimum_reference_differences,
        )
    )
    #A reference estimate is usable only when both conditions are satisfied:
    valid_reference = (
        np.isfinite(reference_per_ramp)
        & (reference_per_ramp >= minimum_reference_signal)
    )

    # --------------------------------------------------------------
    # Identify sustained difference drops
    # --------------------------------------------------------------
    #add a new axis to position 1 (n_ramps,1,nrows,nx)
    threshold = (
        reference_per_ramp[:, None, :, :]
        * np.float32(drop_fraction)
    )

    finite_difference = np.isfinite(differences)
    #A sample becomes True only when all three conditions hold:
    below_threshold = (
        finite_difference
        & valid_reference[:, None, :, :]
        & (differences < threshold)
    )

    reference_stop = min(start_difference + n_reference_differences, nreads - 1)
    # Keep early detections by default for diagnostics. Optionally search only
    # after the reference interval; short ramps then legitimately yield no hit.
    if search_after_reference:
        below_threshold[:, :reference_stop] = False

    # Move the difference axis to the end because first_sustained_true()
    # expects the search axis to be last.
    #(n_ramps, n_differences, nrows, nx) ==> (n_ramps, nrows, nx, n_differences)
    below_threshold_last = np.moveaxis(
        below_threshold,
        1,
        -1,
    )

    #This searches for the first sequence of sustained_window consecutive low differences.
    first_difference_index, detected = first_sustained_true(
        below_threshold_last,
        window=sustained_window,
    )

    # Therefore the corresponding first affected read is k + 1.
    #convert difference index to read index
    saturation_read_per_ramp = np.where(
        detected,
        first_difference_index + 1,
        -1,
    ).astype(np.int32)

    # --------------------------------------------------------------
    # Extract signal value at first saturated read
    # --------------------------------------------------------------
    #Pixels without a detection have index -1, after temporarly asigned to 0 here.
    safe_read_index = np.clip(
        saturation_read_per_ramp,
        0,
        nreads - 1,
    )
    #(nramps,nrows,nx)
    saturation_per_ramp = np.take_along_axis(
        ramps,
        safe_read_index[:, None, :, :],
        axis=1,
    )[:, 0, :, :].astype(np.float32, copy=False)
    #pixels without valid detection assigned with temporary NaN values.
    saturation_per_ramp[~detected] = np.nan

    # --------------------------------------------------------------
    # Combine ramps & count valid detection and valid ramps
    # --------------------------------------------------------------

    detected_and_finite = (
        detected
        & np.isfinite(saturation_per_ramp)
    )
    #Counts how many ramps detected finite saturation for each pixel.
    n_detected = np.sum(
        detected_and_finite,
        axis=0,
        dtype=np.uint32,
    )
    #Counts how many ramps had a usable early-ramp reference difference for each pixel.
    n_valid_ramps = np.sum(
        valid_reference,
        axis=0,
        dtype=np.uint32,
    )
    #Combines the ramp-specific saturation DN estimates using the median.
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        saturation = np.nanmedian(
            saturation_per_ramp,
            axis=0,
        ).astype(np.float32)
        #This first creates an array where invalid reference values are replaced with NaN.
        #Then it calculates the median reference difference across ramps.
        reference_difference = np.nanmedian(
            np.where(
                valid_reference,
                reference_per_ramp,
                np.nan,
            ),
            axis=0,
        ).astype(np.float32)
        
        #This calculates the median saturation-read index
        #across the ramps that actually detected saturation.
        saturation_read_float = np.nanmedian(
            np.where(
                detected,
                saturation_read_per_ramp.astype(np.float32),
                np.nan,
            ),
            axis=0,
        )
    #Creates the final two-dimensional saturation-read map with -1 ==> no valid median detected.
    saturation_read = np.full(
        (nrows, nx),
        -1,
        dtype=np.int32,
    )
    #Identifies pixels where at least one valid saturation-read estimate exists.
    valid_read_median = np.isfinite(saturation_read_float)

    saturation_read[valid_read_median] = np.rint(
        saturation_read_float[valid_read_median]
    ).astype(np.int32)

    # --------------------------------------------------------------
    # Robust ramp-to-ramp scatter
    # --------------------------------------------------------------
    #Calculates the median saturation value but retains the ramp axis as a length-one dimension.
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        sat_median_expanded = np.nanmedian(
            saturation_per_ramp,
            axis=0,
            keepdims=True,
        )
        #calculate MAD
        saturation_mad = np.nanmedian(
            np.abs(
                saturation_per_ramp
                - sat_median_expanded
            ),
            axis=0,
        )

    saturation_scatter = (1.4826 * saturation_mad).astype(np.float32)
    saturation_scatter[n_detected < 3] = np.nan
    #saturation fraction map initialized
    saturation_fraction = np.full((nrows, nx), np.nan, dtype=np.float32)

    valid_denominator = n_valid_ramps > 0

    saturation_fraction[valid_denominator] = (
        n_detected[valid_denominator]
        / n_valid_ramps[valid_denominator]
    )

    # --------------------------------------------------------------
    # Build saturation-quality map
    # --------------------------------------------------------------

    sat_quality = np.zeros(
        (nrows, nx),
        dtype=np.uint32,
    )

    # Retain the unfiltered reference median: filtering first makes LOW_REFERENCE_DIFF unreachable.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        reference_difference = np.nanmedian(reference_per_ramp, axis=0).astype(np.float32)
    no_finite_reference = ~np.isfinite(reference_difference)

    sat_quality[no_finite_reference | ~np.isfinite(saturation)] |= SAT_DQ["NONFINITE"]

    low_reference = (
        np.isfinite(reference_difference)
        & (reference_difference < minimum_reference_signal)
    )

    sat_quality[low_reference] |= SAT_DQ["LOW_REFERENCE_DIFF"]

    too_few_valid_ramps = (
        n_valid_ramps < minimum_valid_ramps
    )

    sat_quality[too_few_valid_ramps] |= (
        SAT_DQ["TOO_FEW_VALID_RAMPS"]
    )

    no_saturation = n_detected == 0

    sat_quality[no_saturation] |= (
        SAT_DQ["NO_SATURATION_FOUND"]
    )

    low_confidence = (
        (n_valid_ramps > 0)
        & (
            saturation_fraction
            < minimum_detection_fraction
        )
    )

    sat_quality[low_confidence | (n_detected < minimum_detected_ramps)] |= SAT_DQ["LOW_CONFIDENCE"]

    too_low = (
        np.isfinite(saturation)
        & (saturation < minimum_saturation_dn)
    )

    sat_quality[too_low] |= SAT_DQ["SATURATION_TOO_LOW"]

    if maximum_saturation_dn is not None:
        too_high = (
            np.isfinite(saturation)
            & (saturation > maximum_saturation_dn)
        )

        sat_quality[too_high] |= SAT_DQ["SATURATION_TOO_HIGH"]

    early_saturation = np.any(
        detected_and_finite & ((saturation_read_per_ramp < skip_initial_reads)
                              | (first_difference_index < reference_stop)), axis=0)

    sat_quality[early_saturation] |= SAT_DQ["EARLY_SATURATION"]

    valid_scatter_test = (
        np.isfinite(saturation)
        & np.isfinite(saturation_scatter)
        & (np.abs(saturation) > 0)
    )

    relative_scatter = np.full(
        (nrows, nx),
        np.nan,
        dtype=np.float32,
    )

    relative_scatter[valid_scatter_test] = (
        saturation_scatter[valid_scatter_test]
        / np.abs(saturation[valid_scatter_test])
    )

    inconsistent_ramps = (
        valid_scatter_test
        & (
            relative_scatter
            > maximum_ramp_scatter_fraction
        )
    )

    sat_quality[inconsistent_ramps] |= (
        SAT_DQ["INCONSISTENT_RAMPS"]
    )

    # Block output is measurement-only. Interpolate once on the full detector.
    return {
        "SATURATION": saturation.copy(),
        "SATURATION_RAW": saturation,
        "SAT_QUALITY": sat_quality,
        "SAT_READ": saturation_read,
        "REF_DIFF": reference_difference,
        "NDETECTED": n_detected,
        "NVALIDRAMPS": n_valid_ramps,
        "SAT_SCATTER": saturation_scatter,
        "SAT_FRACTION": saturation_fraction,
        "SATURATION_PER_RAMP": saturation_per_ramp,
        "SAT_READ_PER_RAMP": saturation_read_per_ramp,
        "REFERENCE_PER_RAMP": reference_per_ramp,
        "N_REFERENCE_VALID": n_reference_valid,

    }

def fill_invalid_saturation_neighbors(
    saturation,
    *,
    minimum_saturation_dn=None,
    maximum_saturation_dn=None,
    maximum_iterations=5,
    final_fallback=None,
    invalid_mask=None,
):
    """Fill rejected/missing pixels from synchronous eight-neighbor medians.

    maximum_iterations=5 limits propagation to five Chebyshev pixels. None
    permits detector-wide propagation. No edge wrapping occurs. invalid_mask
    excludes flagged seeds. final_fallback=None leaves unresolved NaNs;
    'median' uses ORIGINAL accepted measurements, or supply a finite scalar.
    Returns (filled, neighbor_filled, global_filled, originally_invalid).
    No valid seeds is allowed with None/scalar fallback; median fallback raises.
    """

    filled = np.asarray(saturation, dtype=np.float32).copy()

    if filled.ndim != 2:
        raise ValueError("saturation must be a two-dimensional array.")

    valid = np.isfinite(filled)

    if minimum_saturation_dn is not None:
        valid &= filled >= float(minimum_saturation_dn)

    if maximum_saturation_dn is not None:
        valid &= filled <= float(maximum_saturation_dn)

    originally_invalid = ~valid

    if minimum_saturation_dn is not None and maximum_saturation_dn is not None:
        if maximum_saturation_dn < minimum_saturation_dn:
            raise ValueError("Invalid saturation limits.")
    if invalid_mask is not None:
        invalid_mask = np.asarray(invalid_mask, dtype=bool)
        if invalid_mask.shape != filled.shape:
            raise ValueError("invalid_mask must match saturation shape.")
        valid &= ~invalid_mask
    originally_invalid = ~valid
    original_seeds = filled[valid].copy()

    # Remove invalid and unrealistic seed values before interpolation.
    filled[~valid] = np.nan

    neighbor_filled = np.zeros(filled.shape, dtype=bool)
    global_filled = np.zeros(filled.shape, dtype=bool)

    if maximum_iterations is None:
        maximum_iterations = max(filled.shape)
    else:
        maximum_iterations = int(maximum_iterations)

        if maximum_iterations < 0:
            raise ValueError(
                "maximum_iterations must be non-negative or None."
            )

    def shift_with_nan(array, dy, dx):
        """Shift a 2-D array without wrapping across detector edges."""

        shifted = np.full(
            array.shape,
            np.nan,
            dtype=np.float32,
        )

        ny, nx = array.shape

        source_y0 = max(0, -dy)
        source_y1 = min(ny, ny - dy)
        source_x0 = max(0, -dx)
        source_x1 = min(nx, nx - dx)

        target_y0 = max(0, dy)
        target_y1 = min(ny, ny + dy)
        target_x0 = max(0, dx)
        target_x1 = min(nx, nx + dx)

        shifted[
            target_y0:target_y1,
            target_x0:target_x1,
        ] = array[
            source_y0:source_y1,
            source_x0:source_x1,
        ]

        return shifted

    neighbor_offsets = (
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1),
    )

    for _ in range(maximum_iterations):
        unresolved = ~np.isfinite(filled)

        if not np.any(unresolved):
            break

        neighbor_stack = np.stack(
            [
                shift_with_nan(filled, dy, dx)
                for dy, dx in neighbor_offsets
            ],
            axis=0,
        )

        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            neighbor_median = np.nanmedian(
                neighbor_stack,
                axis=0,
            )

        can_fill = (
            unresolved
            & np.isfinite(neighbor_median)
        )

        if not np.any(can_fill):
            break

        filled[can_fill] = neighbor_median[can_fill]
        neighbor_filled[can_fill] = True

    # Catch any large isolated region that could not be reached.
    unresolved = ~np.isfinite(filled)

    if np.any(unresolved) and final_fallback is not None:
        valid_values = original_seeds

        if isinstance(final_fallback, str):
            if final_fallback.lower() != "median":
                raise ValueError(
                    "final_fallback must be 'median', a scalar, or None."
                )

            if valid_values.size == 0:
                raise ValueError("Cannot use median fallback without valid measured seeds.")
            fallback_value = float(np.median(valid_values))
        else:
            fallback_value = float(final_fallback)

        if not np.isfinite(fallback_value):
            raise ValueError(
                "The final saturation fallback must be finite."
            )

        if (
            minimum_saturation_dn is not None
            and fallback_value < minimum_saturation_dn
        ):
            fallback_value = float(minimum_saturation_dn)

        if (
            maximum_saturation_dn is not None
            and fallback_value > maximum_saturation_dn
        ):
            fallback_value = float(maximum_saturation_dn)

        filled[unresolved] = np.float32(fallback_value)
        global_filled[unresolved] = True

    # Final safety clipping.
    if minimum_saturation_dn is not None:
        filled = np.maximum(
            filled,
            np.float32(minimum_saturation_dn),
        )

    if maximum_saturation_dn is not None:
        filled = np.minimum(
            filled,
            np.float32(maximum_saturation_dn),
        )

    return (
        filled.astype(np.float32, copy=False),
        neighbor_filled,
        global_filled,
        originally_invalid,
    )


# Severe measurement flags excluded from interpolation seeds. Low-confidence
# measurements are retained by default; set reject_low_confidence=True to omit.
DEFAULT_REJECT_FLAGS = int(
    SAT_DQ['SATURATION_TOO_LOW'] | SAT_DQ['SATURATION_TOO_HIGH']
    | SAT_DQ['EARLY_SATURATION'] | SAT_DQ['INCONSISTENT_RAMPS']
    | SAT_DQ['TOO_FEW_VALID_RAMPS']
)
MAP_KEYS = ('SATURATION_RAW', 'SAT_QUALITY', 'SAT_READ', 'REF_DIFF',
            'NDETECTED', 'NVALIDRAMPS', 'SAT_SCATTER', 'SAT_FRACTION')


def finalize_saturation_map(products, *, minimum_saturation_dn=2000.0,
                            maximum_saturation_dn=None, maximum_iterations=5,
                            final_fallback=None, reject_flags=DEFAULT_REJECT_FLAGS,
                            reject_low_confidence=False):
    """Return full-detector products with quality-aware filling and intact raw data.

    SAT_REJECTED marks measurements excluded before interpolation; SAT_UNRESOLVED
    identifies final NaNs. Original DQ bits survive filling. Diagnostics always
    describe measurements, never interpolated values. Call once after assembly.
    """
    result = {key: np.array(value, copy=True) for key, value in products.items()}
    raw = result['SATURATION_RAW']
    quality = result['SAT_QUALITY']
    if raw.ndim != 2 or quality.shape != raw.shape:
        raise ValueError('Raw saturation and quality must have matching 2-D shapes.')
    if reject_low_confidence:
        reject_flags |= int(SAT_DQ['LOW_CONFIDENCE'])
    invalid = (quality & np.uint32(reject_flags)) != 0
    filled, neighbors, global_fill, rejected = fill_invalid_saturation_neighbors(
        raw, minimum_saturation_dn=minimum_saturation_dn,
        maximum_saturation_dn=maximum_saturation_dn,
        maximum_iterations=maximum_iterations, final_fallback=final_fallback,
        invalid_mask=invalid)
    quality[neighbors] |= SAT_DQ['FALLBACK_USED'] | SAT_DQ['NEIGHBOR_FILL_USED']
    quality[global_fill] |= SAT_DQ['FALLBACK_USED'] | SAT_DQ['GLOBAL_FILL_USED']
    result.update(SATURATION=filled, SAT_NEIGHBOR_FILLED=neighbors,
                  SAT_GLOBAL_FILLED=global_fill, SAT_ORIGINALLY_INVALID=rejected,
                  SAT_REJECTED=rejected, SAT_UNRESOLVED=~np.isfinite(filled))
    return result


def write_saturation_fits(
    products, output, *, header=None, filenames=None, camera=None,
    mclock=None, exptime=None, calibration_options=None, overwrite=False,
):
    """Write maps, quality diagnostics and calibration metadata.

    calibration_options accepts the same estimation/filling settings used in
    create_saturation_map(). Pass your actual options when changing defaults.
    Existing header cards are preserved unless explicitly overridden here.
    filenames supplies NCOMBINE and input-file history. No ramps are reprocessed.
    """
    primary_header = fits.Header() if header is None else header.copy()
    options = dict(calibration_options or {})
    cards = {
        'drop_fraction': ('SATDROP', 0.85, 'Fractional difference threshold'),
        'sustained_window': ('SATWIN', 3, 'Number of sustained low differences'),
        'start_difference': ('REFSTRT', 1, 'First reference difference; zero-based'),
        'n_reference_differences': ('NREFDIFF', 10, 'Maximum reference differences'),
        'minimum_reference_differences': ('MINREF', 3, 'Minimum finite reference differences'),
        'minimum_reference_signal': ('MINRSIG', 1.0, 'Minimum usable reference difference DN'),
        'minimum_saturation_dn': ('MINSAT', 2000.0, 'Minimum accepted saturation DN'),
        'maximum_saturation_dn': ('MAXSAT', None, 'Upper saturation limit DN; NONE if unset'),
        'minimum_valid_ramps': ('MINRAMPS', 1, 'Minimum ramps with usable references'),
        'minimum_detected_ramps': ('MINDET', 3, 'Minimum detections for confidence'),
        'minimum_detection_fraction': ('SATFRMIN', 0.5, 'Minimum saturation detection fraction'),
        'maximum_ramp_scatter_fraction': ('SATSCMAX', 0.10, 'Maximum relative robust ramp scatter'),
        'skip_initial_reads': ('SKIPRD', 2, 'Early saturation read threshold'),
        'search_after_reference': ('SRCHREF', False, 'Search only after reference interval'),
        'maximum_iterations': ('FILLITER', 5, 'Maximum neighbor-fill iterations'),
        'fallback_saturation_dn': ('FALLBACK', None, 'Global fallback DN, MEDIAN or NONE'),
        'reject_low_confidence': ('REJLOW', False, 'Reject low-confidence measurements'),
        'block_rows': ('BLKROWS', 32, 'Rows processed per measurement block'),
    }
    unknown = set(options) - set(cards)
    if unknown:
        raise ValueError(f'Unknown calibration options: {sorted(unknown)}')
    for name, (keyword, default, comment) in cards.items():
        if name in options or keyword not in primary_header:
            value = options.get(name, default)
            if value is None:
                value = -1 if name == 'maximum_iterations' else 'NONE'
            elif name == 'fallback_saturation_dn' and isinstance(value, str):
                value = value.upper()
            primary_header[keyword] = (value, comment)
    primary_header['CALTYPE'] = ('SATURATION', 'Calibration product type')
    primary_header['SATCONV'] = ('ABSOLUTE', 'Input DN; no pedestal subtraction')
    for keyword, value in (('CAMERA', camera), ('MCLOCK', mclock), ('EXPTIME', exptime)):
        if value is not None:
            primary_header[keyword] = value
    if filenames is not None:
        filenames = list(filenames)
        primary_header['NCOMBINE'] = (len(filenames), 'Number of L11 ramps used')
        primary_header.add_history(f'Combined {len(filenames)} L11 ramps')
        for filename in filenames:
            primary_header.add_history(f'Input: {filename}')
    elif 'NCOMBINE' not in primary_header and 'SATURATION_PER_RAMP' in products:
        primary_header['NCOMBINE'] = (products['SATURATION_PER_RAMP'].shape[0],
                                     'Number of L11 ramps used')
    for line in (
        'SCALES detector saturation map created from L11 ramps',
        'SATURATION_RAW preserves measurements before rejection and filling',
        'Quality-aware neighbor filling performed on the full detector',
        'Original quality flags retained after filling',
        'Early flags include detections within the reference interval',
        'SAT_READ and SATURATION_RAW are independent median summaries',
        'SAT_SCATTER is NaN for fewer than three detections',
    ):
        primary_header.add_history(line)
    # Add input-preprocessing history in the supplied header: it is not inferred.
    extensions = [
        ('SATURATION', 'SATURATION', np.float32),
        ('SATURATION_RAW', 'SATURATION_RAW', np.float32),
        ('SAT_READ', 'SAT_READ', np.int32),
        ('SAT_QUALITY', 'SAT_QUALITY', np.uint32),
        ('REF_DIFF', 'REF_DIFF', np.float32),
        ('NDETECTED', 'NDETECTED', np.uint32),
        ('NVALIDRAMPS', 'NVALIDRAMP', np.uint32),
        ('SAT_SCATTER', 'SATSCAT', np.float32),
        ('SAT_FRACTION', 'SATFRAC', np.float32),
        ('SAT_ORIGINALLY_INVALID', 'ORIGINVAL', np.uint8),
        ('SAT_NEIGHBOR_FILLED', 'NEIGHFILL', np.uint8),
        ('SAT_GLOBAL_FILLED', 'GLOBALFILL', np.uint8),
        ('SAT_UNRESOLVED', 'UNRESOLV', np.uint8),
    ]
    shape = np.asarray(products['SATURATION']).shape
    if len(shape) != 2:
        raise ValueError('SATURATION must be a two-dimensional map.')
    hdus = [fits.PrimaryHDU(header=primary_header)]
    for key, extension_name, dtype in extensions:
        value = np.asarray(products[key], dtype=dtype)
        if value.shape != shape:
            raise ValueError(f'{key} must have shape {shape}.')
        hdu = fits.ImageHDU(value, name=extension_name)
        if key in ('SATURATION', 'SATURATION_RAW', 'REF_DIFF', 'SAT_SCATTER'):
            hdu.header['BUNIT'] = 'DN'
        if key == 'SAT_READ':
            hdu.header['INDEX0'] = (True, 'Zero-based; -1 means no detection')
        hdus.append(hdu)
    names = list(SAT_DQ)
    hdus.append(fits.BinTableHDU.from_columns([
        fits.Column(name='FLAG', format='32A', array=names),
        fits.Column(name='VALUE', format='K', array=[int(SAT_DQ[n]) for n in names]),
    ], name='DQ_DEFINITIONS'))
    fits.HDUList(hdus).writeto(output, overwrite=overwrite, checksum=True)


def create_saturation_map(
    ramps,
    *,
    drop_fraction=0.85,
    sustained_window=3,
    start_difference=1,
    n_reference_differences=10,
    minimum_reference_differences=3,
    minimum_reference_signal=1.0,
    minimum_saturation_dn=2000.0,
    maximum_saturation_dn=None,
    fallback_saturation_dn=None,
    minimum_valid_ramps=1,
    minimum_detection_fraction=0.5,
    maximum_ramp_scatter_fraction=0.10,
    skip_initial_reads=2,
    minimum_detected_ramps=3,
    search_after_reference=False,
    block_rows=32,
    maximum_iterations=5,
    reject_low_confidence=False,
):
    """Create a full-detector map and return the input script's dictionary format.

    ramps has shape (n_ramps, n_reads, nrows, nx). Detection is processed in
    row blocks; interpolation runs only after all blocks are assembled.
    The same keys as the original estimate_saturation_block() are returned,
    plus SATURATION_RAW, SAT_REJECTED and SAT_UNRESOLVED. Per-ramp diagnostics
    have shape (n_ramps, nrows, nx); all other products have shape (nrows, nx).
    No file is written. Use write_saturation_fits(result, path) if desired.

    fallback_saturation_dn=None leaves unresolved pixels as NaN after bounded
    neighbor filling. Set it to a numeric DN value or 'median' to fill the rest.
    Raw measurements and their quality flags are preserved in every case.
    """
    ramps = np.asarray(ramps)
    if ramps.ndim != 4 or any(size < 1 for size in ramps.shape):
        raise ValueError('ramps must have nonempty shape (ramps, reads, rows, columns).')
    if not isinstance(block_rows, (int, np.integer)) or block_rows < 1:
        raise ValueError('block_rows must be a positive integer.')
    nramps, _, ny, nx = ramps.shape
    per_ramp_keys = ('SATURATION_PER_RAMP', 'SAT_READ_PER_RAMP',
                     'REFERENCE_PER_RAMP', 'N_REFERENCE_VALID')
    products = None
    for y0 in range(0, ny, block_rows):
        y1 = min(y0 + block_rows, ny)
        block = estimate_saturation_block(
            ramps[:, :, y0:y1, :],
            drop_fraction=drop_fraction, sustained_window=sustained_window,
            start_difference=start_difference,
            n_reference_differences=n_reference_differences,
            minimum_reference_differences=minimum_reference_differences,
            minimum_reference_signal=minimum_reference_signal,
            minimum_saturation_dn=minimum_saturation_dn,
            maximum_saturation_dn=maximum_saturation_dn,
            minimum_valid_ramps=minimum_valid_ramps,
            minimum_detection_fraction=minimum_detection_fraction,
            maximum_ramp_scatter_fraction=maximum_ramp_scatter_fraction,
            skip_initial_reads=skip_initial_reads,
            minimum_detected_ramps=minimum_detected_ramps,
            search_after_reference=search_after_reference)
        if products is None:
            products = {key: np.empty((ny, nx), dtype=block[key].dtype)
                        for key in MAP_KEYS}
            products.update({key: np.empty((nramps, ny, nx), dtype=block[key].dtype)
                             for key in per_ramp_keys})
        for key in MAP_KEYS:
            products[key][y0:y1, :] = block[key]
        for key in per_ramp_keys:
            products[key][:, y0:y1, :] = block[key]
    result = finalize_saturation_map(
        products, minimum_saturation_dn=minimum_saturation_dn,
        maximum_saturation_dn=maximum_saturation_dn,
        maximum_iterations=maximum_iterations,
        final_fallback=fallback_saturation_dn,
        reject_low_confidence=reject_low_confidence)
    return {
        'SATURATION': result['SATURATION'],
        'SAT_QUALITY': result['SAT_QUALITY'],
        'SAT_READ': result['SAT_READ'],
        'REF_DIFF': result['REF_DIFF'],
        'NDETECTED': result['NDETECTED'],
        'NVALIDRAMPS': result['NVALIDRAMPS'],
        'SAT_SCATTER': result['SAT_SCATTER'],
        'SAT_FRACTION': result['SAT_FRACTION'],
        'SATURATION_PER_RAMP': result['SATURATION_PER_RAMP'],
        'SAT_READ_PER_RAMP': result['SAT_READ_PER_RAMP'],
        'REFERENCE_PER_RAMP': result['REFERENCE_PER_RAMP'],
        'N_REFERENCE_VALID': result['N_REFERENCE_VALID'],
        'SAT_NEIGHBOR_FILLED': result['SAT_NEIGHBOR_FILLED'],
        'SAT_GLOBAL_FILLED': result['SAT_GLOBAL_FILLED'],
        'SAT_ORIGINALLY_INVALID': result['SAT_ORIGINALLY_INVALID'],
        'SATURATION_RAW': result['SATURATION_RAW'],
        'SAT_REJECTED': result['SAT_REJECTED'],
        'SAT_UNRESOLVED': result['SAT_UNRESOLVED'],
    }


def create_saturation_map_from_fits(input_files, output=None, *, extension=0, block_rows=32,
                          overwrite=False, maximum_iterations=5, final_fallback=None,
                          reject_low_confidence=False, **estimator_options):
    """Read FITS cubes (reads, rows, columns), process blocks, then fill globally.

    All cubes must share shape, read cadence, detector configuration, units and
    preprocessing; shape is checked here, other properties must be checked by
    the caller. FITS section reads support scaled integer images. Only one row
    block of all input cubes is loaded at a time. No per-ramp cubes are saved.
    """
    from contextlib import ExitStack
    paths = [Path(p) for p in input_files]
    if not paths or block_rows < 1:
        raise ValueError('Provide input files and a positive block_rows.')
    with ExitStack() as stack:
        cubes = [stack.enter_context(fits.open(p, memmap=False))[extension] for p in paths]
        shape = cubes[0].shape
        if len(shape) != 3 or any(c.shape != shape for c in cubes):
            raise ValueError('All FITS cubes must share shape (reads, rows, columns).')
        _, ny, nx = shape
        products = None
        for y0 in range(0, ny, block_rows):
            y1 = min(y0 + block_rows, ny)
            ramps = np.stack([np.asarray(c.section[:, y0:y1, :], dtype=np.float32)
                              for c in cubes])
            block = estimate_saturation_block(ramps, **estimator_options)
            if products is None:
                products = {k: np.empty((ny, nx), dtype=block[k].dtype) for k in MAP_KEYS}
            for key in MAP_KEYS:
                products[key][y0:y1] = block[key]
            print(f'Saturation measurements: rows {y0}:{y1} / {ny}', flush=True)
    result = finalize_saturation_map(
        products, minimum_saturation_dn=estimator_options.get('minimum_saturation_dn', 2000.),
        maximum_saturation_dn=estimator_options.get('maximum_saturation_dn'),
        maximum_iterations=maximum_iterations, final_fallback=final_fallback,
        reject_low_confidence=reject_low_confidence)
    if output is not None:
        header = fits.Header()
        header['DROPFRC'] = estimator_options.get('drop_fraction', 0.85)
        header['SUSTWIN'] = estimator_options.get('sustained_window', 3)
        header['NRAMPS'] = len(paths)
        header['FILLITER'] = -1 if maximum_iterations is None else maximum_iterations
        header['FALLBACK'] = str(final_fallback)
        header['REJFLAGS'] = DEFAULT_REJECT_FLAGS | (int(SAT_DQ['LOW_CONFIDENCE']) if reject_low_confidence else 0)
        for key, value in estimator_options.items():
            header.add_history(f'{key}={value}')
        for path in paths:
            header.add_history(f'Input: {path}')
        writer_options = dict(estimator_options)
        writer_options.update(block_rows=block_rows, maximum_iterations=maximum_iterations,
                              fallback_saturation_dn=final_fallback,
                              reject_low_confidence=reject_low_confidence)
        write_saturation_fits(result, output, header=header, filenames=paths,
                              calibration_options=writer_options, overwrite=overwrite)
    return result


###################plotting################################################
def plot_sat(
    result, *, title='Saturation calibration', bins=100,
    figsize=(15, 11), map_percentiles=(1, 99), log_counts=True,
    output_path=None, dpi=180):
    """Plot a single 2x2 figure and return (fig, axes).

    Panels: final DN map; accepted raw versus final DN distributions;
    saturation detection fraction; final-map provenance.
    Works with create_saturation_map() and create_saturation_map_from_fits().
    Display limits affect only the DN-map colors, never the data/histograms.
    Each histogram sample represents one detector pixel, not one ramp.
    NaNs are omitted from distributions and displayed gray in detector maps.
    This function does not alter result or interpolate any values.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    required = ('SATURATION', 'SATURATION_RAW', 'SAT_FRACTION',
                'SAT_ORIGINALLY_INVALID', 'SAT_NEIGHBOR_FILLED', 'SAT_GLOBAL_FILLED')
    maps = {key: np.asarray(result[key]) for key in required}
    shape = maps['SATURATION'].shape
    if len(shape) != 2 or min(shape) == 0:
        raise ValueError('SATURATION must be a nonempty 2-D map.')
    if any(value.shape != shape for value in maps.values()):
        raise ValueError('All diagnostic maps must share the detector shape.')
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError('bins must be a positive integer.')
    lo, hi = map_percentiles
    if not (0 <= lo < hi <= 100):
        raise ValueError('map_percentiles must satisfy 0 <= low < high <= 100.')

    final = maps['SATURATION']
    raw = maps['SATURATION_RAW']
    rejected = maps['SAT_ORIGINALLY_INVALID'].astype(bool)
    neighbor = maps['SAT_NEIGHBOR_FILLED'].astype(bool)
    global_fill = maps['SAT_GLOBAL_FILLED'].astype(bool)
    finite_final = np.isfinite(final)
    accepted_raw = np.isfinite(raw) & ~rejected
    raw_values = raw[accepted_raw]
    final_values = final[finite_final]
    npixels = final.size

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#d0d0d0')
    vmin, vmax = (0., 1.)
    if final_values.size:
        vmin, vmax = np.percentile(final_values, (lo, hi))
        if vmin == vmax:
            padding = max(abs(float(vmin)) * 0.01, 1.)
            vmin, vmax = vmin - padding, vmax + padding
    im = axes[0, 0].imshow(np.ma.masked_invalid(final), origin='lower',
                            interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'Final saturation threshold\nFinite: {final_values.size:,}/{npixels:,} pixels')
    fig.colorbar(im, ax=axes[0, 0], label='Saturation threshold (DN)', extend='both')

    ax = axes[0, 1]
    if raw_values.size or final_values.size:
        minima = [float(v.min()) for v in (raw_values, final_values) if v.size]
        maxima = [float(v.max()) for v in (raw_values, final_values) if v.size]
        left, right = min(minima), max(maxima)
        if left == right:
            padding = max(abs(left) * 0.01, 1.)
            left, right = left - padding, right + padding
        edges = np.linspace(left, right, bins + 1)
        if final_values.size:
            ax.hist(final_values, bins=edges, histtype='step', linewidth=1.8,
                    color='#c96c12', label=f'Final map ({final_values.size:,})')
        if raw_values.size:
            ax.hist(raw_values, bins=edges, histtype='step', linewidth=1.6,
                    linestyle='--', color='#2463a6',
                    label=f'Accepted raw measurements ({raw_values.size:,})')
        if log_counts:
            ax.set_yscale('log')
        ax.legend(fontsize=9)
        if final_values.size:
            p16, median, p84 = np.percentile(final_values, (16, 50, 84))
            ax.axvline(median, color='0.4', linewidth=1, linestyle=':')
            ax.text(.98, .95, f'Final median: {median:,.0f} DN\n'
                    f'16–84%: {p16:,.0f}–{p84:,.0f} DN',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    else:
        ax.text(.5, .5, 'No finite saturation values', ha='center', va='center', transform=ax.transAxes)
    ax.set(title='Per-pixel saturation DN distribution', xlabel='Saturation threshold (DN)',
           ylabel='Number of detector pixels')
    ax.grid(alpha=.2)

    fraction = np.ma.masked_invalid(maps['SAT_FRACTION'])
    im = axes[1, 0].imshow(fraction, origin='lower', interpolation='nearest',
                            cmap=cmap, vmin=0, vmax=1)
    axes[1, 0].set_title('Saturation detection fraction\nNDETECTED / NVALIDRAMPS (before filling)')
    fig.colorbar(im, ax=axes[1, 0], label='Detection fraction')

    # Disjoint categories; nonfinite final values always count as unresolved.
    provenance = np.full(shape, 3, dtype=np.uint8)
    provenance[finite_final & ~rejected] = 0
    provenance[finite_final & neighbor] = 1
    provenance[finite_final & global_fill] = 2
    labels = ('Measured', 'Neighbor filled', 'Global fallback', 'Unresolved')
    colors = ('#287a56', '#3478b7', '#e2a32b', '#d0d0d0')
    counts = np.bincount(provenance.ravel(), minlength=4)
    im = axes[1, 1].imshow(provenance, origin='lower', interpolation='nearest',
                            cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(5)-.5, 4))
    axes[1, 1].set_title('Final-map provenance')
    cb = fig.colorbar(im, ax=axes[1, 1], ticks=range(4))
    cb.ax.set_yticklabels([f'{label}: {count:,} ({100*count/npixels:.2f}%)'
                          for label, count in zip(labels, counts)])
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.set(xlabel='Detector column', ylabel='Detector row')
    fig.suptitle(title, fontsize=16)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    return fig, axes

