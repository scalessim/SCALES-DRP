"""
Automatic per-pixel saturation-threshold estimation

The detector is processed in row blocks. Input ramps should have shape
(n_ramps, n_reads, nrows, nx). Currently the input is ACN and 1/f, and linearity corrected.
The main output maps have shape (nrows, nx), 2D saturation map
"""
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
    first_index[~detected] = -1

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
        dtype=np.uint16,
    )

    with np.errstate(invalid="ignore"):
        reference_difference = np.nanmedian(
            reference_region,
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
    minimum_detection_fraction=0.5,
    maximum_ramp_scatter_fraction=0.10,
    skip_initial_reads=2,
):
    """
    Estimate a per-pixel saturation threshold from calibration ramps.

    Saturation is identified when the ramp difference drops below a
    fraction of its normal early-ramp value for a sustained number of
    consecutive differences.

    Parameters
    ----------
    ramps : ndarray
        Ramp data with shape

            (n_ramps, n_reads, nrows, nx)

    drop_fraction : float, optional
        A pixel is considered to be entering saturation when

            current_difference
            <
            drop_fraction * reference_difference

    sustained_window : int, optional
        Number of consecutive low differences required.

    start_difference : int, optional
        First difference used to estimate the early-ramp reference signal.

    n_reference_differences : int, optional
        Number of early differences used to estimate normal ramp behavior.

    minimum_reference_differences : int, optional
        Minimum number of finite early differences required.

    minimum_reference_signal : float, optional
        Minimum positive early-ramp difference required for saturation
        detection.

    minimum_saturation_dn : float, optional
        Saturation estimates below this value are flagged.

    maximum_saturation_dn : float or None, optional
        Saturation estimates above this value are flagged.

        Use None when there is no explicit upper limit.

    fallback_saturation_dn : float or None, optional
        Fallback saturation threshold used when saturation cannot be
        determined.

        If None, unresolved pixels remain NaN.

    minimum_valid_ramps : int, optional
        Minimum number of ramps with valid saturation detections.

    minimum_detection_fraction : float, optional
        Minimum fraction of valid ramps that should detect saturation.
        Pixels below this fraction are marked as low-confidence.

    maximum_ramp_scatter_fraction : float, optional
        Maximum allowed robust scatter between per-ramp saturation
        estimates, expressed relative to the median saturation value.

    skip_initial_reads : int, optional
        Saturation detections before this read index are considered
        suspicious and flagged as early saturation.

    Returns
    -------
    result : dict
        Dictionary containing block-sized detector maps.

        SATURATION
            Final per-pixel saturation threshold, shape ``(nrows, nx)``.

        SAT_QUALITY
            Saturation DQ map, shape ``(nrows, nx)``.

        SAT_READ
            First saturated read index, shape ``(nrows, nx)``.

        REF_DIFF
            Median early-ramp reference difference, shape ``(nrows, nx)``.

        NDETECTED
            Number of ramps with detected saturation.

        NVALIDRAMPS
            Number of ramps with valid reference differences.

        SAT_SCATTER
            Robust scatter among per-ramp saturation estimates.

        SAT_FRACTION
            Fraction of valid ramps with detected saturation.

        SATURATION_PER_RAMP
            Individual saturation estimates with shape
            ``(n_ramps, nrows, nx)``.

        SAT_READ_PER_RAMP
            Individual saturation-read indices with shape
            ``(n_ramps, nrows, nx)``.
    """

    ramps = np.asarray(ramps)

    if ramps.ndim != 4:
        raise ValueError(
            "ramps must have shape "
            "(n_ramps, n_reads, nrows, nx)."
        )

    nramps, nreads, nrows, nx = ramps.shape

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
    ).astype(np.int16)

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
        dtype=np.uint16,
    )
    #Counts how many ramps had a usable early-ramp reference difference for each pixel.
    n_valid_ramps = np.sum(
        valid_reference,
        axis=0,
        dtype=np.uint16,
    )
    #Combines the ramp-specific saturation DN estimates using the median.
    with np.errstate(invalid="ignore"):
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
        dtype=np.int16,
    )
    #Identifies pixels where at least one valid saturation-read estimate exists.
    valid_read_median = np.isfinite(saturation_read_float)

    saturation_read[valid_read_median] = np.rint(
        saturation_read_float[valid_read_median]
    ).astype(np.int16)

    # --------------------------------------------------------------
    # Robust ramp-to-ramp scatter
    # --------------------------------------------------------------
    #Calculates the median saturation value but retains the ramp axis as a length-one dimension.
    with np.errstate(invalid="ignore"):
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

    saturation_scatter = (
        1.4826 * saturation_mad
    ).astype(np.float32)
    #saturation fraction map initialized
    saturation_fraction = np.zeros(
        (nrows, nx),
        dtype=np.float32,
    )

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

    no_finite_reference = ~np.isfinite(reference_difference)

    sat_quality[no_finite_reference] |= SAT_DQ["NONFINITE"]

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

    sat_quality[low_confidence] |= SAT_DQ["LOW_CONFIDENCE"]

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

    early_saturation = (
        saturation_read >= 0
    ) & (
        saturation_read < skip_initial_reads
    )

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

    # --------------------------------------------------------------
    # Repair missing or unrealistic saturation values
    # --------------------------------------------------------------

    (
        saturation,
        neighbor_filled,
        global_filled,
        originally_invalid_saturation,
    ) = fill_invalid_saturation_neighbors(
        saturation,
        minimum_saturation_dn=minimum_saturation_dn,
        maximum_saturation_dn=maximum_saturation_dn,
        maximum_iterations=None,
        final_fallback=(
            fallback_saturation_dn
            if fallback_saturation_dn is not None
            else "median"
        ),
    )

    sat_quality[neighbor_filled] |= (
        SAT_DQ["FALLBACK_USED"]
        | SAT_DQ["NEIGHBOR_FILL_USED"]
    )

    sat_quality[global_filled] |= (
        SAT_DQ["FALLBACK_USED"]
        | SAT_DQ["GLOBAL_FILL_USED"]
    )

    # --------------------------------------------------------------
    # Return all block products
    # --------------------------------------------------------------

    return {
        "SATURATION": saturation,
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
        "SAT_NEIGHBOR_FILLED": neighbor_filled,
        "SAT_GLOBAL_FILLED": global_filled,
        "SAT_ORIGINALLY_INVALID": originally_invalid_saturation,
    }

def fill_invalid_saturation_neighbors(
    saturation,
    *,
    minimum_saturation_dn=None,
    maximum_saturation_dn=None,
    maximum_iterations=None,
    final_fallback="median",
):
    """
    Replace invalid saturation values using neighboring valid pixels.

    Invalid pixels are filled iteratively using the median of their
    available 8-connected neighbors. Repeating the procedure allows valid
    saturation values to propagate into larger invalid regions.

    Parameters
    ----------
    saturation : array-like
        Two-dimensional saturation map.

    minimum_saturation_dn : float or None, optional
        Values below this threshold are considered invalid.

    maximum_saturation_dn : float or None, optional
        Values above this threshold are considered invalid.

    maximum_iterations : int or None, optional
        Maximum number of neighbor-propagation iterations. If None, enough
        iterations are allowed to span the largest image dimension.

    final_fallback : {"median", None} or float, optional
        Action for pixels that remain unresolved after neighbor filling.

        ``"median"``
            Use the median of all valid saturation values.

        float
            Use the supplied scalar saturation threshold.

        None
            Leave unresolved pixels as NaN.

    Returns
    -------
    filled : ndarray
        Saturation map after filling, with dtype float32.

    neighbor_filled : ndarray of bool
        Pixels filled from neighboring saturation values.

    global_filled : ndarray of bool
        Pixels filled using the final global fallback.

    originally_invalid : ndarray of bool
        Pixels that were nonfinite or outside the accepted range before
        filling.
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

    if not np.any(valid):
        raise ValueError(
            "The saturation map contains no finite values within the "
            "requested limits, so neighbor filling cannot be performed."
        )

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

        with np.errstate(invalid="ignore"):
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
        valid_values = filled[np.isfinite(filled)]

        if isinstance(final_fallback, str):
            if final_fallback.lower() != "median":
                raise ValueError(
                    "final_fallback must be 'median', a scalar, or None."
                )

            fallback_value = float(np.nanmedian(valid_values))
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

