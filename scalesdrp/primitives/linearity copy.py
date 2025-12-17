import numpy as np
import os
from tqdm import tqdm
import pkg_resources
from astropy.io import fits
import time
############# create linearity correction coefficents #################
def characterize_detector_linearity_full(
    ramp_cube,
    output_filename="linearity_coeffs_full.fits",
    poly_order=3,
    cutoff_fraction=0.75,
    *,
    min_dynamic_range=50.0,      # DN; below this, treat pixel as flat / unusable
    min_points_for_fit=5,        # min valid reads to attempt a baseline fit
    sat_slope_threshold=3.0,     # DN/read; below this, counts are “not increasing”
    sat_window=4,                # consecutive reads below threshold => saturation
    require_increasing=True,     # reject pixels whose median dY <= 0
    neg_slope_tolerance=-20.0     # DN/read; treat strongly negative ramp as bad
):
    """
    Robust per-pixel linearity characterization.

    For each pixel:
      1) Detect where the ramp stops increasing (saturation / constant region).
      2) Discard saturated reads.
      3) Fit a linear baseline to the low-signal part of the remaining data.
      4) Use the baseline to compute a "linearized" signal L(t).
      5) Split the (M, L) points by a cutoff in L and fit two polynomials:
            COEFFS1: low-signal (M -> L)
            COEFFS2: high-signal (M -> L)

    Handles edge cases:
      - only 2 reads
      - no saturation
      - almost-constant or dead pixels
      - purely linear pixels
      - non-linear pixels without hard saturation

    Parameters
    ----------
    ramp_cube : np.ndarray
        Read cube with shape (n_reads, n_rows, n_cols), in DN.
    output_filename : str
        FITS file to write.
    poly_order : int
        Polynomial order for both segments (COEFFS1/COEFFS2).
    cutoff_fraction : float
        Fraction of max linearized signal L used to split the data for
        two-polynomial fitting.
    min_dynamic_range : float
        Minimum (max - min) DN across the ramp required to attempt a fit.
    min_points_for_fit : int
        Minimum number of *pre-saturation* reads required to fit anything.
    sat_slope_threshold : float
        Threshold in DN/read for declaring that the ramp has effectively
        stopped increasing (used for saturation detection).
    sat_window : int
        Number of consecutive low-derivative reads needed to mark saturation.
    require_increasing : bool
        If True, pixels whose median dY/dt <= 0 are rejected.
    neg_slope_tolerance : float
        If np.nanmin(dY) < neg_slope_tolerance, treat pixel as bad (strongly
        decreasing).

    Returns
    -------
    astropy.io.fits.HDUList
        FITS object with extensions:
          - COEFFS1   (poly_order+1, Ny, Nx)
          - COEFFS2   (poly_order+1, Ny, Nx)
          - CUTOFFS   (Ny, Nx)  cutoff in linearized DN
          - SATURATION (Ny, Nx) estimated saturation level in measured DN
          - SLOPE     (Ny, Nx)  baseline linear slope
          - INTERCEPT (Ny, Nx)  baseline linear intercept
          - GOODPIX   (Ny, Nx)  1 if coefficients are considered valid
    """

    ramp = np.asarray(ramp_cube, dtype=float)
    n_reads, ny, nx = ramp.shape
    reads = np.arange(n_reads, dtype=float)

    # Allocate outputs
    coeffs1 = np.full((poly_order + 1, ny, nx), np.nan, dtype=np.float32)
    coeffs2 = np.full((poly_order + 1, ny, nx), np.nan, dtype=np.float32)
    cutoff_map = np.full((ny, nx), np.nan, dtype=np.float32)
    saturation_map = np.full((ny, nx), np.nan, dtype=np.float32)
    slope_map = np.full((ny, nx), np.nan, dtype=np.float32)
    intercept_map = np.full((ny, nx), np.nan, dtype=np.float32)
    goodpix = np.zeros((ny, nx), dtype=np.uint8)

    def safe_polyfit(x, y, order):
        """Return poly coeffs or None if unfit-able."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if x.size <= order:
            return None
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return None
        try:
            return np.polyfit(x, y, order).astype(np.float32)
        except Exception:
            return None

    for r, c in tqdm(np.ndindex(ny, nx), total=ny * nx, desc="Linearity per pixel"):
        y_all = ramp[:, r, c]

        # --- Finite check ---
        mask_fin = np.isfinite(y_all)
        if not np.any(mask_fin):
            continue

        t = reads[mask_fin]
        y = y_all[mask_fin]
        if y.size < 2:
            # Not enough data for anything
            continue

        # --- Basic dynamic range / flatness check ---
        dr = np.nanmax(y) - np.nanmin(y)
        if dr < min_dynamic_range:
            # Essentially constant / dead / not useful
            continue

        # --- Derivatives for classification ---
        dy = np.diff(y)
        if dy.size == 0:
            continue

        # Strongly decreasing? (bad pixel)
        neg_mask = dy < neg_slope_tolerance
        if np.mean(neg_mask) > 0.3:
            continue

        if require_increasing and np.nanmedian(dy) <= 0:
            # user: "No need to include pixels where count not increasing with time"
            continue

        # --- Step 1: saturation / constant region detection ---
        sat_index = None
        if dy.size >= sat_window:
            for i in range(dy.size - sat_window + 1):
                window_dy = dy[i:i + sat_window]
                # All derivatives below threshold → plateau / saturation
                if np.all(window_dy < sat_slope_threshold):
                    sat_index = i + 1  # plateau begins at read i+1
                    break

        if sat_index is not None:
            # Everything from sat_index onward is "saturated/constant"
            t_valid = t[:sat_index]
            y_valid = y[:sat_index]
            saturation_map[r, c] = y[sat_index]
        else:
            # No detected saturation; use all reads
            t_valid = t
            y_valid = y
            saturation_map[r, c] = y[-1]

        if y_valid.size < min_points_for_fit:
            # Not enough data before saturation to fit reliably
            continue

        # --- Step 2: fit linear baseline using low-signal part ---
        # choose low-signal region as y below some fraction of max
        max_y_valid = np.nanmax(y_valid)
        lin_cut = 0.25 * max_y_valid  # 25% of max as "safe linear"
        lin_mask = y_valid <= lin_cut
        if np.count_nonzero(lin_mask) < 2:
            # fallback: use first few reads
            n_lin = min(10, y_valid.size)
            t_lin = t_valid[:n_lin]
            y_lin = y_valid[:n_lin]
        else:
            t_lin = t_valid[lin_mask]
            y_lin = y_valid[lin_mask]

        # require at least 2 points
        if t_lin.size < 2:
            continue

        a, b = np.polyfit(t_lin, y_lin, 1)  # slope, intercept

        # sanity: if slope is tiny or negative, skip
        if a <= 0:
            continue

        slope_map[r, c] = a
        intercept_map[r, c] = b

        # --- Step 3: compute linearized signal L(t) for all valid points ---
        L_valid = a * t_valid + b

        # --- Step 4: choose cutoff and split into two regions ---
        L_max = np.nanmax(L_valid)
        if not np.isfinite(L_max):
            continue

        cutoff_L = cutoff_fraction * L_max
        cutoff_map[r, c] = cutoff_L

        low_mask = L_valid <= cutoff_L
        high_mask = ~low_mask

        # (M, L) points
        M_low,  L_low  = y_valid[low_mask],  L_valid[low_mask]
        M_high, L_high = y_valid[high_mask], L_valid[high_mask]

        # --- Step 5: polynomial fits M -> L ---
        p1 = safe_polyfit(M_low,  L_low,  poly_order)
        p2 = safe_polyfit(M_high, L_high, poly_order)

        # if one of the segments has too few points, we can:
        #   - fit a single polynomial to all data and assign it to COEFFS1,
        #   - leave COEFFS2 as NaN
        if p1 is None and p2 is None:
            # try global fit
            p_global = safe_polyfit(y_valid, L_valid, poly_order)
            if p_global is None:
                continue
            coeffs1[:, r, c] = p_global
        else:
            if p1 is not None:
                coeffs1[:, r, c] = p1
            if p2 is not None:
                coeffs2[:, r, c] = p2

        # mark as good
        goodpix[r, c] = 1

    # ---------- Pack FITS ----------
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['COMMENT'] = "Per-pixel non-linearity coefficients (measured->linear)"

    hdul = fits.HDUList([
        primary_hdu,
        fits.ImageHDU(coeffs1,      name="COEFFS1"),
        fits.ImageHDU(coeffs2,      name="COEFFS2"),
        fits.ImageHDU(cutoff_map,   name="CUTOFFS"),
        fits.ImageHDU(saturation_map, name="SATURATION"),
        fits.ImageHDU(slope_map,    name="SLOPE"),
        fits.ImageHDU(intercept_map, name="INTERCEPT"),
        fits.ImageHDU(goodpix,      name="GOODPIX"),
    ])

    hdul.writeto(output_filename, overwrite=True)
    return hdul

############### linearity correction #############################################

# Data quality flags (matching your style)
DQ_FLAGS = {
    'GOOD': 0,
    'DO_NOT_USE': 1,
    'SATURATED': 2,
    'NO_LIN_CORR': 1024,
}

def _polyval_stack(coeffs_desc, x3d):
    """
    Evaluate polynomials with Horner’s method.
    coeffs_desc: (P+1, R, C)
    x3d:         (N, R, C)
    returns:     (N, R, C)
    """
    P, R, C = coeffs_desc.shape
    y = np.broadcast_to(coeffs_desc[0], x3d.shape).astype(np.float32)
    for j in range(1, P):
        y = y * x3d + coeffs_desc[j]
    return y

def create_group_dq(science_ramp, sat_map):
    N, H, W = science_ramp.shape
    gdq = np.zeros((N, H, W), dtype=np.uint16)
    sat = science_ramp >= sat_map[None, :, :]

    # mark immediately
    gdq[sat] |= DQ_FLAGS["SATURATED"]

    # propagate saturation forward
    for i in range(1, N):
        prev = (gdq[i-1] & DQ_FLAGS["SATURATED"]) != 0
        gdq[i, prev] |= DQ_FLAGS["SATURATED"]

    return gdq

def enforce_cutoff_in_groupdq(group_dq_raw, cutoff_read_map, min_cutoff_read=3):
    """
    Ensure that reads >= cutoff_read_map[row, col] are marked SATURATED,
    but only if the cutoff:
      - occurs at a reasonable read index (>= min_cutoff_read), and
      - is earlier than the existing physical saturation.
    This avoids wiping out entire columns when cutoff_read_map = 0 or 1.
    """
    N, H, W = group_dq_raw.shape
    out = group_dq_raw.copy()

    sat_flag = DQ_FLAGS["SATURATED"]

    for r in range(H):
        for c in range(W):
            k = int(cutoff_read_map[r, c])

            # find first existing saturation from slope/plateau detection
            sat_idx_arr = np.where((group_dq_raw[:, r, c] & sat_flag) != 0)[0]
            first_sat = sat_idx_arr[0] if sat_idx_arr.size > 0 else N

            # only add extra saturation if:
            #  - cutoff index is valid
            #  - cutoff is not unrealistically early
            #  - cutoff is before the already-known saturation
            if 0 <= k < N and k >= min_cutoff_read and k < first_sat:
                out[k:, r, c] |= sat_flag
    return out

def apply_linearity_correction_twopart_final(
    science_ramp,
    group_dq,
    linearity_hdul,
    *,
    slope_tolerance=0.20,   # |a1 - 1| > 0.2 → bad coeffs
    coeff2_max=1e7          # max |coeffs2| > 1e7 → bad coeffs
):
    """
    Apply per-pixel two-segment polynomial linearity correction, with
    coefficient sanity checks.

    Inputs
    ------
    science_ramp  : (N,H,W) float32
    group_dq      : (N,H,W) uint16
    linearity_hdul: HDUList with COEFFS1, COEFFS2, CUTOFFS, GOODPIX (optional)

    Returns
    -------
    corrected_ramp : (N,H,W) float32
    pixel_dq       : (H,W) uint16  (NO_LIN_CORR flagged)
    cutoff_read_map: (H,W) int16   (first read where M > cutoff DN; N-1 if never)
    """

    science_ramp = science_ramp.astype(np.float32, copy=False)
    N, H, W = science_ramp.shape

    # ---------- Load calibration arrays ----------
    coeffs1 = linearity_hdul["COEFFS1"].data.astype(np.float32)  # (P+1,H,W)
    coeffs2 = linearity_hdul["COEFFS2"].data.astype(np.float32)  # (P+1,H,W)
    cutoffs = linearity_hdul["CUTOFFS"].data.astype(np.float32)  # (H,W)
    if "GOODPIX" in linearity_hdul:
        goodpix = linearity_hdul["GOODPIX"].data.astype(bool)    # (H,W)
    else:
        goodpix = np.ones((H, W), dtype=bool)

    P = coeffs1.shape[0]
    if coeffs2.shape != coeffs1.shape:
        raise ValueError("COEFFS1 and COEFFS2 shapes do not match.")

    # ---------- Initialize pixel DQ ----------
    pixel_dq = np.zeros((H, W), dtype=np.uint16)

    # ---------- Sanity checks on coefficients ----------

    # 1) basic non-finite / GOODPIX mask
    nan_bad = (
        ~np.isfinite(coeffs1).any(axis=0) |
        ~np.isfinite(coeffs2).any(axis=0) |
        ~np.isfinite(cutoffs)
    )
    goodpix_bad = ~goodpix

    # 2) COEFFS1 linear term should be close to 1.0
    a1 = coeffs1[-2]  # (H,W)
    slope_bad = np.abs(a1 - 1.0) > slope_tolerance

    # 3) COEFFS2 coefficients should not be insane
    high2 = np.max(np.abs(coeffs2), axis=0)  # (H,W)
    coeff2_bad = high2 > coeff2_max

    # Combined bad mask
    bad = nan_bad | goodpix_bad | slope_bad | coeff2_bad

    if np.any(bad):
        pixel_dq[bad] |= DQ_FLAGS["NO_LIN_CORR"]

        # Identity polynomial y = x in DESC order: [0,...,0,1,0]
        ident = np.zeros(P, dtype=np.float32)
        if P >= 2:
            ident[-2] = 1.0  # linear term
            ident[-1] = 0.0  # constant
        else:
            ident[0] = 1.0   # degenerate safeguard

        coeffs1[:, bad] = ident[:, None]
        coeffs2[:, bad] = ident[:, None]

    # ---------- Evaluate polynomials ----------
    corrected1 = _polyval_stack(coeffs1, science_ramp)
    corrected2 = _polyval_stack(coeffs2, science_ramp)

    below_cut = science_ramp <= cutoffs[None, :, :]
    corrected_ramp = np.where(below_cut, corrected1, corrected2)

    # ---------- Preserve saturated samples from RAW group_dq ----------
    sat_mask = (group_dq & DQ_FLAGS["SATURATED"]) != 0
    corrected_ramp = np.where(sat_mask, science_ramp, corrected_ramp)

    # ---------- Build cutoff-read map ----------
    cutoff_read_map = np.full((H, W), fill_value=N-1, dtype=np.int16)

    for r in range(H):
        m_row = science_ramp[:, r, :]     # (N,W)
        c_row = cutoffs[r, :]             # (W,)
        over = m_row > c_row[None, :]     # (N,W)
        any_over = over.any(axis=0)
        if not np.any(any_over):
            continue
        idx_first = np.argmax(over[:, any_over], axis=0)
        cutoff_read_map[r, any_over] = idx_first.astype(np.int16)

    return corrected_ramp.astype(np.float32, copy=False), pixel_dq, cutoff_read_map


def create_saturation_map_by_slope(
    science_ramp,
    slope_threshold=1.0,
    window=5,
    min_flat_reads=5,
    min_dynamic_range=200.0,
):
    """
    Detect saturation/plateau by derivative, robustly.

    A pixel is considered saturated starting at the first read where:
      - the smoothed derivative remains < slope_threshold
        for at least `min_flat_reads` consecutive points, AND
      - the signal has increased by at least `min_dynamic_range`
        DN above the first read.

    Returns:
      sat_map : (H, W) float32
        DN level at which saturation begins; np.inf if no saturation found.
    """
    N, H, W = science_ramp.shape
    sat_map = np.full((H, W), np.inf, dtype=np.float32)

    kernel = np.ones(window, dtype=np.float32) / window

    for r in range(H):
        for c in range(W):
            y = science_ramp[:, r, c].astype(np.float32)
            if not np.any(np.isfinite(y)):
                continue

            # Require some dynamic range; otherwise ignore
            dr = float(np.nanmax(y) - np.nanmin(y))
            if dr < min_dynamic_range:
                continue

            dy = np.diff(y)
            if dy.size < window + min_flat_reads:
                continue

            sm = np.convolve(dy, kernel, mode='valid')  # length N-1-window+1

            # Find contiguous runs where sm < threshold
            flat = sm < slope_threshold
            if not np.any(flat):
                continue

            # Find first run of length >= min_flat_reads
            run_start = None
            run_len = 0
            found = False
            for i, val in enumerate(flat):
                if val:
                    if run_start is None:
                        run_start = i
                        run_len = 1
                    else:
                        run_len += 1
                    if run_len >= min_flat_reads:
                        # candidate plateau begins at idx = run_start
                        idx = run_start + window  # map sm index back to y index
                        # ensure plateau is not at the very start
                        if idx < 3:
                            # ignore – too early to be true saturation
                            break
                        # ensure we've risen enough above the first read
                        if (y[idx] - y[0]) >= min_dynamic_range:
                            sat_map[r, c] = y[idx]
                        found = True
                        break
                else:
                    run_start = None
                    run_len = 0

            if not found:
                # leave sat_map[r,c] as np.inf (no good plateau found)
                continue

    return sat_map

def run_linearity_workflow(
    science_ramp,
    linearity_file,
    *,
    slope_tolerance=0.20,
    coeff2_max=1e7,
):
    """
    Full non-linearity workflow:

      1. Build saturation map from RAW ramp.
      2. Build initial GROUPDQ from RAW saturation.
      3. Apply non-linearity correction (with coefficient sanity checks).
      4. Enforce cutoff-read map on GROUPDQ.
      5. If SATURATION HDU exists in calibration, enforce saturation on
         corrected ramp and revert those reads to raw counts.

    Returns
    -------
    corrected_ramp : (N,H,W) float32
    pixel_dq       : (H,W)   uint16
    group_dq       : (N,H,W) uint16  (final GROUPDQ)
    cutoff_read_map: (H,W)   int16
    sat_map        : (H,W)   float32 (RAW saturation DN from slope)
    group_dq_raw   : (N,H,W) uint16  (pre-cutoff GROUPDQ)
    """
    t1=time.time()
    # 1. Build saturation map from RAW ramp
    sat_map = create_saturation_map_by_slope(science_ramp)

    # 2. Build initial DQ (based on RAW saturation)
    group_dq_raw = create_group_dq(science_ramp, sat_map)

    calib_path = pkg_resources.resource_filename('scalesdrp', 'calib/')

    sat_dn = None  # will be filled if SATURATION HDU exists

    # 3. Apply non-linearity correction + pull SATURATION while file is open
    with fits.open(calib_path + linearity_file) as hdul:
        corrected_ramp, pixel_dq, cutoff_read_map = apply_linearity_correction_twopart_final(
            science_ramp,
            group_dq_raw,
            hdul,
            slope_tolerance=slope_tolerance,
            coeff2_max=coeff2_max,
        )

        # Copy SATURATION HDU to a plain numpy array (if present)
        if "SATURATION" in hdul:
            sat_dn = hdul["SATURATION"].data.astype(np.float32)  # (H, W)

    # 4. Enforce cutoff read index on group DQ (using RAW cutoff)
    group_dq = enforce_cutoff_in_groupdq(group_dq_raw, cutoff_read_map)

    # 5. Enforce saturation using *corrected* ramp + sat_dn, if available
    if sat_dn is not None:
        # shape: corrected_ramp (N,H,W); sat_dn (H,W)
        corr_sat_mask = corrected_ramp >= sat_dn[None, :, :]

        # Flag these reads as SATURATED
        group_dq |= np.where(
            corr_sat_mask,
            DQ_FLAGS["SATURATED"],
            0
        ).astype(np.uint16)

        # Keep RAW counts where we’ve declared saturation
        corrected_ramp = np.where(corr_sat_mask, science_ramp, corrected_ramp)
    t2=time.time()
    print(f"Linearity correction completed in {t2 - t1:.3f} seconds.")
    return (
        corrected_ramp.astype(np.float32, copy=False),
        pixel_dq,
        group_dq,
        cutoff_read_map,
        sat_map,
        group_dq_raw,
    )

