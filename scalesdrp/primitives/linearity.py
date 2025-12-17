import numpy as np
from astropy.io import fits
from tqdm import tqdm
import os
import pkg_resources
############## create polynomial #############################

def characterize_detector_linearity_full(
    ramp_cube,
    output_filename="linearity_coeffs_full.fits",
    low_poly_order=1,          # low segment: keep strictly linear by default
    high_poly_order=3,         # high segment: allow curvature
    cutoff_fraction=0.75,
    *,
    # basic quality cuts
    min_dynamic_range=50.0,    # DN; below this, treat pixel as flat / unusable
    min_points_for_fit=5,      # min valid reads before saturation to attempt ANY fit
    require_increasing=True,   # reject pixels whose median dY <= 0
    neg_slope_tolerance=-20.0, # DN/read; treat strongly negative ramps as bad

    # saturation detection
    sat_window=4,              # consecutive low-derivative reads needed to mark saturation
    sat_frac=0.05,             # sat threshold = sat_frac * median(dy); set to None to disable

    # baseline refinement via deviation from line
    initial_frac=0.25,         # use up to 25% of pre-sat signal for provisional line
    baseline_dev_threshold=0.05,   # fractional deviation that defines non-linear break
    baseline_safety_fraction=0.90, # use this fraction of break index as safe baseline region
    min_pred_abs=1.0,          # guard for very small predicted L (to avoid crazy frac_dev)

    # polynomial sanity limits
    low_slope_min=0.5,         # acceptable range for COEFFS1 linear term
    low_slope_max=1.5,
    high_coeff_limit=1e7       # max |COEFFS2| before we reject that fit
):
    """
    Per-pixel linearity characterization.

    For each pixel:
      1) Detect where the ramp stops increasing (saturation / plateau)
         using derivatives dy and a sliding window.
      2) Discard saturated reads; keep only pre-saturation data.
      3) Fit a provisional linear baseline to a very low-signal core
         (up to `initial_frac * max(y_valid)`).
      4) Use this provisional line to compute fractional deviation of all
         pre-saturation points, and find the first read where the deviation
         exceeds `baseline_dev_threshold`.
      5) Define the final baseline region as the reads up to
         `baseline_safety_fraction * break_index`, and refit a line there.
      6) Use this final baseline line to define a linearized signal L(t).
      7) Split (M, L) into two regions using a deviation-based cutoff in L,
         with a fallback based on `cutoff_fraction` of the sequence length.
      8) Fit two polynomials:
            - COEFFS1: low-signal (M -> L), order = low_poly_order
            - COEFFS2: high-signal (M -> L), order = high_poly_order
         enforcing extra sanity checks on slopes and coefficient magnitudes.

    Output FITS HDUs:
      - COEFFS1        (low_poly_order+1, Ny, Nx)
      - COEFFS2        (high_poly_order+1, Ny, Nx)
      - CUTOFFS        (Ny, Nx)  cutoff in linearized DN #switch btw polynomial 1&2
      - SATURATION     (Ny, Nx)  estimated saturation level in measured DN
      - SLOPE          (Ny, Nx)  baseline linear slope
      - INTERCEPT      (Ny, Nx)  baseline linear intercept
      - GOODPIX        (Ny, Nx)  1 if coefficients are considered valid
    """

    ramp = np.asarray(ramp_cube, dtype=float)
    n_reads, ny, nx = ramp.shape
    reads = np.arange(n_reads, dtype=float)

    # Allocate outputs
    coeffs1        = np.full((low_poly_order  + 1, ny, nx), np.nan, dtype=np.float32)
    coeffs2        = np.full((high_poly_order + 1, ny, nx), np.nan, dtype=np.float32)
    cutoff_map     = np.full((ny, nx), np.nan, dtype=np.float32) #DN at which fitting switches
    saturation_map = np.full((ny, nx), np.nan, dtype=np.float32)
    slope_map      = np.full((ny, nx), np.nan, dtype=np.float32)
    intercept_map  = np.full((ny, nx), np.nan, dtype=np.float32)
    goodpix        = np.zeros((ny, nx), dtype=np.uint8)

    def safe_polyfit(x, y, order):
        """Return polynomial coeffs (descending order) or None if unfit-able."""
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

    # Minimum points for each segment (stricter than global min_points_for_fit)
    min_low_points  = max(low_poly_order  + 2, 5)
    min_high_points = max(high_poly_order + 2, 5)

    # ---------- Per-pixel loop ----------
    for r, c in tqdm(np.ndindex(ny, nx), total=ny * nx, desc="Linearity per pixel"):
        y_all = ramp[:, r, c]

        # 1) Basic finite check
        mask_fin = np.isfinite(y_all)
        if not np.any(mask_fin):
            continue

        t = reads[mask_fin]
        y = y_all[mask_fin]
        if y.size < 2:
            continue

        # 2) Dynamic range check
        dr = np.nanmax(y) - np.nanmin(y)
        if dr < min_dynamic_range:
            continue

        # 3) Derivatives
        dy = np.diff(y)
        if dy.size == 0:
            continue

        # 3a) Strongly negative jumps → unstable pixel
        neg_mask = dy < neg_slope_tolerance
        if np.mean(neg_mask) > 0.3:
            continue

        # 3b) Require overall increasing ramp if requested
        if require_increasing and np.nanmedian(dy) <= 0:
            continue

        # 4) Saturation / plateau detection
        sat_index = None
        if sat_frac is not None and dy.size >= sat_window:
            baseline_slope_est = np.nanmedian(dy)
            if baseline_slope_est > 0:
                sat_thresh = sat_frac * baseline_slope_est
                for i in range(dy.size - sat_window + 1):
                    window_dy = dy[i:i + sat_window]
                    if np.all(window_dy < sat_thresh):
                        sat_index = i + 1  # plateau begins at read i+1
                        break

        # 5) Pre-saturation valid data
        if sat_index is not None:
            t_valid = t[:sat_index]
            y_valid = y[:sat_index]
            saturation_map[r, c] = y[sat_index]
        else: #otherwise use all points
            t_valid = t
            y_valid = y
            saturation_map[r, c] = y[-1]

        if y_valid.size < min_points_for_fit:
            continue

        # ---------- Baseline fit (two-stage) ----------

        # 5a) Provisional line from a low-signal core (<= initial_frac * max)
        max_y_valid = np.nanmax(y_valid)
        if not np.isfinite(max_y_valid):
            continue

        initial_cut = initial_frac * max_y_valid # 25% of max
        core_mask = y_valid <= initial_cut
        if np.count_nonzero(core_mask) >= 2:
            t_lin0 = t_valid[core_mask]
            y_lin0 = y_valid[core_mask]
        else:
            n_lin0 = min(10, y_valid.size)
            t_lin0 = t_valid[:n_lin0]
            y_lin0 = y_valid[:n_lin0]

        if t_lin0.size < 2:
            continue

        a0, b0 = np.polyfit(t_lin0, y_lin0, 1)
        if a0 <= 0:
            continue

        # 5b) Fractional deviation across all pre-sat points
        #min_pred_abs=1 will take care of division by small values
        L_prov = a0 * t_valid + b0
        denom_dev = np.maximum(np.abs(L_prov), min_pred_abs)
        frac_dev = np.abs((y_valid - L_prov) / denom_dev)

        # 5c) First read where non-linearity exceeds threshold 5%
        over = np.where(frac_dev > baseline_dev_threshold)[0]
        #neaver exceeds 5%, then low limit set to the end of the read
        if over.size == 0:
            idx_break = y_valid.size
        else:
            idx_break = int(over[0])

        # 5d) Final baseline region as safety fraction of break index
        if idx_break <= 2: #fallback to single linear fit of 10 reads if idx_break<2
            idx_baseline_end = min(10, y_valid.size)
        else: #90% of the stright line
            idx_baseline_end = int(baseline_safety_fraction * idx_break)
            idx_baseline_end = max(2, min(idx_baseline_end, y_valid.size))

        t_lin = t_valid[:idx_baseline_end]
        y_lin = y_valid[:idx_baseline_end]
        if t_lin.size < 2:
            continue

        # 5e) Final baseline slope & intercept
        a, b = np.polyfit(t_lin, y_lin, 1)
        if a <= 0:
            continue

        slope_map[r, c]     = a
        intercept_map[r, c] = b

        # 6) Linearized signal for all pre-sat points
        L_valid = a * t_valid + b

        # 7) Cutoff in L space for low/high segments
        L_max = np.nanmax(L_valid)
        if not np.isfinite(L_max):
            continue

        resid = y_valid - L_valid
        denom_L = np.maximum(np.abs(L_valid), min_pred_abs)
        rel_err = np.abs(resid / denom_L)

        # deviation-based cutoff index (5%)
        over2 = np.where(rel_err > baseline_dev_threshold)[0]
        if over2.size:
            cutoff_index = int(over2[0])
        else:
            # fallback: use a fraction of the sequence length
            cutoff_index = int(cutoff_fraction * len(L_valid))

        # Clamp cutoff index to avoid extremely early or late splits
        #at least 5 points in the lower polynomial
        #at least 1 for higher order
        nL = len(y_valid)
        cutoff_index = max(5, cutoff_index)
        cutoff_index = min(nL - 1, cutoff_index)

        cutoff_M = y_valid[cutoff_index]
        cutoff_map[r, c] = cutoff_M

        low_mask  = L_valid <= cutoff_M
        high_mask = ~low_mask

        M_low,  L_low  = y_valid[low_mask],  L_valid[low_mask]
        M_high, L_high = y_valid[high_mask], L_valid[high_mask]

        # 8) Polynomial fits M -> L with segment-specific point requirements
        p1 = None
        p2 = None

        # --- Low segment ---
        if M_low.size >= min_low_points:
            p1 = safe_polyfit(M_low, L_low, low_poly_order)

            # enforce near-identity slope at low DN
            if p1 is not None and low_poly_order >= 1:
                # Poly coeffs: [a_n, ..., a2, a1, a0]
                a1_low = p1[-2]  # linear term
                if (not np.isfinite(a1_low) or
                        a1_low < low_slope_min or
                        a1_low > low_slope_max): #0.5<a1<1.5
                    p1 = None

        # --- High segment ---
        if M_high.size >= min_high_points:
            p2 = safe_polyfit(M_high, L_high, high_poly_order)
            if p2 is not None:
                if np.max(np.abs(p2)) > high_coeff_limit: #above 1e7
                    p2 = None

        # If both fail, try a single global fit as last resort
        if p1 is None and p2 is None:
            p_global = safe_polyfit(y_valid, L_valid, low_poly_order)
            if p_global is None:
                continue
            coeffs1[:, r, c] = p_global
        else:
            if p1 is not None:
                coeffs1[:, r, c] = p1
            if p2 is not None:
                coeffs2[:, r, c] = p2

        goodpix[r, c] = 1

    # ---------- Pack FITS ----------
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['COMMENT'] = (
        "Per-pixel non-linearity coefficients (measured->linear) "
        "with deviation-based baseline fitting and segment sanity checks."
    )

    hdul = fits.HDUList([
        primary_hdu,
        fits.ImageHDU(coeffs1,        name="COEFFS1"),
        fits.ImageHDU(coeffs2,        name="COEFFS2"),
        fits.ImageHDU(cutoff_map,     name="CUTOFFS"),
        fits.ImageHDU(saturation_map, name="SATURATION"),
        fits.ImageHDU(slope_map,      name="SLOPE"),
        fits.ImageHDU(intercept_map,  name="INTERCEPT"),
        fits.ImageHDU(goodpix,        name="GOODPIX"),
    ])

    hdul.writeto(output_filename, overwrite=True)
    return hdul

################ how to execute ########################################################
#where ramp is a cube of read from where we are creating the coefficients
#linearity_hdul =  characterize_detector_linearity_full(
#    ramp_cube = ramp,
#    output_filename="linearity_coeffs_img_full_order3.fits")


############# linearity correction  #############################################


DQ_FLAGS = {
    'GOOD': 0,
    'DO_NOT_USE': 1,
    'SATURATED': 2,
    'NO_LIN_CORR': 1024,
}


def _polyval_stack(coeffs_desc, x3d):
    """
    coeffs_desc: (P+1, H, W)
    x3d:         (N, H, W) #input read
    returns:     (N, H, W)
    """
    P, H, W = coeffs_desc.shape
    #starting with the highest order polynomial
    y = np.broadcast_to(coeffs_desc[0], x3d.shape).astype(np.float32)
    for j in range(1, P):
        y = y * x3d + coeffs_desc[j]
    return y

def create_saturation_map_by_slope(science_ramp,
                                   slope_threshold=2.0,
                                   window=3):
    """
    Estimate saturation level per pixel using the raw ramp derivative.
    Returns sat_map (H, W) in *measured DN*.

    For each pixel, we:
      - compute dy = diff(y)
      - smooth dy with a boxcar of length `window`
      - find the first index where smoothed dy < slope_threshold
      - take the corresponding y-level as the saturation DN
    """
    N, H, W = science_ramp.shape
    sat_map = np.full((H, W), np.inf, dtype=np.float32)

    kernel = np.ones(window, dtype=np.float32) / float(window)

    for r in range(H):
        for c in range(W):
            y = science_ramp[:, r, c]
            dy = np.diff(y)
            if dy.size < window:
                continue

            sm = np.convolve(dy, kernel, mode="valid")
            flat = np.where(sm < slope_threshold)[0] #first index where sm < slope_threshold
            if flat.size:
                idx = flat[0] + window
                if 0 <= idx < N:
                    sat_map[r, c] = y[idx] #DN value where the pixel stops increasing

    return sat_map


def create_group_dq(science_ramp, sat_map):
    """
    Build GROUPDQ flags using a per-pixel saturation DN map.
    mark all reads from first saturated one as SATURATED
    Parameters
    ----------
    science_ramp : (N, H, W) in measured DN
    sat_map      : (H, W) saturation DN (measured), or +inf for "never"

    Returns
    -------
    gdq : (N, H, W), uint16
        SATURATED bit set from the first read where M >= sat_map onward.
    """
    N, H, W = science_ramp.shape
    gdq = np.zeros((N, H, W), dtype=np.uint16)

    # broadcast sat_map to (N,H,W)
    sat_mask = science_ramp >= sat_map[None, :, :]  # True where measured DN >= sat DN

    # mark immediately
    gdq[sat_mask] |= DQ_FLAGS["SATURATED"]

    # IF THE previous read as saturated, mark all the later ones too
    for i in range(1, N):
        prev_sat = (gdq[i - 1] & DQ_FLAGS["SATURATED"]) != 0 
        gdq[i, prev_sat] |= DQ_FLAGS["SATURATED"]

    return gdq

def enforce_cutoff_in_groupdq(group_dq, cutoff_read_map):
    """
    Ensure that reads >= cutoff_read_map[row, col] are marked SATURATED in GROUPDQ.
    """
    N, H, W = group_dq.shape
    out = group_dq.copy()

    for r in range(H):
        for c in range(W):
            k = cutoff_read_map[r, c]
            if 0 <= k < N:
                out[k:, r, c] |= DQ_FLAGS["SATURATED"]
    return out

def apply_linearity_correction_twopart_final(
    science_ramp,
    group_dq,
    linearity_hdul,
    *,
    coeff2_max=1e8,          # max |COEFFS2| above this => don't use high segment
):
    """
    Apply per-pixel two-segment polynomial linearity correction.

    Calibration file must contain:
      - COEFFS1 : (P1+1, H, W)  low-signal polynomial  (measured -> linear)
      - COEFFS2 : (P2+1, H, W)  high-signal polynomial (measured -> linear)
      - CUTOFFS : (H, W)        *measured* DN where we prefer COEFFS2 over COEFFS1
      - GOODPIX : (H, W) bool   1 = good coefficients (optional, assumed True)
      - (optional) CUTOFF_READ : (H, W) int16
            pre-computed per-pixel read cutoff index (0..N-1)

    Behaviour
    ---------
    * CUTOFFS is **only** used as a threshold in measured DN to decide between
      low vs high polynomial. It is NOT used to compute a read index.

    * The read-index cutoff map is:
        - CUTOFF_READ HDU if present, else
        - first read where GROUPDQ has SATURATED bit set for that pixel;
          if no read is SATURATED, cutoff_read_map = N-1.

    * Pixels with *no usable low segment* (COEFFS1 all-NaN, or cutoff NaN,
      or GOODPIX==False) are:
        - flagged with DQ_FLAGS["NO_LIN_CORR"] in pixel_dq
        - forced to use identity y = x for both segments (no correction).

    * Pixels where the *high* segment looks bad (huge |COEFFS2| or COEFFS2
      all-NaN) but the low segment is fine:
        - are NOT flagged NO_LIN_CORR
        - simply never use COEFFS2; they use COEFFS1 at all counts.

    * Existing SATURATED bits in group_dq are preserved:
        corrected_ramp = original science_ramp where SATURATED is set.

    Parameters
    ----------
    science_ramp : ndarray, shape (N,H,W)
        Input ramp in measured DN.
    group_dq : ndarray, shape (N,H,W), uint16
        GROUPDQ-like flags. SATURATED bits are preserved.
    linearity_hdul : fits.HDUList
        Opened linearity calibration file.

    Returns
    -------
    corrected_ramp : ndarray, (N,H,W), float32
        Linearity-corrected ramp (in "linearized DN").
    pixel_dq       : ndarray, (H,W),   uint16
        Pixel-level DQ flags (e.g. NO_LIN_CORR).
    cutoff_read_map: ndarray, (H,W),   int16
        Per-pixel read-index cutoff:
          - if CUTOFF_READ HDU exists: that value (clipped to [0,N-1])
          - else: first index where GROUPDQ has SATURATED set;
                  if never saturated, N-1.
    """

    # ---------- Shapes ----------
    science_ramp = science_ramp.astype(np.float32, copy=False)
    N, H, W = science_ramp.shape

    coeffs1 = linearity_hdul["COEFFS1"].data.astype(np.float32)  # (P1+1,H,W)
    coeffs2 = linearity_hdul["COEFFS2"].data.astype(np.float32)  # (P2+1,H,W)
    cutoffs = linearity_hdul["CUTOFFS"].data.astype(np.float32)  # (H,W) measured DN

    if "GOODPIX" in linearity_hdul:
        goodpix = linearity_hdul["GOODPIX"].data.astype(bool)    # (H,W)
    else:
        goodpix = np.ones((H, W), dtype=bool)

    P1 = coeffs1.shape[0]
    P2 = coeffs2.shape[0]

    # ---------- Pixel-level DQ ----------
    pixel_dq = np.zeros((H, W), dtype=np.uint16)

    # ---------- Coefficient sanity checks ----------

    # Where does each segment have ANY finite coefficients?
    poly1_has_any = np.isfinite(coeffs1).any(axis=0)   # (H,W)
    poly2_has_any = np.isfinite(coeffs2).any(axis=0)   # (H,W)

    poly1_all_nan = ~poly1_has_any #lower order polynimal was not fitted
    poly2_all_nan = ~poly2_has_any #higher order polynimal was not fitted

    cutoff_nan = ~np.isfinite(cutoffs) #cutoff undefined
    goodpix_bad = ~goodpix #not linearity corrected pixel

    # COEFFS2: coefficients should not be enormous
    abs2 = np.abs(coeffs2)
    abs2[~np.isfinite(abs2)] = np.nan
    high2 = np.nanmax(abs2, axis=0)                    # (H,W)
    coeff2_bad = high2 > coeff2_max

    # Pixels where we *must* fall back to identity:
    #   - cutoff NaN, OR
    #   - GOODPIX is false, OR
    #   - low segment completely missing.
    mask_identity = cutoff_nan | goodpix_bad | poly1_all_nan     # (H,W)

    # Pixels where high segment cannot be trusted, but low is OK:
    high_bad_only = (~mask_identity) & (poly2_all_nan | coeff2_bad)

    # ---------- Apply fallback behaviours ----------

    # 1) Identity pixels: mark NO_LIN_CORR, use y = x for both segments
    if np.any(mask_identity):
        pixel_dq[mask_identity] |= DQ_FLAGS["NO_LIN_CORR"]

        # Identity polynomial y = x for low-order set (P1)
        ident1 = np.zeros(P1, dtype=np.float32)
        if P1 >= 2:
            ident1[-2] = 1.0   # slope
            ident1[-1] = 0.0   # offset
        else:
            ident1[0] = 1.0

        # Identity polynomial y = x for high-order set (P2)
        ident2 = np.zeros(P2, dtype=np.float32)
        if P2 >= 2:
            ident2[-2] = 1.0
            ident2[-1] = 0.0
        else:
            ident2[0] = 1.0

        coeffs1[:, mask_identity] = ident1[:, None]
        coeffs2[:, mask_identity] = ident2[:, None]

    # NOTE: for high_bad_only we *do not* touch the coefficients directly.
    # We will simply never use COEFFS2 for those pixels in the blending step.

    # ---------- Evaluate polynomials on the ramp ----------
    corrected1 = _polyval_stack(coeffs1, science_ramp)  # (N,H,W)
    corrected2 = _polyval_stack(coeffs2, science_ramp)  # (N,H,W)


    corrected_ramp = corrected1.copy()

    #  - high segment not flagged as bad
    valid_high = (~mask_identity) & (~high_bad_only)   # (H,W)
    valid_high_3d = valid_high[None, :, :]             # (N,H,W)

    # Use measured-DN cutoff only to choose which polynomial to apply.
    below_cut = science_ramp <= cutoffs[None, :, :]    # (N,H,W) bool
    use_high = (~below_cut) & valid_high_3d

    #if input read greater than cutoff, use corrected 2, otherwise corrected_ramp
    corrected_ramp = np.where(use_high, corrected2, corrected_ramp)

    # ---------- Preserve SATURATED samples from input GROUPDQ ----------
    sat_mask = (group_dq & DQ_FLAGS["SATURATED"]) != 0  # (N,H,W) bool
    corrected_ramp = np.where(sat_mask, science_ramp, corrected_ramp)

    # ---------- Build cutoff-read map (per pixel) ----------
    # Strategy:
    #   1) If the calibration file provides a CUTOFF_READ HDU, use that.
    #   2) Otherwise, derive it from GROUPDQ SATURATED flags:
    #        first read where SATURATED is set; if never, N-1.

    if "CUTOFF_READ" in linearity_hdul:
        cutoff_read_map = linearity_hdul["CUTOFF_READ"].data.astype(np.int16)
        # clip to valid range
        cutoff_read_map = np.clip(cutoff_read_map, 0, N - 1)
    else:
        cutoff_read_map = np.full((H, W), fill_value=N - 1, dtype=np.int16)
        sat_mask_3d = (group_dq & DQ_FLAGS["SATURATED"]) != 0  # (N,H,W)

        # For each pixel, find the first read where it is saturated.
        # This uses the same saturation logic you used to build group_dq
        # (e.g., derivative-based sat detection), so it's consistent.
        for r in range(H):
            sat_row = sat_mask_3d[:, r, :]       # (N,W)
            any_sat = sat_row.any(axis=0)        # (W,)
            if not np.any(any_sat):
                continue
            first_idx = np.argmax(sat_row[:, any_sat], axis=0)  # (n_sat,)
            cutoff_read_map[r, any_sat] = first_idx.astype(np.int16)

    
    total_pix = H * W
    n_no_lin   = np.count_nonzero(pixel_dq & DQ_FLAGS["NO_LIN_CORR"])
    n_high_bad = np.count_nonzero(high_bad_only & ~mask_identity)

    #print(f"[linearity] NO_LIN_CORR: {n_no_lin} / {total_pix} "
    #      f"({100.0*n_no_lin/total_pix:.2f} %)")
    #print(f"[linearity] Pixels using low-only (high segment disabled): "
    #      f"{n_high_bad} / {total_pix} "
    #      f"({100.0*n_high_bad/total_pix:.2f} %)")

    return corrected_ramp.astype(np.float32, copy=False), pixel_dq, cutoff_read_map



def run_linearity_workflow(science_ramp, linearity_file): #best one
    """
    High-level wrapper for linearity correction:

      1) Find saturation onset from derivatives (sat_map_meas, measured DN).
      2) Build initial GROUPDQ from raw saturation (measured DN).
      3) Apply two-piece polynomial linearity correction (measured -> linear),
         and get a calibration-based read cutoff (cutoff_read_map).
      4) Derive a *physical* cutoff from where the corrected ramp crosses the
         saturation DN (sat_phys), per pixel.
      5) Combine both into an effective cutoff = min(calib_cutoff, phys_cutoff).
      6) Enforce this effective cutoff in GROUPDQ (mark reads >= cutoff SATURATED).
      7) For all reads >= cutoff, force corrected DN to the saturation DN, and
         guarantee that no corrected value exceeds saturation.
    """
    # Ensure float
    science_ramp = science_ramp.astype(np.float32, copy=False)
    N, H, W = science_ramp.shape

    # 1. Derivative-based saturation map in *measured* DN
    sat_map_meas = create_saturation_map_by_slope(science_ramp)   # (H,W)

    # 2. GROUPDQ from raw saturation (measured DN)
    group_dq_raw = create_group_dq(science_ramp, sat_map_meas)    # (N,H,W)

    calib_path = pkg_resources.resource_filename('scalesdrp', 'calib/')
    # 3. Linearity correction and calibration-based cutoff
    sat_dn_meas = None
    with fits.open(calib_path+linearity_file) as hdul:
        if "SATURATION" in hdul:
            sat_dn_meas = hdul["SATURATION"].data.astype(np.float32)  # (H,W)
        corrected_ramp, pixel_dq, cutoff_read_map_cal = apply_linearity_correction_twopart_final(
            science_ramp,
            group_dq_raw,
            hdul,
        )   # corrected_ramp: (N,H,W), cutoff_read_map_cal: (H,W)

    #preferred saturation map from hdul, otherwise from the above step
    if sat_dn_meas is not None:
        sat_phys = sat_dn_meas
    else:
        sat_phys = sat_map_meas         

    sat_phys_3d = sat_phys[None, :, :]  # (1,H,W) for broadcasting

    #filled with last index
    phys_cut_map = np.full((H, W), fill_value=N - 1, dtype=np.int16)

    over = corrected_ramp >= sat_phys_3d   # (N,H,W), True where we exceed saturation
    any_over = over.any(axis=0)            # (H,W) mask of pixels that ever exceed

    if np.any(any_over):
        # For those pixels, find first index along the read axis where saturation happens
        idx_first = np.argmax(over[:, any_over], axis=0)   #return true for the first index
        phys_cut_map[any_over] = idx_first.astype(np.int16)


    cutoff_read_map = np.minimum(
        cutoff_read_map_cal.astype(np.int16),
        phys_cut_map
    )  # (H,W), values in [0, N-1]

    #marks all reads >= cutoff_read_map[y,x] as SATURATED
    group_dq = enforce_cutoff_in_groupdq(group_dq_raw, cutoff_read_map)

    #Force all reads >= cutoff to saturation DN
    read_idx  = np.arange(N, dtype=np.int16)[:, None, None]    # (N,1,1)
    cutoff_3d = cutoff_read_map[None, :, :]                    # (1,H,W)
    beyond_cut = read_idx >= cutoff_3d                         # (N,H,W)

    sat_mask = (group_dq & DQ_FLAGS["SATURATED"]) != 0         # (N,H,W)
    mask_off = sat_mask | beyond_cut                           # any "off-limits" read

    
    corrected_ramp = np.where(mask_off, sat_phys_3d, corrected_ramp)

    #no corrected value exceeds the saturation DN
    corrected_ramp = np.minimum(corrected_ramp, sat_phys_3d)

    # 8. Summary
    total_pixels = H * W
    bad_pix = np.count_nonzero(pixel_dq & DQ_FLAGS["NO_LIN_CORR"])
    good_pix = total_pixels - bad_pix

    print("\n========== Linearity correction summary ==========")
    print(f"Total pixels:      {total_pixels:,}")
    print(f"Corrected pixels:  {good_pix:,} ({100*good_pix/total_pixels:.2f} %)")
    print(f"Fallback pixels:   {bad_pix:,} ({100*bad_pix/total_pixels:.2f} %)")
    print("==================================================\n")

    return (corrected_ramp.astype(np.float32, copy=False),
            pixel_dq,
            group_dq,
            cutoff_read_map,
            sat_map_meas)































