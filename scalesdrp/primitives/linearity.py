from scalesdrp.core.scales_pkg_resources import get_resource_path
import warnings
import numpy as np
from astropy.io import fits
from tqdm import tqdm

################ create non-linearity coefficients #################
def make_linearity_coeffs_final(
    ramp_cube,
    bpm_2d=None,
    output_filename="linearity_coeffs_final.fits",
    *,
    skip_initial_reads=2,
    low_poly_order=1,
    high_poly_order=3,

    # saturation / plateau detection
    sat_window=3,
    sat_drop_fraction=0.65,      # plateau if local derivate dy < 0.65 * early slope
    hard_sat_dn=None,

    # break detection
    linearity_fraction=0.90,     # break where measured/linear_model < 0.90 of the fitted model
    min_points_total=10,
    min_low_points=6,
    min_high_points=6,
    fallback_split_fraction=0.75, #if no real non-linearity found, split 75% of the valid ramp

    # quality cuts
    min_dynamic_range=50.0, #if a pixel change by less than 50 DN, treated as unstable
    max_negative_jump_fraction=0.20,
    negative_jump_sigma=5.0,

    # fitting sanity
    max_coeff_abs=1e8, #reject polynomial fot with coefficients larger than this
    transition_tolerance=5.0, #low/high transition coeff1 and coeff2 should agree within 5 DN
    monotonic_tolerance=-1e-3, #allow negative slopes
    max_rms_low=np.inf, #optional rms upper limit
    max_rms_high=np.inf,

    # behavior
    allow_single_fit=True, #if both low and high fit fails, try to fit a single polynomial over the whole ramp
    require_real_break=False, #if True, pixel with no detected non-linearity break and rejected
):
    """
    Create per-pixel linearity correction coefficients.

    Correction model:
        measured DN M  --->  linearized DN L

    Main idea:
      1. Remove first few unstable reads.
      2. Find saturation/plateau.
      3. Keep only pre-saturation reads.
      4. Fit a baseline line anchored at the first valid read.
      5. Find nonlinearity break where M/L < linearity_fraction.
      6. Fit low and high correction polynomials M -> L.
      7. Normalize polynomial fitting internally for numerical stability.
      8. Convert normalized polynomial back to ordinary measured-DN coefficients.
      9. Store coefficients and diagnostic maps.

    Output coefficient convention:
      COEFFS1 and COEFFS2 are standard np.polyval-compatible coefficients:
          L = np.polyval(coeffs, M)
    """

    # -----------------------------
    # DQ flags
    # -----------------------------
    DQ_BPM_INPUT             = 1 << 0 #input bpm
    DQ_NONFINITE             = 1 << 1 #nan or inf values
    DQ_LOW_DYNAMIC_RANGE     = 1 << 2 #low dynamic range pixels
    DQ_TOO_FEW_POINTS        = 1 << 3 #not enough reads to fit
    DQ_NEGATIVE_JUMPS        = 1 << 4 #negative jumps
    DQ_NO_SATURATION_FOUND   = 1 << 5 #saturation
    DQ_BAD_BASELINE          = 1 << 6 #baseline fit failed
    DQ_NO_BREAK_FOUND        = 1 << 7 #no nolinearity break found
    DQ_BAD_LOWFIT            = 1 << 8 #lower fit failed
    DQ_BAD_HIGHFIT           = 1 << 9 #upper fit failed
    DQ_SINGLE_FIT_USED       = 1 << 10 #used single fit fallback option
    DQ_NONMONOTONIC          = 1 << 11 #Polynomial correction is nonmonotonic.
    DQ_TRANSITION_BAD        = 1 << 12 #low/high transition is bad
    DQ_RMS_TOO_HIGH          = 1 << 13 #polynomial residual is too high
    DQ_UNPHYSICAL_CORR       = 1 << 14 #correction became unphysical

    ramp = np.asarray(ramp_cube, dtype=float)
    if ramp.ndim != 3:
        raise ValueError("ramp_cube must have shape (n_reads, ny, nx)")

    n_reads, ny, nx = ramp.shape
    reads = np.arange(n_reads, dtype=float)

    if bpm_2d is None:
        bpm_2d = np.zeros((ny, nx), dtype=np.uint8)
    else:
        bpm_2d = np.asarray(bpm_2d)
        if bpm_2d.shape != (ny, nx):
            raise ValueError("bpm_2d must have shape (ny, nx)")
        bpm_2d = (bpm_2d != 0).astype(np.uint8)

    coeffs1 = np.full((low_poly_order + 1, ny, nx), np.nan, dtype=np.float32)
    coeffs2 = np.full((high_poly_order + 1, ny, nx), np.nan, dtype=np.float32)

    cutoff_m = np.full((ny, nx), np.nan, dtype=np.float32)
    cutoff_l = np.full((ny, nx), np.nan, dtype=np.float32)

    saturation = np.full((ny, nx), np.nan, dtype=np.float32)
    sat_index = np.full((ny, nx), -1, dtype=np.int16)

    slope_map = np.full((ny, nx), np.nan, dtype=np.float32)
    intercept_map = np.full((ny, nx), np.nan, dtype=np.float32)

    break_index = np.full((ny, nx), -1, dtype=np.int16)
    nvalid_map = np.zeros((ny, nx), dtype=np.int16)
    nlow_map = np.zeros((ny, nx), dtype=np.int16)
    nhigh_map = np.zeros((ny, nx), dtype=np.int16)

    rms1_map = np.full((ny, nx), np.nan, dtype=np.float32)
    rms2_map = np.full((ny, nx), np.nan, dtype=np.float32)

    raw_rms_map = np.full((ny, nx), np.nan, dtype=np.float32)
    corr_rms_map = np.full((ny, nx), np.nan, dtype=np.float32)
    improvement_map = np.full((ny, nx), np.nan, dtype=np.float32)

    dq = np.zeros((ny, nx), dtype=np.uint32)
    goodpix = np.zeros((ny, nx), dtype=np.uint8)

    # -----------------------------
    # Helpers
    # -----------------------------
    def safe_polyfit(x, y, order):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < order + 2:
            return None
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return None
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", np.RankWarning)
                p = np.polyfit(x, y, order)
            if not np.all(np.isfinite(p)):
                return None
            if np.nanmax(np.abs(p)) > max_coeff_abs:
                return None
            return p.astype(np.float64)
        except Exception:
            return None

    def normalized_polyfit_to_original_x(x, y, order):
        """
        Fit y as polynomial of x, but internally normalize x for stability.
        Return standard np.polyval-compatible polynomial in original x.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.size < order + 2:
            return None

        x0 = np.nanmean(x)
        xs = np.nanstd(x)
        if not np.isfinite(xs) or xs <= 0:
            return None

        xn = (x - x0) / xs
        pn = safe_polyfit(xn, y, order)
        if pn is None:
            return None

        # Convert polynomial in xn=(x-x0)/xs to polynomial in x.
        poly = np.poly1d(pn)
        xpoly = np.poly1d([1.0 / xs, -x0 / xs])
        p_orig = poly(xpoly).c

        if not np.all(np.isfinite(p_orig)):
            return None
        if np.nanmax(np.abs(p_orig)) > max_coeff_abs:
            return None

        return p_orig.astype(np.float32)

    def rms_to_line(y):
        y = np.asarray(y, dtype=float)
        t = np.arange(len(y), dtype=float)
        m = np.isfinite(y)
        if np.count_nonzero(m) < 2:
            return np.nan
        p = np.polyfit(t[m], y[m], 1)
        model = np.polyval(p, t[m])
        return float(np.sqrt(np.mean((y[m] - model) ** 2)))

    def rms_resid(p, x, y): #check how stright the ramp is.
        if p is None or len(x) == 0:
            return np.nan
        model = np.polyval(p, x)
        return float(np.sqrt(np.nanmean((y - model) ** 2)))

    def monotonic_ok(p, xmin, xmax, ngrid=128):
        '''
        Check whether a ploynomial correction is monotonic over its valid DN range
        '''
        if p is None:
            return False
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            return False
        grid = np.linspace(xmin, xmax, ngrid)
        vals = np.polyval(p, grid)
        if not np.all(np.isfinite(vals)):
            return False #Accept only if corrected value mostly increases as measured DN increases.
        return np.all(np.diff(vals) > monotonic_tolerance)

    def apply_piecewise(y, p1, p2, cm):
        y = np.asarray(y, dtype=float)
        out = np.full_like(y, np.nan, dtype=float)

        has_p1 = p1 is not None and np.all(np.isfinite(p1))
        has_p2 = p2 is not None and np.all(np.isfinite(p2))

        if not has_p1:
            return out

        low = y <= cm
        out[low] = np.polyval(p1, y[low]) #low DN reads

        if has_p2:
            out[~low] = np.polyval(p2, y[~low]) #high DN use p2 if available.
        else: #otherwise fallback to p1
            out[~low] = np.polyval(p1, y[~low])

        return out

    # -----------------------------
    # Per-pixel loop
    # -----------------------------
    for r, c in tqdm(np.ndindex(ny, nx), total=ny * nx, desc="Linearity coeffs"):

        if bpm_2d[r, c] != 0: #skip if a pixel is bpm true
            dq[r, c] |= DQ_BPM_INPUT
            continue

        y_all = ramp[:, r, c]
        finite = np.isfinite(y_all)

        if np.count_nonzero(finite) < min_points_total:
            dq[r, c] |= DQ_NONFINITE #not enough finite reads
            continue
        #keep only finite reads
        t = reads[finite]
        y = y_all[finite]

        # Skip first reset/unstable reads
        if skip_initial_reads > 0:
            if y.size <= skip_initial_reads + min_points_total:
                dq[r, c] |= DQ_TOO_FEW_POINTS
                continue
            t = t[skip_initial_reads:]
            y = y[skip_initial_reads:]

        if y.size < min_points_total:
            dq[r, c] |= DQ_TOO_FEW_POINTS
            continue

        if np.nanmax(y) - np.nanmin(y) < min_dynamic_range:
            dq[r, c] |= DQ_LOW_DYNAMIC_RANGE
            continue

        dy = np.diff(y)
        if dy.size < sat_window:
            dq[r, c] |= DQ_TOO_FEW_POINTS
            continue

        # Negative-jump rejection using robust scale
        med_dy = np.nanmedian(dy)
        mad_dy = 1.4826 * np.nanmedian(np.abs(dy - med_dy)) #scatter estimate
        if not np.isfinite(mad_dy) or mad_dy <= 0:
            mad_dy = max(abs(med_dy), 1.0)

        #reject pixel with too many negative jumps
        neg_thresh = -negative_jump_sigma * mad_dy
        if np.mean(dy < neg_thresh) > max_negative_jump_fraction:
            dq[r, c] |= DQ_NEGATIVE_JUMPS
            continue

        # -----------------------------
        # Saturation detection
        # -----------------------------
        sat_i = None

        if hard_sat_dn is not None:
            hard_hits = np.where(y >= hard_sat_dn)[0]
            if hard_hits.size > 0:
                sat_i = int(hard_hits[0])

        if sat_i is None:
            n_early = max(3, min(len(dy), len(dy) // 4)) #estimate the early slope
            early_slope = np.nanmedian(dy[:n_early])

            if np.isfinite(early_slope) and early_slope > 0:
                flat_thresh = sat_drop_fraction * early_slope #define flattening threshold
                #find first region where local derivative drops below threshold
                for i in range(len(dy) - sat_window + 1):
                    local = np.nanmedian(dy[i:i + sat_window])
                    if local < flat_thresh:
                        sat_i = i + 1
                        break

        if sat_i is None: #if no saturation use full ramp, saturation is final ramp
            dq[r, c] |= DQ_NO_SATURATION_FOUND
            t_valid = t
            y_valid = y
            sat_index[r, c] = -1
            saturation[r, c] = y[-1]
        else: #use pre saturation reads
            sat_i = int(np.clip(sat_i, 1, len(y) - 1))
            t_valid = t[:sat_i]
            y_valid = y[:sat_i]
            sat_index[r, c] = sat_i
            saturation[r, c] = y[sat_i]

        nvalid = len(y_valid)
        nvalid_map[r, c] = nvalid

        if nvalid < min_points_total: #reject if too few reads
            dq[r, c] |= DQ_TOO_FEW_POINTS
            continue

        # -----------------------------
        # Anchored baseline fit
        #
        # Fit y - y0 = slope * (t - t0)
        # This avoids intercept/bias causing an unphysical baseline.
        # -----------------------------
        t0 = t_valid[0] #shift ramp so first valid read
        y0 = y_valid[0]

        tt = t_valid - t0
        yy = y_valid - y0

        denom = np.sum(tt ** 2)
        if denom <= 0 or not np.isfinite(denom):
            dq[r, c] |= DQ_BAD_BASELINE
            continue

        slope = np.sum(tt * yy) / denom
        intercept = y0 - slope * t0

        if not np.isfinite(slope) or slope <= 0:
            dq[r, c] |= DQ_BAD_BASELINE
            continue

        L_valid = slope * t_valid + intercept #ideal linearized ramp

        if not np.all(np.isfinite(L_valid)):
            dq[r, c] |= DQ_BAD_BASELINE
            continue

        slope_map[r, c] = slope
        intercept_map[r, c] = intercept

        # -----------------------------
        # Break detection by 90% rule for finding non-linearity
        # -----------------------------
        ratio = y_valid / np.maximum(L_valid, 1.0)
        candidates = np.where(ratio < linearity_fraction)[0]

        if candidates.size > 0:
            idx_break = int(candidates[0])
        else: #no break found
            idx_break = nvalid
            dq[r, c] |= DQ_NO_BREAK_FOUND

        break_index[r, c] = idx_break

        if require_real_break and idx_break >= nvalid:
            continue

        # -----------------------------
        # Split where real break found , othrewise split based on fallback_split_fraction
        # -----------------------------
        if idx_break < nvalid:
            split = idx_break
        else:
            split = int(fallback_split_fraction * nvalid)
        #ensure enough points on both sides
        if nvalid >= min_low_points + min_high_points:
            split = max(min_low_points, split)
            split = min(nvalid - min_high_points, split)
        else:
            split = max(1, min(split, nvalid - 1))
        #create measured to-linear target pair
        M_low = y_valid[:split]
        L_low = L_valid[:split]

        M_high = y_valid[split:]
        L_high = L_valid[split:]

        #store transition point
        nlow_map[r, c] = len(M_low)
        nhigh_map[r, c] = len(M_high)

        cutoff_m[r, c] = y_valid[split]
        cutoff_l[r, c] = L_valid[split]

        # -----------------------------
        # Fit polynomial correction M -> L
        # -----------------------------
        p1 = None
        p2 = None

        if len(M_low) >= min_low_points: #fit low region corrections
            p1 = normalized_polyfit_to_original_x(M_low, L_low, low_poly_order)

        if p1 is None:
            dq[r, c] |= DQ_BAD_LOWFIT

        if len(M_high) >= min_high_points: #fit high region correction
            p2 = normalized_polyfit_to_original_x(M_high, L_high, high_poly_order)

        if p2 is None:
            dq[r, c] |= DQ_BAD_HIGHFIT

        # Single fallback if both segment fits failed
        if p1 is None and p2 is None and allow_single_fit:
            p_single = normalized_polyfit_to_original_x(y_valid, L_valid, low_poly_order)
            if p_single is not None:
                p1 = p_single
                dq[r, c] |= DQ_SINGLE_FIT_USED

        if p1 is None and p2 is None:
            continue

        # -----------------------------
        # Smooth transition
        # -----------------------------
        #evaluate both polynomial at cutoff
        if p1 is not None and p2 is not None:
            mcut = cutoff_m[r, c]
            y1 = np.polyval(p1, mcut)
            y2 = np.polyval(p2, mcut)

            if np.isfinite(y1) and np.isfinite(y2):
                jump = y2 - y1

                # Shift high polynomial constant term to match low polynomial.
                p2 = p2.copy()
                p2[-1] -= jump
                #If still discontinuous, reject high polynomial.
                y2_new = np.polyval(p2, mcut)
                if not np.isfinite(y2_new) or abs(y2_new - y1) > transition_tolerance:
                    dq[r, c] |= DQ_TRANSITION_BAD
                    p2 = None
            else:
                dq[r, c] |= DQ_TRANSITION_BAD
                p2 = None

        # -----------------------------
        # Monotonic checks
        # -----------------------------
        if p1 is not None and len(M_low) >= 2: #reject low fit if it is not monotonic
            if not monotonic_ok(p1, np.nanmin(M_low), np.nanmax(M_low)):
                dq[r, c] |= DQ_NONMONOTONIC
                p1 = None

        if p2 is not None and len(M_high) >= 2: #reject high fit if it is not monotonic
            if not monotonic_ok(p2, np.nanmin(M_high), np.nanmax(M_high)):
                dq[r, c] |= DQ_NONMONOTONIC
                p2 = None

        if p1 is None and p2 is None:
            continue

        # -----------------------------
        # Fit residuals
        # -----------------------------
        if p1 is not None:
            rms1 = rms_resid(p1, M_low, L_low)
            rms1_map[r, c] = rms1 #store low-fit rms
            if np.isfinite(rms1) and rms1 > max_rms_low:
                dq[r, c] |= DQ_RMS_TOO_HIGH

        if p2 is not None:
            rms2 = rms_resid(p2, M_high, L_high)
            rms2_map[r, c] = rms2 #store high-fit rms
            if np.isfinite(rms2) and rms2 > max_rms_high:
                dq[r, c] |= DQ_RMS_TOO_HIGH

        # -----------------------------
        # Verify correction does not make valid ramp worse
        # -----------------------------
        y_corr = apply_piecewise(y_valid, p1, p2, cutoff_m[r, c])

        #Measure straightness before and after correction.
        raw_rms = rms_to_line(y_valid)
        corr_rms = rms_to_line(y_corr)

        raw_rms_map[r, c] = raw_rms
        corr_rms_map[r, c] = corr_rms

        if np.isfinite(raw_rms) and np.isfinite(corr_rms) and corr_rms > 0:
            improvement_map[r, c] = raw_rms / corr_rms

        if not np.all(np.isfinite(y_corr)):
            dq[r, c] |= DQ_UNPHYSICAL_CORR #reject NaN
            continue

        # Do not accept if correction creates clear negative steps.
        dycorr = np.diff(y_corr)
        if np.any(dycorr < -max(1.0, 5.0 * mad_dy)):
            dq[r, c] |= DQ_UNPHYSICAL_CORR
            continue

        # Do not accept if correction makes the line residual worse than 25%.
        # Allow small tolerance because correction may be subtle.
        if np.isfinite(raw_rms) and np.isfinite(corr_rms):
            if corr_rms > 1.25 * raw_rms:
                dq[r, c] |= DQ_UNPHYSICAL_CORR
                continue

        # -----------------------------
        # Store coeffs
        # -----------------------------
        if p1 is not None:
            coeffs1[:, r, c] = p1.astype(np.float32)

        if p2 is not None:
            coeffs2[:, r, c] = p2.astype(np.float32)

        fatal = (
            DQ_BPM_INPUT |
            DQ_NONFINITE |
            DQ_LOW_DYNAMIC_RANGE |
            DQ_TOO_FEW_POINTS |
            DQ_NEGATIVE_JUMPS |
            DQ_BAD_BASELINE |
            DQ_NONMONOTONIC |
            DQ_TRANSITION_BAD |
            DQ_UNPHYSICAL_CORR
        )

        if (dq[r, c] & fatal) == 0:
            goodpix[r, c] = 1

    # -----------------------------
    # Write FITS
    # -----------------------------
    phdu = fits.PrimaryHDU()
    hdr = phdu.header
    hdr["COMMENT"] = "Linearity correction coefficients: measured DN -> linearized DN"
    hdr["SKIPRD"] = int(skip_initial_reads)
    hdr["LINFRAC"] = float(linearity_fraction)
    hdr["SATDROP"] = float(sat_drop_fraction)
    hdr["SATWIN"] = int(sat_window)
    hdr["LOWORD"] = int(low_poly_order)
    hdr["HIWORD"] = int(high_poly_order)
    hdr["MINPTS"] = int(min_points_total)
    hdr["MINLOW"] = int(min_low_points)
    hdr["MINHIGH"] = int(min_high_points)
    hdr["FALLSPL"] = float(fallback_split_fraction)

    hdr["DQ0"] = "1<<0 BPM input"
    hdr["DQ1"] = "1<<1 nonfinite"
    hdr["DQ2"] = "1<<2 low dynamic range"
    hdr["DQ3"] = "1<<3 too few points"
    hdr["DQ4"] = "1<<4 negative jumps"
    hdr["DQ5"] = "1<<5 no saturation found"
    hdr["DQ6"] = "1<<6 bad baseline"
    hdr["DQ7"] = "1<<7 no break found"
    hdr["DQ8"] = "1<<8 bad low fit"
    hdr["DQ9"] = "1<<9 bad high fit"
    hdr["DQ10"] = "1<<10 single fit fallback"
    hdr["DQ11"] = "1<<11 nonmonotonic"
    hdr["DQ12"] = "1<<12 bad transition"
    hdr["DQ13"] = "1<<13 RMS too high"
    hdr["DQ14"] = "1<<14 unphysical correction"

    hdul = fits.HDUList([
        phdu,
        fits.ImageHDU(coeffs1, name="COEFFS1"),
        fits.ImageHDU(coeffs2, name="COEFFS2"),
        fits.ImageHDU(cutoff_m, name="CUTOFF_M"),
        fits.ImageHDU(cutoff_l, name="CUTOFF_L"),
        fits.ImageHDU(saturation, name="SATURATION"),
        fits.ImageHDU(sat_index, name="SAT_INDEX"),
        fits.ImageHDU(slope_map, name="SLOPE"),
        fits.ImageHDU(intercept_map, name="INTERCEPT"),
        fits.ImageHDU(break_index, name="BREAK_INDEX"),
        fits.ImageHDU(nvalid_map, name="NVALID"),
        fits.ImageHDU(nlow_map, name="NLOW"),
        fits.ImageHDU(nhigh_map, name="NHIGH"),
        fits.ImageHDU(rms1_map, name="RMS1"),
        fits.ImageHDU(rms2_map, name="RMS2"),
        fits.ImageHDU(raw_rms_map, name="RAW_RMS"),
        fits.ImageHDU(corr_rms_map, name="CORR_RMS"),
        fits.ImageHDU(improvement_map, name="IMPROVE"),
        fits.ImageHDU(dq, name="DQ"),
        fits.ImageHDU(goodpix, name="GOODPIX"),
        fits.ImageHDU(bpm_2d.astype(np.uint8), name="BPM_INPUT"),
    ])

    hdul.writeto(output_filename, overwrite=True)
    return hdul

#input to the non-linearity coefficient creating function 
#is a acn & 1/f corrected cube of reads reaching saturation

##################### correct ########################################
# ----------------------------------------------------------------------
# Linearity DQ bit definitions
# ----------------------------------------------------------------------
LIN_NO_CORR      = 1 << 0   # no valid linearity correction for this pixel/read
LIN_SATURATED    = 1 << 1   # read is at/after SAT_INDEX
LIN_BAD_VALUE    = 1 << 2   # correction produced NaN/inf
LIN_NONMONOTONIC = 1 << 3   # correction made valid ramp non-monotonic
LIN_APPLIED      = 1 << 4   # linearity correction applied to this read

DQ_FLAGS = {
    "DO_NOT_USE": 1 << 0,      # input BPM / unusable pixel
    "NO_LIN_CORR": 1 << 1,     # no valid linearity correction
    "SATURATED": 1 << 2,       # read is at/after SAT_INDEX
    "BAD_LIN_CORR": 1 << 3,    # correction produced invalid value
    "LIN_NONMONO": 1 << 4,     # correction made ramp non-monotonic
    "LIN_APPLIED": 1 << 5,     # correction applied successfully
}

def apply_linearity_coeffs_to_cube(
    input_cube,
    coeff_file,
    bpm_2d=None,
    *,
    invalid_read_behavior="raw",
    use_goodpix=True,
    check_finite=True,
    check_monotonic=True,
    monotonic_tolerance=-1e-3,
    return_aux=False,
):
    """
    Apply measured-DN -> linearized-DN correction to a cube of reads.

    The correction model is:

        M -> L

    where M is the measured DN and L is the linearized DN.

    Coefficient convention:

        L = np.polyval(coeffs, M)

    Parameters
    ----------
    input_cube : ndarray, shape (N, H, W)
        Input ramp cube in measured DN.

    coeff_file : str
        FITS file created by make_linearity_coeffs_final().

    bpm_2d : ndarray, optional, shape (H, W)
        Additional bad-pixel mask. Nonzero pixels are not corrected.

    invalid_read_behavior : {"raw", "flat_last_valid", "nan"}
        What to do for reads at/after SAT_INDEX.

        "raw":
            Keep original measured DN after saturation.

        "flat_last_valid":
            Set reads after saturation to the last valid corrected value.

        "nan":
            Set reads after saturation to NaN.

    use_goodpix : bool
        If True, only apply correction where GOODPIX == 1.

    check_finite : bool
        If True, reject the correction for a pixel if input or corrected
        values are non-finite.

    check_monotonic : bool
        If True, reject the correction for a pixel if corrected valid reads
        contain a clear negative step.

    monotonic_tolerance : float
        Allowed negative step in corrected DN.

    return_aux : bool
        If True, return corrected_cube, lin_dq, and lin_mask.

    Returns
    -------
    corrected_cube : ndarray, shape (N, H, W), float32

    If return_aux=True:
        corrected_cube : ndarray, shape (N, H, W), float32
        lin_dq        : ndarray, shape (N, H, W), uint32
        lin_mask      : ndarray, shape (H, W), bool

    Notes
    -----
    lin_dq is a read-level DQ cube. Ramp fitting can identify saturated reads as:

        sat_mask = (lin_dq & LIN_SATURATED) != 0

    and corrected reads as:

        corrected_reads = (lin_dq & LIN_APPLIED) != 0
    """

    cube = np.asarray(input_cube, dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError("input_cube must have shape (N, H, W)")

    N, H, W = cube.shape

    if invalid_read_behavior not in {"raw", "flat_last_valid", "nan"}:
        raise ValueError(
            "invalid_read_behavior must be 'raw', 'flat_last_valid', or 'nan'"
        )

    if bpm_2d is None:
        bpm_2d = np.zeros((H, W), dtype=bool)
    else:
        bpm_2d = np.asarray(bpm_2d) != 0
        if bpm_2d.shape != (H, W):
            raise ValueError("bpm_2d must have shape (H, W)")

    with fits.open(coeff_file) as hdul:
        coeffs1 = hdul["COEFFS1"].data.astype(np.float64)
        coeffs2 = hdul["COEFFS2"].data.astype(np.float64)
        cutoff_m = hdul["CUTOFF_M"].data.astype(np.float64)
        sat_index = hdul["SAT_INDEX"].data.astype(np.int32)

        if "GOODPIX" in hdul and use_goodpix:
            goodpix = hdul["GOODPIX"].data.astype(bool)
        else:
            goodpix = np.ones((H, W), dtype=bool)

        coeff_bpm = (
            hdul["BPM_INPUT"].data.astype(bool)
            if "BPM_INPUT" in hdul
            else np.zeros((H, W), dtype=bool)
        )

    if coeffs1.shape[1:] != (H, W):
        raise ValueError("COEFFS1 spatial shape does not match input cube")

    if coeffs2.shape[1:] != (H, W):
        raise ValueError("COEFFS2 spatial shape does not match input cube")

    if cutoff_m.shape != (H, W):
        raise ValueError("CUTOFF_M shape does not match input cube")

    if sat_index.shape != (H, W):
        raise ValueError("SAT_INDEX shape does not match input cube")

    corrected = cube.copy().astype(np.float32)

    # Read-level linearity DQ cube.
    lin_dq = np.zeros((N, H, W), dtype=np.uint32)

    # Pixel-level mask: True where at least one valid read was corrected.
    lin_mask = np.zeros((H, W), dtype=bool)

    def has_coeff(p):
        return np.all(np.isfinite(p))

    def apply_piecewise(y, p1, p2, cm):
        """
        Apply piecewise measured-DN -> linearized-DN correction.
        """
        y = np.asarray(y, dtype=np.float64)
        out = np.full_like(y, np.nan, dtype=np.float64)

        if not has_coeff(p1) or not np.isfinite(cm):
            return out

        low = y <= cm

        out[low] = np.polyval(p1, y[low])

        if has_coeff(p2):
            out[~low] = np.polyval(p2, y[~low])
        else:
            out[~low] = np.polyval(p1, y[~low])

        return out

    for r, c in tqdm(
        np.ndindex(H, W),
        total=H * W,
        desc="Applying linearity correction",
    ):

        # ------------------------------------------------------------
        # Skip pixels with no valid correction
        # ------------------------------------------------------------
        if bpm_2d[r, c] or coeff_bpm[r, c] or not goodpix[r, c]:
            lin_dq[:, r, c] |= LIN_NO_CORR
            continue

        p1 = coeffs1[:, r, c]
        p2 = coeffs2[:, r, c]
        cm = cutoff_m[r, c]

        if not has_coeff(p1) or not np.isfinite(cm):
            lin_dq[:, r, c] |= LIN_NO_CORR
            continue

        # ------------------------------------------------------------
        # Valid read range from SAT_INDEX
        #
        # SAT_INDEX is interpreted as the first saturated/invalid read.
        # If SAT_INDEX < 0, no saturation was found, so all reads are used.
        # ------------------------------------------------------------
        k = int(sat_index[r, c])

        if k < 0:
            k = N
        else:
            k = min(k, N)

        if k <= 0:
            lin_dq[:, r, c] |= LIN_NO_CORR
            lin_dq[:, r, c] |= LIN_SATURATED
            continue

        # Mark reads at/after saturation.
        if k < N:
            lin_dq[k:, r, c] |= LIN_SATURATED

        y = cube[:k, r, c]

        # ------------------------------------------------------------
        # Input finite check
        # ------------------------------------------------------------
        if check_finite and not np.all(np.isfinite(y)):
            lin_dq[:k, r, c] |= LIN_NO_CORR | LIN_BAD_VALUE
            continue

        # ------------------------------------------------------------
        # Apply correction to pre-saturation reads
        # ------------------------------------------------------------
        ycorr = apply_piecewise(y, p1, p2, cm)

        if check_finite and not np.all(np.isfinite(ycorr)):
            lin_dq[:k, r, c] |= LIN_NO_CORR | LIN_BAD_VALUE
            continue

        # ------------------------------------------------------------
        # Safety check: correction should not introduce obvious
        # downward steps in the pre-saturation ramp.
        # ------------------------------------------------------------
        if check_monotonic and ycorr.size > 1:
            dycorr = np.diff(ycorr)

            if np.any(dycorr < monotonic_tolerance):
                lin_dq[:k, r, c] |= LIN_NO_CORR | LIN_NONMONOTONIC
                continue

        # ------------------------------------------------------------
        # Write corrected values for pre-saturation reads
        # ------------------------------------------------------------
        corrected[:k, r, c] = ycorr.astype(np.float32)
        lin_dq[:k, r, c] |= LIN_APPLIED
        lin_mask[r, c] = True

        # ------------------------------------------------------------
        # Handle reads at/after saturation
        #
        # These are flagged with LIN_SATURATED regardless of value.
        # Ramp fitting should ignore them using lin_dq.
        # ------------------------------------------------------------
        if k < N:
            if invalid_read_behavior == "raw":
                corrected[k:, r, c] = cube[k:, r, c]

            elif invalid_read_behavior == "flat_last_valid":
                corrected[k:, r, c] = ycorr[-1]

            elif invalid_read_behavior == "nan":
                corrected[k:, r, c] = np.nan

    if return_aux:
        return corrected.astype(np.float32), lin_dq, lin_mask

    return corrected.astype(np.float32)

################### apply correction fast ############################
def apply_linearity_coeffs_to_cube_fast(
    input_cube,
    coeff_file,
    bpm_2d=None,
    *,
    invalid_read_behavior="raw",
    use_goodpix=True,
    check_finite=True,
    check_monotonic=True,
    monotonic_tolerance=-1e-3,
    return_aux=False,
):
    """
    Apply measured-DN -> linearized-DN correction to a cube of reads.

    Fast vectorized version.

    Coefficient convention:
        L = np.polyval(coeffs, M)

    Supports 1st, 2nd, or 3rd order polynomial coefficients.
    """

    cube = np.asarray(input_cube, dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError("input_cube must have shape (N, H, W)")

    N, H, W = cube.shape

    if invalid_read_behavior not in {"raw", "flat_last_valid", "nan"}:
        raise ValueError(
            "invalid_read_behavior must be 'raw', 'flat_last_valid', or 'nan'"
        )

    if bpm_2d is None:
        bpm_2d = np.zeros((H, W), dtype=bool)
    else:
        bpm_2d = np.asarray(bpm_2d) != 0
        if bpm_2d.shape != (H, W):
            raise ValueError("bpm_2d must have shape (H, W)")

    with fits.open(coeff_file) as hdul:
        coeffs1 = hdul["COEFFS1"].data.astype(np.float64)
        coeffs2 = hdul["COEFFS2"].data.astype(np.float64)
        cutoff_m = hdul["CUTOFF_M"].data.astype(np.float64)
        sat_index = hdul["SAT_INDEX"].data.astype(np.int32)

        if "GOODPIX" in hdul and use_goodpix:
            goodpix = hdul["GOODPIX"].data.astype(bool)
        else:
            goodpix = np.ones((H, W), dtype=bool)

        coeff_bpm = (
            hdul["BPM_INPUT"].data.astype(bool)
            if "BPM_INPUT" in hdul
            else np.zeros((H, W), dtype=bool)
        )

    if coeffs1.shape[1:] != (H, W):
        raise ValueError("COEFFS1 spatial shape does not match input cube")

    if coeffs2.shape[1:] != (H, W):
        raise ValueError("COEFFS2 spatial shape does not match input cube")

    if cutoff_m.shape != (H, W):
        raise ValueError("CUTOFF_M shape does not match input cube")

    if sat_index.shape != (H, W):
        raise ValueError("SAT_INDEX shape does not match input cube")

    def eval_poly_image(coeffs, y):
        """
        Evaluate polynomial image coefficients on cube y.

        coeffs shape: (degree + 1, H, W)
        y shape:      (N, H, W)

        Coefficient convention follows np.polyval:
            degree 1: c0*y + c1
            degree 2: c0*y**2 + c1*y + c2
            degree 3: c0*y**3 + c1*y**2 + c2*y + c3
        """

        ncoeff = coeffs.shape[0]

        if ncoeff == 2:
            return (
                coeffs[0][None, :, :] * y
                + coeffs[1][None, :, :]
            )

        elif ncoeff == 3:
            return (
                coeffs[0][None, :, :] * y**2
                + coeffs[1][None, :, :] * y
                + coeffs[2][None, :, :]
            )

        elif ncoeff == 4:
            return (
                coeffs[0][None, :, :] * y**3
                + coeffs[1][None, :, :] * y**2
                + coeffs[2][None, :, :] * y
                + coeffs[3][None, :, :]
            )

        else:
            raise ValueError(
                f"Unsupported polynomial order: coeffs has {ncoeff} coefficients. "
                "Only 1st, 2nd, or 3rd order polynomials are supported."
            )

    corrected = cube.copy()
    lin_dq = np.zeros((N, H, W), dtype=np.uint32)

    # ------------------------------------------------------------
    # Pixel-level validity mask
    # ------------------------------------------------------------
    p1_good = np.all(np.isfinite(coeffs1), axis=0)
    p2_good = np.all(np.isfinite(coeffs2), axis=0)
    cutoff_good = np.isfinite(cutoff_m)

    valid_pix = (
        goodpix
        & ~bpm_2d
        & ~coeff_bpm
        & p1_good
        & cutoff_good
    )

    no_corr_pix = ~valid_pix
    lin_dq[:, no_corr_pix] |= LIN_NO_CORR

    # ------------------------------------------------------------
    # SAT_INDEX handling
    #
    # SAT_INDEX is the first saturated/invalid read.
    # SAT_INDEX < 0 means no saturation was found.
    # ------------------------------------------------------------
    k = sat_index.copy()
    k[k < 0] = N
    k = np.clip(k, 0, N)

    read_index = np.arange(N, dtype=np.int32)[:, None, None]

    pre_sat_read = read_index < k[None, :, :]
    sat_read = read_index >= k[None, :, :]

    lin_dq[sat_read] |= LIN_SATURATED

    valid_read = pre_sat_read & valid_pix[None, :, :]

    # Pixels saturated at or before the first read cannot be corrected
    first_read_bad = valid_pix & (k <= 0)

    if np.any(first_read_bad):
        lin_dq[:, first_read_bad] |= LIN_NO_CORR | LIN_SATURATED
        valid_read[:, first_read_bad] = False

    # ------------------------------------------------------------
    # Evaluate piecewise correction
    # ------------------------------------------------------------
    y = cube.astype(np.float64)

    ycorr1 = eval_poly_image(coeffs1, y)
    ycorr2 = eval_poly_image(coeffs2, y)

    low = y <= cutoff_m[None, :, :]

    ycorr = np.where(low, ycorr1, ycorr2)

    # If COEFFS2 is invalid for a pixel, use COEFFS1 above cutoff.
    if np.any(~p2_good):
        ycorr = np.where(
            (~low) & (~p2_good[None, :, :]),
            ycorr1,
            ycorr,
        )

    # ------------------------------------------------------------
    # Finite safety check
    # ------------------------------------------------------------
    if check_finite:
        bad_value_read = valid_read & (
            ~np.isfinite(y) | ~np.isfinite(ycorr)
        )

        bad_value_pix = np.any(bad_value_read, axis=0)

        if np.any(bad_value_pix):
            lin_dq[:, bad_value_pix] |= LIN_NO_CORR | LIN_BAD_VALUE
            valid_read[:, bad_value_pix] = False

    # ------------------------------------------------------------
    # Monotonicity safety check
    # ------------------------------------------------------------
    if check_monotonic and N > 1:
        dycorr = np.diff(ycorr, axis=0)

        valid_pair = valid_read[1:, :, :] & valid_read[:-1, :, :]
        bad_step = valid_pair & (dycorr < monotonic_tolerance)

        bad_mono_pix = np.any(bad_step, axis=0)

        if np.any(bad_mono_pix):
            lin_dq[:, bad_mono_pix] |= LIN_NO_CORR | LIN_NONMONOTONIC
            valid_read[:, bad_mono_pix] = False

    # ------------------------------------------------------------
    # Write corrected values only for valid pre-saturation reads
    # ------------------------------------------------------------
    corrected[valid_read] = ycorr[valid_read].astype(np.float32)
    lin_dq[valid_read] |= LIN_APPLIED

    lin_mask = np.any(valid_read, axis=0)

    # ------------------------------------------------------------
    # Handle reads at/after saturation
    # ------------------------------------------------------------
    sat_read_valid_pix = sat_read & valid_pix[None, :, :]

    if invalid_read_behavior == "raw":
        pass

    elif invalid_read_behavior == "nan":
        corrected[sat_read_valid_pix] = np.nan

    elif invalid_read_behavior == "flat_last_valid":
        last_valid_index = np.maximum(k - 1, 0)

        rr, cc = np.indices((H, W))
        last_vals = corrected[last_valid_index, rr, cc]

        last_vals_3d = np.broadcast_to(
            last_vals[None, :, :],
            corrected.shape,
        )

        corrected[sat_read_valid_pix] = last_vals_3d[sat_read_valid_pix]

    if return_aux:
        return corrected.astype(np.float32), lin_dq, lin_mask

    return corrected.astype(np.float32)


################ diagnostics plots ##################################
def plot_linearity_correction_examples(
    coeff_file,
    ramp_cube,
    pixels=None,
    *,
    n_examples=4,
    ref_fraction=0.80,
    selection_mode="mixed",
    max_search_pixels=300000,
    random_seed=123,
    savefile=None,
    dpi=200,
):
    """
    Plot example pixels showing how the linearity correction is performed.

    Figure layout
    -------------
    Top row:
        Raw ramp M(t), linear reference L(t), corrected ramp.

    Bottom row:
        Measured-to-linearized mapping L(M), with ramp samples and
        fitted piecewise polynomial correction.

    Parameters
    ----------
    coeff_file : str
        FITS file produced by make_linearity_coeffs_final().

    ramp_cube : ndarray
        Original ramp cube with shape (n_reads, ny, nx).

    pixels : list of tuple, optional
        Specific pixels to plot, e.g. [(500, 500), (1000, 1000)].
        Pixel order is (row, column).

    n_examples : int
        Number of automatically selected example pixels if pixels is None.

    ref_fraction : float
        Reference signal fraction of saturation used to compute correction
        amplitude for pixel ranking.

    selection_mode : {"mixed", "strong", "typical"}
        Automatic pixel selection behavior.

        "mixed":
            Select representative weak, typical, strong, and high-improvement pixels.

        "strong":
            Select pixels with the largest absolute correction amplitude.

        "typical":
            Select pixels near the median correction amplitude.

    max_search_pixels : int
        Maximum number of good pixels used for automatic selection.

    random_seed : int
        Random seed for reproducible subsampling.

    savefile : str, optional
        Save figure to this path if provided.

    dpi : int
        Figure resolution for saved output.
    """

    # ------------------------------------------------------------
    # Load coefficient products
    # ------------------------------------------------------------
    with fits.open(coeff_file) as hdul:
        coeffs1 = hdul["COEFFS1"].data.astype(float)
        coeffs2 = hdul["COEFFS2"].data.astype(float)
        cutoff_m = hdul["CUTOFF_M"].data.astype(float)
        saturation = hdul["SATURATION"].data.astype(float)
        sat_index = hdul["SAT_INDEX"].data.astype(float)
        slope = hdul["SLOPE"].data.astype(float)
        intercept = hdul["INTERCEPT"].data.astype(float)
        goodpix = hdul["GOODPIX"].data.astype(bool)

        raw_rms = hdul["RAW_RMS"].data.astype(float)
        corr_rms = hdul["CORR_RMS"].data.astype(float)
        improve = hdul["IMPROVE"].data.astype(float)

    ramp_cube = np.asarray(ramp_cube, dtype=float)

    if ramp_cube.ndim != 3:
        raise ValueError("ramp_cube must have shape (n_reads, ny, nx)")

    n_reads, ny, nx = ramp_cube.shape

    if goodpix.shape != (ny, nx):
        raise ValueError("Coefficient file shape does not match ramp_cube shape")

    # ------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------
    def has_coeff(p):
        return np.all(np.isfinite(p))

    def apply_pixel_coeff_values(mvals, p1, p2, cm):
        mvals = np.asarray(mvals, dtype=float)
        out = np.full_like(mvals, np.nan, dtype=float)

        if not has_coeff(p1) or not np.isfinite(cm):
            return out

        low = mvals <= cm
        out[low] = np.polyval(p1, mvals[low])

        if has_coeff(p2):
            out[~low] = np.polyval(p2, mvals[~low])
        else:
            out[~low] = np.polyval(p1, mvals[~low])

        return out

    def apply_pixel_correction_ramp(raw, p1, p2, cm, si):
        raw = np.asarray(raw, dtype=float)

        if np.isfinite(si) and si >= 0:
            nuse = min(int(si), raw.size)
        else:
            nuse = raw.size

        raw_use = raw[:nuse]

        if raw_use.size == 0:
            return raw_use, raw_use

        corr = apply_pixel_coeff_values(raw_use, p1, p2, cm)

        if not np.all(np.isfinite(corr)):
            return raw_use, raw_use

        return raw_use, corr

    def fit_line(y):
        y = np.asarray(y, dtype=float)
        t = np.arange(y.size, dtype=float)
        good = np.isfinite(y)

        if np.count_nonzero(good) < 3:
            return None

        p = np.polyfit(t[good], y[good], 1)
        return np.polyval(p, t)

    def correction_amplitude_at_ref(r, c):
        sat = saturation[r, c]

        if not np.isfinite(sat) or sat <= 0:
            return np.nan

        mref = ref_fraction * sat

        p1 = coeffs1[:, r, c]
        p2 = coeffs2[:, r, c]
        cm = cutoff_m[r, c]

        lref = apply_pixel_coeff_values(np.array([mref]), p1, p2, cm)[0]

        if not np.isfinite(lref) or mref == 0:
            return np.nan

        return 100.0 * (lref - mref) / mref

    # ------------------------------------------------------------
    # Automatic pixel selection
    # ------------------------------------------------------------
    if pixels is None:
        candidates = np.argwhere(goodpix)

        if candidates.shape[0] == 0:
            raise RuntimeError("No GOODPIX pixels found")

        rng = np.random.default_rng(random_seed)

        if candidates.shape[0] > max_search_pixels:
            idx = rng.choice(
                candidates.shape[0],
                size=max_search_pixels,
                replace=False,
            )
            candidates = candidates[idx]

        amp = np.array([
            correction_amplitude_at_ref(r, c)
            for r, c in candidates
        ])

        imp = np.array([
            improve[r, c]
            for r, c in candidates
        ])

        good_metric = (
            np.isfinite(amp) &
            np.isfinite(imp) &
            (imp > 0)
        )

        candidates = candidates[good_metric]
        amp = amp[good_metric]
        imp = imp[good_metric]

        if candidates.shape[0] == 0:
            raise RuntimeError("No valid candidate pixels found for examples")

        abs_amp = np.abs(amp)

        chosen = []

        if selection_mode == "strong":
            order = np.argsort(abs_amp)[::-1]
            chosen = [tuple(candidates[i]) for i in order[:n_examples]]

        elif selection_mode == "typical":
            target = np.nanmedian(abs_amp)
            order = np.argsort(np.abs(abs_amp - target))
            chosen = [tuple(candidates[i]) for i in order[:n_examples]]

        elif selection_mode == "mixed":
            # Weak correction
            p20 = np.nanpercentile(abs_amp, 20)
            i_weak = np.argmin(np.abs(abs_amp - p20))
            chosen.append(tuple(candidates[i_weak]))

            # Typical correction
            p50 = np.nanpercentile(abs_amp, 50)
            i_typ = np.argmin(np.abs(abs_amp - p50))
            chosen.append(tuple(candidates[i_typ]))

            # Strong correction
            p90 = np.nanpercentile(abs_amp, 90)
            i_strong = np.argmin(np.abs(abs_amp - p90))
            chosen.append(tuple(candidates[i_strong]))

            # Highest RMS improvement
            i_best = np.nanargmax(imp)
            chosen.append(tuple(candidates[i_best]))

            # Trim or extend
            chosen = chosen[:n_examples]

            if len(chosen) < n_examples:
                order = np.argsort(abs_amp)[::-1]
                for i in order:
                    pix = tuple(candidates[i])
                    if pix not in chosen:
                        chosen.append(pix)
                    if len(chosen) >= n_examples:
                        break
        else:
            raise ValueError("selection_mode must be 'mixed', 'strong', or 'typical'")

        pixels = chosen

    else:
        pixels = [tuple(p) for p in pixels]
        n_examples = len(pixels)

    # ------------------------------------------------------------
    # Make figure
    # ------------------------------------------------------------
    fig, axs = plt.subplots(
        2,
        n_examples,
        figsize=(4.2 * n_examples, 7.2),
        squeeze=False,
    )

    #fig.suptitle(
    #    "Representative examples of linearity correction application",
    #    fontsize=15,
    #)

    for j, (r, c) in enumerate(pixels):
        if not (0 <= r < ny and 0 <= c < nx):
            raise ValueError(f"Pixel {(r, c)} is outside detector shape {(ny, nx)}")

        raw_full = ramp_cube[:, r, c]

        p1 = coeffs1[:, r, c]
        p2 = coeffs2[:, r, c]
        cm = cutoff_m[r, c]
        si = sat_index[r, c]

        raw, corr = apply_pixel_correction_ramp(raw_full, p1, p2, cm, si)

        t = np.arange(raw.size)

        # Linear reference from stored slope/intercept if available
        if np.isfinite(slope[r, c]) and np.isfinite(intercept[r, c]):
            ref = slope[r, c] * t + intercept[r, c]
        else:
            ref = fit_line(raw)

        if ref is None:
            ref = fit_line(raw)

        raw_line = fit_line(raw)
        corr_line = fit_line(corr)

        amp_ref = correction_amplitude_at_ref(r, c)
        imp_val = improve[r, c]
        raw_rms_val = raw_rms[r, c]
        corr_rms_val = corr_rms[r, c]

        # --------------------------------------------------------
        # Top row: ramp correction
        # --------------------------------------------------------
        ax = axs[0, j]

        ax.plot(t, raw, "ko", ms=3, label="Measured ramp $M(t)$")

        if ref is not None:
            ax.plot(t, ref, color="0.45", lw=2, ls="--", label="Linear reference $L(t)$")

        ax.plot(t, corr, "r-", lw=1.8, label="Corrected ramp")

        if np.isfinite(si) and si >= 0:
            ax.axvline(si, color="orange", ls=":", lw=1.5, label="Saturation index")

        ax.set_title(
            f"Pixel ({r}, {c})\n"
            rf"$\Delta_{{80}}$={amp_ref:.2f}\%, "
            rf"improve={imp_val:.2f}$\times$",
            fontsize=10,
        )
        ax.set_xlabel("Read number",fontsize=15)
        ax.set_ylabel("Signal [DN]",fontsize=15)
        ax.grid(alpha=0.25)

        if j == 0:
            ax.legend(fontsize=8, loc="best")

        # --------------------------------------------------------
        # Bottom row: measured-to-linear mapping
        # --------------------------------------------------------
        ax = axs[1, j]

        if ref is not None:
            ax.plot(raw, ref, "ko", ms=3, label="Ramp samples: $M \\rightarrow L$")
        else:
            ax.plot(raw, corr, "ko", ms=3, label="Ramp samples")

        finite_raw = np.isfinite(raw)

        if np.any(finite_raw):
            xmin = np.nanmin(raw[finite_raw])
            xmax = np.nanmax(raw[finite_raw])

            grid = np.linspace(xmin, xmax, 300)

            model = apply_pixel_coeff_values(grid, p1, p2, cm)

            ax.plot(
                grid,
                model,
                "r-",
                lw=2,
                label="Polynomial correction",
            )

            ax.plot(
                [xmin, xmax],
                [xmin, xmax],
                color="0.5",
                ls="--",
                lw=1,
                label="No correction",
            )

            if np.isfinite(cm):
                ax.axvline(
                    cm,
                    color="purple",
                    ls=":",
                    lw=1.5,
                    label="Cutoff $M$",
                )

        ax.set_xlabel("Measured signal $M$ [DN]",fontsize=15)
        ax.set_ylabel("Linearized signal $L$ [DN]",fontsize=15)
        ax.grid(alpha=0.25)

        ax.text(
            0.04,
            0.96,
            f"Raw RMS = {raw_rms_val:.3f} DN\n"
            f"Corr RMS = {corr_rms_val:.3f} DN",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )

        if j == 0:
            ax.legend(fontsize=8, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if savefile is not None:
        plt.savefig(savefile, dpi=dpi, bbox_inches="tight")

    plt.show()

#plot_linearity_correction_examples(
#    coeff_file=path+"lin_coeffs_ifs_slow_cd5.fits",
#    ramp_cube=lin_input_ifs,
#    selection_mode="mixed",
#    n_examples=4,
#    savefile="linearity_correction_examples_ifs.png",
#)

LIN_BPM          = np.uint8(1 << 0)
LIN_IDENTITY     = np.uint8(1 << 1)
LIN_SATURATED    = np.uint8(1 << 2)
LIN_BAD_VALUE    = np.uint8(1 << 3)
LIN_APPLIED      = np.uint8(1 << 4)
LIN_TOO_LARGE    = np.uint8(1 << 5)
DQ_BPM_INPUT           = np.uint32(1 << 0)
DQ_NONFINITE           = np.uint32(1 << 1)
DQ_LOW_DYNAMIC_RANGE   = np.uint32(1 << 2)
DQ_TOO_FEW_POINTS      = np.uint32(1 << 3)
DQ_NEGATIVE_JUMPS      = np.uint32(1 << 4)
DQ_NO_SATURATION_FOUND = np.uint32(1 << 5)
DQ_BAD_BASELINE        = np.uint32(1 << 6)
DQ_NO_BREAK_FOUND      = np.uint32(1 << 7)
DQ_BAD_LOWFIT          = np.uint32(1 << 8)
DQ_BAD_HIGHFIT         = np.uint32(1 << 9)
DQ_IDENTITY_USED       = np.uint32(1 << 10)
DQ_NONMONOTONIC        = np.uint32(1 << 11)
DQ_TRANSITION_BAD      = np.uint32(1 << 12)
DQ_RMS_TOO_HIGH        = np.uint32(1 << 13)
DQ_UNPHYSICAL_CORR     = np.uint32(1 << 14)


# CORR_TYPE values
CORR_INVALID  = np.uint8(0)  # BPM / unusable pixel
CORR_IDENTITY = np.uint8(1)  # safe identity transform L=M
CORR_FITTED   = np.uint8(2)  # accepted two-piece correction

def apply_linearity_coeffs_to_cube_safe_fast1(
    input_cube,
    coeff_file,
    bpm_2d=None,
    *,
    invalid_read_behavior="raw",
    max_corr_frac=None,
    max_corr_abs=None,
    chunk_size=65536,
    return_aux=False,
):
    """
    Apply safe linearity coefficients to a read cube.

    Identity pixels are copied unchanged.

    Fitted corrections are applied only to reads before SAT_INDEX.
    Reads at/after SAT_INDEX are marked with LIN_SATURATED for ramp fitting.

    Returns
    -------
    corrected_cube : float32, shape (N,H,W)

    If return_aux=True:
        corrected_cube, lin_dq, lin_mask
    """

    cube = np.asarray(input_cube, dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError("input_cube must have shape (N,H,W)")

    n_reads, ny, nx = cube.shape
    npix = ny * nx

    if invalid_read_behavior not in {
        "raw",
        "flat_last_valid",
        "nan",
    }:
        raise ValueError(
            "invalid_read_behavior must be "
            "'raw', 'flat_last_valid', or 'nan'"
        )

    if bpm_2d is None:
        bpm = np.zeros((ny, nx), dtype=bool)
    else:
        bpm = np.asarray(bpm_2d) != 0
        if bpm.shape != (ny, nx):
            raise ValueError("bpm_2d must have shape (H,W)")

    with fits.open(coeff_file) as hdul:
        coeffs1 = hdul["COEFFS1"].data.astype(np.float64)
        coeffs2 = hdul["COEFFS2"].data.astype(np.float64)
        cutoff_m = hdul["CUTOFF_M"].data.astype(np.float64)
        sat_index = hdul["SAT_INDEX"].data.astype(np.int32)

        if "CORR_TYPE" in hdul:
            corr_type = hdul["CORR_TYPE"].data.astype(np.uint8)
        else:
            goodpix = hdul["GOODPIX"].data.astype(bool)
            corr_type = np.where(
                goodpix,
                CORR_FITTED,
                CORR_INVALID,
            ).astype(np.uint8)

        coeff_bpm = (
            hdul["BPM_INPUT"].data.astype(bool)
            if "BPM_INPUT" in hdul
            else np.zeros((ny, nx), dtype=bool)
        )

        hdr = hdul[0].header

    if coeffs1.shape[1:] != (ny, nx):
        raise ValueError(
            "Coefficient spatial shape does not match input cube"
        )

    if coeffs2.shape[1:] != (ny, nx):
        raise ValueError(
            "Coefficient spatial shape does not match input cube"
        )

    if cutoff_m.shape != (ny, nx):
        raise ValueError("CUTOFF_M shape mismatch")

    if sat_index.shape != (ny, nx):
        raise ValueError("SAT_INDEX shape mismatch")

    if max_corr_frac is None:
        max_corr_frac = float(hdr.get("MAXCFR", 0.05))

    if max_corr_abs is None:
        max_corr_abs = float(hdr.get("MAXCABS", 50.0))

    corrected = cube.copy()

    # uint8 is sufficient for the current six flags and is much smaller
    # than uint32 for a full read cube.
    lin_dq = np.zeros(
        cube.shape,
        dtype=np.uint8,
    )

    lin_mask = np.zeros((ny, nx), dtype=bool)

    cube_flat = cube.reshape(n_reads, npix)
    corrected_flat = corrected.reshape(n_reads, npix)
    dq_flat = lin_dq.reshape(n_reads, npix)

    bpm_combined = bpm | coeff_bpm
    bpm_flat = bpm_combined.ravel()

    corr_type_flat = corr_type.ravel()
    cutoff_flat = cutoff_m.ravel()
    sat_flat = sat_index.ravel()

    coeffs1_flat = coeffs1.reshape(
        coeffs1.shape[0],
        npix,
    )
    coeffs2_flat = coeffs2.reshape(
        coeffs2.shape[0],
        npix,
    )

    read_number = np.arange(n_reads)[:, None]

    # ------------------------------------------------------------
    # BPM pixels
    # ------------------------------------------------------------
    bpm_indices = np.flatnonzero(bpm_flat)
    if bpm_indices.size:
        dq_flat[:, bpm_indices] |= LIN_BPM

    # ------------------------------------------------------------
    # Identity pixels: leave data exactly unchanged
    # ------------------------------------------------------------
    identity_indices = np.flatnonzero(
        (~bpm_flat)
        & (corr_type_flat == CORR_IDENTITY)
    )

    if identity_indices.size:
        dq_flat[:, identity_indices] |= LIN_IDENTITY
        lin_mask.ravel()[identity_indices] = True

        identity_sat = sat_flat[identity_indices]
        valid_sat = (
            (identity_sat >= 0)
            & (identity_sat < n_reads)
        )

        for start in range(
            0,
            identity_indices.size,
            chunk_size,
        ):
            idx = identity_indices[start:start + chunk_size]
            kval = sat_flat[idx]

            sat_mask = (
                (kval[None, :] >= 0)
                & (read_number >= kval[None, :])
            )

            dq_flat[:, idx] |= (
                sat_mask.astype(np.uint8)
                * LIN_SATURATED
            )

    # ------------------------------------------------------------
    # Fitted pixels
    # ------------------------------------------------------------
    fitted_indices = np.flatnonzero(
        (~bpm_flat)
        & (corr_type_flat == CORR_FITTED)
    )

    for start in tqdm(
        range(0, fitted_indices.size, chunk_size),
        desc="Applying fitted linearity correction",
    ):
        idx = fitted_indices[start:start + chunk_size]

        y = np.asarray(
            cube_flat[:, idx],
            dtype=np.float64,
        )

        cm = cutoff_flat[idx]
        kval = sat_flat[idx]

        # No detected saturation means all reads are valid.
        k_effective = np.where(
            kval < 0,
            n_reads,
            np.clip(kval, 0, n_reads),
        )

        pre_sat = read_number < k_effective[None, :]
        post_sat = ~pre_sat

        # Evaluate polynomial maps with Horner's method.
        low_value = np.zeros_like(y, dtype=np.float64)
        high_value = np.zeros_like(y, dtype=np.float64)

        for coefficient in coeffs1_flat[:, idx]:
            low_value = low_value * y + coefficient[None, :]

        for coefficient in coeffs2_flat[:, idx]:
            high_value = high_value * y + coefficient[None, :]

        mapped = np.where(
            y <= cm[None, :],
            low_value,
            high_value,
        )

        finite_ok = np.isfinite(mapped) & np.isfinite(y)

        delta = mapped - y
        allowed_delta = np.maximum(
            max_corr_abs,
            max_corr_frac * np.maximum(np.abs(y), 1.0),
        )

        amplitude_ok = np.abs(delta) <= allowed_delta

        valid_correction = (
            pre_sat
            & finite_ok
            & amplitude_ok
        )

        # Apply only safe pre-saturation values.
        output_chunk = corrected_flat[:, idx]

        output_chunk[valid_correction] = (
            mapped[valid_correction].astype(np.float32)
        )

        corrected_flat[:, idx] = output_chunk

        dq_chunk = dq_flat[:, idx]

        dq_chunk[valid_correction] |= LIN_APPLIED
        dq_chunk[pre_sat & ~finite_ok] |= LIN_BAD_VALUE
        dq_chunk[pre_sat & finite_ok & ~amplitude_ok] |= LIN_TOO_LARGE
        dq_chunk[post_sat] |= LIN_SATURATED

        dq_flat[:, idx] = dq_chunk

        # A pixel is considered applied if every finite pre-saturation
        # read passed the application checks.
        successful_pixel = np.all(
            (~pre_sat) | valid_correction,
            axis=0,
        )

        lin_mask.ravel()[idx[successful_pixel]] = True

        # Handle post-saturation values.
        if invalid_read_behavior == "nan":
            output_chunk = corrected_flat[:, idx]
            output_chunk[post_sat] = np.nan
            corrected_flat[:, idx] = output_chunk

        elif invalid_read_behavior == "flat_last_valid":
            output_chunk = corrected_flat[:, idx]

            for j, kpix in enumerate(k_effective):
                if 0 < kpix < n_reads:
                    output_chunk[kpix:, j] = output_chunk[kpix - 1, j]

            corrected_flat[:, idx] = output_chunk

        # "raw" requires no assignment because corrected started as a copy.

    if return_aux:
        return corrected, lin_dq, lin_mask

    return corrected

import numpy as np
from astropy.io import fits
from tqdm import tqdm


# ============================================================
# CORR_TYPE values stored in coefficient file
# ============================================================
CORR_INVALID = np.uint8(0)   # BPM / unusable coefficient entry
CORR_IDENTITY = np.uint8(1)  # identity mapping: L = M
CORR_FITTED = np.uint8(2)    # accepted fitted piecewise correction


# ============================================================
# Read-level linearity DQ flags
#
# IMPORTANT:
# LIN_SATURATED must use the same bit value expected by the
# ramp-fitting code.
# ============================================================
LIN_BPM = np.uint8(1 << 0)
LIN_IDENTITY = np.uint8(1 << 1)
LIN_SATURATED = np.uint8(1 << 2)
LIN_BAD_COEFF = np.uint8(1 << 3)
LIN_APPLIED = np.uint8(1 << 4)
LIN_BAD_VALUE = np.uint8(1 << 5)


def apply_linearity_coeffs_to_cube_safe_fast(
    input_cube,
    coeff_file,
    bpm_2d=None,
    *,
    invalid_read_behavior="raw",
    chunk_size=4096,
    return_aux=False,
):
    """
    Apply safe measured-DN -> linearized-DN coefficients to a read cube.

    Correction policy
    -----------------
    CORR_INVALID:
        Leave the complete pixel ramp unchanged and flag it.

    CORR_IDENTITY:
        Leave the complete pixel ramp unchanged. Saturated reads are still
        flagged using SAT_INDEX.

    CORR_FITTED:
        Apply the fitted piecewise polynomial to every finite pre-saturation
        read. If any coefficient or mapped pre-saturation value is invalid,
        leave the entire pixel ramp unchanged.

    No per-read mixture of corrected and uncorrected values is allowed.

    Parameters
    ----------
    input_cube : ndarray, shape (N, H, W)
        Input cube of measured detector reads.

    coeff_file : str
        FITS coefficient file containing COEFFS1, COEFFS2, CUTOFF_M,
        SAT_INDEX, and preferably CORR_TYPE.

    bpm_2d : ndarray, optional, shape (H, W)
        Additional bad-pixel mask. Nonzero pixels are not corrected.

    invalid_read_behavior : {"raw", "flat_last_valid", "nan"}
        Treatment of reads at/after SAT_INDEX.

        "raw"
            Preserve the original values. Recommended before ramp fitting,
            because group DQ will exclude these reads.

        "flat_last_valid"
            Replace post-saturation reads with the final valid corrected read.

        "nan"
            Replace post-saturation reads with NaN.

    chunk_size : int
        Number of fitted pixels processed per chunk. Values around
        2048--8192 are usually suitable for long H2RG ramps.

    return_aux : bool
        If True, return corrected_cube, lin_dq, and lin_mask.

    Returns
    -------
    corrected_cube : ndarray, float32, shape (N, H, W)
        Corrected cube. Pixels that cannot be safely corrected remain equal
        to the input cube.

    lin_dq : ndarray, uint8, shape (N, H, W), optional
        Read-level DQ cube.

    lin_mask : ndarray, bool, shape (H, W), optional
        True only where an actual fitted correction was successfully applied.
        Identity pixels remain False.
    """

    # ------------------------------------------------------------
    # Validate input cube
    # ------------------------------------------------------------
    cube = np.asarray(input_cube, dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError("input_cube must have shape (N, H, W)")

    n_reads, ny, nx = cube.shape
    npix = ny * nx

    if invalid_read_behavior not in {
        "raw",
        "flat_last_valid",
        "nan",
    }:
        raise ValueError(
            "invalid_read_behavior must be "
            "'raw', 'flat_last_valid', or 'nan'"
        )

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    # ------------------------------------------------------------
    # Additional BPM supplied at application time
    # ------------------------------------------------------------
    if bpm_2d is None:
        bpm = np.zeros((ny, nx), dtype=bool)
    else:
        bpm = np.asarray(bpm_2d) != 0

        if bpm.shape != (ny, nx):
            raise ValueError(
                f"bpm_2d must have shape {(ny, nx)}, got {bpm.shape}"
            )

    # ------------------------------------------------------------
    # Load calibration products
    # ------------------------------------------------------------
    with fits.open(coeff_file, memmap=True) as hdul:
        coeffs1 = np.asarray(
            hdul["COEFFS1"].data,
            dtype=np.float64,
        )
        coeffs2 = np.asarray(
            hdul["COEFFS2"].data,
            dtype=np.float64,
        )
        cutoff_m = np.asarray(
            hdul["CUTOFF_M"].data,
            dtype=np.float64,
        )
        sat_index = np.asarray(
            hdul["SAT_INDEX"].data,
            dtype=np.int32,
        )

        coeff_bpm = (
            np.asarray(hdul["BPM_INPUT"].data, dtype=bool)
            if "BPM_INPUT" in hdul
            else np.zeros((ny, nx), dtype=bool)
        )

        if "CORR_TYPE" in hdul:
            corr_type = np.asarray(
                hdul["CORR_TYPE"].data,
                dtype=np.uint8,
            )
        else:
            # Compatibility fallback for older files.
            if "GOODPIX" in hdul:
                goodpix = np.asarray(
                    hdul["GOODPIX"].data,
                    dtype=bool,
                )
            else:
                goodpix = np.ones((ny, nx), dtype=bool)

            p1_finite = np.all(np.isfinite(coeffs1), axis=0)
            p2_finite = np.all(np.isfinite(coeffs2), axis=0)

            corr_type = np.full(
                (ny, nx),
                CORR_INVALID,
                dtype=np.uint8,
            )

            usable = goodpix & p1_finite & p2_finite
            corr_type[usable] = CORR_FITTED

    # ------------------------------------------------------------
    # Shape checks
    # ------------------------------------------------------------
    expected_shape = (ny, nx)

    if coeffs1.shape[1:] != expected_shape:
        raise ValueError(
            "COEFFS1 spatial shape does not match input cube: "
            f"{coeffs1.shape[1:]} versus {expected_shape}"
        )

    if coeffs2.shape[1:] != expected_shape:
        raise ValueError(
            "COEFFS2 spatial shape does not match input cube: "
            f"{coeffs2.shape[1:]} versus {expected_shape}"
        )

    if cutoff_m.shape != expected_shape:
        raise ValueError(
            f"CUTOFF_M must have shape {expected_shape}"
        )

    if sat_index.shape != expected_shape:
        raise ValueError(
            f"SAT_INDEX must have shape {expected_shape}"
        )

    if corr_type.shape != expected_shape:
        raise ValueError(
            f"CORR_TYPE must have shape {expected_shape}"
        )

    if coeff_bpm.shape != expected_shape:
        raise ValueError(
            f"BPM_INPUT must have shape {expected_shape}"
        )

    # ------------------------------------------------------------
    # Initialize output as an exact copy of input
    #
    # This is the main safety fallback. Unless a complete pixel ramp
    # passes all application checks, its values remain unchanged.
    # ------------------------------------------------------------
    corrected = cube.copy()

    lin_dq = np.zeros(
        cube.shape,
        dtype=np.uint8,
    )

    # True only when a fitted correction is applied successfully.
    lin_mask = np.zeros(
        (ny, nx),
        dtype=bool,
    )

    # ------------------------------------------------------------
    # Flatten spatial dimensions
    # ------------------------------------------------------------
    cube_flat = cube.reshape(n_reads, npix)
    corrected_flat = corrected.reshape(n_reads, npix)
    dq_flat = lin_dq.reshape(n_reads, npix)
    lin_mask_flat = lin_mask.ravel()

    bpm_combined = bpm | coeff_bpm
    bpm_flat = bpm_combined.ravel()

    corr_type_flat = corr_type.ravel()
    cutoff_flat = cutoff_m.ravel()
    sat_flat = sat_index.ravel()

    coeffs1_flat = coeffs1.reshape(
        coeffs1.shape[0],
        npix,
    )
    coeffs2_flat = coeffs2.reshape(
        coeffs2.shape[0],
        npix,
    )

    read_number = np.arange(
        n_reads,
        dtype=np.int32,
    )[:, None]

    # ------------------------------------------------------------
    # BPM / invalid pixels
    # ------------------------------------------------------------
    bpm_indices = np.flatnonzero(bpm_flat)

    if bpm_indices.size:
        dq_flat[:, bpm_indices] |= LIN_BPM

    invalid_indices = np.flatnonzero(
        (~bpm_flat)
        & (corr_type_flat == CORR_INVALID)
    )

    if invalid_indices.size:
        dq_flat[:, invalid_indices] |= LIN_BAD_COEFF

    # ------------------------------------------------------------
    # Identity pixels
    #
    # Data remain unchanged. We only propagate saturation information.
    # ------------------------------------------------------------
    identity_indices = np.flatnonzero(
        (~bpm_flat)
        & (corr_type_flat == CORR_IDENTITY)
    )

    for start in range(
        0,
        identity_indices.size,
        chunk_size,
    ):
        idx = identity_indices[start:start + chunk_size]

        dq_flat[:, idx] |= LIN_IDENTITY

        kval = sat_flat[idx]

        # SAT_INDEX < 0 means no detected saturation.
        effective_k = np.where(
            kval < 0,
            n_reads,
            np.clip(kval, 0, n_reads),
        )

        post_sat = (
            read_number >= effective_k[None, :]
        )

        dq_chunk = dq_flat[:, idx]
        dq_chunk[post_sat] |= LIN_SATURATED
        dq_flat[:, idx] = dq_chunk

        if invalid_read_behavior == "nan":
            output_chunk = corrected_flat[:, idx]
            output_chunk[post_sat] = np.nan
            corrected_flat[:, idx] = output_chunk

        elif invalid_read_behavior == "flat_last_valid":
            output_chunk = corrected_flat[:, idx]

            for j, kpix in enumerate(effective_k):
                if 0 < kpix < n_reads:
                    output_chunk[kpix:, j] = output_chunk[kpix - 1, j]

            corrected_flat[:, idx] = output_chunk

        # "raw": no change is required.

    # ------------------------------------------------------------
    # Fitted pixels
    # ------------------------------------------------------------
    fitted_indices = np.flatnonzero(
        (~bpm_flat)
        & (corr_type_flat == CORR_FITTED)
    )

    for start in tqdm(
        range(0, fitted_indices.size, chunk_size),
        desc="Applying fitted linearity correction",
    ):
        idx = fitted_indices[start:start + chunk_size]
        n_chunk = idx.size

        # Input values for this spatial chunk.
        y = np.asarray(
            cube_flat[:, idx],
            dtype=np.float64,
        )

        cm = cutoff_flat[idx]
        kval = sat_flat[idx]

        effective_k = np.where(
            kval < 0,
            n_reads,
            np.clip(kval, 0, n_reads),
        )

        pre_sat = (
            read_number < effective_k[None, :]
        )
        post_sat = ~pre_sat

        # --------------------------------------------------------
        # Pixel-level coefficient validation
        # --------------------------------------------------------
        p1 = coeffs1_flat[:, idx]
        p2 = coeffs2_flat[:, idx]

        coeff_ok = (
            np.all(np.isfinite(p1), axis=0)
            & np.all(np.isfinite(p2), axis=0)
            & np.isfinite(cm)
            & (effective_k > 0)
        )

        # --------------------------------------------------------
        # Prevent polynomial evaluation outside valid reads
        #
        # Post-saturation values are replaced by zero only for polynomial
        # evaluation. Their original values remain in corrected_flat.
        # --------------------------------------------------------
        y_eval = np.where(
            pre_sat,
            y,
            0.0,
        )

        # --------------------------------------------------------
        # Evaluate low branch with Horner's method
        # --------------------------------------------------------
        low_value = np.zeros_like(
            y_eval,
            dtype=np.float64,
        )

        for coefficient_row in p1:
            low_value = (
                low_value * y_eval
                + coefficient_row[None, :]
            )

        # --------------------------------------------------------
        # Evaluate high branch with Horner's method
        # --------------------------------------------------------
        high_value = np.zeros_like(
            y_eval,
            dtype=np.float64,
        )

        for coefficient_row in p2:
            high_value = (
                high_value * y_eval
                + coefficient_row[None, :]
            )

        low_branch = y_eval <= cm[None, :]

        mapped = np.where(
            low_branch,
            low_value,
            high_value,
        )

        # Only pre-saturation reads are relevant.
        read_finite = (
            np.isfinite(y)
            & np.isfinite(mapped)
        )

        # --------------------------------------------------------
        # All-or-nothing pixel decision
        #
        # Every pre-saturation read must be finite. If one fails,
        # the complete pixel ramp remains unchanged.
        # --------------------------------------------------------
        pixel_ok = (
            coeff_ok
            & np.all(
                (~pre_sat) | read_finite,
                axis=0,
            )
        )

        good_columns = np.flatnonzero(pixel_ok)
        bad_columns = np.flatnonzero(~pixel_ok)

        # --------------------------------------------------------
        # Apply every pre-saturation read for accepted pixels
        # --------------------------------------------------------
        if good_columns.size:
            output_chunk = corrected_flat[:, idx]

            apply_mask = (
                pre_sat[:, good_columns]
            )

            mapped_good = mapped[:, good_columns]

            output_good = output_chunk[:, good_columns]
            output_good[apply_mask] = (
                mapped_good[apply_mask].astype(np.float32)
            )

            output_chunk[:, good_columns] = output_good
            corrected_flat[:, idx] = output_chunk

            dq_chunk = dq_flat[:, idx]

            dq_good = dq_chunk[:, good_columns]
            dq_good[apply_mask] |= LIN_APPLIED
            dq_good[post_sat[:, good_columns]] |= LIN_SATURATED

            dq_chunk[:, good_columns] = dq_good
            dq_flat[:, idx] = dq_chunk

            lin_mask_flat[idx[good_columns]] = True

        # --------------------------------------------------------
        # Rejected fitted pixels remain exactly raw
        # --------------------------------------------------------
        if bad_columns.size:
            dq_chunk = dq_flat[:, idx]

            dq_bad = dq_chunk[:, bad_columns]
            dq_bad |= LIN_BAD_VALUE

            dq_bad[post_sat[:, bad_columns]] |= LIN_SATURATED

            dq_chunk[:, bad_columns] = dq_bad
            dq_flat[:, idx] = dq_chunk

        # --------------------------------------------------------
        # Post-saturation data treatment
        # --------------------------------------------------------
        if invalid_read_behavior == "nan":
            output_chunk = corrected_flat[:, idx]
            output_chunk[post_sat] = np.nan
            corrected_flat[:, idx] = output_chunk

        elif invalid_read_behavior == "flat_last_valid":
            output_chunk = corrected_flat[:, idx]

            for j, kpix in enumerate(effective_k):
                if 0 < kpix < n_reads:
                    output_chunk[kpix:, j] = output_chunk[kpix - 1, j]

            corrected_flat[:, idx] = output_chunk

        # "raw": post-saturation values remain equal to input.

    # ------------------------------------------------------------
    # Return output
    # ------------------------------------------------------------
    corrected = corrected_flat.reshape(
        n_reads,
        ny,
        nx,
    )

    lin_dq = dq_flat.reshape(
        n_reads,
        ny,
        nx,
    )

    lin_mask = lin_mask_flat.reshape(
        ny,
        nx,
    )

    if return_aux:
        return (
            corrected.astype(np.float32, copy=False),
            lin_dq,
            lin_mask,
        )

    return corrected.astype(np.float32, copy=False)
