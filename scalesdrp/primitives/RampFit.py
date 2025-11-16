from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import scalesdrp.primitives.fitramp as fitramp
import scalesdrp.primitives.robust as robust
import numpy as np
from astropy.io import fits
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import time
import os
import pkg_resources
from scipy.signal import savgol_filter
import scalesdrp.primitives.reference as reference #1/f and reference pixel correction
import scalesdrp.primitives.linearity as linearity #linearity correction
import scalesdrp.primitives.bpm_correction as bpm #bpm correction
from tqdm import tqdm
from astropy.nddata import StdDevUncertainty

class RampFit(BasePrimitive):

    """
    We adopt the ramp fitting method of Brandt et. al. 2024 for reads greater than 5. 
    This method perform an optimal fit to a pixel’s count rate nondestructively in the
    presence of both read and photon noise. The method construct a covarience matrix by
    estimating the difference in the read in a ramp, propagation of the read noise,
    photon noise and their corelation. And Performs a generalized least squares fit
    to the differences, using the inverse of the covariance matrix as weights.
    This gives optimal weight to each difference. The readnoise per pixel is estimated
    from the drak frames. The jumps are detected iteratively checking the goodness of
    fit at each possible jump location. 
        Args:
            data_image: The (N,H,W) input ramp cube.
            read_noise: The (N_pixels,N_pixels) 2D vector of read noise.
            

        Returns:
            A 2D image of ramp fitted slope
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger

    ############## ramp fit for reads less than 5############################
    # per-row OLS slope uncertainty (1σ) using valid reads ---
    def _ols_row_and_uncert(self,row_reads,             # (N, W) counts
                        valid_reads_mask,      # (N, W) bool
                        t,                     # (N,) seconds
                        sig_row):              # (W,) single-read σ (e-)
        """
        Return (OLS_slope_row, OLS_slope_uncert_row).
        OLS slope uncertainty:  σ_m = σ_read / sqrt( Σ_i (t_i - tbar)^2 ), using only valid reads.
        This is the classic unweighted OLS formula (appropriate when each read has the same σ per pixel).
        """
        N, W = row_reads.shape
        v = valid_reads_mask.astype(bool)

        # counts of valid reads per pixel
        S0 = v.sum(axis=0)                                    # (W,)
        S0_safe = np.maximum(S0, 1)

        # time statistics on valid reads
        St  = (t[:, None] * v).sum(axis=0)                    # (W,)
        tbar = St / S0_safe                                   # (W,)
        Stt_centered = (((t[:, None] - tbar) ** 2) * v).sum(axis=0)  # (W,)

        # OLS slope (per pixel) using valid reads
        y   = np.where(v, row_reads, 0.0)
        Sy  = y.sum(axis=0)
        Sty = (t[:, None] * y).sum(axis=0)
        # slope = Cov(t,y)/Var(t) = (Σ t y - tbar Σ y) / Σ (t - tbar)^2
        num = Sty - tbar * Sy
        den = np.where(Stt_centered > 0, Stt_centered, np.nan)
        slope_row = num / den

        # 1σ on slope (e-/s); if <2 reads valid, set NaN
        slope_unc_row = sig_row / np.sqrt(den)
        slope_unc_row[(S0 < 2) | ~np.isfinite(den)] = np.nan

        return slope_row, slope_unc_row

    ########### ramp fit for read >5 ########################################
    def ramp_fit(self,input_read, total_exptime, SIG_map_scaled, *,
        return_pedestal=True,
        reset_prior_strength=3.0, # prior σ = k * SIG per pixel
        use_sigma_clip=False,  # optional; physics mask usually suffices
        sigma_clip=3.0, max_iter=3, min_reads=5, tile=(128, 128),
        JUMP_THRESH_ONEOMIT=20.25,
        JUMP_THRESH_TWOOMIT=23.8):
        """
        produce two images—slope (countrate) and pedestal (reset) using fitramp everywhere it’s well-posed,
        and fall back to a robust OLS only where fitramp can’t run. Optionally clean reads,
        mask physically impossible read pairs, and (optionally) detect jumps.
    
        Prefer fitramp (pedestal=True) for any number of reads; OLS only as per-pixel fallback.
        - Saturation/rollover handled via Δread > 0 mask.
        - Jump detection layered on top.
        - Finite reset prior from first valid read.
        """
        N, H, W = input_read.shape
        dt = float(total_exptime) / N
        t  = (np.arange(N, dtype=float) + 0.5) * dt
        t1=time.time()
        # (optional) σ-clip (off by default)
        def _sigma_clip_reads(self,cube):
            from tqdm import tqdm
            keep = np.ones_like(cube, dtype=bool)
            Ty, Tx = tile
            for y0 in tqdm(range(0, H, Ty), desc="σ-clip tiles"):
                y1 = min(H, y0+Ty)
                for x0 in range(0, W, Tx):
                    x1 = min(W, x0+Tx)
                    sub = cube[:, y0:y1, x0:x1]
                    n, ty, tx = sub.shape
                    k = ty*tx
                    Y = sub.reshape(n, k)
                    mask = np.isfinite(Y)
                    time_idx = np.arange(n, dtype=np.float32)
                    for _ in range(max_iter):
                        cnt = mask.sum(0)
                        if not np.any(cnt >= min_reads): break
                        S0  = cnt
                        St  = time_idx @ mask
                        Stt = (time_idx**2) @ mask
                        Wy  = Y * mask
                        Sy  = Wy.sum(0)
                        Sty = time_idx @ Wy
                        Var_t = Stt - (St*St)/np.maximum(S0, 1)
                        Cov_ty = Sty - (St*Sy)/np.maximum(S0, 1)
                        b = np.zeros(k, np.float32)
                        ok = Var_t > 0
                        b[ok] = Cov_ty[ok] / Var_t[ok]
                        a = (Sy - b*St) / np.maximum(S0, 1)
                        Yhat = a + np.outer(time_idx, b)
                        resid = Y - Yhat
                        resid[~mask] = np.nan
                        s = np.nanstd(resid, 0)
                        new_mask = (np.abs(resid) < sigma_clip*s) & np.isfinite(Y)
                        if np.array_equal(new_mask, mask): break
                        mask = new_mask
                    keep[:, y0:y1, x0:x1] = mask.reshape(n, ty, tx)
            return keep

        base_valid = np.isfinite(input_read)
        if use_sigma_clip:
            base_valid &= self._sigma_clip_reads(input_read)

        # differences and physics mask
        diffs = input_read[1:] - input_read[:-1]
        ## both reads valid & Δread must be positive
        pair_mask = (base_valid[1:] & base_valid[:-1]) & (diffs > 0)

        # ---------- Reset prior: first valid read; else no prior (σ=∞) ----------
        #If the first resultant is valid, use it as the prior mean on the pedestal.
        #Prior uncertainty is reset_prior_strength × SIG (finite). If first read is bad, set σ=∞ (flat prior).
        #With few usable differences, a finite reset prior makes the joint 
        #(pedestal+slope) fit solvable and better conditioned.

        first_read = input_read[0]
        first_ok = base_valid[0] & np.isfinite(first_read)
        resetval_map = np.where(first_ok, first_read, 0.0)
        resetsig_map = np.where(first_ok, reset_prior_strength * SIG_map_scaled, np.inf)

        C_no  = fitramp.Covar(t, pedestal=False)
        C_ped = fitramp.Covar(t, pedestal=True)

        slope  = np.full((H, W), np.nan, float)
        ped    = np.full((H, W), np.nan, float)
        uncert = np.full((H, W), np.nan, float)

        for i in range(H):
            if i % 128 == 0:
                print(f"Fitting row {i}/{H}...")

            sig_row   = SIG_map_scaled[i, :]
            d_row     = diffs[:, i, :]
            m_row     = pair_mask[:, i, :]
            resetval  = resetval_map[i, :]
            resetsig  = resetsig_map[i, :]

            Wrow = d_row.shape[1]
            row_slope  = np.full(Wrow, np.nan, float)
            row_ped    = np.full(Wrow, np.nan, float)
            row_uncert = np.full(Wrow, np.nan, float)

            # 0) Usable diffs BEFORE jump detection
            usable0 = m_row.sum(axis=0)

            #If a pixel has ≥2 usable diffs, we can get a fitramp-based seed for its slope.
            #Otherwise, seed comes from the robust OLS fallback.
            # 1) Initial seed via fitramp only where well-posed (≥2 diffs). Else OLS seed.
            #The final (pedestal=True) optimization benefits from a decent slope initial guess.
            seed = np.full(Wrow, np.nan, float)
            idx_seed_fit = (usable0 >= 2)
            idx_seed_ols = ~idx_seed_fit

            #Use pedestal=False for speed/robustness; countrateguess=None lets fitramp infer it from the data.
            #If something throws, we switch those pixels to OLS seeding.
            if np.any(idx_seed_fit):
                try:
                    d_sub   = d_row[:, idx_seed_fit]
                    m_sub   = m_row[:, idx_seed_fit]
                    sig_sub = sig_row[idx_seed_fit]
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        init_res = fitramp.fit_ramps(d_sub, C_no, sig_sub,
                            diffs2use=m_sub,
                            countrateguess=None,
                            rescale=True)
                    seed[idx_seed_fit] = init_res.countrate
                except Exception:
                    idx_seed_ols |= idx_seed_fit

            if np.any(idx_seed_ols):
                # compute OLS for the whole row once and slice
                ols_row, _ = self._ols_row_and_uncert(input_read[:, i, :],  # (N,W)
                    base_valid[:, i, :],
                    t, sig_row)
                seed[idx_seed_ols] = ols_row[idx_seed_ols]

            # fill non-finite seeds with OLS
            bad = ~np.isfinite(seed)
            if np.any(bad):
                ols_row, _ = self._ols_row_and_uncert(input_read[:, i, :],
                    base_valid[:, i, :],
                    t, sig_row)
                seed[bad] = ols_row[bad]

            # 2) Jump detection (only where we have ≥1 diff)
            #Likelihood-based cosmic-ray/jump search; returns a binary mask for diffs to keep.
            #Drop diffs that substantially improve χ² when omitted (thresholds ≈ 4.5σ false-positive equivalent).
            m_row2 = m_row.copy()
            idx_jump = (usable0 >= 1)
            if np.any(idx_jump):
                try:
                    d_sub   = d_row[:, idx_jump]
                    m_sub   = m_row[:, idx_jump]
                    sig_sub = sig_row[idx_jump]
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        m2_sub, _ = fitramp.mask_jumps(d_sub, C_no, sig_sub,
                            threshold_oneomit=JUMP_THRESH_ONEOMIT,
                            threshold_twoomit=JUMP_THRESH_TWOOMIT,
                            diffs2use=m_sub)
                    m_row2[:, idx_jump] = m2_sub
                except Exception:
                    pass

            usable_final = m_row2.sum(axis=0)

            # 3) Final fit with pedestal=True
            # Solvable if:
            #   - usable_final >= 1 AND prior finite, OR
            #   - usable_final >= 2 (solvable even without a prior)
            #This is the high-fidelity fit that jointly estimates pedestal and slope using the full covariance model.
            idx_final_fit = (usable_final >= 1) & (np.isfinite(resetsig) | (usable_final >= 2))
            if np.any(idx_final_fit):
                try:
                    d_sub    = d_row[:, idx_final_fit]
                    m_sub    = m_row2[:, idx_final_fit]
                    sig_sub  = sig_row[idx_final_fit]
                    seed_sub = seed[idx_final_fit]
                    r0_sub   = resetval[idx_final_fit]
                    s0_sub   = resetsig[idx_final_fit]
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        final_res = fitramp.fit_ramps(d_sub, C_ped, sig_sub,
                            diffs2use=m_sub,
                            countrateguess=seed_sub,
                            resetval=r0_sub, resetsig=s0_sub,
                            rescale=True)
                    row_slope[idx_final_fit]  = final_res.countrate
                    row_ped[idx_final_fit]    = final_res.pedestal
                    row_uncert[idx_final_fit] = final_res.uncert     # <-- from fitramp
                except Exception:
                    pass

            # 4) Per-pixel fallback: unsolved, zero usable diffs, or excluded by gating
            #If final fit is missing, use OLS slope and the prior mean for pedestal.
            need_fallback = (~np.isfinite(row_slope)) | (usable_final == 0) | (~idx_final_fit)
            if np.any(need_fallback):
                ols_row, ols_unc = self._ols_row_and_uncert(input_read[:, i, :],
                    base_valid[:, i, :],
                    t, sig_row)
                row_slope[need_fallback]  = ols_row[need_fallback]
                row_uncert[need_fallback] = ols_unc[need_fallback]
                row_ped[need_fallback]    = resetval[need_fallback]  # prior mean

            # 5) Final sanitation: any remaining non-finite → seed; then OLS
            bad_final = ~np.isfinite(row_slope)
            if np.any(bad_final):
                ols_row, ols_unc = self._ols_row_and_uncert(input_read[:, i, :],
                    base_valid[:, i, :],
                    t, sig_row)
                row_slope[bad_final]  = ols_row[bad_final]
                row_uncert[bad_final] = ols_unc[bad_final]
                row_ped[bad_final]    = resetval[bad_final]

            slope[i, :]  = row_slope
            ped[i,   :]  = row_ped
            uncert[i, :] = row_uncert
        t2=time.time()
        self.logger.info(f"Ramp fitting completed in {t2 - t1:.3f} seconds.")
        return (slope, ped, uncert) if return_pedestal else (slope, uncert)


    ####################### getting the master files needed from /redux #####################
    def load_single_master_file(self,expected_keywords, master_type):
        base = os.path.join(os.getcwd(), "redux")
        master_type = master_type.upper()
        obs_mode = (expected_keywords.get("OBSMODE") or "").upper()
        ifs_mode = (expected_keywords.get("IFSMODE") or "").upper()

        # canonical endings we will search for
        tail_map = {
            "DARK": "_mdark.fits",
            "BIAS": "_mbias.fits",
            "FLATLAMP": "_mflatlamp.fits",
            "FLATLENS": "_mflatlens.fits",}

        # lenslet flats keep their long names
        lenslet_tails = [
            "LowRes-K_master_lensflat.fits",
            "LowRes-L_master_lensflat.fits",
            "LowRes-M_master_lensflat.fits",
            "LowRes-SED_master_lensflat.fits",
            "LowRes-H2O_master_lensflat.fits",
            "LowRes-PAH_master_lensflat.fits",
            "MedRes-L_master_lensflat.fits",
            "MedRes-M_master_lensflat.fits",]

        if not os.path.isdir(base):
            return (None,None)

        # collect candidate filenames in redux/
        all_files = [f for f in os.listdir(base) if f.lower().endswith(".fits")]

        candidates = []

        if master_type == "LENSFLAT":
            # choose files whose name ends with the matching lenslet flat tail
            # and then we'll let header matching decide
            for f in all_files:
                for tail in lenslet_tails:
                    if f.endswith(tail):
                        candidates.append(os.path.join(base, f))
                        break
        else:
            tail = tail_map.get(master_type)
            if not tail:
                return None
            # match anything like "*_master_dark.fits"
            for f in all_files:
                if f.endswith(tail):
                    candidates.append(os.path.join(base, f))

        if not candidates:
            return (None,None)

        # now open candidates and check headers
        for path in candidates:
            try:
                with fits.open(path) as hdul:
                    hdr = hdul[0].header
                    data = hdul[0].data
                    uncert = hdul['UNCERT'].data
            except Exception:
                continue

            # basic type check: IMTYPE in file must match requested master_type
            # (if file doesn't have IMTYPE, we just skip it)
            imtype = (hdr.get("IMTYPE") or "").upper()
            if imtype and imtype != master_type:
                # not the right kind of master
                continue

            # for lens flats, also check IFSMODE from header vs expected
            if master_type == "LENSFLAT":
                file_ifs = (hdr.get("IFSMODE") or "").upper()
                if ifs_mode and file_ifs and file_ifs != ifs_mode:
                    continue

            # now check the rest of the expected keywords
            mismatch = False
            for key, exp_val in expected_keywords.items():
                if key in ("IMTYPE", "IMTYPE"):  # user-supplied science header may have different type
                    continue
                actual_val = hdr.get(key)
                if actual_val != exp_val and exp_val is not None:
                    mismatch = True
                    break

            if mismatch:
                continue

            # if we reach here, this file matches everything we care about
            return (data,uncert)

        # no candidate matched fully
        return (None,None)

    ####################### calib correction #############################
    def apply_calibration(
        self,
        data,                # (H,W) science data
        sigma_data,          # (H,W) 1σ of science data
        calib,               # (H,W) calibration frame: bias/dark/flat
        sigma_calib,         # (H,W) 1σ of calibration frame
        imtype,              # str from header, e.g. 'BIAS','DARK','FLAT'
        *,
        scale: float = 1.0,  # optional scale for bias/dark (e.g. exposure ratio)
        eps: float = 1e-10,  # avoid divide-by-zero for flats
        clip_nan: bool = True):
        """
        Return (corrected_data, corrected_sigma) with error propagation.

        If IMTYPE is 'BIAS' or 'DARK' (case-insensitive):
            out = data - scale * calib
            σ_out = sqrt( σ_data^2 + (scale^2) * σ_calib^2 )

        If IMTYPE is 'FLAT' (normalized flat):
            out = data / calib
            σ_out = sqrt( σ_data^2 / calib^2 + data^2 * σ_calib^2 / calib^4 )

        Notes:
        - All arrays must be same shape and numeric.
        - For subtraction, units of data & calib (and their σ) must match.
        - For flat-fielding, calib should be normalized (unitless, ~1.0 mean).
        - `scale` is applied only for BIAS/DARK (ignored for FLAT).
        """
        data        = np.asarray(data, dtype=float)
        sigma_data  = np.asarray(sigma_data, dtype=float)
        calib       = np.asarray(calib, dtype=float)
        sigma_calib = np.asarray(sigma_calib, dtype=float)

        kind = (imtype or "").strip().upper()

        if kind in ("BIAS", "DARK"):
            out = data - scale * calib
            sig = np.sqrt(sigma_data**2 + (scale**2) * sigma_calib**2)

        elif kind in ("FLATLAMP", 'FLATLENS'):
            denom = np.where(np.isfinite(calib) & (np.abs(calib) > eps), calib, np.nan)
            out = data / denom
            sig = np.sqrt( (sigma_data**2) / (denom**2) + (data**2) * (sigma_calib**2) / (denom**4) )

        else:
            raise ValueError(f"Unrecognized IMTYPE='{imtype}'. Expected 'BIAS', 'DARK', or 'FLAT'.")

        if clip_nan:
            good = np.isfinite(out) & np.isfinite(sig)
            out = np.where(good, out, np.nan)
            sig = np.where(good, sig, np.nan)
        return out.astype(np.float32), sig.astype(np.float32)

    ####################### flat normalization    ###################
    def normalize_detector_flat(
        self,
        flat,                     # (H,W) master flat (same units as science)
        flat_sigma,               # (H,W) 1σ uncertainty of master flat (same units)
        mask=None,                # (H,W) bool, True = invalid pixel
        method='median',          # 'median' or 'mean' for normalization constant
        clip_sigma=7.0,
        iterations=3,
        eps=1e-12,
        return_norm_stats=False):
        """
        Normalize a master flat and propagate uncertainty.

        For each pixel i:
            f_i_norm = F_i / c
            σ_i_norm^2 ≈ (σ_Fi^2 / c^2) + (F_i^2 * σ_c^2 / c^4)

        where c is the robust central value (median or mean) of the valid pixels,
        estimated after sigma-clipping; σ_c is the standard error of c.

        Notes
        -----
        * Correlation between F_i and c is ignored (good approximation for large N).
        * For 'median':   σ_c ≈ 1.253 * σ_sample / sqrt(N_eff)
        For 'mean'  :   σ_c =      σ_sample / sqrt(N_eff)
        σ_sample is estimated robustly via MAD; falls back to std if needed.
        * Invalid pixels (mask | non-finite | <=0) return NaN in both outputs.
        """
        flat      = np.asarray(flat, dtype=np.float64)
        flat_sigma= np.asarray(flat_sigma, dtype=np.float64)
        if flat.shape != flat_sigma.shape:
            raise ValueError("flat and flat_sigma must have identical shapes")

        invalid = ~np.isfinite(flat) | ~np.isfinite(flat_sigma) | (flat <= 0)
        if mask is not None:
            invalid |= np.asarray(mask, dtype=bool)

        valid = ~invalid
        valid_data = flat[valid]
        if valid_data.size == 0:
            raise ValueError("No valid pixels found in flat for normalization.")

        # --- sigma clip the valid sample to define the normalization pool ---
        clipped = valid_data.copy()
        for _ in range(iterations):
            med = np.median(clipped)
            # robust scatter via MAD; fallback to std if MAD==0
            mad = np.median(np.abs(clipped - med))
            robust_sigma = mad / 0.67448975 if mad > 0 else np.std(clipped)
            if robust_sigma == 0 or not np.isfinite(robust_sigma):
                break
            low, high = med - clip_sigma * robust_sigma, med + clip_sigma * robust_sigma
            keep = (clipped > low) & (clipped < high)
            if np.all(keep):
                break
            clipped = clipped[keep]

        if clipped.size == 0:
            raise ValueError("All flat pixels rejected during clipping; cannot normalize.")

        # normalization constant (c) and its standard error (σ_c)
        if method.lower() == 'median':
            c = np.median(clipped)
            # recompute robust sigma on the final clipped set
            mad = np.median(np.abs(clipped - c))
            sig_sample = mad / 0.67448975 if mad > 0 else np.std(clipped)
            sigma_c = 1.2533141373155001 * sig_sample / np.sqrt(clipped.size)  # ≈ sqrt(pi/2)
        elif method.lower() == 'mean':
            c = np.mean(clipped)
            sig_sample = np.std(clipped, ddof=1) if clipped.size > 1 else 0.0
            sigma_c = sig_sample / np.sqrt(clipped.size) if clipped.size > 0 else np.inf
        else:
            raise ValueError("method must be 'median' or 'mean'")
        if not np.isfinite(c) or c <= 0:
            raise ValueError(f"Invalid normalization constant: {c}")

        # --- propagate uncertainties per pixel ---
        c_safe = float(c)
        c2 = c_safe**2
        c4 = c_safe**4
        # main formulas
        flat_norm = np.full_like(flat, np.nan, dtype=np.float64)
        sigma_norm= np.full_like(flat, np.nan, dtype=np.float64)

        # avoid division by zero / nan propagation; invalid already handled
        denom = np.where(valid, c_safe, np.nan)

        flat_norm[valid]  = flat[valid] / denom[valid]
        # Var(f/c) = Var(f)/c^2 + f^2 * Var(c)/c^4
        sigma_norm[valid] = np.sqrt( (flat_sigma[valid]**2) / c2 + (flat[valid]**2) * (sigma_c**2) / c4 )

        # cast to float32 for storage if desired
        flat_norm  = flat_norm.astype(np.float32)
        sigma_norm = sigma_norm.astype(np.float32)

        if return_norm_stats:
            return flat_norm, sigma_norm, (float(c), float(sigma_c), int(clipped.size))
        return flat_norm, sigma_norm

    ############# even odd swaping (optional) ############################
    def swap_odd_even_columns(self,cube,n_amps=4,do_swap=True):
        if do_swap:
            nreads,n_rows,n_cols = cube.shape
            ramp = np.empty_like(cube)
            block = n_cols // n_amps
            for a in range(n_amps):
                x0, x1 = a * block, (a + 1) * block
                sub = cube[..., x0:x1]
                nsub = sub.shape[-1]
                new_order = []
                for i in range(0, nsub, 2):
                    if i + 1 < nsub:
                        new_order.extend([i + 1, i])
                    else:
                        new_order.append(i)
                ramp[..., x0:x1] = sub[..., new_order]
        return ramp
    ####################################################################################
        
    def _perform(self):

        imtype = self.action.args.ccddata.header['IMTYPE']
        if imtype =='OBJECT':
            total_exptime = self.action.args.ccddata.header['EXPTIME']
            obsmode = self.action.args.ccddata.header['OBSMODE']
            calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
        
            if obsmode =='IMAGING':
                SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits')

            elif obsmode =='IFS':
                SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits')

            input_data = self.action.args.ccddata.data
            #print(input_data.shape)
            self.logger.info("+++++++++++ odd even column swapping +++++++++++")
            sci_im_full_original = self.swap_odd_even_columns(input_data)
            #sci_im_full_original = reference.reffix_hxrg(input_data, nchans=4, fixcol=True)
            #self.logger.info("refpix and 1/f correction completed")
            #saturation_map = linearity.create_saturation_map_by_slope(
            #    science_ramp=sci_im_full_original1,
            #    skip_reads=3,
            #    slope_threshold=2.0
            #    smoothing_window=3)
        
            #final_ramp, final_pixel_dq, final_group_dq = linearity.run_linearity_workflow(
            #    science_ramp=sci_im_full_original1,
            #    saturation_map=saturation_map,
            #    obsmode = obsmode)

            nim_s = sci_im_full_original.shape[0]
            self.logger.info("+++++++++++ ramp fitting started +++++++++++")

            final_slope,final_reset,final_uncert = self.ramp_fit(sci_im_full_original, total_exptime, SIG_map_scaled)
            #neg_bpm = (final_slope < 0).astype(np.uint8)

            self.logger.info("+++++++++++ Bad pixel correction started +++++++++++")
            final_ramp = bpm.apply_full_correction(final_slope,obsmode)

            keywords_unique = {
                key: self.action.args.ccddata.header.get(key)
                for key in ['OBSMODE', 'IFSMODE', 'MCLOCK']}

            m_dark, m_dark_uncert = self.load_single_master_file(keywords_unique, master_type='DARK')
            m_bias, m_bias_uncert = self.load_single_master_file(keywords_unique, master_type='BIAS')
            m_flat, m_flat_uncert = self.load_single_master_file(keywords_unique, master_type='FLATLAMP')

            if m_dark is not None:
                final_ramp, final_uncert = self.apply_calibration(
                    final_ramp,
                    final_uncert,
                    m_dark,
                    m_dark_uncert,
                    imtype='DARK')
                self.action.args.ccddata.header['HISTORY'] = 'Dark subtracted.'
                self.logger.info("+++++++++++ Master dark subtracted +++++++++++")
        
            if m_bias is not None:
                final_ramp, final_uncert = self.apply_calibration(
                    final_ramp,
                    final_uncert,
                    m_bias,
                    m_bias_uncert,
                    imtype='BIAS')
                self.action.args.ccddata.header['HISTORY'] = 'Bias subtracted.'
                self.logger.info("+++++++++++ Master bias subtracted +++++++++++")

            if m_flat is not None:
                norm_flat,norm_flat_uncert = self.normalize_detector_flat(m_flat,m_flat_uncert)
                final_ramp, final_uncert = self.apply_calibration(
                    final_ramp,
                    final_uncert,
                    norm_flat,
                    norm_flat_uncert,
                    imtype='FLATLAMP')
                self.action.args.ccddata.header['HISTORY'] = 'Detector Flat correction applied.'
                self.logger.info("+++++++++++ detector flat correction completed  +++++++++++")

            self.action.args.ccddata.data = final_ramp
            self.action.args.ccddata.uncertainty = StdDevUncertainty(final_uncert.astype(np.float32))

            log_string = RampFit.__module__
            self.action.args.ccddata.header['HISTORY'] = log_string
            self.logger.info(log_string)

            scales_fits_writer(self.action.args.ccddata,
                table=self.action.args.table,
                output_file=self.action.args.name,
                output_dir=self.config.instrument.output_directory,
                suffix="L1_ramp")
            self.logger.info("+++++++++++ slope image FITS file saved +++++++++++")
        else:
            self.logger.info("+++++++++++ No science files detected to process +++++++++++")
        return self.action.args
    # END: class RampFit()
