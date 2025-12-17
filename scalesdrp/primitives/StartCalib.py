from keckdrpframework.primitives.base_primitive import BasePrimitive
import scalesdrp.primitives.fitramp as fitramp
import scalesdrp.primitives.robust as robust
import scalesdrp.primitives.reference as reference #1/f and reference pixel correction
import scalesdrp.primitives.linearity as linearity #linearity correction
import scalesdrp.primitives.bpm_correction as bpm #bpm correction
import pandas as pd
import numpy as np
import pickle
from astropy.io import fits
import warnings
import pkg_resources
from scipy import sparse
import astropy.io.fits as pyfits
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import time
from scipy.signal import savgol_filter
from scipy.ndimage import convolve, median_filter
from scipy.interpolate import griddata
from scipy.stats import median_abs_deviation
from scipy.ndimage import median_filter
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve as astropy_convolve
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
import pkg_resources
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scalesdrp.core.matplot_plotting import mpl_plot, mpl_clear
from tqdm import tqdm
from scalesdrp.primitives.linearity import DQ_FLAGS

class StartCalib(BasePrimitive):
    """
    Estimate the psf centroid of all the calib images images and save
    in two pickle file one for x and one for y centroid values. Currently
    assumes wavelengths will be in filenames. Need to replace that with
    header keywords instead. Also the location of the calib filesneed to fix.
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger
        #print('action arguments in primitive',self.action.args.dirname)

    def group_files_by_header(self, dt):
        """
        Groups files in the data table based on observing mode and key header values.

        Rules:
        - If OBSMODE == 'IMAGING': group by (OBSMODE, IM-FW-1, IMTYPE, EXPTIME)
        - If OBSMODE in ('LOWRES', 'MEDRES'): group by (OBSMODE, IFSMODE, IMTYPE, EXPTIME)
        """

        required_cols = ['OBSMODE', 'IFSMODE', 'IM-FW-1', 'IMTYPE', 'EXPTIME','MCLOCK']
        missing = [c for c in required_cols if c not in dt.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return []

        file_groups = []

        # --- Split by observing mode first
        for mode, mode_df in dt.groupby('OBSMODE'):
            if mode.upper() == 'IMAGING':
                group_keys = ['OBSMODE', 'IM-FW-1', 'IMTYPE', 'EXPTIME','MCLOCK']
                self.logger.info(f"Grouping IMAGING data by {group_keys}")
            elif mode.upper() =='IFS':
                group_keys = ['OBSMODE', 'IFSMODE', 'IMTYPE', 'EXPTIME','MCLOCK']
                self.logger.info(f"Grouping {mode} data by {group_keys}")
            else:
                self.logger.warning(f"Unknown OBSMODE '{mode}'; skipping.")
                continue

            grouped = mode_df.groupby(group_keys)

            # Build output list
            for group_params, sub_df in grouped:
                group_info = {
                    'params': {key.lower(): val for key, val in zip(group_keys, group_params)},
                    'filenames': sub_df.index.tolist()
                }
                file_groups.append(group_info)

        if not file_groups:
            self.logger.warning("No file groups were created.")
        else:
            self.logger.info(f"Created {len(file_groups)} groups from data table.")

        return file_groups

############# ouput fits writing ##############################
    def fits_writer_steps1(self,data,header,output_dir,input_filename,suffix,overwrite=True):
        base_name = os.path.basename(input_filename)
        file_root, file_ext = os.path.splitext(base_name)
        output_filename = f"{file_root}{suffix}{file_ext}"
        redux_output_dir = os.path.join(output_dir, 'redux')
        os.makedirs(redux_output_dir, exist_ok=True)
        output_path = os.path.join(redux_output_dir, output_filename)
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_path, overwrite=overwrite)
        return output_path

    def fits_writer_steps(
        self,
        data,                  # 2D or 3D array
        header,                # FITS header (astropy.io.fits.Header)
        output_dir,            # base output directory
        input_filename,        # original filename (used for naming)
        suffix,                # string appended to filename
        overwrite=True,
        uncert=None            # optional uncertainty array (same shape as data)
    ):
        """
        Write data (and optional uncertainty) to a FITS file inside redux/.

        Parameters
        ----------
        data : ndarray
            calibration array.
        header : fits.Header
            FITS header for the primary HDU.
        output_dir : str
            Output directory (redux/ will be created inside this).
        input_filename : str
            Original filename, used to derive output filename.
        suffix : str
            Suffix appended before extension (e.g. '_mdark', '_mflat').
        overwrite : bool, optional
            Whether to overwrite existing file (default: True).
        uncert : ndarray, optional
            Uncertainty array, same shape as data. Saved as 'UNCERT' extension.

        Returns
        -------
        output_path : str
            Full path to the written FITS file.
        """
        # --- ensure directories exist ---
        os.makedirs(os.path.join(output_dir, "redux"), exist_ok=True)

        # --- construct filename ---
        base_name = os.path.basename(input_filename)
        file_root, file_ext = os.path.splitext(base_name)
        output_filename = f"{file_root}{suffix}{file_ext}"
        output_path = os.path.join(output_dir, "redux", output_filename)

        # --- build HDUList ---
        hdus = [fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=header)]

        # add uncertainty extension if provided
        if uncert is not None:
            if uncert.shape != data.shape:
                warnings.warn(f"Uncertainty shape {uncert.shape} does not match data {data.shape}; skipping UNCERT extension.")
            else:
                hdu_uncert = fits.ImageHDU(data=np.asarray(uncert, dtype=np.float32), name="UNCERT")
                hdus.append(hdu_uncert)
        hdul = fits.HDUList(hdus)
        # --- write to disk ---
        hdul.writeto(output_path, overwrite=overwrite)
        # --- optional: return path for downstream pipeline ---
        return output_path

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

################## ramp fitting ############################
    def ramp_fit(self,input_read, total_exptime, SIG_map_scaled, *,
        return_pedestal=True,
        reset_prior_strength=3.0, # prior σ = k * SIG per pixel
        use_sigma_clip=False,  # optional; physics mask usually suffices
        sigma_clip=3.0, max_iter=3, min_reads=5, tile=(128, 128),
        JUMP_THRESH_ONEOMIT=20.25,
        JUMP_THRESH_TWOOMIT=23.8,
        group_dq=None):
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

        if group_dq is not None:
            saturated = (group_dq & DQ_FLAGS["SATURATED"]) != 0
            base_valid &= ~saturated
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

################## swapping ######################################
    def swap_odd_even_columns(self,cube,n_amps=4,do_swap=True):
        if not do_swap:
            return cube
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

################### master file ###############################
    def build_master_from_stack(
        self,
        data_stack,                 # (N,H,W) or list of (H,W)
        sigma_stack=None,           # None or (N,H,W) or list of (H,W)
        *,
        method="ivw",               # 'ivw' (inverse-variance mean), 'mean', or 'median'
        clip_sigma=5.0,             # sigma threshold for iterative clipping
        iterations=3,               # max clipping iterations
        min_valid=2,                # minimum frames per pixel to keep result
        frame_scales=None,          # optional length-N array of multiplicative scales per frame
        return_mask=False           # optionally return the final inlier mask (N,H,W)
    ):
        """
        Combine a stack of frames into a master calibration frame with uncertainty.

        Parameters
        ----------
        data_stack : array-like
            Stack of input frames, shape (N,H,W) or list of (H,W).
        sigma_stack : array-like or None
        Matching stack of 1σ uncertainties; if None, equal-variance is assumed.
        method : {'ivw','mean','median'}
            ivw   = inverse-variance weighted mean (requires sigma_stack)
            mean  = unweighted mean
            median= unweighted median with robust uncertainty estimate
        clip_sigma : float
            Sigma threshold for iterative outlier rejection (applied along the N axis).
        iterations : int
            Maximum number of clipping iterations.
        min_valid : int
            Minimum number of surviving frames per pixel; otherwise result is NaN.
        frame_scales : array-like or None
            Optional length-N multiplicative scales applied per frame (e.g. normalize by exposure).
        return_mask : bool
            If True, also return the final boolean inlier mask of shape (N,H,W).
        Returns
        -------
        master : (H,W) float32
            Combined master frame.
        master_uncert : (H,W) float32
            Propagated 1σ uncertainty of the master.
        mask (optional) : (N,H,W) bool
            Final inlier mask after clipping (only if return_mask=True).
        """
        # --- normalize inputs to arrays ---
        data = np.asarray(data_stack)
        if data.ndim == 3:
            N, H, W = data.shape
        elif isinstance(data_stack, (list, tuple)) and np.asarray(data_stack[0]).ndim == 2:
            data = np.stack([np.asarray(f) for f in data_stack], axis=0)
            N, H, W = data.shape
        else:
            raise ValueError("data_stack must be (N,H,W) or list of (H,W).")
        if sigma_stack is None:
            sigma = None
        else:
            sigma = np.asarray(sigma_stack)
            if sigma.shape != data.shape:
                # allow list input
                if isinstance(sigma_stack, (list, tuple)) and len(sigma_stack) == N:
                    sigma = np.stack([np.asarray(s) for s in sigma_stack], axis=0)
                else:
                    raise ValueError(f"sigma_stack shape {np.asarray(sigma_stack).shape} must match data {data.shape}.")
        # --- optional per-frame scaling ---
        if frame_scales is not None:
            scales = np.asarray(frame_scales, dtype=float)
            if scales.shape != (N,):
                raise ValueError(f"frame_scales must have shape ({N},).")
            data = data * scales[:, None, None]
            if sigma is not None:
                sigma = sigma * np.abs(scales[:, None, None])

        # --- initial mask: finite data (and finite sigma if provided & > 0) ---
        mask = np.isfinite(data)
        if sigma is not None:
            mask &= np.isfinite(sigma) & (sigma > 0)

        # --- iterative clipping along the stack axis ---
        m = mask.copy()
        for _ in range(max(0, int(iterations))):
            # center estimate by method
            if method == "ivw" and sigma is not None:
                w = np.where(m, 1.0 / np.maximum(sigma, 1e-30)**2, 0.0)
                Wsum = np.sum(w, axis=0)
                center = np.where(Wsum > 0, np.sum(w * data, axis=0) / Wsum, np.nan)
                # weighted residual "sigma": use weighted std
                resid = (data - center[None, :, :])
                var = np.where(Wsum > 0, np.sum(w * resid**2, axis=0) / np.maximum(Wsum, 1e-30), np.nan)
                s = np.sqrt(var)
            elif method == "mean":
                valid = np.where(m, data, np.nan)
                center = np.nanmean(valid, axis=0)
                s = np.nanstd(valid, axis=0)
            elif method == "median":
                valid = np.where(m, data, np.nan)
                center = np.nanmedian(valid, axis=0)
                # robust scale via MAD
                mad = np.nanmedian(np.abs(valid - center[None, :, :]), axis=0)
                s = 1.4826 * mad  # MAD→σ
            else:
                raise ValueError("method must be 'ivw', 'mean', or 'median'.")

            if not np.isfinite(clip_sigma) or clip_sigma <= 0:
                break

            # update mask: keep |resid| <= clip_sigma * s
            # compute residuals
            resid = np.abs(data - center[None, :, :])
            thresh = clip_sigma * np.where(np.isfinite(s), s, np.nan)
            keep = resid <= thresh[None, :, :]
            # always keep currently invalid pixels as False
            m = m & keep

            # stop early if nothing changes
            if np.array_equal(m, keep & mask):
                break

        # --- final combine with remaining mask m ---
        n_eff = np.sum(m, axis=0)

        master = np.full((H, W), np.nan, dtype=np.float64)
        master_unc = np.full((H, W), np.nan, dtype=np.float64)

        good_pix = n_eff >= max(1, int(min_valid))

        if method == "ivw" and sigma is not None:
            w = np.where(m, 1.0 / np.maximum(sigma, 1e-30)**2, 0.0)
            Wsum = np.sum(w, axis=0)
            # master = sum(w*x)/sum(w)
            val = np.where(Wsum > 0, np.sum(w * data, axis=0) / Wsum, np.nan)
            # σ_master = 1/sqrt(sum w)
            unc = np.where(Wsum > 0, 1.0 / np.sqrt(Wsum), np.nan)
            master[good_pix] = val[good_pix]
            master_unc[good_pix] = unc[good_pix]

        elif method == "mean":
            valid = np.where(m, data, np.nan)
            val = np.nanmean(valid, axis=0)
            s = np.nanstd(valid, axis=0)
            # σ_master ≈ s / sqrt(n_eff)
            unc = np.where(n_eff > 0, s / np.sqrt(np.maximum(n_eff, 1)), np.nan)
            master[good_pix] = val[good_pix]
            master_unc[good_pix] = unc[good_pix]

        elif method == "median":
            valid = np.where(m, data, np.nan)
            val = np.nanmedian(valid, axis=0)
            # robust per-pixel spread via MAD
            mad = np.nanmedian(np.abs(valid - val[None, :, :]), axis=0)
            # σ ~ 1.4826*MAD; σ_median ≈ 1.2533*σ/sqrt(n_eff)
            sigma_robust = 1.4826 * mad
            unc = np.where(
                n_eff > 0,
                1.2533 * sigma_robust / np.sqrt(np.maximum(n_eff, 1)),
                np.nan)
            master[good_pix] = val[good_pix]
            master_unc[good_pix] = unc[good_pix]

        # enforce min_valid
        master[~good_pix] = np.nan
        master_unc[~good_pix] = np.nan

        if return_mask:
            return master.astype(np.float32), master_unc.astype(np.float32), m
        return master.astype(np.float32), master_unc.astype(np.float32)

    def optimal_extract_with_error(
        self,
        R_transpose,
        data_image,
        sigma_image,
        read_noise_variance_vector,gain = 1.0):

        self.logger.info('Optimal extraction started')
        start_time1 = time.time()
        data_vector_d = data_image.flatten().astype(np.float64)
        sigma_vector = sigma_image.array.flatten().astype(np.float64)
        variance_from_map = sigma_vector**2
        photon_noise_variance = data_vector_d.clip(min=0) / gain
        total_variance = read_noise_variance_vector + photon_noise_variance
        #total_variance[total_variance <= 0] = 1e-9
        total_variance = read_noise_variance_vector + photon_noise_variance + variance_from_map
        inverse_variance = 1.0 / total_variance
        weighted_data = data_vector_d * inverse_variance
        numerator = R_transpose @ weighted_data
        R_transpose_squared = R_transpose.power(2)
        denominator = R_transpose_squared @ inverse_variance
        denominator_safe = np.maximum(denominator, 1e-9)
        optimized_flux = numerator / denominator_safe
        flux_variance = 1.0 / denominator_safe
        flux_error = np.sqrt(flux_variance)
        end_time1 = time.time()
        t1 = (end_time1 - start_time1)
        self.logger.info(f"Optimal extraction finished in {t1:.4f} seconds.")
        return optimized_flux, flux_error
########################################### START MAIN ######################

    def _perform(self):
        self.logger.info("+++++++++++ SCALES calibration starting +++++++++++")
        dt = self.context.data_set.data_table
        all_groups = self.group_files_by_header(dt)
        if not all_groups:
            self.logger.warning("No file groups found. Nothing to process.")
            return
        
        ols_t_global = None
        organized_groups = {}
        for group in all_groups:
            params = group['params']
            imtype = params.get('imtype', 'UNKNOWN')
            if imtype not in organized_groups:
                organized_groups[imtype] = []
            organized_groups[imtype].append(group)

        processing_order = ['BIAS', 'DARK', 'FLATLAMP','FLATLENS', 'CALUNIT']
        self.logger.info(f"Found groups for IMTYPEs: {list(organized_groups.keys())}")
        bias_ramps=[]
        dark_ramps=[]
        flatlamp_ramps=[]
        flatlen_ramps=[]
        calunit_ramps=[]

        bias_ramps_uncert=[]
        dark_ramps_uncert=[]
        flatlamp_ramps_uncert=[]
        flatlen_ramps_uncert=[]
        calunit_ramps_uncert=[]

        for imtype in processing_order:
            if imtype not in organized_groups:
                continue
            groups_for_this_type = organized_groups[imtype]
            for group in groups_for_this_type:
                params = group['params']
                filenames = group['filenames']
                obsmode = params.get('obsmode', 'UNKNOWN')
                ifsmode = params.get('ifsmode', 'N/A')
                filtername = params.get('im-fw-1', 'N/A')
                exptime = params.get('exptime', 0)
                mclock = params.get('mclock', 0)
                self.logger.info(f"Processing {imtype}: {len(filenames)} files "
                    f"(OBSMODE={obsmode}, IFSMODE={ifsmode}, FILTER={filtername}, EXPTIME={exptime}, MCLOCK={mclock})")
                for filename in filenames:
                    try:
                        with fits.open(filename) as hdulist:
                            #sci_im_full_original1 = hdulist[0]
                            sci_im_full_original1 = hdulist[0].data
                            data_header = hdulist[0].header
                            readtime = data_header['EXPTIME']
                    except Exception as e:
                        self.logger.error(f"Failed to read {filename}: {e}")

                    calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
                    if obsmode =='IMAGING':
                        SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits')

                    elif obsmode =='IFS':
                        SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits')

                    self.logger.info("+++++++++++ odd even swapping +++++++++++")
                    sci_im_full_original2 = self.swap_odd_even_columns(sci_im_full_original1,do_swap=True)

                    self.logger.info("+++++++++++ ACN & 1/f correction started +++++++++++")
                    sci_im_full_original3 = reference.reffix_hxrg(sci_im_full_original2, nchans=4, fixcol=True)

                    self.logger.info("+++++++++++ linearity correction started +++++++++++")
                    if obsmode =='IMAGING':
                        corrected_input_ramp, pixeldq, groupdq, cutoff_map, sat_map = linearity.run_linearity_workflow(
                            sci_im_full_original3,
                            linearity_file="linearity_coeffs_img.fits")

                    if obsmode =='IFS':
                        corrected_input_ramp, pixeldq, groupdq, cutoff_map, sat_map = linearity.run_linearity_workflow(
                            sci_im_full_original3,
                            linearity_file="linearity_coeffs_img.fits")

                    self.logger.info("+++++++++++ ramp fitting started +++++++++++")
                    slope,reset,uncert = self.ramp_fit(
                        corrected_input_ramp,
                        readtime,
                        SIG_map_scaled,
                        group_dq = groupdq)

                    self.logger.info("+++++++++++ Bad pixel correction started +++++++++++")
                    bpm_slope = bpm.apply_full_correction(slope,obsmode)
                    bpm_slope_uncert = bpm.apply_full_correction(uncert,obsmode)
                    
                    self.fits_writer_steps(
                        data=bpm_slope,
                        header=data_header,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix='_L1_ramp',
                        overwrite=True,
                        uncert = bpm_slope_uncert)

                                #self.context.proctab.new_proctab()
                                #name1 = data_header['IMTYPE']
                                
                                #self.context.proctab.update_proctab(
                                #    frame=sci_im_full_original1, suffix="ramp", newtype='name1',
                                #    filename=data_header['OFNAME'])

                                #self.context.proctab.write_proctab(
                                #    tfil=self.config.instrument.procfile)
                                
                    if imtype == 'BIAS':
                        bias_ramps.append(bpm_slope)
                        bias_ramps_uncert.append(bpm_slope_uncert)
                        bias_header = data_header
                    
                    if imtype == 'DARK':
                        dark_ramps.append(bpm_slope)
                        dark_ramps_uncert.append(bpm_slope_uncert)
                        dark_header = data_header

                    if imtype == 'FLATLENS':
                        flatlen_ramps.append(bpm_slope)
                        flatlen_ramps_uncert.append(bpm_slope_uncert)
                        flatlen_header = data_header
                    
                    if imtype == 'FLATLAMP':
                        flatlamp_ramps.append(bpm_slope)
                        flatlamp_ramps_uncert.append(bpm_slope_uncert)
                        flatlamp_header = data_header
                    
                    if imtype == 'CALUNIT':
                        calunit_ramps.append(bpm_slope)
                        calunit_ramps_uncert.append(bpm_slope_uncert)
                        calunit_header = data_header

        if len(dark_ramps) > 0:
            master_dark, master_dark_uncert = self.build_master_from_stack(
                dark_ramps,
                dark_ramps_uncert,
                method='mean')
            dark_header['HISTORY'] = 'Master dark file'
                    
            self.fits_writer_steps(
                data=master_dark,
                header=dark_header,
                output_dir=self.action.args.dirname,
                input_filename=filename,
                suffix='_mdark',
                overwrite=True,
                uncert=master_dark_uncert)
            self.logger.info("+++++++++++ Creating master dark +++++++++++")

        if len(bias_ramps) > 0:
            master_bias, master_bias_uncert = self.build_master_from_stack(
                bias_ramps,
                bias_ramps_uncert,
                method='mean')
            bias_header['HISTORY'] = 'Master bias file'
            self.fits_writer_steps(
                data=master_bias,
                header=bias_header,
                output_dir=self.action.args.dirname,
                input_filename=filename,
                suffix='_mbias',
                overwrite=True,
                uncert=master_bias_uncert)
            self.logger.info("+++++++++++ Creating master bias +++++++++++")

        if len(flatlen_ramps) > 0:
            master_flatlens, master_flatlens_uncert = self.build_master_from_stack(
                flatlen_ramps,
                flatlen_ramps_uncert,
                method='mean')
            flatlens_header['HISTORY'] = 'Master lenslet flat file'
            self.fits_writer_steps(
                data=master_flatlens,
                header=flatlen_header,
                output_dir=self.action.args.dirname,
                input_filename=filename,
                suffix='_mflatlens',
                overwrite=True,
                uncert=master_flatlen_uncert)
            
            self.logger.info("+++++++++++ Creating master lenslet flat +++++++++++")
            self.logger.info("+++++++++++ Creating master lenslet flat cube +++++++++++")

            calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
            readnoise = fits.getdata(calib_path+'sim_readnoise.fits')
            var_read_vector = (readnoise.flatten().astype(np.float64))**2
            GAIN = 1.0#self.action.args.ccddata.header['GAIN']
            
            if ifsmode=='LowRes-K':
                R_for_extract = load_npz(calib_path+'K_C2_rectmat_lowres.npz')
                R_matrix = load_npz(calib_path+'K_QL_rectmat_lowres.npz')
                FLUX_SHAPE_3D = (56, 103, 110)
            elif ifsmode=='LowRes-L':
                R_for_extract = load_npz(calib_path+'L_C2_rectmat_lowres.npz')
                R_matrix = load_npz(calib_path+'L_QL_rectmat_lowres.npz')
                FLUX_SHAPE_3D = (56, 103, 110)
            elif ifsmode=='LowRes-M':
                R_for_extract = load_npz(calib_path+'M_C2_rectmat_lowres.npz')
                R_matrix = load_npz(calib_path+'M_QL_rectmat_lowres.npz')
                FLUX_SHAPE_3D = (56, 103, 110)
            elif ifsmode=='LowRes-SED':
                R_for_extract = load_npz(calib_path+'SED_C2_rectmat_lowres.npz')
                R_matrix = load_npz(calib_path+'SED_QL_rectmat_lowres.npz')
                FLUX_SHAPE_3D = (56, 103, 110)
            elif ifsmode=='LowRes-H2O':
                R_for_extract = load_npz(calib_path+'H2O_C2_rectmat_lowres.npz')
                R_matrix = load_npz(calib_path+'H2O_QL_rectmat_lowres.npz')
                FLUX_SHAPE_3D = (56, 103, 110)
            elif ifsmode=='LowRes-PAH':
                R_for_extract = load_npz(calib_path+'PAH_C2_rectmat_lowres.npz')
                R_matrix = load_npz(calib_path+'PAH_QL_rectmat_lowres.npz')
                FLUX_SHAPE_3D = (56, 103, 110)
            elif ifsmode=='MedRes-K':
                R_for_extract = load_npz(calib_path+'K_C2_rectmat_medres.npz')
                R_matrix = load_npz(calib_path+'K_QL_rectmat_medres.npz')
                FLUX_SHAPE_3D = (1900, 103, 110)
            elif ifsmode=='MedRes-L':
                R_for_extract = load_npz(calib_path+'L_C2_rectmat_medres.npz')
                R_matrix = load_npz(calib_path+'L_QL_rectmat_medres.npz')
                FLUX_SHAPE_3D = (1900, 103, 110)
            elif ifsmode=='MedRes-M':
                R_for_extract = load_npz(calib_path+'M_C2_rectmat_medres.npz')
                R_matrix = load_npz(calib_path+'M_QL_rectmat_medres.npz')
                FLUX_SHAPE_3D = (1900, 103, 110)
            
            A_guess_cube,A_guess_cube_err = self.optimal_extract_with_error(
                R_matrix,
                master_flatlens,
                master_flatlen_uncert,
                var_read_vector)
            A_opt = A_guess_cube.reshape(FLUX_SHAPE_3D)
            A_opt_err = A_guess_cube_err.reshape(FLUX_SHAPE_3D)
            self.fits_writer_steps(
                data=A_opt,
                header=flatlen_header,
                output_dir=self.action.args.dirname,
                input_filename=filename,
                suffix='_cube_flatlens',
                overwrite=True,
                uncert=A_opt_err)

        if len(flatlamp_ramps) > 0:
            master_flatlamp, master_flatlamp_uncert = self.build_master_from_stack(
                flatlamp_ramps,
                flatlamp_ramps_uncert,
                method='mean')
            flatlamp_header['HISTORY'] = 'Master detector flat file'
            self.fits_writer_steps(
                data=master_flatlamp,
                header=flatlamp_header,
                output_dir=self.action.args.dirname,
                input_filename=filename,
                suffix='_mflatlamp',
                overwrite=True,
                uncert=master_flatlamp_uncert)
            self.logger.info("+++++++++++ Creating master detector flat +++++++++++")

        if len(calunit_ramps) > 0:
            master_calunit, master_calunit_uncert = self.build_master_from_stack(
                calunit_ramps,
                calunit_ramps_uncert,
                method='mean')
            calunit_header['HISTORY'] = 'Master calunit file'
            self.fits_writer_steps(
                data=master_calunit,
                header=calunit_header,
                output_dir=self.action.args.dirname,
                input_filename=filename,
                suffix='_mcalunit',
                overwrite=True,
                uncert=master_calunit_uncert)
            self.logger.info("+++++++++++ Creating master monochromator file +++++++++++")
        
        self.logger.info('+++++++++++++ All available Master calibration files are created ++++++++++++++')
        self.logger.info('+++++++++++++ Ready to process the science exposures ++++++++++++++')
        
        log_string = StartCalib.__module__
        self.logger.info(log_string)
        return self.action.args










