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
    def iterative_sigma_weighted_ramp_fit(
        self, ramp1, read_time, gain=3.0, rn=5.0, max_iter=3, tile=(256, 256), do_swap=True):
        t1=time.time()
        n_reads, n_rows, n_cols = ramp1.shape
        read_times = np.linspace(0, read_time, n_reads, dtype=np.float32)
        dt = np.mean(np.diff(read_times))
        ramp = np.empty_like(ramp1)
        n_amps = 4
        block = n_cols // n_amps
        for a in range(n_amps):
            x0, x1 = a * block, (a + 1) * block
            sub = ramp1[..., x0:x1]
            nsub = sub.shape[-1]
            new_order = []
            for i in range(0, nsub, 2):
                if i + 1 < nsub:
                    new_order.extend([i + 1, i])
                else:
                    new_order.append(i)
            ramp[..., x0:x1] = sub[..., new_order]
        slope = np.zeros((n_rows, n_cols), dtype=np.float32)
        bias = np.zeros_like(slope)
        Ty, Tx = tile
        for y0 in range(0, n_rows, Ty):
            y1 = min(n_rows, y0 + Ty)
            for x0 in range(0, n_cols, Tx):
                x1 = min(n_cols, x0 + Tx)
                cube = ramp[:, y0:y1, x0:x1]
                N, ty, tx = cube.shape
                m_tile = np.zeros((ty, tx), dtype=np.float32)
                b_tile = np.zeros_like(m_tile)
                for col_slice, out_slice in [(np.index_exp[:, :, 0::2], (slice(None), slice(0, None, 2))),
                    (np.index_exp[:, :, 1::2], (slice(None), slice(1, None, 2))),]:

                    subcube = cube[col_slice]
                    if subcube.size == 0:
                        continue
                    for iteration in range(max_iter):
                        sig2 = np.maximum(subcube / gain + rn**2, 1e-6)
                        i = np.arange(subcube.shape[0], dtype=np.float32)[:, None, None]
                        S0 = np.sum(1.0 / sig2, axis=0)
                        S1 = np.sum(i / sig2, axis=0)
                        S2 = np.sum(i**2 / sig2, axis=0)
                        S0x = np.sum(subcube / sig2, axis=0)
                        S1x = np.sum(i * subcube / sig2, axis=0)
                        ibar = S1 / S0
                        mdt = (S1x - ibar * S0x) / np.maximum(S2 - ibar**2 * S0, 1e-8)
                        m = mdt / dt
                        b = S0x / S0 - mdt * ibar
                        subcube = np.clip(b[None, :, :] + m[None, :, :] * i * dt, 0, None)
                    m_tile[out_slice] = m
                    b_tile[out_slice] = b
                slope[y0:y1, x0:x1] = m_tile
                bias[y0:y1, x0:x1] = b_tile
        t2=time.time()
        self.logger.info(f"Ramp fitting finished in {t2-t1:.2f} seconds.")
        return slope

    ########### ramp fit for read >5 ########################################

    def ramp_fit(self,input_read, total_exptime, SIG_map_scaled):
        """
        Perform hybrid ramp fitting:
        - tile-based sigma-clipped preprocessing of input reads
        - slope_linear() fallback for all pixels
        - fitramp for valid regions
        - smooth blending and NaN handling
        """
        # === Helper: fast tile-based sigma clipping ===
        def sigma_clip_ramp_inputs(science_ramp, sigma_clip=3.0, max_iter=3, min_reads=5, tile=(128, 128)):
            """
            Iteratively sigma-clip each pixel's ramp along the time axis, tile by tile.
            Returns a boolean mask (True = keep read).
            """
            n_reads, n_rows, n_cols = science_ramp.shape
            keep_mask = np.ones_like(science_ramp, dtype=bool)
            Ty, Tx = tile

            for y0 in tqdm(range(0, n_rows, Ty), desc="σ-clipping tiles"):
                y1 = min(n_rows, y0 + Ty)
                for x0 in range(0, n_cols, Tx):
                    x1 = min(n_cols, x0 + Tx)

                    cube = science_ramp[:, y0:y1, x0:x1]  # (N, ty, tx)
                    N, ty, tx = cube.shape
                    k = ty * tx
                    y = cube.reshape(N, k)
                    mask = np.isfinite(y)

                    for _ in range(max_iter):
                        valid_counts = mask.sum(axis=0)
                        good = valid_counts >= min_reads
                        if not np.any(good):
                            break

                        t = np.arange(N, dtype=np.float32)
                        S0 = mask.sum(axis=0)
                        St = t @ mask
                        Stt = (t**2) @ mask
                        wy = y * mask
                        Sy = wy.sum(axis=0)
                        Sty = t @ wy
                        Var_t = Stt - (St * St) / np.maximum(S0, 1)
                        Cov_ty = Sty - (St * Sy) / np.maximum(S0, 1)
                        b = np.zeros(k, dtype=np.float32)
                        valid = Var_t > 0
                        b[valid] = Cov_ty[valid] / Var_t[valid]
                        a = (Sy - b * St) / np.maximum(S0, 1)

                        y_pred = a + np.outer(t, b)
                        resid = (y - y_pred)
                        resid[~mask] = np.nan
                        std = np.nanstd(resid, axis=0)
                        new_mask = np.abs(resid) < sigma_clip * std
                        new_mask &= np.isfinite(y)
                        if np.array_equal(new_mask, mask):
                            break
                        mask = new_mask

                    keep_mask[:, y0:y1, x0:x1] = mask.reshape(N, ty, tx)

            return keep_mask

        FLUX_SCALING_FACTOR = 1.0
        JUMP_THRESH_ONEOMIT = 20.25
        JUMP_THRESH_TWOOMIT = 23.8

        nim_s = input_read.shape[0]
        read_times = np.linspace(0, total_exptime, nim_s)

        # === Case 1: Few reads (simple linear fit) ===
        if nim_s < 6:
            self.logger.info('Few reads (<6): Performing a stright line fit...')
            output_final = self.iterative_sigma_weighted_ramp_fit(
                input_read,read_time=total_exptime)
        else:
            # === Case 2: Full ramp fitting ===
            self.logger.info("Applying σ-clipping to input ramp (tile-based)...")
            valid_reads_mask = sigma_clip_ramp_inputs(
                input_read, sigma_clip=3.0, max_iter=3, min_reads=5, tile=(128, 128))

            sci_im_scaled = input_read / FLUX_SCALING_FACTOR
            Covar_obj = fitramp.Covar(read_times, pedestal=False)

            # Apply sigma-clipped mask
            masked_input = np.where(valid_reads_mask, sci_im_scaled, np.nan)
            d_sci = masked_input[1:] - masked_input[:-1]

            # === Precompute linear fallback (OLS) ===
            self.logger.info("Computing initial guess for ramp fitting...")
            B_ols_sci = self.iterative_sigma_weighted_ramp_fit(
                sci_im_scaled,
                read_time=total_exptime)

            # === fitramp fitting ===
            output_final = np.empty((sci_im_scaled.shape[1], sci_im_scaled.shape[2]), dtype=float)
            start_time = time.time()

            for i in range(sci_im_scaled.shape[1]):
                if i % 128 == 0:
                    print(f"  Fitting row {i}/{sci_im_scaled.shape[1]}...")

                current_sig_for_row = SIG_map_scaled[i, :]
                diffs_for_row = d_sci[:, i, :]
                countrateguess = B_ols_sci[i, :]

                # convert sigma-clipped mask to diffs2use
                diffs2use = valid_reads_mask[1:, i, :] & valid_reads_mask[:-1, i, :]

                try:
                    # Combine with fitramp’s own jump mask
                    diffs2use_fitramp, _ = fitramp.mask_jumps(
                        diffs_for_row, Covar_obj, current_sig_for_row,
                        threshold_oneomit=JUMP_THRESH_ONEOMIT,
                        threshold_twoomit=JUMP_THRESH_TWOOMIT,)

                    diffs2use &= diffs2use_fitramp

                    result = fitramp.fit_ramps(
                        diffs_for_row, Covar_obj, current_sig_for_row,
                        diffs2use=diffs2use,
                        countrateguess=countrateguess,
                        rescale=True,)

                    valid = np.isfinite(result.countrate) & (
                        result.countrate > 0.05 * np.nanmedian(countrateguess))

                    row_out = np.where(valid, result.countrate, countrateguess)

                except Exception:
                    row_out = countrateguess  # fallback if fitramp fails entirely

                output_final[i, :] = row_out * FLUX_SCALING_FACTOR

            end_time = time.time()
            self.logger.info(f"Ramp fitting completed in {end_time - start_time:.2f} seconds.")
        return output_final

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
            return None

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
            return None

        # now open candidates and check headers
        for path in candidates:
            try:
                with fits.open(path) as hdul:
                    hdr = hdul[0].header
                    data = hdul[0].data
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
            return data

        # no candidate matched fully
        return None

    ####################### flat normalization #############################
    def normalize_detector_flat(self,flat, mask=None, method='median', clip_sigma=7.0, iterations=3):
        flat = np.array(flat, dtype=np.float32)
        invalid = ~np.isfinite(flat) | (flat <= 0)
        if mask is not None:
            invalid |= mask.astype(bool)
        valid = ~invalid
        valid_data = flat[valid]
        if valid_data.size == 0:
            raise ValueError("No valid pixels found in flat for normalization.")
        clipped = valid_data.copy()
        for _ in range(iterations):
            med = np.median(clipped)
            std = np.std(clipped)
            good = (clipped > med - clip_sigma * std) & (clipped < med + clip_sigma * std)
            if np.all(good):
                break
            clipped = clipped[good]
        norm_value = np.median(clipped) if method == 'median' else np.mean(clipped)
        if norm_value <= 0 or not np.isfinite(norm_value):
            raise ValueError(f"Invalid normalization constant: {norm_value}")
        flat_norm = flat / norm_value
        flat_norm[invalid] = np.nan
        return flat_norm
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
        
            output_final = []
            input_data = self.action.args.ccddata.data
            #print(input_data.shape)
            sci_im_full_original = input_data 
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
            read_times = np.linspace(0, total_exptime, nim_s)
            self.logger.info("+++++++++++ ramp fitting started +++++++++++")

            output_final = self.ramp_fit(sci_im_full_original, read_times, SIG_map_scaled)

            #bad_mask_global = (output_final <= 0) | ~np.isfinite(output_final)
            #if np.any(bad_mask_global):
            #    output_final = self.masked_smooth_fast(output_final, bad_mask_global, sigma=1.0)
            #med = np.nanmedian(output_final[output_final > 0])
            #output_final[np.isnan(output_final)] = med

            self.logger.info("+++++++++++ Bad pixel correction started +++++++++++")
            final_ramp = bpm.apply_full_correction(output_final,obsmode)
            self.logger.info("+++++++++++ BPM correction completed +++++++++++")

            keywords_unique = {
                key: self.action.args.ccddata.header.get(key)
                for key in ['OBSMODE', 'IFSMODE', 'MCLOCK']}

            m_dark = self.load_single_master_file(keywords_unique, master_type='DARK')
            m_bias = self.load_single_master_file(keywords_unique, master_type='BIAS')
            m_flat = self.load_single_master_file(keywords_unique, master_type='FLATLAMP')

            if m_dark is not None:
                final_ramp = final_ramp - m_dark
                self.action.args.ccddata.header['HISTORY'] = 'Dark subtracted.'
                self.logger.info("+++++++++++ Master dark subtracted +++++++++++")
        
            if m_bias is not None:
                final_ramp = final_ramp - m_bias
                self.action.args.ccddata.header['HISTORY'] = 'Bias subtracted.'
                self.logger.info("+++++++++++ Master bias subtracted +++++++++++")

            if m_flat is not None:
                norm_flat = self.normalize_detector_flat(m_flat)
                final_ramp = np.true_divide(final_ramp,norm_flat)
                self.action.args.ccddata.header['HISTORY'] = 'Detector Flat correction applied.'
                self.logger.info("+++++++++++ detector flat correction completed  +++++++++++")

            self.action.args.ccddata.data = final_ramp
        
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
