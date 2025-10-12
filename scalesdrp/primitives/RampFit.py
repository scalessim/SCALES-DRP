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
        #self.logger.info("Ramp fitting science data starting")

    def adaptive_weighted_ramp_fit(self,ramp, read_time, cutoff_frac=0.75, sat_level=4096.0, tile=(256,256)):
        """
        Adaptive weighted ramp fit (thread-safe, DRP-compatible version).
        Processes the ramp in tiles to stay memory- and cache-efficient.
        """
        n_reads, n_rows, n_cols = ramp.shape
        read_times = np.linspace(0, read_time, n_reads, dtype=np.float32)
        slope = np.zeros((n_rows, n_cols), dtype=np.float32)
        eps = 1e-6

        Ty, Tx = tile
        for y0 in range(0, n_rows, Ty):
            y1 = min(n_rows, y0 + Ty)
            for x0 in range(0, n_cols, Tx):
                x1 = min(n_cols, x0 + Tx)
                cube = ramp[:, y0:y1, x0:x1]  # (N, ty, tx)
                ty, tx = cube.shape[1:]
                N = cube.shape[0]

                # Find cutoff index per pixel where counts exceed cutoff_frac * sat_level
                cutoff_mask = cube > (cutoff_frac * sat_level)
                # argmax returns 0 if all False — handle that
                cutoff_idx = np.argmax(cutoff_mask, axis=0)
                cutoff_idx[~np.any(cutoff_mask, axis=0)] = N - 1

                # Build weights centered around cutoff index
                idx = np.arange(N)[:, None, None]
                w = np.exp(-((idx - cutoff_idx)**2) / 8.0).astype(np.float32)

                # Weighted sums
                t = read_times[:, None, None]
                y = cube
                S0  = np.sum(w, axis=0)
                St  = np.sum(w * t, axis=0)
                Stt = np.sum(w * t * t, axis=0)
                Sy  = np.sum(w * y, axis=0)
                Sty = np.sum(w * t * y, axis=0)

                denom = S0 * Stt - St * St
                valid = denom > eps
                local_slope = np.full_like(S0, np.nan, dtype=np.float32)
                local_slope[valid] = (S0[valid] * Sty[valid] - St[valid] * Sy[valid]) / denom[valid]
                slope[y0:y1, x0:x1] = local_slope
        return slope

    def _perform(self):
        #self.logger.info("Ramp fitting")

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
        JUMP_THRESH_ONEOMIT = 20.25 #4.5 sigma
        JUMP_THRESH_TWOOMIT = 23.8
        total_exptime = self.action.args.ccddata.header['EXPTIME']
        calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
        SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits')

        input_data = self.action.args.ccddata.data
        print(input_data.shape)
        sci_im_full_original = reference.reffix_hxrg(input_data, nchans=4, fixcol=True)
        self.logger.info("refpix and 1/f correction completed")
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


        if nim_s < 5:
            print('Number of reads are less than 5, starting a stright line fit to the reads')
            output_final = self.adaptive_weighted_ramp_fit(sci_im_full_original, read_times)
            return output_final
        else:
            self.logger.info("Applying σ-clipping to input ramp (tile-based)...")
            valid_reads_mask = sigma_clip_ramp_inputs(
                sci_im_full_original, sigma_clip=3.0, max_iter=3, min_reads=5, tile=(128, 128))

            sci_im_scaled = sci_im_full_original / FLUX_SCALING_FACTOR
            Covar_obj = fitramp.Covar(read_times, pedestal=False)
            masked_input = np.where(valid_reads_mask, sci_im_scaled, np.nan)
            d_sci = masked_input[1:] - masked_input[:-1]
            self.logger.info("Estimating initial guess for ramp fitting")
            B_ols_sci = self.adaptive_weighted_ramp_fit(
                sci_im_scaled,read_time=total_exptime)

            output_final = np.empty((sci_im_scaled.shape[1], sci_im_scaled.shape[2]), dtype=float)
            start_time = time.time()
            for i in range(sci_im_scaled.shape[1]):
                if i % 128 == 0:
                    print(f"  Fitting row {i}/{sci_im_scaled.shape[1]}...")

                current_sig_for_row = SIG_map_scaled[i, :]
                diffs_for_row = d_sci[:, i, :]
                countrateguess = B_ols_sci[i, :]
                diffs2use = valid_reads_mask[1:, i, :] & valid_reads_mask[:-1, i, :]

                try:
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
                    valid = np.isfinite(result.countrate) & (result.countrate > 0.05 * np.nanmedian(countrateguess))

                    row_out = np.where(valid, result.countrate, countrateguess)
                except Exception:
                    row_out = countrateguess
                output_final[i, :] = row_out * FLUX_SCALING_FACTOR
            end_time = time.time()
            self.logger.info(f"Ramp fitting completed in {end_time - start_time:.2f} seconds.")
            #bad_mask_global = (output_final <= 0) | ~np.isfinite(output_final)
            #if np.any(bad_mask_global):
            #    output_final = self.masked_smooth_fast(output_final, bad_mask_global, sigma=1.0)
            #med = np.nanmedian(output_final[output_final > 0])
            #output_final[np.isnan(output_final)] = med

        #ramp_image_ouput = bpm.apply_full_correction(ramp_image_ouput1,obsmode)
        #self.logger.info("+++++++++++ BPM correction completed +++++++++++")


        self.action.args.ccddata.data = output_final
        log_string = RampFit.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)

        scales_fits_writer(self.action.args.ccddata,
            table=self.action.args.table,
            output_file=self.action.args.name,
            output_dir=self.config.instrument.output_directory,
            suffix="_L1_ramp")
        self.logger.info("+++++++++++ slope image FITS file saved +++++++++++")

        return self.action.args
    # END: class RampFit()
