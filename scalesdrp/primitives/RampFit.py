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

    def masked_smooth_fast(self,filled, nanmask, sigma=1.25):
        """Smooth only inside bbox around NaNs; keep valid pixels fixed."""
        out = np.asarray(filled, dtype=np.float32).copy()
        (ys, xs), ok = self._bbox(nanmask)
        if not ok:
            return out
        submask = nanmask[ys, xs]
        subimg  = out[ys, xs]
        sm = gaussian_filter(subimg, sigma=sigma, mode='nearest')
        dist_inside = distance_transform_edt(submask.astype(np.uint8))
        blend = np.clip(dist_inside / (3.0 * sigma), 0.0, 1.0).astype(np.float32)
        subimg[submask] = (1 - blend[submask]) * subimg[submask] + blend[submask] * sm[submask]
        out[ys, xs] = subimg
        #self.logger.info("Ramp fitting completed")
        return out

    def inpaint_nearest_fast(self,img, nanmask=None, pad=16):
        """Do the EDT only on the small bbox around NaNs."""
        out = np.asarray(img, dtype=np.float32).copy()
        if nanmask is None:
            nanmask = ~np.isfinite(out)
        if not np.any(nanmask):
            return out, nanmask
        (ys, xs), ok = self._bbox(nanmask)
        if not ok:
            return out, nanmask
        submask = nanmask[ys, xs]
        subimg  = out[ys, xs]
        _, (iy, ix) = distance_transform_edt(submask, return_indices=True)
        subimg[submask] = subimg[iy[submask], ix[submask]]
        out[ys, xs] = subimg
        return out, nanmask

    def _bbox(self,mask, pad=16):
        ys, xs = np.where(mask)
        if ys.size == 0:
            return (slice(0,0), slice(0,0)), False
        y0 = max(0, ys.min() - pad)
        y1 = min(mask.shape[0], ys.max() + pad + 1)
        x0 = max(0, xs.min() - pad)
        x1 = min(mask.shape[1], xs.max() + pad + 1)
        return (slice(y0, y1), slice(x0, x1)), True

    def _perform(self):
        #self.logger.info("Ramp fitting")

        ols_t_global = None

        def ols_pack_parms(a, b, c): 
            return np.array([a, b, c])

        def ols_unpack_parms(p):
            a_p, b_p, c_p = p
            return a_p, b_p, c_p

        def ols_model_fn(p_model): # Quadratic model
            a_m, b_m, c_m = ols_unpack_parms(p_model)
            global ols_t_global
            if ols_t_global is None:
                raise ValueError("Global time array 'ols_t_global' for OLS model_fn is not set.")
            return a_m + b_m * ols_t_global + c_m * ols_t_global**2

        def fit_slope_image(reads_cube,read_times,read_noise_sigmas):
            """
            Performs a weighted straight-line fit for each pixel in a data cube of reads.
            Args:
                reads_cube:
                    A 3D NumPy array of shape (N, H, W) where N is the number of reads,
                    and H and W are the height and width of the image.
                read_times:
                    A 1D NumPy array of length N, containing the individual time
                    duration for each read.
                read_noise_sigmas:
                    A 2D NumPy array of shape (H, W) containing the read noise standard
                    deviation for each pixel.
            Returns:
                A 2D NumPy array of shape (H, W) representing the best-fit slope image.
            """
            num_reads, height, width = reads_cube.shape
            x = np.cumsum(read_times)
            x_broadcast = x.reshape(num_reads, 1, 1)
            epsilon = 1e-10
            weights = 1.0 / (read_noise_sigmas**2 + epsilon)
            sum_w = num_reads * weights
            sum_w_x = np.sum(weights * x_broadcast, axis=0)
            sum_w_y = np.sum(weights * reads_cube, axis=0)
            sum_w_x_sq = np.sum(weights * x_broadcast**2, axis=0)
            sum_w_xy = np.sum(weights * x_broadcast * reads_cube, axis=0)
            numerator = sum_w_xy - (sum_w_y * sum_w_x) / sum_w
            denominator = sum_w_x_sq - (sum_w_x**2) / sum_w
            slope_image = numerator / (denominator + epsilon)
            return slope_image

        FLUX_SCALING_FACTOR = 1.0
        SATURATION_DARK_OLS = 50000.0 / FLUX_SCALING_FACTOR
        SATURATION_SCIENCE_OLS = 4096.0 / FLUX_SCALING_FACTOR
        DEFAULT_SIG_FALLBACK_SCALED = 5.0 / FLUX_SCALING_FACTOR 
        JUMP_THRESH_ONEOMIT = 20.25 #4.5 sigma
        JUMP_THRESH_TWOOMIT = 23.8
        total_exptime = 300.0 #self.action.args.ccddata.header['RLEXPT']
        calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
        SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits')

        sci_im_full_original = reference.reffix_hxrg(self.action.args.ccddata.data, nchans=4, fixcol=True)
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
            output_fitramp_final = fit_slope_image(sci_im_full_original, read_times,SIG_map_scaled)
        else:
            sci_im_original_units = sci_im_full_original[:nim_s, :, :]
            sci_im_scaled = sci_im_original_units / FLUX_SCALING_FACTOR
            sci_im_with_jumps_scaled = sci_im_scaled.copy()
            ols_t_global = np.arange(nim_s)
            B_ols_sci = np.zeros((2048, 2048), dtype=float)

            for i_r in range(2048):
                for j_c in range(2048):
                    a_g=sci_im_with_jumps_scaled[0,i_r,j_c]; b_g=(sci_im_with_jumps_scaled[1,i_r,j_c]-sci_im_with_jumps_scaled[0,i_r,j_c]) \
                        if nim_s>1 else 1.0; c_g=0.0
                    
                    sp=ols_pack_parms(a_g,b_g,c_g); imdat_p=sci_im_with_jumps_scaled[:,i_r,j_c]
                    w_idx=np.where(imdat_p < SATURATION_SCIENCE_OLS)[0]
                    if len(w_idx)<3: B_ols_sci[i_r,j_c]=np.nan; continue
                    imdat_v = imdat_p[w_idx]
                    def resid_fn_sci_local(p_loc): mimdat_f=ols_model_fn(p_loc); return imdat_v - mimdat_f[w_idx]
                    try: 
                        p_opt_s,ier_s=leastsq(resid_fn_sci_local,sp.copy())
                        if ier_s not in [1,2,3,4]: p_opt_s = np.array([np.nan]*3)
                    except: p_opt_s=np.array([np.nan]*3)
                    _ ,b_fit_s,_ = ols_unpack_parms(p_opt_s); B_ols_sci[i_r,j_c]=b_fit_s
            
            median_B_sci_val=np.nanmedian(B_ols_sci)
            B_ols_sci[np.isnan(B_ols_sci)]= median_B_sci_val if not np.isnan(median_B_sci_val) else 0.0
            B_ols_sci[B_ols_sci<0]=0 

            Covar_obj_sci = fitramp.Covar(read_times, pedestal=False)
            d_sci = sci_im_with_jumps_scaled[1:] - sci_im_with_jumps_scaled[:-1]

            output_fitramp_final = np.empty((2048, 2048), dtype=float)
            start_time = time.time()

            for i in range(2048):
                if i > 0 and i % (max(1, 2048 // 10)) == 0: print(f"  fitramp data row {i+1}/{2048}...")
                
                current_sig_for_row = SIG_map_scaled[i, :] 
                diffs_for_row = d_sci[:, i, :] 
                countrateguess_for_row = B_ols_sci[i, :] 

                diffs2use, countrates_after_masking = fitramp.mask_jumps(
                    diffs_for_row, Covar_obj_sci, current_sig_for_row,
                    threshold_oneomit=JUMP_THRESH_ONEOMIT,
                    threshold_twoomit=JUMP_THRESH_TWOOMIT)


                final_countrateguess_fitramp = countrates_after_masking * (countrates_after_masking > 0)
        
                result = fitramp.fit_ramps(
                    diffs_for_row, Covar_obj_sci, current_sig_for_row,
                    diffs2use=diffs2use,
                    detect_jumps=False, 
                    countrateguess=final_countrateguess_fitramp,
                    rescale=True)


                output_fitramp_final[i, :] = result.countrate * FLUX_SCALING_FACTOR 
            
            end_time = time.time()

            self.logger.info(f"  fitramp on the data took {end_time - start_time:.2f} seconds.")
        
        filled, nanmask = self.inpaint_nearest_fast(output_fitramp_final)
        ramp_image_ouput = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
        self.logger.info("+++++++++++ ramp fitting completed +++++++++++")

        #ramp_image_ouput = bpm.apply_full_correction(ramp_image_ouput1,obsmode)
        #self.logger.info("+++++++++++ BPM correction completed +++++++++++")


        self.action.args.ccddata.data = ramp_image_ouput
        log_string = RampFit.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)

        scales_fits_writer(self.action.args.ccddata,
            table=self.action.args.table,
            output_file=self.action.args.name,
            output_dir=self.config.instrument.output_directory,
            suffix="_L1_ramp")
        self.logger.info("+++++++++++ slope image saved +++++++++++")

        return self.action.args
    # END: class RampFit()
