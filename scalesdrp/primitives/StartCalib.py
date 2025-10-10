from keckdrpframework.primitives.base_primitive import BasePrimitive
import scalesdrp.primitives.fitramp as fitramp
import scalesdrp.primitives.robust as robust
import scalesdrp.primitives.reference as reference #1/f and reference pixel correction

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

    def group_files_by_header(self,dt):
        """
        Groups files in the data table by unique combinations of SCMODE,
        IMTYPE, and EXPTIME.
        """

        grouping_keys = ['FILTER','IMTYPE','EXPTIME']
        if not all(key in dt.columns for key in grouping_keys):
            self.logger.error(f"One or more grouping keys not found in data table: {grouping_keys}")
            return []
        grouped = dt.groupby(grouping_keys)
        file_groups = []

        for group_params, sub_df in grouped:
            group_info = {
                'params': {
                    'scmode': group_params[0],
                    'imtype': group_params[1],
                    'exptime': group_params[2]
                },
                'filenames': sub_df.index.tolist()
            }
            file_groups.append(group_info)
        return file_groups   

    def ols_pack_parms(self,a, b, c): 
        return np.array([a, b, c])

    def ols_unpack_parms(self,p):
        a_p, b_p, c_p = p
        return a_p, b_p, c_p

    def ols_model_fn(self,p_model): # Quadratic model
        a_m, b_m, c_m = ols_unpack_parms(p_model)
        global ols_t_global
        if ols_t_global is None:
            raise ValueError("Global time array 'ols_t_global' for OLS model_fn is not set.")
        return a_m + b_m * ols_t_global + c_m * ols_t_global**2

############# ouput fits writing ##############################
    def fits_writer_steps(self,data,header,output_dir,input_filename,suffix,overwrite=True):
        base_name = os.path.basename(input_filename)
        file_root, file_ext = os.path.splitext(base_name)
        output_filename = f"{file_root}{suffix}{file_ext}"
        redux_output_dir = os.path.join(output_dir, 'redux')
        os.makedirs(redux_output_dir, exist_ok=True)
        output_path = os.path.join(redux_output_dir, output_filename)
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_path, overwrite=overwrite)
        return output_path

############################# linearity correction ##############################
    def apply_linearity_correction(self,raw_ramp_frames,num_sample_pix=5000):
        """
        Loads the raw ramp data, derives a non-linearity correction.
        Args:
            raw_ramp_frames: the 3D ramp data cube.
            num_sample_pix (int, optional):
            Number of random pixels to use for characterization. Defaults to 5000.
        Returns:
            tuple: two arrays for the correction curve:median_fluence and median_correction.
        """    
        num_frames, height, width = raw_ramp_frames.shape
        pixel_max_value = 65536
        sample_y = np.random.randint(0, height, size=num_sample_pix)
        sample_x = np.random.randint(0, width, size=num_sample_pix)
        sample_ramps = raw_ramp_frames[:, sample_y, sample_x].T
        linear_slope_est = sample_ramps[:, 1] - sample_ramps[:, 0]
        frame_times = np.arange(1, num_frames + 1)
        biases = sample_ramps[:, 0] - (linear_slope_est * frame_times[0])
        measured_fluence = sample_ramps - biases[:, np.newaxis]
        ideal_ramps = linear_slope_est[:, np.newaxis] * frame_times
        corrections = ideal_ramps - measured_fluence
        fluence_flat = measured_fluence.flatten()
        corr_flat = corrections.flatten()
        bins = np.linspace(0, pixel_max_value, num=200)
        bin_indices = np.digitize(fluence_flat, bins)
        median_fluence=[]
        median_corr=[]
        for i in range(1, len(bins)):
            in_bin = (bin_indices == i)
            if np.any(in_bin):
                median_fluence.append(np.median(fluence_flat[in_bin]))
                median_corr.append(np.median(corr_flat[in_bin]))
        
        median_fluence = np.array(median_fluence)
        median_corr = np.array(median_corr)
        return median_fluence, median_corr

######################### Creating BPM ###########################
    def generate_bpm(
        self,
        image_stack: np.ndarray,
        stack_name: str = "Image Stack",
        spatial_brightness_factor: float = 5.0,
        spatial_kernel_size: int = 5,
        temporal_sigma_thresh: float = 5.0,
        min_frames_for_temporal: int = 3,
        plot_results: bool = False) -> np.ndarray:
        """
        Generates a Bad Pixel Mask (BPM) from a stack of images.
        - Temporal outliers are found using a standard deviation.
        - Spatial outliers are found using a RELATIVE BRIGHTNESS test: a pixel
          is flagged if it is N times brighter or dimmer than its local median.
        Args:
            image_stack (np.ndarray): 3D stack of images (N_frames, Height, Width).
            stack_name (str): Name for logging/plotting.
            spatial_brightness_factor (float): A pixel is bad if its value is > this
                factor times the local median, or < the local median / this factor.
            spatial_kernel_size (int): Size of the square kernel for spatial median filter.
            temporal_sigma_thresh (float): Sigma threshold for the temporal outlier test.
            min_frames_for_temporal (int): Min frames to run the temporal test.
        Returns:
            np.ndarray: A 2D boolean bad pixel mask where True represents a bad pixel.
        """
        if image_stack.ndim != 3:
            raise ValueError("Input `image_stack` must be a 3D array.")
        n_frames, height, width = image_stack.shape
        final_bpm = np.all(np.isnan(image_stack), axis=0)
        temporal_bpm = np.zeros_like(final_bpm)
        if n_frames >= min_frames_for_temporal and temporal_sigma_thresh > 0:
            pixel_medians = np.nanmedian(image_stack, axis=0, keepdims=True)
            pixel_stds_map = np.nanstd(image_stack, axis=0)
            valid_stds = pixel_stds_map[np.isfinite(pixel_stds_map) & (pixel_stds_map > 0)]
            std_floor = np.median(valid_stds) * 0.01 if len(valid_stds) > 0 else 1e-5
            pixel_stds_map[~np.isfinite(pixel_stds_map) | (pixel_stds_map <= 0)] = max(std_floor, 1e-5)
            deviations = np.abs(image_stack - pixel_medians)
            is_outlier_stack = deviations > (temporal_sigma_thresh * pixel_stds_map)
            temporal_bpm = np.any(is_outlier_stack, axis=0)
            final_bpm |= temporal_bpm
        spatial_bpm = np.zeros_like(final_bpm)
        if spatial_brightness_factor > 1 and spatial_kernel_size >= 3:
            for i in range(n_frames):
                frame = image_stack[i].copy()
                original_nans = np.isnan(frame)
                global_median_val = np.nanmedian(frame)
                if not np.isfinite(global_median_val): global_median_val = 0
                frame_no_nan = np.nan_to_num(frame, nan=global_median_val)
                local_median = median_filter(frame_no_nan, size=spatial_kernel_size, mode='reflect')
                local_median[local_median <= 1e-9] = 1e-9
                with np.errstate(divide='ignore', invalid='ignore'):
                    is_hot_pixel = frame > (spatial_brightness_factor * local_median)
                    is_cold_pixel = frame < (local_median / spatial_brightness_factor)
                is_outlier_this_frame = is_hot_pixel | is_cold_pixel
                is_outlier_this_frame |= original_nans 
                spatial_bpm |= is_outlier_this_frame
            final_bpm |= spatial_bpm
        return final_bpm

############# Corecting BPM ########################################################
    def correct_local_defects(self,image_data_to_correct, bad_pixel_mask, **kwargs):
        verbose = kwargs.get('verbose', True)
        if verbose: print("--- PASS 1: Starting Improved Local Iterative Correction ---")
    
        corrected_image = image_data_to_correct.copy()
        remaining_bpm = bad_pixel_mask.copy()
        initial_bad_count = np.sum(remaining_bpm)
        if initial_bad_count == 0:
            return corrected_image, remaining_bpm 

        current_box_size = kwargs.get('initial_box_size', 5)
        max_box_size = kwargs.get('max_box_size', 11)
        min_good_neighbors_frac = kwargs.get('min_good_neighbors_frac', 0.3) #finetune needed
        max_iterations = kwargs.get('max_iterations', 10) # finetune needed
        stalled_pixels_mask = np.zeros_like(remaining_bpm)

        for growth_pass in range((max_box_size - current_box_size) // 2 + 1):
            num_fixed_this_growth_step = 0
            for i in range(max_iterations):
                num_bad_at_start = np.sum(remaining_bpm)
                if num_bad_at_start == 0: break
            
                replacement_values = median_filter(corrected_image, size=current_box_size, mode='reflect')
                good_pixel_mask = (~remaining_bpm).astype(np.uint8)
                good_neighbor_count = convolve(good_pixel_mask, np.ones((current_box_size, current_box_size)), mode='constant', cval=0)
                min_good_req = int(min_good_neighbors_frac * (current_box_size**2 - 1))
                pixels_to_correct = remaining_bpm & (good_neighbor_count >= min_good_req)
                num_newly_corrected = np.sum(pixels_to_correct)
                if num_newly_corrected == 0: break
                corrected_image[pixels_to_correct] = replacement_values[pixels_to_correct]
                remaining_bpm[pixels_to_correct] = False
                num_fixed_this_growth_step += num_newly_corrected
        
            if np.sum(remaining_bpm) == 0: break
            if num_fixed_this_growth_step == 0:
                stalled_pixels_mask |= remaining_bpm
                current_box_size += 2
                if verbose: print(f"Stalled. Growing box to {current_box_size}x{current_box_size}.")
        
        final_uncorrected_mask = remaining_bpm | stalled_pixels_mask
        if verbose: print(f"--- Pass 1 Finished. Uncorrectable local pixels: {np.sum(final_uncorrected_mask)} ---")
        return corrected_image, final_uncorrected_mask
    
    def fill_global_defects(self,image, bad_pixel_mask):
        num_defects = np.sum(bad_pixel_mask)
        if num_defects == 0:
            return image.copy()
        image_with_nans = image.copy()
        image_with_nans[bad_pixel_mask] = np.nan
        kernel_size_stddev = 3 
        kernel = Gaussian2DKernel(x_stddev=kernel_size_stddev)
        inpainted_image = interpolate_replace_nans(image_with_nans, kernel)
        return inpainted_image

    def apply_full_correction(self,image_to_correct, master_bpm, pass1_kwargs={}):
        corrected_pass1, large_defects_mask = self.correct_local_defects(
        image_to_correct,
        master_bpm,
        **pass1_kwargs)
    
        fully_corrected_image = self.fill_global_defects(
            corrected_pass1,
            large_defects_mask)
        return fully_corrected_image

################## ramp fitting ############################
    ols_t_global = None

    def ols_pack_parms(self,a, b, c): 
        return np.array([a, b, c])

    def ols_unpack_parms(self,p):
        a_p, b_p, c_p = p
        return a_p, b_p, c_p

    def ols_model_fn(self,p_model): # Quadratic model
        a_m, b_m, c_m = ols_unpack_parms(p_model)
        global ols_t_global
        if ols_t_global is None:
            raise ValueError("Global time array 'ols_t_global' for OLS model_fn is not set.")
        return a_m + b_m * ols_t_global + c_m * ols_t_global**2

    def ramp_fit(self,input_read,read_time):
        calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
        SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits')
        FLUX_SCALING_FACTOR = 1.0
        SATURATION_DARK_OLS = 50000.0 / FLUX_SCALING_FACTOR
        SATURATION_SCIENCE_OLS = 4096.0 / FLUX_SCALING_FACTOR
        DEFAULT_SIG_FALLBACK_SCALED = 5.0 / FLUX_SCALING_FACTOR
        JUMP_THRESH_ONEOMIT = 20.25 #4.5 sigma
        JUMP_THRESH_TWOOMIT = 23.8
        sci_im_full_original = reference.reffix_hxrg(input_read, nchans=4, fixcol=True)
        self.logger.info("refpix and 1/f correction completed")
        nim_s = input_read.shape[0]
        if nim_s < 6:
            print('Number of reads are less than 5, starting a stright line fit to the reads')
            reads = input_read[:nim_s, :, :]
            read_times = np.arange(nim_s).astype(float)*read_time
            output_fitramp_final = fit_slope_image(reads, read_times,SIG_map_scaled)
        else:
            sci_im_original_units = input_read[:nim_s, :, :]
            sci_im_scaled = sci_im_original_units / FLUX_SCALING_FACTOR
            sci_im_with_jumps_scaled = sci_im_scaled.copy()
            ols_t_global = np.arange(nim_s)
            readtimes_for_covar_sci = np.arange(nim_s).astype(float)*read_time
            B_ols_sci = np.zeros((2048, 2048), dtype=float)
            for i_r in range(2048):
                for j_c in range(2048):
                    a_g=sci_im_with_jumps_scaled[0,i_r,j_c]; b_g=(sci_im_with_jumps_scaled[1,i_r,j_c]-sci_im_with_jumps_scaled[0,i_r,j_c]) \
                        if nim_s>1 else 1.0; c_g=0.0
                    sp=self.ols_pack_parms(a_g,b_g,c_g); imdat_p=sci_im_with_jumps_scaled[:,i_r,j_c]
                    w_idx=np.where(imdat_p < SATURATION_SCIENCE_OLS)[0]
                    if len(w_idx)<3: B_ols_sci[i_r,j_c]=np.nan; continue
                    imdat_v = imdat_p[w_idx]
                    def resid_fn_sci_local(p_loc): 
                        mimdat_f=self.ols_model_fn(p_loc)
                        return imdat_v - mimdat_f[w_idx]
                    try:
                        p_opt_s,ier_s=leastsq(resid_fn_sci_local,sp.copy())
                        if ier_s not in [1,2,3,4]: p_opt_s = np.array([np.nan]*3)
                    except: p_opt_s=np.array([np.nan]*3)
                    _ ,b_fit_s,_ = self.ols_unpack_parms(p_opt_s); B_ols_sci[i_r,j_c]=b_fit_s
            median_B_sci_val=np.nanmedian(B_ols_sci)
            B_ols_sci[np.isnan(B_ols_sci)]= median_B_sci_val if not np.isnan(median_B_sci_val) else 0.0
            B_ols_sci[B_ols_sci<0]=0
            Covar_obj_sci = fitramp.Covar(readtimes_for_covar_sci, pedestal=False)
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
        return output_fitramp_final

########################################### START MAIN ######################

    def _perform(self):
        self.logger.info("+++++++++++ Calibration Process +++++++++++")
        dt = self.context.data_set.data_table
        all_groups = self.group_files_by_header(dt)
        if not all_groups:
            self.logger.warning("No file groups found. Nothing to process.")
            return
        
        ols_t_global = None
        organized_groups = {}
        for group in all_groups:
            imtype = group['params']['imtype']
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

        for imtype in processing_order:
            if imtype in organized_groups:
                groups_for_this_type = organized_groups[imtype]
                for group in groups_for_this_type:
                    params = group['params']
                    filenames = group['filenames']
                    for filename in filenames:
                        try:
                            with fits.open(filename) as hdulist:
                                #sci_im_full_original1 = hdulist[0]
                                sci_im_full_original1 = hdulist[0].data
                                data_header = hdulist[0].header
                                #readtime = data_header['READTIME']
                        except Exception as e:
                            self.logger.error(f"Failed to read {filename}: {e}")
                        
                        ramp_image_ouput = self.ramp_fit(sci_im_full_original1,read_time=2.0)
                        self.logger.info("+++++++++++ ramp fitting completed +++++++++++")

                        self.fits_writer_steps(
                            data=ramp_image_ouput,
                            header=data_header,
                            output_dir=self.action.args.dirname,
                            input_filename=filename,
                            suffix='_L1_ramp',
                            overwrite=True)
                        final_corrected_image = ramp_image_ouput
                                #self.context.proctab.new_proctab()
                                #name1 = data_header['IMTYPE']
                                
                                #self.context.proctab.update_proctab(
                                #    frame=sci_im_full_original1, suffix="ramp", newtype='name1',
                                #    filename=data_header['OFNAME'])

                                #self.context.proctab.write_proctab(
                                #    tfil=self.config.instrument.procfile)
                                
                        if imtype == 'BIAS':
                            bias_ramps.append(final_corrected_image)

                        if imtype == 'DARK':
                            dark_ramps.append(final_corrected_image)

                        if imtype == 'FLATLENS':
                            flatlen_ramps.append(final_corrected_image)

                        if imtype == 'FLATLAMP':
                            flatlamp_ramps.append(final_corrected_image)

                        if imtype == 'CALUNIT':
                            calunit_ramps.append(final_corrected_image)
        
        bpm_from_darks = self.generate_bpm(
            np.array(dark_ramps),
            stack_name="Dark Stack",
            temporal_sigma_thresh=5.0,
            spatial_brightness_factor=10.0,
            plot_results=False)
        bpm_from_flats = self.generate_bpm(
            np.array(flatlamp_ramps),
            stack_name="Flat Stack",
            temporal_sigma_thresh=5.0,
            spatial_brightness_factor=1.5,
            plot_results=False)

        final_combined_bpm = bpm_from_darks | bpm_from_flats
        hdu = fits.PrimaryHDU(data=final_combined_bpm.astype(np.uint8))
        print(self.action.args.dirname)
        hdu.writeto(calib_path+'bpm.fits', overwrite=True)
        self.logger.info('+++++++++++++ Bad pixel mask created ++++++++++++++')
        master_dark = robust.mean(dark_ramps,axis=0)
        #master_bias = robust.mean(bias_ramps,axis=0)
        #master_flatlen = robust.mean(flatlen_ramps,axis=0)
        master_flatlamp = robust.mean(flatlamp_ramps,axis=0)
        #master_calunit = robust.mean(calunit_ramps,axis=0)
        self.logger.info('+++++++++++++ Master calibration files created ++++++++++++++')
        final_bpm = final_combined_bpm.astype(bool)
        dark_bpm_corrected = self.apply_full_correction(master_dark, final_bpm)
        #flatlen_bpm_corrected = self.apply_full_correction(master_flatlen, final_bpm)
        flatlamp_bpm_corrected = self.apply_full_correction(master_flatlamp, final_bpm)
        #calunit_bpm_corrected = self.apply_full_correction(master_calunit, final_bpm)
        self.logger.info('+++++++++++++ Bad pixel corrected ++++++++++++++')
        
        self.fits_writer_steps(
            data=dark_bpm_corrected,
            header=data_header,
            output_dir=self.action.args.dirname,
            input_filename=filename,
            suffix='_mdark',
            overwrite=True)
        #self.fits_writer_steps(
        #    data=flatlen_bpm_corrected,
        #    header=data_header,
        #    output_dir=self.action.args.dirname,
        #    input_filename=filename,
        #    suffix='_mflatlens',
        #    overwrite=True)
        self.fits_writer_steps(
            data=flatlamp_bpm_corrected,
            header=data_header,
            output_dir=self.action.args.dirname,
            input_filename=filename,
            suffix='_mflatlamp',
            overwrite=True)
        #self.fits_writer_steps(
        #    data=calunit_bpm_corrected,
        #    header=data_header,
        #    output_dir=self.action.args.dirname,
        #    input_filename=filename,
        #    suffix='_mdark',
        #    overwrite=True)
        
        log_string = StartCalib.__module__
        self.logger.info(log_string)
        return self.action.args










