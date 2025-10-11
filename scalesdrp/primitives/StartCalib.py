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

        required_cols = ['OBSMODE', 'IFSMODE', 'IM-FW-1', 'IMTYPE', 'EXPTIME']
        missing = [c for c in required_cols if c not in dt.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return []

        file_groups = []

        # --- Split by observing mode first
        for mode, mode_df in dt.groupby('OBSMODE'):
            if mode.upper() == 'IMAGING':
                group_keys = ['OBSMODE', 'IM-FW-1', 'IMTYPE', 'EXPTIME']
                self.logger.info(f"Grouping IMAGING data by {group_keys}")
            elif mode.upper() in ('LOWRES', 'MEDRES'):
                group_keys = ['OBSMODE', 'IFSMODE', 'IMTYPE', 'EXPTIME']
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

###################### dealing with saturated pixel with nans #############
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

    def ramp_fit(self,input_read,total_exptime):
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
            read_times = np.linspace(0, total_exptime, nim_s)
            output_fitramp_final = fit_slope_image(reads, read_times,SIG_map_scaled)
        else:
            sci_im_original_units = input_read[:nim_s, :, :]
            sci_im_scaled = sci_im_original_units / FLUX_SCALING_FACTOR
            sci_im_with_jumps_scaled = sci_im_scaled.copy()
            ols_t_global = np.arange(nim_s)
            readtimes_for_covar_sci = np.linspace(0, total_exptime, nim_s)
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
                self.logger.info(f"Processing {imtype}: {len(filenames)} files "
                    f"(OBSMODE={obsmode}, IFSMODE={ifsmode}, FILTER={filtername}, EXPTIME={exptime})")
                for filename in filenames:
                    try:
                        with fits.open(filename) as hdulist:
                            #sci_im_full_original1 = hdulist[0]
                            sci_im_full_original1 = hdulist[1].data
                            data_header = hdulist[0].header
                            readtime = data_header['EXPTIME']
                    except Exception as e:
                        self.logger.error(f"Failed to read {filename}: {e}")

                    #saturation_map = linearity.create_saturation_map_by_slope(
                    #    science_ramp=sci_im_full_original1,
                    #    skip_reads=3,
                    #    slope_threshold=2.0,
                    #    smoothing_window=3)

                    #final_ramp, final_pixel_dq, final_group_dq = linearity.run_linearity_workflow(
                    #    science_ramp=sci_im_full_original1,
                    #    saturation_map=saturation_map,
                    #    obsmode = obsmode)

                    slope = self.ramp_fit(sci_im_full_original1,total_exptime=readtime)
                    filled, nanmask = self.inpaint_nearest_fast(slope)
                    ramp_image_ouput = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
                    self.logger.info("+++++++++++ ramp fitting completed +++++++++++")
                    
                    #ramp_image_ouput = bpm.apply_full_correction(ramp_image_ouput1,obsmode)
                    #self.logger.info("+++++++++++ BPM correction completed +++++++++++")

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
        
        #bpm_from_darks = bpm.generate_bpm(
        #    np.array(dark_ramps),
        #    stack_name="Dark Stack",
        #    temporal_sigma_thresh=5.0,
        #    spatial_brightness_factor=10.0,
        #    plot_results=False)
        #bpm_from_flats = bpm.generate_bpm(
        #    np.array(flatlamp_ramps),
        #    stack_name="Flat Stack",
        #    temporal_sigma_thresh=5.0,
        #    spatial_brightness_factor=1.5,
        #    plot_results=False)

        #final_combined_bpm = bpm_from_darks | bpm_from_flats
        #hdu = fits.PrimaryHDU(data=final_combined_bpm.astype(np.uint8))
        #print(self.action.args.dirname)
        #hdu.writeto(calib_path+'bpm.fits', overwrite=True)
        #self.logger.info('+++++++++++++ Bad pixel mask created ++++++++++++++')

        if dark_ramps > 0:
            master_dark = robust.mean(dark_ramps,axis=0)
            self.fits_writer_steps(
                data=master_dark,
                header=data_header,
                output_dir=self.action.args.dirname,
                input_filename=filename,
                suffix='_mdark',
                overwrite=True)

        if bias_ramps > 0:
            master_bias = robust.mean(bias_ramps,axis=0)
            self.fits_writer_steps(
                data=master_bias,
                header=data_header,
                output_dir=self.action.args.dirname,
                input_filename=filename,
                suffix='_mdark',
                overwrite=True)

        if flatlen_ramps > 0:
            master_flatlen = robust.mean(flatlen_ramps,axis=0)
            self.fits_writer_steps(
                data=master_flatlen,
                header=data_header,
                output_dir=self.action.args.dirname,
                input_filename=filename,
                suffix='_mdark',
                overwrite=True)

        if flatlamp_ramps > 0:
            master_flatlamp = robust.mean(flatlamp_ramps,axis=0)
            self.fits_writer_steps(
                data=master_flatlamp,
                header=data_header,
                output_dir=self.action.args.dirname,
                input_filename=filename,
                suffix='_mdark',
                overwrite=True)

        if calunit_ramps > 0:
            master_calunit = robust.mean(calunit_ramps,axis=0)
            self.fits_writer_steps(
                data=master_calunit,
                header=data_header,
                output_dir=self.action.args.dirname,
                input_filename=filename,
                suffix='_mdark',
                overwrite=True)
        
        self.logger.info('+++++++++++++ Master calibration files created ++++++++++++++')
        
        log_string = StartCalib.__module__
        self.logger.info(log_string)
        return self.action.args










