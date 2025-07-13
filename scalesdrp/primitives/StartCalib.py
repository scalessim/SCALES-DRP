from keckdrpframework.primitives.base_primitive import BasePrimitive
#from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import scalesdrp.primitives.fitramp as fitramp
import pandas as pd
import numpy as np
import pickle
from astropy.io import fits
import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)
import pkg_resources
from scipy import sparse
import astropy.io.fits as pyfits
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import time

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

        grouping_keys = ['SCMODE', 'TYPE', 'EXPTIME']
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

    def fits_writer_steps(self,data,header,output_dir,input_filename,suffix,overwrite=True):
        base_name = os.path.basename(input_filename)
        file_root, file_ext = os.path.splitext(base_name)
        output_filename = f"{file_root}{suffix}{file_ext}"
        output_path = os.path.join(output_dir, output_filename)
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_path, overwrite=overwrite)
        return output_path


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

        processing_order = ['BIAS', 'DARK', 'FLATLAMP','FLATLEN', 'CALUNIT']
        self.logger.info(f"Found groups for IMTYPEs: {list(organized_groups.keys())}")
        bias_ramps=[]
        dark_ramps=[]
        flatlamp_ramps=[]
        flatlen_ramps=[]
        calunit_ramps=[]


        FLUX_SCALING_FACTOR = 1.0
        SATURATION_DARK_OLS = 50000.0 / FLUX_SCALING_FACTOR
        SATURATION_SCIENCE_OLS = 4096.0 / FLUX_SCALING_FACTOR
        DEFAULT_SIG_FALLBACK_SCALED = 5.0 / FLUX_SCALING_FACTOR 
        JUMP_THRESH_ONEOMIT = 20.25 #4.5 sigma
        JUMP_THRESH_TWOOMIT = 23.8

        calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
        SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits')


        for imtype in processing_order:
            print(imtype)
            if imtype in organized_groups:
                groups_for_this_type = organized_groups[imtype]
                for group in groups_for_this_type:
                    params = group['params']
                    filenames = group['filenames']
                    for filename in filenames:
                        try:
                            with fits.open(filename) as hdulist:
                                #sci_im_full_original1 = hdulist[0]
                                sci_im_full_original = hdulist[0].data
                                data_header = hdulist[0].header
                                NUM_FRAMES_FROM_SCIENCE = data_header['NREADS']
                        except Exception as e:
                            self.logger.error(f"Failed to read {filename}: {e}")
                        
                        nim_sci_file = sci_im_full_original.shape[0]
                        nim_s = min(NUM_FRAMES_FROM_SCIENCE, nim_sci_file)
                        
                        
                        if nim_s < 2:
                            print('will add a 2nd order fit here')
                        else:
                            sci_im_original_units = sci_im_full_original[:nim_s, :, :]
                            sci_im_scaled = sci_im_original_units / FLUX_SCALING_FACTOR
                            sci_im_with_jumps_scaled = sci_im_scaled.copy()
                            ols_t_global = np.arange(nim_s) #time need to update
                            readtimes_for_covar_sci = np.arange(nim_s).astype(float)
                            B_ols_sci = np.zeros((2048, 2048), dtype=float)
                            

                            for i_r in range(2048):
                                for j_c in range(2048):
                                    a_g=sci_im_with_jumps_scaled[0,i_r,j_c]; b_g=(sci_im_with_jumps_scaled[1,i_r,j_c]-sci_im_with_jumps_scaled[0,i_r,j_c]) if nim_s>1 else 1.0; c_g=0.0
                                    sp=self.ols_pack_parms(a_g,b_g,c_g); imdat_p=sci_im_with_jumps_scaled[:,i_r,j_c]
                                    w_idx=np.where(imdat_p < SATURATION_SCIENCE_OLS)[0]
                                    if len(w_idx)<3: B_ols_sci[i_r,j_c]=np.nan; continue
                                    imdat_v = imdat_p[w_idx]
                                    def resid_fn_sci_local(p_loc): mimdat_f=self.ols_model_fn(p_loc); return imdat_v - mimdat_f[w_idx]
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
                                #print(self.action.args.dirname)
                                self.fits_writer_steps(
                                    data=output_fitramp_final,
                                    header=data_header,
                                    output_dir=self.action.args.dirname,
                                    input_filename=filename,
                                    suffix='_ramp',
                                    overwrite=True)

                                #self.context.proctab.new_proctab()
                                #name1 = data_header['IMTYPE']
                                
                                #self.context.proctab.update_proctab(
                                #    frame=sci_im_full_original1, suffix="ramp", newtype='name1',
                                #    filename=data_header['OFNAME'])

                                #self.context.proctab.write_proctab(
                                #    tfil=self.config.instrument.procfile)
                                
                                #if imtype == 'BIAS':
                                #    bias_ramps.append(output_fitramp_final)

                                #if imtype == 'DARK':
                                #    dark_ramps.append(output_fitramp_final)



                                



            log_string = StartCalib.__module__
            self.logger.info(log_string)
        return self.action.args










