from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import scalesdrp.primitives.fitramp as fitramp
import numpy as np
from astropy.io import fits
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import time
import os
import pkg_resources
import shutil
from pathlib import Path
from collections import defaultdict

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
        self.logger.info("Ramp fitting starting")

    def _perform(self):

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

        def group_fits_cubes_by_header(
            folder_path: str,
            imagetype_keyword: str = 'IMTYPE',
            exptime_keyword: str = 'EXPTIME',
            filter_keyword: str = 'FILTER'
        ) -> dict:
            """
            Scans a directory, finds all 3D FITS cubes, and groups their file paths
            based on the values of header keywords.
            The files are grouped into a nested dictionary structure:
            {imagetype: {filter: {exptime: [filepath1, filepath2, ...]}}}

            Args:
                folder_path (str): The path to the directory containing FITS files.
                imagetype_keyword (str): The FITS header keyword for the image type.
                exptime_keyword (str): The FITS header keyword for exposure time.
                filter_keyword (str): The FITS header keyword for the filter name.
            Returns:
                dict: A nested dictionary containing the grouped file paths.
            """
            folder = Path(folder_path)
            if not folder.is_dir():
                raise FileNotFoundError(f"Directory not found: {folder_path}")
            categorized_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            fits_files = list(folder.glob('*.fits*'))
            for filepath in sorted(fits_files):
                try:
                    with fits.open(filepath) as hdul:
                        if hdul[0].data is None or hdul[0].data.ndim != 3:
                            continue
                        header = hdul[0].header
                        img_type = header.get(imagetype_keyword, 'UNKNOWN').strip().upper()
                        filter_name = header.get(filter_keyword, 'UNKNOWN').strip().upper()
                        exp_time = float(header.get(exptime_keyword, -1.0))
                        categorized_files[img_type][filter_name][exp_time].append(str(filepath))
            return categorized_files        

        def create_job_list_from_groups(
            grouped_files: dict,
            min_files_in_group: int = 1
        ) -> list[dict]:
            """
            Transforms the nested dictionary from group_fits_cubes_by_header into a
            flat list of "job" dictionaries.
            Each job dictionary represents a unique group of files and contains all
            the information needed for subsequent processing.
            Args:
                grouped_files (dict): The nested dictionary from group_fits_cubes_by_header.
                min_files_in_group (int): The minimum number of files a group must have
                                  to be included in the job list.
            Returns:
                list[dict]: A flat list of job dictionaries. Each dictionary has the keys:
                    'imagetype', 'filter', 'exptime', and 'file_list'.
            """
            job_list = []
            for img_type, filter_groups in grouped_files.items():
                for filter_name, exp_groups in filter_groups.items():
                    for exp_time, file_list in exp_groups.items():                
                        if len(file_list) < min_files_in_group:
                            continue
                        job = {
                            'imagetype': img_type,
                            'filter': filter_name,
                            'exptime': exp_time,
                            'file_list': file_list  # This is the array of filenames you wanted
                        }
                        job_list.append(job)
            return job_list

        if all_jobs:
            for job in all_jobs:
                filenames_for_this_group = job['file_list']
                for fits_file_path in filenames_for_this_group:

        NUM_FRAMES_FROM_SCIENCE = self.action.args.ccddata.header['NREADS']
        FLUX_SCALING_FACTOR = 1.0
        SATURATION_DARK_OLS = 50000.0 / FLUX_SCALING_FACTOR
        SATURATION_SCIENCE_OLS = 4096.0 / FLUX_SCALING_FACTOR
        DEFAULT_SIG_FALLBACK_SCALED = 5.0 / FLUX_SCALING_FACTOR 
        JUMP_THRESH_ONEOMIT = 20.25 #4.5 sigma
        JUMP_THRESH_TWOOMIT = 23.8

        calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
        SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits')

        temp_dir = '/Users/athira/Desktop/scales_drp/header/'
        grouped_files = group_fits_cubes_by_header(str(temp_dir))
        all_jobs = create_job_list_from_groups(grouped_files, min_files_in_group=1)

        sci_im_full_original = self.action.args.ccddata.data
        nim_sci_file = sci_im_full_original.shape[0]
        nim_s = min(NUM_FRAMES_FROM_SCIENCE, nim_sci_file)
        if nim_s < 2:
            print('will do a 2nd order fit here')
        else:
            sci_im_original_units = sci_im_full_original[:nim_s, :, :]
            sci_im_scaled = sci_im_original_units / FLUX_SCALING_FACTOR
            sci_im_with_jumps_scaled = sci_im_scaled.copy()
            ols_t_global = np.arange(nim_s)
            readtimes_for_covar_sci = np.arange(nim_s).astype(float) #need to take from the header
            B_ols_sci = np.zeros((2048, 2048), dtype=float)

            for i_r in range(2048):
                for j_c in range(2048):
                    a_g=sci_im_with_jumps_scaled[0,i_r,j_c]; b_g=(sci_im_with_jumps_scaled[1,i_r,j_c]-sci_im_with_jumps_scaled[0,i_r,j_c]) if nim_s>1 else 1.0; c_g=0.0
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

        self.action.args.ccddata.data = output_fitramp_final
        log_string = RampFit.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)


        scales_fits_writer(self.action.args.ccddata,
            table=self.action.args.table,
            output_file=self.action.args.name,
            output_dir=self.config.instrument.output_directory,
            suffix="ramp")

        return self.action.args
    # END: class RampFit()
