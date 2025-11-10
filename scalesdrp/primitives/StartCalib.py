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

###################### ramp fitting reads<5 #############
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

    ################## ramp fitting reads > 5 ############################
    def ramp_fiting(self,input_read,total_exptime):
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
            self.logger.info('Number of reads are less than 5, starting a stright line fit to the reads')
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

    def ramp_fit(self,input_read, total_exptime):
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

        # === Load calibration maps ===
        calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
        SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits')
        FLUX_SCALING_FACTOR = 1.0
        JUMP_THRESH_ONEOMIT = 20.25
        JUMP_THRESH_TWOOMIT = 23.8

        nim_s = input_read.shape[0]
        read_times = np.linspace(0, total_exptime, nim_s)

        # === Case 1: Few reads (simple linear fit) ===
        if nim_s < 6:
            self.logger.info('Few reads (<6): Performing direct slope fit...')
            slope_map = self.iterative_sigma_weighted_ramp_fit(
                input_read,read_time=total_exptime)
            return output_final

        # === Case 2: Full ramp fitting ===
        self.logger.info("Applying σ-clipping to input ramp (tile-based)...")
        valid_reads_mask = sigma_clip_ramp_inputs(
            input_read, sigma_clip=3.0, max_iter=3, min_reads=5, tile=(128, 128)
        )

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
                    threshold_twoomit=JUMP_THRESH_TWOOMIT,
                )
                diffs2use &= diffs2use_fitramp

                result = fitramp.fit_ramps(
                    diffs_for_row, Covar_obj, current_sig_for_row,
                    diffs2use=diffs2use,
                    countrateguess=countrateguess,
                    rescale=True,
                )

                valid = np.isfinite(result.countrate) & (
                    result.countrate > 0.05 * np.nanmedian(countrateguess)
                )
                row_out = np.where(valid, result.countrate, countrateguess)

            except Exception:
                row_out = countrateguess  # fallback if fitramp fails entirely

            output_final[i, :] = row_out * FLUX_SCALING_FACTOR

        end_time = time.time()
        self.logger.info(f"Ramp fitting completed in {end_time - start_time:.2f} seconds.")

        # === Global smooth and cleanup ===
        #bad_mask_global = (output_final <= 0) | ~np.isfinite(output_final)
        #if np.any(bad_mask_global):
        #    output_final = self.masked_smooth_fast(output_final, bad_mask_global, sigma=1.0)
        #med = np.nanmedian(output_final[output_final > 0])
        #output_final[np.isnan(output_final)] = med

        return output_final
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

                    #saturation_map = linearity.create_saturation_map_by_slope(
                    #    science_ramp=sci_im_full_original1,
                    #    skip_reads=3,
                    #    slope_threshold=2.0,
                    #    smoothing_window=3)

                    #final_ramp, final_pixel_dq, final_group_dq = linearity.run_linearity_workflow(
                    #    science_ramp=sci_im_full_original1,
                    #    saturation_map=saturation_map,
                    #    obsmode = obsmode)
                    self.logger.info("+++++++++++ ramp fitting started +++++++++++")
                    slope = self.ramp_fit(sci_im_full_original1,total_exptime=readtime)

                    self.fits_writer_steps(
                        data=slope,
                        header=data_header,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix='_L1_ramp',
                        overwrite=True)
                    final_corrected_image = slope
                                #self.context.proctab.new_proctab()
                                #name1 = data_header['IMTYPE']
                                
                                #self.context.proctab.update_proctab(
                                #    frame=sci_im_full_original1, suffix="ramp", newtype='name1',
                                #    filename=data_header['OFNAME'])

                                #self.context.proctab.write_proctab(
                                #    tfil=self.config.instrument.procfile)
                                
                    if imtype == 'BIAS':
                        bias_ramps.append(final_corrected_image)
                        bias_header = data_header
                    if imtype == 'DARK':
                        dark_ramps.append(final_corrected_image)
                        dark_header = data_header

                    if imtype == 'FLATLENS':
                        flatlen_ramps.append(final_corrected_image)
                        flatlens_header = data_header
                    if imtype == 'FLATLAMP':
                        flatlamp_ramps.append(final_corrected_image)
                        flatlamp_header = data_header
                    if imtype == 'CALUNIT':
                        calunit_ramps.append(final_corrected_image)
                        calunit_header = data_header

                if len(dark_ramps) > 0:
                    master_dark = robust.mean(dark_ramps,axis=0)
                    self.logger.info("+++++++++++ BPM correction started for master dark +++++++++++")
                    bpm_master_dark = bpm.apply_full_correction(master_dark,obsmode)
                    self.logger.info("+++++++++++ BPM correction completed +++++++++++")
                    key = ('DARK', obsmode, ifsmode, filtername, mclock)
                    #dark_header['HISTORY'] = 'Master dark file'
                    
                    self.fits_writer_steps(
                        data=bpm_master_dark,
                        header=dark_header,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix='_mdark',
                        overwrite=True)

                if len(bias_ramps) > 0:
                    master_bias = robust.mean(bias_ramps,axis=0)
                    self.logger.info("+++++++++++ BPM correction started for master bias +++++++++++")
                    bpm_master_bias = bpm.apply_full_correction(master_bias,obsmode)
                    self.logger.info("+++++++++++ BPM correction completed +++++++++++")
                    #bias_header['HISTORY'] = 'Master bias file'
                    self.fits_writer_steps(
                        data=bpm_master_bias,
                        header=bias_header,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix='_mbias',
                        overwrite=True)

                if len(flatlen_ramps) > 0:
                    master_flatlens = robust.mean(flatlen_ramps,axis=0)
                    self.logger.info("+++++++++++ BPM correction started for master lenslet flat +++++++++++")
                    bpm_master_flatlens = bpm.apply_full_correction(master_flatlens,obsmode)
                    self.logger.info("+++++++++++ BPM correction completed +++++++++++")
                    #data_header['HISTORY'] = 'Master lenslet flat file'
                    self.fits_writer_steps(
                        data=bpm_master_flatlens,
                        header=flatlens_header,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix='_mflatlens',
                        overwrite=True)

                if len(flatlamp_ramps) > 0:
                    master_flatlamp = robust.mean(flatlamp_ramps,axis=0)

                    self.logger.info("+++++++++++ BPM correction started for master detector flat +++++++++++")
                    bpm_master_flatlamp = bpm.apply_full_correction(master_flatlamp,obsmode)
                    self.logger.info("+++++++++++ BPM correction completed +++++++++++")
                    #data_header['HISTORY'] = 'Master detector flat file'
                    self.fits_writer_steps(
                        data=bpm_master_flatlamp,
                        header=flatlamp_header,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix='_mflatlamp',
                        overwrite=True)

                if len(calunit_ramps) > 0:
                    master_calunit = robust.mean(calunit_ramps,axis=0)
                    self.logger.info("+++++++++++ BPM correction started for master calunit +++++++++++")
                    bpm_master_calunit = bpm.apply_full_correction(master_calunit,obsmode)
                    self.logger.info("+++++++++++ BPM correction completed +++++++++++")
                    #data_header['HISTORY'] = 'Master calunit file'
                    self.fits_writer_steps(
                        data=bpm_master_calunit,
                        header=calunit_header,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix='_mcalunit',
                        overwrite=True)
        
        self.logger.info('+++++++++++++ All available Master calibration files are created ++++++++++++++')
        self.logger.info('+++++++++++++ Ready to process the science exposures ++++++++++++++')
        
        log_string = StartCalib.__module__
        self.logger.info(log_string)
        return self.action.args










