from keckdrpframework.primitives.base_primitive import BasePrimitive
import scalesdrp.primitives.fitramp as fitramp
import scalesdrp.primitives.robust as robust
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import scalesdrp.primitives.reference as reference #1/f and reference pixel correction
import scalesdrp.primitives.linearity_correct as linearity_correct #linearity correction
import scalesdrp.primitives.bpm_correction as bpm #bpm correction
import scalesdrp.primitives.saturation_correct as saturation_correct
from scalesdrp.primitives.scales_file_primitives import fits_writer_calib
from scalesdrp.core.scales_proctab import Proctab
from scalesdrp.core.scales_pkg_resources import get_resource_path
from scalesdrp.core.matplot_plotting import mpl_plot, mpl_clear
import scalesdrp.primitives.scales_basic as scbasic
import pandas as pd
import numpy as np
import pickle
from importlib.resources import files
from pathlib import Path
from astropy.io import fits
import warnings
from scipy import sparse
import astropy.io.fits as pyfits
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import time
from astropy.nddata import StdDevUncertainty
from scipy.signal import savgol_filter
from scipy.ndimage import convolve, median_filter
from scipy.interpolate import griddata
from scipy.stats import median_abs_deviation
from scipy.ndimage import median_filter
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve as astropy_convolve
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import scalesdrp.primitives.scales_basic as scbasic
import scalesdrp.primitives.robust as robust
from tqdm import tqdm

import logging
log = logging.getLogger("SCALES")
pt = Proctab(logger=log)


class StartCalibSA(BasePrimitive):
    """
    From reads to slope images of al the calibration exposures. This include
    detector flat, dark, bias, and monochromator exposures. This function will
    also make master calibration required for science processing.

    Include a ACN, 1/f correction, linearity correction, ramp fitting, and a bad pixel correction.
    We adopt the ramp fitting method of Brandt et. al. 2024.
    This method perform an optimal fit to a pixel’s count rate nondestructively in the
    presence of both read and photon noise. The method construct a covarience matrix by
    estimating the difference in the read in a ramp, propagation of the read noise,
    photon noise and their corelation. And Performs a generalized least squares fit
    to the differences, using the inverse of the covariance matrix as weights.
    This gives optimal weight to each difference.
    The jumps are detected iteratively checking the goodness of
    fit at each possible jump location.
        Args:
            data_image: The (N,H,W) input ramp cube.

        Returns:
            A 2D image of ramp fitted slope
            A 2D image of uncetainty of the ramp fitted slope
            Data Quality Flags
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger
        if not hasattr(self, "proctab") or self.proctab is None:
            self.proctab = Proctab(logger=self.logger if hasattr(self, "logger") else logging.getLogger("SCALES"))

    def _perform(self):
        imtype = self.action.args.ccddata.header['IMTYPE']
        total_exptime = self.action.args.ccddata.header['EXPTIME']
        obsmode = self.action.args.ccddata.header['CAMERA']
        det_config = self.action.args.ccddata.header['MCLOCK']
        filename = self.action.args.ccddata.header.get("OFNAME")
        package = __name__.split('.')[0]
        det_config = str(det_config).strip()
        calibfilepath = self.context.calib_file_path
        calib_path = str(get_resource_path(package, calibfilepath))+'/'
        plot_dir = self.config.instrument.plot_directory

        create_linearity = bool(self.config.instrument.create_linearity_coeff)
        create_sat = bool(self.config.instrument.create_sat_map)
        create_bpm = bool(self.config.instrument.create_bpm)
        create_rn = bool(self.config.instrument.create_readnoise)
        create_rectmat = bool(self.config.instrument.create_rectmat)

        if obsmode =='Im':
            rmat1 = sparse.load_npz(calib_path+self.context.bpmat_img)
            bpm = self.context.bpm_img
            master_bpm = fits.getdata(calib_path+bpm)

            if det_config =='5.0 MHz':  #fast1.0
                lin_coeff = calib_path+self.context.lin_coeff_img_fast1
                readnoise = self.context.sig_map_img_fast1
                SIG_map_scaled = fits.getdata(calib_path+readnoise)
                gain = self.context.gain_img_fast1
                #rmat1 = sparse.load_npz(calib_path+self.context.bpmat_img_fast1)
                #bpm = self.context.bpm_img_fast1
                #master_bpm = fits.getdata(calib_path+bpm)
                sat_map_file = self.context.sat_map_img_fast1
                with fits.open(calib_path+sat_map_file) as hdul:
                    sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

            elif det_config =='9.0 MHz': #fast0.6
                lin_coeff = calib_path+self.context.lin_coeff_img_fast0p6
                readnoise = self.context.sig_map_img_fast0p6
                SIG_map_scaled = fits.getdata(calib_path+readnoise)
                gain = self.context.gain_img_fast0p6
                #SIG_map_scaled[np.where(np.isnan(SIG_map_scaled)==True)] = np.nanmedian(SIG_map_scaled)
                #rmat1 = sparse.load_npz(calib_path+self.context.bpmat_img_fast0p6)
                #bpm = self.context.bpm_img_fast0p6
                #master_bpm = fits.getdata(calib_path+bpm)
                sat_map_file = self.context.sat_map_img_fast0p6
                with fits.open(calib_path+sat_map_file) as hdul:
                    sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

            elif det_config =='20.0 MHz': #slow5.2
                lin_coeff = calib_path+self.context.lin_coeff_img_slow
                readnoise = self.context.sig_map_img_slow
                SIG_map_scaled = fits.getdata(calib_path+readnoise)
                gain = self.context.gain_img_slow
                #rmat1 = sparse.load_npz(calib_path+self.context.bpmat_img_slow)
                #bpm = self.context.bpm_img_slow
                #master_bpm = fits.getdata(calib_path+bpm)
                sat_map_file = self.context.sat_map_img_slow
                with fits.open(calib_path+sat_map_file) as hdul:
                    sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

            else: #default if MCLCOCK is not one of those specified above
                lin_coeff = calib_path+self.context.lin_coeff_img_fast0p6
                readnoise = self.context.sig_map_img_fast0p6
                SIG_map_scaled = fits.getdata(calib_path+readnoise)
                gain = self.context.gain_img_fast0p6
                #rmat1 = sparse.load_npz(calib_path+self.context.bpmat_img_fast0p6)
                #bpm = self.context.bpm_img_fast0p6
                #master_bpm = fits.getdata(calib_path+bpm)
                sat_map_file = self.context.sat_map_img_fast0p6
                with fits.open(calib_path+sat_map_file) as hdul:
                    sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

        elif obsmode =='IFS':
            rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs)
            bpm = self.context.bpm_ifs
            master_bpm = fits.getdata(calib_path+bpm)

            if det_config =='5.0 MHz':  #fast1.0
                lin_coeff = calib_path+self.context.lin_coeff_ifs_fast1
                readnoise = self.context.sig_map_ifs_fast1
                SIG_map_scaled = fits.getdata(calib_path+readnoise)
                gain = self.context.gain_ifs_fast1
                #rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs_fast1)
                #bpm = self.context.bpm_ifs_fast1
                #master_bpm = fits.getdata(calib_path+bpm)
                sat_map_file = self.context.sat_map_ifs_fast1
                with fits.open(calib_path+sat_map_file) as hdul:
                    sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

            elif det_config =='9.0 MHz': #fast1.0
                lin_coeff = calib_path+self.context.lin_coeff_ifs_fast0p6
                readnoise = self.context.sig_map_ifs_fast0p6
                SIG_map_scaled = fits.getdata(calib_path+readnoise)
                gain = self.context.gain_ifs_fast0p6
                #rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs_fast0p6)
                #bpm = self.context.bpm_ifs_fast0p6
                #master_bpm = fits.getdata(calib_path+bpm)
                sat_map_file = self.context.sat_map_ifs_fast0p6
                with fits.open(calib_path+sat_map_file) as hdul:
                    sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

            elif det_config =='20.0 MHz': #slow
                lin_coeff = calib_path+self.context.lin_coeff_ifs_slow
                readnoise = self.context.sig_map_ifs_slow
                SIG_map_scaled = fits.getdata(calib_path+readnoise)
                gain = self.context.gain_ifs_slow
                #rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs_slow)
                #bpm = self.context.bpm_ifs_slow
                #master_bpm = fits.getdata(calib_path+bpm)
                sat_map_file = self.context.sat_map_ifs_slow
                with fits.open(calib_path+sat_map_file) as hdul:
                    sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

            else: #default
                lin_coeff = calib_path+self.context.lin_coeff_ifs_fast0p6
                readnoise = self.context.sig_map_ifs_fast0p6
                SIG_map_scaled = fits.getdata(calib_path+readnoise)
                gain = self.context.gain_ifs_fast0p6
                #rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs_fast0p6)
                #bpm = self.context.bpm_ifs_fast0p6
                #master_bpm = fits.getdata(calib_path+bpm)
                sat_map_file = self.context.sat_map_ifs_fast0p6
                with fits.open(calib_path+sat_map_file) as hdul:
                    sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

        input_data = self.action.args.ccddata.data
        imtype = str(imtype).strip().upper()
        requested_suffixes = []
        if imtype == "FLATLAMP":
            if create_linearity:
                requested_suffixes.append("_L10")
            if create_sat:
                requested_suffixes.append("_L11")
        if imtype in ("DARK", "FLATLAMP") and create_bpm:
            requested_suffixes.append("_L12")
        if imtype == "DARK" and create_rn:
            requested_suffixes.append("_L13")
        if imtype == "CALUNIT" and create_rectmat:
            requested_suffixes.append("_L1")

        if not requested_suffixes:
            self.logger.info("No calibration products requested for IMTYPE=%s; skipping %s", imtype, filename)
            return self.action.args

        existing_products = {}
        if not self.context.clobber:
            for suffix in requested_suffixes:
                existing_products[suffix] = scbasic.find_existing_proc_file(
                    input_filename=filename,
                    redux_dir=self.config.instrument.output_directory,
                    suffix=suffix,
                )
            if all(existing_products[suffix] is not None for suffix in requested_suffixes):
                self.logger.info("All requested products %s exist for %s; skipping", requested_suffixes, filename)
                return self.action.args

        do_l10 = "_L10" in requested_suffixes
        do_l11 = "_L11" in requested_suffixes
        do_l12 = "_L12" in requested_suffixes
        do_l13 = "_L13" in requested_suffixes
        do_l1 = "_L1" in requested_suffixes

        #self.logger.info("+++++++++++ odd even column swapping +++++++++++")
        if self.config.instrument.do_swap == True:
            input_data = scbasic.swap_odd_even_columns(input_data,do_swap=True)
            self.action.args.ccddata.header['HISTORY'] = 'detector level odd-even correction applied in the DRP'
            self.logger.info("detector level odd-even correction applied in the DRP")
        #print('NaNs in the input data=',np.isnan(input_data).sum())
            
        self.logger.info("refpix and 1/f correction started")            
        sci_im_full_original = reference.reffix_hxrg(
            cube =input_data,
            nchans=self.config.instrument.nchans,
            altcol=self.config.instrument.altcol,
            channelwise=self.config.instrument.channelwise,
            amp_mean_func=scbasic.resolve_mean_func(self.config.instrument.amp_mean_func),
            do_acn=self.config.instrument.do_acn,
            acn_avg_type=self.config.instrument.acn_avg_type,
            acn_mean_func=scbasic.resolve_mean_func(self.config.instrument.acn_mean_func),
            acn_smooth=self.config.instrument.acn_smooth,
            acn_savgol=self.config.instrument.acn_savgol,
            acn_winsize=self.config.instrument.acn_winsize,
            acn_order=self.config.instrument.acn_order,
            resid_colsub=self.config.instrument.resid_colsub,
            fixcol=self.config.instrument.fixcol,
            ref_avg_type=self.config.instrument.ref_avg_type,
            ref_mean_func=scbasic.resolve_mean_func(self.config.instrument.ref_mean_func),
            ref_smooth=self.config.instrument.ref_smooth,
            ref_savgol=self.config.instrument.ref_savgol,
            ref_winsize=self.config.instrument.ref_winsize,
            ref_order=self.config.instrument.ref_order,
            pickup=self.config.instrument.pickup,
            sigma_thresh=self.config.instrument.sigma_thresh,
            dilate_iter=self.config.instrument.dilate_iter,
            highpass_size=self.config.instrument.highpass_size,
            per_amp=self.config.instrument.per_amp)

        hdr = self.action.args.ccddata.header
        hdr['HISTORY'] = 'Refpix and 1/f correction applied'
        hdr['HISTORY'] = (
            f'nchans={self.config.instrument.nchans}, '
            f'altcol={self.config.instrument.altcol}, '
            f'channelwise={self.config.instrument.channelwise}')
        hdr['HISTORY'] = (
            f'amp_mean_func={self.config.instrument.amp_mean_func}')
        hdr['HISTORY'] = (
            f'do_acn={self.config.instrument.do_acn}, '
            f'acn_avg_type={self.config.instrument.acn_avg_type}, '
            f'acn_mean_func={self.config.instrument.acn_mean_func}')
        hdr['HISTORY'] = (
            f'acn_smooth={self.config.instrument.acn_smooth}, '
            f'acn_savgol={self.config.instrument.acn_savgol}, '
            f'acn_winsize={self.config.instrument.acn_winsize}, '
            f'acn_order={self.config.instrument.acn_order}')
        hdr['HISTORY'] = (
            f'resid_colsub={self.config.instrument.resid_colsub}, '
            f'fixcol={self.config.instrument.fixcol}')
        hdr['HISTORY'] = (
            f'ref_avg_type={self.config.instrument.ref_avg_type}, '
            f'ref_mean_func={self.config.instrument.ref_mean_func}')
        hdr['HISTORY'] = (
            f'ref_smooth={self.config.instrument.ref_smooth}, '
            f'ref_savgol={self.config.instrument.ref_savgol}, '
            f'ref_winsize={self.config.instrument.ref_winsize}, '
            f'ref_order={self.config.instrument.ref_order}')
        hdr['HISTORY'] = (
            f'pickup={self.config.instrument.pickup}, '
            f'sigma_thresh={self.config.instrument.sigma_thresh}, '
            f'dilate_iter={self.config.instrument.dilate_iter}')
        hdr['HISTORY'] = (
            f'highpass_size={self.config.instrument.highpass_size}, '
            f'per_amp={self.config.instrument.per_amp}')

        self.logger.info("refpix and 1/f correction completed")

        plot_path = scbasic.plot_read_correction_comparison(
            cube_before=input_data,
            cube_after=sci_im_full_original,
            output_dir=plot_dir,
            filename=f"{self.action.args.name}_acn_onef_comp.png",
            title="SCALES ACN and 1/f Correction",
            unit="DN",
            show=False)
        #print('NaNs in the reference pixel corrected data=',np.isnan(sci_im_full_original).sum())

        #saving ACN&1/f corrected reads to create linearity files
        if do_l10:
            if imtype != 'FLATLAMP' or sci_im_full_original.ndim != 3:
                msg = (
                    "create_linearity_coeff=True requires a 3-D "
                    f"FLATLAMP ramp cube. Received IMTYPE='{imtype}', "
                    f"ndim={sci_im_full_original.ndim}.")
                self.logger.error(msg)
                raise ValueError(msg)
            
            if (self.context.clobber is False and existing_products.get("_L10") is not None):
                self.logger.info(f"L10 already exists for {filename}. ""Skipping L10 write.")

            else:
                self.action.args.ccddata.header = hdr
                self.action.args.ccddata.data = sci_im_full_original

                scales_fits_writer(
                    self.action.args.ccddata,
                    table=self.action.args.table,
                    output_file=self.action.args.name,
                    output_dir=self.config.instrument.output_directory,
                    quality_map=None,
                    chisq=None,
                    suffix="L10")

                scbasic.proctab_update(
                    header=self.action.args.ccddata.header,
                    output_dir=self.config.instrument.output_directory,
                    input_filename=self.action.args.name,
                    suffix="_L10",
                    frame=None,
                    proctab=self.proctab)

                self.logger.info("L10 written for linearity-coefficient creation")

            if not (do_l11 or do_l12 or do_l1):
                self.logger.info( 
                    "No additional calibration products requested. "
                    "Stopping StartCalib.")
                return self.action.args

        if do_l13:
            if imtype != 'DARK' or sci_im_full_original.ndim != 3:
                msg = (
                    "create_readnoise=True requires a 3-D "
                    f"DARK ramp cube. Received IMTYPE='{imtype}', "
                    f"ndim={sci_im_full_original.ndim}.")
                self.logger.error(msg)
                raise ValueError(msg)
            
            if (self.context.clobber is False and existing_products.get("_L13") is not None):
                self.logger.info(f"L13 already exists for {filename}. ""Skipping L13 write.")

            else:
                self.action.args.ccddata.header = hdr
                self.action.args.ccddata.data = sci_im_full_original

                scales_fits_writer(
                    self.action.args.ccddata,
                    table=self.action.args.table,
                    output_file=self.action.args.name,
                    output_dir=self.config.instrument.output_directory,
                    quality_map=None,
                    chisq=None,
                    suffix="L13")

                scbasic.proctab_update(
                    header=self.action.args.ccddata.header,
                    output_dir=self.config.instrument.output_directory,
                    input_filename=self.action.args.name,
                    suffix="_L13",
                    frame=None,
                    proctab=self.proctab)

                self.logger.info("L13 written for readnoise map")

            if not (do_l11 or do_l12 or do_l1):
                self.logger.info( 
                    "No additional calibration products requested. "
                    "Stopping StartCalib.")
                return self.action.args

        self.logger.info("+++++++++++ detector level corrections are completed +++++++++++")
        
        # ============================================================
        # Validate saturation-map creation
        # ============================================================
        if do_l11:            
            if imtype != 'FLATLAMP' or sci_im_full_original.ndim != 3:
                msg = (
                    "create_sat_map=True requires a 3-D FLATLAMP ramp cube. "
                    f"Received IMTYPE='{imtype}', "
                    f"ndim={sci_im_full_original.ndim}.")
                self.logger.error(msg)
                raise ValueError(msg)

            if self.config.instrument.do_linearity != True:
                msg = (
                    "create_sat_map=True requires non-linearity correction. "
                    "Please set do_linearity=True to create a saturation map.")
                self.logger.error(msg)
                raise ValueError(msg)

        raw_before_linearity = None
        corrected_cube = None
        sat_mask = None
        applied_mask = None
        
        if sci_im_full_original.ndim ==2:
            self.action.args.ccddata.data = sci_im_full_original
            self.action.args.ccddata.header['HISTORY'] = 'Non-linearity correction is NOT applied'

        elif sci_im_full_original.ndim ==3:
            if self.config.instrument.do_linearity == True:
                self.logger.info("+++++++++++ linearity correction started +++++++++++")
                raw_before_linearity = sci_im_full_original.copy()

                corrected_cube, pedestal, sat_mask, applied_mask = (
                    linearity_correct.apply_brandt_linearity_reference(
                        input_cube=sci_im_full_original,
                        coefficient_file=lin_coeff,
                        n_pedestal_reads=2,
                        pedestal_start_read=0,
                        saturation_fraction=0.95,
                        saturated_read_behavior="raw",
                        apply_only_successful=False,
                        return_aux=True))

                self.action.args.ccddata.data = corrected_cube
                self.action.args.ccddata.header['HISTORY'] = (f'Linearity correction applied using {os.path.basename(lin_coeff)}')
                self.logger.info("+++++++++++ linearity correction finished +++++++++++")

                #saving ACN&1/f corrected reads to create saturation map
                if do_l11:
                    if (self.context.clobber is False and existing_products.get("_L11") is not None):
                        self.logger.info(f"L11 already exists for {filename}. ""Skipping L11 write.")

                    else:
                        scales_fits_writer(
                            self.action.args.ccddata,
                            table=self.action.args.table,
                            output_file=self.action.args.name,
                            output_dir=self.config.instrument.output_directory,
                            quality_map=None,
                            chisq=None,
                            suffix="L11")

                        scbasic.proctab_update(
                            header=self.action.args.ccddata.header,
                            output_dir=self.config.instrument.output_directory,
                            input_filename=self.action.args.name,
                            suffix="_L11",
                            frame=None,
                            proctab=self.proctab)
                    
                        self.logger.info("L11 written for saturation-map creation.")
                    if not (do_l12 or do_l1):
                        self.logger.info("No additional products requested after L11.")
                        return self.action.args
            else:
                self.action.args.ccddata.data = sci_im_full_original
        

        # ============================================================
        # Validate BPM creation
        # ============================================================
        if do_l12:
            if imtype not in ('DARK', 'FLATLAMP'):
                msg = (
                    "create_bpm=True requires IMTYPE='DARK' or 'FLATLAMP'. "
                    f"Received IMTYPE='{imtype}'.")
                self.logger.error(msg)
                raise ValueError(msg)
        
        ############# RAMP FITTING ###################################################
        quality_map = None
        chisq = None
        sci_im_full_original = self.action.args.ccddata.data
        
        if sci_im_full_original.ndim ==2:
            final_slope = sci_im_full_original
            uncert = scbasic.estimate_uncert_single_read(
                image_dn=final_slope,
                readnoise_map_dn=SIG_map_scaled,
                gain=1.0)
            
            final_slope *= gain   # e-/s
            uncert *= gain        # e-/s

            self.action.args.ccddata.dq = None
            self.action.args.ccddata.chisq = None

        elif sci_im_full_original.ndim ==3:
            
            if self.config.instrument.apply_sat_mask == True:
                self.logger.info("+++++++++++ Masking saturated neighbours started +++++++++++")
                quality_map, good_read_mask = saturation_correct.make_ramp_quality_mask(
                    sci_im_full_original,
                    sat_map,
                    bpm=master_bpm,
                    neighbor_radius=1)

                fig, axes = scbasic.plot_linearity_saturation_diagnostics(
                    raw_cube=raw_before_linearity,
                    corrected_cube=corrected_cube,
                    saturation_mask=sat_mask,
                    applied_mask=applied_mask,
                    good_read_mask=good_read_mask,
                    output_dir=plot_dir,
                    save_prefix=f"{self.action.args.name}_lin_dign",
                    skip_initial_reads=2,
                    minimum_signal_dn=0.0,
                    show=False)
                plt.close(fig)
                
                self.action.args.ccddata.dq = quality_map.astype(np.uint32)
                self.action.args.ccddata.header['HISTORY'] = (f'Saturation map applied using {os.path.basename(sat_map_file)}')
                self.logger.info("+++++++++++ ramp fitting started +++++++++++")
                
                final_slope,final_reset,uncert,chisq = scbasic.ramp_fit(
                    sci_im_full_original,
                    total_exptime,
                    SIG_map_scaled,
                    group_dq = good_read_mask)
                
                final_slope *= gain   # e-/s
                final_reset *= gain   # e-
                uncert *= gain        # e-/s
                
            elif self.config.instrument.apply_sat_mask == False:
                self.logger.info("+++++++++++ Saturation correction is turned off +++++++++++")
                self.action.args.ccddata.dq = None
                self.action.args.ccddata.header['HISTORY'] = (f'Saturation map NOT applied')    
                
                self.logger.info("+++++++++++ ramp fitting started +++++++++++")
                final_slope,final_reset,uncert,chisq = scbasic.ramp_fit(
                    sci_im_full_original,
                    total_exptime,
                    SIG_map_scaled,
                    group_dq = None)
                
                final_slope *= gain   # e-/s
                final_reset *= gain   # e-
                uncert *= gain        # e-/s

            self.action.args.ccddata.header['HISTORY'] =(f'Readnoise map applied using {os.path.basename(readnoise)}')
            self.action.args.ccddata.chisq = chisq

        ###################### BPM ####################################
        # ============================================================
        # CREATE BPM INPUT PRODUCT
        # L12 = ramp-fitted slope BEFORE existing BPM correction
        # ============================================================

        if do_l12:
            if (
                self.context.clobber is False
                and existing_products.get("_L12") is not None):
                self.logger.info(
                    f"L12 already exists for {filename}. "
                    "Skipping L12 write.")
                return self.action.args

            self.action.args.ccddata.data = final_slope
            self.action.args.ccddata.uncertainty = StdDevUncertainty(uncert.astype(np.float32))
            if chisq is not None:
                self.action.args.ccddata.chisq = chisq.astype(np.float32)
            else:
                self.action.args.ccddata.chisq = None

            scales_fits_writer(
                self.action.args.ccddata,
                table=self.action.args.table,
                output_file=self.action.args.name,
                output_dir=self.config.instrument.output_directory,
                quality_map=quality_map,
                chisq=chisq,
                suffix="L12")

            scbasic.proctab_update(
                header=self.action.args.ccddata.header,
                output_dir=self.config.instrument.output_directory,
                input_filename=self.action.args.name,
                suffix="_L12",
                frame=None,
                proctab=self.proctab)

            self.logger.info(
                "L12 written for BPM creation. "
                "Existing BPM correction is skipped.")
            if not do_l1:
                return self.action.args

        if self.config.instrument.apply_bpm == True:
            final_ramp1 = rmat1*np.matrix(final_slope.flatten().reshape([np.prod(final_slope.shape),1]))
            final_ramp = np.array(final_ramp1).reshape(final_slope.shape)

            final_ramp1_uncert = rmat1*np.matrix(uncert.flatten().reshape([np.prod(uncert.shape),1]))
            final_uncert = np.array(final_ramp1_uncert).reshape(uncert.shape)
            self.action.args.ccddata.header['HISTORY'] =(f'bad pixel map applied using {os.path.basename(bpm)}')
            self.logger.info("+++++++++++ Bad pixel correction completed +++++++++++")

            plot_path = scbasic.plot_bad_pixel_correction_comparison(
                slope_before=final_slope,
                slope_after=final_ramp,
                output_dir=plot_dir,
                filename=f"{self.action.args.name}_bpm_comp.png",
                title="SCALES — Bad-Pixel Correction",
                unit="e-/s",
                show=False)
        else:
            self.logger.info("+++++++++++ Bad pixel correction is turned off +++++++++++")
            final_ramp = final_slope
            final_uncert = uncert
            self.action.args.ccddata.header['HISTORY'] =(f'bad pixel map is NOT applied')
            
        
        #print('NaNs in the bpm corrected slope data=',np.isnan(final_ramp).sum())

        if self.config.instrument.subtract_row_median==True:
            for ii in range(len(final_ramp)):
                final_ramp[ii]-=np.nanmedian(final_ramp[ii])

        if self.config.instrument.subtract_col_median==True:
            for ii in range(len(final_ramp[0])):
                final_ramp[:,ii]-=np.nanmedian(final_ramp[:,ii])

        if self.config.instrument.subtract_img_readout_channels==True:
            bounds = 4+510*np.array(range(5))
            bias = np.zeros([2048,2048])
            for i in range(4):
                xstart = bounds[i]
                xstop = bounds[i+1]
                arr = final_ramp[:,xstart:xstop]
                med = np.nanmedian(arr)
                bias[:,xstart:xstop] = med
            final_ramp = final_ramp-bias

        self.action.args.ccddata.data = final_ramp
        self.action.args.ccddata.uncertainty = StdDevUncertainty(final_uncert.astype(np.float32))

        if chisq is not None:
            self.action.args.ccddata.chisq = chisq.astype(np.float32)
        else:
            self.action.args.ccddata.chisq = None

        if do_l1 and (self.context.clobber or existing_products.get("_L1") is None):

            scales_fits_writer(
                self.action.args.ccddata,
                table=self.action.args.table,
                output_file=self.action.args.name,
                output_dir=self.config.instrument.output_directory,
                quality_map=quality_map,
                chisq=chisq,
                suffix="L1")

            scbasic.proctab_update(
                header=self.action.args.ccddata.header,
                output_dir=self.config.instrument.output_directory,
                input_filename=self.action.args.name,
                suffix="_L1",
                frame=None,
                proctab=self.proctab)

            log_string = StartCalibSA.__module__
            self.logger.info(log_string)

        return self.action.args










