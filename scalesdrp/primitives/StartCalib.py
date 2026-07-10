from keckdrpframework.primitives.base_primitive import BasePrimitive
import scalesdrp.primitives.fitramp as fitramp
import scalesdrp.primitives.robust as robust
import scalesdrp.primitives.reference as reference #1/f and reference pixel correction
import scalesdrp.primitives.linearity as linearity #linearity correction
import scalesdrp.primitives.bpm_correction as bpm #bpm correction
from scalesdrp.primitives.scales_file_primitives import fits_writer_calib
from scalesdrp.primitives.linearity import DQ_FLAGS
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
from scipy.signal import savgol_filter
from scipy.ndimage import convolve, median_filter
from scipy.interpolate import griddata
from scipy.stats import median_abs_deviation
from scipy.ndimage import median_filter
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve as astropy_convolve
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.ndimage import distance_transform_edt, gaussian_filter

from tqdm import tqdm

import logging
log = logging.getLogger("SCALES")
pt = Proctab(logger=log)


class StartCalib(BasePrimitive):
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
        self.logger.info("+++++++++++ SCALES calibration starting +++++++++++")
        dt = self.context.data_set.data_table
        all_groups = scbasic.group_files_by_header(dt)
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

        for imtype in processing_order:
            if imtype not in organized_groups:
                continue
            groups_for_this_type = organized_groups[imtype]
            for group in groups_for_this_type:
                params = group['params']
                filenames = group['filenames']
                obsmode = params.get('camera', 'obsmode')

                ifsmode = params.get('dsprsnam', 'N/A')
                filtername = params.get('imgfw2n', 'N/A')
                exptime = params.get('exptime', 0)
                mclock = params.get('mclock', 0)
                wavelength = params.get('monowave', None)
                #self.logger.info(f"Processing {imtype}: {len(filenames)} files "
                #    f"(OBSMODE={obsmode}, IFSMODE={ifsmode}, FILTER={filtername}, EXPTIME={exptime}, MCLOCK={mclock})")
                wl_str = f", WAVELENGTH={wavelength}" if wavelength is not None else ""
                self.logger.info(
                    f"Processing {imtype}: {len(filenames)} files "
                    f"(CAMERA={obsmode}, IFSMODE={ifsmode}, FILTER={filtername},"
                    f"EXPTIME={exptime}, MCLOCK={mclock}{wl_str})")
                group_ramps = []
                group_uncerts = []
                group_header_for_master = None

                for filename in filenames:

                    redux_dir = os.path.join(self.action.args.dirname, "redux")

                    existing_l1_name = scbasic.find_existing_proc_file(
                        input_filename=filename,
                        suffix="_L1",
                        redux_dir=redux_dir)

                    if existing_l1_name is not None:
                        l1_path = os.path.join(
                            self.action.args.dirname,
                            "redux",
                            os.path.basename(existing_l1_name))
                    else:
                        l1_path = scbasic.get_l1_path_from_raw(
                            input_filename = filename,
                            output_dir = redux_dir)

                    if self.context.clobber==False:
                        if os.path.exists(l1_path):
                            self.logger.info(f"Found existing L1 file: {l1_path}")
                            try:
                                l1_slope, l1_uncert, l1_header = scbasic.read_existing_l1(l1_path)
                                group_ramps.append(l1_slope)
                                group_uncerts.append(l1_uncert)

                                if group_header_for_master is None:
                                    group_header_for_master = l1_header.copy()

                                self.logger.info(f"Reusing existing L1 for {filename}. Skipping raw processing.")
                                continue

                            except Exception as e:
                                self.logger.warning(
                                    f"Existing L1 file could not be used: {l1_path}. "
                                    f"Reason: {e}. Reprocessing from raw file.")
                        try:
                            with fits.open(filename) as hdulist:
                                sci_im_full_original1 = hdulist[0].data
                                data_header = hdulist[0].header
                                readtime = data_header['EXPTIME']
                                det_config = data_header['MCLOCK']
                        except Exception as e:
                            self.logger.error(f"Failed to read {filename}: {e}")
                            continue

                    else:
                        try:
                            with fits.open(filename) as hdulist:
                                sci_im_full_original1 = hdulist[0].data
                                data_header = hdulist[0].header
                                readtime = data_header['EXPTIME']
                                det_config = data_header['MCLOCK']
                        except Exception as e:
                            self.logger.error(f"Failed to read {filename}: {e}")



                    package = __name__.split('.')[0]
                    #filepath = 'calib/'
                    #calib_path = str(get_resource_path(package, filepath))+'/'

                    calibfilepath = self.context.calib_file_path
                    calib_path = str(get_resource_path(package, calibfilepath))+'/'
                    if obsmode =='Im':
                        if det_config =='5.0 MHz':  #fast1.0
                            SIG_map_scaled = fits.getdata(calib_path+'readnoise_img_fast1.0_cd5.fits')
                            rmat1 = sparse.load_npz(calib_path+'bpmat_img.npz')
                            lin_coeff = calib_path+"lin_coeffs_img_fast1.0_cd5.fits"
                            master_bpm = fits.getdata(calib_path+'bpm_img_cd4.fits')

                        elif det_config =='9.0 MHz': #fast0.6
                            SIG_map_scaled = fits.getdata(calib_path+'readnoise_img_fast0.6_cd5.fits')
                            lin_coeff = calib_path+"lin_coeffs_img_fast0.6_cd5.fits"
                            rmat1 = sparse.load_npz(calib_path+'bpmat_img.npz')
                            master_bpm = fits.getdata(calib_path+'bpm_img_cd4.fits')

                        elif det_config =='20.0 MHz': #slow
                            SIG_map_scaled = fits.getdata(calib_path+'readnoise_img_slow_cd5.fits')
                            lin_coeff = calib_path+"lin_coeffs_img_slow_cd5.fits"
                            rmat1 = sparse.load_npz(calib_path+'bpmat_img.npz')
                            master_bpm = fits.getdata(calib_path+'bpm_img_cd4.fits')

                        else: #default if MCLCOCK is not the specified one above
                            SIG_map_scaled = fits.getdata(calib_path+'readnoise_img_fast1.0_cd5.fits')
                            lin_coeff = calib_path+"lin_coeffs_img_fast0.6_cd5.fits"
                            rmat1 = sparse.load_npz(calib_path+'bpmat_img.npz')
                            master_bpm = fits.getdata(calib_path+'bpm_img_cd4.fits')

                    elif obsmode =='IFS':
                        if det_config =='5.0 MHz':  #fast1.0
                            SIG_map_scaled = fits.getdata(calib_path+'readnoise_ifs_fast0.6_cd5.fits')
                            rmat1 = sparse.load_npz(calib_path+'bpmat_ifs.npz')
                            lin_coeff = calib_path+"lin_coeffs_ifs_fast1.0_cd5.fits"
                            master_bpm = fits.getdata(calib_path+'bpm_ifs_cd5.fits')

                        elif det_config =='9.0 MHz': #fast1.0
                            SIG_map_scaled = fits.getdata(calib_path+'readnoise_ifs_fast1.0_cd5.fits')
                            SIG_map_scaled[np.where(np.isnan(SIG_map_scaled)==True)] = np.nanmedian(SIG_map_scaled)
                            master_bpm = fits.getdata(calib_path+self.context.bpm_ifs_9mhz)
                            rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs_9mhz)
                            lin_coeff = calib_path+"lin_coeffs_ifs_fast0.6_cd5.fits"
                            #master_bpm = fits.getdata(calib_path+'bpm_ifs_cd5.fits')

                        elif det_config =='20.0 MHz': #slow
                            SIG_map_scaled = fits.getdata(calib_path+'readnoise_ifs_slow_cd5.fits')
                            rmat1 = sparse.load_npz(calib_path+'bpmat_ifs.npz')
                            lin_coeff = calib_path+"lin_coeffs_ifs_slow_cd5.fits"
                            master_bpm = fits.getdata(calib_path+'bpm_ifs_cd5.fits')

                        else: #default
                            SIG_map_scaled = fits.getdata(calib_path+'readnoise_ifs_fast1.0_cd5.fits')
                            rmat1 = sparse.load_npz(calib_path+'bpmat_ifs.npz')
                            lin_coeff = calib_path+"lin_coeffs_ifs_fast0.6_cd5.fits"
                            master_bpm = fits.getdata(calib_path+'bpm_ifs_cd5.fits')

                    #self.logger.info("+++++++++++ odd even swapping +++++++++++")
                    sci_im_full_original2 = scbasic.swap_odd_even_columns(sci_im_full_original1,do_swap=False)

                    self.logger.info("+++++++++++ ACN & 1/f correction started +++++++++++")
                    sci_im_full_original3 = reference.reffix_hxrg(sci_im_full_original2, nchans=4, fixcol=True)
                    data_header['HISTORY'] = "ACN & 1/f correction applied"

                    if  sci_im_full_original3.ndim == 2:

                        final_slope = sci_im_full_original3
                        uncert = scbasic.estimate_uncert_single_read(
                            image_dn=final_slope,
                            readnoise_map_dn=SIG_map_scaled,
                            gain=1.0)

                        dq_2d = None

                    elif sci_im_full_original3.ndim == 3:

                        self.logger.info("+++++++++++ linearity correction started +++++++++++")
                        corrected_cube, lin_dq, lin_mask = linearity.apply_linearity_coeffs_to_cube_fast(
                            input_cube=sci_im_full_original3,
                            coeff_file=lin_coeff,
                            bpm_2d=master_bpm,
                            invalid_read_behavior="raw",
                            use_goodpix=True,
                            return_aux=True)
                        self.logger.info("+++++++++++ ramp fitting started +++++++++++")
                        final_slope,reset,uncert = scbasic.ramp_fit(
                            #corrected_cube,
                            sci_im_full_original3,
                            readtime,
                            SIG_map_scaled,
                            group_dq = lin_dq) #keep group_dq=lin_dq when linearity is on otherwise None

                        dq_2d = np.bitwise_or.reduce(lin_dq, axis=0).astype(np.uint32)

                    self.logger.info("+++++++++++ Bad pixel correction started +++++++++++")

                    #dynamic mask section
                    #transient_mask, local_med, local_sigma, sig = bpm.detect_transient_bad_pixels(
                    #    final_slope,
                    #    master_bpm=master_bpm,
                    #    kernel_size=5,
                    #    sigma_thresh=7.0,
                    #    return_diagnostics=True)
                    #final_mask = master_bpm | transient_mask
                    #rmat = bpm.bpm_correction(final_mask)

                    final_ramp1 = rmat1*np.matrix(final_slope.flatten().reshape([np.prod(final_slope.shape),1]))
                    bpm_slope = np.array(final_ramp1).reshape(final_slope.shape)

                    final_ramp1_uncert = rmat1*np.matrix(uncert.flatten().reshape([np.prod(uncert.shape),1]))
                    bpm_slope_uncert = np.array(final_ramp1_uncert).reshape(uncert.shape)

                    data_header['HISTORY'] = "Default bad pixel correction applied"
                    self.logger.info("+++++++++++ Bad pixel correction completed +++++++++++")

                    fits_writer_calib(
                        data=bpm_slope,
                        header=data_header,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix='_L1',
                        overwrite=True,
                        uncert = bpm_slope_uncert,
                        dq=dq_2d)

                    scbasic.proctab_update(
                        header=data_header,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix="_L1",
                        frame=None,
                        proctab=self.proctab)

                    group_ramps.append(bpm_slope)
                    group_uncerts.append(bpm_slope_uncert)
                    if group_header_for_master is None:
                        group_header_for_master = data_header.copy()

                    self.logger.info("************** next file **************************")

                if len(group_ramps) == 0:
                    self.logger.warning(f"No valid frames to build master for group {imtype} {params}.")
                    continue

                if group_header_for_master is None:
                    raise RuntimeError(
                        "Internal error: group_header_for_master is None "
                        "even though group_ramps is not empty.")

                #expected_keywords = {
                #    "CAMERA": obsmode,
                #    "MCLOCK": mclock}
                #if imtype == "DARK":
                #    expected_keywords["EXPTIME"] = exptime
                #elif imtype == "FLATLENS":
                #    expected_keywords["IFSMODE"] = ifsmode

                #elif imtype == "CALUNIT":
                #    expected_keywords["MONOWAVE"] = wavelength

                #existing_master, existing_master_hdr = scbasic.load_single_master_file_calib(
                #    expected_keywords=expected_keywords,
                #    master_type=imtype)

                #if existing_master is not None:
                #    self.logger.info(
                #        f"Matching master already exists for {imtype} {params}. "
                #        "Skipping master creation.")
                    #continue

                master, master_unc = scbasic.build_master_from_stack(
                    group_ramps,
                    group_uncerts,
                    method='median',
                    iterations=0)

                hdrm = group_header_for_master.copy()
                hdrm['HISTORY'] = f"Master {imtype} built from {len(group_ramps)} frames"
                hdrm['HISTORY'] = f"Group params: CAMERA={obsmode}, IFSMODE={ifsmode},FILTER={filtername}, EXPTIME={exptime}, MCLOCK={mclock}"

                if imtype == 'DARK':

                    hdrm['HISTORY'] = "master detector DARK created"
                    scbasic.proctab_update(
                        header=hdrm,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix="_mdark",
                        frame=None,
                        proctab=self.proctab)

                    fits_writer_calib(
                        data=master,
                        header=hdrm,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix='_mdark',
                        overwrite=True,
                        uncert=master_unc)
                    self.logger.info("+++++++++++ Creating master dark +++++++++++")

                if imtype == 'BIAS':
                    hdrm['HISTORY'] = "master detector bias created"
                    fits_writer_calib(
                        data=master,
                        header=hdrm,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix='_mbias',
                        overwrite=True,
                        uncert=master_unc)
                    scbasic.proctab_update(
                        header=hdrm,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix="_mbias",
                        frame=None,
                        proctab=self.proctab)
                    self.logger.info("+++++++++++ Creating master bias +++++++++++")

                if imtype == 'FLATLENS':
                    hdrm['HISTORY'] = "master lenslet flat created"
                    fits_writer_calib(
                        data=master,
                        header=hdrm,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix='_mflatlens',
                        overwrite=True,
                        uncert=master_unc)
                    scbasic.proctab_update(
                        header=hdrm,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix="_mflatlens",
                        frame=None,
                        proctab=self.proctab)
                    self.logger.info("+++++++++++ Creating master lenslet flat +++++++++++")
                    self.logger.info("+++++++++++ Creating master lenslet flat cube +++++++++++")

                    package = __name__.split('.')[0]
                    simfile = 'sim_readnoise.fits'
                    readnoise = fits.getdata(calib_path+simfile)
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

                    A_guess_cube,A_guess_cube_err = scbasic.optimal_extract_with_error(
                        R_matrix,
                        master_flatlens,
                        master_flatlen_uncert,
                        var_read_vector)
                    A_opt = A_guess_cube.reshape(FLUX_SHAPE_3D)
                    A_opt_err = A_guess_cube_err.reshape(FLUX_SHAPE_3D)
                    fits_writer_calib(
                        data=A_opt,
                        header=hdrm,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix='_cube_flatlens',
                        overwrite=True,
                        uncert=A_opt_err)
                    scbasic.proctab_update(
                        header=hdrm,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix="_cube_flatlens",
                        frame=None,
                        proctab=self.proctab)

                if imtype == 'FLATLAMP':
                    hdrm['HISTORY'] = "master detector flat created"
                    fits_writer_calib(
                        data=master,
                        header=hdrm,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix='_mflatlamp',
                        overwrite=True,
                        uncert=master_unc)
                    scbasic.proctab_update(
                        header=hdrm,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix="_mflatlamp",
                        frame=None,
                        proctab=self.proctab)

                    self.logger.info("+++++++++++ Creating master detector flat +++++++++++")

                if imtype == 'CALUNIT' and wavelength is not None:
                    hdrm['HISTORY'] = f"CALUNIT wavelength group: {wavelength}"
                    wl_val = float(wavelength)
                    wl_str = f"{wl_val:.3f}".rstrip("0").rstrip(".")
                    hdrm['IMTYPE'] = 'MCALUNIT'
                    suffix = f"_{wl_str}_mcalunit"
                    fits_writer_calib(
                        data=master,
                        header=hdrm,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix=suffix,
                        overwrite=True,
                        uncert=master_unc)
                    scbasic.proctab_update(
                        header=hdrm,
                        output_dir=self.action.args.dirname,
                        input_filename=filename,
                        suffix=suffix,
                        frame=None,
                        proctab=self.proctab)
                    self.logger.info("+++++++++++ Creating master monochromator file +++++++++++")

        self.logger.info('+++++++++++++ All available Master calibration files are created ++++++++++++++')

        log_string = StartCalib.__module__
        self.logger.info(log_string)
        return self.action.args










