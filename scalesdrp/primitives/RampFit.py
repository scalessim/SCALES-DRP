from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import scalesdrp.primitives.fitramp as fitramp
import scalesdrp.primitives.robust as robust
import scalesdrp.primitives.scales_basic as scbasic
import numpy as np
from astropy.io import fits
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import time
import os
from scipy.signal import savgol_filter
from scipy import sparse
import scalesdrp.primitives.reference as reference #1/f and reference pixel correction
import scalesdrp.primitives.linearity as linearity #linearity correction
import scalesdrp.primitives.bpm_correction as bpm #bpm correction
from tqdm import tqdm
from astropy.nddata import StdDevUncertainty
from scalesdrp.primitives.linearity import DQ_FLAGS

from scalesdrp.core.scales_proctab import Proctab
from scalesdrp.core.scales_pkg_resources import get_resource_path
import logging
log = logging.getLogger("SCALES")
pt = Proctab(logger=log)
from multiprocessing import Pool

class RampFit(BasePrimitive):

    """
    This function convert a raw read to a slope image of an OBJECT.
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
        if imtype =='OBJECT':
            total_exptime = self.action.args.ccddata.header['EXPTIME']
            obsmode = self.action.args.ccddata.header['CAMERA']
            det_config = self.action.args.ccddata.header['MCLOCK']
            package = __name__.split('.')[0]
            det_config = str(det_config).strip()

            calibfilepath = self.context.calib_file_path
            calib_path = str(get_resource_path(package, calibfilepath))+'/'
            if obsmode =='Im':
                if det_config =='5.0 MHz':  #fast1.0
                    SIG_map_scaled = fits.getdata(calib_path+self.context.sig_map_img_fast1)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_img_fast1)
                    lin_coeff = calib_path+self.context.lin_coeff_img_fast1
                    master_bpm = fits.getdata(calib_path+self.context.bpm_img_fast1)

                elif det_config =='9.0 MHz': #fast0.6
                    SIG_map_scaled = fits.getdata(calib_path+self.context.sig_map_img_fast0p6)
                    SIG_map_scaled[np.where(np.isnan(SIG_map_scaled)==True)] = np.nanmedian(SIG_map_scaled)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_img_fast0p6)
                    lin_coeff = calib_path+self.context.lin_coeff_img_fast0p6
                    master_bpm = fits.getdata(calib_path+self.context.bpm_img_fast0p6)

                elif det_config =='20.0 MHz': #slow
                    SIG_map_scaled = fits.getdata(calib_path+self.context.sig_map_img_slow)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_img_slow)
                    lin_coeff = calib_path+self.context.lin_coeff_img_slow
                    master_bpm = fits.getdata(calib_path+self.context.bpm_img_slow)

                else: #default if MCLCOCK is not one of those specified above
                    SIG_map_scaled = fits.getdata(calib_path+self.context.sig_map_img_fast0p6)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_img_fast0p6)
                    lin_coeff = calib_path+self.context.lin_coeff_img_fast0p6
                    master_bpm = fits.getdata(calib_path+self.context.bpm_img_fast0p6)

            elif obsmode =='IFS':
                if det_config =='5.0 MHz':  #fast1.0
                    SIG_map_scaled = fits.getdata(calib_path+self.context.sig_map_ifs_fast1)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs_fast1)
                    lin_coeff = calib_path+self.context.lin_coeff_ifs_fast1
                    master_bpm = fits.getdata(calib_path+self.context.bpm_ifs_fast1)

                elif det_config =='9.0 MHz': #fast1.0
                    SIG_map_scaled = fits.getdata(calib_path+self.context.sig_map_ifs_fast0p6)
                    if True in np.isnan(SIG_map_scaled):
                        print('nans in sig map')
                        stop
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs_fast0p6)
                    lin_coeff = calib_path+self.context.lin_coeff_ifs_fast0p6
                    master_bpm = fits.getdata(calib_path+self.context.bpm_ifs_fast0p6)

                elif det_config =='20.0 MHz': #slow
                    SIG_map_scaled = fits.getdata(calib_path+self.context.sig_map_ifs_slow)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs_slow)
                    lin_coeff = calib_path+self.context.lin_coeff_ifs_slow
                    master_bpm = fits.getdata(calib_path+self.context.bpm_ifs_slow)

                else: #default
                    SIG_map_scaled = fits.getdata(calib_path+self.context.sig_map_ifs_fast0p6)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs_fast0p6)
                    lin_coeff = calib_path+self.context.lin_coeff_ifs_fast0p6
                    master_bpm = fits.getdata(calib_path+self.context.bpm_ifs_fast0p6)

            input_data = self.action.args.ccddata.data

            filename = self.action.args.ccddata.header.get("OFNAME")



            existing_l1_name = scbasic.find_existing_proc_file(
                input_filename=filename,
                suffix="_L1",
                redux_dir=self.config.instrument.output_directory)


            if existing_l1_name is not None:
                l1_path = existing_l1_name
            else:
                l1_path = scbasic.get_l1_path_from_raw(
                    input_filename = filename,
                    output_dir = self.config.instrument.output_directory)

            if self.context.clobber==False:
                if os.path.exists(l1_path):
                    self.logger.info(f"Found existing L1 file: {l1_path}")
                    try:
                        l1_slope, l1_uncert, l1_header = scbasic.read_existing_l1(l1_path)
                        self.action.args.ccddata.data = l1_slope
                        self.action.args.ccddata.header = l1_header

                        self.action.args.ccddata.uncertainty = StdDevUncertainty(l1_uncert)

                        self.logger.info(f"Reusing existing L1 for {filename}. Skipping raw processing.")
                        return self.action.args

                    except Exception as e:
                        self.logger.warning(
                                    f"Existing L1 file could not be used: {l1_path}. "
                                    f"Reason: {e}. Reprocessing from raw file.")

            #self.logger.info("+++++++++++ odd even column swapping +++++++++++")
            sci_im_full_original1 = scbasic.swap_odd_even_columns(input_data,do_swap=False)

            self.logger.info("refpix and 1/f correction started")
            sci_im_full_original = reference.reffix_hxrg(sci_im_full_original1, nchans=4)
            self.action.args.ccddata.header['HISTORY'] = 'Refpix and 1/f correction applied'
            self.logger.info("refpix and 1/f correction completed")

            if sci_im_full_original.ndim ==2:
                final_slope = sci_im_full_original
                uncert = scbasic.estimate_uncert_single_read(
                    image_dn=final_slope,
                    readnoise_map_dn=SIG_map_scaled,
                    gain=1.0)
                self.action.args.ccddata.dq = None

            elif sci_im_full_original.ndim ==3:
                self.logger.info("+++++++++++ linearity correction started +++++++++++")

                corrected_cube, lin_dq, lin_mask = linearity.apply_linearity_coeffs_to_cube_safe_fast(
                    input_cube=sci_im_full_original,
                    coeff_file=lin_coeff,
                    bpm_2d=master_bpm,
                    invalid_read_behavior="raw",
                    chunk_size=4096,
                    return_aux=True)

                self.action.args.ccddata.header['HISTORY'] = 'Non-linearity correction applied'
                self.logger.info("+++++++++++ linearity correction finished +++++++++++")
                self.logger.info("+++++++++++ ramp fitting started +++++++++++")

                final_slope,final_reset,uncert = scbasic.ramp_fit(
                    corrected_cube,
                    #sci_im_full_original,
                    total_exptime,
                    SIG_map_scaled,
                    group_dq = lin_dq) #change group_dq=lin_dq

                self.action.args.ccddata.dq = np.bitwise_or.reduce(lin_dq, axis=0).astype(np.uint32)

            self.logger.info("+++++++++++ Bad pixel correction started +++++++++++")
            #final_ramp = bpm.apply_full_correction(final_slope,obsmode)
            #transient_mask, local_med, local_sigma, sig = bpm.detect_transient_bad_pixels(
            #    final_slope,
            #    master_bpm=master_bpm,
            #    kernel_size=5,
            #    sigma_thresh=7.0,
            #    return_diagnostics=True)
            #final_mask = master_bpm | transient_mask
            #rmat = bpm.bpm_correction(final_mask)
            final_ramp1 = rmat1*np.matrix(final_slope.flatten().reshape([np.prod(final_slope.shape),1]))
            final_ramp = np.array(final_ramp1).reshape(final_slope.shape)

            final_ramp1_uncert = rmat1*np.matrix(uncert.flatten().reshape([np.prod(uncert.shape),1]))
            final_uncert = np.array(final_ramp1_uncert).reshape(uncert.shape)

            self.action.args.ccddata.header['HISTORY'] = 'Bad pixel correction applied'

            self.logger.info("+++++++++++ Bad pixel correction completed +++++++++++")
            keywords_unique = {
                key: self.action.args.ccddata.header.get(key)
                for key in ['CAMERA', 'MCLOCK', 'EXPTIME']}

            m_dark, m_dark_uncert = scbasic.load_single_master_file(keywords_unique, master_type='DARK')
            m_bias, m_bias_uncert = scbasic.load_single_master_file(keywords_unique, master_type='BIAS')
            m_flat, m_flat_uncert = scbasic.load_single_master_file(keywords_unique, master_type='FLATLAMP')

            if m_dark is not None:
                final_ramp, final_uncert = scbasic.apply_calibration(
                    final_ramp,
                    final_uncert,
                    m_dark,
                    m_dark_uncert,
                    imtype='DARK')
                self.action.args.ccddata.header['HISTORY'] = 'Dark subtracted.'
                self.logger.info("+++++++++++ Master dark subtracted +++++++++++")

            if m_bias is not None:
                final_ramp, final_uncert = scbasic.apply_calibration(
                    final_ramp,
                    final_uncert,
                    m_bias,
                    m_bias_uncert,
                    imtype='BIAS')
                self.action.args.ccddata.header['HISTORY'] = 'Bias subtraction applied.'
                self.logger.info("+++++++++++ Master bias subtracted +++++++++++")

            if m_flat is not None:
                norm_flat,norm_flat_uncert = scbasic.normalize_detector_flat(m_flat,m_flat_uncert)
                final_ramp, final_uncert = scbasic.apply_calibration(
                    final_ramp,
                    final_uncert,
                    norm_flat,
                    norm_flat_uncert,
                    imtype='FLATLAMP')
                self.action.args.ccddata.header['HISTORY'] = 'Detector Flat correction applied.'
                self.logger.info("+++++++++++ detector flat correction completed  +++++++++++")

            if self.context.subtract_row_median==True:
                for ii in range(len(final_ramp)):
                    final_ramp[ii]-=np.nanmedian(final_ramp[ii])

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

            log_string = RampFit.__module__
            self.action.args.ccddata.header['HISTORY'] = log_string
            self.logger.info(log_string)

            scales_fits_writer(
                self.action.args.ccddata,
                table=self.action.args.table,
                output_file=self.action.args.name,
                output_dir=self.config.instrument.output_directory,
                suffix="L1")

            scbasic.proctab_update(
                header=self.action.args.ccddata.header,
                output_dir=self.config.instrument.output_directory,
                input_filename=self.action.args.name,
                suffix="_L1",
                frame=None,
                proctab=self.proctab)

            self.logger.info("+++++++++++ slope image FITS file saved +++++++++++")
        else:
            self.logger.info("+++++++++++ No science files detected to process +++++++++++")
        return self.action.args
    # END: class RampFit()
