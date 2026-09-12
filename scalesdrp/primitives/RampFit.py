from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import scalesdrp.primitives.fitramp as fitramp
import scalesdrp.primitives.scales_basic as scbasic
import numpy as np
from astropy.io import fits
import time
import os
from scipy import sparse
import scalesdrp.primitives.saturation_correct as saturation_correct
import scalesdrp.primitives.bpm_correction as bpm #bpm correction
from tqdm import tqdm
from astropy.nddata import StdDevUncertainty
from scalesdrp.core.scales_proctab import Proctab
from scalesdrp.core.scales_pkg_resources import get_resource_path
import logging
log = logging.getLogger("SCALES")
pt = Proctab(logger=log)
from multiprocessing import Pool

class RampFit(BasePrimitive):

    """
    This function starts with a linearity corrected cube. Then first perform
        1. Masking the pixels and nesrest neighbours get saturated using the
           saturation map externally provided. You can turn on/off the step using the
           config file. Major module involved is 
           `import scalesdrp.primitives.saturation_correct as saturation_correct`

        2. Ramp fitting:We adopt the ramp fitting method of Brandt et. al. 2024.
           This method perform an optimal fit to a pixel’s count rate nondestructively in the
           presence of both read and photon noise. The method construct a covarience matrix by
           estimating the difference in the read in a ramp, propagation of the read noise,
           photon noise and their corelation. And Performs a generalized least squares fit
           to the differences, using the inverse of the covariance matrix as weights.
           This gives optimal weight to each difference. The jumps are detected iteratively
           checking the goodness of fit at each possible jump location. Major module involved is
           `import scalesdrp.primitives.fitramp as fitramp` and 
           `import scalesdrp.primitives.scales_basic as scbasic`.

        3. Bad pixel correction using existing bad pixel map. You can turn on/off the step using the
           config file.

        Args:
            data_image: The (N,H,W) input ramp cube (linearity corrected).

        Returns:
            A 2D image of ramp fitted slope
            A 2D image of uncetainty of the ramp fitted slope
            A 2D image of best-fit chi-square values of ramp fitting
            A 2D map of Data Quality Flag
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
            filename = self.action.args.ccddata.header.get("OFNAME")
            package = __name__.split('.')[0]
            det_config = str(det_config).strip()

            calibfilepath = self.context.calib_file_path
            calib_path = str(get_resource_path(package, calibfilepath))+'/'
            if obsmode =='Im':
                if det_config =='5.0 MHz':  #fast1.0
                    readnoise = self.context.sig_map_img_fast1
                    SIG_map_scaled = fits.getdata(calib_path+readnoise)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_img_fast1)
                    bpm = self.context.bpm_img_fast1
                    master_bpm = fits.getdata(calib_path+bpm)
                    sat_map_file = self.context.sat_map_img_fast1
                    with fits.open(calib_path+sat_map_file) as hdul:
                        sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

                elif det_config =='9.0 MHz': #fast0.6
                    readnoise = self.context.sig_map_img_fast0p6
                    SIG_map_scaled = fits.getdata(calib_path+readnoise)
                    SIG_map_scaled[np.where(np.isnan(SIG_map_scaled)==True)] = np.nanmedian(SIG_map_scaled)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_img_fast0p6)
                    bpm = self.context.bpm_img_fast0p6
                    master_bpm = fits.getdata(calib_path+bpm)
                    sat_map_file = self.context.sat_map_img_fast0p6
                    with fits.open(calib_path+sat_map_file) as hdul:
                        sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

                elif det_config =='20.0 MHz': #slow
                    readnoise = self.context.sig_map_img_slow
                    SIG_map_scaled = fits.getdata(calib_path+readnoise)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_img_slow)
                    bpm = self.context.bpm_img_slow
                    master_bpm = fits.getdata(calib_path+bpm)
                    sat_map_file = self.context.sat_map_img_slow
                    with fits.open(calib_path+sat_map_file) as hdul:
                        sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

                else: #default if MCLCOCK is not one of those specified above
                    readnoise = self.context.sig_map_img_fast0p6
                    SIG_map_scaled = fits.getdata(calib_path+readnoise)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_img_fast0p6)
                    bpm = self.context.bpm_img_fast0p6
                    master_bpm = fits.getdata(calib_path+bpm)
                    sat_map_file = self.context.sat_map_img_fast0p6
                    with fits.open(calib_path+sat_map_file) as hdul:
                        sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

            elif obsmode =='IFS':
                if det_config =='5.0 MHz':  #fast1.0
                    readnoise = self.context.sig_map_ifs_fast1
                    SIG_map_scaled = fits.getdata(calib_path+readnoise)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs_fast1)
                    bpm = self.context.bpm_ifs_fast1
                    master_bpm = fits.getdata(calib_path+bpm)
                    sat_map_file = self.context.sat_map_ifs_fast1
                    with fits.open(calib_path+sat_map_file) as hdul:
                        sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

                elif det_config =='9.0 MHz': #fast1.0
                    readnoise = self.context.sig_map_ifs_fast0p6
                    SIG_map_scaled = fits.getdata(calib_path+readnoise)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs_fast0p6)
                    bpm = self.context.bpm_ifs_fast0p6
                    master_bpm = fits.getdata(calib_path+bpm)
                    sat_map_file = self.context.sat_map_ifs_fast0p6
                    with fits.open(calib_path+sat_map_file) as hdul:
                        sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

                elif det_config =='20.0 MHz': #slow
                    readnoise = self.context.sig_map_ifs_slow
                    SIG_map_scaled = fits.getdata(calib_path+readnoise)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs_slow)
                    bpm = self.context.bpm_ifs_slow
                    master_bpm = fits.getdata(calib_path+bpm)
                    sat_map_file = self.context.sat_map_ifs_slow
                    with fits.open(calib_path+sat_map_file) as hdul:
                        sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

                else: #default
                    readnoise = self.context.sig_map_ifs_fast0p6
                    SIG_map_scaled = fits.getdata(calib_path+readnoise)
                    rmat1 = sparse.load_npz(calib_path+self.context.bpmat_ifs_fast0p6)
                    bpm = self.context.bpm_ifs_fast0p6
                    master_bpm = fits.getdata(calib_path+bpm)
                    sat_map_file = self.context.sat_map_ifs_fast0p6
                    with fits.open(calib_path+sat_map_file) as hdul:
                        sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)

            sci_im_full_original = self.action.args.ccddata.data
            
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
            
            if sci_im_full_original.ndim ==2:
                final_slope = sci_im_full_original
                uncert = scbasic.estimate_uncert_single_read(
                    image_dn=final_slope,
                    readnoise_map_dn=SIG_map_scaled,
                    gain=1.0)
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
                    self.action.args.ccddata.dq = quality_map.astype(np.uint32)
                    self.action.args.ccddata.header['HISTORY'] = (f'Saturation map applied using {os.path.basename(sat_map_file)}')
                    self.logger.info("+++++++++++ ramp fitting started +++++++++++")
                    final_slope,final_reset,uncert,chisq = scbasic.ramp_fit(
                        sci_im_full_original,
                        total_exptime,
                        SIG_map_scaled,
                        group_dq = good_read_mask)
                
                elif self.config.instrument.apply_sat_mask == False:

                    self.action.args.ccddata.dq = None
                    self.action.args.ccddata.header['HISTORY'] = (f'Saturation map NOT applied')    
                
                    self.logger.info("+++++++++++ ramp fitting started +++++++++++")
                    final_slope,final_reset,uncert,chisq = scbasic.ramp_fit(
                        sci_im_full_original,
                        total_exptime,
                        SIG_map_scaled,
                        group_dq = None)

                self.action.args.ccddata.header['HISTORY'] =(f'Readnoise map applied using {os.path.basename(readnoise)}')

                #print('NaNs in the slope data=',np.isnan(final_slope).sum())
                self.action.args.ccddata.chisq = chisq

            self.logger.info("+++++++++++ Bad pixel correction started +++++++++++")

            if self.config.instrument.apply_bpm == True:
                final_ramp1 = rmat1*np.matrix(final_slope.flatten().reshape([np.prod(final_slope.shape),1]))
                final_ramp = np.array(final_ramp1).reshape(final_slope.shape)

                final_ramp1_uncert = rmat1*np.matrix(uncert.flatten().reshape([np.prod(uncert.shape),1]))
                final_uncert = np.array(final_ramp1_uncert).reshape(uncert.shape)
                self.action.args.ccddata.header['HISTORY'] =(f'bad pixel map applied using {os.path.basename(bpm)}')
            
            else:
                final_ramp = final_slope
                final_uncert = uncert
                self.action.args.ccddata.header['HISTORY'] =(f'bad pixel map is NOT applied')
            
            self.logger.info("+++++++++++ Bad pixel correction completed +++++++++++")
            #print('NaNs in the bpm corrected slope data=',np.isnan(final_ramp).sum())

            self.action.args.ccddata.data = final_ramp
            self.action.args.ccddata.uncertainty = final_uncert
            
            log_string = RampFit.__module__
            self.action.args.ccddata.header['HISTORY'] = log_string
            self.logger.info(log_string)
        else:
            self.logger.info("+++++++++++ No science files detected to process +++++++++++")
        return self.action.args
    # END: class RampFit()
