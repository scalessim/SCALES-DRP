from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import scalesdrp.primitives.robust as robust
import scalesdrp.primitives.scales_basic as scbasic
import numpy as np
from astropy.io import fits
import time
import os
from tqdm import tqdm
from astropy.nddata import StdDevUncertainty
from scalesdrp.core.scales_proctab import Proctab
from scalesdrp.core.scales_pkg_resources import get_resource_path
import logging
log = logging.getLogger("SCALES")


class CalibCorrect(BasePrimitive):

    """
    This function applies master detector bias, master dark, and masster detector flat
    to the bad pixel corrected slope image from the previous step. You can turn on/off
    each of this using config file. Appropriate master files are selected from the redux/ 
    or calib/ folder in the order of listed here. 
        Args:
            data_image: The (H,W) bad pixel corrected slope image.

        Returns:
            A 2D image after the calibration applied
            A 2D image of uncetainty assosciated
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

            final_ramp = self.action.args.ccddata.data
            final_uncert = self.action.args.ccddata.uncertainty
            chisq = self.action.args.ccddata.chisq
            quality_map = self.action.args.ccddata.dq

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

            keywords_unique = {
                key: self.action.args.ccddata.header.get(key)
                for key in ['CAMERA', 'MCLOCK', 'EXPTIME']}

            if self.config.instrument.apply_dark == True:
                m_dark, m_dark_uncert = scbasic.load_single_master_file(keywords_unique, master_type='DARK')
                if m_dark is not None:
                    final_ramp, final_uncert = scbasic.apply_calibration(
                        final_ramp,
                        final_uncert,
                        m_dark,
                        m_dark_uncert,
                        imtype='DARK')
                    self.action.args.ccddata.header['HISTORY'] = 'Dark subtracted.'
                    self.logger.info("+++++++++++ Master dark subtracted +++++++++++")

            if self.config.instrument.apply_bias == True:
                m_bias, m_bias_uncert = scbasic.load_single_master_file(keywords_unique, master_type='BIAS')
                if m_bias is not None:
                    final_ramp, final_uncert = scbasic.apply_calibration(
                        final_ramp,
                        final_uncert,
                        m_bias,
                        m_bias_uncert,
                        imtype='BIAS')
                    self.action.args.ccddata.header['HISTORY'] = 'Bias subtraction applied.'
                    self.logger.info("+++++++++++ Master bias subtracted +++++++++++")

            if self.config.instrument.apply_det_flat == True:
                m_flat, m_flat_uncert = scbasic.load_single_master_file(keywords_unique, master_type='FLATLAMP')
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
            #self.action.args.ccddata.uncertainty = StdDevUncertainty(final_uncert.astype(np.float32))
            self.action.args.ccddata.uncertainty = StdDevUncertainty(final_uncert.array.astype(np.float32))
            self.action.args.ccddata.chisq = StdDevUncertainty(chisq.astype(np.float32))

            log_string = CalibCorrect.__module__
            self.action.args.ccddata.header['HISTORY'] = log_string
            self.logger.info(log_string)

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

            self.logger.info("+++++++++++ slope image FITS file saved +++++++++++")
        else:
            self.logger.info("+++++++++++ No science files detected to process +++++++++++")
        return self.action.args
    # END: class CalibCorrect()
