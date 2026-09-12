from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import scalesdrp.primitives.robust as robust
import scalesdrp.primitives.scales_basic as scbasic
import numpy as np
import time
import os
import scalesdrp.primitives.linearity_correct as linearity_correct #linearity correction
from astropy.nddata import StdDevUncertainty
from scalesdrp.core.scales_proctab import Proctab
from scalesdrp.core.scales_pkg_resources import get_resource_path
import logging
log = logging.getLogger("SCALES")

class LinearityCorrect(BasePrimitive):

    """
    This function start with  detector level corrected cube of read and perform
    a linearity correction using previously estimated linearity correction coefficients
    from a set of detector level corrected flat reads. So if you are skipping the 
    detector level correction step, use a linearity coefficient file derived from 
    a set of flat reads without detector level correction.
    Major module involved is `import scalesdrp.primitives.linearity_correct`
    You can turn on/off the step using the config file.

        Args:
            data_image: The (N,H,W) input ramp cube (detector level corrected).

        Returns:
            ata_image: The (N,H,W) linearity corrected ramp cube.
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
                    lin_coeff = calib_path+self.context.lin_coeff_img_fast1

                elif det_config =='9.0 MHz': #fast0.6
                    lin_coeff = calib_path+self.context.lin_coeff_img_fast0p6

                elif det_config =='20.0 MHz': #slow
                    lin_coeff = calib_path+self.context.lin_coeff_img_slow

                else: #default if MCLCOCK is not one of those specified above
                    lin_coeff = calib_path+self.context.lin_coeff_img_fast0p6

            elif obsmode =='IFS':
                if det_config =='5.0 MHz':  #fast1.0
                    lin_coeff = calib_path+self.context.lin_coeff_ifs_fast1

                elif det_config =='9.0 MHz': #fast1.0
                    lin_coeff = calib_path+self.context.lin_coeff_ifs_fast0p6

                elif det_config =='20.0 MHz': #slow
                    lin_coeff = calib_path+self.context.lin_coeff_ifs_slow

                else: #default
                    lin_coeff = calib_path+self.context.lin_coeff_ifs_fast0p6

            sci_im_full_original = self.action.args.ccddata.data

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
            
            if sci_im_full_original.ndim ==2:
                self.action.args.ccddata.data = sci_im_full_original
                self.action.args.ccddata.header['HISTORY'] = 'Non-linearity correction is NOT applied'
            
            elif sci_im_full_original.ndim ==3:
                if self.config.instrument.do_linearity == True:
                    self.logger.info("+++++++++++ linearity correction started +++++++++++")
                    corrected_cube, pedestal, sat_mask, applied_mask = (
                        linearity_correct.apply_brandt_linearity_reference(
                            input_cube=sci_im_full_original,
                            coefficient_file=lin_coeff,
                            n_pedestal_reads=2,
                            pedestal_start_read=0,
                            saturation_fraction=0.95,
                            saturated_read_behavior="raw",
                            apply_only_successful=False,
                            return_aux=True,
                            )
                        )
                    self.action.args.ccddata.data = corrected_cube
                else:
                    self.action.args.ccddata.data = sci_im_full_original

                #print('NaNs in the linearity corrected data=',np.isnan(corrected_cube).sum())
                self.action.args.ccddata.header['HISTORY'] = (f'Linearity correction applied using {os.path.basename(lin_coeff)}')
                self.logger.info("+++++++++++ linearity correction finished +++++++++++")
                

            log_string = LinearityCorrect.__module__
            self.action.args.ccddata.header['HISTORY'] = log_string
            self.logger.info(log_string)

        else:
            self.logger.info("+++++++++++ No science files detected to process +++++++++++")
        return self.action.args
    # END: class LinearityCorrect()
