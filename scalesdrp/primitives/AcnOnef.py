from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import scalesdrp.primitives.scales_basic as scbasic
import scalesdrp.primitives.robust as robust
import numpy as np
import time
import os
import scalesdrp.primitives.reference as reference #1/f and reference pixel correction
from astropy.nddata import StdDevUncertainty
from scalesdrp.core.scales_proctab import Proctab
from scalesdrp.core.scales_pkg_resources import get_resource_path
import logging
log = logging.getLogger("SCALES")


class AcnOnef(BasePrimitive):

    """
    This function correct a raw read for channel bias, alternative column noise, and 1/f noise.
    The core of this function is `import scalesdrp.primitives.reference as reference`. 
    No external calibration files are involved.
        Args:
            data_image: The (N,H,W) input ramp cube.

        Returns:
            data_image: The (N,H,W) detector level corrected ramp cube.
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
            print('NaNs in the reference pixel corrected data=',np.isnan(sci_im_full_original).sum())
            self.action.args.ccddata.header = hdr
            self.action.args.ccddata.data = sci_im_full_original

            log_string = AcnOnef.__module__
            self.action.args.ccddata.header['HISTORY'] = log_string
            self.logger.info(log_string)

            self.logger.info("+++++++++++ detector level corrections are completed +++++++++++")
        else:
            self.logger.info("+++++++++++ No science files detected to process +++++++++++")
        return self.action.args
    # END: class AcnOnef()
