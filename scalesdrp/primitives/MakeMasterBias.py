from keckdrpframework.models.arguments import Arguments
from keckdrpframework.primitives.base_img import BaseImg
from scalesdrp.primitives.scales_file_primitives import scales_fits_reader, \
    scales_fits_writer, parse_imsec, strip_fname
from scalesdrp.core.scales_plotting import save_plot

from bokeh.plotting import figure
import ccdproc
import numpy as np
from scipy.stats import sigmaclip
from astropy.stats import mad_std
import time
import os


class MakeMasterBias(BaseImg):
    """
    Stack bias frames into a master bias frame.

    Generate a master bias image from overscan-subtracted and trimmed bias
    frames (\*_intb.fits) based on the instrument config parameter
    bias_min_nframes, which defaults to 7.  The combine method for biases is
    'average' and so cosmic rays may be present, especially in RED channel data.
    A high sigma clipping of 2.0 is used to help with the CRs.

    Uses the ccdproc.combine routine to perform the stacking.

    Writes out a \*_mbias.fits file and records a master bias frame in the proc
    table.


    """

    def __init__(self, action, context):
        BaseImg.__init__(self, action, context)
        self.logger = context.pipeline_logger

    def _pre_condition(self):
        """
        Checks if we can build a stacked frame based on the processing table
        """
        # Get bias count
        self.logger.info("Checking precondition for MakeMasterBias")
        self.combine_list = self.context.proctab.search_proctab(
            frame=self.action.args.ccddata, target_type='BIAS',
            target_group=self.action.args.groupid)
        self.logger.info(f"pre condition got {len(self.combine_list)},"
                         f" expecting {self.action.args.min_files}")
        # Did we meet our pre-condition?
        if len(self.combine_list) >= self.action.args.min_files:
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation
        """
        method = 'average'
        sig_up = 2.0        # default upper sigma rejection limit
        suffix = self.action.args.new_type.lower()

        combine_list = list(self.combine_list['filename'])
        # get master bias output name
        # mbname = combine_list[-1].split('.fits')[0] + '_' + suffix + '.fits'
        mbname = strip_fname(combine_list[0]) + '_' + suffix + '.fits'
        # mbname = master_bias_name(self.action.args.ccddata)
        bsec, dsec, tsec, direc, amps, aoff = self.action.args.map_ccd

        # loop over amps
        stack = []
        stackf = []
        for bias in combine_list:
            inbias = bias.split('.fits')[0] + '_intb.fits'
            stackf.append(inbias)
            # using [0] drops the table and leaves just the image
            stack.append(scales_fits_reader(
                os.path.join(self.context.config.instrument.cwd, 'redux',
                             inbias))[0])

        stacked = ccdproc.combine(stack, method=method, sigma_clip=True,
                                  sigma_clip_low_thresh=None,
                                  sigma_clip_high_thresh=sig_up,
                                  sigma_clip_func=np.ma.median,
                                  sigma_clip_dev_func=mad_std)
        stacked.header['IMTYPE'] = self.action.args.new_type
        stacked.header['NSTACK'] = (len(combine_list),
                                    'number of images stacked')
        stacked.header['STCKMETH'] = (method, 'method used for stacking')
        stacked.header['STCKSIGU'] = (sig_up,
                                      'Upper sigma rejection for stacking')
        for ii, fname in enumerate(stackf):
            fname_base = os.path.basename(fname)
            stacked.header['STACKF%d' % (ii + 1)] = (fname_base,
                                                     "stack input file")

        # for readnoise stats use 2nd and 3rd bias
        diff = stack[1].data.astype(np.float32) - \
            stack[2].data.astype(np.float32)
        namps = stack[1].header['NVIDINP']
        if len(amps) != namps:
            self.logger.warning("Amp count disagreement!")
        for ia in amps:
            # get gain
            gain = stacked.header['GAIN%d' % ia]
            # get amp section
            sec, rfor = parse_imsec(stacked.header['ATSEC%d' % ia])
            noise = diff[sec[0]:(sec[1]+1), sec[2]:(sec[3]+1)]
            noise = np.reshape(noise, noise.shape[0]*noise.shape[1]) * \
                gain / 1.414
            # get stats on noise
            c, low, upp = sigmaclip(noise, low=3.5, high=3.5)
            bias_rn = c.std()
            self.logger.info("Amp%d read noise from bias in e-: %.3f" %
                             (ia, bias_rn))
            stacked.header['BIASRN%d' % ia] = \
                (float("%.3f" % bias_rn), "RN in e- from bias")

        log_string = MakeMasterBias.__module__
        stacked.header['HISTORY'] = log_string
        self.logger.info(log_string)

        scales_fits_writer(stacked, output_file=mbname,
                         output_dir=self.config.instrument.output_directory)
        self.context.proctab.update_proctab(frame=stacked, suffix=suffix,
                                            newtype=self.action.args.new_type,
                                            filename=stacked.header['OFNAME'])
        self.context.proctab.write_proctab(tfil=self.config.instrument.procfile)
        return Arguments(name=mbname)
    # END: class ProcessBias()
