from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import parse_imsec, \
    scales_fits_writer


class CorrectGain(BasePrimitive):
    """
    Convert raw data numbers to electrons.

    Uses the ATSECn FITS header keywords to divide image into amp regions and
    then corrects each region with the corresponding GAINn keyword.  Updates the
    following FITS header keywords:

        * GAINCOR: sets to ``True`` if operation performed.
        * BUNIT: sets to `electron`.
        * HISTORY: records the operation.

    Uses the following configuration parameter:

        * saveintims: if set to ``True`` write out a \*_gain.fits file with gain corrected.  Default is ``False``.

    Updates image in returned arguments.

    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger

    def _perform(self):
        # Header keyword to update
        key = 'GAINCOR'
        keycom = 'Gain corrected?'
        # print(self.action.args.ccddata.header)
        number_of_amplifiers = 2.0
        bsec, dsec, tsec, direc, amps, aoff = self.action.args.map_ccd


        self.action.args.ccddata.header[key] = (True, keycom)
        self.action.args.ccddata.header['BUNIT'] = ('electron', 'Pixel units')
        self.action.args.ccddata.unit = 'electron'

        log_string = CorrectGain.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)

        if self.config.instrument.saveintims:
            scales_fits_writer(self.action.args.ccddata,
                             table=self.action.args.table,
                             output_file=self.action.args.name,
                             output_dir=self.config.instrument.output_directory,
                             suffix="gain")
        return self.action.args
    # END: class CorrectGain()
