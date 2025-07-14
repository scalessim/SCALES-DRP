from keckdrpframework.primitives.base_primitive import BasePrimitive
import numpy as np
import math
import time


class SubtractOverscan(BasePrimitive):
    """
    Determines overscan offset and subtracts it from the image.

    Uses the BIASSEC header keyword to determine where to calculate the overscan
    offset.  Subtracts the overscan offset and records the value in the header.
    In addition, performs a polynomial fit and uses the residuals to determine
    the read noise in the overscan.  Records the overscan readnoise in the
    header as well.

    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger

    def _perform(self):
        # image sections for each amp
        bsec, dsec, tsec, direc, amps, aoff = self.action.args.map_ccd #overscan, science,
        namps = len(amps)
        # polynomial fit order
        if namps == 4:
            porder = 2
        else:
            porder = 7
        minoscanpix = self.config.instrument.minoscanpix # The minimum number of overscan pixels required to even attempt a fit.
        oscanbuf = self.config.instrument.oscanbuf #A buffer to exclude pixels at the very edge of the overscan region
        frameno = self.action.args.ccddata.header['FRAMENO']
        # header keyword to update
        key = 'OSCANSUB' #FITS header keyword to document that this step has been performed.
        keycom = 'Overscan subtracted?'
        # is it performed?
        performed = True
        # loop over amps
        plts = []   # plots for each amp

        for ia in amps:
            
            iac = ia - aoff #convert amp number to python index
            # get gain
            gain = self.action.args.ccddata.header['GAIN%d' % ia] #gain for the current amplifier
            #  A check to ensure the overscan region is large enough to be processed.
            if (bsec[iac][3] - bsec[iac][2]) > minoscanpix: 
                # pull out an overscan vector
                x0 = bsec[iac][2] + oscanbuf
                x1 = bsec[iac][3] - oscanbuf
                y0 = bsec[iac][0]
                y1 = bsec[iac][1] + 1
                # get overscan value to subtract
                osval = int(np.nanmedian(
                    self.action.args.ccddata.data[y0:y1, x0:x1]))
                # vector to fit for determining read noise along each row
                osvec = np.nanmedian(
                    self.action.args.ccddata.data[y0:y1, x0:x1], axis=1) #calculates the median along the rows
                nsam = x1 - x0
                xx = np.arange(len(osvec), dtype=np.float)
                # fit it, avoiding first 50 px to avoid electronic settling artifacts
                if direc[iac][0]:
                    # forward read skips first 50 px
                    oscoef = np.polyfit(xx[50:], osvec[50:], porder)
                    # generate fitted overscan vector for full range
                    osfit = np.polyval(oscoef, xx)
                    # calculate residuals
                    resid = (osvec[50:] - osfit[50:]) * math.sqrt(nsam) * \
                        gain / 1.414
                else:
                    # reverse read skips last 50 px
                    oscoef = np.polyfit(xx[:-50], osvec[:-50], porder)
                    # generate fitted overscan vector for full range
                    osfit = np.polyval(oscoef, xx) # overscan structure across its entire length.
                    # calculate residuals
                    resid = (osvec[:-50] - osfit[:-50]) * math.sqrt(nsam) * \
                        gain / 1.414


                sdrs = float("%.3f" % np.std(resid))
                self.logger.info("Img # %05d, Amp %d [%d:%d, %d:%d]" % (frameno,
                                                                        ia,
                                                                        x0, x1,
                                                                        y0, y1))
                self.logger.info("Amp%d oscan counts (DN): %d" %
                                 (ia, osval))
                self.logger.info("Amp%d Read noise from oscan in e-: %.3f" %
                                 (ia, sdrs))
                self.action.args.ccddata.header['OSCNRN%d' % ia] = \
                    (sdrs, "amp%d RN in e- from oscan" % ia)
                self.action.args.ccddata.header['OSCNVAL%d' % ia] = \
                    (osval, "amp%d oscan counts (DN)" % ia)


                dy0, dy1 = dsec[iac][0], dsec[iac][1] + 1
                dx0, dx1 = dsec[iac][2], dsec[iac][3] + 1

                science_data_slice = self.action.args.ccddata.data[dy0:dy1, dx0:dx1]
                osfit_column = osfit[:, np.newaxis]
                science_data_slice -= osfit_column


        self.action.args.ccddata.header[key] = (performed, keycom)

        log_string = SubtractOverscan.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)

        return self.action.args
    # END: class SubtractOverscan()
