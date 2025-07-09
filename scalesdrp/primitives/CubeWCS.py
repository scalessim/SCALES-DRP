from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import warnings
from astropy.coordinates import Angle
import astropy.units as u

class CubeWCS(BasePrimitive):
    """

    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger
        

    def _perform(self):

        self.logger.info("Adding WCS info to the header")
        tab = self.context.proctab.search_proctab(
            frame=self.action.args.ccddata, target_type='OBJECT',
            nearest=True)
        self.logger.info("%d object frames found" % len(tab))

        is_obj = ('OBJECT' in self.action.args.ccddata.header['IMTYPE'])
        
        
    


        data_cube = self.action.args.ccddata.data

    




        self.action.args.ccddata.data = d

        log_string = CubeWCS.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)


        if is_obj:	    
            scales_fits_writer(self.action.args.ccddata, table=self.action.args.table,output_file=self.action.args.name,output_dir=self.config.instrument.output_directory,suffix="wcs")

        self.context.proctab.update_proctab(frame=self.action.args.ccddata, suffix="wcs", newtype='OBJECT',
                filename=self.action.args.ccddata.header['OFNAME'])
        self.context.proctab.write_proctab(
                tfil=self.config.instrument.procfile)

        return self.action.args
    # END: class CubeWCS()
