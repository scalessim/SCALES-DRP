from keckdrpframework.models.arguments import Arguments
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Table
# from astropy import units as u
import numpy as np
from datetime import datetime

from keckdrpframework.primitives.base_primitive import BasePrimitive
import os
import logging
import pkg_resources
import subprocess
from pathlib import Path

logger = logging.getLogger('SCALES')


def parse_imsec(section=None):
    """
    Parse image section FITS header keyword into useful tuples.

    Take into account one-biased IRAF-style image section keywords and the
    possibility that a third element (strid) may be present and generate the
    x and y limits as well as the stride for each axis that are useful for
    python image slices.

    Args:
        section (str): square-bracket enclosed string with colon range
        specifiers and comma delimiters.

    :returns:
        - list: (int) y0, y1, x0, x1 - zero-biased (python) slice limits.
        - list: (int) y-stride, x-stride - strides for each axis.

    """
    xsec = section[1:-1].split(',')[0]
    ysec = section[1:-1].split(',')[1]
    xparts = xsec.split(':')
    yparts = ysec.split(':')
    p1 = int(xparts[0])
    p2 = int(xparts[1])
    p3 = int(yparts[0])
    p4 = int(yparts[1])
    # check for scale factor
    if len(xparts) == 3:
        xstride = int(xparts[2])
    else:
        xstride = 1
    if len(yparts) == 3:
        ystride = int(yparts[2])
    else:
        ystride = 1
    # is x axis in descending order?
    if p1 > p2:
        x0 = p2 - 1
        x1 = p1 - 1
        xstride = -abs(xstride)
    # x axis in ascending order
    else:
        x0 = p1 - 1
        x1 = p2 - 1
    # is y axis in descending order
    if p3 > p4:
        y0 = p4 - 1
        y1 = p3 - 1
        ystride = -abs(ystride)
    # y axis in ascending order
    else:
        y0 = p3 - 1
        y1 = p4 - 1
    # ensure no negative indices
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    # package output with python axis ordering
    sec = (y0, y1, x0, x1)
    stride = (ystride, xstride)

    return sec, stride

class KCCDData(CCDData):
    """
    A container for SCALES images based on the CCDData object that adds
    the `noskysub` frame to allow parallel processing of sky-subtracted
    and un-sky-subtracted SCALES images.

    Attributes:
        noskysub (`numpy.ndarray` or None): Optional un-sky-subtracted frame
            that is created at the sky subtraction stage and processed in
            parallel with the primary frame.
            Default is ``None``.

    """

    def __init__(self, *args, **kwd):
        super().__init__(*args, **kwd)
        self._noskysub = None

    @property
    def noskysub(self):
        return self._noskysub

    @noskysub.setter
    def noskysub(self, value):
        self._noskysub = value

class AddToCalDataFrame(BasePrimitive):
    """

    """

    def __init__(self, action, context):
            BasePrimitive.__init__(self, action, context)
            self.logger = context.pipeline_logger


    def _perform(self):
        if self.context.data_set is None:
            self.context.data_set = DataSet(None, self.logger, self.config,
            self.context.event_queue)
        
        self.context.data_set.append_item(self.action.args.name)
        print(self.context.data_set.data_table)
        self.logger.info(
            "------------------- Ingesting file %s -------------------" %
            self.action.args.name)

        return self.action.args

class ingest_file(BasePrimitive):
    """
    File ingestion class for SCALES images.

    Args:
        action (str): Pipeline action
        context: Pipeline context which includes arguments for the action.

    Attributes:
        logger: SCALES pipeline logger

    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger


    def get_keyword(self, keyword):
        """
        Get a keyword from ingested headers.

        Args:
            keyword (str): Keyword to fetch from header

        Returns:
            Keyword value if present, otherwise ``None``.
        """
        try:
            keyval = self.ccddata.header[keyword]
        except KeyError:
            keyval = None
        # return self.context.data_set.get_info_column(self.name, keyword)
        return keyval

    def scmode(self):
        """
        Get SCALES Mode

        """

        scmode = self.get_keyword('SCMODE').upper()
        if 'LowRes-SED' in scmode:
            return 0


    def imtype(self):
        """Return the value of the `IMTYPE` FITS header keyword."""
        return self.get_keyword('IMTYPE')


    def _perform(self):
        # if self.context.data_set is None:
        #    self.context.data_set = DataSet(None, self.logger, self.config,
        #    self.context.event_queue)
        # self.context.data_set.append_item(self.action.args.name)
        self.logger.info(
            "------------------- Ingesting file %s -------------------" %
            self.action.args.name)

        self.action.event._recurrent = True

        self.name = self.action.args.name
        out_args = Arguments()

        ccddata, table = scales_fits_reader(self.name)

        # save the ccd data into an object
        # that can be shared across the functions
        self.ccddata = ccddata

        out_args.ccddata = ccddata
        out_args.table = table

        imtype = self.get_keyword("IMTYPE")
        scmode = self.get_keyword("SCMODE")
        
        if imtype is None:
            fname = os.path.basename(self.action.args.name)
            self.logger.warn(f"Unknown IMTYPE {fname}")
            self.action.event._reccurent = False

        if self.check_if_file_can_be_processed(imtype) is False:

            if self.config.instrument.continuous or \
                    self.config.instrument.wait_for_event:
                self.logger.warn("Input frame cannot be reduced. Rescheduling")
                self.action.new_event = None
                return None
            else:
                self.logger.warn("Input frame cannot be reduced. Exiting")
                self.action.new_event = None
                self.action.event._recurrent = False
                return None
        else:
            self.action.event._recurrent = False

        out_args.name = self.action.args.name
        out_args.imtype = imtype
        out_args.scmode = scmode
        return out_args


    def check_if_file_can_be_processed(self, imtype):
        """
        For a given image type, ensure that processing can proceed.

        Based on `IMTYPE` keyword, makes a call to proctab to see if
        pre-requisite images are present.

        Returns:
            (bool): ``True`` if processing can proceed, ``False`` if not.

        """
        if imtype == 'CALUNIT':
            return True 
        return True


def scales_fits_reader(file):
    """A reader for KCCDData objects.

    Currently, this is a separate function, but should probably be
    registered as a reader similar to fits_ccddata_reader.

    Args:
        file (str): The filename (or pathlib.Path) of the FITS file to open.

    Raises:
        FileNotFoundError: if file not found or
        OSError: if file not accessible

    Returns:
        (KCCDData, FITS table): All relevant frames in a single KCCDData object
        and a FITS table of exposure events, if present otherwise ``None``.

    """
    try:
        hdul = fits.open(file)
    except (FileNotFoundError, OSError) as e:
        print(e)
        raise e
    read_imgs = 0
    read_tabs = 0
    # primary image
    ccddata = KCCDData(hdul['PRIMARY'].data, meta=hdul['PRIMARY'].header,
                       unit='adu')
    

    # check for other legal components
    if 'UNCERT' in hdul:
        ccddata.uncertainty = hdul['UNCERT'].data
        read_imgs += 1
    if 'FLAGS' in hdul:
        ccddata.flags = hdul['FLAGS'].data
        read_imgs += 1
    if 'MASK' in hdul:
        ccddata.mask = hdul['MASK'].data
        read_imgs += 1
    if 'NOSKYSUB' in hdul:
        ccddata.noskysub = hdul['NOSKYSUB'].data
        read_imgs += 1
    if 'Exposure Events' in hdul:
        table = hdul['Exposure Events']
        read_tabs += 1
    else:
        table = None
    
    # prepare for floating point
    ccddata.data = ccddata.data.astype(np.float64)
    
    ## Fix red headers
    ##fix_header(ccddata)
    
    logger.info("<<< read %d imgs and %d tables out of %d hdus in %s" %
                (read_imgs, read_tabs, len(hdul), file))
    return ccddata, table


def fix_header(ccddata):
    """
    Fix header keywords for DRP use.

    """
    # are we lowres?
    if 'LOWRES' in ccddata.header['MODE'].upper():
        gainmul = ccddata.header['GAINMUL']
        namps = ccddata.header['NVIDINP']
        for ia in range(namps):
            ampid = ccddata.header['AMPID%d' % (ia+1)]
            gain = blue_amp_gain[gainmul][ampid]
            ccddata.header['GAIN%d' % (ia+1)] = gain
    # are we medres?
    elif 'MEDRES' in ccddata.header['MODE'].upper():
        # Fix red headers during Caltech AIT
        if 'TELESCOP' not in ccddata.header:
            # Add DCS keywords
            ccddata.header['TARGNAME'] = ccddata.header['OBJECT']
            dateend = ccddata.header['DATE-END']
            de = datetime.fromisoformat(dateend)
            day_frac = de.hour / 24. + de.minute / 1440. + de.second / 86400.
            jd = datetime.date(de).toordinal() + day_frac + 1721424.5
            mjd = jd - 2400000.5
            ccddata.header['MJD'] = mjd
        # Add NVIDINP
        ccddata.header['NVIDINP'] = ccddata.header['TAPLINES']
        # Add GAINMUL
        ccddata.header['GAINMUL'] = 1
        # Add CCDMODE
        if 'CDSSPEED' in ccddata.header:
            ccddata.header['CCDMODE'] = ccddata.header['CDSSPEED']
        else:
            ccddata.header['CCDMODE'] = 0
    else:
        print("ERROR -- illegal CAMERA keyword: %s" % ccddata.header['CAMERA'])
