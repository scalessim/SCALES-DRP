from keckdrpframework.models.arguments import Arguments
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Table
# from astropy import units as u
import numpy as np
from datetime import datetime
from astropy.nddata import StdDevUncertainty
from keckdrpframework.primitives.base_primitive import BasePrimitive
import os
import logging
import subprocess
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

logger = logging.getLogger('SCALES')

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

    def camera(self):
        """
        Get camera ID number.

        Uses `CAMERA` FITS header keyword.

        Returns:
            0 for Blue channel, 1 for Red, and -1 for Unknown.
        """
        camera = self.get_keyword('CAMERA').upper()
        if 'Im' in camera:
            return 0
        elif 'IFS' in camera:
            return 1
        else:
            return -1

    def namps(self):
        """
        Return value of `NVIDINP` FITS header keyword.
        """
        return self.get_keyword('NVIDINP')

    def nasmask(self):
        """
        Query if mask was inserted for image.

        Calls camera() to determine channel, and then uses `BNASNAM` for Blue
        channel or `RNASNAM` for Red channel.

        Raises:
            ValueError: if channel is Unknown.

        Returns:
            (bool): ``True`` if 'Mask' in, ``False`` if not.

        """
        if self.camera() == 0:  # lowres
            if 'Mask' in self.get_keyword('BNASNAM'):
                return True
            else:
                return False
        elif self.camera() == 1:  # medres
            if 'Mask' in self.get_keyword('RNASNAM'):
                return True
            else:
                return False
        else:
            raise ValueError("unable to determine mask: CAMERA undefined")

    def numopen(self):
        """Returns value of `NUMOPEN` FITS header keyword."""
        return self.get_keyword('NUMOPEN')

    def shufrows(self):
        """Returns value of `SHUFROWS` FITS header keyword."""
        return self.get_keyword('SHUFROWS')

    def ampmode(self):
        """Returns value of `AMPMODE` FITS header keyword."""
        return self.get_keyword('AMPMODE')

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

        # Are we already in proctab?
        out_args.in_proctab = self.context.proctab.in_proctab(frame=ccddata)
        if out_args.in_proctab:
            out_args.last_suffix = self.context.proctab.last_suffix(
                frame=ccddata)
            if len(out_args.last_suffix) > 0:
                self.logger.info("Last suffix is %s" % out_args.last_suffix)
                out_dir = self.config.instrument.output_directory
                last_file = os.path.join(out_dir,
                                         self.name.split('.fits')[0] + '_' +
                                         out_args.last_suffix + '.fits')
                if 'opt_cube' not in out_args.last_suffix:
                    self.logger.info("Ingest stub for %s" % last_file)
                    # ccddata, table = scales_fits_reader(last_file)
                else:
                    self.logger.info("Processing completed in %s" % last_file)
        else:
            out_args.last_suffix = ""

        # save the ccd data into an object
        # that can be shared across the functions
        self.ccddata = ccddata

        out_args.ccddata = ccddata
        out_args.table = table

        imtype = self.get_keyword("IMTYPE")
        groupid = self.get_keyword("GROUPID")

        if groupid is None:
            groupid = "NONE"
        else:
            if len(groupid) <= 0:
                groupid = "NONE"
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
        out_args.groupid = groupid
        # CAMERA
        out_args.camera = self.camera()

        out_args.numopen = self.numopen()
        # AMPMODE
        out_args.ampmode = self.ampmode()
        return out_args

    def apply(self):
        """
        Apply method for class.

        Checks _pre_condition().  If ``True``, then call _perform() and
        collect output.  Then check _post_condition().

        Returns:
            output from _perform, or ``None`` if there was an exception.

        """
        if self._pre_condition():
            try:
                output = self._perform()
            except ValueError as e:
                self.logger.warn("UNABLE TO INGEST THE FILE")
                self.logger.warn("Reason: %s" % e)
                return None
            if self._post_condition():
                self.output = output
        return self.output

    def check_if_file_can_be_processed(self, imtype):
        """
        For a given image type, ensure that processing can proceed.

        Based on `IMTYPE` keyword, makes a call to proctab to see if
        pre-requisite images are present.

        Returns:
            (bool): ``True`` if processing can proceed, ``False`` if not.

        """

        if imtype == 'ARCLAMP':
            # continuum bars
            contbars_frames = self.context.proctab.search_proctab(
                frame=self.ccddata, target_type='MCBARS', nearest=True)
            if len(contbars_frames) > 0:
                return True
            else:
                self.logger.warn("Cannot reduce ARCLAMP frame. "
                                 "Missing master continuum bars. ")
                return False

        if imtype in ['FLATLAMP', 'TWIFLAT', 'DOMEFLAT']:
            # bias frames
            bias_frames = self.context.proctab.search_proctab(
                frame=self.ccddata, target_type='MBIAS', nearest=True)
            arc_frames = self.context.proctab.search_proctab(
                frame=self.ccddata, target_type='MARC', nearest=True)
            if len(bias_frames) > 0 and len(arc_frames) > 0:
                return True
            else:
                self.logger.warn(f"Cannot reduce {imtype} frame.")
                if len(bias_frames) <= 0:
                    self.logger.warn(f"Missing master bias.")
                if len(arc_frames) <= 0:
                    self.logger.warn(f"Missing master arc")
                return False

        return True


def scales_fits_reader(file):
    """

    A reader for KCCDData objects that handles both 2D images and
    3D data cubes.

    It reads the primary HDU and checks its dimensionality. It then looks for
    standard extensions (UNCERT, FLAGS, MASK) and validates that their
    dimensions are consistent with the primary data.

    Args:
        file (str): The filename (pathlib.Path) of the FITS file to open.

    Raises:
        FileNotFoundError: if file not found or OSError if not accessible.
        ValueError: if an extension has a shape inconsistent with the primary HDU.

    Returns:
        (KCCDData, FITS table): All relevant frames in a single KCCDData object
        and a FITS table of exposure events, if present otherwise ``None``.

    """
    try:
        hdul = fits.open(file)
    except (FileNotFoundError, OSError) as e:
        print(e)
        raise e
    try:
        primary_hdu = hdul[0]
    except IndexError:
        hdul.close()
        raise ValueError(f"FITS file '{file}' appears to be empty or corrupted.")

    if primary_hdu is None or primary_hdu.data is None:
        hdul.close()
        raise ValueError(f"FITS file '{file}' does not have a valid PRIMARY HDU with data.")

    #primary_data = primary_hdu.data
    #primary_shape = primary_data.shape
    #primary_ndim = primary_data.ndim
    
    #print(f"  Primary data found with {primary_ndim} dimensions and shape {primary_shape}.")
    cube_hdu = None
    for i, hdu in enumerate(hdul[:2]):
        if hasattr(hdu, "data") and hdu.data is not None:
            if hdu.data.ndim == 3:
                cube_hdu = hdu
                print(f"[INFO] Found 3D data in HDU {i} with shape {hdu.data.shape}")
                break
    if cube_hdu is None:
        hdul.close()
        raise ValueError(f"FITS file '{file}' contains no valid 3D data cube in HDU 0 or 1.")

    #ccddata = KCCDData(np.array(primary_data), meta=primary_hdu.header, unit='adu')
    ccddata = KCCDData(np.array(cube_hdu.data, dtype=np.float64),meta=primary_hdu.header, unit='adu')

    read_imgs = 1
    read_tabs = 0

    # check for other legal components
    if 'UNCERT' in hdul:
        uncert_data = hdul['UNCERT'].data
        ccddata.uncertainty = StdDevUncertainty(uncert_data)
        read_imgs += 1

    if 'FLAGS' in hdul:
        ccddata.flags = hdul['FLAGS'].data
        read_imgs += 1
    if 'MASK' in hdul:
        ccddata.mask = hdul['MASK'].data
        read_imgs += 1
    
    #if 'NOSKYSUB' in hdul:
    #    ccddata.noskysub = hdul['NOSKYSUB'].data
    #    read_imgs += 1
    if 'Exposure Events' in hdul:
        table = hdul['Exposure Events']
        read_tabs += 1
    else:
        table = None
    # prepare for floating point
    ccddata.data = ccddata.data.astype(np.float64)
    return ccddata, table


def read_table(input_dir=None, file_name=None):
    """
    Read FITS table

    Uses astropy.Table module to read in FITS table.

    Raises:
        FileNotFoundError: if file not found.

    Returns:
        (FITS Table): table read in or ``None`` if unsuccessful.

    """
    # Set up return table
    input_file = os.path.join(input_dir, file_name)
    logger.info("Trying to read table: %s" % input_file)
    try:
        retab = Table.read(input_file, format='fits')
    except FileNotFoundError:
        logger.warning("No table to read")
        retab = None
    return retab


def scales_fits_writer(ccddata, table=None, output_file=None, output_dir=None,
                     suffix=None):
    """
    A writer for KCCDData or CCDData objects.

    Updates history in FITS header with pipeline version and git repo version
    and date.

    Converts float64 data to float32.

    Uses object to_hdu() method to generate hdu list and then checks if various
    extra frames are present (flags, noskysub) and adds them to the hdu list
    prior to writing out with hdu list writeto() method.

    Note:
        Currently fits tables are not written out.

    Args:
        ccddata (KCCDData or CCDData): object to write out.
        table (FITS Table): currently not used.
        output_file (str): base filename to write to.
        output_dir (str): directory into which to write.
        suffix (str): a suffix to append to output_file string.


    """
    # Determine if the version info is already in the header
    contains_version = False
    for h in ccddata.header["HISTORY"]:
        if "scalesdrp version" in h:
            contains_version = True

    if not contains_version:
        # Add setup.py version number to header
        try:
            ver = version("scalesdrp")
        except PackageNotFoundError:
            ver = "unknown"
        #version = pkg_resources.get_distribution('scalesdrp').version
        ccddata.header.add_history(f"scalesdrp version={ver}")

        # Get string filepath to .git dir, relative to this primitive
        primitive_loc = os.path.dirname(os.path.abspath(__file__))
        git_loc = primitive_loc[:-18] + ".git"

        # Attempt to gather git version information
        git1 = subprocess.run(["git", "--git-dir", git_loc, "describe",
                               "--tags", "--long"], capture_output=True)
        git2 = subprocess.run(["git", "--git-dir", git_loc, "log", "-1",
                               "--format=%cd"], capture_output=True)
        
        # If all went well, save to the header
        if not bool(git1.stderr) and not bool(git2.stderr):
            git_v = git1.stdout.decode('utf-8')[:-1]
            git_d = git2.stdout.decode('utf-8')[:-1]
            ccddata.header.add_history(f"git version={git_v}")
            ccddata.header.add_history(f"git date={git_d}")
        else:
            logger.debug("Package not installed from a git repo, skipping")

    # If there is a data array, and the type of that array is a 64-bit float,
    # force it to 32 bits.
    if ccddata.data is not None and ccddata.data.dtype == np.float64:
        ccddata.data = ccddata.data.astype(np.float32)
    # If there is an uncertainty array, and the values within
    # (the .array property), make it 32 bits.
    if ccddata.uncertainty is not None and ccddata.uncertainty.array.dtype == np.float64:
        ccddata.uncertainty.array = ccddata.uncertainty.array.astype(np.float32)
    
    out_file = os.path.join(output_dir, os.path.basename(output_file))
    if suffix is not None:
        (main_name, extension) = os.path.splitext(out_file)
        out_file = main_name + "_" + suffix + extension
    hdus_to_save = ccddata.to_hdu()
    dq = getattr(ccddata, "dq", None)
    if dq is not None:
        dq = np.asarray(dq)
        if not np.issubdtype(dq.dtype, np.integer):
            dq = dq.astype(np.uint32)
        hdu_dq = fits.ImageHDU(
            dq,
            name="DQ",
            do_not_scale_image_data=True)
        hdu_dq.header["EXTDESC"] = "Data quality bit mask"
        hdu_dq.header["DQ0"] = "bit 0: no linearity correction"
        hdu_dq.header["DQ1"] = "bit 1: saturated read"
        hdu_dq.header["DQ2"] = "bit 2: bad linearity value"
        hdu_dq.header["DQ3"] = "bit 3: non-monotonic correction"
        hdu_dq.header["DQ4"] = "bit 4: linearity correction applied"
        hdus_to_save.append(hdu_dq)

    logger.info(">>> Saving %d hdus to %s" % (len(hdus_to_save), out_file))
    hdus_to_save.writeto(out_file, overwrite=True)



def fits_writer_calib(
    data,                  # 2D or 3D array
    header,                # FITS header (astropy.io.fits.Header)
    output_dir,            # base output directory
    input_filename,        # original filename (used for naming)
    suffix,                # string appended to filename
    overwrite=True,
    uncert=None,
    dq=None,
):
    """
    Write data (and optional uncertainty) to a FITS file inside redux/.

    Parameters
    ----------
    data : ndarray
        calibration array.
    header : fits.Header
        FITS header for the primary HDU.
    output_dir : str
        Output directory (redux/ will be created inside this).
    input_filename : str
        Original filename, used to derive output filename.
    suffix : str
        Suffix appended before extension (e.g. '_mdark', '_mflat').
    overwrite : bool, optional
        Whether to overwrite existing file (default: True).
    uncert : ndarray, optional
        Uncertainty array, same shape as data. Saved as 'UNCERT' extension.

    Returns
    -------
    output_path : str
        Full path to the written FITS file.
    """
    # --- ensure directories exist ---
    os.makedirs(os.path.join(output_dir, "redux"), exist_ok=True)

    # --- construct filename ---
    base_name = os.path.basename(input_filename)
    file_root, file_ext = os.path.splitext(base_name)
    output_filename = f"{file_root}{suffix}{file_ext}"
    output_path = os.path.join(output_dir, "redux", output_filename)

    # --- build HDUList ---
    hdus = [fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=header)]

    if dq is not None:
        dq = np.asarray(dq)
        if not np.issubdtype(dq.dtype, np.integer):
            dq = dq.astype(np.uint32)

        dq_hdu = fits.ImageHDU(
            data=dq,
            name="DQ",
            do_not_scale_image_data=True)
        dq_hdu.header["EXTDESC"] = "Data quality bit mask"
        dq_hdu.header["DQ0"] = "bit 0: do not use / input BPM"
        dq_hdu.header["DQ1"] = "bit 1: no linearity correction"
        dq_hdu.header["DQ2"] = "bit 2: saturated read"
        dq_hdu.header["DQ3"] = "bit 3: bad linearity correction value"
        dq_hdu.header["DQ4"] = "bit 4: non-monotonic linearity correction"
        dq_hdu.header["DQ5"] = "bit 5: linearity correction applied"
        hdus.append(dq_hdu)

    # add uncertainty extension if provided
    if uncert is not None:
        uncert = np.asarray(uncert)
        if uncert.shape != data.shape:
            warnings.warn(f"Uncertainty shape {uncert.shape} does not match data {data.shape}; skipping UNCERT extension.")
        else:
            hdu_uncert = fits.ImageHDU(data=np.asarray(uncert, dtype=np.float32), name="UNCERT")
            hdus.append(hdu_uncert)
    hdul = fits.HDUList(hdus)
    # --- write to disk ---
    hdul.writeto(output_path, overwrite=overwrite)
    # --- optional: return path for downstream pipeline ---
    return output_path
