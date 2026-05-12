from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import scalesdrp.primitives.reference as reference
import scalesdrp.primitives.robust as robust

import pandas as pd
import numpy as np
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from importlib.resources import files
from pathlib import Path
import os
from scipy.optimize import minimize
from scipy import sparse
from scipy.sparse import load_npz
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time
from scipy.optimize import leastsq
from scipy.signal import savgol_filter
import scalesdrp.primitives.bpm_correction as bpm #bpm correction
from scalesdrp.core.scales_proctab import Proctab
import logging
from astropy.wcs import WCS
from astropy.coordinates import Angle
import astropy.units as u
log = logging.getLogger("SCALES")
pt = Proctab(logger=log)

class QuickLook(BasePrimitive):
    """
	Quicklook  extraction : Quick ramp fitted image and optimal extracted cube.
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger

        if not hasattr(self, "proctab") or self.proctab is None:
            self.proctab = Proctab(logger=self.logger if hasattr(self, "logger") else logging.getLogger("SCALES"))
            #log = getattr(self, "logger", None) or logging.getLogger("SCALES")
            #self.proctab = Proctab(logger=log)

    def fits_writer_steps1(self,data,header,output_dir,input_filename,suffix,overwrite=True):
        base_name = os.path.basename(input_filename)
        file_root, file_ext = os.path.splitext(base_name)
        output_filename = f"{file_root}{suffix}{file_ext}"
        redux_output_dir = os.path.join(output_dir, 'ql_redux')
        os.makedirs(redux_output_dir, exist_ok=True)
        output_path = os.path.join(redux_output_dir, output_filename)
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_path, overwrite=overwrite)
        self.logger.info("+++++++++++ FITS file saved +++++++++++")
        return output_path


    def fits_writer_steps(
        self,
        data,
        header,
        output_dir,
        input_filename,
        suffix,
        overwrite=True,
        *,
        frame=None,                 # preferred: Frame object with .header
        proctab=None,               # Proctab instance (e.g. self.proctab)
        proctab_path=None,          # optional
        newtype=None,               # optional override IMTYPE
    ):
        # --------------------------------------------------
        # 1) Build output path
        # --------------------------------------------------
        base_name = os.path.basename(input_filename)
        file_root, file_ext = os.path.splitext(base_name)
        output_filename = f"{file_root}{suffix}{file_ext}"

        redux_output_dir = os.path.join(output_dir, "ql_redux")
        os.makedirs(redux_output_dir, exist_ok=True)
        output_path = os.path.join(redux_output_dir, output_filename)

        # --------------------------------------------------
        # 2) Write FITS file
        # --------------------------------------------------
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_path, overwrite=overwrite)
        self.logger.info("+++++++++++ FITS file saved +++++++++++: %s", output_path)

        # --------------------------------------------------
        # 3) Proctab bookkeeping (first-run safe)
        # --------------------------------------------------
        if proctab is not None:

            # Decide where the proc table lives
            if proctab_path is None:
                proctab_path = os.path.join(redux_output_dir, "ql_scales.proc")

            # ALWAYS: read if exists, else create new
            try:
                proctab.read_proctab(proctab_path)
            except Exception as e:
                self.logger.warning(
                    "Could not read proctab (%s); creating a new one: %s",
                    proctab_path, str(e)
                )
                proctab.new_proctab()

            # Choose a frame/header source for the update
            if frame is not None:
                use_frame = frame
            else:
                # Minimal shim to satisfy update_proctab(frame=...)
                class _FrameShim:
                    def __init__(self, hdr):
                        self.header = hdr
                use_frame = _FrameShim(header)

            # Update + write (never break pipeline)
            try:
                proctab.update_proctab(
                    use_frame,
                    suffix=suffix,
                    filename=output_filename,
                    newtype=newtype,
                )
                proctab.write_proctab(proctab_path)
                self.logger.info("Proctab updated: %s", proctab_path)

            except Exception as e:
                self.logger.warning(
                    "Proctab update failed for %s: %s",
                    output_path, str(e)
                )

        return output_path

    def optimal_extract_with_error(self,
        R_transpose: sp.spmatrix, 
        data_image: np.ndarray, 
        read_noise_variance_vector: np.ndarray, 
        gain: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs classic (Horne 1986) optimal extraction and calculates the
        corresponding 1-sigma error for each flux element.
        Args:
            R_transpose: The (N_fluxes, N_pixels) sparse rectification matrix.
            data_image: The (H,W) input data image.
            read_noise_variance_vector: The (N_pixels,) 1D vector of read noise variance.
            gain: The detector gain.
        Returns:
            A tuple containing:
            - optimized_flux (np.ndarray): The extracted 1D flux array.
            - flux_error (np.ndarray): The corresponding 1D array of 1-sigma errors.
        """
        self.logger.info('Optimal extraction started')
        start_time1 = time.time()
        data_vector_d = data_image.flatten().astype(np.float64)
        photon_noise_variance = data_vector_d.clip(min=0) / gain
        total_variance = read_noise_variance_vector + photon_noise_variance
        total_variance[total_variance <= 0] = 1e-9  
        inverse_variance = 1.0 / total_variance
        weighted_data = data_vector_d #* inverse_variance
        numerator = R_transpose @ weighted_data
        R_transpose_squared = R_transpose.power(2)
        denominator = R_transpose_squared @ inverse_variance
        denominator_safe = np.maximum(denominator, 1e-9)
        optimized_flux = numerator / denominator_safe
        #flux_variance = 1.0 / denominator_safe
        #flux_error = np.sqrt(flux_variance)
        end_time1 = time.time()
        t1 = (end_time1 - start_time1)
        self.logger.info(f"Optimal extraction finished in {t1:.4f} seconds.")
        return optimized_flux

    def iterative_sigma_weighted_ramp_fit1(self,ramp, read_time, gain=3.0, rn=5.0, max_iter=3, tile=(256, 256)):
        n_reads, n_rows, n_cols = ramp.shape
        read_times = np.linspace(0, read_time, n_reads, dtype=np.float32)
        dt = np.mean(np.diff(read_times))
        slope = np.zeros((n_rows, n_cols), dtype=np.float32)
        bias = np.zeros_like(slope)
        Ty, Tx = tile
        for y0 in range(0, n_rows, Ty):
            y1 = min(n_rows, y0 + Ty)
            for x0 in range(0, n_cols, Tx):
                x1 = min(n_cols, x0 + Tx)
                cube = ramp[:, y0:y1, x0:x1]  # (N, ty, tx)
                N, ty, tx = cube.shape
                shape = (ty, tx)
                m = np.zeros(shape, dtype=np.float32)
                b = np.zeros(shape, dtype=np.float32)
                for iteration in range(max_iter):
                    sig2 = np.maximum(cube / gain + rn**2, 1e-6)
                    i = np.arange(N, dtype=np.float32)[:, None, None]
                    S0  = np.sum(1.0 / sig2, axis=0)
                    S1  = np.sum(i / sig2, axis=0)
                    S2  = np.sum(i**2 / sig2, axis=0)
                    S0x = np.sum(cube / sig2, axis=0)
                    S1x = np.sum(i * cube / sig2, axis=0)
                    ibar = S1 / S0
                    mdt = (S1x - ibar * S0x) / np.maximum(S2 - ibar**2 * S0, 1e-8)
                    m = mdt / dt
                    b = S0x / S0 - mdt * ibar
                    cube_model = b[None, :, :] + m[None, :, :] * i * dt
                    cube = np.clip(cube_model, 0, None)  # keep stable iteration
                slope[y0:y1, x0:x1] = m
                bias[y0:y1, x0:x1] = b
        return slope

    def iterative_sigma_weighted_ramp_fit(self,ramp, read_time, gain=3.0, rn=5.0, tile=(256, 256), return_bias=False):
        ramp = np.asarray(ramp)
        if ramp.ndim != 3:
            raise ValueError("ramp must have shape (N_reads, N_rows, N_cols).")

        N, n_rows, n_cols = ramp.shape
        read_times = np.linspace(0.0, read_time, N, dtype=np.float32)
        dt = np.mean(np.diff(read_times))
        slope = np.zeros((n_rows, n_cols), dtype=np.float32)
        bias  = np.zeros_like(slope)
        Ty, Tx = tile
        i = np.arange(N, dtype=np.float32)[:, None, None]
        for y0 in range(0, n_rows, Ty):
            y1 = min(n_rows, y0 + Ty)
            for x0 in range(0, n_cols, Tx):
                x1 = min(n_cols, x0 + Tx)
                cube = ramp[:, y0:y1, x0:x1]
                sig2 = np.maximum(cube / gain + rn**2, 1e-6)
                w = 1.0 / sig2
                S0  = np.sum(w, axis=0)
                S1  = np.sum(i * w, axis=0)
                S2  = np.sum(i**2 * w, axis=0)
                S0x = np.sum(w * cube, axis=0)
                S1x = np.sum(i * w * cube, axis=0)
                ibar = S1 / S0
                denom = np.maximum(S2 - ibar**2 * S0, 1e-8)
                mdt = (S1x - ibar * S0x) / denom
                m = mdt / dt
                b_loc = S0x / S0 - mdt * ibar
                slope[y0:y1, x0:x1] = m
                bias[y0:y1, x0:x1]  = b_loc
        if return_bias:
            return slope, bias
        else:
            return slope

    ################## swapping ######################################
    def swap_odd_even_columns(self,cube,n_amps=4,do_swap=True):
        if not do_swap:
            return cube

        if cube.ndim ==2:
            n_rows,n_cols = cube.shape
            ramp = np.empty_like(cube)
            block = n_cols // n_amps
            for a in range(n_amps):
                x0, x1 = a * block, (a + 1) * block
                sub = cube[x0:x1]
                nsub = sub.shape[-1]
                new_order = []
                for i in range(0, nsub, 2):
                    if i + 1 < nsub:
                        new_order.extend([i + 1, i])
                    else:
                        new_order.append(i)
                ramp[x0:x1] = sub[..., new_order]

        elif cube.ndim ==3:
            nreads,n_rows,n_cols = cube.shape
            ramp = np.empty_like(cube)
            block = n_cols // n_amps
            for a in range(n_amps):
                x0, x1 = a * block, (a + 1) * block
                sub = cube[..., x0:x1]
                nsub = sub.shape[-1]
                new_order = []
                for i in range(0, nsub, 2):
                    if i + 1 < nsub:
                        new_order.extend([i + 1, i])
                    else:
                        new_order.append(i)
                ramp[..., x0:x1] = sub[..., new_order]
        return ramp

    def optimal_extract_fast(
        self,
        R_transpose: sp.spmatrix,
        data_image: np.ndarray) -> np.ndarray:

        self.logger.info("Optimal extraction started")
        t0 = time.time()
        data_vector = np.asarray(data_image, dtype=np.float32).ravel()
        numerator = R_transpose @ data_vector
        #denominator = R2_transpose @ np.ones(data_vector.size, dtype=np.float32)
        #denominator_safe = np.maximum(denominator, 1e-9)
        optimized_flux = numerator #/ denominator_safe
        self.logger.info(f"Optimal extraction finished in {time.time() - t0:.4f} seconds.")
        return optimized_flux

    def _parse_sky_coord(
        self,
        coord_val: [str, float],
        is_ra: bool = False) -> float:
        """
        Helper function to robustly parse a sky coordinate value.

        It can handle:
        - Floats or integers (assumed to be in degrees).
        - Strings in various formats recognized by astropy.coordinates.Angle
        (e.g., '17:45:40.04', '-29d00m28.1s').

        Args:
            coord_val (str or float): The coordinate value from the FITS header.
            is_ra (bool): Flag to indicate if this is Right Ascension, to give
                      priority to hour-angle parsing for ambiguous strings.

        Returns:
            float: The coordinate value in degrees.
        """
        if isinstance(coord_val, (int, float)):
            return float(coord_val)
    
        if not isinstance(coord_val, str):
            raise TypeError(f"Coordinate must be a string or number, but got {type(coord_val)}.")

        try:
            if is_ra and ('h' in coord_val.lower() or ':' in coord_val):
                return Angle(coord_val, unit=u.hourangle).degree
            else:
                return Angle(coord_val, unit=u.deg).degree
        except (u.UnitsError, ValueError) as e:
            raise ValueError(f"Could not parse coordinate string '{coord_val}'. Error: {e}")

    def create_scales_wcs(
        self,
        cube_shape: tuple,
        header: fits.Header,
        center_map_is_zero_indexed: bool = True):
        """
        Create PC + CDELT WCS for a SCALES IFS cube.

        Cube shape:
            cube_shape = (n_wave, ny, nx)

        WCS convention:
            Axis 1 = RA
            Axis 2 = DEC
            Axis 3 = wavelength

        Wavelength unit:
            CUNIT3 = 'um'

        Pipeline-readable wavelength keywords:
            WAVSTART = wavelength start in micron
            WAVEND   = wavelength end in micron
            DWAVE    = wavelength step in micron/pixel
        """
        SCALES_PLATE_SCALE_ARCSEC = 0.02

        #order (y,x)
        center_map = {
            "LowRes-K": (54, 54),
            "LowRes-L":   (50, 60),
            "LowRes-M":   (50, 60),
            "LowRes-SED":   (50, 60),
            "LowRes-KL": (50, 60),
            "LowRes-PAH": (50, 60),
            "MedRes-K":   (50, 60),
            "MedRes-L":   (50, 60),
            "MedRes-M":   (50, 60),
        }

        default_center_yx = (54, 54)


        wave_config_um = {
            "LowRes-K": {"start": 2.0,  "end": 5.0},
            "LowRes-L":   {"start": 1.95, "end": 2.45},
            "LowRes-M":   {"start": 2.9,  "end": 4.15},
            "LowRes-SED":   {"start": 4.5,  "end": 5.2},
            "LowRes-KL": {"start": 2.0,  "end": 3.7},
            "LowRes-PAH": {"start": 3.1,  "end": 3.5},
            "MedRes-K":   {"start": 2.0,  "end": 2.4},
            "MedRes-L":   {"start": 2.9,  "end": 4.15},
            "MedRes-M":   {"start": 4.5,  "end": 5.2},
        }

        n_wave, ny, nx = cube_shape

        for key in ["RA", "DEC"]:
            if key not in header:
                raise ValueError(f"Essential FITS keyword '{key}' is missing.")

        # convertd ra and dec into degrees
        crval_ra = self._parse_sky_coord(header["RA"], is_ra=True)
        crval_dec = self._parse_sky_coord(header["DEC"], is_ra=False)

        # Spatial scale to degrees
        pixel_scale_deg = SCALES_PLATE_SCALE_ARCSEC / 3600.0

        # Position angle from degrees to radians
        pa_deg = float(header.get("PA", header.get("PARANG", 0.0)))
        pa_rad = np.deg2rad(pa_deg)

        cos_pa = np.cos(pa_rad)
        sin_pa = np.sin(pa_rad)

        # SCALES mode
        ifs_mode = header.get("IFSMODE", header.get("FILTER", "default")).strip()

        # Reference spatial pixel (IFSMODE name case insensitive)
        center_map_lower = {k.lower(): v for k, v in center_map.items()}
        crpix_y, crpix_x = center_map_lower.get(
            ifs_mode.lower(),
            default_center_yx,
        )

        # FITS CRPIX is 1-indexed
        if center_map_is_zero_indexed:
            crpix_x += 1
            crpix_y += 1

        # Wavelength range in microns
        wave_cfg = wave_config_um.get(ifs_mode)

        if wave_cfg is not None:
            wavstart_um = float(wave_cfg["start"])
            wavend_um = float(wave_cfg["end"])
        else:
            wavstart_um = float(header.get("WAVSTART", 2.0))
            wavend_um = float(header.get("WAVEND", wavstart_um + n_wave - 1))

        if n_wave > 1:
            dwave_um = (wavend_um - wavstart_um) / (n_wave - 1)
        else:
            dwave_um = 1.0

        crpix_wave = 1.0

        # --------------------------------------------------
        # PC + CDELT WCS
        # --------------------------------------------------
        # CDELT carries pixel scale.
        # PC carries rotation and axis coupling.
        #
        # Negative CDELT1 follows the usual astronomical convention:
        # increasing x corresponds to decreasing RA.
        # --------------------------------------------------

        wcs_dict = {
            "WCSAXES": 3,

            "CTYPE1": "RA---TAN",#Axis 1 is right ascension with tangent-plane projection.
            "CTYPE2": "DEC--TAN",#Axis 2 is declination with tangent-plane projection.
            "CTYPE3": "WAVE", #Axis 3 wavelength

            "CUNIT1": "deg",# RA in degrees
            "CUNIT2": "deg",# DEC in degrees
            "CUNIT3": "um", # wavelength in micro meters

            "CRVAL1": crval_ra,#these are the world coordinates of the reference pixel
            "CRVAL2": crval_dec,
            "CRVAL3": wavstart_um,

            "CRPIX1": crpix_x,#reference pixel coordinates corresponding to CRVAL
            "CRPIX2": crpix_y,
            "CRPIX3": crpix_wave,
            #The negative sign is standard for RA because increasing image
            #x usually corresponds to decreasing RA on the sky.
            "CDELT1": -pixel_scale_deg, #coordinate increment per pixel
            "CDELT2":  pixel_scale_deg, #dec increaes with y
            "CDELT3":  dwave_um, #wavelength increment

            "PC1_1": cos_pa, #rotation matrix for the spatial axis
            "PC1_2": -sin_pa,
            "PC1_3": 0.0,

            "PC2_1": sin_pa, #rotation matrix for the spatial axis
            "PC2_2": cos_pa,
            "PC2_3": 0.0,

            "PC3_1": 0.0, #Spectral PC term
            "PC3_2": 0.0,
            "PC3_3": 1.0,
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fits.verify.VerifyWarning)
            wcs = WCS(wcs_dict, naxis=3)

        wave_info = {
            "IFSMODE": ifs_mode,
            "WAVSTART": wavstart_um,
            "WAVEND": wavend_um,
            "DWAVE": dwave_um,
            "WAVEUNIT": "um",
        }

        return wcs, wave_info

    def wcs_header_update(
        self,
        data_cube,
        input_header: fits.Header,
        wcs: WCS,
        wave_info: dict,
    ):
        """
        Write SCALES cube with clean WCS.

        Final written spectral WCS is forced to microns so DS9 displays
        wavelength slider values in um.
        """
        SCALES_PLATE_SCALE_ARCSEC =0.02
        output_header = input_header.copy()

        # Remove old/conflicting WCS first
        #output_header = remove_old_wcs_keywords(output_header)

        # Add WCS generated by Astropy
        output_header.update(wcs.to_header())

        # --------------------------------------------------
        # Force final spectral WCS to microns for DS9 display
        # --------------------------------------------------
        output_header["CTYPE3"] = ("WAVE", "Wavelength axis")
        output_header["CUNIT3"] = ("um", "Wavelength unit")
        output_header["CRVAL3"] = (
            float(wave_info["WAVSTART"]),
            "Reference wavelength in micron",
        )
        output_header["CRPIX3"] = (
            1.0,
            "Reference wavelength pixel",
        )
        output_header["CDELT3"] = (
            float(wave_info["DWAVE"]),
            "Wavelength step in micron/pixel",
        )

        # Stable spectral PC matrix for DS9
        output_header["PC3_3"] = (1.0, "Spectral axis scale")
        output_header["PC3_1"] = (0.0, "No spectral-spatial coupling")
        output_header["PC3_2"] = (0.0, "No spectral-spatial coupling")
        output_header["PC1_3"] = (0.0, "No spatial-spectral coupling")
        output_header["PC2_3"] = (0.0, "No spatial-spectral coupling")

        # Remove any CD spectral terms if Astropy added them
        for key in ["CD3_1", "CD3_2", "CD3_3", "CD1_3", "CD2_3"]:
            if key in output_header:
                del output_header[key]

        # --------------------------------------------------
        # Pipeline-readable metadata
        # --------------------------------------------------
        output_header["PIXSCALE"] = (
            SCALES_PLATE_SCALE_ARCSEC,
            "SCALES plate scale in arcsec/spaxel",
        )

        output_header["IFSMODE"] = (
            wave_info["IFSMODE"],
            "SCALES IFS observing mode",
        )

        output_header["WAVSTART"] = (
            float(wave_info["WAVSTART"]),
            "Wavelength start in micron",
        )

        output_header["WAVEND"] = (
            float(wave_info["WAVEND"]),
            "Wavelength end in micron",
        )

        output_header["DWAVE"] = (
            float(wave_info["DWAVE"]),
            "Wavelength step in micron/pixel",
        )

        output_header["WAVEUNIT"] = (
            "um",
            "Pipeline wavelength unit",
        )

        output_header["WCSCORR"] = (
            True,
            "WCS keywords written",
        )

        output_header["WCSTYPE"] = (
            "INITIAL",
            "Initial SCALES cube WCS",
        )
        self.logger.info("WCS coordinates created")
        return output_header


    ########################################################################

    def _perform(self):
        self.logger.info("+++++++++++ Quicklook Started +++++++++++")

        input_data = self.action.args.name
        output_dir = os.path.dirname(input_data)
        filename = os.path.basename(input_data)

        calib_path = str(files("scalesdrp").joinpath("calib"))+ "/"
        #calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
        SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits') #IFS readnoise map
        read_noise_var = SIG_map_scaled.flatten().astype(np.float64)**2
        rmat_img = sparse.load_npz(calib_path+'bpmat_img.npz')
        rmat_ifs = sparse.load_npz(calib_path+'bpmat_ifs.npz')
        #R_matrix_lowres_k = load_npz(calib_path+'K_QL_rectmat_lowres.npz')
        R_matrix_lowres_l = load_npz(calib_path+'L_QL_rectmat_lowres.npz') #real
        #R_matrix_lowres_m = load_npz(calib_path+'M_QL_rectmat_lowres.npz')
        #R_matrix_lowres_sed = load_npz(calib_path+'SED_QL_rectmat_lowres.npz')
        #R_matrix_lowres_kl = load_npz(calib_path+'KL_QL_rectmat_lowres.npz')
        #R_matrix_lowres_pah = load_npz(calib_path+'PAH_QL_rectmat_lowres.npz')
        #R_matrix_medres_k = load_npz(calib_path+'K_QL_rectmat_medres.npz')
        #R_matrix_medres_l = load_npz(calib_path+'L_QL_rectmat_medres.npz')
        #R_matrix_medres_m = load_npz(calib_path+'M_QL_rectmat_medres.npz')


        with fits.open(input_data) as hdul:
            hdr = hdul[0].header
            obs_mode = hdr.get("CAMERA", "")
            ifs_mode = hdr.get("IFSMODE", "")
            #last_file =  hdr.get("LASTFILE", "")
            read_time = hdr.get("EXPTIME", "")
            #file_name = hdr.get("OFNAME", "")
            obj =       hdr.get("IMTYPE", "")

            NUM_FRAMES_FROM_SCIENCE = hdr.get("NREADS", "")
            print(f"OBSMODE = {obs_mode}")
            slope = None
            n_ext = len(hdul)
            print('number of extension = ',n_ext)
            t0 = time.time()
            if (
                filename.endswith(".fits")
                and "_dramp" not in filename
                and "_qramp" not in filename):
                if n_ext == 1:
                    data_1 = hdul[0].data
                    if data_1 is None:
                        raise ValueError("No data in primary HDU.")
                    elif data_1.ndim == 2:
                        self.logger.info("Found a single frame.")
                        data_11 = self.swap_odd_even_columns(data_1,do_swap=True)
                        slope_filled1 = reference.reffix_hxrg(data_11, nchans=4, fixcol=True)
                        self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")

                    elif data_1.ndim == 3:
                        data_11 = self.swap_odd_even_columns(data_1,do_swap=True)
                        img_corr = reference.reffix_hxrg(data_11, nchans=4, fixcol=True)
                        self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                        slope_filled1 = self.iterative_sigma_weighted_ramp_fit(
                            img_corr,
                            read_time=read_time)

                        t1 = time.time()
                        self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                    else:
                        raise ValueError(f"Unexpected data shape: {data_1.shape}")
                elif n_ext >= 2:
                    img2d = hdul[1].data
                    ramp3d = hdul[0].data
                    if img2d.ndim != 2 or ramp3d.ndim != 3:
                        raise ValueError(f"Expected (2D, 3D) shapes, got {img2d.shape}, {ramp3d.shape}")

                    data_11 = self.swap_odd_even_columns(ramp3d,do_swap=True)
                    img_corr = reference.reffix_hxrg(data_11, nchans=4, fixcol=True)
                    self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                    slope_filled1 = self.iterative_sigma_weighted_ramp_fit(
                        img_corr,
                        read_time=read_time)
                    t1 = time.time()
                    self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")

        if obs_mode == "Im":
            self.logger.info("BPM correction started")            
            slope_filled2 = rmat_img*np.matrix(slope_filled1.flatten().reshape([np.prod(slope_filled1.shape),1]))
            slope_filled = np.array(slope_filled2).reshape(slope_filled1.shape)
            self.logger.info("BPM correction completed")
            self.fits_writer_steps(
                data=slope_filled,
                header=hdr,
                output_dir=output_dir,
                input_filename=filename,
                suffix='_ql',
                proctab=self.proctab,
                overwrite=True)

        if obs_mode == "IFS":
            self.logger.info("BPM correction started")
            slope_filled2 = rmat_ifs*np.matrix(slope_filled1.flatten().reshape([np.prod(slope_filled1.shape),1]))
            slope_filled = np.array(slope_filled2).reshape(slope_filled1.shape)
            self.logger.info("BPM correction completed")

            self.fits_writer_steps(
                data=slope_filled,
                header=hdr,
                output_dir=output_dir,
                input_filename=filename,
                suffix='_ql',
                proctab=self.proctab,
                overwrite=True)

            if ifs_mode == "LowRes-K":
                print("IFSMODE is", ifs_mode)
                print("IMTYPE is", obj)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "K_QL_rectmat_lowres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    cube1 = self.optimal_extract_fast(
                        R_matrix_lowres_k,
                        slope_filled)

                    cube= cube1.reshape(56, 103, 110)
                    wcs, wave_info = self.create_scales_wcs(
                        cube_shape=cube.shape,
                        header=hdr)
                    header = self.wcs_header_update(
                        data_cube=cube,
                        input_header=hdr,
                        wcs=wcs,
                        wave_info=wave_info)
                    self.fits_writer_steps(
                        data=cube,
                        header=header,
                        output_dir=output_dir,
                        input_filename=filename,
                        suffix='_ql_cube',
                        proctab=self.proctab,
                        overwrite=True)

            elif ifs_mode == "LowRes-L":
                print("IFSMODE is", ifs_mode)
                
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "L_QL_rectmat_lowres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    cube1 = self.optimal_extract_fast(
                        R_matrix_lowres_l,
                        slope_filled)

                    cube= cube1.reshape(56,103,110)
                    wcs, wave_info = self.create_scales_wcs(
                        cube_shape=cube.shape,
                        header=hdr)
                    header = self.wcs_header_update(
                        data_cube=cube,
                        input_header=hdr,
                        wcs=wcs,
                        wave_info=wave_info)
                    
                    self.fits_writer_steps(
                        data=cube,
                        header=header,
                        output_dir=output_dir,
                        input_filename=filename,
                        suffix='_ql_cube',
                        proctab=self.proctab,
                        overwrite=True)

            elif ifs_mode == "LowRes-M":
                self.logger.info("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "M_QL_rectmat_lowres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):
                    
                    cube1 = self.optimal_extract_fast(
                        R_matrix_lowres_m,
                        slope_filled)

                    cube= cube1.reshape(56,103,110)
                    wcs, wave_info = self.create_scales_wcs(
                        cube_shape=cube.shape,
                        header=hdr)
                    header = self.wcs_header_update(
                        data_cube=cube,
                        input_header=hdr,
                        wcs=wcs,
                        wave_info=wave_info)
                    self.fits_writer_steps(
                        data=cube,
                        header=header,
                        output_dir=output_dir,
                        input_filename=filename,
                        suffix='_ql_cube',
                        proctab=self.proctab,
                        overwrite=True)

            elif ifs_mode == "LowRes-SED":
                print("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "SED_QL_rectmat_lowres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    cube1 = self.optimal_extract_fast(
                        R_matrix_lowres_sed,
                        slope_filled)

                    cube= cube1.reshape(56,103,110)
                    wcs, wave_info = self.create_scales_wcs(
                        cube_shape=cube.shape,
                        header=hdr)
                    header = self.wcs_header_update(
                        data_cube=cube,
                        input_header=hdr,
                        wcs=wcs,
                        wave_info=wave_info)
                    self.fits_writer_steps(
                        data=cube,
                        header=header,
                        output_dir=output_dir,
                        input_filename=filename,
                        suffix='_ql_cube',
                        proctab=self.proctab,
                        overwrite=True)

            elif ifs_mode == "LowRes-KL":
                print("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "KL_QL_rectmat_lowres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    cube1 = self.optimal_extract_fast(
                        R_matrix_lowres_kl,
                        slope_filled)

                    cube= cube1.reshape(54,108,108)
                    wcs, wave_info = self.create_scales_wcs(
                        cube_shape=cube.shape,
                        header=hdr)
                    header = self.wcs_header_update(
                        data_cube=cube,
                        input_header=hdr,
                        wcs=wcs,
                        wave_info=wave_info)
                    self.fits_writer_steps(
                        data=cube,
                        header=header,
                        output_dir=output_dir,
                        input_filename=filename,
                        suffix='_ql_cube',
                        proctab=self.proctab,
                        overwrite=True)

            elif ifs_mode == "LowRes-PAH":
                print("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "PAH_QL_rectmat_lowres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    cube1 = self.optimal_extract_fast(
                        R_matrix_lowres_pah,
                        slope_filled)

                    cube= cube1.reshape(54,108,108)
                    wcs, wave_info = self.create_scales_wcs(
                        cube_shape=cube.shape,
                        header=hdr)
                    header = self.wcs_header_update(
                        data_cube=cube,
                        input_header=hdr,
                        wcs=wcs,
                        wave_info=wave_info)
                    self.fits_writer_steps(
                        data=cube,
                        header=header,
                        output_dir=output_dir,
                        input_filename=filename,
                        suffix='_ql_cube',
                        proctab=self.proctab,
                        overwrite=True)

            elif ifs_mode == "MedRes-K":
                print("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "K_QL_rectmat_medres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    cube1 = self.optimal_extract_fast(
                        R_matrix_medres_k,
                        slope_filled)

                    cube= cube1.reshape(1900,18,17)
                    wcs, wave_info = self.create_scales_wcs(
                        cube_shape=cube.shape,
                        header=hdr)
                    header = self.wcs_header_update(
                        data_cube=cube,
                        input_header=hdr,
                        wcs=wcs,
                        wave_info=wave_info)
                    self.fits_writer_steps(
                        data=cube,
                        header=header,
                        output_dir=output_dir,
                        input_filename=filename,
                        proctab=self.proctab,
                        suffix='_ql_cube',
                        overwrite=True)

            elif ifs_mode == "MedRes-L":
                print("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "L_QL_rectmat_medres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    cube1 = self.optimal_extract_fast(
                        R_matrix_medres_l,
                        slope_filled)

                    cube= cube1.reshape(1900,18,17)
                    wcs, wave_info = self.create_scales_wcs(
                        cube_shape=cube.shape,
                        header=hdr)
                    header = self.wcs_header_update(
                        data_cube=cube,
                        input_header=hdr,
                        wcs=wcs,
                        wave_info=wave_info)
                    self.fits_writer_steps(
                        data=cube,
                        header=header,
                        output_dir=output_dir,
                        input_filename=filename,
                        proctab=self.proctab,
                        suffix='_ql_cube',
                        overwrite=True)

            elif ifs_mode == "MedRes-M":
                print("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "M_QL_rectmat_medres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    cube1 = self.optimal_extract_fast(
                        R_matrix_medres_m,
                        slope_filled)

                    cube= cube1.reshape(1900,18,17)
                    wcs, wave_info = self.create_scales_wcs(
                        cube_shape=cube.shape,
                        header=hdr)
                    header = self.wcs_header_update(
                        data_cube=cube,
                        input_header=hdr,
                        wcs=wcs,
                        wave_info=wave_info)
                    self.fits_writer_steps(
                        data=cube,
                        header=header,
                        output_dir=output_dir,
                        input_filename=filename,
                        suffix='_ql_cube',
                        proctab=self.proctab,
                        overwrite=True)
        else:
            self.logger.info("Unknown MODE of observation")

        log_string = QuickLook.__module__
        self.logger.info(log_string)

        return self.action.args
    # END: class QuickLook()
