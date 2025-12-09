from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer

import numpy as np
import pkg_resources
import pickle
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import load_npz
import scipy.sparse as sp
from astropy.nddata import CCDData
import astropy.units as u
from scipy.optimize import lsq_linear 
import time
from astropy.nddata import StdDevUncertainty
from astropy.coordinates import Angle
from astropy.wcs import WCS
import os

class SpectralExtract(BasePrimitive):
    """
	This primitive will perform spectral cube extraction using Optimal extraction and
    the chi square extraction method. A linear WCS informations are updated to the 
    the final output header. More details are listed below individual functions.  
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger
    
    ################# Optimal extraction and error #########################
    def optimal_extract_with_error(
        self,
        R_transpose: sp.spmatrix, 
        data_image: np.ndarray,
        sigma_image: np.ndarray,
        read_noise_variance_vector: np.ndarray, 
        gain: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
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
        sigma_vector = sigma_image.array.flatten().astype(np.float64)
        variance_from_map = sigma_vector**2
        photon_noise_variance = data_vector_d.clip(min=0) / gain
        #total_variance = read_noise_variance_vector + photon_noise_variance
        #are we doubling the readnoise below?
        total_variance = read_noise_variance_vector + photon_noise_variance + variance_from_map
        total_variance[total_variance <= 0] = 1e-9  
        inverse_variance = 1.0 / total_variance
        weighted_data = data_vector_d * inverse_variance ## d_i / σ_i^2
        numerator = R_transpose @ weighted_data # Σ R_ki d_i / σ_i^2
        R_transpose_squared = R_transpose.power(2)
        denominator = R_transpose_squared @ inverse_variance # Σ R_ki^2 / σ_i^2
        denominator_safe = np.maximum(denominator, 1e-9)
        optimized_flux = numerator / denominator_safe
        flux_variance = 1.0 / denominator_safe
        flux_error = np.sqrt(flux_variance)
        end_time1 = time.time()
        t1 = (end_time1 - start_time1)
        self.logger.info(f"Optimal extraction finished in {t1:.4f} seconds.")
        return optimized_flux, flux_error

    ############### chi square extraction #########################

    def solve_bounded_weighted_nnls(
        self,
        R_matrix: sp.spmatrix,
        data_vector: np.ndarray,
        read_noise_variance_vector: np.ndarray,
        gain: float,
        A_guess: np.ndarray,  
        bound_factor: float = 1.0, 
        tolerance: float = 1e-6) -> np.ndarray:
        """
        Args:
            R_matrix: The (N_pixels, N_fluxes) sparse rectification matrix.
            data_vector: The (H,W) input data vector.
            read_noise_variance_vector: The (N_pixels,) 1D vector of read noise variance.
            gain: The detector gain, used for calculating photon noise.
            A_guess: Amplutide guess from the optimal extraction 1D array (N_fluxes,)
            bound_factor:  How many times the guess to set the upper bound
        Returns:
            Returns the best-fit amplitude
        """ 
        self.logger.info("\nSolving with BOUNDED weighted non-negative least squares...")
        photon_noise_variance = data_vector.clip(min=0) / gain
        total_variance = read_noise_variance_vector + photon_noise_variance
        total_variance[total_variance <= 0] = 1e-9
        weights = 1.0 / np.sqrt(total_variance)
        W = sp.diags(weights, format='csr')
        R_prime = W @ R_matrix
        d_prime = W @ data_vector
        lower_bounds = 0
        upper_bounds = np.maximum(0, A_guess) * bound_factor
        upper_bounds += 1e-9 
        bounds = (lower_bounds, upper_bounds)
        start_time = time.time()
        lsq_options = {'tol': tolerance, 'verbose': 0}
        res = lsq_linear(R_prime, d_prime, bounds=bounds, **lsq_options) 
        end_time = time.time()
        t = (end_time - start_time)/60.0
        self.logger.info(f"Bounded lsq_linear finished in {t:.4f} mins.")
        return res.x

    ############### chi sqaure error estimation #########################

    def calculate_error_flux_cube(
        self,
        R_matrix: sp.spmatrix,
        flux_vector_A: np.ndarray,
        var_read_vector: np.ndarray,
        flux_shape_3d: tuple,
        gain: float = 1.0) -> np.ndarray:
        """
        This provides a fast and reliable estimate of the standard deviation for each
        flux element from a least-squares fit.
        The variance of each flux parameter A_j is approximated as:
        Var(A_j) ≈ 1 / H_jj
        where H is the Hessian matrix H_jj = Σ_i (R_ij^2 / σ_i^2).
        Args:
            R_matrix: The (N_pixels, N_fluxes) sparse rectification matrix.
            flux_vector_A: The (N_fluxes,) solved flux vector.
            var_read_vector: The (N_pixels,) vector of read noise variance.
            flux_shape_3d: The 3D shape of the final flux cube (e.g., (54, 108, 108)).
            gain: The detector gain, used for calculating photon noise.
        Returns:
            A 3D NumPy array containing the standard deviation (error) for each flux element.
        """
        self.logger.info("\nCalculating the error flux cube...")
        start_time = time.time()
        model_data_vector = R_matrix @ flux_vector_A
        model_data_vector[model_data_vector < 0] = 0
        var_photon_vector = model_data_vector / gain
        total_var_vector = var_read_vector + var_photon_vector
        total_var_vector[total_var_vector <= 0] = np.inf
        R_squared = R_matrix.power(2)
        inverse_variance_vector = 1.0 / total_var_vector
        hessian_diagonal = R_squared.T @ inverse_variance_vector
        flux_variance_vector = np.zeros_like(hessian_diagonal)
        valid_mask = hessian_diagonal > 0
        flux_variance_vector[valid_mask] = 1.0 / hessian_diagonal[valid_mask]
        flux_variance_vector[~valid_mask] = np.inf
        error_flux_vector = np.sqrt(flux_variance_vector)
        error_cube = error_flux_vector.reshape(flux_shape_3d)
        end_time = time.time()
        self.logger.info(f"Error cube calculation finished in {end_time - start_time:.2f} seconds.")
        return error_cube

    ##################### WCS ##############################################
    def _parse_sky_coord(self,coord_val: [str, float], is_ra: bool = False) -> float:
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

    def create_wcs_flexible(
        self,
        cube_shape: tuple,
        header: fits.Header,
        center_map: dict,
        default_center_yx: tuple) -> WCS:
        """
        Creates a standard FITS WCS object for a generic IFU data cube,
        robustly handling different input formats and keywords.
        This function automatically parses various RA/Dec formats and provides
        sensible defaults for optional FITS keywords.
        Args:
            cube_shape (tuple): The (n_wave, ny, nx) shape of the data cube.
            header (fits.Header): The FITS header of the observation.
            center_map (dict): A case-insensitive dictionary mapping filter names
            to their (y, x) reference pixel coordinates.
            default_center_yx (tuple): The default (y, x) reference pixel to use if
            the filter is not found in the center_map.
        Returns:
            astropy.wcs.WCS: A fully constructed WCS object.
            Raises ValueError if essential keywords are missing.
        """
        self.logger.info("Creating Flexible WCS for IFU data cube.")
        required_keys = ['RA', 'DEC', 'PIXSCALE']
        for key in required_keys:
            if key not in header:
                raise ValueError(f"Essential FITS keyword '{key}' is missing from the header.")
        try:
            crval_ra = self._parse_sky_coord(header['RA'], is_ra=True)
            crval_dec = self._parse_sky_coord(header['DEC'])
        except ValueError as e:
            raise ValueError(f"Failed to create WCS due to coordinate parsing error: {e}")
        pixel_scale_arcsec = float(header['PIXSCALE'])
        position_angle_deg = float(header.get('PA', 0.0)) # Default PA to 0 if not present
        crval_wave = float(header.get('CRVAL1', header.get('CRVAL3', 1.0)))
        cdelt_wave = float(header.get('CDELT1', header.get('CDELT3', header.get('CD3_3', 1.0))))
        crpix_wave = float(header.get('CRPIX1', header.get('CRPIX3', 1.0)))
        cunit_wave = header.get('CUNIT3', header.get('CUNIT1', 'nm'))
        filter_name = header.get('FILTER', 'default').strip().lower()
        center_map_lower = {k.lower(): v for k, v in center_map.items()}
        crpix_sky_yx = center_map_lower.get(filter_name, default_center_yx)
        crpix_y, crpix_x = crpix_sky_yx
        wcs_dict = {
            'CTYPE1': 'RA---TAN', 'CUNIT1': 'deg', 'CRVAL1': crval_ra,  'CRPIX1': crpix_x,
            'CTYPE2': 'DEC--TAN', 'CUNIT2': 'deg', 'CRVAL2': crval_dec, 'CRPIX2': crpix_y,
            'CTYPE3': 'WAVE',     'CUNIT3': cunit_wave,'CRVAL3': crval_wave,'CRPIX3': crpix_wave}
    
        scale_deg = pixel_scale_arcsec / 3600.0
        pa_rad = np.deg2rad(position_angle_deg)
        cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
    
        wcs_dict['CDELT1'] = -scale_deg
        wcs_dict['CDELT2'] = scale_deg
        wcs_dict['CDELT3'] = cdelt_wave
    
        wcs_dict['PC1_1'] = cos_pa
        wcs_dict['PC1_2'] = -sin_pa
        wcs_dict['PC2_1'] = sin_pa
        wcs_dict['PC2_2'] = cos_pa
        
        wcs_dict.update({'PC1_3': 0, 'PC2_3': 0, 'PC3_1': 0, 'PC3_2': 0, 'PC3_3': 1})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fits.verify.VerifyWarning)
            w = WCS(wcs_dict, naxis=3)
        return w

    ################# search for lenslet flat #######################################
    def load_and_normalize_lenslet_flat(
        self,
        ifsmode,
        *,
        clip_sigma=7.0,
        iterations=3,
        method='median'):
        """
        Find <prefix>_cube_flatlens.fits by IFSMODE, ensure it's a 3D cube,
        load (data, uncert), normalize each slice, and propagate uncertainties.
        Search order: ./redux/ then pkg calib/.
        Returns (flat_norm, flat_norm_uncert) or (None, None).
        """
        norm_ifsmode = (ifsmode or "").strip().upper().replace("_", "-")

        def _try_dir(base_dir):
            if not os.path.isdir(base_dir):
                return None, None, None
            # look for any file ending in _cube_flatlens.fits
            candidates = [f for f in os.listdir(base_dir)
                    if f.endswith("_cube_flatlens.fits") and os.path.isfile(os.path.join(base_dir, f))]
            for fname in candidates:
                path = os.path.join(base_dir, fname)
                try:
                    with fits.open(path) as hdul:
                        hdr = hdul[0].header
                        file_ifs = (hdr.get("IFSMODE") or "").strip().upper().replace("_", "-")
                        if file_ifs != norm_ifsmode:
                            continue

                        data = hdul[0].data
                        if data is None:
                            self.logger.debug(f"Skipping {fname}: primary data is None.")
                            continue
                        if data.ndim != 3:
                            self.logger.debug(f"Skipping {fname}: not a cube (ndim={data.ndim}).")
                            continue

                        # try to get uncertainty cube with same shape
                        uncert = None
                        if "UNCERT" in hdul and hdul["UNCERT"].data is not None:
                            if hdul["UNCERT"].data.shape == data.shape:
                                uncert = hdul["UNCERT"].data
                            else:
                                self.logger.debug(f"Skipping UNCERT in {fname}: shape mismatch {hdul['UNCERT'].data.shape} vs {data.shape}.")
                        elif len(hdul) > 1 and getattr(hdul[1], "data", None) is not None:
                            if hdul[1].data.shape == data.shape:
                                uncert = hdul[1].data
                            else:
                                self.logger.debug(f"Skipping ext[1] as UNCERT in {fname}: shape mismatch {hdul[1].data.shape} vs {data.shape}.")

                        return path, data, uncert
                except Exception as e:
                    self.logger.debug(f"Error reading {fname}: {e}")
                    continue
            return None, None, None

        # 1) ./redux
        path_used, flat, uflat = _try_dir(os.path.join(os.getcwd(), "redux"))
        # 2) pkg calib if not found
        if flat is None:
            pkg_dir = pkg_resources.resource_filename('scalesdrp', 'calib/')
            path_used, flat, uflat = _try_dir(pkg_dir)

        if flat is None:
            self.logger.warning(f"No lenslet flat cube found for IFSMODE={ifsmode}.")
            return None, None

        self.logger.info(f"Loaded lenslet flat cube: {os.path.basename(path_used)}")

        flat = np.asarray(flat, dtype=np.float64)
        uflat = None if uflat is None else np.asarray(uflat, dtype=np.float64)

        N, Y, X = flat.shape
        flat_norm = np.full_like(flat, np.nan, dtype=np.float64)
        flat_norm_uncert = None if uflat is None else np.full_like(uflat, np.nan, dtype=np.float64)

        for k in range(N):
            f = flat[k]
            invalid = ~np.isfinite(f) | (f <= 0)
            vals = f[~invalid]
            if vals.size == 0:
                continue

            # iterative sigma clip
            clipped = vals.copy()
            for _ in range(iterations):
                med = np.median(clipped); std = np.std(clipped)
                if not np.isfinite(med) or not np.isfinite(std) or std == 0:
                    break
                keep = (clipped > med - clip_sigma*std) & (clipped < med + clip_sigma*std)
                if np.all(keep):
                    break
                clipped = clipped[keep]
                if clipped.size == 0:
                    break
            if clipped.size == 0:
                continue

            a = np.median(clipped) if method.lower() == 'median' else np.mean(clipped)
            if not np.isfinite(a) or a <= 0:
                continue

            # estimate σ_a from clipped scatter
            s = np.std(clipped); N_eff = max(1, clipped.size)
            sa = s / np.sqrt(N_eff)

            # normalize slice
            flat_norm[k] = f / a
            flat_norm[k, invalid] = np.nan

            # uncertainty propagation if available
            if uflat is not None:
                uf = uflat[k]
                if uf is not None and uf.shape == f.shape:
                    term1 = (uf / a)**2
                    term2 = ((f * sa) / (a**2))**2
                    flat_norm_uncert[k] = np.sqrt(term1 + term2)
                    flat_norm_uncert[k, invalid] = np.nan

        self.logger.info(f"Normalized lenslet flat cube for IFSMODE={ifsmode}.")
        return flat_norm, flat_norm_uncert

    ##################### flat correction to the cube ##############################
    def apply_flatlens(
        self,
        data,                # (H,W) or (N,H,W) science
        sigma_data,          # same shape as data (1σ)
        calib,               # (H,W) or (N,H,W) lenslet flat (prefer normalized)
        sigma_calib,         # same shape as calib (1σ); can be None
        imtype,              # should be 'FLATLENS'
        *,
        eps: float = 1e-10,  # guard for near-zero flats
        clip_nan: bool = True):
        """
        Divide science by lenslet-flat (supports 2D or 3D cubes) with uncertainty propagation:
            out = data / F
            σ_out^2 = (σ_data / F)^2 + (data * σ_F / F^2)^2
        Shapes:
            - data, sigma_data: (H,W) or (N,H,W)
            - calib, sigma_calib: (H,W) or (N,H,W)
            (Broadcasting is supported for (H,W) calib across N if needed.)
        """
        kind = (imtype or "").strip().upper()
        if kind != "FLATLENS":
            raise ValueError(f"IMTYPE='{imtype}' not supported here; expected 'FLATLENS'.")
            return data.astype(np.float32), sigma_data.astype(np.float32)

        if calib is None or (isinstance(calib, np.ndarray) and calib.size == 0):
            self.logger.warning("No flatlens data provided; skipping flat correction.")
            return data.astype(np.float32), sigma_data.astype(np.float32)
        
        # Cast to float arrays; allow sigma_calib=None (treated as zeros)
        data        = np.asarray(data, dtype=float)
        sigma_data  = np.asarray(sigma_data, dtype=float)
        calib       = np.asarray(calib, dtype=float)
        sigma_calib = np.zeros_like(calib, dtype=float) if sigma_calib is None else np.asarray(sigma_calib, dtype=float)

        try:
            if data.shape != sigma_data.shape:
                raise ValueError(f"sigma_data shape {sigma_data.shape} must match data {data.shape}.")
            if calib.shape != data.shape:
                # Allow (H,W) flat to apply to (N,H,W) science
                if calib.ndim == 2 and data.ndim == 3 and calib.shape == data.shape[-2:]:
                    calib = np.broadcast_to(calib, data.shape)
                    sigma_calib = np.broadcast_to(sigma_calib, data.shape) if sigma_calib.ndim == 2 else sigma_calib
                else:
                    raise ValueError(f"calib shape {calib.shape} not compatible with data {data.shape}.")
            if sigma_calib.shape != calib.shape:
                # Allow (H,W) σ_flat to broadcast to (N,H,W)
                if sigma_calib.ndim == 2 and calib.ndim == 3 and sigma_calib.shape == calib.shape[-2:]:
                    sigma_calib = np.broadcast_to(sigma_calib, calib.shape)
                else:
                    raise ValueError(f"sigma_calib shape {sigma_calib.shape} not compatible with calib {calib.shape}.")
        except Exception as e:
            self.logger.warning(f"[apply_flatlens] Shape mismatch — skipping flat correction: {e}")
            return data.astype(np.float32), sigma_data.astype(np.float32)

        # Guard against invalid flat values
        finite_flat = np.isfinite(calib)
        if not np.any(finite_flat):
            self.logger.warning("Flatlens data invalid or all NaN; skipping correction.")
            return data.astype(np.float32), sigma_data.astype(np.float32)

        safe_flat = np.where(finite_flat & (np.abs(calib) > eps), calib, np.nan)

        # Division
        out = data / safe_flat

        # Uncertainty propagation
        # σ_out^2 = (σ_data / F)^2 + (data * σ_F / F^2)^2
        term1 = (sigma_data / safe_flat) ** 2
        term2 = ((data * sigma_calib) / (safe_flat ** 2)) ** 2
        sig = np.sqrt(term1 + term2)

        if clip_nan:
            good = np.isfinite(out) & np.isfinite(sig)
            out = np.where(good, out, np.nan)
            sig = np.where(good, sig, np.nan)

        self.logger.info("Successfully applied lenslet flat correction.")
        return out.astype(np.float32), sig.astype(np.float32)

    #################################################################################

    def _perform(self):
        #self.logger.info("Spectral Extraction Started")
        #tab = self.context.proctab.search_proctab(
        #    frame=self.action.args.ccddata, target_type='OBJECT',
        #    nearest=True)

        obsmode = self.action.args.ccddata.header['OBSMODE']

        if obsmode=='IFS':
            SCALES_CENTER_MAP = {
                'LowRes-SED': (54, 54),
                'LowRes-K': (50, 60),
                'LowRes-L': (50, 60),
                'LowRes-M': (50, 60),
                'LowRes-H20': (50, 60),
                'LowRes-PAH': (50, 60),
                'MedRes-K': (50, 60),
                'MedRes-L': (50, 60),
                'MedRes-M': (50, 60),}

            SCALES_DEFAULT_CENTER = (54, 54)
            calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
            readnoise = fits.getdata(calib_path+'sim_readnoise.fits')
            #var_read_vector = (readnoise.flatten().astype(np.float64))**2
            sigma_image = self.action.args.ccddata.uncertainty
            var_read_vector = (sigma_image.array.flatten().astype(np.float64))**2+(readnoise.flatten().astype(np.float64))**2
            GAIN = 1.0#self.action.args.ccddata.header['GAIN']

            data_image = self.action.args.ccddata.data
            data_vector_d = data_image.flatten().astype(np.float64)
            sigma_image = self.action.args.ccddata.uncertainty


            ifsmode = self.action.args.ccddata.header['IFSMODE']
            if ifsmode=='LowRes-K':
                R_for_extract = load_npz(calib_path+'K_C2_rectmat_lowres.npz')
                R_matrix = load_npz(calib_path+'K_QL_rectmat_lowres.npz')
                FLUX_SHAPE_3D = (56, 103, 110)
            elif ifsmode=='LowRes-L':
                R_for_extract = load_npz(calib_path+'L_C2_rectmat_lowres.npz')
                R_matrix = load_npz(calib_path+'L_QL_rectmat_lowres.npz')
                FLUX_SHAPE_3D = (56, 103, 110)
            elif ifsmode=='LowRes-M':
                R_for_extract = load_npz(calib_path+'M_C2_rectmat_lowres.npz')
                R_matrix = load_npz(calib_path+'M_QL_rectmat_lowres.npz')
                FLUX_SHAPE_3D = (56, 103, 110)
            elif ifsmode=='LowRes-SED':
                R_for_extract = load_npz(calib_path+'SED_C2_rectmat_lowres.npz')
                R_matrix = load_npz(calib_path+'SED_QL_rectmat_lowres.npz')
                FLUX_SHAPE_3D = (56, 103, 110)
            elif ifsmode=='LowRes-H2O':
                R_for_extract = load_npz(calib_path+'H2O_C2_rectmat_lowres.npz')
                R_matrix = load_npz(calib_path+'H2O_QL_rectmat_lowres.npz')
                FLUX_SHAPE_3D = (56, 103, 110)
            elif ifsmode=='LowRes-PAH':
                R_for_extract = load_npz(calib_path+'PAH_C2_rectmat_lowres.npz')
                R_matrix = load_npz(calib_path+'PAH_QL_rectmat_lowres.npz')
                FLUX_SHAPE_3D = (56, 103, 110)
            elif ifsmode=='MedRes-K':
                R_for_extract = load_npz(calib_path+'K_C2_rectmat_medres.npz')
                R_matrix = load_npz(calib_path+'K_QL_rectmat_medres.npz')
                FLUX_SHAPE_3D = (1900, 103, 110)
            elif ifsmode=='MedRes-L':
                R_for_extract = load_npz(calib_path+'L_C2_rectmat_medres.npz')
                R_matrix = load_npz(calib_path+'L_QL_rectmat_medres.npz')
                FLUX_SHAPE_3D = (1900, 103, 110)
            elif ifsmode=='MedRes-M':
                R_for_extract = load_npz(calib_path+'M_C2_rectmat_medres.npz')
                R_matrix = load_npz(calib_path+'M_QL_rectmat_medres.npz')
                FLUX_SHAPE_3D = (1900, 103, 110)
            
            A_guess_cube,A_guess_cube_err = self.optimal_extract_with_error(
                R_matrix,
                data_image,
                sigma_image,
                var_read_vector)

            A_guess_vector = A_guess_cube.flatten()
        
            A_opt = A_guess_cube.reshape(FLUX_SHAPE_3D)
            A_opt_err = A_guess_cube_err.reshape(FLUX_SHAPE_3D)

            #A_optimal_nnls = self.solve_bounded_weighted_nnls(
            #    R_for_extract, data_vector_d, var_read_vector, GAIN, A_guess_vector)

            #Amp_chi_square = A_optimal_nnls.reshape(FLUX_SHAPE_3D)

            #Amp_chi_square_err = self.calculate_error_flux_cube(
            #    R_matrix=R_for_extract,
            #    flux_vector_A=A_optimal_nnls,
            #    var_read_vector=var_read_vector,
            #    flux_shape_3d=FLUX_SHAPE_3D,
            #    gain=GAIN)

            norm_flatlens,norm_flatlens_uncert = self.load_and_normalize_lenslet_flat(ifsmode)

            A_opt, A_opt_err = self.apply_flatlens(
                A_opt,
                A_opt_err,
                norm_flatlens,
                norm_flatlens_uncert,
                imtype='FLATLENS')

            #Amp_chi_square, Amp_chi_square_err = self.apply_flatlens(
            #    Amp_chi_square,
            #    Amp_chi_square_err,
            #    norm_flatlens,
            #    norm_flatlens_uncert)

            #chi_rslt = CCDData(
            #    data=Amp_chi_square,
            #    uncertainty=StdDevUncertainty(Amp_chi_square_err),
            #    meta=self.action.args.ccddata.header,
            #    unit='adu')

            opt_rslt = CCDData(
                data=A_opt,
                uncertainty=StdDevUncertainty(A_opt_err),
                meta=self.action.args.ccddata.header,
                unit='adu')

            self.action.args.ccddata.data = A_opt

            my_wcs = self.create_wcs_flexible(
                    cube_shape=FLUX_SHAPE_3D,
                    header=self.action.args.ccddata.header,
                    center_map=SCALES_CENTER_MAP,
                    default_center_yx=SCALES_DEFAULT_CENTER)

            final_header = my_wcs.to_header()
            self.action.args.ccddata.header.update(final_header)

            log_string = SpectralExtract.__module__
            self.action.args.ccddata.header['HISTORY'] = log_string
            self.action.args.ccddata.header['HISTORY'] = 'WCS keywords updated (purely linear).'
            self.logger.info(log_string)


            #scales_fits_writer(ccddata = chi_rslt, 
            #    table=self.action.args.table,
            #    output_file=self.action.args.name,
            #    output_dir=self.config.instrument.output_directory,
            #    suffix="chi_cube")
 
            scales_fits_writer(ccddata = opt_rslt, 
                table=self.action.args.table,
                output_file=self.action.args.name,
                output_dir=self.config.instrument.output_directory,
                suffix="opt_cube")

        #self.context.proctab.update_proctab(frame=self.action.args.ccddata, suffix="cube", newtype='OBJECT',
        #        filename=self.action.args.ccddata.header['OFNAME'])
        #self.context.proctab.write_proctab(
        #        tfil=self.config.instrument.procfile)
        return self.action.args
    # END: class LeastExtract()
