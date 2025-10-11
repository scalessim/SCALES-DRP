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

class SpectralExtract(BasePrimitive):
    """
	This primitive will perform spectral cube extraction using Optimal extraction and
    the chi square extraction method. A linear WCS informations are updated to the 
    the final output header. More details are listed below individual functions.  
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger
        #self.logger.info("Spectral Extract: object created")

    def _perform(self):
        self.logger.info("Spectral Extraction Started")
        tab = self.context.proctab.search_proctab(
            frame=self.action.args.ccddata, target_type='OBJECT',
            nearest=True)
        #self.logger.info("%d object frames found" % len(tab))

        is_obj = ('OBJECT' in self.action.args.ccddata.header['IMTYPE'])


        def optimal_extract_with_error(
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
            weighted_data = data_vector_d * inverse_variance
            numerator = R_transpose @ weighted_data
            R_transpose_squared = R_transpose.power(2)
            denominator = R_transpose_squared @ inverse_variance
            denominator_safe = np.maximum(denominator, 1e-9)
            optimized_flux = numerator / denominator_safe
            flux_variance = 1.0 / denominator_safe
            flux_error = np.sqrt(flux_variance)
            end_time1 = time.time()
            t1 = (end_time1 - start_time1)
            self.logger.info(f"Optimal extraction finished in {t1:.4f} seconds.")
            return optimized_flux, flux_error

        def solve_bounded_weighted_nnls(
            R_matrix: sp.spmatrix,
            data_vector: np.ndarray,
            read_noise_variance_vector: np.ndarray,
            gain: float,
            A_guess: np.ndarray,  
            bound_factor: float = 1.0, 
            tolerance: float = 1e-6
        ) -> np.ndarray:
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

        
        def calculate_error_flux_cube(R_matrix: sp.spmatrix,
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

        def _parse_sky_coord(coord_val: [str, float], is_ra: bool = False) -> float:
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
            cube_shape: tuple,
            header: fits.Header,
            center_map: dict,
            default_center_yx: tuple
        ) -> WCS:
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
                crval_ra = _parse_sky_coord(header['RA'], is_ra=True)
                crval_dec = _parse_sky_coord(header['DEC'])
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


        if is_obj:

            SCALES_CENTER_MAP = {
                'LOWRES-SED': (54, 54),
                'LOWRES-K': (50, 60),
                'LOWRES-L': (50, 60),
                'LOWRES-M': (50, 60),
                'LOWRES-H20': (50, 60),
                'LOWRES-PAH': (50, 60),
                'MEDRES-K': (50, 60),
                'MEDRES-L': (50, 60),
                'MEDRES-M': (50, 60),
            }
    
            SCALES_DEFAULT_CENTER = (54, 54)
            calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
            readnoise = fits.getdata(calib_path+'sim_readnoise.fits')
            var_read_vector = (readnoise.flatten().astype(np.float64))**2
            GAIN = self.action.args.ccddata.header['GAIN']
            IMG_DIM = 2048
            FLUX_SHAPE_3D = (54, 108, 108)
            N_PIXELS = IMG_DIM * IMG_DIM
    
            data_image = self.action.args.ccddata.data
            data_vector_d = data_image.flatten().astype(np.float64)

            R_for_extract = load_npz(calib_path+'QLmat_new.npz')
            R_matrix = load_npz(calib_path+'C2mat_new.npz')
        
            A_guess_cube,A_guess_cube_err = optimal_extract_with_error(R_for_extract, data_image, var_read_vector)
            A_guess_vector = A_guess_cube.flatten()
        
            A_opt = A_guess_cube.reshape(FLUX_SHAPE_3D)
            A_opt_err = A_guess_cube_err.reshape(FLUX_SHAPE_3D)

            A_optimal_nnls = solve_bounded_weighted_nnls(
                R_matrix, data_vector_d, var_read_vector, GAIN, A_guess_vector)

            Amp_chi_square = A_optimal_nnls.reshape(FLUX_SHAPE_3D)

            Amp_chi_square_err = calculate_error_flux_cube(
                R_matrix=R_matrix,
                flux_vector_A=A_optimal_nnls,
                var_read_vector=var_read_vector,
                flux_shape_3d=FLUX_SHAPE_3D,
                gain=GAIN)

            chi_rslt = CCDData(
                data=Amp_chi_square,
                uncertainty=StdDevUncertainty(Amp_chi_square_err),
                meta=self.action.args.ccddata.header,
                unit='adu')

            opt_rslt = CCDData(
                data=A_opt,
                uncertainty=StdDevUncertainty(A_opt_err),
                meta=self.action.args.ccddata.header,
                unit='adu')

            self.action.args.ccddata.data = A_opt

            my_wcs = create_wcs_flexible(
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


            scales_fits_writer(ccddata = chi_rslt, 
                table=self.action.args.table,
                output_file=self.action.args.name,
                output_dir=self.config.instrument.output_directory,
                suffix="chi_cube")
 
            scales_fits_writer(ccddata = opt_rslt, 
                table=self.action.args.table,
                output_file=self.action.args.name,
                output_dir=self.config.instrument.output_directory,
                suffix="opt_cube")

        self.context.proctab.update_proctab(frame=self.action.args.ccddata, suffix="cube", newtype='OBJECT',
                filename=self.action.args.ccddata.header['OFNAME'])
        self.context.proctab.write_proctab(
                tfil=self.config.instrument.procfile)
        return self.action.args
    # END: class LeastExtract()
