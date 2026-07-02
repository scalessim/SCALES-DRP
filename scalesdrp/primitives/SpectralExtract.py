from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer

import numpy as np
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
import scalesdrp.primitives.scales_basic as scbasic
from scalesdrp.core.scales_proctab import Proctab
from scalesdrp.core.scales_pkg_resources import get_resource_path
import logging
log = logging.getLogger("SCALES")
pt = Proctab(logger=log)

class SpectralExtract(BasePrimitive):
    """
	This primitive will perform spectral cube extraction using Optimal extraction and
    the chi square extraction method. A linear WCS informations are updated to the 
    the final output header. More details are listed below.
    Args:
        data_image: The (H,W) input slope image & uncertainty.
            
    Returns:
        A 3D cube with two spatial and one spectral dimension
        A 3D uncertainty cube
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger

        if not hasattr(self, "proctab") or self.proctab is None:
            self.proctab = Proctab(logger=self.logger if hasattr(self, "logger") else logging.getLogger("SCALES"))


    def _perform(self):

        obsmode = self.action.args.ccddata.header['CAMERA']

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
            package = __name__.split('.')[0]
            filepath = 'calib/'
            calfile = 'sim_readnoise.fits'
            calib_path = str(get_resource_path(package, filepath))+'/'
            readnoise = fits.getdata(calib_path+calfile)
            #var_read_vector = (readnoise.flatten().astype(np.float64))**2
            sigma_image = self.action.args.ccddata.uncertainty
            var_read_vector = (sigma_image.array.flatten().astype(np.float64))**2+(readnoise.flatten().astype(np.float64))**2
            GAIN = 1.0#self.action.args.ccddata.header['GAIN']

            data_image = self.action.args.ccddata.data
            data_vector_d = data_image.flatten().astype(np.float64)
            sigma_image = self.action.args.ccddata.uncertainty

            ifsmode = self.action.args.ccddata.header['IFSMODE']
            print(ifsmode)
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
                R_for_extract = load_npz(calib_path+'C2_rmat_k_260604.npz')
                R_matrix = load_npz(calib_path+'ql_rmat_k_260604.npz')
                FLUX_SHAPE_3D = (179, 17, 18)
            elif ifsmode=='MedRes-L':
                R_for_extract = load_npz(calib_path+'L_C2_rectmat_medres.npz')
                R_matrix = load_npz(calib_path+'L_QL_rectmat_medres.npz')
                FLUX_SHAPE_3D = (1900, 17, 18)
            elif ifsmode=='MedRes-M':
                R_for_extract = load_npz(calib_path+'M_C2_rectmat_medres.npz')
                R_matrix = load_npz(calib_path+'M_QL_rectmat_medres.npz')
                FLUX_SHAPE_3D = (1900, 17, 18)

            filename = self.action.args.ccddata.header.get("OFNAME")

            existing_l1_name = scbasic.find_existing_proc_file(
                input_filename=filename,
                suffix="_opt_L2",
                redux_dir=self.config.instrument.output_directory)
            
            if existing_l1_name is not None:
                l1_path = os.path.join(
                    self.config.instrument.output_directory,
                    os.path.basename(existing_l1_name))
            else:
                l1_path = scbasic.get_l2_path_from_raw(
                    input_filename = filename,
                    output_dir = self.config.instrument.output_directory)    

            if os.path.exists(l1_path):
                self.logger.info(f"Found existing L2 file: {l1_path}")
                try:
                    l1_slope, l1_uncert, l1_header = scbasic.read_existing_l2(l1_path)
                    self.action.args.ccddata.data = l1_slope
                    self.action.args.ccddata.header = l1_header
                    self.action.args.ccddata.uncertainty = StdDevUncertainty(l1_uncert)

                    self.logger.info(f"Reusing existing L2 for {filename}. Skipping raw processing.")
                    
                    return self.action.args

                except Exception as e:
                    self.logger.warning(
                                f"Existing L2 file could not be used: {l1_path}. "
                                f"Reason: {e}. Reprocessing from raw file.")

            A_guess_cube,A_guess_cube_err = scbasic.optimal_extract_with_error(
                R_matrix,
                data_image,
                sigma_image,
                var_read_vector)

            A_guess_vector = A_guess_cube.flatten()
        
            A_opt = A_guess_cube.reshape(FLUX_SHAPE_3D)
            A_opt_err = A_guess_cube_err.reshape(FLUX_SHAPE_3D)

            A_optimal_nnls = scbasic.solve_bounded_weighted_nnls(
                R_for_extract, data_vector_d, var_read_vector, GAIN, A_guess_vector)

            Amp_chi_square = A_optimal_nnls.reshape(FLUX_SHAPE_3D)

            Amp_chi_square_err = scbasic.calculate_error_flux_cube(
                R_matrix=R_for_extract,
                flux_vector_A=A_optimal_nnls,
                var_read_vector=var_read_vector,
                flux_shape_3d=FLUX_SHAPE_3D,
                gain=GAIN)

            norm_flatlens,norm_flatlens_uncert = scbasic.load_and_normalize_lenslet_flat(ifsmode)

            A_opt, A_opt_err = scbasic.apply_flatlens(
                A_opt,
                A_opt_err,
                norm_flatlens,
                norm_flatlens_uncert,
                imtype='FLATLENS')

            Amp_chi_square, Amp_chi_square_err = scbasic.apply_flatlens(
                Amp_chi_square,
                Amp_chi_square_err,
                norm_flatlens,
                norm_flatlens_uncert,
                imtype='FLATLENS')

            wcs, wave_info = scbasic.create_scales_wcs(
                cube_shape=A_opt.shape,
                header=self.action.args.ccddata.header)
            
            final_header = scbasic.wcs_header_update(
                data_cube=A_opt,
                input_header=self.action.args.ccddata.header,
                wcs=wcs,
                wave_info=wave_info)
            
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

            self.action.args.ccddata.header.update(final_header)

            log_string = SpectralExtract.__module__
            self.action.args.ccddata.header['HISTORY'] = log_string
            self.action.args.ccddata.header['HISTORY'] = 'WCS keywords updated (purely linear).'
            self.action.args.ccddata.header['HISTORY'] = 'Spectral extraction performed using default rectmat.'
            self.logger.info(log_string)


            scales_fits_writer(ccddata = chi_rslt, 
                table=self.action.args.table,
                output_file=self.action.args.name,
                output_dir=self.config.instrument.output_directory,
                suffix="chi_L2")
            
            scbasic.proctab_update(
                header=self.action.args.ccddata.header,
                output_dir=self.config.instrument.output_directory,
                input_filename=self.action.args.name,
                suffix="_chi_L2",
                frame=None,
                proctab=self.proctab)

            scales_fits_writer(ccddata = opt_rslt, 
                table=self.action.args.table,
                output_file=self.action.args.name,
                output_dir=self.config.instrument.output_directory,
                suffix="opt_L2")

            scbasic.proctab_update(
                header=self.action.args.ccddata.header,
                output_dir=self.config.instrument.output_directory,
                input_filename=self.action.args.name,
                suffix="_opt_L2",
                frame=None,
                proctab=self.proctab)

        return self.action.args
    # END: class LeastExtract()
