from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import scalesdrp.primitives.reference as reference
import scalesdrp.primitives.robust as robust

import pandas as pd
import numpy as np
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pkg_resources
import os
from scipy.optimize import minimize
from scipy import sparse
from scipy.sparse import load_npz
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time
from scipy.optimize import leastsq
from scipy.signal import savgol_filter
from scalesdrp.core.matplot_plotting import mpl_plot, mpl_clear
import matplotlib.pyplot as plt

class QuickLook(BasePrimitive):
    """
	Quicklook  extraction : Quick ramp fitted image and optimal extracted cube.
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger

    def fits_writer_steps(self,data,header,output_dir,input_filename,suffix,overwrite=True):
        base_name = os.path.basename(input_filename)
        file_root, file_ext = os.path.splitext(base_name)
        output_filename = f"{file_root}{suffix}{file_ext}"
        redux_output_dir = os.path.join(output_dir, 'redux')
        os.makedirs(redux_output_dir, exist_ok=True)
        output_path = os.path.join(redux_output_dir, output_filename)
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_path, overwrite=overwrite)
        self.logger.info("+++++++++++ FITS file saved +++++++++++")
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


    ########################################################################

    def _perform(self):
        self.logger.info("+++++++++++ Quicklook Started +++++++++++")

        input_data = self.action.args.name
        output_dir = os.path.dirname(input_data)
        filename = os.path.basename(input_data)

        calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
        SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits') #IFS readnoise map
        read_noise_var = SIG_map_scaled.flatten().astype(np.float64)**2
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
            rmat = sparse.load_npz(calib_path+'bpmat_img.npz')
            slope_filled2 = rmat*np.matrix(slope_filled1.flatten().reshape([np.prod(slope_filled1.shape),1]))
            slope_filled = np.array(slope_filled2).reshape(slope_filled1.shape)
            self.logger.info("BPM correction completed")
            self.fits_writer_steps(
                data=slope_filled,
                header=hdr,
                output_dir=output_dir,
                input_filename=filename,
                suffix='_ql',
                overwrite=True)

        if obs_mode == "IFS":
            self.logger.info("BPM correction started")
            rmat = sparse.load_npz(calib_path+'bpmat_ifs.npz')
            slope_filled2 = rmat*np.matrix(slope_filled1.flatten().reshape([np.prod(slope_filled1.shape),1]))
            slope_filled = np.array(slope_filled2).reshape(slope_filled1.shape)
            self.logger.info("BPM correction completed")
            self.fits_writer_steps(
                data=slope_filled1,
                header=hdr,
                output_dir=output_dir,
                input_filename=filename,
                suffix='_ql',
                overwrite=True)

            if ifs_mode == "LowRes-K":
                print("IFSMODE is", ifs_mode)
                print("IMTYPE is", obj)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "K_QL_rectmat_lowres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    R_matrix = load_npz(calib_path+'K_QL_rectmat_lowres.npz')
                    cube1,error1 = self.optimal_extract_with_error(
                        R_matrix,
                        slope_filled,
                        read_noise_var)

                    cube= cube1.reshape(56, 103, 110)
                    self.fits_writer_steps(
                        data=cube,
                        header=hdr,
                        output_dir=output_dir,
                        input_filename=filename,suffix='_ql_cube',
                        overwrite=True)

            elif ifs_mode == "LowRes-L":
                print("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "L_QL_rectmat_lowres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    R_matrix = load_npz(calib_path+'L_QL_rectmat_lowres.npz')
                    cube1,error1 = self.optimal_extract_with_error(
                        R_matrix,
                        slope_filled,
                        read_noise_var)

                    cube= cube1.reshape(56,103,110)
                    self.fits_writer_steps(
                        data=cube,
                        header=hdr,
                        output_dir=output_dir,
                        input_filename=filename,suffix='_ql_cube',
                        overwrite=True)

            elif ifs_mode == "LowRes-M":
                self.logger.info("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "M_QL_rectmat_lowres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    R_matrix = load_npz(calib_path+'M_QL_rectmat_lowres.npz')
                    cube1,error1 = self.optimal_extract_with_error(
                        R_matrix,
                        slope_filled,
                        read_noise_var)

                    cube= cube1.reshape(56,103,110)
                    self.fits_writer_steps(
                        data=cube,
                        header=hdr,
                        output_dir=output_dir,
                        input_filename=filename,suffix='_ql_cube',
                        overwrite=True)

            elif ifs_mode == "LowRes-SED":
                print("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "SED_QL_rectmat_lowres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):
                    R_matrix = load_npz(calib_path+'SED_QL_rectmat_lowres.npz')
                    cube1,error1 = self.optimal_extract_with_error(
                        R_matrix,
                        slope_filled,
                        read_noise_var)

                    cube= cube1.reshape(56,103,110)
                    self.fits_writer_steps(
                        data=cube,
                        header=hdr,
                        output_dir=output_dir,
                        input_filename=filename,suffix='_ql_cube',
                        overwrite=True)

            elif ifs_mode == "LowRes-KL":
                print("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "KL_QL_rectmat_lowres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    R_matrix = load_npz(calib_path+'KL_QL_rectmat_lowres.npz')
                    cube1,error1 = self.optimal_extract_with_error(
                        R_matrix,
                        slope_filled,
                        read_noise_var)

                    cube= cube1.reshape(54,108,108)
                    self.fits_writer_steps(
                        data=cube,
                        header=hdr,
                        output_dir=output_dir,
                        input_filename=filename,suffix='_ql_cube',
                        overwrite=True)

            elif ifs_mode == "LowRes-PAH":
                print("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "PAH_QL_rectmat_lowres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    R_matrix = load_npz(calib_path+'PAH_QL_rectmat_lowres.npz')
                    cube1,error1 = self.optimal_extract_with_error(
                        R_matrix,
                        slope_filled,
                        read_noise_var)

                    cube= cube1.reshape(54,108,108)
                    self.fits_writer_steps(
                        data=cube,
                        header=hdr,
                        output_dir=output_dir,
                        input_filename=filename,
                        suffix='_ql_cube',
                        overwrite=True)

            elif ifs_mode == "MedRes-K":
                print("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "K_QL_rectmat_medres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    R_matrix = load_npz(calib_path+'K_QL_rectmat_medres.npz')
                    cube1,error1 = self.optimal_extract_with_error(
                        R_matrix,
                        slope_filled,
                        read_noise_var)

                    cube= cube1.reshape(1900,18,17)
                    self.fits_writer_steps(
                        data=cube,
                        header=hdr,
                        output_dir=output_dir,
                        input_filename=filename,suffix='_ql_cube',
                        overwrite=True)

            elif ifs_mode == "MedRes-L":
                print("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "L_QL_rectmat_medres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    R_matrix = load_npz(calib_path+'L_QL_rectmat_medres.npz')
                    cube1,error1 = self.optimal_extract_with_error(
                        R_matrix,
                        slope_filled,
                        read_noise_var)

                    cube= cube1.reshape(1900,18,17)
                    self.fits_writer_steps(
                        data=cube,
                        header=hdr,
                        output_dir=output_dir,
                        input_filename=filename,suffix='_ql_cube',
                        overwrite=True)

            elif ifs_mode == "MedRes-M":
                print("IFSMODE is", ifs_mode)
                if (
                    slope_filled is not None
                    and os.path.exists(os.path.join(calib_path, "M_QL_rectmat_medres.npz"))
                    and (obj == "OBJECT" or obj == "FLATLEN")):

                    R_matrix = load_npz(calib_path+'M_QL_rectmat_medres.npz')
                    cube1,error1 = self.optimal_extract_with_error(
                        R_matrix,
                        slope_filled,
                        read_noise_var)

                    cube= cube1.reshape(1900,18,17)
                    self.fits_writer_steps(
                        data=cube,
                        header=hdr,
                        output_dir=output_dir,
                        input_filename=filename,suffix='_ql_cube',
                        overwrite=True)
        else:
            self.logger.info("Unknown MODE of observation")

        log_string = QuickLook.__module__
        self.logger.info(log_string)

        return self.action.args
    # END: class QuickLook()
