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

    def plot_png_save(self,data,output_dir,input_filename,suffix,overwrite=True):
        base_name = os.path.basename(input_filename)
        file_root, file_ext = os.path.splitext(base_name)
        file_ext = '.png'
        output_filename = f"{file_root}{suffix}{file_ext}"
        plot_output_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plot_output_dir, exist_ok=True)
        output_path = os.path.join(plot_output_dir, output_filename)
        fig = plt.figure(figsize=(8, 8))
        im = plt.imshow(data,origin='lower')
        cbar = plt.colorbar(im, label='DN/s')
        cbar.ax.tick_params(labelsize=14)
        plt.title(f"{file_root}{suffix}", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(output_path)
        #plt.show()
        #self.logger.info("+++++++++++ Image saved +++++++++++")
        return 

    #def plot_png_save(self,data,output_dir,input_filename,suffix):
    #    import threading
    #    print("Thread name:", threading.current_thread().name)
    #    base_name = os.path.basename(input_filename)
    #    file_root,_ = os.path.splitext(base_name)
    #    output_filename = f"{file_root}{suffix}.png"
    #    plot_output_dir = os.path.join(output_dir, 'plots')
    #    os.makedirs(plot_output_dir, exist_ok=True)
    #    output_path = os.path.join(plot_output_dir, output_filename)
    #    fig,ax = plt.subplots(figsize=(8, 8))
    #    im = ax.imshow(data,origin='lower',cmap='viridis')
    #    cbar = plt.colorbar(im, label='DN/s')
    #    cbar.ax.tick_params(labelsize=14)
    #    ax.tick_params(axis="both", labelsize=14)
    #    ax.set_title(f"{file_root}{suffix}", fontsize=14)
    #    mpl_plot(fig=fig,show=True, save=True, filename=output_path)
    #    mpl_clear()
    #    return output_path

    def adaptive_weighted_ramp_fit(self,ramp, read_time, cutoff_frac=0.75, sat_level=4096.0, tile=(256,256)):
        """
        Adaptive weighted ramp fit (thread-safe, DRP-compatible version).
        Processes the ramp in tiles to stay memory- and cache-efficient.
        """
        n_reads, n_rows, n_cols = ramp.shape
        read_times = np.linspace(0, read_time, n_reads, dtype=np.float32)
        slope = np.zeros((n_rows, n_cols), dtype=np.float32)
        eps = 1e-6

        Ty, Tx = tile
        for y0 in range(0, n_rows, Ty):
            y1 = min(n_rows, y0 + Ty)
            for x0 in range(0, n_cols, Tx):
                x1 = min(n_cols, x0 + Tx)
                cube = ramp[:, y0:y1, x0:x1]  # (N, ty, tx)
                ty, tx = cube.shape[1:]
                N = cube.shape[0]

                # Find cutoff index per pixel where counts exceed cutoff_frac * sat_level
                cutoff_mask = cube > (cutoff_frac * sat_level)
                # argmax returns 0 if all False — handle that
                cutoff_idx = np.argmax(cutoff_mask, axis=0)
                cutoff_idx[~np.any(cutoff_mask, axis=0)] = N - 1

                # Build weights centered around cutoff index
                idx = np.arange(N)[:, None, None]
                w = np.exp(-((idx - cutoff_idx)**2) / 8.0).astype(np.float32)

                # Weighted sums
                t = read_times[:, None, None]
                y = cube
                S0  = np.sum(w, axis=0)
                St  = np.sum(w * t, axis=0)
                Stt = np.sum(w * t * t, axis=0)
                Sy  = np.sum(w * y, axis=0)
                Sty = np.sum(w * t * y, axis=0)

                denom = S0 * Stt - St * St
                valid = denom > eps
                local_slope = np.full_like(S0, np.nan, dtype=np.float32)
                local_slope[valid] = (S0[valid] * Sty[valid] - St[valid] * Sy[valid]) / denom[valid]
                slope[y0:y1, x0:x1] = local_slope

        return slope


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

    def iterative_sigma_weighted_ramp_fit4(self,ramp, read_time, gain=3.0, rn=5.0, max_iter=3, tile=(256, 256)):
        n_reads, n_rows, n_cols = ramp.shape
        read_times = np.linspace(0, read_time, n_reads, dtype=np.float32)
        dt = np.mean(np.diff(read_times))
        slope = np.zeros((n_rows, n_cols), dtype=np.float32)
        bias = np.zeros_like(slope)

        ie = np.index_exp[:, :, 0::2]
        io = np.index_exp[:, :, 1::2]

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

    def iterative_sigma_weighted_ramp_fit(
        self, ramp1, read_time, gain=3.0, rn=5.0, max_iter=3, tile=(256, 256), do_swap=True):

        n_reads, n_rows, n_cols = ramp1.shape
        read_times = np.linspace(0, read_time, n_reads, dtype=np.float32)
        dt = np.mean(np.diff(read_times))
        ramp = np.empty_like(ramp1)
        n_amps = 4
        block = n_cols // n_amps
        for a in range(n_amps):
            x0, x1 = a * block, (a + 1) * block
            sub = ramp1[..., x0:x1]
            nsub = sub.shape[-1]
            new_order = []
            for i in range(0, nsub, 2):
                if i + 1 < nsub:
                    new_order.extend([i + 1, i])
                else:
                    new_order.append(i)
            ramp[..., x0:x1] = sub[..., new_order]
        slope = np.zeros((n_rows, n_cols), dtype=np.float32)
        bias = np.zeros_like(slope)
        Ty, Tx = tile
        for y0 in range(0, n_rows, Ty):
            y1 = min(n_rows, y0 + Ty)
            for x0 in range(0, n_cols, Tx):
                x1 = min(n_cols, x0 + Tx)
                cube = ramp[:, y0:y1, x0:x1]
                N, ty, tx = cube.shape
                m_tile = np.zeros((ty, tx), dtype=np.float32)
                b_tile = np.zeros_like(m_tile)
                for col_slice, out_slice in [(np.index_exp[:, :, 0::2], (slice(None), slice(0, None, 2))),
                    (np.index_exp[:, :, 1::2], (slice(None), slice(1, None, 2))),]:

                    subcube = cube[col_slice]
                    if subcube.size == 0:
                        continue
                    for iteration in range(max_iter):
                        sig2 = np.maximum(subcube / gain + rn**2, 1e-6)
                        i = np.arange(subcube.shape[0], dtype=np.float32)[:, None, None]
                        S0 = np.sum(1.0 / sig2, axis=0)
                        S1 = np.sum(i / sig2, axis=0)
                        S2 = np.sum(i**2 / sig2, axis=0)
                        S0x = np.sum(subcube / sig2, axis=0)
                        S1x = np.sum(i * subcube / sig2, axis=0)
                        ibar = S1 / S0
                        mdt = (S1x - ibar * S0x) / np.maximum(S2 - ibar**2 * S0, 1e-8)
                        m = mdt / dt
                        b = S0x / S0 - mdt * ibar
                        subcube = np.clip(b[None, :, :] + m[None, :, :] * i * dt, 0, None)
                    m_tile[out_slice] = m
                    b_tile[out_slice] = b
                slope[y0:y1, x0:x1] = m_tile
                bias[y0:y1, x0:x1] = b_tile
        return slope



    def _perform(self):
        self.logger.info("+++++++++++ Quicklook Started +++++++++++")

        input_data = self.action.args.name
        output_dir = os.path.dirname(input_data)
        filename = os.path.basename(input_data)
        #print(input_data)
        #print(output_dir)
        #print(filename)
        calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
        SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits')
        read_noise_var = SIG_map_scaled.flatten().astype(np.float64)**2
        with fits.open(input_data) as hdul:
            hdr = hdul[0].header
            obs_mode = hdr.get("CAMERA", "")
            ifs_mode = 'LowRes-K'#hdr.get("IFSMODE", "")
            #last_file =  hdr.get("LASTFILE", "")
            read_time = hdr.get("EXPTIME", "")
            #file_name = hdr.get("OFNAME", "")
            #obj =       hdr.get("OBJECT", "")

            NUM_FRAMES_FROM_SCIENCE = hdr.get("NREADS", "")
            print(f"OBSMODE = {obs_mode}")
            slope = None
            n_ext = len(hdul)
            print('number of extension = ',n_ext)
            t0 = time.time()
            if filename == filename:
                if obs_mode == "Im":
                    if n_ext == 1:
                        data_1 = hdul[0].data
                        if data_1 is None:
                            raise ValueError("No data in primary HDU.")
                        if data_1.ndim == 2:
                            self.plot_png_save(
                                data = data_1,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_server',
                                overwrite=True)
                        elif data_1.ndim == 3:
                            img_corr = reference.reffix_hxrg(data_1, nchans=4, fixcol=True)
                            slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                img_corr,read_time=read_time)
                            t1 = time.time()
                            self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                            #self.plot_png_save(
                            #    data = slope_filled,
                            #    output_dir=output_dir,
                            #    input_filename=filename,
                            #    suffix='_quicklook',
                            #    overwrite=True)
                            self.fits_writer_steps(
                                data=slope_filled,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)
                        else:
                            raise ValueError(f"Unexpected data shape: {data_1.shape}")
                    elif n_ext >= 2:
                        img2d = hdul[1].data
                        ramp3d = hdul[0].data
                        if img2d.ndim != 2 or ramp3d.ndim != 3:
                            raise ValueError(f"Expected (2D, 3D) shapes, got {img2d.shape}, {ramp3d.shape}")
                        self.plot_png_save(
                            data = img2d,
                            output_dir=output_dir,
                            input_filename=filename,
                            suffix='_server')
                        img_corr = reference.reffix_hxrg(ramp3d, nchans=4, fixcol=True)
                        self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                        slope_filled = self.iterative_sigma_weighted_ramp_fit(
                            img_corr,read_time=read_time)
                        t1 = time.time()
                        self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                        #self.plot_png_save(
                        #    data = slope_filled,
                        #    output_dir=output_dir,
                        #    input_filename=filename,
                        #    suffix='_quicklook')
                        self.fits_writer_steps(
                            data=slope_filled,
                            header=hdr,
                            output_dir=output_dir,
                            input_filename=filename,
                            suffix='_quicklook',
                            overwrite=True)

                elif obs_mode == "IFS":
                    if ifs_mode == "LowRes-K":
                        print("IFSMODE is", ifs_mode)
                        if n_ext == 1:
                            data_1 = hdul[0].data
                            if data_1 is None:
                                raise ValueError("No data in primary HDU.")
                            if data_1.ndim == 2:
                                self.plot_png_save(
                                    data = data_1,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_server',
                                    overwrite=True)
                            elif data_1.ndim == 3:
                                img_corr = reference.reffix_hxrg(data_1, nchans=4, fixcol=True)
                                self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                                slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                    img_corr,
                                    read_time=read_time)
                                t1 = time.time()
                                self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                                #self.plot_png_save(
                                #    data = slope_filled,
                                #    output_dir=output_dir,
                                #    input_filename=filename,
                                #    suffix='_quicklook',
                                #    overwrite=True)
                                self.fits_writer_steps(
                                    data=slope_filled,
                                    header=hdr,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_quicklook',
                                    overwrite=True)
                            else:
                                raise ValueError(f"Unexpected data shape: {data_1.shape}")
                        elif n_ext >= 2:
                            img2d = hdul[1].data
                            ramp3d = hdul[0].data
                            if img2d.ndim != 2 or ramp3d.ndim != 3:
                                raise ValueError(f"Expected (2D, 3D) shapes, got {img2d.shape}, {ramp3d.shape}")
                            #self.plot_png_save(
                            #    data = img2d,
                            #    output_dir=output_dir,
                            #    input_filename=filename,
                            #    suffix='_server',
                            #    overwrite=True)
                            img_corr = reference.reffix_hxrg(ramp3d, nchans=4, fixcol=True)
                            self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                            slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                img_corr,
                                read_time=read_time)
                            t1=time.time()
                            self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                            #self.plot_png_save(
                            #    data = slope_filled,
                            #    output_dir=output_dir,
                            #    input_filename=filename,
                            #    suffix='_quicklook',
                            #    overwrite=True)
                            self.fits_writer_steps(
                                data=slope_filled,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)
                        if (
                            slope_filled is not None
                            and os.path.exists(os.path.join(calib_path, "LowRes-K_QL_rectmat.npz"))
                            and (obj == "SCIENCE" or obj == "FLATLEN")):

                            R_matrix = load_npz(calib_path+'LowRes-K_QL_rectmat.npz')
                            print("Quicklook optimal extraction started for",ifs_mode)
                            cube1,error1 = self.optimal_extract_with_error(R_matrix,slope_filled,read_noise_var)
                            cube= cube1.reshape(54,108,108)
                            self.fits_writer_steps(
                                data=cube,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quick_cube',
                                overwrite=True)
                    elif ifs_mode == "LowRes-L":
                        print("IFSMODE is", ifs_mode)
                        if n_ext == 1:
                            data_1 = hdul[0].data
                            if data_1 is None:
                                raise ValueError("No data in primary HDU.")
                            if data_1.ndim == 2:
                                self.plot_png_save(
                                    data = data_1,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_server',
                                    overwrite=True)
                            elif data_1.ndim == 3:
                                img_corr = reference.reffix_hxrg(data_1, nchans=4, fixcol=True)
                                self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                                slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                    img_corr,
                                    read_time=read_time)
                                t1=time.time()
                                self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                                #self.plot_png_save(
                                #    data = slope_filled,
                                #    output_dir=output_dir,
                                #    input_filename=filename,
                                #    suffix='_quicklook',
                                #    overwrite=True)
                                self.fits_writer_steps(
                                    data=slope_filled,
                                    header=hdr,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_quicklook',
                                    overwrite=True)
                            else:
                                raise ValueError(f"Unexpected data shape: {data_1.shape}")
                        elif n_ext >= 2:
                            img2d = hdul[1].data
                            ramp3d = hdul[0].data
                            if img2d.ndim != 2 or ramp3d.ndim != 3:
                                raise ValueError(f"Expected (2D, 3D) shapes, got {img2d.shape}, {ramp3d.shape}")
                            self.plot_png_save(
                                data = img2d,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_server',
                                overwrite=True)
                            img_corr = reference.reffix_hxrg(ramp3d, nchans=4, fixcol=True)
                            self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                            slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                img_corr,
                                read_time=read_time)
                            t1=time.time()
                            self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                            self.plot_png_save(
                                data = slope_filled,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)
                            self.fits_writer_steps(
                                data=slope_filled,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)

                        if (
                            slope_filled is not None
                            and os.path.exists(os.path.join(calib_path, "LowRes-L_QL_rectmat.npz"))
                            and (obj == "SCIENCE" or obj == "FLATLEN")):

                            R_matrix = load_npz(calib_path+'LowRes-L_QL_rectmat.npz')
                            print("Quicklook optimal extraction started for",ifs_mode)
                            cube1,error1 = self.optimal_extract_with_error(R_matrix,slope_filled,read_noise_var)
                            cube= cube1.reshape(54,108,108)
                            self.fits_writer_steps(
                                data=cube,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,suffix='_quick_cube',
                                overwrite=True)

                    elif ifs_mode == "LowRes-M":
                        print("IFSMODE is", ifs_mode)
                        if n_ext == 1:
                            data_1 = hdul[0].data
                            if data_1 is None:
                                raise ValueError("No data in primary HDU.")
                            if data_1.ndim == 2:
                                self.plot_png_save(
                                    data = data_1,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_server',
                                    overwrite=True)
                            elif data_1.ndim == 3:
                                img_corr = reference.reffix_hxrg(data_1, nchans=4, fixcol=True)
                                self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                                slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                    img_corr,
                                    read_time=read_time)
                                t1=time.time()
                                self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                                #self.plot_png_save(
                                #    data = slope_filled,
                                #    output_dir=output_dir,
                                #    input_filename=filename,
                                #    suffix='_quicklook',
                                #    overwrite=True)
                                self.fits_writer_steps(
                                    data=slope_filled,
                                    header=hdr,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_quicklook',
                                    overwrite=True)
                            else:
                                raise ValueError(f"Unexpected data shape: {data_1.shape}")
                        elif n_ext >= 2:
                            img2d = hdul[1].data
                            ramp3d = hdul[0].data
                            if img2d.ndim != 2 or ramp3d.ndim != 3:
                                raise ValueError(f"Expected (2D, 3D) shapes, got {img2d.shape}, {ramp3d.shape}")
                            self.plot_png_save(
                                data = img2d,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_server',
                                overwrite=True)
                            img_corr = reference.reffix_hxrg(ramp3d, nchans=4, fixcol=True)
                            self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                            slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                img_corr,
                                read_time=read_time)
                            t1=time.time()
                            self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                            #self.plot_png_save(
                            #    data = slope_filled,
                            #    output_dir=output_dir,
                            #    input_filename=filename,
                            #    suffix='_quicklook',
                            #    overwrite=True)
                            self.fits_writer_steps(
                                data=slope_filled,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)

                        if (
                            slope_filled is not None
                            and os.path.exists(os.path.join(calib_path, "LowRes-M_QL_rectmat.npz"))
                            and (obj == "SCIENCE" or obj == "FLATLEN")):

                            R_matrix = load_npz(calib_path+'LowRes-M_QL_rectmat.npz')
                            print("Quicklook optimal extraction started for",ifs_mode)
                            cube1,error1 = self.optimal_extract_with_error(R_matrix,slope_filled,read_noise_var)
                            cube= cube1.reshape(54,108,108)
                            self.fits_writer_steps(
                                data=cube,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quick_cube',
                                overwrite=True)
                    elif ifs_mode == "LowRes-KLM":
                        print("IFSMODE is", ifs_mode)
                        if n_ext == 1:
                            data_1 = hdul[0].data
                            if data_1 is None:
                                raise ValueError("No data in primary HDU.")
                            if data_1.ndim == 2:
                                self.plot_png_save(
                                    data = data_1,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_server',
                                    overwrite=True)
                            elif data_1.ndim == 3:
                                img_corr = reference.reffix_hxrg(data_1, nchans=4, fixcol=True)
                                self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                                slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                    img_corr,
                                    read_time=read_time)
                                t1  = time.time()
                                self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                                self.plot_png_save(
                                    data = slope_filled,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_quicklook',
                                    overwrite=True)
                                self.fits_writer_steps(
                                    data=slope_filled,
                                    header=hdr,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_quicklook',
                                    overwrite=True)
                            else:
                                raise ValueError(f"Unexpected data shape: {data_1.shape}")
                        elif n_ext >= 2:
                            img2d = hdul[1].data
                            ramp3d = hdul[0].data
                            if img2d.ndim != 2 or ramp3d.ndim != 3:
                                raise ValueError(f"Expected (2D, 3D) shapes, got {img2d.shape}, {ramp3d.shape}")
                            self.plot_png_save(
                                data = img2d,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_server',
                                overwrite=True)
                            img_corr = reference.reffix_hxrg(ramp3d, nchans=4, fixcol=True)
                            self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                            slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                img_corr,
                                read_time=read_time)
                            t1=time.time()
                            self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                            #self.plot_png_save(
                            #    data = slope_filled,
                            #    output_dir=output_dir,
                            #    input_filename=filename,
                            #    suffix='_quicklook',
                            #    overwrite=True)
                            self.fits_writer_steps(
                                data=slope_filled,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)

                        if (
                            slope_filled is not None
                            and os.path.exists(os.path.join(calib_path, "LowRes-KLM_QL_rectmat.npz"))
                            and (obj == "SCIENCE" or obj == "FLATLEN")):

                            R_matrix = load_npz(calib_path+'LowRes-KLM_QL_rectmat.npz')
                            print("Quicklook optimal extraction started for",ifs_mode)
                            cube1,error1 = self.optimal_extract_with_error(R_matrix,slope_filled,read_noise_var)
                            cube= cube1.reshape(54,108,108)
                            self.fits_writer_steps(
                                data=cube,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quick_cube',
                                overwrite=True)

                    elif ifs_mode == "LowRes-KL":
                        print("IFSMODE is", ifs_mode)
                        if n_ext == 1:
                            data_1 = hdul[0].data
                            if data_1 is None:
                                raise ValueError("No data in primary HDU.")
                            if data_1.ndim == 2:
                                self.plot_png_save(
                                    data = data_1,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_server',
                                    overwrite=True)
                            elif data_1.ndim == 3:
                                img_corr = reference.reffix_hxrg(data_1, nchans=4, fixcol=True)
                                self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                                slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                    img_corr,
                                    read_time=read_time)
                                t1=time.time()
                                self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                                #self.plot_png_save(
                                #    data = slope_filled,
                                #    output_dir=output_dir,
                                #    input_filename=filename,
                                #    suffix='_quicklook',
                                #   overwrite=True)
                                self.fits_writer_steps(
                                    data=slope_filled,
                                    header=hdr,
                                    output_dir=output_dir,
                                    input_filename=filename,suffix='_quicklook',
                                    overwrite=True)
                            else:
                                raise ValueError(f"Unexpected data shape: {data_1.shape}")
                        elif n_ext >= 2:
                            img2d = hdul[1].data
                            ramp3d = hdul[0].data
                            if img2d.ndim != 2 or ramp3d.ndim != 3:
                                raise ValueError(f"Expected (2D, 3D) shapes, got {img2d.shape}, {ramp3d.shape}")
                            self.plot_png_save(
                                data = img2d,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_server',
                                overwrite=True)
                            img_corr = reference.reffix_hxrg(ramp3d, nchans=4, fixcol=True)
                            self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                            slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                img_corr,
                                read_time=read_time)
                            t1=time.time()
                            self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                            self.plot_png_save(
                                data = slope,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)
                            self.fits_writer_steps(
                                data=slope,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)

                        if (
                            slope_filled is not None
                            and os.path.exists(os.path.join(calib_path, "LowRes-KL_QL_rectmat.npz"))
                            and (obj == "SCIENCE" or obj == "FLATLEN")):

                            R_matrix = load_npz(calib_path+'LowRes-KL_QL_rectmat.npz')
                            print("Quicklook optimal extraction started for",ifs_mode)
                            cube1,error1 = self.optimal_extract_with_error(R_matrix,slope_filled,read_noise_var)
                            cube= cube1.reshape(54,108,108)
                            self.fits_writer_steps(
                                data=cube,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quick_cube',
                                overwrite=True)
                    elif ifs_mode == "LowRes-Ls":
                        print("IFSMODE is", ifs_mode)
                        if n_ext == 1:
                            data_1 = hdul[0].data
                            if data_1 is None:
                                raise ValueError("No data in primary HDU.")
                            if data_1.ndim == 2:
                                self.plot_png_save(
                                    data = data_1,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_server',
                                    overwrite=True)
                            elif data_1.ndim == 3:
                                img_corr = reference.reffix_hxrg(data_1, nchans=4, fixcol=True)
                                self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                                slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                    img_corr,
                                    read_time=read_time)
                                t1=time.time()
                                self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                                #self.plot_png_save(
                                #    data = slope_filled,
                                #    output_dir=output_dir,
                                #    input_filename=filename,
                                #    suffix='_quicklook',
                                #    overwrite=True)
                                self.fits_writer_steps(
                                    data=slope_filled,
                                    header=hdr,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_quicklook',
                                    overwrite=True)
                            else:
                                raise ValueError(f"Unexpected data shape: {data_1.shape}")
                        elif n_ext >= 2:
                            img2d = hdul[0].data
                            ramp3d = hdul[1].data
                            if img2d.ndim != 2 or ramp3d.ndim != 3:
                                raise ValueError(f"Expected (2D, 3D) shapes, got {img2d.shape}, {ramp3d.shape}")
                            self.plot_png_save(
                                data = img2d,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_server',
                                overwrite=True)
                            img_corr = reference.reffix_hxrg(ramp3d, nchans=4, fixcol=True)
                            self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                            slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                img_corr,
                                read_time=read_time)
                            t1=time.time()
                            self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                            self.plot_png_save(
                                data = slope,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)
                            self.fits_writer_steps(
                                data=slope,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)

                        if (
                            slope_filled is not None
                            and os.path.exists(os.path.join(calib_path, "LowRes-Ls_QL_rectmat.npz"))
                            and (obj == "SCIENCE" or obj == "FLATLEN")):

                            R_matrix = load_npz(calib_path+'LowRes-Ls_QL_rectmat.npz')
                            print("Quicklook optimal extraction started for",ifs_mode)
                            cube1,error1 = self.optimal_extract_with_error(R_matrix,slope_filled,read_noise_var)
                            cube= cube1.reshape(54,108,108)
                            self.fits_writer_steps(
                                data=cube,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quick_cube',
                                overwrite=True)

                elif obs_mode == "MEDRES":
                    if ifs_mode == "MedRes-K":
                        print("IFSMODE is", ifs_mode)
                        if n_ext == 1:
                            data_1 = hdul[0].data
                            if data_1 is None:
                                raise ValueError("No data in primary HDU.")
                            if data_1.ndim == 2:
                                self.plot_png_save(
                                    data = data_1,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_server',
                                    overwrite=True)
                            elif data_1.ndim == 3:
                                img_corr = reference.reffix_hxrg(data_1, nchans=4, fixcol=True)
                                self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                                slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                    img_corr,
                                    read_time=read_time)
                                t1=time.time()
                                self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                                self.plot_png_save(
                                    data = slope_filled,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_quicklook',
                                    overwrite=True)
                                self.fits_writer_steps(
                                    data=slope_filled,
                                    header=hdr,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_quicklook',
                                    overwrite=True)
                            else:
                                raise ValueError(f"Unexpected data shape: {data_1.shape}")
                        elif n_ext >= 2:
                            img2d = hdul[1].data
                            ramp3d = hdul[0].data
                            if img2d.ndim != 2 or ramp3d.ndim != 3:
                                raise ValueError(f"Expected (2D, 3D) shapes, got {img2d.shape}, {ramp3d.shape}")
                            self.plot_png_save(
                                data = img2d,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_server',
                                overwrite=True)
                            img_corr = reference.reffix_hxrg(ramp3d, nchans=4, fixcol=True)
                            self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                            slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                img_corr,
                                read_time=read_time)
                            t1 = time.time()
                            self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                            self.plot_png_save(
                                data = slope_filled,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)
                            self.fits_writer_steps(
                                data=slope_filled,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,suffix='_quicklook',
                                overwrite=True)

                        #if slope is not None and os.path.exists(os.path.join(calib_path, "QLmat_new.npz")):
                        #    R_matrix = load_npz(calib_path+'QLmat_new.npz')
                        #    print("Quicklook optimal extraction started for",ifs_mode)
                        #    cube1,error1 = self.optimal_extract_with_error(R_matrix,slope,read_noise_var)
                        #    cube= cube1.reshape(54,108,108)
                        #    self.fits_writer_steps(
                        #        data=cube,
                        #        header=hdr,
                        #        output_dir=output_dir,
                        #        input_filename=filename,suffix='_quick_cube',
                        #        overwrite=True)
                    if ifs_mode == "MedRes-L":
                        print("IFSMODE is", ifs_mode)
                        if n_ext == 1:
                            data_1 = hdul[0].data
                            if data_1 is None:
                                raise ValueError("No data in primary HDU.")
                            if data_1.ndim == 2:
                                self.plot_png_save(
                                    data = data_1,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_server',
                                    overwrite=True)
                            elif data_1.ndim == 3:
                                img_corr = reference.reffix_hxrg(data_1, nchans=4, fixcol=True)
                                self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                                slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                    img_corr,
                                    read_time=read_time)
                                t1=time.time()
                                self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                                self.plot_png_save(
                                    data = slope_filled,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_quicklook',
                                    overwrite=True)
                                self.fits_writer_steps(
                                    data=slope_filled,
                                    header=hdr,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_quicklook',
                                    overwrite=True)
                            else:
                                raise ValueError(f"Unexpected data shape: {data_1.shape}")
                        elif n_ext >= 2:
                            img2d = hdul[1].data
                            ramp3d = hdul[0].data
                            if img2d.ndim != 2 or ramp3d.ndim != 3:
                                raise ValueError(f"Expected (2D, 3D) shapes, got {img2d.shape}, {ramp3d.shape}")
                            self.plot_png_save(
                                data = img2d,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_server',
                                overwrite=True)
                            img_corr = reference.reffix_hxrg(ramp3d, nchans=4, fixcol=True)
                            self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                            slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                img_corr,
                                read_time=read_time)
                            t1=time.time()
                            self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                            self.plot_png_save(
                                data = slope_filled,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)
                            self.fits_writer_steps(
                                data=slope_filled,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)

                        #if slope is not None and os.path.exists(os.path.join(calib_path, "QLmat_new.npz")):
                        #    R_matrix = load_npz(calib_path+'QLmat_new.npz')
                        #    print("Quicklook optimal extraction started for",ifs_mode)
                        #    cube1,error1 = self.optimal_extract_with_error(R_matrix,slope,read_noise_var)
                        #    cube= cube1.reshape(54,108,108)
                        #    self.fits_writer_steps(
                        #        data=cube,
                        #        header=hdr,
                        #        output_dir=output_dir,
                        #        input_filename=filename,suffix='_quick_cube',
                        #        overwrite=True)
                    if ifs_mode == "MedRes-M":
                        print("IFSMODE is", ifs_mode)
                        if n_ext == 1:
                            data_1 = hdul[0].data
                            if data_1 is None:
                                raise ValueError("No data in primary HDU.")
                            if data_1.ndim == 2:
                                self.plot_png_save(
                                    data = data_1,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_server',
                                    overwrite=True)
                            elif data_1.ndim == 3:
                                img_corr = reference.reffix_hxrg(data_1, nchans=4, fixcol=True)
                                self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                                slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                    img_corr,
                                    read_time=read_time)
                                t1=time.time()
                                self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                                self.plot_png_save(
                                    data = slope_filled,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_quicklook',
                                    overwrite=True) 
                                self.fits_writer_steps(
                                    data=slope_filled,
                                    header=hdr,
                                    output_dir=output_dir,
                                    input_filename=filename,
                                    suffix='_quicklook',
                                    overwrite=True)
                            else:
                                raise ValueError(f"Unexpected data shape: {data_1.shape}")
                        elif n_ext >= 2:
                            img2d = hdul[1].data
                            ramp3d = hdul[0].data
                            if img2d.ndim != 2 or ramp3d.ndim != 3:
                                raise ValueError(f"Expected (2D, 3D) shapes, got {img2d.shape}, {ramp3d.shape}")
                            self.plot_png_save(
                                data = img2d,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_server',
                                overwrite=True)
                            img_corr = reference.reffix_hxrg(ramp3d, nchans=4, fixcol=True)
                            self.logger.info("+++++++++++ ACN & 1/f Correction applied +++++++++++")
                            slope_filled = self.iterative_sigma_weighted_ramp_fit(
                                img_corr,
                                read_time=read_time)
                            t1=time.time()
                            self.logger.info(f"Ramp fitting finished in {t1-t0:.2f} seconds.")
                            self.plot_png_save(
                                data = slope_filled,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)
                            self.fits_writer_steps(
                                data=slope_filled,
                                header=hdr,
                                output_dir=output_dir,
                                input_filename=filename,
                                suffix='_quicklook',
                                overwrite=True)

                        #if slope is not None and os.path.exists(os.path.join(calib_path, "QLmat_new.npz")):
                        #    R_matrix = load_npz(calib_path+'QLmat_new.npz')
                        #    print("Quicklook optimal extraction started for",ifs_mode)
                        #    cube1,error1 = self.optimal_extract_with_error(R_matrix,slope,read_noise_var)
                        #    cube= cube1.reshape(54,108,108)
                        #    self.fits_writer_steps(
                        #        data=cube,
                        #        header=hdr,
                        #        output_dir=output_dir,
                        #        input_filename=filename,suffix='_quick_cube',
                        #        overwrite=True)
                else:
                    raise ValueError(f"Unknown OBSMODE: {obs_mode}")

            else:
                self.logger.info("+++++++++ Waiting for the fits file to finish readout ++++++++")
        self.logger.info("+++++++++++ Quicklook Completed for the current FITS file +++++++++++")
        log_string = QuickLook.__module__
        #self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)

        return self.action.args
    # END: class QuickLook()
