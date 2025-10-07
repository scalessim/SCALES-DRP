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
from scipy.ndimage import distance_transform_edt, gaussian_filter

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
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(output_path)
        #plt.show()
        #self.logger.info("+++++++++++ Image saved +++++++++++")
        return 

    def masked_smooth_fast(self,filled, nanmask, sigma=1.25):
        """Smooth only inside bbox around NaNs; keep valid pixels fixed."""
        out = np.asarray(filled, dtype=np.float32).copy()
        (ys, xs), ok = self._bbox(nanmask)
        if not ok:
            return out
        submask = nanmask[ys, xs]
        subimg  = out[ys, xs]
        sm = gaussian_filter(subimg, sigma=sigma, mode='nearest')
        dist_inside = distance_transform_edt(submask.astype(np.uint8))
        blend = np.clip(dist_inside / (3.0 * sigma), 0.0, 1.0).astype(np.float32)
        subimg[submask] = (1 - blend[submask]) * subimg[submask] + blend[submask] * sm[submask]
        out[ys, xs] = subimg
        #self.logger.info("Ramp fitting completed")
        return out

    def inpaint_nearest_fast(self,img, nanmask):
        """Do the EDT only on the small bbox around NaNs."""
        out = np.asarray(img, dtype=np.float32).copy()
        (ys, xs), ok = self._bbox(nanmask)
        if not ok:
            return out
        submask = nanmask[ys, xs]
        subimg  = out[ys, xs]
        _, (iy, ix) = distance_transform_edt(submask, return_indices=True)
        subimg[submask] = subimg[iy[submask], ix[submask]]
        out[ys, xs] = subimg
        return out

    def _bbox(self,mask, pad=16):
        ys, xs = np.where(mask)
        if ys.size == 0:
            return (slice(0,0), slice(0,0)), False
        y0 = max(0, ys.min() - pad)
        y1 = min(mask.shape[0], ys.max() + pad + 1)
        x0 = max(0, xs.min() - pad)
        x1 = min(mask.shape[1], xs.max() + pad + 1)
        return (slice(y0, y1), slice(x0, x1)), True

    def slope_linear(self,
        ims, *,
        read_time=None, times=None,
        sat_thresh=4096.0,   # ADC full-well (DN)
        nl_frac=0.7,         # Use only DN < nl_frac * sat_thresh (e.g. 0.6–0.8)
        drop_first_n=3,      # Skip early unstable reads
        min_reads=6,         # Require at least this many valid reads
        tile=(256, 256),     # Process in spatial tiles to reduce RAM
        dtype=np.float32,):
        """
        Per-pixel linear fit y(t) = a + b*t using only the *strictly-linear* portion
        of each pixel's ramp. Works for large 3D cubes efficiently.

        Returns
        -------
        slope : (Y,X) array, float32
            Fitted slope (DN/s)
        nanmask : (Y,X) boolean array
            True where the slope could not be reliably fit
        """

        ims = np.asarray(ims, dtype=dtype)
        assert ims.ndim == 3, "ims must be (N, Y, X)"
        N0, Y, X = ims.shape

        # --- Time axis in seconds ---
        if times is None:
            if read_time is None:
                raise ValueError("Provide read_time or times")
            t = np.arange(N0, dtype=np.float32) * float(read_time)
        else:
            t = np.asarray(times, dtype=np.float32)
            if t.shape != (N0,):
                raise ValueError("times must have shape (N,)")

        # Drop first few reads (settling)
        if drop_first_n > 0:
            ims = ims[drop_first_n:]
            t = t[drop_first_n:]
        N = ims.shape[0]

        # Precompute time powers
        t1 = t
        t2 = t1 * t1
        lin_dn = None if (sat_thresh is None or nl_frac is None) else np.float32(nl_frac * sat_thresh)

        # Prepare outputs
        slope = np.full((Y, X), np.nan, dtype=np.float32)
        nanmask = np.ones((Y, X), dtype=bool)

        # --- Tile loop to keep memory low ---
        Ty, Tx = tile
        for y0 in range(0, Y, Ty):
            y1 = min(Y, y0 + Ty)
            for x0 in range(0, X, Tx):
                x1 = min(X, x0 + Tx)

                cube = ims[:, y0:y1, x0:x1]     # (N, ty, tx)
                ty, tx = cube.shape[1:]
                k = ty * tx
                y = cube.reshape(N, k)          # (N, k)

                # Valid samples: finite and below nonlinearity DN
                valid = np.isfinite(y)
                if lin_dn is not None:
                    valid &= (y < lin_dn)

                w = valid.astype(np.float32)

                # Need enough valid reads
                S0 = w.sum(axis=0)
                good = (S0 >= min_reads)
                if not np.any(good):
                    nanmask[y0:y1, x0:x1] = True
                    continue

                # Weighted sums
                St  = t1 @ w                   # Σ w t
                Stt = t2 @ w                   # Σ w t²
                wy  = w * y
                Sy  = wy.sum(axis=0)           # Σ w y
                Sty = t1 @ wy                  # Σ w t y

                # Centered linear fit per pixel
                Var_t = Stt - (St * St) / np.maximum(S0, 1)
                Cov_ty = Sty - (St * Sy) / np.maximum(S0, 1)

                b = np.full(k, np.nan, dtype=np.float32)
                valid_fit = good & (Var_t > 0)
                b[valid_fit] = Cov_ty[valid_fit] / Var_t[valid_fit]

                slope[y0:y1, x0:x1] = b.reshape(ty, tx)
                nanmask[y0:y1, x0:x1] = ~valid_fit.reshape(ty, tx)

        return slope, nanmask


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
            obs_mode = hdr.get("OBSMODE", "")
            ifs_mode = hdr.get("IFSMODE", "")
            last_file =  hdr.get("LASTFILE", "")
            read_time = hdr.get("READTIME", "")
            file_name = hdr.get("OFNAME", "")

            NUM_FRAMES_FROM_SCIENCE = hdr.get("NREADS", "")
            print(f"OBSMODE = {obs_mode}")
            slope = None
            n_ext = len(hdul)
            print('number of extension = ',n_ext)
            t0 = time.time()
            if file_name == filename:
                if obs_mode == "IMAGING":
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
                            slope, nanmask = self.slope_linear(
                                img_corr,
                                read_time=read_time,
                                sat_thresh=4000.0,
                                nl_frac=0.7,
                                drop_first_n=3,
                                min_reads=6,
                                tile=(512, 512))

                            filled = self.inpaint_nearest_fast(slope, nanmask)
                            slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                        slope, nanmask = self.slope_linear(
                            img_corr,
                            read_time=read_time,
                            sat_thresh=4000.0,
                            nl_frac=0.7,
                            drop_first_n=3,
                            min_reads=6,
                            tile=(512, 512))
                        filled = self.inpaint_nearest_fast(slope, nanmask)
                        slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                            input_filename=filename,
                            suffix='_quicklook',
                            overwrite=True)

                elif obs_mode == "LOWRES":
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
                                slope, nanmask = self.slope_linear(
                                    img_corr,
                                    read_time=read_time,
                                    sat_thresh=4000.0,
                                    nl_frac=0.7,
                                    drop_first_n=3,
                                    min_reads=6,
                                    tile=(512, 512))
                                filled = self.inpaint_nearest_fast(slope, nanmask)
                                slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                            slope, nanmask = self.slope_linear(
                                img_corr,
                                read_time=read_time,
                                sat_thresh=4000.0,
                                nl_frac=0.7,
                                drop_first_n=3,
                                min_reads=6,
                                tile=(512, 512))
                            filled = self.inpaint_nearest_fast(slope, nanmask)
                            slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                        if slope is not None and os.path.exists(os.path.join(calib_path, "QLmat_new.npz")):
                            R_matrix = load_npz(calib_path+'QLmat_new.npz')
                            print("Quicklook optimal extraction started for",ifs_mode)
                            cube1,error1 = self.optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                                slope, nanmask = self.slope_linear(
                                    img_corr,
                                    read_time=read_time,
                                    sat_thresh=4000.0,
                                    nl_frac=0.7,
                                    drop_first_n=3,
                                    min_reads=6,
                                    tile=(512, 512))
                                
                                filled = self.inpaint_nearest_fast(slope, nanmask)
                                slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                            slope, nanmask = self.slope_linear(
                                img_corr,
                                read_time=read_time,
                                sat_thresh=4000.0,
                                nl_frac=0.7,
                                drop_first_n=3,
                                min_reads=6,
                                tile=(512, 512))

                            filled = self.inpaint_nearest_fast(slope, nanmask)
                            slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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

                        if slope is not None and os.path.exists(os.path.join(calib_path, "QLmat_new.npz")):
                            R_matrix = load_npz(calib_path+'QLmat_new.npz')
                            print("Quicklook optimal extraction started for",ifs_mode)
                            cube1,error1 = self.optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                                slope, nanmask = self.slope_linear(
                                    img_corr,
                                    read_time=read_time,
                                    sat_thresh=4000.0,
                                    nl_frac=0.7,
                                    drop_first_n=3,
                                    min_reads=6,
                                    tile=(512, 512))
                                
                                filled = self.inpaint_nearest_fast(slope, nanmask)
                                slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                            slope, nanmask = self.slope_linear(
                                img_corr,
                                read_time=read_time,
                                sat_thresh=4000.0,
                                nl_frac=0.7,
                                drop_first_n=3,
                                min_reads=6,
                                tile=(512, 512))
                            filled = self.inpaint_nearest_fast(slope, nanmask)
                            slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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

                        if slope is not None and os.path.exists(os.path.join(calib_path, "QLmat_new.npz")):
                            R_matrix = load_npz(calib_path+'QLmat_new.npz')
                            print("Quicklook optimal extraction started for",ifs_mode)
                            cube1,error1 = self.optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                                slope, nanmask = self.slope_linear(
                                    img_corr,
                                    read_time=read_time,
                                    sat_thresh=4000.0,
                                    nl_frac=0.7,
                                    drop_first_n=3,
                                    min_reads=6,
                                    tile=(512, 512))
                                
                                filled = self.inpaint_nearest_fast(slope, nanmask)
                                slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                            slope, nanmask = self.slope_linear(
                                img_corr,
                                read_time=read_time,
                                sat_thresh=4000.0,
                                nl_frac=0.7,
                                drop_first_n=3,
                                min_reads=6,
                                tile=(512, 512))

                            filled = self.inpaint_nearest_fast(slope, nanmask)
                            slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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

                        if slope is not None and os.path.exists(os.path.join(calib_path, "QLmat_new.npz")):
                            R_matrix = load_npz(calib_path+'QLmat_new.npz')
                            print("Quicklook optimal extraction started for",ifs_mode)
                            cube1,error1 = self.optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                                slope, nanmask = self.slope_linear(
                                    img_corr,
                                    read_time=read_time,
                                    sat_thresh=4000.0,
                                    nl_frac=0.7,
                                    drop_first_n=3,
                                    min_reads=6,
                                    tile=(512, 512))
                                filled = self.inpaint_nearest_fast(slope, nanmask)
                                slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                                    input_filename=filename,suffix='_quicklook',
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
                            slope, nanmask = self.slope_linear(
                                img_corr,
                                read_time=read_time,
                                sat_thresh=4000.0,
                                nl_frac=0.7,
                                drop_first_n=3,
                                min_reads=6,
                                tile=(512, 512))
                            filled = self.inpaint_nearest_fast(slope, nanmask)
                            slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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

                        if slope is not None and os.path.exists(os.path.join(calib_path, "QLmat_new.npz")):
                            R_matrix = load_npz(calib_path+'QLmat_new.npz')
                            print("Quicklook optimal extraction started for",ifs_mode)
                            cube1,error1 = self.optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                                slope, nanmask = self.slope_linear(
                                    img_corr,
                                    read_time=read_time,
                                    sat_thresh=4000.0,
                                    nl_frac=0.7,
                                    drop_first_n=3,
                                    min_reads=6,
                                    tile=(512, 512))

                                filled = self.inpaint_nearest_fast(slope, nanmask)
                                slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                            slope, nanmask = self.slope_linear(
                                img_corr,
                                read_time=read_time,
                                sat_thresh=4000.0,
                                nl_frac=0.7,
                                drop_first_n=3,
                                min_reads=6,
                                tile=(512, 512))
                            filled = self.inpaint_nearest_fast(slope, nanmask)
                            slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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

                        if slope is not None and os.path.exists(os.path.join(calib_path, "QLmat_new.npz")):
                            R_matrix = load_npz(calib_path+'QLmat_new.npz')
                            print("Quicklook optimal extraction started for",ifs_mode)
                            cube1,error1 = self.optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                                slope, nanmask = self.slope_linear(
                                    img_corr,
                                    read_time=read_time,
                                    sat_thresh=4000.0,
                                    nl_frac=0.7,
                                    drop_first_n=3,
                                    min_reads=6,
                                    tile=(512, 512))
                                
                                filled = self.inpaint_nearest_fast(slope, nanmask)
                                slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                            slope, nanmask = self.slope_linear(
                                img_corr,
                                read_time=read_time,
                                sat_thresh=4000.0,
                                nl_frac=0.7,
                                drop_first_n=3,
                                min_reads=6,
                                tile=(512, 512))
                            filled = self.inpaint_nearest_fast(slope, nanmask)
                            slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                                slope, nanmask = self.slope_linear(
                                    img_corr,
                                    read_time=read_time,
                                    sat_thresh=4000.0,
                                    nl_frac=0.7,
                                    drop_first_n=3,
                                    min_reads=6,
                                    tile=(512, 512))

                                filled = self.inpaint_nearest_fast(slope, nanmask)
                                slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                            slope, nanmask = self.slope_linear(
                                img_corr,
                                read_time=read_time,
                                sat_thresh=4000.0,
                                nl_frac=0.7,
                                drop_first_n=3,
                                min_reads=6,
                                tile=(512, 512))
                            filled = self.inpaint_nearest_fast(slope, nanmask)
                            slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                                slope, nanmask = self.slope_linear(
                                    img_corr,
                                    read_time=read_time,
                                    sat_thresh=4000.0,
                                    nl_frac=0.7,
                                    drop_first_n=3,
                                    min_reads=6,
                                    tile=(512, 512))
                                filled = self.inpaint_nearest_fast(slope, nanmask)
                                slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
                            slope, nanmask = self.slope_linear(
                                img_corr,
                                read_time=read_time,
                                sat_thresh=4000.0,
                                nl_frac=0.7,
                                drop_first_n=3,
                                min_reads=6,
                                tile=(512, 512))
                            filled = self.inpaint_nearest_fast(slope, nanmask)
                            slope_filled = self.masked_smooth_fast(filled, nanmask, sigma=1.25)
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
