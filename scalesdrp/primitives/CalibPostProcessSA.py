from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
from scalesdrp.primitives.scales_file_primitives import fits_writer_calib
import scalesdrp.primitives.saturation_create as saturation_create
import scalesdrp.primitives.linearity_create as linearity_create
import scalesdrp.primitives.readnoise_create as rn_create
import scalesdrp.primitives.bpm_create as bpm_create
import scalesdrp.primitives.scales_basic as scbasic
from scalesdrp.core.scales_pkg_resources import get_resource_path
import scalesdrp.primitives.robust as robust
import numpy as np
import time
import os
from astropy.io import fits
from scipy import sparse
import scalesdrp.primitives.reference as reference #1/f and reference pixel correction
from astropy.nddata import StdDevUncertainty
from scalesdrp.core.scales_proctab import Proctab
import logging
log = logging.getLogger("SCALES")
import matplotlib.pyplot as plt

class CalibPostProcessSA(BasePrimitive):

    """
    This function correct a raw read for channel bias, alternative column noise, and 1/f noise.
    The core of this function is `import scalesdrp.primitives.reference as reference`. 
    No external calibration files are involved.
        Args:
            data_image: The (N,H,W) input ramp cube.

        Returns:
            data_image: The (N,H,W) detector level corrected ramp cube.
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger

        if not hasattr(self, "proctab") or self.proctab is None:
            self.proctab = Proctab(logger=self.logger if hasattr(self, "logger") else logging.getLogger("SCALES"))

    def _perform(self):
        output_dir = self.config.instrument.output_directory
        plot_dir = self.config.instrument.plot_directory
        package = __name__.split('.')[0]
        calibfilepath = self.context.calib_file_path
        calib_dir = str(get_resource_path(package, calibfilepath))+'/'    
        create_linearity = bool(self.config.instrument.create_linearity_coeff)
        create_sat = bool(self.config.instrument.create_sat_map)
        create_bpm = bool(self.config.instrument.create_bpm)
        create_readnoise = bool(self.config.instrument.create_readnoise)
        create_rectmat = bool(self.config.instrument.create_rectmat)

        linearity_groups = []
        saturation_files = []
        bpm_groups = []
        l1_groups = []
        readnoise_grp=[]

        proctab_path = os.path.join(output_dir,"scales.proc")
        self.proctab.read_proctab(proctab_path)
        if self.proctab.proctab is None:
            raise RuntimeError(f"Could not read processing table: {proctab_path}")
        proctab = self.proctab.proctab.to_pandas()

        #-------------------------------------------------------------------------
        if create_linearity:

            linearity_files = scbasic.validate_linearity_inputs(
                proctab=proctab,
                output_dir=output_dir,
                min_files=self.config.instrument.min_lin_files,
                min_reads=self.config.instrument.min_lin_reads)

            for group in linearity_files:
                camera = group["params"]["camera"]
                mclock = group["params"]["mclock"]
                exptime = group["params"]["exptime"]
                filenames = group["filenames"]
                paths = group["paths"]
                
                lin_data = []
                hdrm=None
                for filename, filepath in zip(filenames, paths):
                    #filepath = os.path.join(output_dir,filename)
                    
                    if not os.path.isfile(filepath):
                        msg = (
                            f"L11 file listed in processing table "
                            f"does not exist: {filepath}")
                        self.logger.error(msg)
                        raise RuntimeError(msg)

                    with fits.open(filepath,memmap=False) as hdul:
                        data = np.asarray(hdul[0].data,dtype=np.float32)
                        hdrm = hdul[0].header.copy()

                        if data.ndim != 3:
                            raise ValueError(
                                f"L10 DATA must be 3-D. "
                                f"{filename} has shape {data.shape}.")
                        if hdrm is None:
                            hdrm = hdul[0].header.copy()

                        lin_data.append(data)
                
                lin_data = np.stack(lin_data,axis=0)
                
                self.logger.info(
                    f"Creating linearity coefficients from {len(filenames)} "
                    f"L11 ramps: CAMERA={camera}, "
                    f"MCLOCK={mclock}, EXPTIME={exptime}:"
                    "WAIT WAIT WAIT")
                
                if camera=='IFS':
                    bpm = self.context.bpm_ifs
                    master_bpm = fits.getdata(calib_dir+bpm)
                    if mclock == '9.0 MHz':
                        lin_filename=self.context.lin_coeff_ifs_fast0p6
                        readnoise = self.context.sig_map_ifs_fast0p6
                        gain = self.context.gain_ifs_fast0p6
                        sat_map_file = self.context.sat_map_ifs_fast0p6

                    if mclock == '5.0 MHz':
                        lin_filename=self.context.lin_coeff_ifs_fast1
                        readnoise = self.context.sig_map_ifs_fast1
                        gain = self.context.gain_ifs_fast1
                        sat_map_file = self.context.sat_map_ifs_fast1

                    if mclock == '20.0 MHz':
                        lin_filename=self.context.lin_coeff_ifs_slow
                        readnoise = self.context.sig_map_ifs_slow
                        gain = self.context.gain_ifs_slow
                        sat_map_file = self.context.sat_map_ifs_slow

                if camera=='Im':
                    bpm = self.context.bpm_img
                    master_bpm = fits.getdata(calib_dir+bpm)
                    if mclock == '9.0 MHz':
                        lin_filename=self.context.lin_coeff_img_fast0p6
                        readnoise = self.context.sig_map_img_fast0p6
                        gain = self.context.gain_img_fast0p6
                        sat_map_file = self.context.sat_map_img_fast0p6

                    if mclock == '5.0 MHz':
                        lin_filename=self.context.lin_coeff_img_fast1
                        readnoise = self.context.sig_map_img_fast1
                        gain = self.context.gain_img_fast1
                        sat_map_file = self.context.sat_map_img_fast1

                    if mclock == '20.0 MHz':
                        lin_filename=self.context.lin_coeff_img_slow
                        readnoise = self.context.sig_map_img_slow
                        gain = self.context.gain_img_slow
                        sat_map_file = self.context.sat_map_img_slow

                SIG_map = fits.getdata(calib_dir+readnoise)
                with fits.open(calib_dir+sat_map_file) as hdul:
                    sat_map = np.asarray(hdul["SATURATION"].data, dtype=float)
                lin_output_path = os.path.join(calib_dir,lin_filename)
                
                results = linearity_create.run_linearity_calibration(
                    ramps=lin_data,
                    saturation_map=sat_map,
                    bad_pixel_map=master_bpm,
                    gain=gain,
                    readnoise=SIG_map,
                    polynomial_terms=(2, 3, 4),
                    output_fits=lin_output_path,
                    block_rows=32,
                    read_times=None,
                    skip_initial_reads=2,
                    n_pedestal_reads=2,
                    saturation_fraction=0.95,
                    minimum_reads_per_ramp=self.config.instrument.min_lin_reads,
                    minimum_total_samples=None,
                    minimum_valid_ramps=self.config.instrument.min_lin_files,
                    require_positive_signal=True,
                    sigma_clip=5.0,
                    maximum_iterations=3,
                    maximum_reduced_chisq=None,
                    check_monotonic=True,
                    derivative_tolerance=-1.0e-6,
                    n_monotonic_test_points=256,
                    rcond=None,
                    progress=True,
                    overwrite=True,
                    metadata={
                    "SATFRAC": 0.95,
                    "SKIPREAD": 2,
                    "NPEDREAD": 2})

                fig1, fig2 = linearity_create.plot_lin_ref_dia(
                    results,
                    signal_dn=2500.0,
                    output_dir=plot_dir,
                    show=False,
                    save_prefix=f"{lin_filename}_diagnostic_fig1")

                fig3, axes = linearity_create.plot_linearity_ramp_validation(
                    lin_data,
                    results,
                    pixels=(1024, 1024),
                    read_times=None,
                    saturation_fraction=0.95,
                    skip_initial_reads=2,
                    n_pedestal_reads=2,
                    output_dir=plot_dir,
                    save_prefix=f"{lin_filename}_diagnostic_fig2",
                    show=False)
                diagnostics = linearity_create.diagnose_linearity_chisq(
                    results=results,
                    ramps=lin_data,
                    readnoise=SIG_map,
                    gain=gain,
                    output_dir=plot_dir,
                    filename=f"{lin_filename}_chisq_diagnostic.png",
                    sample_read=10,
                    show=False)

                plt.close(diagnostics["figure"])
        #--------------------------------------------------------------------------
        if create_sat:
            saturation_files = scbasic.validate_saturation_inputs(
                proctab=proctab,
                output_dir=output_dir,
                min_reads=self.config.instrument.min_sat_reads,
                min_files=self.config.instrument.minimum_detected_ramps,
                suffix="_L11")

            for group in saturation_files:
                camera = group["params"]["camera"]
                mclock = group["params"]["mclock"]
                exptime = group["params"]["exptime"]
                filenames = group["filenames"]
                paths = group["paths"]
                
                sat_data = []
                hdrm=None
                for filename, filepath in zip(filenames, paths):
                    #filepath = os.path.join(output_dir,filename)
                    
                    if not os.path.isfile(filepath):
                        msg = (
                            f"L11 file listed in processing table "
                            f"does not exist: {filepath}")
                        self.logger.error(msg)
                        raise RuntimeError(msg)

                    with fits.open(filepath,memmap=False) as hdul:
                        data = np.asarray(hdul[0].data,dtype=np.float32)
                        hdrm = hdul[0].header.copy()

                        if data.ndim != 3:
                            raise ValueError(
                                f"L11 DATA must be 3-D. "
                                f"{filename} has shape {data.shape}.")
                        if hdrm is None:
                            hdrm = hdul[0].header.copy()

                        sat_data.append(data)
                
                sat_data = np.stack(sat_data,axis=0)
                
                self.logger.info(
                    f"Creating saturation map from {len(filenames)} "
                    f"L11 ramps: CAMERA={camera}, "
                    f"MCLOCK={mclock}, EXPTIME={exptime}:"
                    "WAIT..........................")

                options = dict(
                    drop_fraction=self.config.instrument.drop_fraction,
                    minimum_saturation_dn=self.config.instrument.minimum_saturation_dn,
                    sustained_window=3,
                    block_rows=32,
                    maximum_iterations=5,
                    minimum_detected_ramps=self.config.instrument.minimum_detected_ramps,
                    fallback_saturation_dn=None)

                result = saturation_create.create_saturation_map(
                    sat_data,
                    **options)

                if camera=='IFS':
                    if mclock == '9.0 MHz':
                        sat_filename=self.context.sat_map_ifs_fast0p6
                    if mclock == '5.0 MHz':
                        sat_filename=self.context.sat_map_ifs_fast1
                    if mclock == '20.0 MHz':
                        sat_filename=self.context.sat_map_ifs_slow

                if camera=='Im':
                    if mclock == '9.0 MHz':
                        sat_filename=self.context.sat_map_img_fast0p6
                    if mclock == '5.0 MHz':
                        sat_filename=self.context.sat_map_img_fast1
                    if mclock == '20.0 MHz':
                        sat_filename=self.context.sat_map_img_slow


                sat_output_path = os.path.join(calib_dir,sat_filename)
                sat_header = hdrm.copy()
                sat_header.add_history("Input ramps were detector-corrected and linearity-corrected")
                saturation_create.write_saturation_fits(
                    result,
                    sat_output_path,
                    header=sat_header,
                    filenames=sat_filename,
                    camera=camera,
                    mclock=mclock,
                    exptime=exptime,
                    calibration_options=options,
                    overwrite=True)
                fig, axes = saturation_create.plot_sat(
                    result,
                    title=f"Saturation calibration: {camera}, {mclock} MHz",
                    output_path=os.path.join(
                        plot_dir,f"{os.path.splitext(sat_filename)[0]}_diagnostics.png"))
                plt.close(fig)
                self.logger.info(f"Saturation calibration map written to {sat_output_path}")

        #-------------------------------------------------------------------------------------------
        if create_bpm:

            bpm_groups = scbasic.validate_bpm_inputs(
                proctab=proctab,
                output_dir=output_dir,
                min_darks=self.config.instrument.bpm_min_darks,
                min_flats=self.config.instrument.bpm_min_flats)

            for group in bpm_groups:
                camera = group["params"]["camera"]
                mclock = group["params"]["mclock"]
                dark_files = group["dark_files"]
                dark_paths = group["dark_paths"]
                flat_files = group["flat_files"]
                flat_paths = group["flat_paths"]

                self.logger.info(
                    f"Creating BPM: CAMERA={camera}, MCLOCK={mclock}, "
                    f"N_DARK={len(dark_files)}, "
                    f"N_FLAT={len(flat_files)}")

                bpm_dark = []
                dark_headers = []

                for filename, filepath in zip(dark_files, dark_paths):
                    with fits.open(filepath, memmap=True) as hdul:
                        data = np.asarray(hdul[0].data,dtype=np.float32).copy()
                        header = hdul[0].header.copy()

                    if data.shape != (2048, 2048):
                        raise ValueError(
                            f"Invalid DARK slope shape: "
                            f"{filename}: {data.shape}")

                    bpm_dark.append(data)
                    dark_headers.append(header)

                bpm_flat = []
                flat_headers = []

                for filename, filepath in zip(flat_files, flat_paths):
                    with fits.open(filepath, memmap=True) as hdul:
                        data = np.asarray(hdul[0].data,dtype=np.float32).copy()
                        header = hdul[0].header.copy()

                    if data.shape != (2048, 2048):
                        raise ValueError(
                            f"Invalid FLATLAMP slope shape: "
                            f"{filename}: {data.shape}")
                    bpm_flat.append(data)
                    flat_headers.append(header)
                bpm_dark = np.stack(bpm_dark, axis=0)
                bpm_flat = np.stack(bpm_flat, axis=0)
                
                self.logger.info(f"DARK stack shape: {bpm_dark.shape}")
                self.logger.info(f"FLAT stack shape: {bpm_flat.shape}")

            if camera == "IFS":
                self.logger.info("Started Creating IFS bad pixel map")

                dark_params = dict(
                    spatial_sigma_thresh=self.config.instrument.ifs_dark_spatial_sigma_thresh,
                    spatial_kernel_size=self.config.instrument.ifs_dark_spatial_kernel_size,
                    min_frames_bad_spatial=self.config.instrument.ifs_dark_min_frames_bad_spatial,
                    normalize_flats=False,
                    row_correction=False,
                    row_suppression_fraction=0.20,
                    dark_hot_sigma=self.config.instrument.ifs_dark_hot_sigma,
                    dark_cold_sigma=self.config.instrument.ifs_dark_cold_sigma,
                    temporal_sigma_thresh=self.config.instrument.ifs_dark_temporal_sigma_thresh)

                flat_params = dict(
                    temporal_sigma_thresh=self.config.instrument.ifs_flat_temporal_sigma_thresh,
                    spatial_sigma_thresh=self.config.instrument.ifs_flat_spatial_sigma_thresh,
                    spatial_kernel_size=self.config.instrument.ifs_flat_spatial_kernel_size,
                    min_frames_bad_spatial=self.config.instrument.ifs_flat_min_frames_bad_spatial,
                    normalize_flats=self.config.instrument.ifs_normalize_flats,
                    flat_norm_kernel_size=self.config.instrument.ifs_flat_norm_kernel_size,
                    row_correction=False,
                    row_suppression_fraction=0.20,
                    flat_low_response_thresh=self.config.instrument.ifs_flat_low_response_thresh,
                    flat_high_response_thresh=self.config.instrument.ifs_flat_high_response_thresh)

                dark_bpm, dark_dq, dark_masks, dark_products = (
                    bpm_create.generate_bpm(
                        bpm_dark,
                        stack_name="Dark Stack",
                        mode="dark",
                        **dark_params))

                flat_bpm, flat_dq, flat_masks, flat_products = (
                    bpm_create.generate_bpm(
                        bpm_flat,
                        stack_name="Flat Stack",
                        mode="flat",
                        **flat_params))

                combined_dq = dark_dq | flat_dq
                combined_bpm = combined_dq != 0

                self.logger.info(f"Dark BPM pixels: {np.sum(dark_bpm)}")
                self.logger.info(f"Flat BPM pixels: {np.sum(flat_bpm)}")
                self.logger.info(f"Combined BPM pixels: {np.sum(combined_bpm)}")

                bpm_header = bpm_create.create_bpm_header(
                    camera=camera,
                    mclock=mclock,
                    dark_files=dark_files,
                    flat_files=flat_files,
                    dark_params=dark_params,
                    flat_params=flat_params)

                bpm_header["NBADPIX"] = (int(np.sum(combined_bpm)),"Total flagged pixels")
                bpm_header["BADFRAC"] = (float(np.mean(combined_bpm)),"Fraction of flagged pixels")
                bpm_output_path = os.path.join(calib_dir,"bpm_IFS_cd6.fits")
                #fits.PrimaryHDU(data=combined_bpm.astype(np.uint16),header=bpm_header,).writeto(bpm_output_path,overwrite=True)
                fits.HDUList([
                    fits.PrimaryHDU(
                        data=combined_bpm.astype(np.uint16),
                        header=bpm_header),
                    fits.ImageHDU(
                        data=combined_dq.astype(np.uint16),
                        name="DQ"),]).writeto(bpm_output_path,overwrite=True)

                rmat_output_path = os.path.join(calib_dir,self.config.instrument.bpm_ifs)
                rmat = bpm_create.bpm_correction(combined_bpm)
                sparse.save_npz(rmat_output_path,rmat)

                os.makedirs(plot_dir, exist_ok=True)
                diagnostics = bpm_create.plot_bpm_diagnostics(
                    dark_products=dark_products,
                    flat_products=flat_products,
                    dark_masks=dark_masks,
                    flat_masks=flat_masks,
                    output_dir=plot_dir,
                    prefix=f"bpm_{camera}_{mclock}",
                    flat_low_response_thresh=flat_params["flat_low_response_thresh"],
                    flat_high_response_thresh=flat_params["flat_high_response_thresh"],
                    dark_min_frames_bad_spatial=dark_params["min_frames_bad_spatial"],
                    flat_min_frames_bad_spatial=flat_params["min_frames_bad_spatial"],
                    save_pdf=False)
                self.logger.info(f"BPM diagnostics saved: {diagnostics['png_path']}")
                plt.close(diagnostics["figure"])
                self.logger.info(f"Finished Creating IFS bad pixel map: "f"{bpm_output_path}")

            if camera == "Im":
                self.logger.info("Started Creating Imager bad pixel map")

                dark_params = dict(
                    spatial_sigma_thresh=self.config.instrument.im_dark_spatial_sigma_thresh,
                    spatial_kernel_size=self.config.instrument.im_dark_spatial_kernel_size,
                    min_frames_bad_spatial=self.config.instrument.im_dark_min_frames_bad_spatial,
                    normalize_flats=False,
                    row_correction=self.config.instrument.im_dark_row_correction,
                    row_suppression_fraction=self.config.instrument.im_dark_row_suppression_fraction,
                    dark_hot_sigma=self.config.instrument.im_dark_hot_sigma,
                    dark_cold_sigma=self.config.instrument.im_dark_cold_sigma,
                    temporal_sigma_thresh=self.config.instrument.im_dark_temporal_sigma_thresh)

                flat_params = dict(
                    temporal_sigma_thresh=self.config.instrument.im_flat_temporal_sigma_thresh,
                    spatial_sigma_thresh=self.config.instrument.im_flat_spatial_sigma_thresh,
                    spatial_kernel_size=self.config.instrument.im_flat_spatial_kernel_size,
                    min_frames_bad_spatial=self.config.instrument.im_flat_min_frames_bad_spatial,
                    normalize_flats=self.config.instrument.im_normalize_flats,
                    flat_norm_kernel_size=self.config.instrument.im_flat_norm_kernel_size,
                    row_correction=self.config.instrument.im_flat_row_correction,
                    row_suppression_fraction=self.config.instrument.im_flat_row_suppression_fraction,
                    flat_low_response_thresh=self.config.instrument.im_flat_low_response_thresh,
                    flat_high_response_thresh=self.config.instrument.im_flat_high_response_thresh)

                dark_bpm, dark_dq, dark_masks, dark_products = (
                    bpm_create.generate_bpm(
                        bpm_dark,
                        stack_name="Dark Stack",
                        mode="dark",
                        **dark_params))

                flat_bpm, flat_dq, flat_masks, flat_products = (
                    bpm_create.generate_bpm(
                        bpm_flat,
                        stack_name="Flat Stack",
                        mode="flat",
                        **flat_params))

                combined_dq = dark_dq | flat_dq
                combined_bpm = combined_dq != 0

                self.logger.info(f"Dark BPM pixels: {np.sum(dark_bpm)}")
                self.logger.info(f"Flat BPM pixels: {np.sum(flat_bpm)}")
                self.logger.info(f"Combined BPM pixels: {np.sum(combined_bpm)}")

                bpm_header = bpm_create.create_bpm_header(
                    camera=camera,
                    mclock=mclock,
                    dark_files=dark_files,
                    flat_files=flat_files,
                    dark_params=dark_params,
                    flat_params=flat_params)

                bpm_header["NBADPIX"] = (int(np.sum(combined_bpm)),"Total flagged pixels")
                bpm_header["BADFRAC"] = (float(np.mean(combined_bpm)),"Fraction of flagged pixels")
                bpm_output_path = os.path.join(calib_dir,"bpm_img_cd6.fits")
                #fits.PrimaryHDU(data=combined_bpm.astype(np.uint16),header=bpm_header,).writeto(bpm_output_path,overwrite=True)
                fits.HDUList([
                    fits.PrimaryHDU(
                        data=combined_bpm.astype(np.uint16),
                        header=bpm_header),
                    fits.ImageHDU(
                        data=combined_dq.astype(np.uint16),
                        name="DQ"),]).writeto(bpm_output_path,overwrite=True)

                rmat_output_path = os.path.join(calib_dir,self.config.instrument.bpm_img)
                rmat = bpm_create.bpm_correction(combined_bpm)
                sparse.save_npz(rmat_output_path,rmat)
                os.makedirs(plot_dir, exist_ok=True)
                diagnostics = bpm_create.plot_bpm_diagnostics(
                    dark_products=dark_products,
                    flat_products=flat_products,
                    dark_masks=dark_masks,
                    flat_masks=flat_masks,
                    output_dir=plot_dir,
                    prefix=f"bpm_{camera}_{mclock}",
                    flat_low_response_thresh=flat_params["flat_low_response_thresh"],
                    flat_high_response_thresh=flat_params["flat_high_response_thresh"],
                    dark_min_frames_bad_spatial=dark_params["min_frames_bad_spatial"],
                    flat_min_frames_bad_spatial=flat_params["min_frames_bad_spatial"],
                    save_pdf=False)
                self.logger.info(f"BPM diagnostics saved: {diagnostics['png_path']}")
                plt.close(diagnostics["figure"])                
                self.logger.info(f"Finished Creating Im bad pixel map: "f"{bpm_output_path}")

        #---------------------------------------------------------------------------------------------
        if create_readnoise:

            readnoise_files = scbasic.validate_rn_inputs(
                proctab=proctab,
                output_dir=output_dir,
                min_reads=self.config.instrument.min_rn_reads,
                min_files=self.config.instrument.minimum_rn_ramps,
                suffix="_L13")
            
            self.logger.info("Started creating the readnoise map")
            
            for group in readnoise_files:
                camera = group["params"]["camera"]
                mclock = group["params"]["mclock"]
                exptime = group["params"]["exptime"]
                filenames = group["filenames"]
                paths = group["paths"]
                
                rn_data = []
                hdrm=None
                for filename, filepath in zip(filenames, paths):
                    #filepath = os.path.join(output_dir,filename)
                    
                    if not os.path.isfile(filepath):
                        msg = (
                            f"L13 file listed in processing table "
                            f"does not exist: {filepath}")
                        self.logger.error(msg)
                        raise RuntimeError(msg)

                    with fits.open(filepath,memmap=False) as hdul:
                        data = np.asarray(hdul[0].data,dtype=np.float32)
                        hdrm = hdul[0].header.copy()

                        if data.ndim != 3:
                            raise ValueError(
                                f"L13 DATA must be 3-D. "
                                f"{filename} has shape {data.shape}.")
                        if hdrm is None:
                            hdrm = hdul[0].header.copy()

                        rn_data.append(data)
                
                rn_data = np.stack(rn_data,axis=0)
                
                self.logger.info(
                    f"Creating readnoise map from {len(filenames)} "
                    f"L13 ramps: CAMERA={camera}, "
                    f"MCLOCK={mclock}, EXPTIME={exptime}:"
                    "WAIT..................................")

                if camera=='IFS':
                    if mclock == '9.0 MHz':
                        rn_filename=self.context.sig_map_ifs_fast0p6
                        gain = self.context.gain_ifs_fast0p6
                    if mclock == '5.0 MHz':
                        rn_filename=self.context.sig_map_ifs_fast1
                        gain = self.context.gain_ifs_fast1
                    if mclock == '20.0 MHz':
                        rn_filename=self.context.sig_map_ifs_slow
                        gain = self.context.gain_ifs_slow

                if camera=='Im':
                    if mclock == '9.0 MHz':
                        rn_filename=self.context.sig_map_img_fast0p6
                        gain = self.context.gain_img_fast0p6
                    if mclock == '5.0 MHz':
                        rn_filename=self.context.sig_map_img_fast1
                        gain = self.context.gain_img_fast1
                    if mclock == '20.0 MHz':
                        rn_filename=self.context.sig_map_img_slow
                        gain = self.context.gain_img_slow

                rn_output_path = os.path.join(calib_dir,rn_filename)
                rn, cds_noise, n_good, bad_rn, header = (
                    rn_create.readnoise_from_cds_dark_exposures(
                        rn_data,
                        source_header=hdrm,
                        gain=1,
                        read0=1,
                        read1=2,
                        subtract_global=self.config.instrument.rn_sub_global,
                        subtract_amp=self.config.instrument.rn_sub_amp,
                        subtract_row=self.config.instrument.rn_sub_row,
                        method="pair_diff",
                        min_good=self.config.instrument.min_rn_reads,
                        interpolation_kernel=3))

                hdul = fits.HDUList([fits.PrimaryHDU(data=rn.astype(np.float32),header=header,),
                    fits.ImageHDU(data=cds_noise.astype(np.float32),name="CDS_NOISE_DN",),
                    fits.ImageHDU(data=n_good.astype(np.int32),name="N_GOOD",),
                    fits.ImageHDU(data=bad_rn.astype(np.uint8),name="BAD_RN",),])

                hdul.writeto(rn_output_path, overwrite=True)
                rn_plot_path = rn_create.plot_readnoise_diagnostics(
                    rn=rn,
                    cds_noise=cds_noise,
                    n_good=n_good,
                    bad_rn=bad_rn,
                    output_dir=plot_dir,
                    filename=f"rn_{camera}_{mclock}_fig1.png",
                    title="SCALES Read-Noise Calibration",
                    dpi=200,
                    show=False)
                self.logger.info(f"Read-noise calibration saved to {rn_output_path}")


        #----------------------------------------------------------------------------------------------
        if create_rectmat:
            l1_dt = scbasic.get_proc_entries(proctab,suffix="_L1")
            if len(l1_dt) == 0:
                msg = (
                    "No L1 products were found in the processing table. "
                    "Cannot create master calibration products.")
                self.logger.error(msg)
                raise RuntimeError(msg)

            l1_groups = scbasic.group_files_by_header(l1_dt)

            if len(l1_groups) == 0:
                msg = (
                    "L1 products were found, but no valid "
                    "calibration groups could be created.")
                self.logger.error(msg)
                raise RuntimeError(msg)

            self.logger.info(f"Created {len(l1_groups)} L1 calibration groups.")

            for group_index, group in enumerate(l1_groups, start=1):
                filenames = group["filenames"]
                params = group["params"]
                camera = group["params"]["camera"]
                mclock = group["params"]["mclock"]
                exptime = group["params"]["exptime"]
                filenames = group["filenames"]
                if not self.context.clobber:
                    existing_master = (
                        scbasic.find_existing_master_for_l1_group(
                            l1_filenames=filenames,
                            redux_dir=output_dir))
                    if existing_master is not None:
                        self.logger.info(
                            f"Master calibration already exists "
                            f"for group {group_index}/{len(l1_groups)}: "
                            f"{existing_master}. "
                            "Skipping master creation.")
                        continue
                
                self.logger.info(
                    f"Building master calibration "
                    f"{group_index}/{len(l1_groups)} "
                    f"from {len(filenames)} L1 files.")
                self.logger.info(
                    f"Group parameters: {params}")
                
                data_stack = []
                uncert_stack = []
                chisq_stack = []
                dq_stack = []
                hdrm = None

                for filename in filenames:
                    filepath = os.path.join(output_dir,filename)
                    
                    if not os.path.isfile(filepath):
                        msg = (
                            f"L1 file listed in processing table "
                            f"does not exist: {filepath}")
                        self.logger.error(msg)
                        raise RuntimeError(msg)

                    with fits.open(filepath,memmap=False) as hdul:
                        data = np.asarray(hdul[0].data,dtype=np.float32)
                        hdrm = hdul[0].header.copy()

                        if data.ndim != 2:
                            raise ValueError(
                                f"L1 DATA must be 2-D. "
                                f"{filename} has shape {data.shape}.")

                        data_stack.append(data)

                        if "UNCERT" not in hdul:
                            raise RuntimeError(
                                f"UNCERT extension missing from "
                                f"{filename}.")
                        uncert = np.asarray(hdul["UNCERT"].data,dtype=np.float32)

                        if uncert.shape != data.shape:
                            raise ValueError(
                                f"UNCERT shape {uncert.shape} does not "
                                f"match DATA shape {data.shape} in "
                                f"{filename}.")
                        uncert_stack.append(uncert)
                        
                        if "CHISQ" not in hdul:
                            raise RuntimeError(
                                f"CHISQ extension missing from "
                                f"{filename}.")
                        chisq = np.asarray(
                            hdul["CHISQ"].data,
                            dtype=np.float32)

                        if chisq.shape != data.shape:
                            raise ValueError(
                                f"CHISQ shape {chisq.shape} does not "
                                f"match DATA shape {data.shape} in "
                                f"{filename}.")
                        chisq_stack.append(chisq)

                        if "DQ" not in hdul:
                            raise RuntimeError(
                                f"DQ extension missing from "
                                f"{filename}.")

                        dq = np.asarray(
                            hdul["DQ"].data,
                            dtype=np.uint16)

                        if dq.shape != data.shape:
                            raise ValueError(
                                f"DQ shape {dq.shape} does not "
                                f"match DATA shape {data.shape} in "
                                f"{filename}.")
                        dq_stack.append(dq)

                data_stack = np.stack(data_stack,axis=0)
                uncert_stack = np.stack(uncert_stack,axis=0)
                chisq_stack = np.stack(chisq_stack,axis=0)
                dq_stack = np.stack(dq_stack,axis=0)
                n_slopes = data_stack.shape[0]
                
                self.logger.info(f"Master input stack shape: {data_stack.shape}")
                combine_method = self.config.instrument.master_comb_met

                (master,
                master_uncert,
                master_chisq,
                master_dq) = scbasic.build_master_from_stack(
                data_stack=data_stack,
                sigma_stack=uncert_stack,
                chisq_stack=chisq_stack,
                quality_stack=dq_stack,
                method=combine_method)

                self.logger.info(f"Master calibration created for group {group_index}.")
                imtype = str(params["imtype"]).strip().upper()

                if imtype == "CALUNIT":
                    monowave = str(params["monowave"]).strip().upper()
                    wl_val = float(monowave)
                    wl_str = f"{wl_val:.3f}".rstrip("0").rstrip(".")
                    suffix = f"_{wl_str}_mcalunit"
                    hdrm["HISTORY"] = "Master detector CALUNIT created"
                    hdrm["HISTORY"] = (f"Combined {n_slopes} L1 calunit slope images")
                    hdrm["HISTORY"] = (f"Master combination method: {combine_method}")
                    hdrm["HISTORY"] = (f"Master wavelength combined: {monowave}")
                    hdrm["HISTORY"] = ("UNCERT propagated during master combination")
                    hdrm["HISTORY"] = ("CHISQ propagated as median of contributing L1 values")
                    hdrm["HISTORY"] = ("DQ propagated using bitwise OR of input L1 DQ maps")

                    fits_writer_calib(
                        data=master,
                        header=hdrm,
                        redux_dir=output_dir,
                        input_filename=filename,
                        suffix=suffix,
                        uncert=master_uncert,
                        dq=master_dq,
                        chisq=master_chisq)

                    scbasic.proctab_update(
                        header=hdrm,
                        output_dir=output_dir,
                        input_filename=filename,
                        suffix=suffix,
                        frame=None,
                        proctab=self.proctab)
                    self.logger.info(
                        f"Master CALUNIT created from {n_slopes} "
                        f"L1 slope images using {combine_method}.")

                    plot_path = scbasic.plot_master_calib(
                        master=master,
                        master_dq=master_dq,
                        output_dir=plot_dir,
                        filename=f"master_{camera}_{mclock}_{imtype}.png",
                        title=f"Master Calibration: {camera} | {mclock} | {imtype}",
                        unit="DN/s",
                        exclude_bad=True,
                        show=False)
                    self.logger.info(f"Master calibration diagnostic saved to {plot_path}")

        if not (create_linearity or create_sat or create_bpm or create_readnoise or create_rectmat):
            self.logger.info("+++++++++++ All calibration process are turned off nothing to process +++++++++++")
            return self.action.args

        log_string = CalibPostProcessSA.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)

        self.logger.info("+++++++++++ All calibration processing are completed +++++++++++")

        return self.action.args
    # END: class CalibPostProcessSA()
