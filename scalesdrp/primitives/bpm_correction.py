import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve, median_filter
from scipy.interpolate import griddata
import os
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve as astropy_convolve
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.stats import median_abs_deviation
import time
from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
from importlib.resources import files
from pathlib import Path
import time
from scipy import sparse
from scipy.ndimage import median_filter

######### bpm correction untill CD4 ############################
def correct_local_defects_pass1_improved(image_data_to_correct, bad_pixel_mask, **kwargs):

    verbose = kwargs.get('verbose', True)
    if verbose: print("--- PASS 1: Starting Improved Local Iterative Correction ---")
    
    corrected_image = image_data_to_correct.copy()
    remaining_bpm = bad_pixel_mask.copy()
    initial_bad_count = np.sum(remaining_bpm)
    if initial_bad_count == 0:
        return corrected_image, remaining_bpm # Return empty mask

    current_box_size = kwargs.get('initial_box_size', 5)
    max_box_size = kwargs.get('max_box_size', 11)
    min_good_neighbors_frac = kwargs.get('min_good_neighbors_frac', 0.3)
    max_iterations = kwargs.get('max_iterations', 10) # More iterations can help

    # Keep track of pixels that are permanently unfixable by this method
    stalled_pixels_mask = np.zeros_like(remaining_bpm)

    for growth_pass in range((max_box_size - current_box_size) // 2 + 1):
        num_fixed_this_growth_step = 0
        for i in range(max_iterations):
            num_bad_at_start = np.sum(remaining_bpm)
            if num_bad_at_start == 0: break
            
            replacement_values = median_filter(corrected_image, size=current_box_size, mode='reflect')
            good_pixel_mask = (~remaining_bpm).astype(np.uint8)
            good_neighbor_count = convolve(good_pixel_mask, np.ones((current_box_size, current_box_size)), mode='constant', cval=0)
            min_good_req = int(min_good_neighbors_frac * (current_box_size**2 - 1))
            
            pixels_to_correct = remaining_bpm & (good_neighbor_count >= min_good_req)
            
            num_newly_corrected = np.sum(pixels_to_correct)
            if num_newly_corrected == 0: break
            
            corrected_image[pixels_to_correct] = replacement_values[pixels_to_correct]
            remaining_bpm[pixels_to_correct] = False
            num_fixed_this_growth_step += num_newly_corrected
        
        if np.sum(remaining_bpm) == 0: break
        
        # If we stalled, increase box size. Mark the remaining pixels as stalled for now.
        if num_fixed_this_growth_step == 0:
            stalled_pixels_mask |= remaining_bpm
            current_box_size += 2
            if verbose: print(f"Stalled. Growing box to {current_box_size}x{current_box_size}.")
        
    final_uncorrected_mask = remaining_bpm | stalled_pixels_mask
    if verbose: print(f"--- Pass 1 Finished. Uncorrectable local pixels: {np.sum(final_uncorrected_mask)} ---")
    
    return corrected_image, final_uncorrected_mask


def fill_global_defects_pass2_inpainting(image, bad_pixel_mask):
    
    num_defects = np.sum(bad_pixel_mask)
    if num_defects == 0:
        print("\n--- PASS 2: No defects to fill. Skipping. ---")
        return image.copy()
        
    #print(f"\n--- PASS 2: Inpainting {num_defects} large-scale defects using astropy.convolution ---")
    
    # Inpainting works by replacing NaN values.
    image_with_nans = image.copy()
    image_with_nans[bad_pixel_mask] = np.nan
    

    kernel_size_stddev = 3 # The standard deviation of the Gaussian in pixels
    kernel = Gaussian2DKernel(x_stddev=kernel_size_stddev)
    
    inpainted_image = interpolate_replace_nans(image_with_nans, kernel)
    
    #print("--- Pass 2 Inpainting Finished. ---")
    return inpainted_image

def apply_full_correction(image_to_correct,obsmode, pass1_kwargs={}):
    """
    Applies the improved full two-pass BPM correction workflow.
    """
    t1=time.time()
    #calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
    calib_path = str(files("scalesdrp").joinpath("calib"))+ "/"

    if obsmode=='IMAGING':
        master_bpm = fits.getdata(calib_path+'bpm_new_5.fits').astype(bool)
        #master_bpm = np.bitwise_or(master_bpm1, neg_bpm)
    else:
        master_bpm = fits.getdata(calib_path+'cd3_bpm_ifs_5mhz.fits').astype(bool)
        #master_bpm = np.bitwise_or(master_bpm1, neg_bpm)

    corrected_pass1, large_defects_mask = correct_local_defects_pass1_improved(
        image_to_correct,
        master_bpm,
        **pass1_kwargs
    )
    
    # Pass 2 uses this failure mask to fill in the large holes.
    fully_corrected_image = fill_global_defects_pass2_inpainting(
        corrected_pass1,
        large_defects_mask
    )
    t2=time.time()
    print(f"Bad pixel correction completed in {t2 - t1:.2f} seconds.")

    return fully_corrected_image

######################### Creating BPM untill cd4 ###########################

def generate_bpm_relative(
    image_stack: np.ndarray,
    stack_name: str = "Image Stack",
    spatial_brightness_factor: float = 5.0,
    spatial_kernel_size: int = 5,
    temporal_sigma_thresh: float = 5.0,
    min_frames_for_temporal: int = 3,
    plot_results: bool = True
) -> np.ndarray:
    """
    Generates a Bad Pixel Mask (BPM) from a stack of images.

    - Temporal outliers are found using a standard deviation.
    - Spatial outliers are found using a RELATIVE BRIGHTNESS test: a pixel
      is flagged if it is N times brighter or dimmer than its local median.

    Args:
        image_stack (np.ndarray): 3D stack of images (N_frames, Height, Width).
        stack_name (str): Name for logging/plotting.
        spatial_brightness_factor (float): A pixel is bad if its value is > this
                                           factor times the local median, or <
                                           the local median / this factor.
        spatial_kernel_size (int): Size of the square kernel for spatial median filter.
        temporal_sigma_thresh (float): Sigma threshold for the temporal outlier test.
        min_frames_for_temporal (int): Min frames to run the temporal test.

    Returns:
        np.ndarray: A 2D boolean bad pixel mask where True represents a bad pixel.
    """
    if image_stack.ndim != 3:
        raise ValueError("Input `image_stack` must be a 3D array.")
        
    n_frames, height, width = image_stack.shape

    final_bpm = np.all(np.isnan(image_stack), axis=0)
    print(f"Flagged {np.sum(final_bpm)} pixels that are initially NaN in all frames.")

    # --- Criterion 1: Temporal Outliers ---
    temporal_bpm = np.zeros_like(final_bpm)
    if n_frames >= min_frames_for_temporal and temporal_sigma_thresh > 0:
        start_time = time.time()
        print(f"1. Finding Temporal Outliers (sigma > {temporal_sigma_thresh})...")
        pixel_medians = np.nanmedian(image_stack, axis=0, keepdims=True)
        pixel_stds_map = np.nanstd(image_stack, axis=0)
        valid_stds = pixel_stds_map[np.isfinite(pixel_stds_map) & (pixel_stds_map > 0)]
        std_floor = np.median(valid_stds) * 0.01 if len(valid_stds) > 0 else 1e-5
        pixel_stds_map[~np.isfinite(pixel_stds_map) | (pixel_stds_map <= 0)] = max(std_floor, 1e-5)
        
        deviations = np.abs(image_stack - pixel_medians)
        is_outlier_stack = deviations > (temporal_sigma_thresh * pixel_stds_map)
        temporal_bpm = np.any(is_outlier_stack, axis=0)
        
        print(f"   + Found {np.sum(temporal_bpm)} temporal outliers in {time.time() - start_time:.2f}s.")
        final_bpm |= temporal_bpm

    # --- Criterion 2: Spatial Outliers using Relative Brightness ---
    spatial_bpm = np.zeros_like(final_bpm)
    if spatial_brightness_factor > 1 and spatial_kernel_size >= 3:
        start_time = time.time()
        print(f"2. Finding Spatial Outliers (relative brightness factor > {spatial_brightness_factor})...")
        
        for i in range(n_frames):
            frame = image_stack[i].copy()
            original_nans = np.isnan(frame)
            
            # Use a global median for NaN replacement to not affect local calculations
            global_median_val = np.nanmedian(frame)
            if not np.isfinite(global_median_val): global_median_val = 0
            
            frame_no_nan = np.nan_to_num(frame, nan=global_median_val)
            
            # a) Calculate the local median for every pixel
            local_median = median_filter(frame_no_nan, size=spatial_kernel_size, mode='reflect')
            
            local_median[local_median <= 1e-9] = 1e-9

            with np.errstate(divide='ignore', invalid='ignore'):
                is_hot_pixel = frame > (spatial_brightness_factor * local_median)
                is_cold_pixel = frame < (local_median / spatial_brightness_factor)
            
            is_outlier_this_frame = is_hot_pixel | is_cold_pixel
            
            is_outlier_this_frame |= original_nans 
            
            spatial_bpm |= is_outlier_this_frame

        print(f"   + Found {np.sum(spatial_bpm)} spatial outliers in {time.time() - start_time:.2f}s.")
        final_bpm |= spatial_bpm
    print(f"Total unique pixels flagged for {stack_name}: {np.sum(final_bpm)}")

    return final_bpm

############### bpm creation CD4 onwards ######################
def robust_sigma_from_mad(values, floor=1e-6):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return floor

    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    sigma = 1.4826 * mad

    return max(sigma, floor)


def generate_bpm_robust_v2(
    image_stack,
    stack_name="Image Stack",
    mode="dark",

    # temporal instability test
    temporal_sigma_thresh=6.0,
    min_frames_bad_temporal=2,

    # spatial local outlier test
    spatial_sigma_thresh=6.0,
    spatial_kernel_size=7,
    min_frames_bad_spatial=2,

    # dark persistent hot/cold test
    dark_hot_sigma=8.0,
    dark_cold_sigma=8.0,

    # flat response test
    normalize_flats=True,
    flat_norm_kernel_size=31,
    flat_low_response_thresh=0.5,
    flat_high_response_thresh=1.5,

    # general
    mad_floor_fraction=0.05,
):
    """
    Generate a master BPM from a stack of dark or flat slope images.

    Parameters
    ----------
    image_stack : ndarray
        Shape = (n_frames, ny, nx)
    mode : {"dark", "flat"}
        dark: detect hot/cold/dark-current outliers and unstable pixels.
        flat: detect low/high-response pixels after illumination normalization.

    Returns
    -------
    final_bpm : 2D bool ndarray
        True = bad pixel.
    """
    if image_stack.ndim != 3:
        raise ValueError("image_stack must have shape (n_frames, ny, nx)")

    if mode not in {"dark", "flat"}:
        raise ValueError("mode must be 'dark' or 'flat'")

    stack = np.asarray(image_stack, dtype=np.float32)
    n_frames, ny, nx = stack.shape

    if spatial_kernel_size % 2 == 0:
        spatial_kernel_size += 1
    if flat_norm_kernel_size % 2 == 0:
        flat_norm_kernel_size += 1

    print(f"\n=== Generating BPM for: {stack_name} ===")
    print(f"Mode: {mode}")
    print(f"Stack shape: {stack.shape}")

    # --------------------------------------------------
    # 0. NaNs / non-finite pixels
    # --------------------------------------------------
    all_nan_bpm = np.all(~np.isfinite(stack), axis=0)
    any_nan_bpm = np.any(~np.isfinite(stack), axis=0)

    final_bpm = all_nan_bpm.copy()

    print(f"All-frame NaN pixels: {np.sum(all_nan_bpm)}")
    print(f"Any-frame NaN pixels: {np.sum(any_nan_bpm)}")

    # --------------------------------------------------
    # 1. Median image of the stack
    # --------------------------------------------------
    stack_median = np.nanmedian(stack, axis=0)

    # --------------------------------------------------
    # 2. Temporal instability test
    #    This detects pixels unstable across frames.
    #    It does NOT catch pixels that are always hot/dead.
    # --------------------------------------------------
    start = time.time()

    deviations = np.abs(stack - stack_median[None, :, :])
    pixel_mad = np.nanmedian(deviations, axis=0)
    pixel_sigma = 1.4826 * pixel_mad

    finite_sigma = pixel_sigma[np.isfinite(pixel_sigma) & (pixel_sigma > 0)]
    if finite_sigma.size > 0:
        sigma_floor = max(np.nanmedian(finite_sigma) * mad_floor_fraction, 1e-6)
    else:
        sigma_floor = 1e-6

    pixel_sigma[~np.isfinite(pixel_sigma) | (pixel_sigma <= 0)] = sigma_floor

    temporal_outlier_stack = deviations > (
        temporal_sigma_thresh * pixel_sigma[None, :, :]
    )

    temporal_count = np.sum(temporal_outlier_stack, axis=0)
    temporal_bpm = temporal_count >= min_frames_bad_temporal

    print(
        f"Temporal unstable pixels: {np.sum(temporal_bpm)} "
        f"({time.time() - start:.2f}s)"
    )

    final_bpm |= temporal_bpm

    # --------------------------------------------------
    # 3. Mode-specific persistent tests
    # --------------------------------------------------
    dark_level_bpm = np.zeros((ny, nx), dtype=bool)
    flat_response_bpm = np.zeros((ny, nx), dtype=bool)
    flat_response = None

    if mode == "dark":
        # Persistent hot/cold dark pixels relative to whole dark median image.
        finite = np.isfinite(stack_median)

        global_dark_med = np.nanmedian(stack_median[finite])
        global_dark_sigma = robust_sigma_from_mad(stack_median[finite])

        dark_hot = stack_median > (
            global_dark_med + dark_hot_sigma * global_dark_sigma
        )
        dark_cold = stack_median < (
            global_dark_med - dark_cold_sigma * global_dark_sigma
        )

        dark_level_bpm = dark_hot | dark_cold
        dark_level_bpm[~finite] = True

        print(f"Persistent dark hot pixels: {np.sum(dark_hot)}")
        print(f"Persistent dark cold pixels: {np.sum(dark_cold)}")
        print(f"Persistent dark-level BPM: {np.sum(dark_level_bpm)}")

        final_bpm |= dark_level_bpm

    elif mode == "flat":
        # Persistent low/high-response pixels after removing illumination pattern.
        flat_med = stack_median.copy()

        finite = np.isfinite(flat_med)
        fill_value = np.nanmedian(flat_med[finite]) if np.any(finite) else 1.0

        flat_fill = flat_med.copy()
        flat_fill[~finite] = fill_value

        if normalize_flats:
            illum = median_filter(
                flat_fill,
                size=flat_norm_kernel_size,
                mode="reflect",
            )
            illum[~np.isfinite(illum) | (illum <= 1e-6)] = 1e-6
            flat_response = flat_fill / illum
        else:
            norm = np.nanmedian(flat_fill[finite]) if np.any(finite) else 1.0
            if norm <= 0:
                norm = 1.0
            flat_response = flat_fill / norm

        flat_low = flat_response < flat_low_response_thresh
        flat_high = flat_response > flat_high_response_thresh

        flat_response_bpm = flat_low | flat_high
        flat_response_bpm[~finite] = True

        print(f"Flat low-response pixels: {np.sum(flat_low)}")
        print(f"Flat high-response pixels: {np.sum(flat_high)}")
        print(f"Flat response BPM: {np.sum(flat_response_bpm)}")

        final_bpm |= flat_response_bpm

    # --------------------------------------------------
    # 4. Spatial local outlier test with persistence
    # --------------------------------------------------
    start = time.time()
    spatial_count = np.zeros((ny, nx), dtype=np.uint16)

    for i in range(n_frames):
        frame = stack[i].copy()
        original_bad = ~np.isfinite(frame)

        finite = np.isfinite(frame)
        fill_value = np.nanmedian(frame[finite]) if np.any(finite) else 0.0

        frame_fill = frame.copy()
        frame_fill[~finite] = fill_value

        if mode == "flat" and normalize_flats:
            illum = median_filter(
                frame_fill,
                size=flat_norm_kernel_size,
                mode="reflect",
            )
            illum[~np.isfinite(illum) | (illum <= 1e-6)] = 1e-6
            test_frame = frame_fill / illum
        else:
            test_frame = frame_fill

        local_med = median_filter(
            test_frame,
            size=spatial_kernel_size,
            mode="reflect",
        )

        abs_dev = np.abs(test_frame - local_med)

        local_mad = median_filter(
            abs_dev,
            size=spatial_kernel_size,
            mode="reflect",
        )

        local_sigma = 1.4826 * local_mad

        finite_local_sigma = local_sigma[
            np.isfinite(local_sigma) & (local_sigma > 0)
        ]

        if finite_local_sigma.size > 0:
            local_sigma_floor = max(
                np.nanmedian(finite_local_sigma) * mad_floor_fraction,
                1e-6,
            )
        else:
            local_sigma_floor = 1e-6

        local_sigma[
            ~np.isfinite(local_sigma) | (local_sigma <= 0)
        ] = local_sigma_floor

        spatial_outlier = abs_dev > (spatial_sigma_thresh * local_sigma)
        spatial_outlier |= original_bad

        spatial_count += spatial_outlier.astype(np.uint16)

    spatial_bpm = spatial_count >= min_frames_bad_spatial

    print(
        f"Spatial persistent outliers: {np.sum(spatial_bpm)} "
        f"({time.time() - start:.2f}s)"
    )

    final_bpm |= spatial_bpm

    # --------------------------------------------------
    # 5. Final summary
    # --------------------------------------------------
    print(f"Total unique bad pixels for {stack_name}: {np.sum(final_bpm)}")
    print(f"Fraction flagged: {100.0 * np.sum(final_bpm) / final_bpm.size:.4f}%")
    return final_bpm

#input is a cube of quicklook slope images, odd even corrected, not bpm corrected.

#sigma cutoff ifs dark = temporal_sigma_thresh=10.0, spatial_sigma_thresh=8.0
#sigma cutoff ifs flat = temporal_sigma_thresh=8.0, spatial_sigma_thresh=8.0

#sigma cutoff img dark = temporal_sigma_thresh=10.0, spatial_sigma_thresh=8.0
#sigma cutoff img flat = temporal_sigma_thresh=10.0, spatial_sigma_thresh=8.0

#bpm_from_darks = generate_bpm_robust_v2(
#    dark_stack,
#    stack_name="Dark Stack",
#    mode="dark",

#    temporal_sigma_thresh=10.0,
#    min_frames_bad_temporal=2,

#    spatial_sigma_thresh=8.0,
#    spatial_kernel_size=7,
#    min_frames_bad_spatial=2,

#    dark_hot_sigma=8.0,
#    dark_cold_sigma=8.0,

#    normalize_flats=False)

#bpm_from_flats = generate_bpm_robust_v2(
#    flat_stack,
#    stack_name="Flat Stack",
#    mode="flat",

#    temporal_sigma_thresh=8.0,
#    min_frames_bad_temporal=2,

#    spatial_sigma_thresh=8.0,
#    spatial_kernel_size=7,
#    min_frames_bad_spatial=2,

#    normalize_flats=True,
#    flat_norm_kernel_size=31,
#    flat_low_response_thresh=0.5,
#    flat_high_response_thresh=1.5)

#final_bpm = bpm_from_darks | bpm_from_flats

#fits.PrimaryHDU(final_bpm.astype(np.uint8)).writeto('filename.fits',overwrite=True)
#############R.matrix based .npz FILE bpm correction #########################

def bpm_correction_v1(bpmap):
    #calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
    #calib_path = str(files("scalesdrp").joinpath("calib"))+ "/"
    #if obsmode=='IMAGING':
    #    bpmap = pyfits.getdata(calib_path+'bpm_new_5.fits')
    #elif obsmode=='IFS':
    #    bpmap = pyfits.getdata(calib_path+'cd3_bpm_ifs_5mhz.fits')

    ypix = bpmap.shape[0]
    xpix = bpmap.shape[1]
    matrowinds = []
    matcolinds = []
    matvals = []
    for i in range(len(bpmap)):
        for j in range(len(bpmap[0])):
            if bpmap[i,j] == 1.0:
                vals = []
                weights = []
                indsx = []
                indsy = []
                sbox = 1
                goodbox = False
                while goodbox == False:
                    sbox += 2
                    #print(sbox)
                    xs,xe = j-sbox//2,j+sbox//2+1
                    ys,ye = i-sbox//2,i+sbox//2+1
                    if xs < 0:
                        xs = 0
                    if xe >= xpix:
                        xe = xpix
                    if ys < 0:
                        ys = 0
                    if ye >= xpix:
                        ye = xpix
                    #print(xs,xe,ys,ye)
                    box = bpmap[ys:ye,xs:xe]
                    if len(np.where(box!=1.0)[0]) > 3:
                        goodbox = True
                #plt.imshow(box)
                #plt.colorbar()
                #plt.show()
                #stop

                for yy in range(ys,ye):
                    for xx in range(xs,xe):
                        if bpmap[yy,xx]==0:
                            dist = np.sqrt((yy-i)**2 + (xx-j)**2)
                            weights.append(1/dist)
                            indsx.append(xx)
                            indsy.append(yy)
                avginds = np.ravel_multi_index((indsy,indsx),(ypix,xpix))
                vals = np.array(weights)/np.sum(weights)
                pixind = np.ravel_multi_index(([i],[j]),(ypix,xpix))
                #print(vals,np.sum(vals),indsx,indsy)
                #stop
                for k in range(len(vals)):
                    matrowinds.append(pixind[0])
                    matcolinds.append(avginds[k])
                    matvals.append(vals[k])
            elif bpmap[i,j] == 0.0:
                pixind = np.ravel_multi_index(([i],[j]),(ypix,xpix))
                matrowinds.append(pixind[0])
                matcolinds.append(pixind[0])
                matvals.append(1.0)
    rmat = sparse.csr_matrix((matvals,(matrowinds,matcolinds)),shape=(np.prod(bpmap.shape),np.prod(bpmap.shape)))
    #sparse.save_npz('bpmat_ifs1.npz',rmat)
    return rmat


def bpm_correction(bpmap, min_good_pixels=4, max_box_size=21):
    """
    Build a sparse matrix that replaces BPM pixels with an inverse-distance
    weighted average of nearby good pixels.

    bpmap: 2D array
        1 = bad pixel, 0 = good pixel
    """
    bpmap = np.asarray(bpmap)
    ypix, xpix = bpmap.shape

    matrowinds = []
    matcolinds = []
    matvals = []

    for i in range(ypix):
        for j in range(xpix):

            pixind = np.ravel_multi_index((i, j), (ypix, xpix))

            if bpmap[i, j] == 1:
                found_good_box = False
                sbox = 1

                while not found_good_box:
                    sbox += 2

                    if sbox > max_box_size:
                        # If no good neighbors found, leave pixel unchanged
                        matrowinds.append(pixind)
                        matcolinds.append(pixind)
                        matvals.append(1.0)
                        found_good_box = True
                        break

                    half = sbox // 2

                    xs = max(0, j - half)
                    xe = min(xpix, j + half + 1)

                    ys = max(0, i - half)
                    ye = min(ypix, i + half + 1)   # FIXED: use ypix

                    box = bpmap[ys:ye, xs:xe]

                    if np.sum(box == 0) >= min_good_pixels:
                        found_good_box = True

                if sbox <= max_box_size:
                    weights = []
                    indsy = []
                    indsx = []

                    for yy in range(ys, ye):
                        for xx in range(xs, xe):
                            if bpmap[yy, xx] == 0:
                                dist = np.sqrt((yy - i)**2 + (xx - j)**2)

                                if dist == 0:
                                    continue

                                weights.append(1.0 / dist)
                                indsy.append(yy)
                                indsx.append(xx)

                    weights = np.asarray(weights, dtype=float)

                    if weights.size > 0:
                        weights /= np.sum(weights)
                        avginds = np.ravel_multi_index((indsy, indsx), (ypix, xpix))

                        for colind, w in zip(avginds, weights):
                            matrowinds.append(pixind)
                            matcolinds.append(colind)
                            matvals.append(w)
                    else:
                        matrowinds.append(pixind)
                        matcolinds.append(pixind)
                        matvals.append(1.0)

            else:
                matrowinds.append(pixind)
                matcolinds.append(pixind)
                matvals.append(1.0)

    rmat = sparse.csr_matrix(
        (matvals, (matrowinds, matcolinds)),
        shape=(ypix * xpix, ypix * xpix),
    )
    #sparse.save_npz('bpmat_ifs.npz',rmat)
    return rmat
#############################################################################
def detect_transient_bad_pixels(
    image,
    master_bpm=None,
    kernel_size=5,
    sigma_thresh=7.0,
    mad_floor=1e-6,
    ignore_negative=False,
    return_diagnostics=False,
):
    """
    Detect transient bad pixels in a single 2D image using a local median + MAD test.

    A pixel is flagged if it is a strong local outlier relative to nearby pixels.
    This is useful for one-exposure defects that are not present in the master BPM.

    Parameters
    ----------
    image : 2D ndarray
        Input image.
    master_bpm : 2D bool ndarray or None, optional
        Persistent bad-pixel mask. These pixels are excluded from local statistics
        and are not re-flagged by the transient detector.
    kernel_size : int, optional
        Odd kernel size for local statistics.
    sigma_thresh : float, optional
        Outlier threshold in units of robust local sigma.
    mad_floor : float, optional
        Minimum local sigma floor to avoid division problems.
    ignore_negative : bool, optional
        If True, detect only positive outliers.
        If False, detect both positive and negative outliers.
    return_diagnostics : bool, optional
        If True, also return local median, local sigma, and significance map.

    Returns
    -------
    transient_mask : 2D bool ndarray
        True where a transient bad pixel is detected.

    If return_diagnostics=True, also returns:
    local_median : 2D ndarray
        Local median image.
    local_sigma : 2D ndarray
        Local robust sigma image.
    significance : 2D ndarray
        Signed significance map: (image - local_median) / local_sigma
    """
    image = np.asarray(image, dtype=np.float32)

    if kernel_size % 2 == 0:
        kernel_size += 1

    if master_bpm is None:
        master_bpm = np.zeros_like(image, dtype=bool)
    else:
        master_bpm = np.asarray(master_bpm, dtype=bool)
        if master_bpm.shape != image.shape:
            raise ValueError(
                f"Shape mismatch: image {image.shape}, master_bpm {master_bpm.shape}"
            )

    transient_mask = np.zeros_like(image, dtype=bool)

    # Build a working image for local statistics
    work = image.copy()
    finite_vals = work[np.isfinite(work) & (~master_bpm)]

    #if there is no usable pixels present
    if finite_vals.size == 0:
        if return_diagnostics:
            local_median = np.full_like(image, np.nan, dtype=np.float32)
            local_sigma = np.full_like(image, np.nan, dtype=np.float32)
            significance = np.full_like(image, np.nan, dtype=np.float32)
            return transient_mask, local_median, local_sigma, significance
        return transient_mask

    #compute the global median fill value (temporary)
    global_med = np.median(finite_vals)

    # Exclude master BPM and non-finite pixels from local-stat estimation
    excluded_for_stats = master_bpm | (~np.isfinite(work))
    #assign the global median vaue to the excluded pixels
    work[excluded_for_stats] = global_med

    # Local median
    local_median = median_filter(work, size=kernel_size, mode="reflect")

    # Local robust sigma from MAD, 'reflect' will take care of boundary artifacts
    #local absolute deviation
    abs_dev = np.abs(work - local_median)
    #natural variation of local deviation
    local_mad = median_filter(abs_dev, size=kernel_size, mode="reflect")
    #Gaussian distribution
    local_sigma = 1.4826 * local_mad
    #adding a minimum floor to avoid artificial errors
    local_sigma[~np.isfinite(local_sigma) | (local_sigma < mad_floor)] = mad_floor

    #signiificance map
    significance = (image - local_median) / local_sigma
    significance[~np.isfinite(significance)] = np.nan

    if ignore_negative:
        transient_mask = significance > sigma_thresh
    else:
        transient_mask = np.abs(significance) > sigma_thresh

    # Do not re-flag master BPM pixels here; keep the roles separate
    transient_mask[master_bpm] = False

    # Always flag non-finite pixels in this image as transient bad
    transient_mask[~np.isfinite(image)] = True

    if return_diagnostics:
        return transient_mask, local_median, local_sigma, significance

    return transient_mask
