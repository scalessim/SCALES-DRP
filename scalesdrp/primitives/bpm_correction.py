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
import pkg_resources
import time

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
    calib_path = pkg_resources.resource_filename('scalesdrp','calib/')

    if obsmode=='IMAGING':
    	master_bpm = fits.getdata(calib_path+'cd3_bpm_ifs_5mhz.fits').astype(bool)
    else:
    	master_bpm = fits.getdata(calib_path+'cd3_bpm_ifs_5mhz.fits').astype(bool)

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

######################### Creating BPM ###########################

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



