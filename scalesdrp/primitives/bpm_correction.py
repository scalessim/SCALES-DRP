import pandas as pd
import numpy as np
import pickle
from astropy.io import fits
import warnings
import pkg_resources
from scipy import sparse
import astropy.io.fits as pyfits
import os
import matplotlib.pyplot as plt
import time
from scipy.signal import savgol_filter
from scipy.ndimage import convolve, median_filter
from scipy.interpolate import griddata
from scipy.stats import median_abs_deviation
from scipy.ndimage import median_filter
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve as astropy_convolve
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
import pkg_resources
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scalesdrp.core.matplot_plotting import mpl_plot, mpl_clear

def correct_local_defects(image_data_to_correct, bad_pixel_mask, **kwargs):
	verbose = kwargs.get('verbose', True)
	if verbose: print("--- PASS 1: Starting Improved Local Iterative Correction ---")
	corrected_image = image_data_to_correct.copy()
	remaining_bpm = bad_pixel_mask.copy()
	initial_bad_count = np.sum(remaining_bpm)
	if initial_bad_count == 0:
		return corrected_image, remaining_bpm

	current_box_size = kwargs.get('initial_box_size', 5)
	max_box_size = kwargs.get('max_box_size', 11)
	min_good_neighbors_frac = kwargs.get('min_good_neighbors_frac', 0.3) #finetune needed
	max_iterations = kwargs.get('max_iterations', 10) # finetune needed
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
		if num_fixed_this_growth_step == 0:
			stalled_pixels_mask |= remaining_bpm
			current_box_size += 2
			if verbose: print(f"Stalled. Growing box to {current_box_size}x{current_box_size}.")
	final_uncorrected_mask = remaining_bpm | stalled_pixels_mask
	if verbose: print(f"--- Pass 1 Finished. Uncorrectable local pixels: {np.sum(final_uncorrected_mask)} ---")
	return corrected_image, final_uncorrected_mask
    
def fill_global_defects(image, bad_pixel_mask):
	num_defects = np.sum(bad_pixel_mask)
	if num_defects == 0:
		return image.copy()
	image_with_nans = image.copy()
	image_with_nans[bad_pixel_mask] = np.nan
	kernel_size_stddev = 3
	kernel = Gaussian2DKernel(x_stddev=kernel_size_stddev)
	inpainted_image = interpolate_replace_nans(image_with_nans, kernel)
	return inpainted_image

def apply_full_correction(image_to_correct,obsmode,pass1_kwargs={}):
	print("Starting BPM corrections")
	calib_path = pkg_resources.resource_filename('scalesdrp','calib/')

	if obsmode == 'IMAGING':
		with fits.open(calib_path + "bpm_new_5.fits") as hdul:
			master_bpm = hdul[0].data.astype(np.uint32)
	else:
		with fits.open(calib_path + "bpm_new_5.fits") as hdul:
			master_bpm = hdul[0].data.astype(np.uint32)

	corrected_pass1, large_defects_mask = correct_local_defects(
		image_to_correct,
		master_bpm,
		**pass1_kwargs)

	fully_corrected_image = fill_global_defects(
		corrected_pass1,
		large_defects_mask)
	return fully_corrected_image

######################### Creating BPM ###########################
def generate_bpm(
	image_stack: np.ndarray,
	stack_name: str = "Image Stack",
	spatial_brightness_factor: float = 5.0,
	spatial_kernel_size: int = 5,
	temporal_sigma_thresh: float = 5.0,
	min_frames_for_temporal: int = 3,
	plot_results: bool = False) -> np.ndarray:
    """
    Generates a Bad Pixel Mask (BPM) from a stack of images.
    - Temporal outliers are found using a standard deviation.
    - Spatial outliers are found using a RELATIVE BRIGHTNESS test: a pixel
    is flagged if it is N times brighter or dimmer than its local median.
    Args:
        image_stack (np.ndarray): 3D stack of images (N_frames, Height, Width).
        stack_name (str): Name for logging/plotting.
        spatial_brightness_factor (float): A pixel is bad if its value is > this
            factor times the local median, or < the local median / this factor.
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
    temporal_bpm = np.zeros_like(final_bpm)
    if n_frames >= min_frames_for_temporal and temporal_sigma_thresh > 0:
    	pixel_medians = np.nanmedian(image_stack, axis=0, keepdims=True)
    	pixel_stds_map = np.nanstd(image_stack, axis=0)
    	valid_stds = pixel_stds_map[np.isfinite(pixel_stds_map) & (pixel_stds_map > 0)]
    	std_floor = np.median(valid_stds) * 0.01 if len(valid_stds) > 0 else 1e-5
    	pixel_stds_map[~np.isfinite(pixel_stds_map) | (pixel_stds_map <= 0)] = max(std_floor, 1e-5)
    	deviations = np.abs(image_stack - pixel_medians)
    	is_outlier_stack = deviations > (temporal_sigma_thresh * pixel_stds_map)
    	temporal_bpm = np.any(is_outlier_stack, axis=0)
    	final_bpm |= temporal_bpm
    spatial_bpm = np.zeros_like(final_bpm)
    if spatial_brightness_factor > 1 and spatial_kernel_size >= 3:
    	for i in range(n_frames):
    		frame = image_stack[i].copy()
    		original_nans = np.isnan(frame)
    		global_median_val = np.nanmedian(frame)
    		if not np.isfinite(global_median_val): global_median_val = 0
    		frame_no_nan = np.nan_to_num(frame, nan=global_median_val)
    		local_median = median_filter(frame_no_nan, size=spatial_kernel_size, mode='reflect')
    		local_median[local_median <= 1e-9] = 1e-9
    		with np.errstate(divide='ignore', invalid='ignore'):
    			is_hot_pixel = frame > (spatial_brightness_factor * local_median)
    			is_cold_pixel = frame < (local_median / spatial_brightness_factor)
    		is_outlier_this_frame = is_hot_pixel | is_cold_pixel
    		is_outlier_this_frame |= original_nans 
    		spatial_bpm |= is_outlier_this_frame
    	final_bpm |= spatial_bpm
    return final_bpm





