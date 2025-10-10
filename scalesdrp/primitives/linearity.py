import numpy as np
from astropy.io import fits
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pkg_resources

DQ_FLAGS = {
    'GOOD': 0,
    'DO_NOT_USE': 1, # bad pixel, do not use
    'SATURATED': 2, #pixel saturated during the exposure
    'JUMP_DET': 4, #jump detected during the exposure
    'DROPOUT': 8, #data lost in transmission
    'OUTLIER': 16, #flagged as outlier
    'PERSISTENCE': 32, #High persistance
    'AD_FLOOR': 64,  #Below A/D floor
    'CHARGELOSS': 128, # charge migration
    'HOT': 256, 
    'DEAD': 512,
    'NO_LIN_CORR': 1024,
}
def create_saturation_map_by_slope(
    science_ramp,
    skip_reads=3,
    slope_threshold=2.0,
    smoothing_window=3):
    n_reads, n_rows, n_cols = science_ramp.shape
    if smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be an odd number.")
    saturation_map = np.full((n_rows, n_cols), 1e9, dtype=np.float32)
    for r, c in tqdm(np.ndindex(n_rows, n_cols), total=n_rows * n_cols, desc="Creating Saturation Map for linearity correction"):
        pixel_ramp = science_ramp[:, r, c]
        if len(pixel_ramp) < smoothing_window + 1:
            continue
        diffs = np.diff(pixel_ramp)
        smoothed_diffs = np.convolve(diffs, np.ones(smoothing_window)/smoothing_window, mode='valid')
        flat_indices = np.where(smoothed_diffs < slope_threshold)[0]
        if len(flat_indices) > 0:
            first_flat_idx = flat_indices[0]
            saturation_read_index = first_flat_idx + (smoothing_window // 2) + 1
            if saturation_read_index < skip_reads:
                saturation_map[r, c] = 0.0
            elif saturation_read_index < len(pixel_ramp):
                saturation_level = pixel_ramp[saturation_read_index]
                saturation_map[r, c] = saturation_level
    return saturation_map

def create_group_dq(science_ramp, saturation_map):
    group_dq = np.zeros(science_ramp.shape, dtype=np.uint8)
    saturation_mask = (science_ramp >= saturation_map[np.newaxis, :, :])
    group_dq[saturation_mask] = np.bitwise_or(group_dq[saturation_mask], DQ_FLAGS['SATURATED'])
    n_reads = science_ramp.shape[0] 
    for i in range(1, n_reads):
        previous_read_saturation = np.bitwise_and(group_dq[i-1], DQ_FLAGS['SATURATED'])
        group_dq[i] = np.bitwise_or(group_dq[i], previous_read_saturation)
    num_sat_final = np.count_nonzero(np.bitwise_and(group_dq[-1], DQ_FLAGS['SATURATED']))
    print(f"--- GROUPDQ created. Found {num_sat_final} saturated pixels in the final read. ---")
    return group_dq


def apply_linearity_correction_twopart(science_ramp,group_dq,obsmode):
    """
    Applies a two-part polynomial correction with robust DQ handling.

    This function uses two sets of coefficients and a cutoff level to linearize
    the signal. It is a vectorized implementation that incorporates handling
    for saturated pixels and bad pixels identified in the input and reference
    file data quality (DQ) arrays.

    Parameters
    ----------
    science_ramp : np.ndarray
        The 3D science data ramp (n_reads, n_rows, n_cols).
    group_dq : np.ndarray
        The 3D group DQ array, used to identify saturated pixels.
    pixel_dq : np.ndarray
        The 2D pixel DQ array for the science data.
    linearity_hdul : astropy.io.fits.HDUList
        The FITS HDUList object containing 'COEFFS1', 'COEFFS2', 'CUTOFFS',
        and 'DQ' extensions.

    Returns
    -------
    corrected_ramp : np.ndarray
        The linearly corrected science data ramp.
    output_pixel_dq : np.ndarray
        The updated 2D pixel data quality array.
    """
    calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
    with fits.open(calib_path + "bpm_new_5.fits") as hdul:
        pixel_dq = hdul[0].data.astype(np.uint32)

    linearity_hdul_img  = fits.open(calib_path+"linearity_coeffs_img.fits")
    linearity_hdul_ifs  = fits.open(calib_path+"linearity_coeffs_img.fits")

    if obsmode == 'IMAGING':
        linearity_hdul = linearity_hdul_img
    else:
        linearity_hdul = linearity_hdul_ifs

    coeffs1 = linearity_hdul['COEFFS1'].data.copy()
    coeffs2 = linearity_hdul['COEFFS2'].data.copy()
    cutoffs = linearity_hdul['CUTOFFS'].data
    output_pixel_dq = pixel_dq.copy() #bpm
    lin_dq = np.zeros_like(cutoffs, dtype=np.uint32)
    output_pixel_dq = np.bitwise_or(output_pixel_dq,lin_dq)
    flagged_mask = np.bitwise_and(lin_dq,DQ_FLAGS["NO_LIN_CORR"]).astype(bool)
    nan_mask = np.any(np.isnan(coeffs1), axis=0) | np.any(np.isnan(coeffs2), axis=0)
    zero_mask = (coeffs1[1, :, :] == 0) if coeffs1.shape[0] > 1 else np.zeros_like(flagged_mask)
    bad_pixel_mask = flagged_mask | nan_mask | zero_mask
    output_pixel_dq[bad_pixel_mask] = np.bitwise_or(output_pixel_dq[bad_pixel_mask], DQ_FLAGS["NO_LIN_CORR"])
    if np.any(bad_pixel_mask):
        ben_cor = np.zeros(coeffs1.shape[0], dtype=coeffs1.dtype)
        if len(ben_cor) > 1: 
            ben_cor[1] = 1.0
        coeffs1[:, bad_pixel_mask] = ben_cor[:, np.newaxis]
        coeffs2[:, bad_pixel_mask] = ben_cor[:, np.newaxis]
    n_coeffs1 = coeffs1.shape[0]
    corrected_vals_1 = np.full_like(science_ramp, coeffs1[n_coeffs1 - 1])
    for j in range(n_coeffs1 - 2, -1, -1):
        corrected_vals_1 = corrected_vals_1 * science_ramp + coeffs1[j]
    n_coeffs2 = coeffs2.shape[0]
    corrected_vals_2 = np.full_like(science_ramp, coeffs2[n_coeffs2 - 1])
    for j in range(n_coeffs2 - 2, -1, -1):
        corrected_vals_2 = corrected_vals_2 * science_ramp + coeffs2[j]
    below_cutoff_mask = science_ramp <= cutoffs
    corrected_ramp = np.where(below_cutoff_mask, corrected_vals_1, corrected_vals_2)
    saturation_mask = np.bitwise_and(group_dq, DQ_FLAGS["SATURATED"]).astype(bool)
    final_corrected_ramp = np.where(saturation_mask, science_ramp, corrected_ramp)
    print('linearity correction is applied')
    return final_corrected_ramp, output_pixel_dq


def run_linearity_workflow(
    science_ramp,
    saturation_map,
    obsmode):        
    group_dq = create_group_dq(science_ramp, saturation_map)
    corrected_ramp, output_pixel_dq = apply_linearity_correction_twopart(
        science_ramp=science_ramp, group_dq=group_dq,obsmode=obsmode)
    return corrected_ramp, output_pixel_dq, group_dq

