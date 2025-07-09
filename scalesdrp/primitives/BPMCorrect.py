from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer
import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve, median_filter
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve as astropy_convolve
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
import pkg_resources


class BPMCorrect(BasePrimitive):
    """
    Performs bad pixel correction by replacing the median filter value of 
    a box size varing from 5-11 with a minimum good neigbhours of 0.3% 
    on the box defined. The uncorrected BPM positions will replace with an interpolation.
    Args:
        BPM: 2D boolen bad pixel map
        data_image: The (H,W) input data image.
    Returns:
        - bad pixel corrected input 2D image.
    """
    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger
        self.logger.info("Starting Bad Pixel Correction")

    def _perform(self):

        def correct_local_defects_pass1_improved(image_data_to_correct, bad_pixel_mask, **kwargs):
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


        def fill_global_defects_pass2_inpainting(image, bad_pixel_mask):
            num_defects = np.sum(bad_pixel_mask)
            if num_defects == 0:
                return image.copy()
    
            image_with_nans = image.copy()
            image_with_nans[bad_pixel_mask] = np.nan
            kernel_size_stddev = 3 
            kernel = Gaussian2DKernel(x_stddev=kernel_size_stddev)
            inpainted_image = interpolate_replace_nans(image_with_nans, kernel)
            return inpainted_image

        def apply_full_correction_improved(image_to_correct, master_bpm, pass1_kwargs={}):
            print(f"--- Applying full two-pass correction to image ---")
    
            corrected_pass1, large_defects_mask = correct_local_defects_pass1_improved(
                image_to_correct,
                master_bpm,
                **pass1_kwargs)
    
            fully_corrected_image = fill_global_defects_pass2_inpainting(
                corrected_pass1,
                large_defects_mask)
            return fully_corrected_image

        calib_path = pkg_resources.resource_filename('scalesdrp','calib/')
        BPM = fits.getdata(calib_path+'bpm.fits')
        final_bpm = BPM.astype(bool)
        input_data = self.action.args.ccddata.data
        data_corrected = apply_full_correction_improved(input_data, final_bpm)
        self.action.args.ccddata.data = data_corrected

        log_string = BPMCorrect.__module__
        self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)

        scales_fits_writer(self.action.args.ccddata,
            table=self.action.args.table,
            output_file=self.action.args.name,
            output_dir=self.config.instrument.output_directory,
            suffix="bpm")

        return self.action.args
    # END: class BPMCorrect()
