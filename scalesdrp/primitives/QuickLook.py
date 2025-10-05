from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer

import pandas as pd
import numpy as np
import pickle
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pkg_resources
import os
from scipy.optimize import minimize
from scipy import sparse
import astropy.io.fits as pyfits
from scipy.sparse import load_npz
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time
import scalesdrp.primitives.fitramp as fitramp
import scalesdrp.primitives.robust as robust
from scipy.optimize import leastsq
import pkg_resources
from scipy.signal import savgol_filter

class QuickLook(BasePrimitive):
    """
	Quicklook  extraction : produce 3d cube
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger
        print("Optimal Extract object created")

    def slope_fit_weighted(self,ims,read_time):
        """
        Weighted quadratic ramp fit: y(t) = a + b*t + c*t^2
        Saturated reads get weight = 0.
        Returns B (the slope term) as an image.
    
        Parameters
        ----------
        ims : ndarray
            Input data cube of shape (N, Y, X).
        thresh : float
            Saturation threshold (default=4096 DN).
    
        Returns
        -------
        B : ndarray
            2D slope image of shape (Y, X).
        """
        start_time2 = time.time()
        thresh = 4096.0
        N, Y, X = ims.shape
        P = Y * X
        t = np.arange(N, dtype=np.float64)*read_time
        #print(t)

        # Normalize time axis for numerical stability
        t = (t - t.mean()) / t.std()

        # Design matrix [1, t, t^2] shape (N, 3)
        M = np.stack([np.ones(N), t, t**2], axis=1)  # (N, 3)

        # Flatten spatial dims
        y = ims.reshape(N, P)                        # (N, P)

        # Weights: 1 if below thresh, else 0
        w = (ims < thresh).astype(np.float64).reshape(N, P)

        # Apply sqrt(weights) correctly
        Mw = M[:, :, None] * np.sqrt(w[:, None, :])  # (N, 3, P)
        yw = y * np.sqrt(w)                          # (N, P)

        # Compute normal equations in batch
        XtX = np.einsum("nip, njp -> ijp", Mw, Mw)   # (3, 3, P)
        Xty = np.einsum("nip, np -> ip", Mw, yw)     # (3, P)

        # Solve for coefficients (a, b, c) per pixel
        coeffs = np.full((3, P), np.nan, dtype=np.float64)
        for k in range(P):
            try:
                coeffs[:, k] = np.linalg.solve(XtX[:, :, k], Xty[:, k])
            except np.linalg.LinAlgError:
                # Singular matrix → leave NaN
                pass
        A, B, C = coeffs.reshape(3, Y, X)
        end_time2 = time.time()
        t2 = (end_time2 - start_time2)
        self.logger.info(f"quicklook ramp fitting finished in {t2:.4f} seconds.")
        return B

    def fits_writer_steps(self,data,header,output_dir,input_filename,suffix,overwrite=True):
        base_name = os.path.basename(input_filename)
        file_root, file_ext = os.path.splitext(base_name)
        output_filename = f"{file_root}{suffix}{file_ext}"
        redux_output_dir = os.path.join(output_dir, 'redux')
        os.makedirs(redux_output_dir, exist_ok=True)
        output_path = os.path.join(redux_output_dir, output_filename)
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_path, overwrite=overwrite)
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
        plt.imshow(data,origin='lower')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(output_path)
        return 

    def _perform(self):
        self.logger.info("+++++++++++ Quicklook Started +++++++++++")

        ols_t_global = None

        def ols_pack_parms(a, b, c): 
            return np.array([a, b, c])

        def ols_unpack_parms(p):
            a_p, b_p, c_p = p
            return a_p, b_p, c_p

        def ols_model_fn(p_model): # Quadratic model
            a_m, b_m, c_m = ols_unpack_parms(p_model)
            global ols_t_global
            if ols_t_global is None:
                raise ValueError("Global time array 'ols_t_global' for OLS model_fn is not set.")
            return a_m + b_m * ols_t_global + c_m * ols_t_global**2

        def apply_linearity_correction(raw_ramp_frames,num_sample_pix=5000):
            """
            Loads the raw ramp data, derives a non-linearity correction.
            Args:
                raw_ramp_frames: the 3D ramp data cube.
                num_sample_pix (int, optional):
                Number of random pixels to use for characterization. Defaults to 5000.
            Returns:
                tuple: two arrays for the correction curve:median_fluence and median_correction.
            """
            
            num_frames, height, width = raw_ramp_frames.shape
            pixel_max_value = 65536
            sample_y = np.random.randint(0, height, size=num_sample_pix)
            sample_x = np.random.randint(0, width, size=num_sample_pix)
            sample_ramps = raw_ramp_frames[:, sample_y, sample_x].T
            linear_slope_est = sample_ramps[:, 1] - sample_ramps[:, 0]
            frame_times = np.arange(1, num_frames + 1)
            biases = sample_ramps[:, 0] - (linear_slope_est * frame_times[0])
            measured_fluence = sample_ramps - biases[:, np.newaxis]
            ideal_ramps = linear_slope_est[:, np.newaxis] * frame_times
            corrections = ideal_ramps - measured_fluence
            fluence_flat = measured_fluence.flatten()
            corr_flat = corrections.flatten()
            bins = np.linspace(0, pixel_max_value, num=200)
            bin_indices = np.digitize(fluence_flat, bins)
            median_fluence=[]
            median_corr=[]
            for i in range(1, len(bins)):
                in_bin = (bin_indices == i)
                if np.any(in_bin):
                    median_fluence.append(np.median(fluence_flat[in_bin]))
                    median_corr.append(np.median(corr_flat[in_bin]))
            median_fluence = np.array(median_fluence)
            median_corr = np.array(median_corr)
            return median_fluence, median_corr

        def fit_slope_image(reads_cube,read_times,read_noise_sigmas):
            """
            Performs a weighted straight-line fit for each pixel in a data cube of reads.
            Args:
                reads_cube:
                    A 3D NumPy array of shape (N, H, W) where N is the number of reads,
                    and H and W are the height and width of the image.
                read_times:
                    A 1D NumPy array of length N, containing the individual time
                    duration for each read.
                read_noise_sigmas:
                    A 2D NumPy array of shape (H, W) containing the read noise standard
                    deviation for each pixel.
            Returns:
                A 2D NumPy array of shape (H, W) representing the best-fit slope image.
            """
            num_reads, height, width = reads_cube.shape
            x = np.cumsum(read_times)
            x_broadcast = x.reshape(num_reads, 1, 1)
            epsilon = 1e-10
            weights = 1.0 / (read_noise_sigmas**2 + epsilon)
            sum_w = num_reads * weights
            sum_w_x = np.sum(weights * x_broadcast, axis=0)
            sum_w_y = np.sum(weights * reads_cube, axis=0)
            sum_w_x_sq = np.sum(weights * x_broadcast**2, axis=0)
            sum_w_xy = np.sum(weights * x_broadcast * reads_cube, axis=0)
            numerator = sum_w_xy - (sum_w_y * sum_w_x) / sum_w
            denominator = sum_w_x_sq - (sum_w_x**2) / sum_w
            slope_image = numerator / (denominator + epsilon)
            return slope_image

        def reffix_hxrg(cube, nchans=4, in_place=False, fixcol=True, **kwargs):
            """Reference pixel correction function
            This function performs a reference pixel correction
            on HAWAII-[1,2,4]RG detector data read out using N outputs.
            Top and bottom reference pixels are used first to remove 
            channel offsets.
            Parameters
            ----------
            cube : ndarray
                Input datacube. Can be two or three dimensions (nz,ny,nx).
            in_place : bool
                Perform calculations in place. Input array is overwritten.
            nchans : int
                Number of output amplifier channels in the detector. Default=4.
            fixcol : bool
                Perform reference column corrections?
            Keyword Args
            ------------
            altcol : bool
                Calculate separate reference values for even/odd columns. (default: True)
            supermean : bool
                Add back the overall mean of the reference pixels. (default: False)
            top_ref : bool
                Include top reference rows when correcting channel offsets. (default: True)
            bot_ref : bool
                Include bottom reference rows when correcting channel offsets. (default: True)
            ntop : int
                Specify the number of top reference rows. (default: 4)
            nbot : int
                Specify the number of bottom reference rows. (default: 4)
            mean_func : func
                Function used to calculate averages. (default: `robust.mean`)
            left_ref : bool
                Include left reference cols when correcting 1/f noise. (default: True)
            right_ref : bool
                Include right reference cols when correcting 1/f noise. (default: True)
            nleft : int
                Specify the number of left reference columns. (default: 4)
            nright : int
                Specify the number of right reference columns. (default: 4)
            perint : bool
                Smooth side reference pixel per integration, otherwise do frame-by-frame.
                (default: False)
            avg_type :str
                Type of side column averaging to perform to determine ref pixel drift. 
                Allowed values are 'pixel', 'frame', or 'int' (default: 'frame'):
                    * 'int'   : Subtract the avg value of all side ref pixels in ramp.
                    * 'frame' : For each frame, get avg of side ref pixels and subtract framewise.
                    * 'pixel' : For each ref pixel, subtract its avg value from all frames.
            savgol : bool
                Use Savitsky-Golay filter method rather than FFT. (default: True)
            winsize : int
                Size of the window filter. (default: 31)
            order : int
                Order of the polynomial used to fit the samples. (default: 3)
            """
            ndim = len(cube.shape)
            if 'float' not in cube.dtype.name:
                type_in = cube.dtype.name
                copy = (not in_place)
                cube = cube.astype(float, copy=copy)
                type_out = cube.dtype.name

            if not in_place:
                cube = np.copy(cube)

            # Remove channel offsets
            cube = reffix_amps(cube, nchans=nchans, in_place=True, **kwargs)

            # Fix 1/f noise using vertical reference pixels
            if fixcol:
                cube = ref_filter(cube, nchans=nchans, in_place=True, **kwargs)
            return cube

        def reffix_amps(cube, nchans=4, in_place=True, altcol=True, supermean=False,
            top_ref=True, bot_ref=True, ntop=4, nbot=4, **kwargs):
            """Correct amplifier offsets
            Matches all amplifier outputs of the detector to a common level.
            This routine subtracts the average of the top and bottom reference rows
            for each amplifier and frame individually.
            By default, reference pixel corrections are performed in place since it's
            faster and consumes less memory.
            Parameters
            ----------
            cube : ndarray
                Input datacube. Can be two or three dimensions (nz,ny,nx).
            nchans : int
                Number of output amplifier channels in the detector. Default=4.
            altcol : bool
                Calculate separate reference values for even/odd columns.
            supermean : bool
                Add back the overall mean of the reference pixels.
            in_place : bool
                Perform calculations in place. Input array is overwritten.
            top_ref : bool
                Include top reference rows when correcting channel offsets.
            bot_ref : bool
                Include bottom reference rows when correcting channel offsets.
            ntop : int
                Specify the number of top reference rows.
            nbot : int
                Specify the number of bottom reference rows.
            Keyword Args
            ------------
            mean_func : func
                Function used to calculate averages.
            """
            if not in_place:
                cube = np.copy(cube)
            ndim = len(cube.shape)
            if ndim==2:
                ny,nx = cube.shape
                nz = 1
                cube = cube.reshape((nz,ny,nx))
            elif ndim==3:
                nz, ny, nx = cube.shape
            else:
                raise ValueError('Input data can only have 2 or 3 dimensions.Found {} dimensions.'.format(ndim))        
            chsize = int(nx / nchans)
            # Number of reference rows to use
            # Set nt or nb equal to 0 if we don't want to use either
            nt = ntop if top_ref else 0
            nb = nbot if bot_ref else 0
            if (nt+nb)==0: 
                print("No reference pixels available for use. Returning...")
                return
            # Slice out reference pixels
            refs_bot = cube[:,:nb,:]
            refs_top = cube[:,-nt:,:]
            if nt==0:
                refs_all = refs_bot
            elif nb==0:
                refs_all = refs_top
            else:
                refs_all = np.hstack((refs_bot, refs_top))
            assert refs_all.shape[1] == (nb+nt)

            # Supermean
            # the average of the average is the DC level of the output channel
            smean = robust.mean(refs_all) if supermean else 0.0
    
            # Calculate avg reference values for each frame and channel
            refs_amps_avg = calc_avg_amps(refs_all, cube.shape, nchans=nchans, altcol=altcol, **kwargs)
        
            for ch in range(nchans):
                # Channel indices
                ich1 = ch*chsize
                ich2 = ich1 + chsize
                #print(ich1)
                # In-place subtraction of channel medians
                if altcol:
                    for i in range(nz):
                        cube[i,:,ich1:ich2-1:2] -= refs_amps_avg[0][ch,i]
                        cube[i,:,ich1+1:ich2:2] -= refs_amps_avg[1][ch,i]
                else:
                    for i in range(nz):
                        cube[i,:,ich1:ich2] -= refs_amps_avg[ch,i]
            # Add back supermean
            if supermean: 
                cube += smean
            cube = cube.squeeze()
            return cube

        def ref_filter(cube, nchans=4, in_place=True, avg_type='frame', perint=False, 
            edge_wrap=False, left_ref=True, right_ref=True, nleft=4, nright=4, **kwargs):
            """Optimal Smoothing
            Performs an optimal filtering of the vertical reference pixel to 
            reduce 1/f noise (horizontal stripes).
            FFT method adapted from M. Robberto IDL code:
            http://www.stsci.edu/~robberto/Main/Software/IDL4pipeline/
            Parameters
            ----------
            cube : ndarray
                Input datacube. Can be two or three dimensions (nz,ny,nx).
            nchans : int
                Number of output amplifier channels in the detector. Default=4.
            in_place : bool
                Perform calculations in place. Input array is overwritten.    
            perint : bool
                Smooth side reference pixel per integration, 
                otherwise do frame-by-frame.
            avg_type : str
                Type of ref col averaging to perform. Allowed values are
                'pixel', 'frame', or 'int'.
            left_ref : bool
                Include left reference cols when correcting 1/f noise.
            right_ref : bool
                Include right reference cols when correcting 1/f noise.
            nleft : int
                Specify the number of left reference columns.
            nright : int
                Specify the number of right reference columns.
            Keyword Arguments
            =================
            savgol : bool
                Using Savitsky-Golay filter method rather than FFT.
            winsize : int
                Size of the window filter.
            order : int
                Order of the polynomial used to fit the samples.
            mean_func : func
                Function to use to calculate averages of reference columns.
            """               
            if not in_place:
                cube = np.copy(cube)
            ndim = len(cube.shape)
            if ndim==2:
                ny,nx = cube.shape
                nz = 1
                cube = cube.reshape((nz,ny,nx))
            elif ndim==3:
                nz, ny, nx = cube.shape
            else:
                raise ValueError('Input data can only have 2 or 3 dimensions. Found {} dimensions.'.format(ndim))        

            # Number of reference rows to use
            # Set nt or nb equal to 0 if we don't want to use either
            nl = nleft  if left_ref  else 0
            nr = nright if right_ref else 0
            assert nl>=0, 'Number of left reference pixels must not be negative.'
            assert nr>=0, 'Number of right reference pixels must not be negative.'
            if (nl+nr)==0: 
                print("No reference pixels available for use. Returning...")
                return
            # Slice out reference pixel columns
            refs_left  = cube[:,:,:nl]  if nl>0 else None
            refs_right = cube[:,:,-nr:] if nr>0 else None
            refvals = calc_avg_cols(refs_left, refs_right, avg_type, **kwargs)

            # The delta time doesn't seem to make any difference in the final data product
            # Just for vizualization purposes...
            delt = 10E-6 * (nx/nchans + 12.)
            refvals_smoothed = calc_col_smooth(refvals, cube.shape, perint=perint, 
                edge_wrap=edge_wrap, delt=delt, **kwargs)
            # Final correction
            #for i,im in enumerate(cube): im -= refvals_smoothed[i].reshape([ny,1])
            cube -= refvals_smoothed.reshape([nz,ny,1])
            cube = cube.squeeze()
            return cube

        def calc_avg_amps(refs_all, data_shape, nchans=4, altcol=True, mean_func=robust.mean, **kwargs):
            """Calculate amplifier averages
            Save the average reference value for each amplifier in each frame.
            Assume by default that alternating columns are offset from each other,
            so we save two arrays: self.refs_amps_avg1 and self.refs_amps_avg2. 
            Each array has a size of (namp, ngroup).
            Parameters
            ----------
            refs_all : ndarray
                The top and/or bottom references pixels order 
                in a shape (nz, nref_rows, nx)
            data_shape : tuple
                Shape of the data array: (nz, ny, nx).
            nchans : int
                Number of amplifier output channels.
            altcol : bool
                Calculate separate reference values for even/odd columns? 
                Default=True.
            mean_func : func
                Function used to calculate averages.
            """
            nz, ny, nx = data_shape
            chsize = int(nx / nchans)
            if altcol:
                refs_amps_avg1 = []
                refs_amps_avg2 = []
                for ch in range(nchans):
                    # Channel indices
                    ich1 = ch*chsize
                    ich2 = ich1 + chsize
                    # Slice out alternating columns
                    refs_ch1 = refs_all[:,:,ich1:ich2-1:2].reshape((nz,-1))
                    refs_ch2 = refs_all[:,:,ich1+1:ich2:2].reshape((nz,-1))

                    # Take the resistant mean
                    chavg1 = mean_func(refs_ch1,axis=1)
                    chavg2 = mean_func(refs_ch2,axis=1)    
                    refs_amps_avg1.append(chavg1)
                    refs_amps_avg2.append(chavg2)
                return (np.array(refs_amps_avg1), np.array(refs_amps_avg2))
            else:
                refs_amps_avg = []
                for ch in range(nchans):
                    # Channel indices
                    ich1 = ch*chsize
                    ich2 = ich1 + chsize
                    refs_ch = refs_all[:,:,ich1:ich2].reshape((nz,-1))
                    # Take the resistant mean and reshape for broadcasting
                    chavg = mean_func(refs_ch,axis=1).reshape([-1,1,1])
                    refs_amps_avg.append(chavg)
                return np.array(refs_amps_avg)
        
        def calc_avg_cols(refs_left=None, refs_right=None, avg_type='frame',
            mean_func=np.median, **kwargs):
            """Calculate average of column references
            Determine the average values for the column references, which
            is subsequently used to estimate the 1/f noise contribution.
            Parameters
            ----------
            refs_left : ndarray
                Left reference columns.
            refs_right : ndarray
                Right reference columns.
            avg_type : str
                Type of ref column averaging to perform to determine ref pixel variation. 
                Allowed values are 'pixel', 'frame', or 'int'.
                'pixel' : For each ref pixel, subtract its avg value from all frames.
                'frame' : For each frame, get avg ref pixel values and subtract framewise.
                'int'   : Calculate avg of all ref pixels within the ramp and subtract.
            mean_func : func
                Function to use to calculate averages of reference columns
            """
            # Which function to use for calculating averages?
            # mean_func = robust.mean
            # mean_func = np.median

            # In this context, nl and nr are either 0 (False) or 1 (True)
            nl = 0 if refs_left is None else 1
            nr = 0 if refs_right is None else 1

            # Left and right reference pixels
            # Make a copy so as to not modify the original data?
            if nl>0: refs_left  = np.copy(refs_left)
            if nr>0: refs_right = np.copy(refs_right)

            # Set the average of left and right reference pixels to zero
            # By default, pixel averaging is best for large groups
            if avg_type is None:
                avg_type = 'frame'
            if refs_left is not None:
                nz, ny, nchan = refs_left.shape
            else:
                nz, ny, nchan = refs_right.shape
            # If there is only 1 frame, then we have to do "per frame" averaging.
            # Set to "per int", which produces the same result as "per frame" for nz=1.
            if nz==1:
                avg_type = 'int'
            # Remove average ref pixel values
            # Average over entire integration
            if 'int' in avg_type:
                if nl>0: refs_left  -= mean_func(refs_left)
                if nr>0: refs_right -= mean_func(refs_right)
            # Average over each frame
            elif 'frame' in avg_type:
                if nl>0: refs_left_mean  = mean_func(refs_left.reshape((nz,-1)), axis=1)
                if nr>0: refs_right_mean = mean_func(refs_right.reshape((nz,-1)), axis=1)
                # Subtract estimate of each ref pixel "intrinsic" value
                for i in range(nz):
                    if nl>0: refs_left[i]  -= refs_left_mean[i]
                    if nr>0: refs_right[i] -= refs_right_mean[i]
            # Take the average of each reference pixel 
            elif 'pix' in avg_type:
                if nl>0: refs_left_mean  = mean_func(refs_left, axis=0)
                if nr>0: refs_right_mean = mean_func(refs_right, axis=0)
                # Subtract estimate of each ref pixel "intrinsic" value
                for i in range(nz):
                    if nl>0: refs_left[i]  -= refs_left_mean
                    if nr>0: refs_right[i] -= refs_right_mean
            if nl==0:
                refs_side_avg = refs_right.mean(axis=2)
            elif nr==0:
                refs_side_avg = refs_left.mean(axis=2)    
            else:
                refs_side_avg = (refs_right.mean(axis=2) + refs_left.mean(axis=2)) / 2
            return refs_side_avg

        def calc_col_smooth(refvals, data_shape, perint=False, edge_wrap=False, 
            delt=5.24E-4, savgol=False, winsize=31, order=3, **kwargs):
            """Perform optimal smoothing of side ref pix 
            Generates smoothed version of column reference values.
            Smooths values from calc_avg_cols() via FFT.
            Parameters
            ----------
            refvals : ndarray
                Averaged column reference pixels
            data_shape : tuple
                Shape of original data (nz,ny,nx)    
            Keyword Arguments
            =================
            perint : bool
                Smooth side reference pixel per int, otherwise per frame.
            edge_wrap : bool
                Add a partial frames to the beginning and end of each averaged
                time series pixels in order to get rid of edge effects.
            delt : float
                Time between reference pixel samples. 
            savgol : bool
                Using Savitsky-Golay filter method rather than FFT.
            winsize : int
                Size of the window filter.
            order : int
                Order of the polynomial used to fit the samples.
            """
            nz,ny,nx = data_shape
            if perint: # per integration
                if edge_wrap: # Wrap around to avoid edge effects
                    refvals2 = np.vstack((refvals[0][::-1], refvals, refvals[-1][::-1]))
                    if savgol: # SavGol filter
                        refvals_smoothed2 = savgol_filter(refvals2.ravel(), winsize, order, delta=1)
                    else: # Or "optimal" smoothing algorithm
                        refvals_smoothed2 = smooth_fft(refvals2, delt)
                    refvals_smoothed = refvals_smoothed2[ny:-ny].reshape(refvals.shape)
                else:
                    if savgol:
                        refvals_smoothed = savgol_filter(refvals.ravel(), winsize, order, delta=1)
                    else:
                        refvals_smoothed = smooth_fft(refvals, delt)
                    refvals_smoothed = refvals_smoothed.reshape(refvals.shape)
            else:
                refvals_smoothed = []
                if edge_wrap: # Wrap around to avoid edge effects
                    for ref in refvals:
                        ref2 = np.concatenate((ref[:ny//2][::-1], ref, ref[ny//2:][::-1]))
                        if savgol:
                            ref_smth = savgol_filter(ref2, winsize, order, delta=1)
                        else:
                            ref_smth = smooth_fft(ref2, delt)
                        refvals_smoothed.append(ref_smth[ny//2:ny//2+ny])
                    refvals_smoothed = np.array(refvals_smoothed)
                else:
                    for ref in refvals:
                        if savgol:
                            ref_smth = savgol_filter(ref, winsize, order, delta=1)
                        else: 
                            ref_smth = smooth_fft(ref, delt)
                        refvals_smoothed.append(ref_smth)
                    refvals_smoothed = np.array(refvals_smoothed)
            return refvals_smoothed
    
        def smooth_fft(data, delt, first_deriv=False, second_deriv=False):
            """Optimal smoothing algorithm  
            Smoothing algorithm to perform optimal filtering of the 
            vertical reference pixel to reduce 1/f noise (horizontal stripes),
            based on the Kosarev & Pantos algorithm. This assumes that the
            data to be filtered/smoothed has been sampled evenly.
            If first_deriv is set, then returns two results
            if second_deriv is set, then returns three results.
            Adapted from M. Robberto IDL code:
            http://www.stsci.edu/~robberto/Main/Software/IDL4pipeline/
            Parameters
            ----------
            data : ndarray
                Signal to be filtered.
            delt : float
                Delta time between samples.
            first_deriv : bool
                Return the first derivative.    
            second_deriv : bool
                Return the second derivative (along with first).
            """
            Dat = data.flatten()
            N = Dat.size
            Pi2 = 2*np.pi
            OMEGA = Pi2 / (N*delt)
            X = np.arange(N) * delt
            ##------------------------------------------------
            ## Center and Baselinefit of the data
            ##------------------------------------------------
            Dat_m = Dat - np.mean(Dat)
            SLOPE = (Dat_m[-1] - Dat_m[0]) / (N-2)
            Dat_b = Dat_m - Dat_m[0] - SLOPE * X / delt
            ##------------------------------------------------
            ## Compute fft- / power- spectrum
            ##------------------------------------------------
            Dat_F = np.fft.rfft(Dat_b) #/ N
            Dat_P = np.abs(Dat_F)**2
            ##------------------------------------------------
            ## Noise spectrum from 'half' to 'full'
            ## Mind: half means N/4, full means N/2
            ##------------------------------------------------
            i1 = int((N-1) / 4)
            i2 = int((N-1) / 2) + 1
            Sigma = np.sum(Dat_P[i1:i2])
            Noise = Sigma / ((N-1)/2 - (N-1)/4)
            ##------------------------------------------------
            ## Get Filtercoeff. according to Kosarev/Pantos
            ## Find the J0, start search at i=1 (i=0 is the mean)
            ##------------------------------------------------
            J0 = 2
            for i in np.arange(1, int(N/4)+1):
                sig0, sig1, sig2, sig3 = Dat_P[i:i+4]
                if (sig0<Noise) and ((sig1<Noise) or (sig2<Noise) or (sig3<Noise)):
                    J0 = i
                    break
            ##------------------------------------------------
            ## Compute straight line extrapolation to log(Dat_P)
            ##------------------------------------------------
            ii = np.arange(1,J0+1)
            logvals = np.log(Dat_P[1:J0+1])
            XY = np.sum(ii * logvals)
            XX = np.sum(ii**2)
            S  = np.sum(logvals)
            # Find parameters A1, B1
            XM = (2. + J0) / 2
            YM = S / J0
            A1 = (XY - J0*XM*YM) / (XX - J0*XM*XM)
            B1 = YM - A1 * XM
            # Compute J1, the frequency for which straight
            # line extrapolation drops 20dB below noise
            J1 = int(np.ceil((np.log(0.01*Noise) - B1) / A1 ))
            if J1<J0:
                J1 = J0+1
            ##------------------------------------------------
            ## Compute the Kosarev-Pantos filter windows
            ## Frequency-ranges: 0 -- J0 | J0+1 -- J1 | J1+1 -- N2
            ##------------------------------------------------
            nvals = int((N-1)/2 + 1)
            LOPT = np.zeros_like(Dat_P)
            LOPT[0:J0+1] = Dat_P[0:J0+1] / (Dat_P[0:J0+1] + Noise)
            i_arr = np.arange(J1-J0) + J0+1
            LOPT[J0+1:J1+1] = np.exp(A1*i_arr+B1) / (np.exp(A1*i_arr+B1) + Noise)
            ##--------------------------------------------------------------------
            ## De-noise the Spectrum with the filter
            ## Calculate the first and second derivative (i.e. multiply by iW)
            ##--------------------------------------------------------------------
            # first loop gives smoothed data
            # second loop produces first derivative
            # third loop produces second derivative
            if second_deriv:
                ndiff = 3
            elif first_deriv:
                ndiff = 2
            else:
                ndiff = 1
            
            for diff in range(ndiff):
                Fltr_Spectrum = np.zeros_like(Dat_P,dtype=complex)
                # make the filter complex
                i1 = 1; n2 = int((N-1)/2)+1; i2 = i1+n2 
                FltrCoef = LOPT[i1:].astype(np.complex128)
                # differentitation in frequency domain
                iW = ((np.arange(n2)+i1)*OMEGA*1j)**diff
                # multiply spectrum with filter coefficient
                Fltr_Spectrum[i1:] = Dat_F[i1:] * FltrCoef * iW
                # Fltr_Spectrum[0] values
                # The derivatives of Fltr_Spectrum[0] are 0
                # Mean if diff = 0
                Fltr_Spectrum[0] = 0 if diff>0 else Dat_F[0]
                # Inverse fourier transform back in time domain
                Dat_T = np.fft.irfft(Fltr_Spectrum)
                #Dat_T[-1] = np.real(Dat_T[0]) + 1j*np.imag(Dat_T[-1])
                # This ist the smoothed time series (baseline added)
                if diff==0:
                    Smoothed_Data = np.real(Dat_T) + Dat[0] + SLOPE * X / delt
                elif diff==1:
                    First_Diff = np.real(Dat_T) + SLOPE / delt
                elif diff==2:
                    Secnd_Diff = np.real(Dat_T)
            if second_deriv:
                return Smoothed_Data, First_Diff, Secnd_Diff
            elif first_deriv:
                return Smoothed_Data, First_Diff
            else:
                return Smoothed_Data        

        def optimal_extract_with_error(
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

        def ramp_output(input_read,read_time):
            SIG_map_scaled = fits.getdata(calib_path+'sim_readnoise.fits')
            FLUX_SCALING_FACTOR = 1.0
            SATURATION_DARK_OLS = 50000.0 / FLUX_SCALING_FACTOR
            SATURATION_SCIENCE_OLS = 4096.0 / FLUX_SCALING_FACTOR
            DEFAULT_SIG_FALLBACK_SCALED = 5.0 / FLUX_SCALING_FACTOR
            JUMP_THRESH_ONEOMIT = 20.25 #4.5 sigma
            JUMP_THRESH_TWOOMIT = 23.8
            sci_im_full_original = reffix_hxrg(input_read, nchans=4, fixcol=True)
            self.logger.info("refpix and 1/f correction completed")
            nim_s = input_read.shape[0]
            if nim_s < 200:
                print('Number of reads are less than 5, starting a stright line fit to the reads')
                reads = input_read[:nim_s, :, :]
                read_times = np.arange(nim_s).astype(float)*read_time
                output_fitramp_final = fit_slope_image(reads, read_times,SIG_map_scaled)
            else:
                sci_im_original_units = input_read[:nim_s, :, :]
                sci_im_scaled = sci_im_original_units / FLUX_SCALING_FACTOR
                sci_im_with_jumps_scaled = sci_im_scaled.copy()
                ols_t_global = np.arange(nim_s)
                readtimes_for_covar_sci = np.arange(nim_s).astype(float)*read_time
                B_ols_sci = np.zeros((2048, 2048), dtype=float)
                for i_r in range(2048):
                    for j_c in range(2048):
                        a_g=sci_im_with_jumps_scaled[0,i_r,j_c]; b_g=(sci_im_with_jumps_scaled[1,i_r,j_c]-sci_im_with_jumps_scaled[0,i_r,j_c]) \
                            if nim_s>1 else 1.0; c_g=0.0
                        sp=ols_pack_parms(a_g,b_g,c_g); imdat_p=sci_im_with_jumps_scaled[:,i_r,j_c]
                        w_idx=np.where(imdat_p < SATURATION_SCIENCE_OLS)[0]
                        if len(w_idx)<3: B_ols_sci[i_r,j_c]=np.nan; continue
                        imdat_v = imdat_p[w_idx]
                        def resid_fn_sci_local(p_loc): 
                            mimdat_f=ols_model_fn(p_loc)
                            return imdat_v - mimdat_f[w_idx]
                        try:
                            p_opt_s,ier_s=leastsq(resid_fn_sci_local,sp.copy())
                            if ier_s not in [1,2,3,4]: p_opt_s = np.array([np.nan]*3)
                        except: p_opt_s=np.array([np.nan]*3)
                        _ ,b_fit_s,_ = ols_unpack_parms(p_opt_s); B_ols_sci[i_r,j_c]=b_fit_s
                median_B_sci_val=np.nanmedian(B_ols_sci)
                B_ols_sci[np.isnan(B_ols_sci)]= median_B_sci_val if not np.isnan(median_B_sci_val) else 0.0
                B_ols_sci[B_ols_sci<0]=0
                Covar_obj_sci = fitramp.Covar(readtimes_for_covar_sci, pedestal=False)
                d_sci = sci_im_with_jumps_scaled[1:] - sci_im_with_jumps_scaled[:-1]
                output_fitramp_final = np.empty((2048, 2048), dtype=float)
                start_time = time.time()
                for i in range(2048):
                    if i > 0 and i % (max(1, 2048 // 10)) == 0: print(f"  fitramp data row {i+1}/{2048}...")
                    current_sig_for_row = SIG_map_scaled[i, :]
                    diffs_for_row = d_sci[:, i, :]
                    countrateguess_for_row = B_ols_sci[i, :]

                    diffs2use, countrates_after_masking = fitramp.mask_jumps(
                        diffs_for_row, Covar_obj_sci, current_sig_for_row,
                        threshold_oneomit=JUMP_THRESH_ONEOMIT,
                        threshold_twoomit=JUMP_THRESH_TWOOMIT)

                    final_countrateguess_fitramp = countrates_after_masking * (countrates_after_masking > 0)
                    result = fitramp.fit_ramps(
                        diffs_for_row, Covar_obj_sci, current_sig_for_row,
                        diffs2use=diffs2use,
                        detect_jumps=False,
                        countrateguess=final_countrateguess_fitramp,
                        rescale=True)
                    output_fitramp_final[i, :] = result.countrate * FLUX_SCALING_FACTOR

                end_time = time.time()
                self.logger.info(f"  fitramp on the data took {end_time - start_time:.2f} seconds.")
            return output_fitramp_final


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

            NUM_FRAMES_FROM_SCIENCE = hdr.get("NREADS", "")
            print(f"OBSMODE = {obs_mode}")
            slope = None
            n_ext = len(hdul)
            print('number of extension = ',n_ext)
            if last_file == "TRUE":
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
                            #sci_im_full_original = reffix_hxrg(data_1, nchans=4, fixcol=True)
                            slope = self.slope_fit_weighted(data_1,read_time)
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

                        slope = ramp_output(ramp3d,read_time)
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
                                slope = self.slope_fit_weighted(data_1,read_time)
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
                            slope = self.slope_fit_weighted(ramp3d,read_time)
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
                            cube1,error1 = optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                                slope = self.slope_fit_weighted(data_1,read_time)
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
                            slope = self.slope_fit_weighted(ramp3d,read_time)
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
                            cube1,error1 = optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                                slope = self.slope_fit_weighted(data_1,read_time)
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
                            slope = self.slope_fit_weighted(ramp3d,read_time)
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
                            cube1,error1 = optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                                slope = self.slope_fit_weighted(data_1,read_time)
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
                            slope = self.slope_fit_weighted(ramp3d,read_time)
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
                            cube1,error1 = optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                            elif data.ndim == 3:
                                slope = self.slope_fit_weighted(data_1,read_time)
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
                            slope = self.slope_fit_weighted(ramp3d,read_time)
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
                            cube1,error1 = optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                                slope = self.slope_fit_weighted(data_1,read_time)
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
                            slope = self.slope_fit_weighted(ramp3d,read_time)
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
                            cube1,error1 = optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                                slope = self.slope_fit_weighted(data_1,read_time)
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
                            slope = self.slope_fit_weighted(ramp3d,read_time)
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
                                input_filename=filename,suffix='_quicklook',
                                overwrite=True)

                        #if slope is not None and os.path.exists(os.path.join(calib_path, "QLmat_new.npz")):
                        #    R_matrix = load_npz(calib_path+'QLmat_new.npz')
                        #    print("Quicklook optimal extraction started for",ifs_mode)
                        #    cube1,error1 = optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                                slope = self.slope_fit_weighted(data_1,read_time)
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
                            slope = self.slope_fit_weighted(ramp3d,read_time)
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

                        #if slope is not None and os.path.exists(os.path.join(calib_path, "QLmat_new.npz")):
                        #    R_matrix = load_npz(calib_path+'QLmat_new.npz')
                        #    print("Quicklook optimal extraction started for",ifs_mode)
                        #    cube1,error1 = optimal_extract_with_error(R_matrix,slope,read_noise_var)
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
                                slope = self.slope_fit_weighted(data_1,read_time)
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
                            slope = self.slope_fit_weighted(ramp3d,read_time)
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

                        #if slope is not None and os.path.exists(os.path.join(calib_path, "QLmat_new.npz")):
                        #    R_matrix = load_npz(calib_path+'QLmat_new.npz')
                        #    print("Quicklook optimal extraction started for",ifs_mode)
                        #    cube1,error1 = optimal_extract_with_error(R_matrix,slope,read_noise_var)
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

        log_string = QuickLook.__module__
        #self.action.args.ccddata.header['HISTORY'] = log_string
        self.logger.info(log_string)

        return self.action.args
    # END: class QuickLook()
