import pandas as pd
import numpy as np
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pkg_resources
import os
from scipy.optimize import minimize
import astropy.io.fits as pyfits
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time
import scalesdrp.primitives.robust as robust
from scipy.optimize import leastsq
import pkg_resources
from scipy.signal import savgol_filter



def reffix_hxrg(cube, nchans=4, in_place=False, fixcol=True, **kwargs):
	ndim = len(cube.shape)
	if 'float' not in cube.dtype.name:
		type_in = cube.dtype.name
		copy = (not in_place)
		cube = cube.astype(float, copy=copy)
		type_out = cube.dtype.name
	if not in_place:
		cube = np.copy(cube)
	cube = reffix_amps(cube, nchans=nchans, in_place=True, **kwargs)
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
	nz, ny, nx = data_shape
	chsize = int(nx / nchans)
	if altcol:
		refs_amps_avg1 = []
		refs_amps_avg2 = []
		for ch in range(nchans):
			ich1 = ch*chsize
			ich2 = ich1 + chsize
			refs_ch1 = refs_all[:,:,ich1:ich2-1:2].reshape((nz,-1))
			refs_ch2 = refs_all[:,:,ich1+1:ich2:2].reshape((nz,-1))
			chavg1 = mean_func(refs_ch1,axis=1)
			chavg2 = mean_func(refs_ch2,axis=1)
			refs_amps_avg1.append(chavg1)
			refs_amps_avg2.append(chavg2)
		return (np.array(refs_amps_avg1), np.array(refs_amps_avg2))
	else:
		refs_amps_avg = []
		for ch in range(nchans):
			ich1 = ch*chsize
			ich2 = ich1 + chsize
			refs_ch = refs_all[:,:,ich1:ich2].reshape((nz,-1))
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




