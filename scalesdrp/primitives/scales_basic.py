from scalesdrp.primitives.linearity import DQ_FLAGS
from scalesdrp.core.scales_proctab import Proctab
from scalesdrp.core.scales_pkg_resources import get_resource_path
from scalesdrp.core.matplot_plotting import mpl_plot, mpl_clear
import scalesdrp.primitives.fitramp as fitramp
import scipy.sparse as sp
import pandas as pd
import numpy as np
import pickle
from importlib.resources import files
from pathlib import Path
from astropy.io import fits
import warnings
import astropy.units as u
from scipy import sparse
from scipy.optimize import lsq_linear
from astropy.coordinates import Angle
from astropy.wcs import WCS
import astropy.io.fits as pyfits
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import time
from scipy.signal import savgol_filter
from scipy.ndimage import convolve, median_filter
from scipy.interpolate import griddata
from scipy.stats import median_abs_deviation
from scipy.ndimage import median_filter
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve as astropy_convolve
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.ndimage import distance_transform_edt, gaussian_filter
from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments

############# functions to check avoid processing again #########################
def read_existing_l1(l1_path):

    with fits.open(l1_path) as hdul:
        slope = hdul[0].data
        header = hdul[0].header

        if slope is None:
            raise RuntimeError("Primary HDU has no data.")

        if slope.ndim != 2:
            raise RuntimeError(f"Expected 2D slope image, got {slope.ndim}D.")

        if not np.any(np.isfinite(slope)):
            raise RuntimeError("Slope image has no finite values.")

        uncert = None
        for hdu in hdul[1:]:
            if hdu.name.upper() in ["UNCERT", "ERROR", "ERR", "SIGMA"]:
                uncert = hdu.data
                break

        if uncert is None:
            raise RuntimeError("No uncertainty extension found.")

        if uncert.shape != slope.shape:
            raise RuntimeError(
               f"Uncertainty shape {uncert.shape} does not match slope shape {slope.shape}."
            )

    return slope, uncert, header

def read_existing_l2(l2_path):

    with fits.open(l2_path) as hdul:
        slope = hdul[0].data
        header = hdul[0].header

        if slope is None:
            raise RuntimeError("Primary HDU has no data.")

        if slope.ndim != 3:
            raise RuntimeError(f"Expected 3D cube image, got {slope.ndim}D.")

        if not np.any(np.isfinite(slope)):
            raise RuntimeError("Cube image has no finite values.")

        uncert = None
        for hdu in hdul[1:]:
            if hdu.name.upper() in ["UNCERT", "ERROR", "ERR", "SIGMA"]:
                uncert = hdu.data
                break

        if uncert is None:
            raise RuntimeError("No uncertainty extension found.")

        if uncert.shape != slope.shape:
            raise RuntimeError(
               f"Uncertainty shape {uncert.shape} does not match slope shape {slope.shape}."
            )

    return slope, uncert, header

def _strip_fits_extension(filename):
    base = os.path.basename(filename)
    for ext in [".fits"]:
        if base.endswith(ext):
            return base[:-len(ext)]
    return os.path.splitext(base)[0]


def get_l1_path_from_raw(input_filename, output_dir):
    stem = _strip_fits_extension(input_filename)
    return os.path.join(output_dir,f"{stem}_L1.fits")

def get_l2_path_from_raw(input_filename, output_dir):
    stem = _strip_fits_extension(input_filename)
    return os.path.join(output_dir,f"{stem}_opt_L2.fits")

def find_existing_proc_file(input_filename, suffix,redux_dir):
    #redux_dir = os.path.join(output_dir, "redux")
    proctab_path = os.path.join(redux_dir, "scales.proc")

    if not os.path.exists(proctab_path):
        return None

    try:
        proc = pd.read_csv(
            proctab_path,
            sep="|",
            engine="python",
            skipinitialspace=True
        )
    except Exception as e:
        print(
            f"Could not read proctab {proctab_path}: {e}"
        )
        return None

    # clean column names
    proc.columns = [c.strip() for c in proc.columns]

    # remove empty columns caused by leading/trailing |
    proc = proc.loc[:, [c for c in proc.columns if c != ""]]

    if "filename" not in proc.columns or "SUFF" not in proc.columns:
        print(f"Could not search proctab because filename/SUFF columns are missing.")
        return None

    proc["filename"] = proc["filename"].astype(str).str.strip()
    proc["SUFF"] = proc["SUFF"].astype(str).str.strip()

    raw_stem = _strip_fits_extension(input_filename)

    matches = proc[
        (proc["SUFF"] == suffix)
        &
        (proc["filename"].str.contains(raw_stem, regex=False))
    ]

    if len(matches) == 0:
        return None

    found_name = matches.iloc[-1]["filename"]
    return os.path.join(redux_dir, found_name)

##############################################################################

def load_single_master_file_calib(expected_keywords, master_type):
    """
    Load one matching master calibration file from ./redux or package calib/.
    """

    master_type = master_type.upper()

    tail_map = {
        "DARK": "_mdark.fits",
        "BIAS": "_mbias.fits",
        "FLATLAMP": "_mflatlamp.fits",
        "FLATLENS": "_mflatlens.fits",
        "CALUNIT": "_mcalunit.fits",
    }

    tail = tail_map.get(master_type)
    if tail is None:
        return (None, None)

    #search_dirs = [] #
    search_dirs = [os.path.join(os.getcwd(), "redux")]
    #package = __name__.split(".")[0]
    #calib_path = str(get_resource_path(package, "calib/"))
    #search_dirs.append([os.path.join(os.getcwd(), "redux")])

    for base in search_dirs:
        if not os.path.isdir(base):
            continue

        candidates = [
            os.path.join(base, fname)
            for fname in os.listdir(base)
            if fname.lower().endswith(".fits") and fname.endswith(tail)
        ]

        for path in candidates:
            try:
                with fits.open(path) as hdul:
                    hdr = hdul[0].header
                    data = hdul[0].data
                    uncert = hdul["UNCERT"].data if "UNCERT" in hdul else None
            except Exception:
                continue

            file_imtype = (hdr.get("IMTYPE") or "").upper()
            if file_imtype != master_type:
                continue

            mismatch = False

            for key, expected_value in expected_keywords.items():
                if expected_value is None:
                    continue

                if key == "EXPTIME" and master_type != "DARK":
                    continue

                if key == "IFSMODE" and master_type != "FLATLENS":
                    continue

                if key == "MONOWAVE" and master_type != "CALUNIT":
                    continue

                actual_value = hdr.get(key)

                if key in ["EXPTIME", "MONOWAVE"]:
                    try:
                        if not np.isclose(
                            float(actual_value),
                            float(expected_value),
                            rtol=0,
                            atol=1e-6,
                        ):
                            mismatch = True
                            break
                    except Exception:
                        mismatch = True
                        break
                else:
                    if str(actual_value).strip() != str(expected_value).strip():
                        mismatch = True
                        break

            if mismatch:
                continue

            return (data, uncert)

    return (None, None)

def fits_headers_to_dataframe(directory, pattern="*.fits", recursive=False, include_filename=True):
    """
    Build a pandas DataFrame from the headers of all FITS files in a directory.

    Parameters
    ----------
    directory : str or Path
        Directory containing the FITS files.
    pattern : str, optional
        Glob pattern to match files (default "*.fits"). Use "*.fit" or
        "*.fits.gz" etc. if your files use different extensions.
    recursive : bool, optional
        If True, search subdirectories as well.
    include_filename : bool, optional
        If True, add a "filename" column with each file's name.

    Returns
    -------
    pd.DataFrame
        One row per FITS file, one column per unique header keyword.
        Missing keywords are filled with NaN.
    """
    directory = Path(directory)
    glob_fn = directory.rglob if recursive else directory.glob
    files = sorted(glob_fn(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}")

    rows = []
    for f in files:
        with fits.open(f) as hdul:
            header = hdul[0].header  # primary HDU header
            row = dict(header)
            if include_filename:
                row["filename"] = f.name
            rows.append(row)

    df = pd.DataFrame(rows)

    if include_filename:
        # move filename to the front for readability
        cols = ["filename"] + [c for c in df.columns if c != "filename"]
        df = df[cols]

    return df


#######################################################################################
def swap_odd_even_columns(cube,n_amps=4,do_swap=True):
    if not do_swap:
        return cube
    nreads,n_rows,n_cols = cube.shape
    ramp = np.empty_like(cube)
    block = n_cols // n_amps
    for a in range(n_amps):
        x0, x1 = a * block, (a + 1) * block
        sub = cube[..., x0:x1]
        nsub = sub.shape[-1]
        new_order = []
        for i in range(0, nsub, 2):
            if i + 1 < nsub:
                new_order.extend([i + 1, i])
            else:
                new_order.append(i)
        ramp[..., x0:x1] = sub[..., new_order]
    return ramp

############### spectral extraction ###############################################

def optimal_extract_with_error(
    R_transpose,
    data_image,
    sigma_image,
    read_noise_variance_vector,gain = 1.0):

    print('Optimal extraction started')
    start_time1 = time.time()
    data_vector_d = data_image.flatten().astype(np.float64)
    sigma_vector = sigma_image.array.flatten().astype(np.float64)
    variance_from_map = sigma_vector**2
    photon_noise_variance = data_vector_d.clip(min=0) / gain
    total_variance = read_noise_variance_vector + photon_noise_variance + variance_from_map
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
    print(f"Optimal extraction finished in {t1:.4f} seconds.")
    return optimized_flux, flux_error

def solve_bounded_weighted_nnls(
    R_matrix: sp.spmatrix,
    data_vector: np.ndarray,
    read_noise_variance_vector: np.ndarray,
    gain: float,
    A_guess: np.ndarray,
    bound_factor: float = 1.0,
    tolerance: float = 1e-6) -> np.ndarray:
    """
    Args:
        R_matrix: The (N_pixels, N_fluxes) sparse rectification matrix.
        data_vector: The (H,W) input data vector.
        read_noise_variance_vector: The (N_pixels,) 1D vector of read noise variance.
        gain: The detector gain, used for calculating photon noise.
        A_guess: Amplutide guess from the optimal extraction 1D array (N_fluxes,)
        bound_factor:  How many times the guess to set the upper bound
    Returns:
        Returns the best-fit amplitude
    """
    print("\nSolving with BOUNDED weighted non-negative least squares...")
    photon_noise_variance = data_vector.clip(min=0) / gain
    total_variance = read_noise_variance_vector + photon_noise_variance
    total_variance[total_variance <= 0] = 1e-9
    weights = 1.0 / np.sqrt(total_variance)
    W = sp.diags(weights, format='csr')
    R_prime = W @ R_matrix
    d_prime = W @ data_vector
    lower_bounds = 0
    upper_bounds = np.maximum(0, A_guess) * bound_factor
    upper_bounds += 1e-9
    bounds = (lower_bounds, upper_bounds)
    start_time = time.time()
    lsq_options = {'tol': tolerance, 'verbose': 0}
    res = lsq_linear(R_prime, d_prime, bounds=bounds, **lsq_options)
    end_time = time.time()
    t = (end_time - start_time)/60.0
    print(f"Bounded lsq_linear finished in {t:.4f} mins.")
    return res.x

    ############### chi sqaure error estimation #########################
def calculate_error_flux_cube(
    R_matrix: sp.spmatrix,
    flux_vector_A: np.ndarray,
    var_read_vector: np.ndarray,
    flux_shape_3d: tuple,
    gain: float = 1.0) -> np.ndarray:
    """
    This provides a fast and reliable estimate of the standard deviation for each
    flux element from a least-squares fit.
    The variance of each flux parameter A_j is approximated as:
    Var(A_j) ≈ 1 / H_jj
    where H is the Hessian matrix H_jj = Σ_i (R_ij^2 / σ_i^2).
    Args:
        R_matrix: The (N_pixels, N_fluxes) sparse rectification matrix.
        flux_vector_A: The (N_fluxes,) solved flux vector.
        var_read_vector: The (N_pixels,) vector of read noise variance.
        flux_shape_3d: The 3D shape of the final flux cube (e.g., (54, 108, 108)).
        gain: The detector gain, used for calculating photon noise.
    Returns:
        A 3D NumPy array containing the standard deviation (error) for each flux element.
    """
    print("\nCalculating the error flux cube...")
    start_time = time.time()
    model_data_vector = R_matrix @ flux_vector_A
    model_data_vector[model_data_vector < 0] = 0
    var_photon_vector = model_data_vector / gain
    total_var_vector = var_read_vector + var_photon_vector
    total_var_vector[total_var_vector <= 0] = np.inf
    R_squared = R_matrix.power(2)
    inverse_variance_vector = 1.0 / total_var_vector
    hessian_diagonal = R_squared.T @ inverse_variance_vector
    flux_variance_vector = np.zeros_like(hessian_diagonal)
    valid_mask = hessian_diagonal > 0
    flux_variance_vector[valid_mask] = 1.0 / hessian_diagonal[valid_mask]
    flux_variance_vector[~valid_mask] = np.inf
    error_flux_vector = np.sqrt(flux_variance_vector)
    error_cube = error_flux_vector.reshape(flux_shape_3d)
    end_time = time.time()
    print(f"Error cube calculation finished in {end_time - start_time:.2f} seconds.")
    return error_cube
##################################################################################################
def get_base_output_dir(output_dir):
    output_dir = os.path.abspath(output_dir)
    if os.path.basename(output_dir) == "redux":
        return os.path.dirname(output_dir)
    return output_dir


def proctab_update(
	header,output_dir,input_filename,
	suffix,frame=None,proctab=None,
	proctab_path=None,newtype=None):

	output_dir = get_base_output_dir(output_dir)
	base_name = os.path.basename(input_filename)
	file_root, file_ext = os.path.splitext(base_name)
	output_filename = f"{file_root}{suffix}{file_ext}"
	redux_output_dir = os.path.join(output_dir, "redux")
	os.makedirs(redux_output_dir, exist_ok=True)
	output_path = os.path.join(redux_output_dir, output_filename)

	if proctab is not None:
		if proctab_path is None:
			proctab_path = os.path.join(redux_output_dir, "scales.proc")
		try:
			proctab.read_proctab(proctab_path)
		except Exception as e:
			proctab.new_proctab()

		if frame is not None:
			use_frame = frame
		else:
			class _FrameShim():
				def __init__(self,hdr):
					self.header = hdr
			use_frame = _FrameShim(header)
		try:
			proctab.update_proctab(
				use_frame,
				suffix=suffix,
				filename=output_filename,
				newtype=newtype)
			proctab.write_proctab(proctab_path)
			print("Proctab updated: %s", proctab_path)
		except Exception as e:
			print("Proctab update failed for %s: %s",output_path, str(e))
	return output_path


def build_master_from_stack(
    data_stack,                 # (N,H,W) or list of (H,W)
    sigma_stack=None,           # None or (N,H,W) or list of (H,W)
    *,
    method="ivw",               # 'ivw' (inverse-variance mean), 'mean', or 'median'
    clip_sigma=5.0,             # sigma threshold for iterative clipping
    iterations=3,               # max clipping iterations
    min_valid=2,                # minimum frames per pixel to keep result
    frame_scales=None,          # optional length-N array of multiplicative scales per frame
    return_mask=False           # optionally return the final inlier mask (N,H,W)
):
    """
    Combine a stack of frames into a master calibration frame with uncertainty.

    Parameters
    ----------
    data_stack : array-like
        Stack of input frames, shape (N,H,W) or list of (H,W).
    sigma_stack : array-like or None
    Matching stack of 1σ uncertainties; if None, equal-variance is assumed.
    method : {'ivw','mean','median'}
        ivw   = inverse-variance weighted mean (requires sigma_stack)
        mean  = unweighted mean
        median= unweighted median with robust uncertainty estimate
    clip_sigma : float
        Sigma threshold for iterative outlier rejection (applied along the N axis).
    iterations : int
        Maximum number of clipping iterations.
    min_valid : int
        Minimum number of surviving frames per pixel; otherwise result is NaN.
    frame_scales : array-like or None
        Optional length-N multiplicative scales applied per frame (e.g. normalize by exposure).
    return_mask : bool
        If True, also return the final boolean inlier mask of shape (N,H,W).
    Returns
    -------
    master : (H,W) float32
        Combined master frame.
    master_uncert : (H,W) float32
        Propagated 1σ uncertainty of the master.
    mask (optional) : (N,H,W) bool
        Final inlier mask after clipping (only if return_mask=True).
    """
    # --- normalize inputs to arrays ---
    data = np.asarray(data_stack)
    if True in np.isnan(data_stack): print('nan in data trying to stack')
    if True in np.isnan(sigma_stack): print('nan in sigma trying to stack')
    if data.ndim == 2:
        return data

    if data.ndim == 3:
        N, H, W = data.shape
    elif isinstance(data_stack, (list, tuple)) and np.asarray(data_stack[0]).ndim == 2:
        data = np.stack([np.asarray(f) for f in data_stack], axis=0)
        N, H, W = data.shape
    else:
        raise ValueError("data_stack must be (N,H,W) or list of (H,W).")
    if sigma_stack is None:
        sigma = None
    else:
        sigma = np.asarray(sigma_stack)
        if sigma.shape != data.shape:
            # allow list input
            if isinstance(sigma_stack, (list, tuple)) and len(sigma_stack) == N:
                sigma = np.stack([np.asarray(s) for s in sigma_stack], axis=0)
            else:
                raise ValueError(f"sigma_stack shape {np.asarray(sigma_stack).shape} must match data {data.shape}.")
    # --- optional per-frame scaling ---
    if frame_scales is not None:
        scales = np.asarray(frame_scales, dtype=float)
        if scales.shape != (N,):
            raise ValueError(f"frame_scales must have shape ({N},).")
        data = data * scales[:, None, None]
        if sigma is not None:
            sigma = sigma * np.abs(scales[:, None, None])

    # --- initial mask: finite data (and finite sigma if provided & > 0) ---
    mask = np.isfinite(data)
    if sigma is not None:
        mask &= np.isfinite(sigma) & (sigma > 0)

    # --- iterative clipping along the stack axis ---
    m = mask.copy()
    for _ in range(max(0, int(iterations))):
        # center estimate by method
        if method == "ivw" and sigma is not None:
            w = np.where(m, 1.0 / np.maximum(sigma, 1e-30)**2, 0.0)
            Wsum = np.sum(w, axis=0)
            center = np.where(Wsum > 0, np.sum(w * data, axis=0) / Wsum, np.nan)
            # weighted residual "sigma": use weighted std
            resid = (data - center[None, :, :])
            var = np.where(Wsum > 0, np.sum(w * resid**2, axis=0) / np.maximum(Wsum, 1e-30), np.nan)
            s = np.sqrt(var)
        elif method == "mean":
            valid = np.where(m, data, np.nan)
            center = np.nanmean(valid, axis=0)
            s = np.nanstd(valid, axis=0)
        elif method == "median":
            valid = np.where(m, data, np.nan)
            center = np.nanmedian(valid, axis=0)
            # robust scale via MAD
            mad = np.nanmedian(np.abs(valid - center[None, :, :]), axis=0)
            s = 1.4826 * mad  # MAD→σ
        else:
            raise ValueError("method must be 'ivw', 'mean', or 'median'.")

        if not np.isfinite(clip_sigma) or clip_sigma <= 0:
            break

        # update mask: keep |resid| <= clip_sigma * s
        # compute residuals
        resid = np.abs(data - center[None, :, :])
        thresh = clip_sigma * np.where(np.isfinite(s), s, np.nan)
        keep = resid <= thresh[None, :, :]
        # always keep currently invalid pixels as False
        m = m & keep

        # stop early if nothing changes
        if np.array_equal(m, keep & mask):
            break

    # --- final combine with remaining mask m ---
    n_eff = np.sum(m, axis=0)

    master = np.full((H, W), np.nan, dtype=np.float64)
    master_unc = np.full((H, W), np.nan, dtype=np.float64)

    good_pix = n_eff >= max(1, int(min_valid))

    if method == "ivw" and sigma is not None:
        w = np.where(m, 1.0 / np.maximum(sigma, 1e-30)**2, 0.0)
        Wsum = np.sum(w, axis=0)
        # master = sum(w*x)/sum(w)
        val = np.where(Wsum > 0, np.sum(w * data, axis=0) / Wsum, np.nan)
        unc = np.where(Wsum > 0, 1.0 / np.sqrt(Wsum), np.nan)
        master[good_pix] = val[good_pix]
        master_unc[good_pix] = unc[good_pix]

    elif method == "mean":
        valid = np.where(m, data, np.nan)
        val = np.nanmean(valid, axis=0)
        s = np.nanstd(valid, axis=0)
        unc = np.where(n_eff > 0, s / np.sqrt(np.maximum(n_eff, 1)), np.nan)
        master[good_pix] = val[good_pix]
        master_unc[good_pix] = unc[good_pix]

    elif method == "median":
        valid = np.where(m, data, np.nan)
        val = np.nanmedian(valid, axis=0)
        # robust per-pixel spread via MAD
        mad = np.nanmedian(np.abs(valid - val[None, :, :]), axis=0)
        sigma_robust = 1.4826 * mad
        unc = np.where(
            n_eff > 0,
            1.2533 * sigma_robust / np.sqrt(np.maximum(n_eff, 1)),
            np.nan)
        master[good_pix] = val[good_pix]
        master_unc[good_pix] = unc[good_pix]



    # enforce min_valid
    #master[~good_pix] = np.nan
    #master_unc[~good_pix] = np.nan

    if return_mask:
        return master.astype(np.float32), master_unc.astype(np.float32), m
    return master.astype(np.float32), master_unc.astype(np.float32)

########################### ramp fitting #####################################################
def _ols_row_and_uncert(row_reads,valid_reads_mask,t,sig_row):

    #if True in np.isnan(sig_row):
    #    print('input sigmas have nans')
    N, W = row_reads.shape
    v = valid_reads_mask.astype(bool)
    S0 = v.sum(axis=0)
    S0_safe = np.maximum(S0, 1)
    St  = (t[:, None] * v).sum(axis=0)
    tbar = St / S0_safe
    Stt_centered = (((t[:, None] - tbar) ** 2) * v).sum(axis=0)
    y   = np.where(v, row_reads, 0.0)
    Sy  = y.sum(axis=0)
    Sty = (t[:, None] * y).sum(axis=0)
    num = Sty - tbar * Sy
    den = np.where(Stt_centered > 0, Stt_centered, np.nan)
    slope_row = num / den
    #if True in np.isnan(slope_row):
    #    print('divide by den put nans into slope')
    slope_unc_row = sig_row / np.sqrt(den)
    #if True in np.isnan(slope_unc_row):
    #    print('is den negative?',np.unique(den < 0))
    #    print('is den zero?',np.unique(den==0))
    #    print('divide by sqrt(den) made nans')
    slope_unc_row[(S0 < 2) | ~np.isfinite(den)] = np.nan
    #if True in np.isnan(slope_unc_row):
    #    print('S0 mask made nans')
    return slope_row, slope_unc_row


def ramp_fit(input_read, total_exptime, SIG_map_scaled, *,
    return_pedestal=True,
    reset_prior_strength=3.0, # prior σ = k * SIG per pixel
    use_sigma_clip=False,  # optional; physics mask usually suffices
    sigma_clip=3.0, max_iter=3, min_reads=5, tile=(128, 128),
    JUMP_THRESH_ONEOMIT=20.25,
    JUMP_THRESH_TWOOMIT=23.8,
    group_dq=None):
    """
    produce two images—slope (countrate) and pedestal (reset) using fitramp everywhere it’s well-posed,
    and fall back to a robust OLS only where fitramp can’t run. Optionally clean reads,
    mask physically impossible read pairs, and (optionally) detect jumps.

    Prefer fitramp (pedestal=True) for any number of reads; OLS only as per-pixel fallback.
    - Saturation/rollover handled via Δread > 0 mask.
    - Jump detection layered on top.
    - Finite reset prior from first valid read.
    """
    N, H, W = input_read.shape
    dt = float(total_exptime) / N
    t  = (np.arange(N, dtype=float) + 0.5) * dt
    t1=time.time()
    # (optional) σ-clip (off by default)
    def _sigma_clip_reads(cube):
        from tqdm import tqdm
        keep = np.ones_like(cube, dtype=bool)
        Ty, Tx = tile
        for y0 in tqdm(range(0, H, Ty), desc="σ-clip tiles"):
            y1 = min(H, y0+Ty)
            for x0 in range(0, W, Tx):
                x1 = min(W, x0+Tx)
                sub = cube[:, y0:y1, x0:x1]
                n, ty, tx = sub.shape
                k = ty*tx
                Y = sub.reshape(n, k)
                mask = np.isfinite(Y)
                time_idx = np.arange(n, dtype=np.float32)
                for _ in range(max_iter):
                    cnt = mask.sum(0)
                    if not np.any(cnt >= min_reads): break
                    S0  = cnt
                    St  = time_idx @ mask
                    Stt = (time_idx**2) @ mask
                    Wy  = Y * mask
                    Sy  = Wy.sum(0)
                    Sty = time_idx @ Wy
                    Var_t = Stt - (St*St)/np.maximum(S0, 1)
                    Cov_ty = Sty - (St*Sy)/np.maximum(S0, 1)
                    b = np.zeros(k, np.float32)
                    ok = Var_t > 0
                    b[ok] = Cov_ty[ok] / Var_t[ok]
                    a = (Sy - b*St) / np.maximum(S0, 1)
                    Yhat = a + np.outer(time_idx, b)
                    resid = Y - Yhat
                    resid[~mask] = np.nan
                    s = np.nanstd(resid, 0)
                    new_mask = (np.abs(resid) < sigma_clip*s) & np.isfinite(Y)
                    if np.array_equal(new_mask, mask): break
                    mask = new_mask
                keep[:, y0:y1, x0:x1] = mask.reshape(n, ty, tx)
        return keep

    base_valid = np.isfinite(input_read)
    if use_sigma_clip:
        base_valid &= _sigma_clip_reads(input_read)

    if group_dq is not None:
        saturated = (group_dq & DQ_FLAGS["SATURATED"]) != 0
        base_valid &= ~saturated
    # differences and physics mask
    diffs = input_read[1:] - input_read[:-1]
    ## both reads valid & Δread must be positive
    pair_mask = (base_valid[1:] & base_valid[:-1]) & (diffs > 0)

    # ---------- Reset prior: first valid read; else no prior (σ=∞) ----------
    #If the first resultant is valid, use it as the prior mean on the pedestal.
    #Prior uncertainty is reset_prior_strength × SIG (finite). If first read is bad, set σ=∞ (flat prior).
    #With few usable differences, a finite reset prior makes the joint
    #(pedestal+slope) fit solvable and better conditioned.

    first_read = input_read[0]
    first_ok = base_valid[0] & np.isfinite(first_read)
    resetval_map = np.where(first_ok, first_read, 0.0)
    resetsig_map = np.where(first_ok, reset_prior_strength * SIG_map_scaled, np.inf)

    C_no  = fitramp.Covar(t, pedestal=False)
    C_ped = fitramp.Covar(t, pedestal=True)

    slope  = np.full((H, W), np.nan, float)
    ped    = np.full((H, W), np.nan, float)
    uncert = np.full((H, W), np.nan, float)

    for i in range(H):
        if i % 128 == 0:
            print(f"Fitting row {i}/{H}...")

        sig_row   = SIG_map_scaled[i, :]
        d_row     = diffs[:, i, :]
        m_row     = pair_mask[:, i, :]
        resetval  = resetval_map[i, :]
        resetsig  = resetsig_map[i, :]

        Wrow = d_row.shape[1]
        row_slope  = np.full(Wrow, np.nan, float)
        row_ped    = np.full(Wrow, np.nan, float)
        row_uncert = np.full(Wrow, np.nan, float)

        # 0) Usable diffs BEFORE jump detection
        usable0 = m_row.sum(axis=0)

        #If a pixel has ≥2 usable diffs, we can get a fitramp-based seed for its slope.
        #Otherwise, seed comes from the robust OLS fallback.
        # 1) Initial seed via fitramp only where well-posed (≥2 diffs). Else OLS seed.
        #The final (pedestal=True) optimization benefits from a decent slope initial guess.
        seed = np.full(Wrow, np.nan, float)
        idx_seed_fit = (usable0 >= 2)
        idx_seed_ols = ~idx_seed_fit

        #Use pedestal=False for speed/robustness; countrateguess=None lets fitramp infer it from the data.
        #If something throws, we switch those pixels to OLS seeding.
        if np.any(idx_seed_fit):
            try:
                d_sub   = d_row[:, idx_seed_fit]
                m_sub   = m_row[:, idx_seed_fit]
                sig_sub = sig_row[idx_seed_fit]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    init_res = fitramp.fit_ramps(d_sub, C_no, sig_sub,
                        diffs2use=m_sub,
                        countrateguess=None,
                        rescale=True)
                seed[idx_seed_fit] = init_res.countrate
            except Exception:
                idx_seed_ols |= idx_seed_fit

        if np.any(idx_seed_ols):
            # compute OLS for the whole row once and slice
            ols_row, _ = _ols_row_and_uncert(input_read[:, i, :],  # (N,W)
                base_valid[:, i, :],
                t, sig_row)
            seed[idx_seed_ols] = ols_row[idx_seed_ols]

        # fill non-finite seeds with OLS
        bad = ~np.isfinite(seed)
        if np.any(bad):
            ols_row, _ = _ols_row_and_uncert(input_read[:, i, :],
                base_valid[:, i, :],
                t, sig_row)
            seed[bad] = ols_row[bad]

        # 2) Jump detection (only where we have ≥1 diff)
        #Likelihood-based cosmic-ray/jump search; returns a binary mask for diffs to keep.
        #Drop diffs that substantially improve χ² when omitted (thresholds ≈ 4.5σ false-positive equivalent).
        m_row2 = m_row.copy()
        idx_jump = (usable0 >= 1)
        if np.any(idx_jump):
            try:
                d_sub   = d_row[:, idx_jump]
                m_sub   = m_row[:, idx_jump]
                sig_sub = sig_row[idx_jump]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    m2_sub, _ = fitramp.mask_jumps(d_sub, C_no, sig_sub,
                        threshold_oneomit=JUMP_THRESH_ONEOMIT,
                        threshold_twoomit=JUMP_THRESH_TWOOMIT,
                        diffs2use=m_sub)
                m_row2[:, idx_jump] = m2_sub
            except Exception:
                pass

        usable_final = m_row2.sum(axis=0)

        # 3) Final fit with pedestal=True
        # Solvable if:
        #   - usable_final >= 1 AND prior finite, OR
        #   - usable_final >= 2 (solvable even without a prior)
        #This is the high-fidelity fit that jointly estimates pedestal and slope using the full covariance model.
        idx_final_fit = (usable_final >= 1) & (np.isfinite(resetsig) | (usable_final >= 2))
        if np.any(idx_final_fit):
            try:
                d_sub    = d_row[:, idx_final_fit]
                m_sub    = m_row2[:, idx_final_fit]
                sig_sub  = sig_row[idx_final_fit]
                seed_sub = seed[idx_final_fit]
                r0_sub   = resetval[idx_final_fit]
                s0_sub   = resetsig[idx_final_fit]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    final_res = fitramp.fit_ramps(d_sub, C_ped, sig_sub,
                        diffs2use=m_sub,
                        countrateguess=seed_sub,
                        resetval=r0_sub, resetsig=s0_sub,
                        rescale=True)
                row_slope[idx_final_fit]  = final_res.countrate
                row_ped[idx_final_fit]    = final_res.pedestal
                row_uncert[idx_final_fit] = final_res.uncert     # <-- from fitramp
            except Exception:
                pass

        # 4) Per-pixel fallback: unsolved, zero usable diffs, or excluded by gating
        #If final fit is missing, use OLS slope and the prior mean for pedestal.
        need_fallback = (~np.isfinite(row_slope)) | (usable_final == 0) | (~idx_final_fit)
        if np.any(need_fallback):
            ols_row, ols_unc = _ols_row_and_uncert(input_read[:, i, :],
                base_valid[:, i, :],
                t, sig_row)
            row_slope[need_fallback]  = ols_row[need_fallback]
            row_uncert[need_fallback] = ols_unc[need_fallback]
            row_ped[need_fallback]    = resetval[need_fallback]  # prior mean

        # 5) Final sanitation: any remaining non-finite → seed; then OLS
        bad_final = ~np.isfinite(row_slope)
        if np.any(bad_final):
            ols_row, ols_unc = _ols_row_and_uncert(input_read[:, i, :],
                base_valid[:, i, :],
                t, sig_row)
            row_slope[bad_final]  = ols_row[bad_final]
            row_uncert[bad_final] = ols_unc[bad_final]
            row_ped[bad_final]    = resetval[bad_final]

        slope[i, :]  = row_slope
        ped[i,   :]  = row_ped
        uncert[i, :] = row_uncert
    t2=time.time()
    print(f"Ramp fitting completed in {t2 - t1:.3f} seconds.")
    return (slope, ped, uncert) if return_pedestal else (slope, uncert)

######################################################################################
def estimate_uncert_single_read(
    image_dn,
    readnoise_map_dn,
    gain=3.0,
):
    """
    Estimate uncertainty for a single-read image when image and read-noise map
    are both in DN.

    sigma_DN = sqrt(RN_DN^2 + signal_DN / gain)
    """

    image_dn = np.asarray(image_dn, dtype=float)
    readnoise_map_dn = np.asarray(readnoise_map_dn, dtype=float)

    if image_dn.shape != readnoise_map_dn.shape:
        raise ValueError(
            f"image shape {image_dn.shape} does not match "
            f"readnoise map shape {readnoise_map_dn.shape}"
        )

    shot_var_dn2 = np.maximum(image_dn, 0.0) / gain
    read_var_dn2 = readnoise_map_dn**2

    uncert_dn = np.sqrt(read_var_dn2 + shot_var_dn2)

    return uncert_dn.astype(np.float32)

##########################################################################################

def normalize_detector_flat(
    flat,                     # (H,W) master flat (same units as science)
    flat_sigma,               # (H,W) 1σ uncertainty of master flat (same units)
    mask=None,                # (H,W) bool, True = invalid pixel
    method='median',          # 'median' or 'mean' for normalization constant
    clip_sigma=7.0,
    iterations=3,
    eps=1e-12,
    return_norm_stats=False):
    """
    Normalize a master flat and propagate uncertainty.

    For each pixel i:
        f_i_norm = F_i / c
        σ_i_norm^2 ≈ (σ_Fi^2 / c^2) + (F_i^2 * σ_c^2 / c^4)

    where c is the robust central value (median or mean) of the valid pixels,
    estimated after sigma-clipping; σ_c is the standard error of c.

    Notes
    -----
    * Correlation between F_i and c is ignored (good approximation for large N).
    * For 'median':   σ_c ≈ 1.253 * σ_sample / sqrt(N_eff)
    For 'mean'  :   σ_c =      σ_sample / sqrt(N_eff)
    σ_sample is estimated robustly via MAD; falls back to std if needed.
    * Invalid pixels (mask | non-finite | <=0) return NaN in both outputs.
    """
    flat      = np.asarray(flat, dtype=np.float64)
    flat_sigma= np.asarray(flat_sigma, dtype=np.float64)
    if flat.shape != flat_sigma.shape:
        raise ValueError("flat and flat_sigma must have identical shapes")

    invalid = ~np.isfinite(flat) | ~np.isfinite(flat_sigma) | (flat <= 0)
    if mask is not None:
        invalid |= np.asarray(mask, dtype=bool)

    valid = ~invalid
    valid_data = flat[valid]
    if valid_data.size == 0:
        raise ValueError("No valid pixels found in flat for normalization.")

    # --- sigma clip the valid sample to define the normalization pool ---
    clipped = valid_data.copy()
    for _ in range(iterations):
        med = np.median(clipped)
        # robust scatter via MAD; fallback to std if MAD==0
        mad = np.median(np.abs(clipped - med))
        robust_sigma = mad / 0.67448975 if mad > 0 else np.std(clipped)
        if robust_sigma == 0 or not np.isfinite(robust_sigma):
            break
        low, high = med - clip_sigma * robust_sigma, med + clip_sigma * robust_sigma
        keep = (clipped > low) & (clipped < high)
        if np.all(keep):
            break
        clipped = clipped[keep]

    if clipped.size == 0:
        raise ValueError("All flat pixels rejected during clipping; cannot normalize.")

    # normalization constant (c) and its standard error (σ_c)
    if method.lower() == 'median':
        c = np.median(clipped)
        # recompute robust sigma on the final clipped set
        mad = np.median(np.abs(clipped - c))
        sig_sample = mad / 0.67448975 if mad > 0 else np.std(clipped)
        sigma_c = 1.2533141373155001 * sig_sample / np.sqrt(clipped.size)  # ≈ sqrt(pi/2)
    elif method.lower() == 'mean':
        c = np.mean(clipped)
        sig_sample = np.std(clipped, ddof=1) if clipped.size > 1 else 0.0
        sigma_c = sig_sample / np.sqrt(clipped.size) if clipped.size > 0 else np.inf
    else:
        raise ValueError("method must be 'median' or 'mean'")
    if not np.isfinite(c) or c <= 0:
        raise ValueError(f"Invalid normalization constant: {c}")

    # --- propagate uncertainties per pixel ---
    c_safe = float(c)
    c2 = c_safe**2
    c4 = c_safe**4
    # main formulas
    flat_norm = np.full_like(flat, np.nan, dtype=np.float64)
    sigma_norm= np.full_like(flat, np.nan, dtype=np.float64)

    # avoid division by zero / nan propagation; invalid already handled
    denom = np.where(valid, c_safe, np.nan)

    flat_norm[valid]  = flat[valid] / denom[valid]
    # Var(f/c) = Var(f)/c^2 + f^2 * Var(c)/c^4
    sigma_norm[valid] = np.sqrt( (flat_sigma[valid]**2) / c2 + (flat[valid]**2) * (sigma_c**2) / c4 )

    # cast to float32 for storage if desired
    flat_norm  = flat_norm.astype(np.float32)
    sigma_norm = sigma_norm.astype(np.float32)

    if return_norm_stats:
        return flat_norm, sigma_norm, (float(c), float(sigma_c), int(clipped.size))
    return flat_norm, sigma_norm

####################### calib correction #############################
def apply_calibration(
    data,                # (H,W) science data
    sigma_data,          # (H,W) 1σ of science data
    calib,               # (H,W) calibration frame: bias/dark/flat
    sigma_calib,         # (H,W) 1σ of calibration frame
    imtype,              # str from header, e.g. 'BIAS','DARK','FLAT'
    *,
    scale: float = 1.0,  # optional scale for bias/dark (e.g. exposure ratio)
    eps: float = 1e-10,  # avoid divide-by-zero for flats
    clip_nan: bool = True):
    """
    Return (corrected_data, corrected_sigma) with error propagation.

    If IMTYPE is 'BIAS' or 'DARK' (case-insensitive):
        out = data - scale * calib
        σ_out = sqrt( σ_data^2 + (scale^2) * σ_calib^2 )

    If IMTYPE is 'FLAT' (normalized flat):
        out = data / calib
        σ_out = sqrt( σ_data^2 / calib^2 + data^2 * σ_calib^2 / calib^4 )

    Notes:
    - All arrays must be same shape and numeric.
    - For subtraction, units of data & calib (and their σ) must match.
    - For flat-fielding, calib should be normalized (unitless, ~1.0 mean).
    - `scale` is applied only for BIAS/DARK (ignored for FLAT).
    """
    data        = np.asarray(data, dtype=float)
    sigma_data  = np.asarray(sigma_data, dtype=float)
    calib       = np.asarray(calib, dtype=float)
    sigma_calib = np.asarray(sigma_calib, dtype=float)

    kind = (imtype or "").strip().upper()

    if kind in ("BIAS", "DARK"):
        out = data - scale * calib
        sig = np.sqrt(sigma_data**2 + (scale**2) * sigma_calib**2)

    elif kind in ("FLATLAMP"):
        denom = np.where(np.isfinite(calib) & (np.abs(calib) > eps), calib, np.nan)
        out = data / denom
        sig = np.sqrt( (sigma_data**2) / (denom**2) + (data**2) * (sigma_calib**2) / (denom**4) )

    else:
        raise ValueError(f"Unrecognized IMTYPE='{imtype}'. Expected 'BIAS', 'DARK', or 'FLAT'.")

    if clip_nan:
        good = np.isfinite(out) & np.isfinite(sig)
        out = np.where(good, out, np.nan)
        sig = np.where(good, sig, np.nan)
    return out.astype(np.float32), sig.astype(np.float32)

############### load master files ##########################################

def load_single_master_file(expected_keywords, master_type):
    """
    Load one matching master calibration file from ./redux or calib/.

    Matching rules:
    - IMTYPE is determined from master_type.
    - CAMERA and MCLOCK must match for all master files.
    - EXPTIME must match only for DARK.
    - If no matching file is found, return (None, None).
    """

    master_type = master_type.upper()

    tail_map = {
        "DARK": "_mdark.fits",
        "BIAS": "_mbias.fits",
        "FLATLAMP": "_mflatlamp.fits",
    }

    tail = tail_map.get(master_type)
    if tail is None:
        return (None, None)

    # Search redux first, then package calib/
    search_dirs = [os.path.join(os.getcwd(), "redux")]
    package = __name__.split(".")[0]
    filepath = "calib/"
    calib_path = str(get_resource_path(package, filepath))
    search_dirs.append(calib_path)
    for base in search_dirs:
        if not os.path.isdir(base):
            continue

        candidates = [
            os.path.join(base, fname)
            for fname in os.listdir(base)
            if fname.lower().endswith(".fits") and fname.endswith(tail)
        ]

        for path in candidates:
            try:
                with fits.open(path) as hdul:
                    hdr = hdul[0].header
                    data = hdul[0].data
                    uncert = hdul["UNCERT"].data if "UNCERT" in hdul else None
            except Exception:
                continue

        # --------------------------------------------------
        # IMTYPE check comes from requested master_type
        # --------------------------------------------------
            file_imtype = (hdr.get("IMTYPE") or "").upper()

            expected_imtype = master_type

            if file_imtype != expected_imtype:
                continue

            mismatch = False

            for key, expected_value in expected_keywords.items():
                if expected_value is None:
                    continue

            # EXPTIME only matters for DARK
                if key == "EXPTIME" and master_type != "DARK":
                    continue

                actual_value = hdr.get(key)

                if key == "EXPTIME":
                    try:
                        if not np.isclose(
                            float(actual_value),
                            float(expected_value),
                            rtol=0,
                            atol=1e-6,
                        ):
                            mismatch = True
                            break
                    except Exception:
                        mismatch = True
                        break
                else:
                    if actual_value != expected_value:
                        mismatch = True
                        break

            if mismatch:
                continue

            return (data, uncert)
    return (None, None)
#################################################################################
def ifsmode_select(self,modslnam, dsprsnam):
    modslnam = modslnam.strip()
    dsprsnam = dsprsnam.strip()
    grating = dsprsnam.split("-")[0]

    if modslnam == "MedRes":
        band = grating[0]
        ifsmode = f"{modslnam}-{band}"
    elif modslnam == "LowRes":
        band = grating
        ifsmode = f"{modslnam}-{band}"
    return ifsmode

############## lenslet flat may change #####################################################
def load_and_normalize_lenslet_flat(
    ifsmode,
    *,
    clip_sigma=7.0,
    iterations=3,
    method='median'):
    """
    Find <prefix>_cube_flatlens.fits by IFSMODE, ensure it's a 3D cube,
    load (data, uncert), normalize each slice, and propagate uncertainties.
    Search order: ./redux/ then pkg calib/.
    Returns (flat_norm, flat_norm_uncert) or (None, None).
    """
    norm_ifsmode = (ifsmode or "").strip().upper().replace("_", "-")

    def _try_dir(base_dir):
        if not os.path.isdir(base_dir):
            return None, None, None
        # look for any file ending in _cube_flatlens.fits
        candidates = [f for f in os.listdir(base_dir)
                if f.endswith("_cube_flatlens.fits") and os.path.isfile(os.path.join(base_dir, f))]
        for fname in candidates:
            path = os.path.join(base_dir, fname)
            try:
                with fits.open(path) as hdul:
                    hdr = hdul[0].header
                    file_ifs = (hdr.get("IFSMODE") or "").strip().upper().replace("_", "-")
                    if file_ifs != norm_ifsmode:
                        continue

                    data = hdul[0].data
                    if data is None:
                        self.logger.debug(f"Skipping {fname}: primary data is None.")
                        continue
                    if data.ndim != 3:
                        self.logger.debug(f"Skipping {fname}: not a cube (ndim={data.ndim}).")
                        continue

                    # try to get uncertainty cube with same shape
                    uncert = None
                    if "UNCERT" in hdul and hdul["UNCERT"].data is not None:
                        if hdul["UNCERT"].data.shape == data.shape:
                            uncert = hdul["UNCERT"].data
                        else:
                            self.logger.debug(f"Skipping UNCERT in {fname}: shape mismatch {hdul['UNCERT'].data.shape} vs {data.shape}.")
                    elif len(hdul) > 1 and getattr(hdul[1], "data", None) is not None:
                        if hdul[1].data.shape == data.shape:
                            uncert = hdul[1].data
                        else:
                            self.logger.debug(f"Skipping ext[1] as UNCERT in {fname}: shape mismatch {hdul[1].data.shape} vs {data.shape}.")

                    return path, data, uncert
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                continue
        return None, None, None

    # 1) ./redux
    path_used, flat, uflat = _try_dir(os.path.join(os.getcwd(), "redux"))
    # 2) pkg calib if not found
    if flat is None:
        package = __name__.split('.')[0]
        filedir = 'calib/'
        pkg_dir = str(get_resource_path(package, filedir))
        path_used, flat, uflat = _try_dir(pkg_dir)

    if flat is None:
        print(f"No lenslet flat cube found for IFSMODE={ifsmode}.")
        return None, None

    print(f"Loaded lenslet flat cube: {os.path.basename(path_used)}")

    flat = np.asarray(flat, dtype=np.float64)
    uflat = None if uflat is None else np.asarray(uflat, dtype=np.float64)

    N, Y, X = flat.shape
    flat_norm = np.full_like(flat, np.nan, dtype=np.float64)
    flat_norm_uncert = None if uflat is None else np.full_like(uflat, np.nan, dtype=np.float64)

    for k in range(N):
        f = flat[k]
        invalid = ~np.isfinite(f) | (f <= 0)
        vals = f[~invalid]
        if vals.size == 0:
            continue

        # iterative sigma clip
        clipped = vals.copy()
        for _ in range(iterations):
            med = np.median(clipped); std = np.std(clipped)
            if not np.isfinite(med) or not np.isfinite(std) or std == 0:
                break
            keep = (clipped > med - clip_sigma*std) & (clipped < med + clip_sigma*std)
            if np.all(keep):
                break
            clipped = clipped[keep]
            if clipped.size == 0:
                break
        if clipped.size == 0:
            continue

        a = np.median(clipped) if method.lower() == 'median' else np.mean(clipped)
        if not np.isfinite(a) or a <= 0:
            continue

        # estimate σ_a from clipped scatter
        s = np.std(clipped); N_eff = max(1, clipped.size)
        sa = s / np.sqrt(N_eff)

        # normalize slice
        flat_norm[k] = f / a
        flat_norm[k, invalid] = np.nan

        # uncertainty propagation if available
        if uflat is not None:
            uf = uflat[k]
            if uf is not None and uf.shape == f.shape:
                term1 = (uf / a)**2
                term2 = ((f * sa) / (a**2))**2
                flat_norm_uncert[k] = np.sqrt(term1 + term2)
                flat_norm_uncert[k, invalid] = np.nan

    print(f"Normalized lenslet flat cube for IFSMODE={ifsmode}.")
    return flat_norm, flat_norm_uncert

##################### flat correction to the cube ##############################
def apply_flatlens(
    data,                # (H,W) or (N,H,W) science
    sigma_data,          # same shape as data (1σ)
    calib,               # (H,W) or (N,H,W) lenslet flat (prefer normalized)
    sigma_calib,         # same shape as calib (1σ); can be None
    imtype,              # should be 'FLATLENS'
    *,
    eps: float = 1e-10,  # guard for near-zero flats
    clip_nan: bool = True):
    """
    Divide science by lenslet-flat (supports 2D or 3D cubes) with uncertainty propagation:
        out = data / F
        σ_out^2 = (σ_data / F)^2 + (data * σ_F / F^2)^2
    Shapes:
        - data, sigma_data: (H,W) or (N,H,W)
        - calib, sigma_calib: (H,W) or (N,H,W)
        (Broadcasting is supported for (H,W) calib across N if needed.)
    """
    kind = (imtype or "").strip().upper()
    if kind != "FLATLENS":
        raise ValueError(f"IMTYPE='{imtype}' not supported here; expected 'FLATLENS'.")
        return data.astype(np.float32), sigma_data.astype(np.float32)

    if calib is None or (isinstance(calib, np.ndarray) and calib.size == 0):
        print("No flatlens data provided; skipping flat correction.")
        return data.astype(np.float32), sigma_data.astype(np.float32)

    # Cast to float arrays; allow sigma_calib=None (treated as zeros)
    data        = np.asarray(data, dtype=float)
    sigma_data  = np.asarray(sigma_data, dtype=float)
    calib       = np.asarray(calib, dtype=float)
    sigma_calib = np.zeros_like(calib, dtype=float) if sigma_calib is None else np.asarray(sigma_calib, dtype=float)

    try:
        if data.shape != sigma_data.shape:
            raise ValueError(f"sigma_data shape {sigma_data.shape} must match data {data.shape}.")
        if calib.shape != data.shape:
            # Allow (H,W) flat to apply to (N,H,W) science
            if calib.ndim == 2 and data.ndim == 3 and calib.shape == data.shape[-2:]:
                calib = np.broadcast_to(calib, data.shape)
                sigma_calib = np.broadcast_to(sigma_calib, data.shape) if sigma_calib.ndim == 2 else sigma_calib
            else:
                raise ValueError(f"calib shape {calib.shape} not compatible with data {data.shape}.")
        if sigma_calib.shape != calib.shape:
            # Allow (H,W) σ_flat to broadcast to (N,H,W)
            if sigma_calib.ndim == 2 and calib.ndim == 3 and sigma_calib.shape == calib.shape[-2:]:
                sigma_calib = np.broadcast_to(sigma_calib, calib.shape)
            else:
                raise ValueError(f"sigma_calib shape {sigma_calib.shape} not compatible with calib {calib.shape}.")
    except Exception as e:
        print(f"[apply_flatlens] Shape mismatch — skipping flat correction: {e}")
        return data.astype(np.float32), sigma_data.astype(np.float32)

    # Guard against invalid flat values
    finite_flat = np.isfinite(calib)
    if not np.any(finite_flat):
        print("Flatlens data invalid or all NaN; skipping correction.")
        return data.astype(np.float32), sigma_data.astype(np.float32)

    safe_flat = np.where(finite_flat & (np.abs(calib) > eps), calib, np.nan)

    # Division
    out = data / safe_flat

    # Uncertainty propagation
    # σ_out^2 = (σ_data / F)^2 + (data * σ_F / F^2)^2
    term1 = (sigma_data / safe_flat) ** 2
    term2 = ((data * sigma_calib) / (safe_flat ** 2)) ** 2
    sig = np.sqrt(term1 + term2)

    if clip_nan:
        good = np.isfinite(out) & np.isfinite(sig)
        out = np.where(good, out, np.nan)
        sig = np.where(good, sig, np.nan)

    print("Successfully applied lenslet flat correction.")
    return out.astype(np.float32), sig.astype(np.float32)

############################# WCS ###########################################################
def _parse_sky_coord(coord_val: [str, float], is_ra: bool = False) -> float:
    """
    Helper function to robustly parse a sky coordinate value.
    It can handle:
        - Floats or integers (assumed to be in degrees).
        - Strings in various formats recognized by astropy.coordinates.Angle
            (e.g., '17:45:40.04', '-29d00m28.1s').
    Args:
        coord_val (str or float): The coordinate value from the FITS header.
        is_ra (bool): Flag to indicate if this is Right Ascension, to give
        priority to hour-angle parsing for ambiguous strings.
    Returns:
        float: The coordinate value in degrees.
    """
    if isinstance(coord_val, (int, float)):
        return float(coord_val)
    if not isinstance(coord_val, str):
        raise TypeError(f"Coordinate must be a string or number, but got {type(coord_val)}.")

    try:
        if is_ra and ('h' in coord_val.lower() or ':' in coord_val):
            return Angle(coord_val, unit=u.hourangle).degree
        else:
            return Angle(coord_val, unit=u.deg).degree
    except (u.UnitsError, ValueError) as e:
        raise ValueError(f"Could not parse coordinate string '{coord_val}'. Error: {e}")


def create_scales_wcs(
    cube_shape: tuple,
    header: fits.Header,
    center_map_is_zero_indexed: bool = True):
    """
    Create PC + CDELT WCS for a SCALES IFS cube.

    Cube shape:
        cube_shape = (n_wave, ny, nx)

    WCS convention:
        Axis 1 = RA
        Axis 2 = DEC
        Axis 3 = wavelength

    Wavelength unit:
        CUNIT3 = 'um'

    Pipeline-readable wavelength keywords:
        WAVSTART = wavelength start in micron
        WAVEND   = wavelength end in micron
        DWAVE    = wavelength step in micron/pixel
    """
    SCALES_PLATE_SCALE_ARCSEC = 0.02

    #order (y,x)
    center_map = {
        "LowRes-K": (54, 54),
        "LowRes-L":   (50, 60),
        "LowRes-M":   (50, 60),
        "LowRes-SED":   (50, 60),
        "LowRes-KL": (50, 60),
        "LowRes-PAH": (50, 60),
        "MedRes-K":   (50, 60),
        "MedRes-L":   (50, 60),
        "MedRes-M":   (50, 60),
    }

    default_center_yx = (54, 54)

    wave_config_um = {
        "LowRes-K": {"start": 1.95,  "end": 2.45},
        "LowRes-L":   {"start": 2.9, "end": 4.15},
        "LowRes-M":   {"start": 4.5,  "end": 5.2},
        "LowRes-SED":   {"start": 2.0,  "end": 5.0},
        "LowRes-KL": {"start": 2.0,  "end": 3.7},
        "LowRes-PAH": {"start": 3.1,  "end": 3.5},
        "MedRes-K":   {"start": 2.0,  "end": 2.4},
        "MedRes-L":   {"start": 2.9,  "end": 4.15},
        "MedRes-M":   {"start": 4.5,  "end": 5.2},
    }

    n_wave, ny, nx = cube_shape

    # convertd ra and dec into degrees
    ra_str = header.get('RA', '00:00:00.0')
    dec_str = header.get('DEC', '00:00:00.0')

    crval_ra = _parse_sky_coord(ra_str, is_ra=True)
    crval_dec = _parse_sky_coord(dec_str, is_ra=False)

    # Spatial scale to degrees
    pixel_scale_deg = SCALES_PLATE_SCALE_ARCSEC / 3600.0

    # Position angle from degrees to radians
    pa_deg = float(header.get("PA", header.get("PARANG", 0.0)))
    pa_rad = np.deg2rad(pa_deg)

    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)

    # SCALES mode
    ifs_mode = header.get("IFSMODE", header.get("FILTER", "default")).strip()

    # Reference spatial pixel (IFSMODE name case insensitive)
    center_map_lower = {k.lower(): v for k, v in center_map.items()}
    crpix_y, crpix_x = center_map_lower.get(
        ifs_mode.lower(),
        default_center_yx,
    )

    # FITS CRPIX is 1-indexed
    if center_map_is_zero_indexed:
        crpix_x += 1
        crpix_y += 1

    # Wavelength range in microns
    wave_cfg = wave_config_um.get(ifs_mode)

    if wave_cfg is not None:
        wavstart_um = float(wave_cfg["start"])
        wavend_um = float(wave_cfg["end"])
    else:
        wavstart_um = float(header.get("WAVSTART", 2.0))
        wavend_um = float(header.get("WAVEND", wavstart_um + n_wave - 1))

    if n_wave > 1:
        dwave_um = (wavend_um - wavstart_um) / (n_wave - 1)
    else:
        dwave_um = 1.0
    crpix_wave = 1.0

    # --------------------------------------------------
    # PC + CDELT WCS
    # --------------------------------------------------
    # CDELT carries pixel scale.
    # PC carries rotation and axis coupling.
    #
    # Negative CDELT1 follows the usual astronomical convention:
    # increasing x corresponds to decreasing RA.
    # --------------------------------------------------

    wcs_dict = {
        "WCSAXES": 3,

        "CTYPE1": "RA---TAN",#Axis 1 is right ascension with tangent-plane projection.
        "CTYPE2": "DEC--TAN",#Axis 2 is declination with tangent-plane projection.
        "CTYPE3": "WAVE", #Axis 3 wavelength

        "CUNIT1": "deg",# RA in degrees
        "CUNIT2": "deg",# DEC in degrees
        "CUNIT3": "um", # wavelength in micro meters

        "CRVAL1": crval_ra,#these are the world coordinates of the reference pixel
        "CRVAL2": crval_dec,
        "CRVAL3": wavstart_um,

        "CRPIX1": crpix_x,#reference pixel coordinates corresponding to CRVAL
        "CRPIX2": crpix_y,
        "CRPIX3": crpix_wave,
        #The negative sign is standard for RA because increasing image
        #x usually corresponds to decreasing RA on the sky.
        "CDELT1": -pixel_scale_deg, #coordinate increment per pixel
        "CDELT2":  pixel_scale_deg, #dec increaes with y
        "CDELT3":  dwave_um, #wavelength increment

        "PC1_1": cos_pa, #rotation matrix for the spatial axis
        "PC1_2": -sin_pa,
        "PC1_3": 0.0,

        "PC2_1": sin_pa, #rotation matrix for the spatial axis
        "PC2_2": cos_pa,
        "PC2_3": 0.0,

        "PC3_1": 0.0, #Spectral PC term
        "PC3_2": 0.0,
        "PC3_3": 1.0,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fits.verify.VerifyWarning)
        wcs = WCS(wcs_dict, naxis=3)

    wave_info = {
        "IFSMODE": ifs_mode,
        "WAVSTART": wavstart_um,
        "WAVEND": wavend_um,
        "DWAVE": dwave_um,
        "WAVEUNIT": "um",
    }

    return wcs, wave_info


def wcs_header_update(
    data_cube,
    input_header: fits.Header,
    wcs: WCS,
    wave_info: dict,
):
    """
    Write SCALES cube with clean WCS.

    Final written spectral WCS is forced to microns so DS9 displays
    wavelength slider values in um.
    """
    SCALES_PLATE_SCALE_ARCSEC =0.02
    output_header = input_header.copy()

    # Remove old/conflicting WCS first
    #output_header = remove_old_wcs_keywords(output_header)

    # Add WCS generated by Astropy
    output_header.update(wcs.to_header())

    # --------------------------------------------------
    # Force final spectral WCS to microns for DS9 display
    # --------------------------------------------------
    output_header["CTYPE3"] = ("WAVE", "Wavelength axis")
    output_header["CUNIT3"] = ("um", "Wavelength unit")
    output_header["CRVAL3"] = (
        float(wave_info["WAVSTART"]),
        "Reference wavelength in micron",
    )
    output_header["CRPIX3"] = (
        1.0,
        "Reference wavelength pixel",
    )
    output_header["CDELT3"] = (
        float(wave_info["DWAVE"]),
        "Wavelength step in micron/pixel",
    )

    # Stable spectral PC matrix for DS9
    output_header["PC3_3"] = (1.0, "Spectral axis scale")
    output_header["PC3_1"] = (0.0, "No spectral-spatial coupling")
    output_header["PC3_2"] = (0.0, "No spectral-spatial coupling")
    output_header["PC1_3"] = (0.0, "No spatial-spectral coupling")
    output_header["PC2_3"] = (0.0, "No spatial-spectral coupling")

    # Remove any CD spectral terms if Astropy added them
    for key in ["CD3_1", "CD3_2", "CD3_3", "CD1_3", "CD2_3"]:
        if key in output_header:
            del output_header[key]

    # --------------------------------------------------
    # Pipeline-readable metadata
    # --------------------------------------------------
    output_header["PIXSCALE"] = (
        SCALES_PLATE_SCALE_ARCSEC,
        "SCALES plate scale in arcsec/spaxel",
    )

    output_header["IFSMODE"] = (
        wave_info["IFSMODE"],
        "SCALES IFS observing mode",
    )

    output_header["WAVSTART"] = (
        float(wave_info["WAVSTART"]),
        "Wavelength start in micron",
    )

    output_header["WAVEND"] = (
        float(wave_info["WAVEND"]),
        "Wavelength end in micron",
    )

    output_header["DWAVE"] = (
        float(wave_info["DWAVE"]),
        "Wavelength step in micron/pixel",
    )

    output_header["WAVEUNIT"] = (
        "um",
        "Pipeline wavelength unit",
    )

    output_header["WCSCORR"] = (
        True,
        "WCS keywords written",
    )

    output_header["WCSTYPE"] = (
        "INITIAL",
        "Initial SCALES cube WCS",
    )
    print("WCS coordinates created")
    return output_header


################### GROUPING files for calib input ################################################
def group_files_by_header(dt):
    """
    Groups files in the data table based on CAMERA and key header values.

    Rules:
    - CAMERA == 'Im'  : group by (CAMERA, IM-FW-1, IMTYPE, EXPTIME, MCLOCK)
    - CAMERA == 'IFS' : group by (CAMERA, IFSMODE, IMTYPE, EXPTIME, MCLOCK)
    - Additionally, if IMTYPE == 'CALUNIT', include WAVELENGTH in grouping.
    """
    # Base required columns
    required_cols = ['CAMERA', 'MODSLNAM', 'DSPRSNAM', 'IMTYPE', 'EXPTIME', 'MCLOCK']
    missing = [c for c in required_cols if c not in dt.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return []

    # Only require WAVELENGTH if there are CALUNIT rows
    has_calunit = dt['IMTYPE'].astype(str).str.upper().eq('CALUNIT').any()
    if has_calunit and 'MONOWAVE' not in dt.columns:
        print("Missing required column: ['MONOWAVE'] (needed for CALUNIT grouping)")
        return []

    file_groups = []

    # --- Split by camera first
    for cam, cam_df in dt.groupby('CAMERA'):
        cam_u = str(cam) #.upper()

        if cam_u == 'Im':
            base_keys = ['CAMERA', 'IMGFW2N', 'IMTYPE', 'EXPTIME', 'MCLOCK']
            print(f"Grouping Imager data by {base_keys}")
        elif cam_u == 'IFS':
            base_keys = ['CAMERA', 'DSPRSNAM', 'IMTYPE', 'EXPTIME', 'MCLOCK']
            print(f"Grouping IFS data by {base_keys}")
        else:
            print(f"Unknown CAMERA '{cam}' skipping.")
            continue
        # Split into CALUNIT vs non-CALUNIT so we can add WAVELENGTH only for CALUNIT
        imtype_u = cam_df['IMTYPE'].astype(str).str.upper()
        df_cal = cam_df[imtype_u.eq('CALUNIT')]
        df_non = cam_df[~imtype_u.eq('CALUNIT')]

        # Non-CALUNIT groups
        if len(df_non) > 0:
            for group_params, sub_df in df_non.groupby(base_keys):
                file_groups.append({
                    'params': {k.lower(): v for k, v in zip(base_keys, group_params)},
                    'filenames': sub_df.index.tolist()
                })

        # CALUNIT groups (add WAVELENGTH)
        if len(df_cal) > 0:
            cal_keys = base_keys + ['MONOWAVE']
            print(f"Grouping CALUNIT data by {cal_keys}")
            for group_params, sub_df in df_cal.groupby(cal_keys):
                file_groups.append({
                    'params': {k.lower(): v for k, v in zip(cal_keys, group_params)},
                    'filenames': sub_df.index.tolist()
                })

    if not file_groups:
        print("No file groups were created.")
    else:
        print(f"Created {len(file_groups)} groups from data table.")
    return file_groups


