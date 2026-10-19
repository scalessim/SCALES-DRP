from scalesdrp.core.scales_proctab import Proctab
from scalesdrp.core.scales_pkg_resources import get_resource_path
from scalesdrp.core.matplot_plotting import mpl_plot, mpl_clear
import scalesdrp.primitives.fitramp as fitramp
import scalesdrp.primitives.robust as robust
import scipy.sparse as sp
import pandas as pd
import numpy as np
import pickle
from scipy.stats import trim_mean
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
import scalesdrp.primitives.reference as reference
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import (binary_dilation, binary_opening, label,median_filter, gaussian_filter)
############# functions to check avoid processing again #########################
def read_existing_l1(l1_path):

    with fits.open(l1_path) as hdul:
        slope = hdul[0].data
        header = hdul[0].header.copy()

        if slope is None:
            raise RuntimeError("Primary HDU has no data.")

        if slope.ndim != 2:
            raise RuntimeError(
                f"Expected 2D slope image, got {slope.ndim}D."
            )

        if not np.any(np.isfinite(slope)):
            raise RuntimeError(
                "Slope image has no finite values."
            )

        # -------------------------
        # UNCERTAINTY
        # -------------------------
        uncert = None

        for hdu in hdul[1:]:
            if hdu.name.upper() in [
                "UNCERT", "ERROR", "ERR", "SIGMA"
            ]:
                uncert = hdu.data
                break

        if uncert is None:
            raise RuntimeError(
                "No uncertainty extension found."
            )

        if uncert.shape != slope.shape:
            raise RuntimeError(
                f"Uncertainty shape {uncert.shape} does not "
                f"match slope shape {slope.shape}."
            )

        # -------------------------
        # DQ
        # -------------------------
        dq = None

        if "DQ" in hdul:
            if hdul["DQ"].data is not None:
                dq = np.asarray(
                    hdul["DQ"].data,
                    dtype=np.uint32
                )

                if dq.shape != slope.shape:
                    raise RuntimeError(
                        f"DQ shape {dq.shape} does not "
                        f"match slope shape {slope.shape}."
                    )

        # -------------------------
        # CHISQ
        # -------------------------
        chisq = None

        if "CHISQ" in hdul:
            if hdul["CHISQ"].data is not None:
                chisq = np.asarray(
                    hdul["CHISQ"].data,
                    dtype=np.float32
                )

                if chisq.shape != slope.shape:
                    raise RuntimeError(
                        f"CHISQ shape {chisq.shape} does not "
                        f"match slope shape {slope.shape}."
                    )

        slope = np.asarray(
            slope,
            dtype=np.float32
        )

        uncert = np.asarray(
            uncert,
            dtype=np.float32
        )

    return slope, uncert, dq, chisq, header

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

def find_existing_proc_file1(
    input_filename,
    redux_dir,
    suffix=None,
):
    """
    Find an existing processed product for a raw input file.

    Parameters
    ----------
    input_filename : str
        Original raw FITS filename.

    redux_dir : str
        Directory containing scales.proc and processed products.

    suffix : str or None, optional
        If specified, search only for this suffix
        (e.g. '_L1', '_L10', '_L11', '_L12', '_L2').

        If None, search all supported processing levels.

    Returns
    -------
    str or None
        Full path to an existing processed product.
        None if no matching product exists.
    """

    proctab_path = os.path.join(
        redux_dir,
        "scales.proc"
    )

    if not os.path.isfile(proctab_path):
        return None

    try:
        proc = pd.read_csv(
            proctab_path,
            sep="|",
            engine="python",
            skipinitialspace=True,
        )

    except Exception as exc:
        print(
            f"Could not read proctab "
            f"{proctab_path}: {exc}"
        )
        return None

    # ---------------------------------------------------------
    # Clean processing table
    # ---------------------------------------------------------

    proc.columns = [
        str(c).strip()
        for c in proc.columns
    ]

    proc = proc.loc[
        :,
        [c for c in proc.columns if c != ""]
    ]

    if (
        "filename" not in proc.columns
        or "SUFF" not in proc.columns
    ):
        print(
            "Could not search proctab because "
            "filename/SUFF columns are missing."
        )
        return None

    proc["filename"] = (
        proc["filename"]
        .astype(str)
        .str.strip()
    )

    proc["SUFF"] = (
        proc["SUFF"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ---------------------------------------------------------
    # Raw filename stem
    # ---------------------------------------------------------

    raw_stem = _strip_fits_extension(
        os.path.basename(input_filename)
    )

    # ---------------------------------------------------------
    # Select suffix(es) to search
    # ---------------------------------------------------------

    valid_suffixes = [
        "_L1",
        "_L10",
        "_L11",
        "_L12",
        "_L13",
        "_OPT_L2",
        "_CHI_L2",
        "_mcalunit",
        "_mbias",
        "_mflatlamp",
        "_mdark",
    ]

    if suffix is None:

        suffixes_to_search = valid_suffixes

    else:

        suffix = str(suffix).strip().upper()

        if suffix not in valid_suffixes:
            raise ValueError(
                f"Unsupported processing suffix: {suffix}. "
                f"Expected one of {valid_suffixes}."
            )

        suffixes_to_search = [suffix]

    # ---------------------------------------------------------
    # Search proc table
    # ---------------------------------------------------------

    matches = proc[
        proc["SUFF"].isin(suffixes_to_search)
        &
        proc["filename"].str.contains(
            raw_stem,
            regex=False,
            na=False,
        )
    ]

    if len(matches) == 0:
        return None

    # ---------------------------------------------------------
    # Confirm that the actual FITS file exists
    # ---------------------------------------------------------

    for _, row in matches.iloc[::-1].iterrows():

        found_name = str(
            row["filename"]
        ).strip()

        found_path = os.path.join(
            redux_dir,
            found_name
        )

        if os.path.isfile(found_path):
            return found_path

    return None


def find_existing_proc_file(
    input_filename,
    redux_dir,
    suffix=None,
):
    """
    Find an existing processed FITS product using scales.proc.

    Parameters
    ----------
    input_filename : str
        Original raw FITS filename.

    redux_dir : str
        Directory containing scales.proc and processed FITS files.

    suffix : str, list, tuple, set, or None, optional
        Processing suffix(es) to search.

        Examples:
            "_L1"
            "_mdark"
            ["_mdark", "_mflatlamp", "_mcalunit"]

        If None, search all supported processing levels.

    Returns
    -------
    str or None
        Full path to the first matching existing FITS product.
        Returns None if no matching file exists.

    Notes
    -----
    - Suffix matching is case-insensitive.
    - Only files registered in scales.proc are considered.
    - The actual FITS file must exist on disk.
    - If multiple files match, the most recently listed
      matching product in scales.proc is returned.
    """

    redux_dir = os.path.abspath(os.fspath(redux_dir))

    proctab_path = os.path.join(
        redux_dir,
        "scales.proc"
    )

    if not os.path.isfile(proctab_path):
        return None

    # ---------------------------------------------------------
    # Read processing table
    # ---------------------------------------------------------

    try:
        proc = pd.read_csv(
            proctab_path,
            sep="|",
            engine="python",
            skipinitialspace=True,
        )

    except Exception as exc:
        print(
            f"Could not read proctab "
            f"{proctab_path}: {exc}"
        )
        return None

    # ---------------------------------------------------------
    # Clean processing table
    # ---------------------------------------------------------

    proc.columns = [
        str(c).strip()
        for c in proc.columns
    ]

    proc = proc.loc[
        :,
        [c for c in proc.columns if c != ""]
    ].copy()

    if (
        "filename" not in proc.columns
        or "SUFF" not in proc.columns
    ):
        print(
            "Could not search proctab because "
            "filename/SUFF columns are missing."
        )
        return None

    proc["filename"] = (
        proc["filename"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    proc["SUFF"] = (
        proc["SUFF"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ---------------------------------------------------------
    # Supported processing suffixes
    # ---------------------------------------------------------

    valid_suffixes = [
        "_L1",
        "_L10",
        "_L11",
        "_L12",
        "_L13",
        "_OPT_L2",
        "_CHI_L2",
        "_mcalunit",
        "_mbias",
        "_mflatlamp",
        "_mdark",
    ]

    valid_upper = {
        s.upper()
        for s in valid_suffixes
    }

    # ---------------------------------------------------------
    # Select suffixes
    # ---------------------------------------------------------

    if suffix is None:

        suffixes_to_search = list(valid_upper)

    elif isinstance(suffix, (list, tuple, set)):

        suffixes_to_search = [
            str(s).strip().upper()
            for s in suffix
        ]

    else:

        suffixes_to_search = [
            str(suffix).strip().upper()
        ]

    invalid_suffixes = [
        s for s in suffixes_to_search
        if s not in valid_upper
    ]

    if invalid_suffixes:
        raise ValueError(
            f"Unsupported processing suffix(es): "
            f"{invalid_suffixes}. "
            f"Supported suffixes: {valid_suffixes}"
        )

    if not suffixes_to_search:
        return None

    # ---------------------------------------------------------
    # Determine original raw filename stem
    # ---------------------------------------------------------

    raw_stem = _strip_fits_extension(
        os.path.basename(
            os.fspath(input_filename)
        )
    )

    # ---------------------------------------------------------
    # Match filenames and processing suffixes
    # ---------------------------------------------------------

    # Require the processed filename to begin with the
    # original raw filename stem followed by a separator.
    #
    # Examples:
    #
    # raw:
    #   ss260605_04802.fits
    #
    # processed:
    #   ss260605_04802_L1.fits
    #   ss260605_04802_L1_2370.25_mcalunit.fits

    proc_names = proc["filename"].map(
        lambda name: os.path.basename(name)
    )

    filename_match = (
        proc_names.str.startswith(raw_stem + "_")
        | proc_names.eq(raw_stem + ".fits")
        | proc_names.eq(raw_stem + ".fits.gz")
    )

    suffix_match = proc["SUFF"].isin(
        suffixes_to_search
    )

    matches = proc.loc[
        filename_match & suffix_match
    ]

    if matches.empty:
        return None

    # ---------------------------------------------------------
    # Confirm FITS file exists
    # ---------------------------------------------------------

    for _, row in matches.iloc[::-1].iterrows():

        found_name = str(
            row["filename"]
        ).strip()

        if os.path.isabs(found_name):
            found_path = found_name
        else:
            found_path = os.path.join(
                redux_dir,
                found_name
            )

        if os.path.isfile(found_path):
            return found_path

    return None
##############################################################################

def load_single_master_file_calib(expected_keywords, master_type):
    """
    Load one matching master calibration file from ./redux or package calib/.
    """

    master_type = master_type.upper()

    tail_map = {
        "DARK": "_L1_mdark.fits",
        "BIAS": "_L1_mbias.fits",
        "FLATLAMP": "_L1_mflatlamp.fits",
        "FLATLENS": "_L1_mflatlens.fits",
        "CALUNIT": "_L1_mcalunit.fits",
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
    R_matrix,
    data_image,
    sigma_image,
):
    """
    Extract a cube and propagate the detector uncertainty.

    Assumptions
    -----------
    - R_matrix is a detector-to-cube rectification matrix.
    - R_matrix contains nonnegative, already-normalized extraction weights.
    - sigma_image is the ramp-fit 1-sigma uncertainty and already includes
      the relevant read-noise and photon-noise terms.

    Parameters
    ----------
    R_matrix : scipy sparse matrix
        Shape (N_cube_voxels, N_detector_pixels).

    data_image : ndarray
        Two-dimensional detector slope image.

    sigma_image : ndarray or uncertainty object
        Two-dimensional ramp-fit 1-sigma uncertainty image.

    cube_shape : tuple
        Output cube shape, for example (179, 17, 18).

    Returns
    -------
    flux_cube : ndarray
        Extracted flux cube.

    error_cube : ndarray
        Propagated 1-sigma error cube.
    """

    print("Optimal extraction started")
    start_time = time.time()

    data_vector = np.asarray(
        data_image,
        dtype=np.float64,
    ).ravel()

    if hasattr(sigma_image, "array"):
        sigma_vector = np.asarray(
            sigma_image.array,
            dtype=np.float64,
        ).ravel()
    else:
        sigma_vector = np.asarray(
            sigma_image,
            dtype=np.float64,
        ).ravel()

    # Invalid science pixels contribute zero to the extracted cube.
    data_safe = np.where(
        np.isfinite(data_vector),
        data_vector,
        0.0,
    )

    # Extract flux using the existing rectification matrix.
    flux_vector = np.asarray(
        R_matrix @ data_safe
    ).ravel()

    # Ramp-fit variance.
    variance_vector = sigma_vector**2

    # Invalid uncertainties contribute no propagated variance.
    variance_safe = np.where(
        np.isfinite(variance_vector) & (variance_vector >= 0),
        variance_vector,
        0.0,
    )

    # Propagate independent detector-pixel variances:
    #
    # Var(f_j) = sum_i R_ji^2 Var(d_i)
    R_squared = R_matrix.multiply(R_matrix)

    flux_variance_vector = np.asarray(
        R_squared @ variance_safe
    ).ravel()

    error_vector = np.sqrt(
        np.maximum(flux_variance_vector, 0.0)
    )

    elapsed = time.time() - start_time

    print(
        f"Optimal extraction finished in {elapsed:.4f} seconds."
    )

    return flux_vector, error_vector


def optimal_extract_horne(
    R_matrix,
    data_image,
    sigma_image,
):
    """
    Perform Horne-style optimal extraction.

    For each cube voxel k, the extracted flux is

        F_k = sum_i [P_ki * d_i / sigma_i^2]
              --------------------------------
              sum_i [P_ki^2 / sigma_i^2]

    where
        d_i       : detector pixel value
        sigma_i   : 1-sigma uncertainty of detector pixel i
        P_ki      : normalized detector profile for cube voxel k

    The formal uncertainty is

        sigma_F,k = 1 / sqrt(
                        sum_i [P_ki^2 / sigma_i^2]
                    )

    Assumptions
    -----------
    - R_matrix has shape
          (N_cube_voxels, N_detector_pixels)

    - Each row of R_matrix represents the detector profile P_ki
      associated with one cube voxel.

    - The profiles are non-negative and appropriately normalized.

    - sigma_image contains the detector-pixel 1-sigma uncertainties
      from the ramp fitting.

    - Detector-pixel errors are independent.

    Parameters
    ----------
    R_matrix : scipy.sparse matrix
        Rectification/profile matrix with shape
        (N_cube_voxels, N_detector_pixels).

    data_image : ndarray
        Two-dimensional detector slope image.

    sigma_image : ndarray or uncertainty object
        Two-dimensional detector 1-sigma uncertainty image.

    Returns
    -------
    flux_vector : ndarray
        Horne optimally extracted flux for each cube voxel.

    error_vector : ndarray
        Formal 1-sigma uncertainty for each extracted cube voxel.
    """

    print("Horne optimal extraction started")
    start_time = time.time()

    # ------------------------------------------------------------
    # Flatten detector data
    # ------------------------------------------------------------
    data_vector = np.asarray(
        data_image,
        dtype=np.float64,
    ).ravel()

    if hasattr(sigma_image, "array"):
        sigma_vector = np.asarray(
            sigma_image.array,
            dtype=np.float64,
        ).ravel()
    else:
        sigma_vector = np.asarray(
            sigma_image,
            dtype=np.float64,
        ).ravel()

    # ------------------------------------------------------------
    # Check dimensions
    # ------------------------------------------------------------
    if R_matrix.shape[1] != data_vector.size:
        raise ValueError(
            f"R_matrix has {R_matrix.shape[1]} detector pixels, "
            f"but data_image contains {data_vector.size} pixels."
        )

    if sigma_vector.size != data_vector.size:
        raise ValueError(
            "sigma_image and data_image must contain the same "
            "number of detector pixels."
        )

    # ------------------------------------------------------------
    # Construct valid-pixel mask
    # ------------------------------------------------------------
    valid = (
        np.isfinite(data_vector)
        & np.isfinite(sigma_vector)
        & (sigma_vector > 0)
    )

    # Inverse variance:
    #
    #     w_i = 1 / sigma_i^2
    #
    inv_variance = np.zeros_like(sigma_vector)

    inv_variance[valid] = (
        1.0 / sigma_vector[valid]**2
    )

    # Invalid detector data are set to zero. Since their
    # inverse variance is also zero, they do not contribute.
    data_safe = np.where(
        valid,
        data_vector,
        0.0,
    )

    # ------------------------------------------------------------
    # Horne numerator
    #
    #     numerator_k =
    #         sum_i P_ki d_i / sigma_i^2
    #
    # ------------------------------------------------------------
    weighted_data = data_safe * inv_variance

    numerator = np.asarray(
        R_matrix @ weighted_data
    ).ravel()

    # ------------------------------------------------------------
    # Horne denominator
    #
    #     denominator_k =
    #         sum_i P_ki^2 / sigma_i^2
    #
    # ------------------------------------------------------------
    R_squared = R_matrix.multiply(R_matrix)

    denominator = np.asarray(
        R_squared @ inv_variance
    ).ravel()

    # ------------------------------------------------------------
    # Extracted flux
    #
    #     F_k = numerator_k / denominator_k
    #
    # ------------------------------------------------------------
    flux_vector = np.full(
        denominator.shape,
        np.nan,
        dtype=np.float64,
    )

    good = (
        np.isfinite(denominator)
        & (denominator > 0)
    )

    flux_vector[good] = (
        numerator[good] / denominator[good]
    )

    # ------------------------------------------------------------
    # Formal Horne uncertainty
    #
    #     sigma_F,k =
    #         1 / sqrt(denominator_k)
    #
    # ------------------------------------------------------------
    error_vector = np.full(
        denominator.shape,
        np.nan,
        dtype=np.float64,
    )

    error_vector[good] = (
        1.0 / np.sqrt(denominator[good])
    )

    elapsed = time.time() - start_time

    print(
        f"Horne optimal extraction finished in "
        f"{elapsed:.4f} seconds."
    )

    return flux_vector, error_vector

def optimal_extract_fast(
    R_transpose: sp.spmatrix,
    data_image: np.ndarray) -> np.ndarray:
    t0 = time.time()
    optimized_flux = np.array(R_transpose*np.matrix(data_image.reshape([2048*2048,1])))
    return optimized_flux

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
    sigma_stack=None,           # None or (N,H,W)
    chisq_stack=None,           # None or (N,H,W)
    quality_stack=None,         # None or (N,H,W), uint16 DQ maps
    *,
    method="ivw",               # 'ivw', 'mean', or 'median'
    clip_sigma=5.0,             # sigma threshold for iterative clipping
    iterations=3,               # maximum clipping iterations
    min_valid=2,                # minimum surviving frames per pixel
    frame_scales=None,          # optional multiplicative scale per frame
    return_mask=False,          # optionally return final combination mask
):
    """
    Combine a stack of L1 calibration products into a master calibration.

    Each L1 product may contain:

        DATA    : calibrated 2-D image
        UNCERT  : 1-sigma uncertainty
        CHISQ   : ramp-fit chi-square diagnostic
        DQ      : uint16 detector quality map

    The DATA stack is combined using inverse-variance weighting, mean,
    or median after iterative sigma clipping.

    UNCERT is propagated according to the selected combination method.

    CHISQ is treated as a diagnostic quantity and is propagated using
    the median of the CHISQ values corresponding to samples that survive
    the DATA sigma-clipping.

    DQ is also diagnostic at this stage and does NOT reject samples from
    the master combination. The master DQ is constructed using a bitwise
    OR across the input L1 DQ maps, preserving any detector-quality flag
    that occurred in any contributing exposure.

    Parameters
    ----------
    data_stack : array-like
        Stack of input frames with shape (N,H,W), or a single (H,W)
        frame.

    sigma_stack : array-like or None
        Stack of 1-sigma uncertainties with the same shape as data_stack.
        Required when method='ivw'.

    chisq_stack : array-like or None
        Stack of CHISQ images with the same shape as data_stack.

    quality_stack : array-like or None
        Stack of uint16 DQ maps with the same shape as data_stack.

        Current DQ definition:

            bit 0 : BPM
            bit 1 : pixel directly saturated during the ramp
            bit 2 : saturated neighbor forced earlier ramp cutoff

        Bits may be combined.

        DQ does not participate in the numerical master combination.
        The output master DQ is the bitwise OR of the input DQ maps.

    method : {'ivw', 'mean', 'median'}
        Combination method.

        ivw
            Inverse-variance weighted mean.

        mean
            Unweighted mean.

        median
            Unweighted median.

    clip_sigma : float
        Sigma threshold for iterative outlier rejection along the
        frame axis. Set <= 0 or non-finite to disable clipping.

    iterations : int
        Maximum number of clipping iterations.

    min_valid : int
        Minimum number of surviving numerical samples required to
        produce a valid master pixel.

    frame_scales : array-like or None
        Optional length-N multiplicative scale applied to DATA.
        UNCERT is scaled by the corresponding absolute scale.
        CHISQ and DQ are not scaled.

    return_mask : bool
        If True, also return the final per-frame combination mask
        with shape (N,H,W).

    Returns
    -------
    master : ndarray, float32
        Master DATA image, shape (H,W).

    master_uncert : ndarray, float32
        Propagated 1-sigma uncertainty, shape (H,W).

    master_chisq : ndarray or None, float32
        Median CHISQ of samples surviving the DATA clipping.

    master_dq : ndarray or None, uint16
        Bitwise OR of the input L1 DQ maps.

    final_mask : ndarray, bool, optional
        Returned only when return_mask=True.
        True indicates that the corresponding L1 DATA sample
        contributed to the final master calculation.
    """

    # ============================================================
    # DATA
    # ============================================================

    data = np.asarray(data_stack, dtype=np.float64)

    if data.ndim == 2:
        data = data[None, :, :]

    if data.ndim != 3:
        raise ValueError(
            "data_stack must have shape (N,H,W) "
            "or be a single (H,W) frame."
        )

    N, H, W = data.shape

    if np.isnan(data).any():
        print("NaN in DATA stack while creating master.")


    # ============================================================
    # UNCERTAINTY
    # ============================================================

    if sigma_stack is None:

        sigma = None

        if method == "ivw":
            raise ValueError(
                "method='ivw' requires sigma_stack."
            )

    else:

        sigma = np.asarray(
            sigma_stack,
            dtype=np.float64
        )

        if sigma.ndim == 2:
            sigma = sigma[None, :, :]

        if sigma.shape != data.shape:
            raise ValueError(
                f"sigma_stack shape {sigma.shape} "
                f"must match data_stack shape {data.shape}."
            )

        if np.isnan(sigma).any():
            print("NaN in UNCERTAINTY stack while creating master.")


    # ============================================================
    # CHISQ
    # ============================================================

    if chisq_stack is None:

        chisq = None

    else:

        chisq = np.asarray(
            chisq_stack,
            dtype=np.float64
        )

        if chisq.ndim == 2:
            chisq = chisq[None, :, :]

        if chisq.shape != data.shape:
            raise ValueError(
                f"chisq_stack shape {chisq.shape} "
                f"must match data_stack shape {data.shape}."
            )


    # ============================================================
    # DQ / QUALITY MAP
    # ============================================================

    if quality_stack is None:

        quality = None
        master_dq = None

    else:

        quality = np.asarray(
            quality_stack,
            dtype=np.uint16
        )

        if quality.ndim == 2:
            quality = quality[None, :, :]

        if quality.shape != data.shape:
            raise ValueError(
                f"quality_stack shape {quality.shape} "
                f"must match data_stack shape {data.shape}."
            )

        # Preserve every DQ condition encountered in the L1 stack.
        #
        # Example:
        #     frame 1 : 000
        #     frame 2 : 010  SATURATED
        #     frame 3 : 100  SAT_NEIGHBOR
        #
        #     master  : 110
        #
        master_dq = np.bitwise_or.reduce(
            quality,
            axis=0
        ).astype(np.uint16)


    # ============================================================
    # OPTIONAL PER-FRAME SCALING
    # ============================================================

    if frame_scales is not None:

        scales = np.asarray(
            frame_scales,
            dtype=np.float64
        )

        if scales.shape != (N,):
            raise ValueError(
                f"frame_scales must have shape ({N},)."
            )

        data = (
            data *
            scales[:, None, None]
        )

        if sigma is not None:
            sigma = (
                sigma *
                np.abs(scales[:, None, None])
            )

        # CHISQ is diagnostic and is not scaled.
        # DQ is a bitmask and is not scaled.


    # ============================================================
    # INITIAL NUMERICAL VALIDITY MASK
    # ============================================================

    # IMPORTANT:
    # DQ is intentionally NOT included in this mask.
    #
    # At the L1 stage detector corrections have already been
    # performed. The DQ map is retained as observer-facing
    # information rather than being used to reject corrected
    # pixels from master construction.

    mask = np.isfinite(data)

    if sigma is not None:
        mask &= (
            np.isfinite(sigma)
            & (sigma > 0)
        )

    # Working mask for iterative clipping.
    m = mask.copy()


    # ============================================================
    # ITERATIVE SIGMA CLIPPING
    # ============================================================

    for _ in range(max(0, int(iterations))):

        old_m = m.copy()

        # --------------------------------------------------------
        # IVW
        # --------------------------------------------------------

        if method == "ivw":

            if sigma is None:
                raise ValueError(
                    "method='ivw' requires sigma_stack."
                )

            w = np.where(
                m,
                1.0 / np.maximum(
                    sigma,
                    1e-30
                )**2,
                0.0
            )

            wsum = np.sum(
                w,
                axis=0
            )

            numerator = np.sum(
                np.where(
                    m,
                    w * data,
                    0.0
                ),
                axis=0
            )

            center = np.divide(
                numerator,
                wsum,
                out=np.full(
                    (H, W),
                    np.nan,
                    dtype=np.float64
                ),
                where=wsum > 0
            )

            residual = (
                data -
                center[None, :, :]
            )

            var_numerator = np.sum(
                np.where(
                    m,
                    w * residual**2,
                    0.0
                ),
                axis=0
            )

            variance = np.divide(
                var_numerator,
                wsum,
                out=np.full(
                    (H, W),
                    np.nan,
                    dtype=np.float64
                ),
                where=wsum > 0
            )

            scatter = np.sqrt(
                variance
            )


        # --------------------------------------------------------
        # MEAN
        # --------------------------------------------------------

        elif method == "mean":

            valid = np.where(
                m,
                data,
                np.nan
            )

            center = np.nanmean(
                valid,
                axis=0
            )

            scatter = np.nanstd(
                valid,
                axis=0
            )


        # --------------------------------------------------------
        # MEDIAN
        # --------------------------------------------------------

        elif method == "median":

            valid = np.where(
                m,
                data,
                np.nan
            )

            center = np.nanmedian(
                valid,
                axis=0
            )

            mad = np.nanmedian(
                np.abs(
                    valid -
                    center[None, :, :]
                ),
                axis=0
            )

            scatter = (
                1.4826 * mad
            )


        else:

            raise ValueError(
                "method must be 'ivw', 'mean', or 'median'."
            )


        # --------------------------------------------------------
        # CLIPPING DISABLED
        # --------------------------------------------------------

        if (
            not np.isfinite(clip_sigma)
            or clip_sigma <= 0
        ):
            break


        # --------------------------------------------------------
        # UPDATE CLIPPING MASK
        # --------------------------------------------------------

        residual = np.abs(
            data -
            center[None, :, :]
        )

        threshold = (
            clip_sigma *
            scatter
        )

        keep = (
            residual <=
            threshold[None, :, :]
        )

        # A sample must have been numerically valid initially
        # and must pass the clipping criterion.
        m &= keep


        # --------------------------------------------------------
        # STOP WHEN NOTHING CHANGES
        # --------------------------------------------------------

        if np.array_equal(
            m,
            old_m
        ):
            break


    # ============================================================
    # NUMBER OF SURVIVING SAMPLES
    # ============================================================

    n_eff = np.sum(
        m,
        axis=0
    )

    good_pix = (
        n_eff >= max(
            1,
            int(min_valid)
        )
    )


    # ============================================================
    # INITIALIZE OUTPUT ARRAYS
    # ============================================================

    master = np.full(
        (H, W),
        np.nan,
        dtype=np.float64
    )

    master_unc = np.full(
        (H, W),
        np.nan,
        dtype=np.float64
    )


    # ============================================================
    # FINAL IVW COMBINATION
    # ============================================================

    if method == "ivw":

        w = np.where(
            m,
            1.0 / np.maximum(
                sigma,
                1e-30
            )**2,
            0.0
        )

        wsum = np.sum(
            w,
            axis=0
        )

        numerator = np.sum(
            np.where(
                m,
                w * data,
                0.0
            ),
            axis=0
        )

        value = np.divide(
            numerator,
            wsum,
            out=np.full(
                (H, W),
                np.nan,
                dtype=np.float64
            ),
            where=wsum > 0
        )

        uncertainty = np.divide(
            1.0,
            np.sqrt(wsum),
            out=np.full(
                (H, W),
                np.nan,
                dtype=np.float64
            ),
            where=wsum > 0
        )

        master[good_pix] = (
            value[good_pix]
        )

        master_unc[good_pix] = (
            uncertainty[good_pix]
        )


    # ============================================================
    # FINAL MEAN COMBINATION
    # ============================================================

    elif method == "mean":

        valid = np.where(
            m,
            data,
            np.nan
        )

        value = np.nanmean(
            valid,
            axis=0
        )

        scatter = np.nanstd(
            valid,
            axis=0
        )

        uncertainty = np.divide(
            scatter,
            np.sqrt(n_eff),
            out=np.full(
                (H, W),
                np.nan,
                dtype=np.float64
            ),
            where=n_eff > 0
        )

        master[good_pix] = (
            value[good_pix]
        )

        master_unc[good_pix] = (
            uncertainty[good_pix]
        )


    # ============================================================
    # FINAL MEDIAN COMBINATION
    # ============================================================

    elif method == "median":

        valid = np.where(
            m,
            data,
            np.nan
        )

        value = np.nanmedian(
            valid,
            axis=0
        )

        mad = np.nanmedian(
            np.abs(
                valid -
                value[None, :, :]
            ),
            axis=0
        )

        sigma_robust = (
            1.4826 * mad
        )

        uncertainty = np.divide(
            1.2533 * sigma_robust,
            np.sqrt(n_eff),
            out=np.full(
                (H, W),
                np.nan,
                dtype=np.float64
            ),
            where=n_eff > 0
        )

        master[good_pix] = (
            value[good_pix]
        )

        master_unc[good_pix] = (
            uncertainty[good_pix]
        )


    # ============================================================
    # PROPAGATE CHISQ
    # ============================================================

    if chisq is not None:

        # Use the same frame-selection mask that was determined
        # from the DATA combination.
        #
        # CHISQ itself does not control clipping.

        valid_chisq = np.where(
            m & np.isfinite(chisq),
            chisq,
            np.nan
        )

        master_chisq = np.nanmedian(
            valid_chisq,
            axis=0
        )

        # No valid master DATA -> no meaningful master CHISQ.
        master_chisq[
            ~good_pix
        ] = np.nan

        master_chisq = (
            master_chisq.astype(
                np.float32
            )
        )

    else:

        master_chisq = None


    # ============================================================
    # ENFORCE min_valid
    # ============================================================

    master[
        ~good_pix
    ] = np.nan

    master_unc[
        ~good_pix
    ] = np.nan


    # ============================================================
    # OUTPUT TYPES
    # ============================================================

    master = master.astype(
        np.float32
    )

    master_unc = master_unc.astype(
        np.float32
    )

    if master_dq is not None:
        master_dq = master_dq.astype(
            np.uint16
        )


    # ============================================================
    # RETURN
    # ============================================================

    if return_mask:

        return (
            master,
            master_unc,
            master_chisq,
            master_dq,
            m,
        )

    return (
        master,
        master_unc,
        master_chisq,
        master_dq,
    )
########################### ramp fitting #####################################################
def _ols_row_and_uncert1(row_reads,valid_reads_mask,t,sig_row):

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

##################################################################################
def ramp_fit(
    input_read,
    total_exptime,
    SIG_map_scaled,
    *,
    return_pedestal=True,
    reset_prior_strength=3.0,
    use_sigma_clip=False,
    sigma_clip=3.0,
    max_iter=3,
    min_reads=5,
    tile=(128, 128),
    JUMP_THRESH_ONEOMIT=20.25,
    JUMP_THRESH_TWOOMIT=23.8,
    group_dq=None,
):
    """
    Fit detector ramps and return slope, uncertainty, chi-square,
    and optionally pedestal.

    The preferred fit is fitramp with pedestal=True. If that fit
    cannot be performed for a pixel, a robust OLS fit is used as
    a fallback.

    Parameters
    ----------
    input_read : ndarray, shape (Nreads, H, W)
        Input detector ramp.

    total_exptime : float
        Total exposure time.

    SIG_map_scaled : ndarray, shape (H, W)
        Per-pixel noise estimate used by the ramp fitter.

    return_pedestal : bool, optional
        If True, return pedestal in addition to slope, uncertainty,
        and chi-square.

    reset_prior_strength : float, optional
        Reset prior sigma is:

            reset_prior_strength * SIG_map_scaled

    use_sigma_clip : bool, optional
        Apply an initial sigma-clipping mask to the reads.

    sigma_clip : float, optional
        Sigma threshold used by the optional clipping.

    max_iter : int, optional
        Maximum number of sigma-clipping iterations.

    min_reads : int, optional
        Minimum number of reads used during sigma clipping.

    tile : tuple, optional
        Tile size used for sigma clipping.

    JUMP_THRESH_ONEOMIT : float, optional
        fitramp threshold for one-omit jump detection.

    JUMP_THRESH_TWOOMIT : float, optional
        fitramp threshold for two-omit jump detection.

    group_dq : ndarray, bool, shape (Nreads, H, W), optional
        Dynamic read-validity mask.

        True  = use read
        False = reject read

        The dynamic mask is only applied to a pixel if at least one
        valid read difference remains. Otherwise the unmasked ramp is
        used for that pixel.

    Returns
    -------
    If return_pedestal=True:

        slope, pedestal, uncertainty, chisq

    otherwise:

        slope, uncertainty, chisq
    """

    N, H, W = input_read.shape

    dt = float(total_exptime) / N

    t = (
        np.arange(N, dtype=float) + 0.5
    ) * dt

    t1 = time.time()

    # ================================================================
    # Helper: sigma clipping
    # ================================================================

    def _sigma_clip_reads(cube):

        from tqdm import tqdm

        keep = np.ones_like(
            cube,
            dtype=bool
        )

        Ty, Tx = tile

        for y0 in tqdm(
            range(0, H, Ty),
            desc="σ-clip tiles"
        ):

            y1 = min(
                H,
                y0 + Ty
            )

            for x0 in range(0, W, Tx):

                x1 = min(
                    W,
                    x0 + Tx
                )

                sub = cube[
                    :,
                    y0:y1,
                    x0:x1
                ]

                n, ty, tx = sub.shape

                k = ty * tx

                Y = sub.reshape(
                    n,
                    k
                )

                mask = np.isfinite(Y)

                time_idx = np.arange(
                    n,
                    dtype=np.float32
                )

                for _ in range(max_iter):

                    cnt = mask.sum(axis=0)

                    if not np.any(
                        cnt >= min_reads
                    ):
                        break

                    S0 = cnt

                    St = (
                        time_idx @ mask
                    )

                    Stt = (
                        (time_idx ** 2) @ mask
                    )

                    Wy = Y * mask

                    Sy = Wy.sum(axis=0)

                    Sty = (
                        time_idx @ Wy
                    )

                    Var_t = (
                        Stt
                        - (St * St)
                        / np.maximum(S0, 1)
                    )

                    Cov_ty = (
                        Sty
                        - (St * Sy)
                        / np.maximum(S0, 1)
                    )

                    b = np.zeros(
                        k,
                        dtype=np.float32
                    )

                    ok = Var_t > 0

                    b[ok] = (
                        Cov_ty[ok]
                        / Var_t[ok]
                    )

                    a = (
                        Sy - b * St
                    ) / np.maximum(S0, 1)

                    Yhat = (
                        a
                        + np.outer(
                            time_idx,
                            b
                        )
                    )

                    resid = Y - Yhat

                    resid[~mask] = np.nan

                    s = np.nanstd(
                        resid,
                        axis=0
                    )

                    new_mask = (
                        (np.abs(resid) < sigma_clip * s)
                        & np.isfinite(Y)
                    )

                    if np.array_equal(
                        new_mask,
                        mask
                    ):
                        break

                    mask = new_mask

                keep[
                    :,
                    y0:y1,
                    x0:x1
                ] = mask.reshape(
                    n,
                    ty,
                    tx
                )

        return keep

    # ================================================================
    # Helper: OLS chi-square
    # ================================================================
    
    def _ols_chisq_row(
        reads_row,
        valid_row,
        t,
        slope_row,
        sig_row,
    ):
        """
        Vectorized OLS chi-square for one detector row.

        Parameters
        ----------
        reads_row : ndarray, shape (N, W)
        valid_row : ndarray, bool, shape (N, W)
        t : ndarray, shape (N,)
        slope_row : ndarray, shape (W,)
        sig_row : ndarray, shape (W,)

        Returns
        -------
        chisq_row : ndarray, shape (W,)
        """

        # Valid points
        mask = (
            valid_row
            & np.isfinite(reads_row)
        )

        nvalid = mask.sum(axis=0)

        # --------------------------------------------------------------
        # OLS intercept corresponding to supplied slope
        #
        # intercept = mean(y - slope*t)
        # over valid reads
        # --------------------------------------------------------------

        model_without_intercept = (
            t[:, None] * slope_row[None, :]
        )

        residual0 = (
            reads_row
            - model_without_intercept
        )

        # Ignore invalid reads
        residual0_masked = np.where(
            mask,
            residual0,
            0.0
        )

        intercept = np.full(
            reads_row.shape[1],
            np.nan,
            dtype=float
        )

        good_count = nvalid > 0

        intercept[good_count] = (
            residual0_masked[:, good_count].sum(axis=0)
            / nvalid[good_count]
        )

        # --------------------------------------------------------------
        # Full model and residual
        # --------------------------------------------------------------

        model = (
            intercept[None, :]
            + model_without_intercept
        )

        residual = (
            reads_row - model
        )

        # --------------------------------------------------------------
        # Chi-square
        # --------------------------------------------------------------

        valid_sigma = (
            np.isfinite(sig_row)
            & (sig_row > 0)
        )

        good = (
            (nvalid >= 2)
            & np.isfinite(slope_row)
            & np.isfinite(intercept)
            & valid_sigma
        )

        chisq_row = np.full(
            reads_row.shape[1],
            np.nan,
            dtype=float
        )

        normalized_resid2 = (
            residual
            / sig_row[None, :]
        ) ** 2

        normalized_resid2 = np.where(
            mask,
            normalized_resid2,
            0.0
        )

        chisq_all = np.sum(
            normalized_resid2,
            axis=0
        )

        chisq_row[good] = (
            chisq_all[good]
        )

        return chisq_row

    # ================================================================
    # Initial read-validity mask
    # ================================================================

    base_valid_nomask = np.isfinite(
        input_read
    )

    if use_sigma_clip:

        base_valid_nomask &= (
            _sigma_clip_reads(
                input_read
            )
        )

    # ================================================================
    # Read differences
    # ================================================================

    diffs = (
        input_read[1:]
        - input_read[:-1]
    )

    # ================================================================
    # Dynamic saturation / group DQ mask
    # ================================================================

    n_fallback = 0

    if group_dq is not None:

        group_dq = np.asarray(
            group_dq,
            dtype=bool
        )

        if (
            group_dq.shape
            != input_read.shape
        ):
            raise ValueError(
                f"group_dq shape "
                f"{group_dq.shape} "
                f"!= input_read shape "
                f"{input_read.shape}"
            )

        # Apply the dynamic mask
        masked_valid = (
            base_valid_nomask
            & group_dq
        )

        # Valid differences after dynamic masking
        masked_pair = (
            masked_valid[1:]
            & masked_valid[:-1]
            & (diffs > 0)
        )

        # Dynamic mask may only be applied where at least
        # one valid difference remains.
        can_apply_mask = np.any(
            masked_pair,
            axis=0
        )

        # Did group_dq actually remove anything?
        dq_changes_pixel = np.any(
            base_valid_nomask
            & ~group_dq,
            axis=0
        )

        # Pixels where applying group_dq would leave
        # no usable difference.
        fallback_pixels = (
            dq_changes_pixel
            & ~can_apply_mask
        )

        n_fallback = np.count_nonzero(
            fallback_pixels
        )

        # Apply group_dq only where it leaves a usable ramp.
        base_valid = np.where(
            can_apply_mask[None, :, :],
            masked_valid,
            base_valid_nomask
        )

    else:

        base_valid = (
            base_valid_nomask
        )

    # ================================================================
    # Physical difference mask
    # ================================================================

    pair_mask = (
        base_valid[1:]
        & base_valid[:-1]
        & (diffs > 0)
    )

    if n_fallback > 0:

        print(
            f"Dynamic saturation mask ignored for "
            f"{n_fallback:,} pixels because it would "
            f"remove all valid read differences."
        )

    # ================================================================
    # Reset prior
    # ================================================================

    first_read = input_read[0]

    first_ok = (
        base_valid[0]
        & np.isfinite(first_read)
    )

    resetval_map = np.where(
        first_ok,
        first_read,
        0.0
    )

    resetsig_map = np.where(
        first_ok,
        reset_prior_strength
        * SIG_map_scaled,
        np.inf
    )

    # ================================================================
    # fitramp covariance objects
    # ================================================================

    C_no = fitramp.Covar(
        t,
        pedestal=False
    )

    C_ped = fitramp.Covar(
        t,
        pedestal=True
    )

    # ================================================================
    # Output arrays
    # ================================================================

    slope = np.full(
        (H, W),
        np.nan,
        dtype=float
    )

    ped = np.full(
        (H, W),
        np.nan,
        dtype=float
    )

    uncert = np.full(
        (H, W),
        np.nan,
        dtype=float
    )

    chisq = np.full(
        (H, W),
        np.nan,
        dtype=float
    )

    # ================================================================
    # Ramp fit row-by-row
    # ================================================================

    for i in range(H):

        if i % 128 == 0:
            print(
                f"Fitting row {i}/{H}..."
            )

        sig_row = (
            SIG_map_scaled[i, :]
        )

        d_row = (
            diffs[:, i, :]
        )

        m_row = (
            pair_mask[:, i, :]
        )

        resetval = (
            resetval_map[i, :]
        )

        resetsig = (
            resetsig_map[i, :]
        )

        Wrow = d_row.shape[1]

        row_slope = np.full(
            Wrow,
            np.nan,
            dtype=float
        )

        row_ped = np.full(
            Wrow,
            np.nan,
            dtype=float
        )

        row_uncert = np.full(
            Wrow,
            np.nan,
            dtype=float
        )

        row_chisq = np.full(
            Wrow,
            np.nan,
            dtype=float
        )

        # ------------------------------------------------------------
        # 0) Number of usable differences before jump detection
        # ------------------------------------------------------------

        usable0 = m_row.sum(
            axis=0
        )

        # ------------------------------------------------------------
        # 1) Initial slope seed
        # ------------------------------------------------------------

        seed = np.full(
            Wrow,
            np.nan,
            dtype=float
        )

        idx_seed_fit = (
            usable0 >= 2
        )

        idx_seed_ols = (
            ~idx_seed_fit
        )

        # ------------------------------------------------------------
        # fitramp seed
        # ------------------------------------------------------------

        if np.any(idx_seed_fit):

            try:

                d_sub = d_row[
                    :,
                    idx_seed_fit
                ]

                m_sub = m_row[
                    :,
                    idx_seed_fit
                ]

                sig_sub = sig_row[
                    idx_seed_fit
                ]

                with warnings.catch_warnings():

                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning
                    )

                    init_res = fitramp.fit_ramps(
                        d_sub,
                        C_no,
                        sig_sub,
                        diffs2use=m_sub,
                        countrateguess=None,
                        rescale=True
                    )

                seed[
                    idx_seed_fit
                ] = init_res.countrate

            except Exception:

                idx_seed_ols |= (
                    idx_seed_fit
                )

        # ------------------------------------------------------------
        # OLS seed
        # ------------------------------------------------------------

        if np.any(idx_seed_ols):

            ols_row, _ = (
                _ols_row_and_uncert(
                    input_read[:, i, :],
                    base_valid[:, i, :],
                    t,
                    sig_row
                )
            )

            seed[
                idx_seed_ols
            ] = ols_row[
                idx_seed_ols
            ]

        # ------------------------------------------------------------
        # Fill failed fitramp seeds with OLS
        # ------------------------------------------------------------

        bad_seed = (
            ~np.isfinite(seed)
        )

        if np.any(bad_seed):

            ols_row, _ = (
                _ols_row_and_uncert(
                    input_read[:, i, :],
                    base_valid[:, i, :],
                    t,
                    sig_row
                )
            )

            seed[
                bad_seed
            ] = ols_row[
                bad_seed
            ]

        # ------------------------------------------------------------
        # 2) Jump detection
        # ------------------------------------------------------------

        m_row2 = m_row.copy()

        idx_jump = (
            usable0 >= 1
        )

        if np.any(idx_jump):

            try:

                d_sub = d_row[
                    :,
                    idx_jump
                ]

                m_sub = m_row[
                    :,
                    idx_jump
                ]

                sig_sub = sig_row[
                    idx_jump
                ]

                with warnings.catch_warnings():

                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning
                    )

                    m2_sub, _ = (
                        fitramp.mask_jumps(
                            d_sub,
                            C_no,
                            sig_sub,
                            threshold_oneomit=(
                                JUMP_THRESH_ONEOMIT
                            ),
                            threshold_twoomit=(
                                JUMP_THRESH_TWOOMIT
                            ),
                            diffs2use=m_sub
                        )
                    )

                m_row2[
                    :,
                    idx_jump
                ] = m2_sub

            except Exception:
                pass

        usable_final = (
            m_row2.sum(
                axis=0
            )
        )

        # ------------------------------------------------------------
        # 3) Final fitramp fit with pedestal
        # ------------------------------------------------------------

        idx_final_fit = (
            (usable_final >= 1)
            &
            (
                np.isfinite(resetsig)
                |
                (usable_final >= 2)
            )
        )

        if np.any(idx_final_fit):

            try:

                d_sub = d_row[
                    :,
                    idx_final_fit
                ]

                m_sub = m_row2[
                    :,
                    idx_final_fit
                ]

                sig_sub = sig_row[
                    idx_final_fit
                ]

                seed_sub = seed[
                    idx_final_fit
                ]

                r0_sub = resetval[
                    idx_final_fit
                ]

                s0_sub = resetsig[
                    idx_final_fit
                ]

                with warnings.catch_warnings():

                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning
                    )

                    final_res = fitramp.fit_ramps(
                        d_sub,
                        C_ped,
                        sig_sub,
                        diffs2use=m_sub,
                        countrateguess=seed_sub,
                        resetval=r0_sub,
                        resetsig=s0_sub,
                        rescale=True
                    )

                row_slope[
                    idx_final_fit
                ] = final_res.countrate

                row_ped[
                    idx_final_fit
                ] = final_res.pedestal

                row_uncert[
                    idx_final_fit
                ] = final_res.uncert

                row_chisq[
                    idx_final_fit
                ] = final_res.chisq

            except Exception:
                pass

        # ------------------------------------------------------------
        # 4) OLS fallback
        # ------------------------------------------------------------

        need_fallback = (
            (~np.isfinite(row_slope))
            |
            (usable_final == 0)
            |
            (~idx_final_fit)
        )

        if np.any(need_fallback):

            ols_row, ols_unc = (
                _ols_row_and_uncert(
                    input_read[:, i, :],
                    base_valid[:, i, :],
                    t,
                    sig_row
                )
            )

            ols_chisq = (
                _ols_chisq_row(
                    input_read[:, i, :],
                    base_valid[:, i, :],
                    t,
                    ols_row,
                    sig_row
                )
            )

            row_slope[
                need_fallback
            ] = ols_row[
                need_fallback
            ]

            row_uncert[
                need_fallback
            ] = ols_unc[
                need_fallback
            ]

            row_ped[
                need_fallback
            ] = resetval[
                need_fallback
            ]

            row_chisq[
                need_fallback
            ] = ols_chisq[
                need_fallback
            ]

        # ------------------------------------------------------------
        # 5) Final sanitation
        # ------------------------------------------------------------

        bad_final = (
            ~np.isfinite(
                row_slope
            )
        )

        if np.any(bad_final):

            ols_row, ols_unc = (
                _ols_row_and_uncert(
                    input_read[:, i, :],
                    base_valid[:, i, :],
                    t,
                    sig_row
                )
            )

            ols_chisq = (
                _ols_chisq_row(
                    input_read[:, i, :],
                    base_valid[:, i, :],
                    t,
                    ols_row,
                    sig_row
                )
            )

            row_slope[
                bad_final
            ] = ols_row[
                bad_final
            ]

            row_uncert[
                bad_final
            ] = ols_unc[
                bad_final
            ]

            row_ped[
                bad_final
            ] = resetval[
                bad_final
            ]

            row_chisq[
                bad_final
            ] = ols_chisq[
                bad_final
            ]

        # ------------------------------------------------------------
        # Save row
        # ------------------------------------------------------------

        slope[i, :] = (
            row_slope
        )

        ped[i, :] = (
            row_ped
        )

        uncert[i, :] = (
            row_uncert
        )

        chisq[i, :] = (
            row_chisq
        )

    # ================================================================
    # Finished
    # ================================================================

    t2 = time.time()

    print(
        f"Ramp fitting completed in "
        f"{t2 - t1:.3f} seconds."
    )

    if return_pedestal:

        return (
            slope,
            ped,
            uncert,
            chisq
        )

    return (
        slope,
        uncert,
        chisq
    )
##################################################################################

import time
import warnings
import numpy as np

FIT_DQ = {
    'NO_USABLE_PAIR': np.uint16(1 << 0),
    'OLS_USED': np.uint16(1 << 1),
    'FIT_FAILED': np.uint16(1 << 2),
    'INVALID_NOISE': np.uint16(1 << 3),
    'JUMP_REJECTION': np.uint16(1 << 4),
    'PLACEHOLDER': np.uint16(1 << 5),
    'JUMP_CHECK_FAILED': np.uint16(1 << 6),
}
# Pixels with these bits must be excluded from science extraction.
FIT_REJECT_BITS = int(FIT_DQ['NO_USABLE_PAIR'] | FIT_DQ['FIT_FAILED']
                      | FIT_DQ['INVALID_NOISE'] | FIT_DQ['PLACEHOLDER'])


def _ols_row_and_uncert(reads, valid, t, sig):
    """OLS on accepted reads; known independent per-read noise sets uncertainty.

    Returns NaN internally where fewer than two reads/time leverage remain.
    The public fitter converts those failures to flagged finite placeholders.
    """
    mask = np.asarray(valid, bool) & np.isfinite(reads)
    count = mask.sum(axis=0)
    safe_y = np.where(mask, reads, 0.)
    den = np.maximum(count, 1)
    tmean = np.sum(np.where(mask, t[:, None], 0.), axis=0) / den
    ymean = safe_y.sum(axis=0) / den
    tc = t[:, None] - tmean
    yc = np.where(mask, reads - ymean, 0.)
    leverage = np.sum(np.where(mask, tc * tc, 0.), axis=0)
    good = (count >= 2) & (leverage > 0) & np.isfinite(sig) & (sig > 0)
    slope = np.full(reads.shape[1], np.nan)
    uncertainty = slope.copy()
    slope[good] = np.sum(np.where(mask, tc * yc, 0.), axis=0)[good] / leverage[good]
    uncertainty[good] = sig[good] / np.sqrt(leverage[good])
    return slope, uncertainty


def ramp_fit2(
    input_read,
    total_exptime,
    SIG_map_scaled,
    *,
    return_pedestal=True,
    reset_prior_strength=3.0,
    use_sigma_clip=False,
    sigma_clip=3.0,
    max_iter=3,
    min_reads=5,
    tile=(128, 128),
    JUMP_THRESH_ONEOMIT=20.25,
    JUMP_THRESH_TWOOMIT=23.8,
    group_dq=None,
    return_dq=True,
    invalid_uncertainty=1.0e30,
):
    """Fit ramps without restoring saturation- or jump-rejected samples.

    input_read: (reads, rows, columns); group_dq: same shape, True=use.
    Timing retains your original convention dt=total_exptime/N.
    Verify that convention against detector read timestamps.

    All returned arrays are finite. Failed pixels have slope=0, pedestal=0,
    uncertainty=invalid_uncertainty, chisq=-1; these are placeholders, not
    measurements. FIT_REJECT_BITS must be excluded from downstream extraction.
    A successful two-read OLS fit remains allowed when a usable pair exists.
    Negative/zero finite differences are allowed to avoid positive-noise bias.
    OLS does not run if jump detection has removed any additional differences.

    With return_dq=True (default), append fit_dq to the original return tuple:
    slope, pedestal, uncertainty, chisq, fit_dq (return_pedestal=True), or
    slope, uncertainty, chisq, fit_dq otherwise. False retains the old tuple
    shape but omits essential information about failed/placeholder pixels.
    fit_dq is separate from the saturation quality_map; their bit meanings
    differ and must be translated before combining into the pipeline's DQ.
    """
    input_read = np.asarray(input_read, dtype=float)
    SIG_map_scaled = np.asarray(SIG_map_scaled, dtype=float)
    if input_read.ndim != 3 or input_read.shape[0] < 2:
        raise ValueError('input_read requires at least two reads and three dimensions.')
    if SIG_map_scaled.shape != input_read.shape[1:]:
        raise ValueError('SIG_map_scaled must match the detector shape.')
    if not np.isfinite(total_exptime) or total_exptime <= 0:
        raise ValueError('total_exptime must be finite and positive.')
    if not np.isfinite(invalid_uncertainty) or invalid_uncertainty <= 0:
        raise ValueError('invalid_uncertainty must be finite and positive.')
    if not np.isfinite(reset_prior_strength) or reset_prior_strength <= 0:
        raise ValueError('reset_prior_strength must be finite and positive.')
    if group_dq is not None:
        group_dq = np.asarray(group_dq, dtype=bool)
        if group_dq.shape != input_read.shape:
            raise ValueError('group_dq must match input_read shape.')

    N, H, W = input_read.shape

    dt = float(total_exptime) / N

    t = (
        np.arange(N, dtype=float) + 0.5
    ) * dt

    t1 = time.time()

    # ================================================================
    # Helper: sigma clipping
    # ================================================================

    def _sigma_clip_reads(cube):

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs):
                return iterable

        keep = np.ones_like(
            cube,
            dtype=bool
        )

        Ty, Tx = tile

        for y0 in tqdm(
            range(0, H, Ty),
            desc="σ-clip tiles"
        ):

            y1 = min(
                H,
                y0 + Ty
            )

            for x0 in range(0, W, Tx):

                x1 = min(
                    W,
                    x0 + Tx
                )

                sub = cube[
                    :,
                    y0:y1,
                    x0:x1
                ]

                n, ty, tx = sub.shape

                k = ty * tx

                Y = sub.reshape(
                    n,
                    k
                )

                mask = np.isfinite(Y)

                time_idx = np.arange(
                    n,
                    dtype=np.float32
                )

                for _ in range(max_iter):

                    cnt = mask.sum(axis=0)

                    if not np.any(
                        cnt >= min_reads
                    ):
                        break

                    S0 = cnt

                    St = (
                        time_idx @ mask
                    )

                    Stt = (
                        (time_idx ** 2) @ mask
                    )

                    Wy = np.where(mask, Y, 0.0)

                    Sy = Wy.sum(axis=0)

                    Sty = (
                        time_idx @ Wy
                    )

                    Var_t = (
                        Stt
                        - (St * St)
                        / np.maximum(S0, 1)
                    )

                    Cov_ty = (
                        Sty
                        - (St * Sy)
                        / np.maximum(S0, 1)
                    )

                    b = np.zeros(
                        k,
                        dtype=np.float32
                    )

                    ok = Var_t > 0

                    b[ok] = (
                        Cov_ty[ok]
                        / Var_t[ok]
                    )

                    a = (
                        Sy - b * St
                    ) / np.maximum(S0, 1)

                    Yhat = (
                        a
                        + np.outer(
                            time_idx,
                            b
                        )
                    )

                    resid = Y - Yhat

                    resid[~mask] = np.nan

                    s = np.nanstd(
                        resid,
                        axis=0
                    )

                    candidate = mask & (np.abs(resid) <= sigma_clip * s)
                    # Preserve zero-scatter ramps and avoid clipping below min_reads.
                    update = (cnt >= min_reads) & np.isfinite(s) & (s > 0)
                    update &= candidate.sum(axis=0) >= min_reads
                    new_mask = np.where(update[None, :], candidate, mask)

                    if np.array_equal(
                        new_mask,
                        mask
                    ):
                        break

                    mask = new_mask

                keep[
                    :,
                    y0:y1,
                    x0:x1
                ] = mask.reshape(
                    n,
                    ty,
                    tx
                )

        return keep

    # ================================================================
    # Helper: OLS chi-square
    # ================================================================
    
    def _ols_chisq_row(
        reads_row,
        valid_row,
        t,
        slope_row,
        sig_row,
    ):
        """
        Vectorized OLS chi-square for one detector row.

        Parameters
        ----------
        reads_row : ndarray, shape (N, W)
        valid_row : ndarray, bool, shape (N, W)
        t : ndarray, shape (N,)
        slope_row : ndarray, shape (W,)
        sig_row : ndarray, shape (W,)

        Returns
        -------
        chisq_row : ndarray, shape (W,)
        """

        # Valid points
        mask = (
            valid_row
            & np.isfinite(reads_row)
        )

        nvalid = mask.sum(axis=0)

        # --------------------------------------------------------------
        # OLS intercept corresponding to supplied slope
        #
        # intercept = mean(y - slope*t)
        # over valid reads
        # --------------------------------------------------------------

        model_without_intercept = (
            t[:, None] * slope_row[None, :]
        )

        residual0 = (
            reads_row
            - model_without_intercept
        )

        # Ignore invalid reads
        residual0_masked = np.where(
            mask,
            residual0,
            0.0
        )

        intercept = np.full(
            reads_row.shape[1],
            np.nan,
            dtype=float
        )

        good_count = nvalid > 0

        intercept[good_count] = (
            residual0_masked[:, good_count].sum(axis=0)
            / nvalid[good_count]
        )

        # --------------------------------------------------------------
        # Full model and residual
        # --------------------------------------------------------------

        model = (
            intercept[None, :]
            + model_without_intercept
        )

        residual = (
            reads_row - model
        )

        # --------------------------------------------------------------
        # Chi-square
        # --------------------------------------------------------------

        valid_sigma = (
            np.isfinite(sig_row)
            & (sig_row > 0)
        )

        good = (
            (nvalid >= 2)
            & np.isfinite(slope_row)
            & np.isfinite(intercept)
            & valid_sigma
        )

        chisq_row = np.full(
            reads_row.shape[1],
            np.nan,
            dtype=float
        )

        normalized_resid2 = (
            residual
            / sig_row[None, :]
        ) ** 2

        normalized_resid2 = np.where(
            mask,
            normalized_resid2,
            0.0
        )

        chisq_all = np.sum(
            normalized_resid2,
            axis=0
        )

        chisq_row[good] = (
            chisq_all[good]
        )

        return chisq_row

    # ================================================================
    # Initial read-validity mask
    # ================================================================

    base_valid_nomask = np.isfinite(input_read)

    if use_sigma_clip:
        clip_cube = input_read if group_dq is None else np.where(group_dq, input_read, np.nan)
        base_valid_nomask &= _sigma_clip_reads(clip_cube)

    # ================================================================
    # Read differences
    # ================================================================

    diffs = (input_read[1:] - input_read[:-1])

    # ================================================================
    # Dynamic saturation / group DQ mask
    # ================================================================
    base_valid = base_valid_nomask.copy()


    if group_dq is not None:
        group_dq = np.asarray(group_dq,dtype=bool)
        if (group_dq.shape != input_read.shape):
            raise ValueError(
                f"group_dq shape "
                f"{group_dq.shape} "
                f"!= input_read shape "
                f"{input_read.shape}")

        # Apply the dynamic mask
        base_valid &= group_dq

    pair_mask = (
        base_valid[1:]
        & base_valid[:-1]
        & np.isfinite(diffs)
        )

    no_usable_pair = ~np.any(pair_mask, axis=0)
    n_no_usable_pair = np.count_nonzero(no_usable_pair)
    if n_no_usable_pair > 0:
        print(
            f"{n_no_usable_pair:,} pixels have no usable "
            "read differences after masking."
        )

    fit_dq = np.zeros((H, W), dtype=np.uint16)
    fit_dq[no_usable_pair] |= FIT_DQ['NO_USABLE_PAIR']
    noise_ok = np.isfinite(SIG_map_scaled) & (SIG_map_scaled > 0)
    fit_dq[~noise_ok] |= FIT_DQ['INVALID_NOISE']

    # ================================================================
    # Reset prior
    # ================================================================

    first_read = input_read[0]

    first_ok = (
        base_valid[0]
        & np.isfinite(first_read)
    )

    resetval_map = np.where(
        first_ok,
        first_read,
        0.0
    )

    resetsig_map = np.where(
        first_ok,
        reset_prior_strength
        * SIG_map_scaled,
        np.inf
    )

    # ================================================================
    # fitramp covariance objects
    # ================================================================

    C_no = fitramp.Covar(
        t,
        pedestal=False
    )

    C_ped = fitramp.Covar(
        t,
        pedestal=True
    )

    # ================================================================
    # Output arrays
    # ================================================================

    slope = np.full(
        (H, W),
        np.nan,
        dtype=float
    )

    ped = np.full(
        (H, W),
        np.nan,
        dtype=float
    )

    uncert = np.full(
        (H, W),
        np.nan,
        dtype=float
    )

    chisq = np.full(
        (H, W),
        np.nan,
        dtype=float
    )

    # ================================================================
    # Ramp fit row-by-row
    # ================================================================

    for i in range(H):

        if i % 128 == 0:
            print(
                f"Fitting row {i}/{H}..."
            )

        sig_row = (
            SIG_map_scaled[i, :]
        )

        # Supply neutral zeros for excluded differences; NaN*False is NaN.
        d_row = np.where(pair_mask[:, i, :], diffs[:, i, :], 0.)

        m_row = (
            pair_mask[:, i, :]
        )

        resetval = (
            resetval_map[i, :]
        )

        resetsig = (
            resetsig_map[i, :]
        )

        Wrow = d_row.shape[1]

        row_slope = np.full(
            Wrow,
            np.nan,
            dtype=float
        )

        row_ped = np.full(
            Wrow,
            np.nan,
            dtype=float
        )

        row_uncert = np.full(
            Wrow,
            np.nan,
            dtype=float
        )

        row_chisq = np.full(
            Wrow,
            np.nan,
            dtype=float
        )

        # ------------------------------------------------------------
        # 0) Number of usable differences before jump detection
        # ------------------------------------------------------------

        usable0 = m_row.sum(
            axis=0
        )

        # ------------------------------------------------------------
        # 1) Initial slope seed
        # ------------------------------------------------------------

        seed = np.full(
            Wrow,
            np.nan,
            dtype=float
        )

        idx_seed_fit = (usable0 >= 2) & noise_ok[i]

        idx_seed_ols = (
            ~idx_seed_fit
        )

        # ------------------------------------------------------------
        # fitramp seed
        # ------------------------------------------------------------

        if np.any(idx_seed_fit):

            try:

                d_sub = d_row[
                    :,
                    idx_seed_fit
                ]

                m_sub = m_row[
                    :,
                    idx_seed_fit
                ]

                sig_sub = sig_row[
                    idx_seed_fit
                ]

                with warnings.catch_warnings():

                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning
                    )

                    init_res = fitramp.fit_ramps(
                        d_sub,
                        C_no,
                        sig_sub,
                        diffs2use=m_sub,
                        countrateguess=None,
                        rescale=True
                    )

                seed[
                    idx_seed_fit
                ] = init_res.countrate

            except Exception:

                idx_seed_ols |= (
                    idx_seed_fit
                )

        # ------------------------------------------------------------
        # OLS seed
        # ------------------------------------------------------------

        if np.any(idx_seed_ols):

            ols_row, _ = (
                _ols_row_and_uncert(
                    input_read[:, i, :],
                    base_valid[:, i, :],
                    t,
                    sig_row
                )
            )

            seed[
                idx_seed_ols
            ] = ols_row[
                idx_seed_ols
            ]

        # ------------------------------------------------------------
        # Fill failed fitramp seeds with OLS
        # ------------------------------------------------------------

        bad_seed = (
            ~np.isfinite(seed)
        )

        if np.any(bad_seed):

            ols_row, _ = (
                _ols_row_and_uncert(
                    input_read[:, i, :],
                    base_valid[:, i, :],
                    t,
                    sig_row
                )
            )

            seed[
                bad_seed
            ] = ols_row[
                bad_seed
            ]

        # ------------------------------------------------------------
        # 2) Jump detection
        # ------------------------------------------------------------

        m_row2 = m_row.copy()

        idx_jump = (usable0 >= 2) & noise_ok[i]

        if np.any(idx_jump):

            try:

                d_sub = d_row[
                    :,
                    idx_jump
                ]

                m_sub = m_row[
                    :,
                    idx_jump
                ]

                sig_sub = sig_row[
                    idx_jump
                ]

                with warnings.catch_warnings():

                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning
                    )

                    m2_sub, _ = (
                        fitramp.mask_jumps(
                            d_sub,
                            C_no,
                            sig_sub,
                            threshold_oneomit=(
                                JUMP_THRESH_ONEOMIT
                            ),
                            threshold_twoomit=(
                                JUMP_THRESH_TWOOMIT
                            ),
                            diffs2use=m_sub
                        )
                    )

                m_row2[
                    :,
                    idx_jump
                ] = np.asarray(m2_sub, dtype=bool) & m_sub

            except Exception:
                fit_dq[i, idx_jump] |= FIT_DQ['JUMP_CHECK_FAILED']

        # No later step may restore an excluded read difference.
        m_row2 &= m_row
        jump_removed = np.any(m_row & ~m_row2, axis=0)
        fit_dq[i, jump_removed] |= FIT_DQ['JUMP_REJECTION']
        usable_final = (
            m_row2.sum(
                axis=0
            )
        )

        # ------------------------------------------------------------
        # 3) Final fitramp fit with pedestal
        # ------------------------------------------------------------

        idx_final_fit = (
            (usable_final >= 1)
            & noise_ok[i]
            & np.isfinite(seed)
            &
            (
                np.isfinite(resetsig)
                |
                (usable_final >= 2)
            )
        )

        if np.any(idx_final_fit):

            try:

                d_sub = d_row[
                    :,
                    idx_final_fit
                ]

                m_sub = m_row2[
                    :,
                    idx_final_fit
                ]

                sig_sub = sig_row[
                    idx_final_fit
                ]

                seed_sub = seed[
                    idx_final_fit
                ]

                r0_sub = resetval[
                    idx_final_fit
                ]

                s0_sub = resetsig[
                    idx_final_fit
                ]

                with warnings.catch_warnings():

                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning
                    )

                    final_res = fitramp.fit_ramps(
                        d_sub,
                        C_ped,
                        sig_sub,
                        diffs2use=m_sub,
                        countrateguess=seed_sub,
                        resetval=r0_sub,
                        resetsig=s0_sub,
                        rescale=True
                    )

                row_slope[
                    idx_final_fit
                ] = final_res.countrate

                row_ped[
                    idx_final_fit
                ] = final_res.pedestal

                row_uncert[
                    idx_final_fit
                ] = final_res.uncert

                row_chisq[
                    idx_final_fit
                ] = final_res.chisq

            except Exception:
                pass

        # ------------------------------------------------------------
        # 4) OLS fallback
        # ------------------------------------------------------------

        final_ok = (np.isfinite(row_slope) & np.isfinite(row_ped)
                    & np.isfinite(row_uncert) & (row_uncert > 0)
                    & np.isfinite(row_chisq) & (row_chisq >= 0))
        need_fallback = (~final_ok & (usable_final >= 1)
                         & noise_ok[i] & ~jump_removed)

        if np.any(need_fallback):

            ols_row, ols_unc = (
                _ols_row_and_uncert(
                    input_read[:, i, :],
                    base_valid[:, i, :],
                    t,
                    sig_row
                )
            )

            ols_chisq = (
                _ols_chisq_row(
                    input_read[:, i, :],
                    base_valid[:, i, :],
                    t,
                    ols_row,
                    sig_row
                )
            )

            row_slope[
                need_fallback
            ] = ols_row[
                need_fallback
            ]

            row_uncert[
                need_fallback
            ] = ols_unc[
                need_fallback
            ]

            ols_mask = base_valid[:, i, :]
            ols_intercept = np.sum(np.where(
                ols_mask, input_read[:, i, :] - t[:, None] * ols_row[None, :], 0.),
                axis=0) / np.maximum(ols_mask.sum(axis=0), 1)
            row_ped[need_fallback] = ols_intercept[need_fallback]
            fit_dq[i, need_fallback] |= FIT_DQ['OLS_USED']

            row_chisq[
                need_fallback
            ] = ols_chisq[
                need_fallback
            ]

        # ------------------------------------------------------------
        # 5) Final sanitation
        # ------------------------------------------------------------

        bad_final = (
            ~np.isfinite(row_slope) | ~np.isfinite(row_ped)
            | ~np.isfinite(row_uncert) | (row_uncert <= 0)
            | ~np.isfinite(row_chisq) | (row_chisq < 0)
            | (usable_final == 0) | ~noise_ok[i]
        )
        fit_dq[i, bad_final] |= FIT_DQ['FIT_FAILED'] | FIT_DQ['PLACEHOLDER']
        row_slope[bad_final] = 0.0
        row_ped[bad_final] = 0.0
        row_uncert[bad_final] = invalid_uncertainty
        row_chisq[bad_final] = -1.0

        # ------------------------------------------------------------
        # Save row
        # ------------------------------------------------------------

        slope[i, :] = (
            row_slope
        )

        ped[i, :] = (
            row_ped
        )

        uncert[i, :] = (
            row_uncert
        )

        chisq[i, :] = (
            row_chisq
        )

    # ================================================================
    # Finished
    # ================================================================

    t2 = time.time()

    print(
        f"Ramp fitting completed in "
        f"{t2 - t1:.3f} seconds."
    )

    values = (slope, ped, uncert, chisq) if return_pedestal else (slope, uncert, chisq)
    return values

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

def select_ifsmode(modslnam, dsprsnam, filtername=None):
    """
    Determine SCALES IFSMODE from MODSLNAM and DSPRSNAM.
    FILTER is used only for pol-prism configurations.

    Returns
    -------
    str or None
        IFS mode, or None for unknown/unsupported configurations.
    """

    ifsmode_map = {
        # Medium-resolution modes
        ("MedRes", "K-grating"): "MedRes-K",
        ("MedRes", "L-grating"): "MedRes-L",
        ("MedRes", "M-grating"): "MedRes-M",

        # Low-resolution modes
        ("LowRes", "K-prism"):   "LowRes-K",
        ("LowRes", "KL-prism"):  "LowRes-KL",
        ("LowRes", "KLM-prism"): "LowRes-KLM",
        ("LowRes", "Ls-prism"):  "LowRes-Ls",
        ("LowRes", "L-prism"):   "LowRes-L",
        ("LowRes", "M-prism"):   "LowRes-M",
    }

    modslnam = str(modslnam).strip()
    dsprsnam = str(dsprsnam).strip()

    # Check only MODSLNAM and DSPRSNAM
    if modslnam.upper() == "UNKNOWN" or dsprsnam.upper() == "UNKNOWN":
        return None

    # FILTER is relevant only for polarimetric mode
    if dsprsnam == "pol-prism":
        if modslnam != "LowRes":
            return None

        filtername = str(filtername).strip()

        if filtername == "KL":
            return "LowRes-KL"
        elif filtername == "M":
            return "LowRes-M"

        return None

    # All other modes ignore FILTER
    return ifsmode_map.get((modslnam, dsprsnam), None)

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
        "LowRes-L":   {"start": 2.91157407, "end": 4.13842593},
        "LowRes-M":   {"start": 4.5,  "end": 5.2},
        "LowRes-SED":   {"start": 2.0,  "end": 5.0},
        "LowRes-KL": {"start": 2.0,  "end": 3.7},
        "LowRes-PAH": {"start": 3.1,  "end": 3.5},
        "MedRes-K":   {"start": 2.02,  "end": 2.38},
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
    Group L1 calibration products based on detector/instrument
    configuration.

    Grouping rules
    --------------
    Imager:
        CAMERA, IMGFW2N, IMTYPE, EXPTIME, MCLOCK

    IFS:
        CAMERA, DSPRSNAM, MODSLNAM, IMTYPE, EXPTIME, MCLOCK

    CALUNIT:
        MONOWAVE is additionally included in the grouping.

    Parameters
    ----------
    dt : pandas.DataFrame
        Processing-table subset containing L1 products.

    Returns
    -------
    file_groups : list of dict
        Each group has the form:

        {
            'params': {
                grouping keyword: value,
                ...
            },
            'filenames': [...]
        }
    """

    # ---------------------------------------------------------
    # Columns required for every file
    # ---------------------------------------------------------

    required_cols = [
        'filename',
        'CAMERA',
        'IMTYPE',
        'EXPTIME',
        'MCLOCK'
    ]

    missing = [
        col for col in required_cols
        if col not in dt.columns
    ]

    if missing:
        print(
            f"Missing required columns: {missing}"
        )
        return []

    # ---------------------------------------------------------
    # CALUNIT requires MONOWAVE
    # ---------------------------------------------------------

    imtype_upper = (
        dt['IMTYPE']
        .astype(str)
        .str.strip()
        .str.upper()
    )

    has_calunit = imtype_upper.eq('CALUNIT').any()

    if has_calunit and 'MONOWAVE' not in dt.columns:
        print(
            "Missing required column: ['MONOWAVE'] "
            "(needed for CALUNIT grouping)"
        )
        return []

    file_groups = []

    # ---------------------------------------------------------
    # Split by camera
    # ---------------------------------------------------------

    for cam, cam_df in dt.groupby(
        'CAMERA',
        dropna=False
    ):

        cam_name = str(cam).strip()
        cam_upper = cam_name.upper()

        # -----------------------------------------------------
        # Imager
        # -----------------------------------------------------

        if cam_upper == 'IM':

            camera_required = ['IMGFW2N']

            missing = [
                col for col in camera_required
                if col not in cam_df.columns
            ]

            if missing:
                print(
                    f"Missing Imager grouping columns: {missing}"
                )
                continue

            base_keys = [
                'CAMERA',
                'IMGFW2N',
                'IMTYPE',
                'EXPTIME',
                'MCLOCK'
            ]

            print(
                f"Grouping Imager data by {base_keys}"
            )

        # -----------------------------------------------------
        # IFS
        # -----------------------------------------------------

        elif cam_upper == 'IFS':

            camera_required = [
                'DSPRSNAM',
                'MODSLNAM'
            ]

            missing = [
                col for col in camera_required
                if col not in cam_df.columns
            ]

            if missing:
                print(
                    f"Missing IFS grouping columns: {missing}"
                )
                continue

            base_keys = [
                'CAMERA',
                'DSPRSNAM',
                'MODSLNAM',
                'IMTYPE',
                'EXPTIME',
                'MCLOCK'
            ]

            print(
                f"Grouping IFS data by {base_keys}"
            )

        else:

            print(
                f"Unknown CAMERA '{cam_name}', skipping."
            )
            continue

        # -----------------------------------------------------
        # Separate CALUNIT from other calibration types
        # -----------------------------------------------------

        imtype_upper = (
            cam_df['IMTYPE']
            .astype(str)
            .str.strip()
            .str.upper()
        )

        df_cal = cam_df[
            imtype_upper.eq('CALUNIT')
        ]

        df_non = cam_df[
            ~imtype_upper.eq('CALUNIT')
        ]

        # -----------------------------------------------------
        # DARK / FLATLAMP / BIAS etc.
        # -----------------------------------------------------

        if len(df_non) > 0:

            for group_params, sub_df in df_non.groupby(
                base_keys,
                dropna=False
            ):

                filenames = (
                    sub_df['filename']
                    .astype(str)
                    .str.strip()
                    .tolist()
                )

                file_groups.append({
                    'params': {
                        key.lower(): value
                        for key, value in zip(
                            base_keys,
                            group_params
                        )
                    },
                    'filenames': filenames
                })

        # -----------------------------------------------------
        # CALUNIT
        # -----------------------------------------------------

        if len(df_cal) > 0:

            cal_keys = base_keys + ['MONOWAVE']

            print(
                f"Grouping CALUNIT data by {cal_keys}"
            )

            for group_params, sub_df in df_cal.groupby(
                cal_keys,
                dropna=False
            ):

                filenames = (
                    sub_df['filename']
                    .astype(str)
                    .str.strip()
                    .tolist()
                )

                file_groups.append({
                    'params': {
                        key.lower(): value
                        for key, value in zip(
                            cal_keys,
                            group_params
                        )
                    },
                    'filenames': filenames
                })

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    if not file_groups:

        print(
            "No file groups were created."
        )

    else:

        print(
            f"Created {len(file_groups)} groups "
            "from data table."
        )

    return file_groups

def resolve_mean_func(name):
    if callable(name):
        return name

    funcs = {
        "mean": np.mean,
        "np.mean": np.mean,

        "nanmean": np.nanmean,
        "np.nanmean": np.nanmean,

        "median": np.median,
        "np.median": np.median,

        "nanmedian": np.nanmedian,
        "np.nanmedian": np.nanmedian,

        "robust_mean": robust.mean,
        "robust.mean": robust.mean,

        "robust_median": robust.median,
        "robust.median": robust.median,
    }

    try:
        return funcs[name]
    except KeyError:
        raise ValueError(
            f"Unknown mean function: {name}. "
            f"Valid choices are: {list(funcs.keys())}"
        )


def get_proc_entries(proctab, suffix, imtype=None):
    """
    Select entries from the processing table by suffix
    and optionally by IMTYPE.
    """

    required_cols = ["SUFF", "IMTYPE", "filename"]

    missing = [
        col for col in required_cols
        if col not in proctab.columns
    ]

    if missing:
        raise ValueError(
            f"Processing table is missing required columns: {missing}"
        )

    # Normalize requested suffix
    suffix = str(suffix).strip().upper()

    # Normalize proc-table suffix
    suff = (
        proctab["SUFF"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    print("Requested suffix:", repr(suffix))
    print("Available SUFF values:", suff.unique())

    mask = suff == suffix

    if imtype is not None:

        imtype = str(imtype).strip().upper()

        imtypes = (
            proctab["IMTYPE"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        mask &= imtypes == imtype

    result = proctab.loc[mask].copy()

    print(
        f"Found {len(result)} processing-table entries "
        f"for suffix={suffix}"
        + (
            f", IMTYPE={imtype}"
            if imtype is not None
            else ""
        )
    )

    return result


def validate_linearity_inputs(
    proctab,
    output_dir,
    min_files=5,
    min_reads=30,
    suffix="_L10",
):
    """
    Validate input files for creating detector linearity coefficients.

    Requirements
    ------------
    - IMTYPE must be FLATLAMP.
    - Processing suffix must be L10.
    - FITS data must be a 3-D ramp cube.
    - Each cube must contain at least `min_reads` reads.
    - Files must have matching:
          CAMERA
          MCLOCK
          EXPTIME
    - At least `min_files` valid cubes must exist within a matching group.

    Parameters
    ----------
    proctab : pandas.DataFrame
        SCALES processing table.

    output_dir : str
        Directory containing the processed FITS files.

    min_files : int, optional
        Minimum number of matching ramp cubes required.
        Default is 5.

    min_reads : int, optional
        Minimum number of reads required in each ramp cube.
        Default is 30.

    suffix : str, optional
        Processing suffix for linearity inputs.
        Default is '_L10'.

    Returns
    -------
    valid_groups : list of dict
        Each dictionary contains:

            {
                'params': {
                    'camera': ...,
                    'mclock': ...,
                    'exptime': ...
                },
                'filenames': [...],
                'paths': [...]
            }

        Only groups satisfying all requirements are returned.

    Raises
    ------
    RuntimeError
        If no valid group satisfies the requirements.
    """

    dt = get_proc_entries(
        proctab,
        suffix=suffix,
        imtype="FLATLAMP",
    )

    if len(dt) == 0:
        raise RuntimeError(
            f"Cannot create linearity coefficients: "
            f"no FLATLAMP {suffix} entries were found."
        )

    required_cols = [
        "CAMERA",
        "MCLOCK",
        "EXPTIME",
    ]

    missing = [col for col in required_cols if col not in dt.columns]

    if missing:
        raise ValueError(
            f"Processing table is missing columns required "
            f"for linearity grouping: {missing}"
        )

    valid_rows = []

    for idx, row in dt.iterrows():

        filename = str(row["filename"]).strip()
        path = os.path.join(output_dir, filename)

        if not os.path.isfile(path):
            print(
                f"WARNING: skipping linearity input; "
                f"file does not exist: {path}"
            )
            continue

        try:
            with fits.open(path, memmap=True) as hdul:

                data = hdul[0].data

                if data is None:
                    print(
                        f"WARNING: skipping {filename}; "
                        "FITS data is empty."
                    )
                    continue

                if data.ndim != 3:
                    print(
                        f"WARNING: skipping {filename}; "
                        f"expected 3-D cube, got shape={data.shape}."
                    )
                    continue

                nreads = data.shape[0]

                if nreads < min_reads:
                    print(
                        f"WARNING: skipping {filename}; "
                        f"{nreads} reads found, "
                        f"minimum required is {min_reads}."
                    )
                    continue

        except Exception as exc:
            print(
                f"WARNING: could not validate {filename}: {exc}"
            )
            continue

        valid_rows.append(idx)

    if not valid_rows:
        raise RuntimeError(
            "Cannot create linearity coefficients: "
            "no valid 3-D FLATLAMP ramp cubes were found."
        )

    valid_dt = dt.loc[valid_rows].copy()

    group_keys = [
        "CAMERA",
        "MCLOCK",
        "EXPTIME",
    ]

    valid_groups = []

    for group_params, sub_df in valid_dt.groupby(
        group_keys,
        dropna=False,
    ):

        nfiles = len(sub_df)

        if nfiles < min_files:
            print(
                f"WARNING: skipping linearity group "
                f"{group_params}: {nfiles} valid cubes found; "
                f"minimum required is {min_files}."
            )
            continue

        filenames = (
            sub_df["filename"]
            .astype(str)
            .str.strip()
            .tolist()
        )

        paths = [
            os.path.join(output_dir, filename)
            for filename in filenames
        ]

        valid_groups.append(
            {
                "params": {
                    key.lower(): value
                    for key, value in zip(
                        group_keys,
                        group_params,
                    )
                },
                "filenames": filenames,
                "paths": paths,
            }
        )

    if not valid_groups:
        raise RuntimeError(
            "Cannot create linearity coefficients. "
            "No CAMERA/MCLOCK/EXPTIME group contains "
            f"at least {min_files} valid FLATLAMP "
            f"{suffix} cubes with at least "
            f"{min_reads} reads each."
        )

    return valid_groups


def validate_saturation_inputs(
    proctab,
    output_dir,
    min_files=5,
    min_reads=30,
    suffix="_L11",
):
    """
    Validate and group input files for creating saturation maps.

    Requirements
    ------------
    - IMTYPE must be FLATLAMP.
    - Processing suffix must be L11.
    - FITS data must be a 3-D ramp cube.
    - Each cube must contain at least `min_reads` reads.
    - Valid cubes are grouped by exact:
          CAMERA, MCLOCK, EXPTIME
    - Each group must contain at least `min_files` valid cubes.

    Parameters
    ----------
    proctab : pandas.DataFrame
        SCALES processing table.

    output_dir : str
        Directory containing the processed FITS files.

    min_files : int, optional
        Minimum number of valid ramp cubes required per group.
        Default is 5.

    min_reads : int, optional
        Minimum number of reads required in each ramp cube.
        Default is 30.

    suffix : str, optional
        Processing suffix.
        Default is '_L11'.

    Returns
    -------
    valid_groups : list of dict
        Valid saturation-map groups:

            [
                {
                    "params": {
                        "camera": ...,
                        "mclock": ...,
                        "exptime": ...
                    },
                    "filenames": [...],
                    "paths": [...]
                },
                ...
            ]

    Raises
    ------
    RuntimeError
        If no valid saturation-map group exists.
    """

    # ---------------------------------------------------------
    # Find FLATLAMP L11 products
    # ---------------------------------------------------------

    dt = get_proc_entries(
        proctab,
        suffix=suffix,
        imtype="FLATLAMP",
    )

    if len(dt) == 0:
        raise RuntimeError(
            f"Cannot create saturation map: "
            f"no FLATLAMP {suffix} entries were found."
        )

    # ---------------------------------------------------------
    # Check required grouping columns
    # ---------------------------------------------------------

    group_keys = [
        "CAMERA",
        "MCLOCK",
        "EXPTIME",
    ]

    missing = [
        col for col in group_keys
        if col not in dt.columns
    ]

    if missing:
        raise RuntimeError(
            "Cannot create saturation map: "
            f"processing table is missing required "
            f"grouping columns: {missing}"
        )

    # ---------------------------------------------------------
    # Validate individual L11 ramp cubes
    # ---------------------------------------------------------

    valid_indices = []

    for idx, row in dt.iterrows():

        filename = str(
            row["filename"]
        ).strip()

        path = os.path.join(
            output_dir,
            filename
        )

        if not os.path.isfile(path):
            print(
                f"WARNING: skipping saturation input; "
                f"file does not exist: {path}"
            )
            continue

        try:

            with fits.open(
                path,
                memmap=True
            ) as hdul:

                data = hdul[0].data

                if data is None:
                    print(
                        f"WARNING: skipping {filename}; "
                        "FITS data is empty."
                    )
                    continue

                if data.ndim != 3:
                    print(
                        f"WARNING: skipping {filename}; "
                        f"expected 3-D cube, "
                        f"got shape={data.shape}."
                    )
                    continue

                nreads = data.shape[0]

                if nreads < min_reads:
                    print(
                        f"WARNING: skipping {filename}; "
                        f"{nreads} reads found, "
                        f"minimum required is "
                        f"{min_reads}."
                    )
                    continue

        except Exception as exc:

            print(
                f"WARNING: could not validate "
                f"{filename}: {exc}"
            )
            continue

        # File passed validation
        valid_indices.append(idx)

    # ---------------------------------------------------------
    # Make table containing only valid L11 cubes
    # ---------------------------------------------------------

    if not valid_indices:
        raise RuntimeError(
            "Cannot create saturation map. "
            f"No valid FLATLAMP {suffix} 3-D cubes "
            f"with at least {min_reads} reads were found."
        )

    valid_dt = dt.loc[
        valid_indices
    ].copy()

    # ---------------------------------------------------------
    # Group by CAMERA + MCLOCK + EXPTIME
    # ---------------------------------------------------------

    valid_groups = []

    for group_params, sub_df in valid_dt.groupby(
        group_keys,
        dropna=False
    ):

        camera, mclock, exptime = group_params

        nfiles = len(sub_df)

        # -----------------------------------------------------
        # Require enough ramps in this configuration
        # -----------------------------------------------------

        if nfiles < min_files:

            print(
                "WARNING: skipping saturation-map group "
                f"CAMERA={camera}, "
                f"MCLOCK={mclock}, "
                f"EXPTIME={exptime}; "
                f"found {nfiles} valid ramps, "
                f"minimum required is {min_files}."
            )

            continue

        filenames = (
            sub_df["filename"]
            .astype(str)
            .str.strip()
            .tolist()
        )

        paths = [
            os.path.join(
                output_dir,
                filename
            )
            for filename in filenames
        ]

        valid_groups.append(
            {
                "params": {
                    "camera": camera,
                    "mclock": mclock,
                    "exptime": exptime,
                },
                "filenames": filenames,
                "paths": paths,
            }
        )

    # ---------------------------------------------------------
    # Make sure at least one valid group exists
    # ---------------------------------------------------------

    if not valid_groups:
        raise RuntimeError(
            "Cannot create saturation map. "
            "No valid CAMERA/MCLOCK/EXPTIME group "
            f"contains at least {min_files} FLATLAMP "
            f"{suffix} ramp cubes with at least "
            f"{min_reads} reads each."
        )

    return valid_groups


def validate_bpm_inputs(
    proctab,
    output_dir,
    min_darks=5,
    min_flats=5,
    suffix="_L12",
):
    """
    Validate slope images for creating a bad-pixel map (BPM).

    Requirements
    ------------
    - Processing suffix must be L12.
    - Only DARK and FLATLAMP inputs are used.
    - Each FITS input must contain a 2-D slope image.
    - CAMERA and MCLOCK must match.
    - Each valid group must contain at least:
          `min_darks` DARK slopes
          `min_flats` FLATLAMP slopes

    Parameters
    ----------
    proctab : pandas.DataFrame
        SCALES processing table.

    output_dir : str
        Directory containing processed FITS files.

    min_darks : int, optional
        Minimum DARK slopes required per CAMERA/MCLOCK group.
        Default is 5.

    min_flats : int, optional
        Minimum FLATLAMP slopes required per CAMERA/MCLOCK group.
        Default is 5.

    suffix : str, optional
        Processing suffix.
        Default is '_L12'.

    Returns
    -------
    valid_groups : list of dict
        Each valid group has the form:

            {
                'params': {
                    'camera': ...,
                    'mclock': ...
                },

                'dark_files': [...],
                'dark_paths': [...],

                'flat_files': [...],
                'flat_paths': [...]
            }

    Raises
    ------
    RuntimeError
        If no CAMERA/MCLOCK group contains enough valid
        DARK and FLATLAMP slopes.
    """

    dt = get_proc_entries(
        proctab,
        suffix=suffix,
    )

    if len(dt) == 0:
        raise RuntimeError(
            f"Cannot create BPM: "
            f"no {suffix} entries were found."
        )

    required_cols = [
        "CAMERA",
        "MCLOCK",
        "IMTYPE",
        "filename",
    ]
    missing = [col for col in required_cols if col not in dt.columns]

    if missing:
        raise ValueError(
            f"Processing table is missing columns required "
            f"for BPM grouping: {missing}"
        )

    # Keep only DARK and FLATLAMP
    imtype_upper = (
        dt["IMTYPE"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dt = dt[
        imtype_upper.isin(
            ["DARK", "FLATLAMP"]
        )
    ].copy()

    if len(dt) == 0:
        raise RuntimeError(
            "Cannot create BPM: no DARK or FLATLAMP "
            f"{suffix} entries were found."
        )

    valid_rows = []

    for idx, row in dt.iterrows():

        filename = str(row["filename"]).strip()
        path = os.path.join(output_dir, filename)

        if not os.path.isfile(path):
            print(
                f"WARNING: skipping BPM input; "
                f"file does not exist: {path}"
            )
            continue

        valid_rows.append(idx)

    if not valid_rows:
        raise RuntimeError(
            "Cannot create BPM: no valid 2-D "
            "DARK/FLATLAMP slopes were found."
        )

    valid_dt = dt.loc[valid_rows].copy()

    group_keys = [
        "CAMERA",
        "MCLOCK",
    ]

    valid_groups = []

    for group_params, sub_df in valid_dt.groupby(
        group_keys,
        dropna=False,
    ):

        imtype_upper = (
            sub_df["IMTYPE"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        dark_df = sub_df[
            imtype_upper == "DARK"
        ]

        flat_df = sub_df[
            imtype_upper == "FLATLAMP"
        ]

        n_dark = len(dark_df)
        n_flat = len(flat_df)

        if n_dark < min_darks or n_flat < min_flats:

            print(
                f"WARNING: skipping BPM group {group_params}: "
                f"{n_dark} DARK and {n_flat} FLATLAMP "
                f"slopes found; require at least "
                f"{min_darks} DARK and {min_flats} FLATLAMP."
            )

            continue

        dark_files = (
            dark_df["filename"]
            .astype(str)
            .str.strip()
            .tolist()
        )

        flat_files = (
            flat_df["filename"]
            .astype(str)
            .str.strip()
            .tolist()
        )

        valid_groups.append(
            {
                "params": {
                    key.lower(): value
                    for key, value in zip(
                        group_keys,
                        group_params,
                    )
                },

                "dark_files": dark_files,

                "dark_paths": [
                    os.path.join(output_dir, filename)
                    for filename in dark_files
                ],

                "flat_files": flat_files,

                "flat_paths": [
                    os.path.join(output_dir, filename)
                    for filename in flat_files
                ],
            }
        )

    if not valid_groups:
        raise RuntimeError(
            "Cannot create BPM. "
            "No CAMERA/MCLOCK group contains at least "
            f"{min_darks} valid DARK {suffix} slopes and "
            f"{min_flats} valid FLATLAMP {suffix} slopes."
        )

    return valid_groups

def validate_rn_inputs(
    proctab,
    output_dir,
    min_files=5,
    min_reads=2,
    suffix="_L13",
):
    """
    Validate and group input files for creating readnoise maps.

    Requirements
    ------------
    - IMTYPE must be DARK.
    - Processing suffix must be L13.
    - FITS data must be a 3-D ramp cube.
    - Each cube must contain at least `min_reads` reads.
    - Valid cubes are grouped by exact:
          CAMERA, MCLOCK, EXPTIME
    - Each group must contain at least `min_files` valid cubes.

    Parameters
    ----------
    proctab : pandas.DataFrame
        SCALES processing table.

    output_dir : str
        Directory containing the processed FITS files.

    min_files : int, optional
        Minimum number of valid ramp cubes required per group.
        Default is 2.

    min_reads : int, optional
        Minimum number of reads required in each ramp cube.
        Default is 30.

    suffix : str, optional
        Processing suffix.
        Default is '_L11'.

    Returns
    -------
    valid_groups : list of dict
        Valid saturation-map groups:

            [
                {
                    "params": {
                        "camera": ...,
                        "mclock": ...,
                        "exptime": ...
                    },
                    "filenames": [...],
                    "paths": [...]
                },
                ...
            ]

    Raises
    ------
    RuntimeError
        If no valid saturation-map group exists.
    """

    # ---------------------------------------------------------
    # Find FLATLAMP L11 products
    # ---------------------------------------------------------

    dt = get_proc_entries(
        proctab,
        suffix=suffix,
        imtype="DARK",
    )

    if len(dt) == 0:
        raise RuntimeError(
            f"Cannot create saturation map: "
            f"no DARK {suffix} entries were found."
        )

    # ---------------------------------------------------------
    # Check required grouping columns
    # ---------------------------------------------------------

    group_keys = [
        "CAMERA",
        "MCLOCK",
        "EXPTIME",
    ]

    missing = [
        col for col in group_keys
        if col not in dt.columns
    ]

    if missing:
        raise RuntimeError(
            "Cannot create saturation map: "
            f"processing table is missing required "
            f"grouping columns: {missing}"
        )

    # ---------------------------------------------------------
    # Validate individual L11 ramp cubes
    # ---------------------------------------------------------

    valid_indices = []

    for idx, row in dt.iterrows():

        filename = str(
            row["filename"]
        ).strip()

        path = os.path.join(
            output_dir,
            filename
        )

        if not os.path.isfile(path):
            print(
                f"WARNING: skipping saturation input; "
                f"file does not exist: {path}"
            )
            continue

        try:

            with fits.open(
                path,
                memmap=True
            ) as hdul:

                data = hdul[0].data

                if data is None:
                    print(
                        f"WARNING: skipping {filename}; "
                        "FITS data is empty."
                    )
                    continue

                if data.ndim != 3:
                    print(
                        f"WARNING: skipping {filename}; "
                        f"expected 3-D cube, "
                        f"got shape={data.shape}."
                    )
                    continue

                nreads = data.shape[0]

                if nreads < min_reads:
                    print(
                        f"WARNING: skipping {filename}; "
                        f"{nreads} reads found, "
                        f"minimum required is "
                        f"{min_reads}."
                    )
                    continue

        except Exception as exc:

            print(
                f"WARNING: could not validate "
                f"{filename}: {exc}"
            )
            continue

        # File passed validation
        valid_indices.append(idx)

    # ---------------------------------------------------------
    # Make table containing only valid L11 cubes
    # ---------------------------------------------------------

    if not valid_indices:
        raise RuntimeError(
            "Cannot create saturation map. "
            f"No valid FLATLAMP {suffix} 3-D cubes "
            f"with at least {min_reads} reads were found."
        )

    valid_dt = dt.loc[
        valid_indices
    ].copy()

    # ---------------------------------------------------------
    # Group by CAMERA + MCLOCK + EXPTIME
    # ---------------------------------------------------------

    valid_groups = []

    for group_params, sub_df in valid_dt.groupby(
        group_keys,
        dropna=False
    ):

        camera, mclock, exptime = group_params

        nfiles = len(sub_df)

        # -----------------------------------------------------
        # Require enough ramps in this configuration
        # -----------------------------------------------------

        if nfiles < min_files:

            print(
                "WARNING: skipping rn-map group "
                f"CAMERA={camera}, "
                f"MCLOCK={mclock}, "
                f"EXPTIME={exptime}; "
                f"found {nfiles} valid ramps, "
                f"minimum required is {min_files}."
            )

            continue

        filenames = (
            sub_df["filename"]
            .astype(str)
            .str.strip()
            .tolist()
        )

        paths = [
            os.path.join(
                output_dir,
                filename
            )
            for filename in filenames
        ]

        valid_groups.append(
            {
                "params": {
                    "camera": camera,
                    "mclock": mclock,
                    "exptime": exptime,
                },
                "filenames": filenames,
                "paths": paths,
            }
        )
    # ---------------------------------------------------------
    # Make sure at least one valid group exists
    # ---------------------------------------------------------

    if not valid_groups:
        raise RuntimeError(
            "Cannot create readnoise map. "
            "No valid CAMERA/MCLOCK/EXPTIME group "
            f"contains at least {min_files} DARK "
            f"{suffix} ramp cubes with at least "
            f"{min_reads} reads each."
        )

    return valid_groups

def find_existing_master_for_l1_group(
    l1_filenames,
    redux_dir,
    master_type=None,
    check_ncombine=True,
):
    """
    Find an existing master calibration corresponding to a group
    of L1 calibration files.

    Parameters
    ----------
    l1_filenames : list of str
        L1 filenames belonging to one calibration group.

    redux_dir : str or Path
        Directory containing scales.proc and calibration products.

    master_type : str or None
        One of 'DARK', 'FLATLAMP', 'CALUNIT', or 'BIAS'.
        If None, determine from the L1 entries' IMTYPE.

    check_ncombine : bool
        If True, compare the master's NCOMBINE keyword with
        the number of L1 files, when that keyword exists.

    Returns
    -------
    str or None
        Existing master calibration FITS path, or None.

    Notes
    -----
    Matches by calibration type, detector configuration,
    exposure time, and relevant instrument settings.

    For CALUNIT, also matches MONOWAVE.

    This checks calibration-group compatibility. Exact input-file
    provenance requires storing the contributing L1 filenames
    in the master product.
    """

    redux_dir = Path(redux_dir)
    proctab_path = redux_dir / "scales.proc"

    if not proctab_path.is_file():
        return None

    if not l1_filenames:
        return None

    # ---------------------------------------------------------
    # Read processing table
    # ---------------------------------------------------------

    try:
        proc = pd.read_csv(
            proctab_path,
            sep="|",
            engine="python",
            skipinitialspace=True,
        )
    except Exception as exc:
        print(f"Could not read {proctab_path}: {exc}")
        return None

    proc.columns = proc.columns.astype(str).str.strip()

    proc = proc.loc[
        :,
        ~proc.columns.str.startswith("Unnamed:")
    ].copy()

    required = {
        "filename",
        "SUFF",
        "IMTYPE",
        "CAMERA",
        "MCLOCK",
        "EXPTIME",
    }

    if not required.issubset(proc.columns):
        raise ValueError(
            f"Missing proctab columns: "
            f"{sorted(required - set(proc.columns))}"
        )

    proc["filename"] = (
        proc["filename"].fillna("").astype(str).str.strip()
    )

    proc["SUFF"] = (
        proc["SUFF"].fillna("").astype(str).str.strip()
    )

    proc["IMTYPE"] = (
        proc["IMTYPE"].fillna("").astype(str).str.strip().str.upper()
    )

    # ---------------------------------------------------------
    # Identify the requested L1 files
    # ---------------------------------------------------------

    l1_names = {
        Path(name).name
        for name in l1_filenames
    }

    l1_rows = proc.loc[
        proc["filename"].map(
            lambda name: Path(name).name in l1_names
        )
        & proc["SUFF"].str.upper().eq("_L1")
    ].copy()

    found_names = set(l1_rows["filename"].map(
        lambda name: Path(name).name
    ))

    if found_names != l1_names:
        missing = sorted(l1_names - found_names)
        print(f"L1 files missing from proctab: {missing}")
        return None

    # ---------------------------------------------------------
    # Determine master type
    # ---------------------------------------------------------

    imtypes = l1_rows["IMTYPE"].unique()

    if len(imtypes) != 1:
        raise ValueError(
            f"Mixed IMTYPE values in L1 group: {imtypes}"
        )

    imtype = imtypes[0]

    if master_type is None:
        master_type = imtype
    else:
        master_type = str(master_type).strip().upper()

    if master_type != imtype:
        raise ValueError(
            f"Requested {master_type} master for "
            f"{imtype} L1 files."
        )

    suffix_map = {
        "DARK": "_mdark",
        "FLATLAMP": "_mflatlamp",
        "CALUNIT": "_mcalunit",
        "BIAS": "_mbias",
    }

    if master_type not in suffix_map:
        raise ValueError(
            f"Unsupported master type: {master_type}"
        )

    master_suffix = suffix_map[master_type]

    # ---------------------------------------------------------
    # Determine calibration matching keys
    # ---------------------------------------------------------

    # Required detector configuration
    group_keys = [
        "CAMERA",
        "MCLOCK",
        "EXPTIME",
    ]

    # Include instrument configuration for illuminated products.
    if master_type in ("FLATLAMP", "CALUNIT"):
        group_keys.extend([
            "MODSLNAM",
            "DSPRSNAM",
        ])

    # Each CALUNIT wavelength requires its own master.
    if master_type == "CALUNIT":
        group_keys.append("MONOWAVE")

    # Verify columns exist
    missing_keys = [
        key for key in group_keys
        if key not in proc.columns
    ]

    if missing_keys:
        raise ValueError(
            f"Missing calibration keys: {missing_keys}"
        )

    # Verify all input L1 files have the same configuration
    for key in group_keys:

        values = l1_rows[key]

        if values.isna().any():
            print(f"Missing {key} in L1 group.")
            return None

        if key in ("EXPTIME", "MONOWAVE"):
            numeric = pd.to_numeric(values, errors="coerce")

            if (
                numeric.isna().any()
                or not np.allclose(
                    numeric,
                    numeric.iloc[0],
                    rtol=0,
                    atol=1e-5,
                )
            ):
                raise ValueError(
                    f"Inconsistent {key} in L1 group."
                )

        else:
            normalized = values.astype(str).str.strip()

            if normalized.nunique() != 1:
                raise ValueError(
                    f"Inconsistent {key} in L1 group: "
                    f"{normalized.unique().tolist()}"
                )

    # ---------------------------------------------------------
    # Select existing master entries
    # ---------------------------------------------------------

    # Accept both:
    #    _mdark
    #    _2370.25_mcalunit
    #
    # Avoid confusing L1 files with master products.

    master_rows = proc.loc[
        proc["IMTYPE"].eq(master_type)
        & proc["SUFF"].str.lower().str.endswith(master_suffix)
    ].copy()

    if master_rows.empty:
        return None

    # ---------------------------------------------------------
    # Match configuration
    # ---------------------------------------------------------

    for key in group_keys:

        expected = l1_rows[key].iloc[0]

        if key in ("EXPTIME", "MONOWAVE"):

            values = pd.to_numeric(
                master_rows[key],
                errors="coerce",
            )

            mask = np.isclose(
                values.to_numpy(dtype=float),
                float(expected),
                rtol=0,
                atol=1e-5,
            )

        else:

            mask = (
                master_rows[key]
                .astype(str)
                .str.strip()
                .eq(str(expected).strip())
            )

        master_rows = master_rows.loc[mask]

        if master_rows.empty:
            return None

    # ---------------------------------------------------------
    # Verify the actual master FITS file
    # ---------------------------------------------------------

    for _, row in master_rows.iloc[::-1].iterrows():

        filename = str(row["filename"]).strip()
        path = Path(filename)

        if not path.is_absolute():
            path = redux_dir / path

        if not path.is_file():
            continue

        if check_ncombine:

            try:
                with fits.open(path, memmap=False) as hdul:
                    ncombine = hdul[0].header.get(
                        "NCOMBINE"
                    )

                if (
                    ncombine is not None
                    and int(ncombine) != len(l1_names)
                ):
                    continue

            except Exception as exc:
                print(
                    f"Could not validate master {path}: {exc}"
                )
                continue

        return str(path)

    return None

def validate_master_inputs(
    proctab,
    output_dir,
    imtype=None,
    min_files=1,
    suffix="_L1",
):
    """
    Validate and group L1 calibration products for master creation.

    Parameters
    ----------
    proctab : pandas.DataFrame
        SCALES processing table.

    output_dir : str
        Directory containing L1 FITS files.

    imtype : str or None, optional
        Calibration type to select, e.g. 'DARK', 'FLATLAMP',
        'CALUNIT', or 'BIAS'.

        If None, include all IMTYPE values, grouped separately.

    min_files : int, optional
        Minimum number of valid L1 files per group.

    suffix : str, optional
        Processing suffix. Default is '_L1'.

    Returns
    -------
    valid_groups : list of dict
        Each group contains:

        {
            "params": {
                "imtype": ...,
                "camera": ...,
                "mclock": ...,
                "exptime": ...,
                ...
            },
            "filenames": [...],
            "paths": [...]
        }

    Notes
    -----
    - Only existing 2D FITS images are accepted.
    - Groups are separated by IMTYPE, CAMERA, MCLOCK,
      and EXPTIME.
    - FLATLAMP and CALUNIT are additionally grouped by
      MODSLNAM, DSPRSNAM, and FILTER.
    - CALUNIT is additionally grouped by MONOWAVE.
    """

    # ---------------------------------------------------------
    # Select L1 processing entries
    # ---------------------------------------------------------

    dt = get_proc_entries(
        proctab,
        suffix=suffix,
        imtype=imtype,
    )

    if len(dt) == 0:
        raise RuntimeError(
            f"No {imtype or 'any IMTYPE'} {suffix} "
            "processing entries were found."
        )

    dt = dt.copy()

    dt["IMTYPE"] = (
        dt["IMTYPE"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    required_columns = [
        "filename",
        "IMTYPE",
        "CAMERA",
        "MCLOCK",
        "EXPTIME",
    ]

    missing = [
        col for col in required_columns
        if col not in dt.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required proctab columns: {missing}"
        )

    # ---------------------------------------------------------
    # Validate individual L1 images
    # ---------------------------------------------------------

    valid_indices = []

    for idx, row in dt.iterrows():

        filename = str(row["filename"]).strip()

        path = os.path.join(
            output_dir,
            filename
        )

        if not os.path.isfile(path):
            print(
                f"WARNING: L1 file does not exist: {path}"
            )
            continue

        try:
            with fits.open(path, memmap=True) as hdul:

                data = hdul[0].data

                if data is None:
                    print(
                        f"WARNING: empty L1 file: {filename}"
                    )
                    continue

                if data.ndim != 2:
                    print(
                        f"WARNING: skipping {filename}; "
                        f"expected 2D L1 image, "
                        f"got shape={data.shape}"
                    )
                    continue

                if not np.any(np.isfinite(data)):
                    print(
                        f"WARNING: skipping {filename}; "
                        "no finite pixels."
                    )
                    continue

        except Exception as exc:
            print(
                f"WARNING: cannot validate "
                f"{filename}: {exc}"
            )
            continue

        valid_indices.append(idx)

    if not valid_indices:
        raise RuntimeError(
            f"No valid 2D {suffix} images found."
        )

    valid_dt = dt.loc[valid_indices].copy()

    # ---------------------------------------------------------
    # Group by IMTYPE and configuration
    # ---------------------------------------------------------

    valid_groups = []

    for current_imtype, type_df in valid_dt.groupby(
        "IMTYPE",
        dropna=False,
    ):

        group_keys = [
            "CAMERA",
            "MCLOCK",
            "EXPTIME",
        ]

        if current_imtype in ("FLATLAMP", "CALUNIT"):
            group_keys.extend([
                "MODSLNAM",
                "DSPRSNAM",
                "FILTER",
            ])

        if current_imtype == "CALUNIT":
            group_keys.append("MONOWAVE")

        missing_keys = [
            key for key in group_keys
            if key not in type_df.columns
        ]

        if missing_keys:
            raise RuntimeError(
                f"Missing grouping columns for "
                f"{current_imtype}: {missing_keys}"
            )

        # -----------------------------------------------------
        # Create groups
        # -----------------------------------------------------

        for group_values, sub_df in type_df.groupby(
            group_keys,
            dropna=False,
            sort=False,
        ):

            if not isinstance(group_values, tuple):
                group_values = (group_values,)

            nfiles = len(sub_df)

            if nfiles < min_files:
                print(
                    f"WARNING: skipping {current_imtype} "
                    f"group {group_values}; "
                    f"found {nfiles} L1 files, "
                    f"minimum required is {min_files}."
                )
                continue

            filenames = (
                sub_df["filename"]
                .astype(str)
                .str.strip()
                .tolist()
            )

            paths = [
                os.path.join(output_dir, name)
                for name in filenames
            ]

            params = {
                "imtype": current_imtype,
            }

            for key, value in zip(
                group_keys,
                group_values,
            ):
                params[key.lower()] = value

            valid_groups.append(
                {
                    "params": params,
                    "filenames": filenames,
                    "paths": paths,
                }
            )

    # ---------------------------------------------------------
    # Ensure valid groups exist
    # ---------------------------------------------------------

    if not valid_groups:
        raise RuntimeError(
            f"No valid {suffix} calibration groups "
            f"with at least {min_files} files."
        )

    return valid_groups
################plotting###########################################################
def plot_master_calib(
    master,
    output_dir,
    *,
    master_dq=None,
    filename="master_calibration.png",
    title="Master Calibration",
    unit="DN/s",
    exclude_bad=True,
    percentile_range=(1, 99),
    bins=100,
    dpi=200,
    show=False,
):
    """
    Plot a master calibration image and its pixel-value distribution.

    Parameters
    ----------
    master : ndarray
        2D master calibration image.

    output_dir : str or Path
        Directory where the figure will be saved.

    master_dq : ndarray, optional
        2D data-quality mask.
        Pixels with nonzero DQ are excluded from statistics
        when exclude_bad=True.

    filename : str
        Output PNG filename.

    title : str
        Figure title.

    unit : str
        Physical unit of the master calibration image.

    exclude_bad : bool
        Exclude DQ-flagged pixels from the histogram and
        percentile calculations.

    percentile_range : tuple
        Percentiles used for image color scaling.

    bins : int
        Number of histogram bins.

    dpi : int
        Figure resolution.

    show : bool
        Display the figure interactively.

    Returns
    -------
    pathlib.Path
        Path to the saved diagnostic figure.
    """

    master = np.asarray(master, dtype=np.float64)

    if master.ndim != 2:
        raise ValueError(
            f"master must be a 2D image. Got {master.shape}."
        )

    # ---------------------------------------------------------
    # Construct valid-pixel mask
    # ---------------------------------------------------------

    valid = np.isfinite(master)

    if master_dq is not None:

        master_dq = np.asarray(master_dq)

        if master_dq.shape != master.shape:
            raise ValueError(
                "master_dq must have the same shape as master."
            )

        if exclude_bad:
            valid &= (master_dq == 0)

    if not np.any(valid):
        raise ValueError(
            "No valid pixels available for plotting."
        )

    values = master[valid]

    # ---------------------------------------------------------
    # Compute statistics
    # ---------------------------------------------------------

    median = float(np.median(values))
    mean = float(np.mean(values))
    std = float(np.std(values))

    p5, p95 = np.percentile(values, [5, 95])

    vmin, vmax = np.percentile(
        values, percentile_range
    )

    if vmax <= vmin:
        delta = max(abs(vmin) * 0.01, 1e-6)
        vmin -= delta
        vmax += delta

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5.5),
        constrained_layout=True,
    )

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
    )

    # ---------------------------------------------------------
    # Panel 1: Master calibration map
    # ---------------------------------------------------------

    ax = axes[0]

    im = ax.imshow(
        master,
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(unit)

    ax.set_title("Master Calibration Map")
    ax.set_xlabel("Detector X (pixel)")
    ax.set_ylabel("Detector Y (pixel)")

    # ---------------------------------------------------------
    # Panel 2: Histogram
    # ---------------------------------------------------------

    ax = axes[1]

    ax.hist(
        values,
        bins=bins,
        range=(vmin, vmax),
        color="steelblue",
        alpha=0.75,
        edgecolor="none",
    )

    ax.axvline(
        median,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Median = {median:.3g} {unit}",
    )

    ax.axvline(
        p5,
        color="black",
        linestyle=":",
        linewidth=1.3,
        label=f"P5 = {p5:.3g}",
    )

    ax.axvline(
        p95,
        color="black",
        linestyle=":",
        linewidth=1.3,
        label=f"P95 = {p95:.3g}",
    )

    ax.set_title("Pixel-Value Distribution")
    ax.set_xlabel(f"Pixel value ({unit})")
    ax.set_ylabel("Number of pixels")

    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    ax.text(
        0.98, 0.97,
        f"Mean = {mean:.3g}\n"
        f"Std = {std:.3g}\n"
        f"Valid = {np.count_nonzero(valid):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="lightgray",
        ),
    )

    # ---------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    savepath = output_dir / filename

    fig.savefig(
        savepath,
        dpi=dpi,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)

    return savepath

def plot_bad_pixel_correction_comparison(
    slope_before,
    slope_after,
    output_dir,
    *,
    filename="bad_pixel_correction_comparison.png",
    title="SCALES Bad-Pixel Correction",
    unit="DN/s",
    percentile_range=(1, 99),
    dpi=200,
    show=False,
):
    """
    Compare 2D slope images before and after bad-pixel correction.

    Parameters
    ----------
    slope_before : ndarray
        2D slope image before bad-pixel correction.

    slope_after : ndarray
        2D slope image after bad-pixel correction.

    output_dir : str or Path
        Directory where the diagnostic figure is saved.

    filename : str
        Output PNG filename.

    title : str
        Figure title.

    unit : str
        Slope image unit, normally DN/s.

    percentile_range : tuple
        Percentiles used to determine the common color limits.

    dpi : int
        Output figure resolution.

    show : bool
        Display the figure interactively.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """

    # ---------------------------------------------------------
    # 1. Validate input images
    # ---------------------------------------------------------

    slope_before = np.asarray(
        slope_before, dtype=np.float64
    )

    slope_after = np.asarray(
        slope_after, dtype=np.float64
    )

    if slope_before.ndim != 2 or slope_after.ndim != 2:
        raise ValueError(
            "Both slope images must be 2D."
        )

    if slope_before.shape != slope_after.shape:
        raise ValueError(
            "Slope image shapes do not match: "
            f"{slope_before.shape} vs {slope_after.shape}"
        )

    p_low, p_high = percentile_range

    if not (0 <= p_low < p_high <= 100):
        raise ValueError(
            "Invalid percentile_range."
        )

    # ---------------------------------------------------------
    # 2. Determine common display limits
    # ---------------------------------------------------------

    valid_before = slope_before[
        np.isfinite(slope_before)
    ]

    valid_after = slope_after[
        np.isfinite(slope_after)
    ]

    if valid_before.size == 0 or valid_after.size == 0:
        raise ValueError(
            "Both slope images must contain finite pixels."
        )

    combined = np.concatenate(
        [valid_before, valid_after]
    )

    vmin, vmax = np.percentile(
        combined, [p_low, p_high]
    )

    if vmax <= vmin:
        delta = max(abs(vmin) * 0.01, 1e-6)
        vmin -= delta
        vmax += delta

    # ---------------------------------------------------------
    # 3. Create two-panel figure
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6),
        constrained_layout=True,
    )

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
    )

    # ---------------------------------------------------------
    # Panel 1: Before bad-pixel correction
    # ---------------------------------------------------------

    im0 = axes[0].imshow(
        slope_before,
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    axes[0].set_title(
        "Before Bad-Pixel Correction"
    )

    axes[0].set_xlabel("Detector X (pixel)")
    axes[0].set_ylabel("Detector Y (pixel)")

    # ---------------------------------------------------------
    # Panel 2: After bad-pixel correction
    # ---------------------------------------------------------

    im1 = axes[1].imshow(
        slope_after,
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    axes[1].set_title(
        "After Bad-Pixel Correction"
    )

    axes[1].set_xlabel("Detector X (pixel)")
    axes[1].set_ylabel("Detector Y (pixel)")

    # ---------------------------------------------------------
    # 4. Shared colorbar
    # ---------------------------------------------------------

    cbar = fig.colorbar(
        im1,
        ax=axes,
        fraction=0.035,
        pad=0.02,
        shrink=0.85,
    )

    cbar.set_label(unit)

    # ---------------------------------------------------------
    # 5. Save figure
    # ---------------------------------------------------------

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    savepath = output_dir / filename

    fig.savefig(
        savepath,
        dpi=dpi,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)

    return savepath



def plot_read_correction_comparison(
    cube_before,
    cube_after,
    output_dir,
    *,
    filename="read_correction_comparison.png",
    title="SCALES Detector Noise Correction",
    unit="DN",
    percentile_range=(1, 99),
    difference_percentile=99,
    dpi=200,
    show=False,
):
    """
    Compare detector reads before and after ACN / 1/f correction.

    Parameters
    ----------
    cube_before : ndarray
        Raw read cube with shape (n_reads, ny, nx).

    cube_after : ndarray
        Corrected read cube with the same shape.

    output_dir : str or Path
        Directory where the diagnostic figure is saved.

    filename : str
        Output figure filename.

    title : str
        Figure title.

    unit : str
        Detector signal unit, normally DN.

    percentile_range : tuple
        Percentiles used to determine the common color limits
        for before and after images in each row.

    difference_percentile : float
        Percentile of absolute differences used for symmetric
        difference-image color limits.

    dpi : int
        Output figure resolution.

    show : bool
        Display figure interactively.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.

    Notes
    -----
    The difference is defined as:

        difference = before - after

    This represents the signal removed by the correction.

    The function does not modify either input cube.
    """

    # ---------------------------------------------------------
    # 1. Validate input cubes
    # ---------------------------------------------------------

    cube_before = np.asarray(cube_before)
    cube_after = np.asarray(cube_after)

    if cube_before.ndim != 3 or cube_after.ndim != 3:
        raise ValueError(
            "Both inputs must be 3D cubes with shape "
            "(n_reads, ny, nx)."
        )

    if cube_before.shape != cube_after.shape:
        raise ValueError(
            "Input cube shapes do not match: "
            f"{cube_before.shape} vs {cube_after.shape}"
        )

    n_reads, ny, nx = cube_before.shape

    if n_reads < 2:
        raise ValueError(
            "At least two reads are required."
        )

    p_low, p_high = percentile_range

    if not (0 <= p_low < p_high <= 100):
        raise ValueError(
            "Invalid percentile_range."
        )

    if not (0 < difference_percentile <= 100):
        raise ValueError(
            "difference_percentile must be in (0, 100]."
        )

    # ---------------------------------------------------------
    # 2. Select first and last reads
    # ---------------------------------------------------------

    read_indices = [0, n_reads - 1]
    read_labels = [
        "First Read",
        "Last Read",
    ]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 11),
        constrained_layout=True,
    )

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
    )

    # ---------------------------------------------------------
    # 3. Plot each selected read
    # ---------------------------------------------------------

    for row, (read_idx, read_label) in enumerate(
        zip(read_indices, read_labels)
    ):

        before = np.asarray(
            cube_before[read_idx],
            dtype=np.float64,
        )

        after = np.asarray(
            cube_after[read_idx],
            dtype=np.float64,
        )

        difference = before - after

        # -----------------------------------------------------
        # Common display scale for before and after
        # -----------------------------------------------------

        combined = np.concatenate([
            before[np.isfinite(before)],
            after[np.isfinite(after)],
        ])

        if combined.size == 0:
            raise ValueError(
                f"No finite values in read {read_idx}."
            )

        vmin, vmax = np.percentile(
            combined,
            [p_low, p_high],
        )

        if vmax <= vmin:
            delta = max(abs(vmin) * 0.01, 1e-6)
            vmin -= delta
            vmax += delta

        # -----------------------------------------------------
        # Symmetric difference scale centered at zero
        # -----------------------------------------------------

        finite_diff = difference[
            np.isfinite(difference)
        ]

        if finite_diff.size == 0:
            raise ValueError(
                f"No finite difference values "
                f"in read {read_idx}."
            )

        diff_limit = np.percentile(
            np.abs(finite_diff),
            difference_percentile,
        )

        if not np.isfinite(diff_limit) or diff_limit <= 0:
            diff_limit = 1e-6

        diff_norm = TwoSlopeNorm(
            vmin=-diff_limit,
            vcenter=0,
            vmax=diff_limit,
        )

        # -----------------------------------------------------
        # Panel 1: Before correction
        # -----------------------------------------------------

        ax = axes[row, 0]

        im = ax.imshow(
            before,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_title(
            f"{read_label} (Read {read_idx}) — Before"
        )

        fig.colorbar(
            im,
            ax=ax,
            label=unit,
            fraction=0.046,
            pad=0.04,
        )

        # -----------------------------------------------------
        # Panel 2: After correction
        # -----------------------------------------------------

        ax = axes[row, 1]

        im = ax.imshow(
            after,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_title(
            f"{read_label} (Read {read_idx}) — After"
        )

        fig.colorbar(
            im,
            ax=ax,
            label=unit,
            fraction=0.046,
            pad=0.04,
        )

        # -----------------------------------------------------
        # Panel 3: Difference image
        # -----------------------------------------------------

        ax = axes[row, 2]

        im = ax.imshow(
            difference,
            origin="lower",
            cmap="RdBu_r",
            norm=diff_norm,
            interpolation="nearest",
        )

        ax.set_title(
            f"{read_label} — Before − After"
        )

        fig.colorbar(
            im,
            ax=ax,
            label=unit,
            fraction=0.046,
            pad=0.04,
        )

        # -----------------------------------------------------
        # Axis labels
        # -----------------------------------------------------

        for ax in axes[row]:

            ax.set_xlabel("Detector X (pixel)")
            ax.set_ylabel("Detector Y (pixel)")

            ax.set_xlim(0, nx - 1)
            ax.set_ylim(0, ny - 1)

    # ---------------------------------------------------------
    # 4. Save figure
    # ---------------------------------------------------------

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    savepath = output_dir / filename

    fig.savefig(
        savepath,
        dpi=dpi,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)

    return savepath


def plot_linearity_saturation_diagnostics(
    raw_cube, corrected_cube, saturation_mask, applied_mask, good_read_mask,
    *, output_dir="linearity_plots", save_prefix="linearity_diagnostic",
    skip_initial_reads=2, minimum_signal_dn=0.0, dpi=180, show=False,
):
    """Plot 2x2 detector maps of correction and ramp-fitting read availability.

    Inputs are the arrays returned by apply_brandt_linearity_reference(...,
    return_aux=True) and make_ramp_quality_mask(...). Shapes are (N,H,W)
    for cubes/read masks and (H,W) for applied_mask.

    Fractional correction uses 100*(corrected-raw)/(raw-pedestal), only on
    pre-saturation reads with abs(signal)>=minimum_signal_dn, where the
    pedestal is estimated from the first two reads (matching the shown call).
    Saturated/raw-preserved reads are excluded from correction statistics.

    Read availability comes from good_read_mask, which includes BPM,
    saturation, neighbor rejection, and nonfinite read exclusions. The lost
    read count is measured relative to finite raw reads, after the initial
    excluded reads; it is not exclusively attributable to saturation.
    """
    raw = np.asarray(raw_cube)
    corrected = np.asarray(corrected_cube)
    sat = np.asarray(saturation_mask, dtype=bool)
    applied = np.asarray(applied_mask, dtype=bool)
    good = np.asarray(good_read_mask, dtype=bool)
    if raw.ndim != 3 or corrected.shape != raw.shape or sat.shape != raw.shape or good.shape != raw.shape:
        raise ValueError("raw_cube, corrected_cube, saturation_mask and good_read_mask must all have shape (Nreads,H,W)")
    nreads, h, w = raw.shape
    if applied.shape != (h, w):
        raise ValueError("applied_mask must have shape (H,W)")
    if not (0 <= skip_initial_reads < nreads):
        raise ValueError("skip_initial_reads must be in [0, nreads)")
    if minimum_signal_dn < 0:
        raise ValueError("minimum_signal_dn must be non-negative")

    # Pedestal convention matches n_pedestal_reads=2, pedestal_start_read=0.
    with np.errstate(invalid="ignore"):
        pedestal = np.nanmedian(raw[:2].astype(np.float64), axis=0)

    # Streaming accumulators avoid constructing a full float64 difference cube.
    frac_sum = np.zeros((h, w), dtype=np.float64)
    frac_count = np.zeros((h, w), dtype=np.int32)
    max_abs_dn = np.full((h, w), np.nan, dtype=np.float32)
    n_finite = np.zeros((h, w), dtype=np.int32)
    n_good = np.zeros((h, w), dtype=np.int32)

    for i in range(skip_initial_reads, nreads):
        r = raw[i].astype(np.float64)
        c = corrected[i].astype(np.float64)
        finite = np.isfinite(r)
        n_finite += finite
        n_good += good[i] & finite & np.isfinite(c)
        signal = r - pedestal
        eligible = (applied & ~sat[i] & finite & np.isfinite(c)
                    & np.isfinite(signal) & (np.abs(signal) >= minimum_signal_dn))
        if not np.any(eligible):
            continue
        delta = c - r
        frac = np.zeros((h, w), dtype=np.float64)
        np.divide(100.0 * delta, signal, out=frac, where=eligible)
        frac_sum[eligible] += frac[eligible]
        frac_count[eligible] += 1
        abs_delta = np.abs(delta[eligible]).astype(np.float32)
        existing = max_abs_dn[eligible]
        max_abs_dn[eligible] = np.where(np.isfinite(existing), np.maximum(existing, abs_delta), abs_delta)

    median_label = "Mean fractional correction [%]"
    mean_frac = np.full((h, w), np.nan, dtype=np.float32)
    np.divide(frac_sum, frac_count, out=mean_frac, where=frac_count > 0)
    # Lost reads include BPM, nonfinite reads, and neighbor-induced masking.
    n_lost = np.maximum(n_finite - n_good, 0)
    n_good = n_good.astype(np.float32)
    n_lost = n_lost.astype(np.float32)
    n_good[~np.any(np.isfinite(raw[skip_initial_reads:]), axis=0)] = np.nan

    print(f"Linearity diagnostics: applied pixels={np.count_nonzero(applied):,}/{h*w:,}, "
          f"pixels with eligible reads={np.count_nonzero(frac_count):,}, "
          f"nonzero corrections={np.count_nonzero(np.isfinite(max_abs_dn) & (max_abs_dn > 0)):,}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    def display(ax, data, title, label, *, cmap="viridis", center_zero=False, vmax=None):
        finite = np.asarray(data)[np.isfinite(data)]
        kw = {}
        if center_zero and finite.size:
            limit = float(np.percentile(np.abs(finite), 99))
            if limit > 0:
                kw["norm"] = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        elif vmax is not None:
            kw.update(vmin=0, vmax=vmax)
        elif finite.size and np.nanmin(finite) >= 0:
            hi = float(np.percentile(finite, 99.5))
            kw.update(vmin=0, vmax=hi if hi > 0 else 1.0)
        elif finite.size:
            lo, hi = np.percentile(finite, [1, 99])
            if hi > lo:
                kw.update(vmin=float(lo), vmax=float(hi))
        if not finite.size:
            im = ax.imshow(np.zeros_like(data), origin="lower", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
            ax.text(0.5, 0.5, "NO ELIGIBLE DATA\nCheck applied_mask, saturation_mask,\nand minimum_signal_dn", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, bbox=dict(facecolor="white", alpha=.9))
        else:
            im = ax.imshow(np.ma.masked_invalid(data), origin="lower", interpolation="nearest", cmap=cmap, **kw)
            fig.colorbar(im, ax=ax, label=label, shrink=0.82)
        ax.set(title=title, xlabel="Column", ylabel="Row")

    display(axes[0, 0], mean_frac, median_label, "%", cmap="coolwarm", center_zero=True)
    display(axes[0, 1], max_abs_dn, "Maximum |corrected - raw| (pre-saturation)", "DN", cmap="magma")
    display(axes[1, 0], n_good, "Usable reads for ramp fitting", "Number of reads", vmax=nreads-skip_initial_reads)
    display(axes[1, 1], n_lost, "Finite reads rejected by ramp-quality mask", "Number of reads", vmax=nreads-skip_initial_reads)
    fig.suptitle(f"Linearity correction and ramp-quality diagnostics ({nreads} reads)", fontsize=15)

    if save_prefix is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        prefix = Path(str(save_prefix)).stem
        path = destination / f"{prefix}_linearity_saturation_maps.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved diagnostic: {path}")
    if show:
        plt.show()
    return fig, axes


def plot_linearity_correction_diagnostics(
    raw_cube,
    corrected_cube,
    saturation_mask,
    applied_mask,
    output_dir="plots",
    save_prefix="linearity",
    skip_initial_reads=2,
    minimum_signal_dn=0.0,
    show=False,
):
    """
    Three-panel diagnostic for applied linearity correction.

    Panels
    ------
    1. Mean fractional correction [%]
    2. Maximum absolute correction [DN]
    3. Number of corrected reads per pixel

    Does not require a ramp-quality mask.

    Fractional correction is calculated relative to the
    pedestal-subtracted signal estimated from the first
    two reads. The diagnostic excludes saturated reads.

    Parameters
    ----------
    raw_cube, corrected_cube : ndarray
        Input and linearity-corrected cubes (N, H, W).
    saturation_mask : ndarray
        Boolean saturation mask (N, H, W).
    applied_mask : ndarray
        Boolean map (H, W) indicating pixels where
        linearity coefficients were applied.
    output_dir : str or Path
        Directory for the saved figure.
    save_prefix : str
        Output filename prefix.
    skip_initial_reads : int
        Number of initial reads excluded from diagnostics.
    minimum_signal_dn : float
        Minimum absolute pedestal-subtracted signal used
        for fractional correction.
    show : bool
        Display figure interactively.

    Returns
    -------
    fig, axes
        Matplotlib figure and three axes.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from matplotlib.colors import TwoSlopeNorm

    raw = np.asarray(raw_cube)
    corrected = np.asarray(corrected_cube)

    if raw.ndim != 3 or corrected.shape != raw.shape:
        raise ValueError(
            "raw_cube and corrected_cube must have "
            "identical 3D shapes."
        )

    nreads, ny, nx = raw.shape

    if not 0 <= skip_initial_reads < nreads:
        raise ValueError("Invalid skip_initial_reads.")

    sat = np.asarray(saturation_mask, dtype=bool)
    applied = np.asarray(applied_mask, dtype=bool)

    if sat.shape != raw.shape:
        raise ValueError("saturation_mask shape mismatch.")

    if applied.shape != (ny, nx):
        raise ValueError("applied_mask shape mismatch.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the same pedestal convention as the correction
    # routine: mean of the first two raw reads.
    pedestal = np.mean(
        raw[:2].astype(np.float64), axis=0
    )

    fraction_sum = np.zeros((ny, nx), dtype=np.float64)
    fraction_count = np.zeros((ny, nx), dtype=np.int32)

    maximum_correction = np.zeros(
        (ny, nx), dtype=np.float64
    )

    corrected_read_count = np.zeros(
        (ny, nx), dtype=np.int32
    )

    # Process one read at a time to avoid large temporary
    # arrays for a 2048 x 2048 detector.
    for i in range(skip_initial_reads, nreads):

        signal = raw[i].astype(np.float64) - pedestal
        delta = (
            corrected[i].astype(np.float64)
            - raw[i].astype(np.float64)
        )

        valid = (
            applied
            & ~sat[i]
            & np.isfinite(signal)
            & np.isfinite(delta)
        )

        corrected_read_count += valid.astype(np.int32)

        maximum_correction = np.maximum(
            maximum_correction,
            np.where(valid, np.abs(delta), 0.0)
        )

        valid_fraction = (
            valid
            & (np.abs(signal) > minimum_signal_dn)
            & (signal != 0)
        )

        fraction = np.zeros((ny, nx), dtype=np.float64)

        np.divide(
            100.0 * delta,
            signal,
            out=fraction,
            where=valid_fraction
        )

        fraction_sum += np.where(
            valid_fraction, fraction, 0.0
        )

        fraction_count += valid_fraction.astype(np.int32)

    mean_fraction = np.full(
        (ny, nx), np.nan, dtype=np.float64
    )

    np.divide(
        fraction_sum,
        fraction_count,
        out=mean_fraction,
        where=fraction_count > 0
    )

    maximum_correction = np.where(
        corrected_read_count > 0,
        maximum_correction,
        np.nan
    )

    # Robust color limits
    finite_fraction = mean_fraction[
        np.isfinite(mean_fraction)
    ]

    if finite_fraction.size:
        frac_limit = np.percentile(
            np.abs(finite_fraction), 99
        )
        frac_limit = max(frac_limit, 1e-12)
    else:
        frac_limit = 1.0

    finite_maximum = maximum_correction[
        np.isfinite(maximum_correction)
    ]

    if finite_maximum.size:
        max_limit = np.percentile(
            finite_maximum, 99
        )
        max_limit = max(max_limit, 1e-12)
    else:
        max_limit = 1.0

    fig, axes = plt.subplots(
        1, 3, figsize=(19, 6), constrained_layout=True
    )

    im0 = axes[0].imshow(
        mean_fraction,
        origin="lower",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(
            vmin=-frac_limit,
            vcenter=0.0,
            vmax=frac_limit
        ),
        interpolation="nearest"
    )

    axes[0].set_title("Mean fractional correction [%]")
    fig.colorbar(im0, ax=axes[0], label="%")

    im1 = axes[1].imshow(
        maximum_correction,
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=max_limit,
        interpolation="nearest"
    )

    axes[1].set_title(
        "Maximum |corrected - raw| (pre-saturation)"
    )
    fig.colorbar(im1, ax=axes[1], label="DN")

    im2 = axes[2].imshow(
        corrected_read_count,
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=nreads - skip_initial_reads,
        interpolation="nearest"
    )

    axes[2].set_title("Reads eligible for linearity correction")
    fig.colorbar(
        im2, ax=axes[2], label="Number of reads"
    )

    for ax in axes:
        ax.set_xlabel("Detector column")
        ax.set_ylabel("Detector row")

    fig.suptitle(
        f"Linearity correction diagnostics ({nreads} reads)",
        fontsize=15
    )

    output_file = (
        output_dir / f"{save_prefix}_linearity_diagnostics.png"
    )

    fig.savefig(output_file, dpi=150)

    print(f"Saved linearity diagnostic: {output_file}")
    print(f"Applied pixels: {np.count_nonzero(applied):,}")
    print(
        "Pixels with eligible reads: "
        f"{np.count_nonzero(corrected_read_count):,}"
    )

    if not show:
        plt.close(fig)
    else:
        plt.show()

    return fig, axes


def plot_good_reads_for_ramp_fitting(
    good_read_mask,
    output_dir="plots",
    save_prefix="ramp_quality",
    skip_initial_reads=2,
    show=False,
):
    """
    Plot the number of usable reads per detector pixel
    after applying the ramp-quality mask.

    Parameters
    ----------
    good_read_mask : ndarray of bool
        Boolean cube (N, H, W). True means usable.
    output_dir : str or Path
        Output directory.
    save_prefix : str
        Output filename prefix.
    skip_initial_reads : int
        Number of initial reads excluded.
    show : bool
        Display the figure.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    good = np.asarray(good_read_mask, dtype=bool)

    if good.ndim != 3:
        raise ValueError(
            "good_read_mask must be a 3D boolean cube."
        )

    nreads = good.shape[0]

    if not 0 <= skip_initial_reads < nreads:
        raise ValueError("Invalid skip_initial_reads.")

    usable_reads = np.count_nonzero(
        good[skip_initial_reads:], axis=0
    )

    maximum_reads = nreads - skip_initial_reads

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(9, 8), constrained_layout=True
    )

    im = ax.imshow(
        usable_reads,
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=maximum_reads,
        interpolation="nearest"
    )

    ax.set_title(
        f"Usable reads for ramp fitting "
        f"(maximum {maximum_reads})"
    )

    ax.set_xlabel("Detector column")
    ax.set_ylabel("Detector row")

    fig.colorbar(
        im,
        ax=ax,
        label="Number of usable reads"
    )

    output_file = (
        output_dir / f"{save_prefix}_good_reads.png"
    )

    fig.savefig(output_file, dpi=150)

    print(f"Saved good-read map: {output_file}")
    print(
        f"Median usable reads: "
        f"{np.median(usable_reads):.1f}"
    )
    print(
        f"Pixels with zero usable reads: "
        f"{np.count_nonzero(usable_reads == 0):,}"
    )

    if not show:
        plt.close(fig)
    else:
        plt.show()

    return fig, ax

def plot_extracted_cube_diagnostic(
    cube, output_directory, *, wavelength=None, combine="median",
    box_size=5, detection_sigma=4.0, min_component_pixels=3,
    persistence_fraction=0.15, dilation_iterations=1,
    spectral_sigma=6.0, save_prefix="extracted_cube", show=False,
):
    """Plot collapsed IFS image and robust aperture-median spectrum.

    Assumes cube shape (n_wave, ny, nx), with no spectral reordering.
    Wavelength, if supplied, must be a 1D array of length n_wave.
    Returns (fig, axes, info).  info includes source center and aperture.

    Source finding uses robust channel-by-channel spatial contrast and
    wavelength persistence, then connected-component filtering. The
    spectrum is the median of valid aperture spaxels at each channel;
    strong spatial outliers are rejected separately in each channel.
    """
    data = np.asarray(cube)
    if data.ndim != 3:
        raise ValueError("cube must have shape (n_wave, ny, nx)")
    nw, ny, nx = data.shape
    if nw < 3 or ny < 3 or nx < 3:
        raise ValueError("cube dimensions must each be at least 3")
    if combine not in ("median", "sum"):
        raise ValueError("combine must be 'median' or 'sum'")
    if box_size < 3 or box_size % 2 != 1:
        raise ValueError("box_size must be odd and >= 3")
    if wavelength is None:
        wave = np.arange(nw)
        xlabel = "Wavelength channel"
    else:
        wave = np.asarray(wavelength, dtype=float)
        if wave.shape != (nw,) or not np.all(np.isfinite(wave)):
            raise ValueError("wavelength must be finite and length n_wave")
        xlabel = "Wavelength (input units)"

    # A channel must contain enough valid spaxels to contribute to detection.
    finite = np.isfinite(data)
    channel_good = finite.sum(axis=(1, 2)) >= max(9, int(0.1 * ny * nx))
    if not channel_good.any():
        raise ValueError("Insufficient finite cube samples")

    # Per-channel robust spatial background and scale. Avoid full-cube
    # temporary allocations by accumulating a 2D persistence map.
    persistence = np.zeros((ny, nx), dtype=np.int32)
    for k in np.flatnonzero(channel_good):
        plane = np.asarray(data[k], dtype=float)
        values = plane[np.isfinite(plane)]
        center = np.median(values)
        sigma = 1.4826 * np.median(np.abs(values - center))
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        # Smooth each channel to suppress isolated single-spaxel hits.
        filled = np.where(np.isfinite(plane), plane, center)
        smooth = median_filter(filled, size=3, mode="nearest")
        persistence += (smooth > center + detection_sigma * sigma)

    required = max(2, int(np.ceil(persistence_fraction * channel_good.sum())))
    detected = persistence >= required
    components, ncomponents = label(detected)
    candidates = []
    for cid in range(1, ncomponents + 1):
        mask = components == cid
        count = int(mask.sum())
        if count >= min_component_pixels:
            candidates.append((int(persistence[mask].sum()), count, cid))
    if not candidates:
        raise ValueError(
            "No persistent spatially extended source detected. "
            "Try lowering detection_sigma/persistence_fraction, or inspect "
            "the cube and bad-spaxel mask; no peak was selected blindly."
        )
    _, _, best_id = max(candidates)
    component = components == best_id
    yy, xx = np.nonzero(component)
    weights = persistence[component].astype(float)
    cy = int(np.rint(np.average(yy, weights=weights)))
    cx = int(np.rint(np.average(xx, weights=weights)))

    # Restrict extraction to a compact box, dilating the persistent source
    # within it. This avoids pulling in unrelated nearby bright regions.
    half = box_size // 2
    y0, y1 = max(0, cy-half), min(ny, cy+half+1)
    x0, x1 = max(0, cx-half), min(nx, cx+half+1)
    box = np.zeros((ny, nx), dtype=bool)
    box[y0:y1, x0:x1] = True
    aperture = binary_dilation(component, iterations=dilation_iterations) & box
    if aperture.sum() < 3:
        aperture = box

    # Collapse the cube for the display only (sum is not background-subtracted).
    with np.errstate(invalid="ignore"):
        if combine == "median":
            collapsed = np.nanmedian(data, axis=0)
        else:
            collapsed = np.nansum(data, axis=0).astype(float)
            collapsed[finite.sum(axis=0) == 0] = np.nan

    spectrum = np.full(nw, np.nan)
    n_used = np.zeros(nw, dtype=int)
    for k in range(nw):
        vals = np.asarray(data[k][aperture], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        med = np.median(vals)
        mad = 1.4826 * np.median(np.abs(vals-med))
        if mad > 0 and vals.size >= 5:
            vals = vals[np.abs(vals-med) <= spectral_sigma*mad]
        if vals.size:
            spectrum[k] = np.median(vals)
            n_used[k] = vals.size

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    valid_image = collapsed[np.isfinite(collapsed)]
    vmin, vmax = np.percentile(valid_image, [2, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1
    im = axes[0].imshow(collapsed, origin="lower", cmap="viridis",
                        vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].contour(aperture.astype(float), levels=[0.5], colors=["red"],
                    linewidths=1.5)
    axes[0].plot(cx, cy, "+", color="white", markersize=12, mew=2)
    axes[0].set(title=f"Wavelength-{combine} image", xlabel="X spaxel",
                ylabel="Y spaxel")
    fig.colorbar(im, ax=axes[0], label="Flux (cube units)")
    axes[1].plot(wave, spectrum, color="black", lw=1.1)
    axes[1].set(title=f"Aperture-median spectrum ({int(aperture.sum())} spaxels)",
                xlabel=xlabel, ylabel="Median flux (cube units)")
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"Extracted IFS cube diagnostic | center=({cy}, {cx})")
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{save_prefix}_cube_diagnostic.png"
    fig.savefig(path, dpi=160)
    if show:
        plt.show()
    info = dict(center_row=cy, center_col=cx, aperture_mask=aperture,
                spectrum=spectrum, wavelength=wave, valid_spaxels=n_used,
                persistence_map=persistence, filename=str(path))
    if not show:
        plt.close(fig)
    return fig, axes, info

