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
from scalesdrp.core.scales_pkg_resources import get_resource_path
import time
from scipy import sparse
from scipy.ndimage import median_filter, binary_dilation, label
######### bpm correction ############################
# ============================================================
# DQ BIT DEFINITIONS
# ============================================================
DQ_BITS = {
    "NONFINITE":        1,
    "UNSTABLE":         2,
    "HOT":              4,
    "COLD":             8,
    "LOW_QE":           16,
    "HIGH_RESPONSE":    32,
    "SPATIAL_OUTLIER":  64,
    "ADJ_HOT":          128,
    "ADJ_LOW_QE":       256,
    "ADJ_OPEN":         512,
    "CLUSTER":          1024,
    "REFERENCE_PIXEL":  2048,
}


def robust_sigma_from_mad(values, floor=1e-6):
    '''
    This estimates a robust standard deviation using Median Absolute Deviation.
    '''
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return floor

    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    #Converts MAD to Gaussian-equivalent sigma
    return max(1.4826 * mad, floor)


def add_bit(dq_map, mask, bit_name):
    #This adds a DQ flag to selected pixels.
    dq_map[mask] |= DQ_BITS[bit_name]


def make_reference_pixel_mask(shape, ref_width=4):
    #Creates a reference-pixel mask for the 4-pixel detector border.
    ny, nx = shape

    ref_mask = np.zeros((ny, nx), dtype=bool)
    ref_mask[:, :ref_width] = True
    ref_mask[:, nx-ref_width:] = True
    ref_mask[:ref_width, :] = True
    ref_mask[ny-ref_width:, :] = True

    science_mask = ~ref_mask
    return ref_mask, science_mask


def expand_neighbors(mask, science_mask=None, radius=1):
    #This flags neighbors around bad pixels. radius 1==> 3x3 box
    if radius <= 0:
        expanded = mask.copy()
    else:
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        expanded = binary_dilation(mask, structure=structure)

    adjacent = expanded & (~mask)

    if science_mask is not None:
        adjacent &= science_mask
    #return neighbour only mask
    return adjacent


def detect_clusters(mask, min_cluster_size=6, science_mask=None):
    #detect connected groups on bad pixels
    work = mask.copy()

    if science_mask is not None:
        work &= science_mask
    #define 8 connected neighbours
    structure = np.ones((3, 3), dtype=int)
    labels, nlab = label(work, structure=structure)

    cluster_mask = np.zeros_like(mask, dtype=bool)
    #loop through each connected component
    for lab in range(1, nlab + 1):
        comp = labels == lab
        #if the connected component is large enough
        if np.sum(comp) >= min_cluster_size:
            cluster_mask |= comp

    return cluster_mask


def save_category_maps(output_dir, prefix, masks, dq_map, final_bpm, products=None):
    os.makedirs(output_dir, exist_ok=True)

    fits.PrimaryHDU(final_bpm.astype(np.uint8)).writeto(
        os.path.join(output_dir, f"{prefix}_bpm_bool.fits"),
        overwrite=True,
    )

    fits.PrimaryHDU(dq_map.astype(np.uint16)).writeto(
        os.path.join(output_dir, f"{prefix}_dq.fits"),
        overwrite=True,
    )

    for name, mask in masks.items():
        fits.PrimaryHDU(mask.astype(np.uint8)).writeto(
            os.path.join(output_dir, f"{prefix}_{name}.fits"),
            overwrite=True,
        )

    if products is not None:
        for name, arr in products.items():
            if arr is None:
                continue
            fits.PrimaryHDU(arr).writeto(
                os.path.join(output_dir, f"{prefix}_{name}.fits"),
                overwrite=True,
            )

# ============================================================

def remove_frame_common_mode(
    stack,
    stats_mask,
    remove_global=True,
    remove_rows=True,
    remove_cols=False,
):
    """
    This one used to estimate unstable pixels.
    Remove frame-to-frame common-mode structure before estimating
    temporal instability.

    This removes frmae to frame structure before measuring temporal instability
    like the cross hatching stripes of the imager.

    Parameters
    ----------
    stack : ndarray
        Shape = (n_frames, ny, nx)
    stats_mask : 2D bool
        True where pixels should be used for statistics.
    remove_global : bool
        Subtract one global median per frame.
    remove_rows : bool
        Subtract one row median per frame.
    remove_cols : bool
        Subtract one column median per frame.

    Returns
    -------
    clean_stack : ndarray
        Common-mode corrected stack.
    """
    clean = np.asarray(stack, dtype=np.float32).copy()
    n_frames, ny, nx = clean.shape

    for k in range(n_frames):
        frame = clean[k].copy()

        good = stats_mask & np.isfinite(frame)

        if remove_global and np.any(good):
            frame -= np.nanmedian(frame[good])

        if remove_rows:
            for y in range(ny):
                row_good = stats_mask[y, :] & np.isfinite(frame[y, :])
                if np.any(row_good):
                    frame[y, :] -= np.nanmedian(frame[y, row_good])

        if remove_cols:
            for x in range(nx):
                col_good = stats_mask[:, x] & np.isfinite(frame[:, x])
                if np.any(col_good):
                    frame[:, x] -= np.nanmedian(frame[col_good, x])

        clean[k] = frame
    #return common mode corrected stack
    return clean


def detect_unstable_pixels_from_temporal_scatter(
    stack,
    stats_mask,
    temporal_sigma_thresh=10.0,
    mad_floor=1e-6,
    remove_global=True,
    remove_rows=True,
    remove_cols=False,
):
    """
    Detect temporally unstable pixels using excess temporal scatter.

    Old method:
        Count how many frames a pixel deviates from its own temporal median.

    Problem:
        Common-mode row/stripe variations can look like unstable pixels.

    New method:
        1. Remove frame-level global/row/column structure.
        2. Compute per-pixel temporal scatter.
        3. Compare each pixel's temporal scatter against the detector population.

    This detects pixels that are truly noisy/flickering relative to other pixels.
    """
    #clean the common mode
    clean_stack = remove_frame_common_mode(
        stack,
        stats_mask=stats_mask,
        remove_global=remove_global,
        remove_rows=remove_rows,
        remove_cols=remove_cols,
    )
    #for each pixel, calaculate the temporal MAD across the frames
    temporal_median = np.nanmedian(clean_stack, axis=0)

    temporal_mad = np.nanmedian(
        np.abs(clean_stack - temporal_median[None, :, :]),
        axis=0,
    )

    temporal_sigma_map = 1.4826 * temporal_mad

    valid = np.isfinite(temporal_sigma_map) & stats_mask

    unstable = np.zeros_like(stats_mask, dtype=bool)

    if not np.any(valid):
        return unstable, temporal_sigma_map, np.nan
    #median temporal noise level
    population_median = np.nanmedian(temporal_sigma_map[valid])
    #scatter of the temporal noise values
    population_sigma = robust_sigma_from_mad(
        temporal_sigma_map[valid],
        floor=mad_floor,
    )
    #define unstable threshold
    threshold = population_median + temporal_sigma_thresh * population_sigma

    unstable = temporal_sigma_map > threshold
    unstable &= stats_mask

    print(f"Temporal scatter median: {population_median:.6g}")
    print(f"Temporal scatter sigma:  {population_sigma:.6g}")
    print(f"Temporal unstable thresh: {threshold:.6g}")
    #Returns unstable mask, temporal sigma map, and threshold
    return unstable, temporal_sigma_map, threshold


# ============================================================

def generate_bpm_classified_v4(
    image_stack,
    stack_name="Image Stack",
    mode="dark",

    # reference pixels
    ref_width=4,
    flag_reference_pixels=True,
    exclude_reference_from_stats=True,

    # temporal instability
    temporal_sigma_thresh=10.0,
    temporal_remove_global=True,
    temporal_remove_rows=True,
    temporal_remove_cols=False,

    # spatial local outlier
    spatial_sigma_thresh=8.0,
    spatial_kernel_size=7,
    min_frames_bad_spatial=4,

    # dark hot/cold
    dark_hot_sigma=8.0,
    dark_cold_sigma=8.0,

    # flat response / QE
    normalize_flats=True,
    flat_norm_kernel_size=31,
    flat_low_response_thresh=0.4,
    flat_high_response_thresh=1.5,

    # neighbor and cluster logic
    neighbor_radius=0,
    cluster_min_size=6,

    # general
    mad_floor_fraction=0.05,
):
    """
    Generate classified detector BPM from dark & flat slope-image stacks.

    Categories
    ----------
    NONFINITE
        NaN/Inf in calibration frames.

    UNSTABLE
        Excess temporal scatter after removing frame-level common-mode,
        row, and optionally column structure.

    HOT
        Dark-stack only. Persistent high dark current.

    COLD
        Dark-stack only. Persistent low dark signal.

    LOW_QE
        Flat-stack only. Low normalized flat response.

    HIGH_RESPONSE
        Flat-stack only. High normalized flat response / open-like pixel.

    SPATIAL_OUTLIER
        Persistent local spatial outlier across frames.

    ADJ_*
        Neighbor pixels around selected defect classes.

    CLUSTER
        Connected group of core bad pixels.

    REFERENCE_PIXEL
        Four-pixel detector reference border.

    Returns
    -------
    final_bpm : 2D bool
    dq_map : 2D uint16
    masks : dict
    products : dict
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

    print(f"\n=== Generating classified BPM for: {stack_name} ===")
    print(f"Mode: {mode}")
    print(f"Stack shape: {stack.shape}")

    #create reference mask and science mask
    ref_mask, science_mask = make_reference_pixel_mask((ny, nx), ref_width=ref_width)

    stats_mask = science_mask.copy() if exclude_reference_from_stats else np.ones((ny, nx), dtype=bool)

    dq_map = np.zeros((ny, nx), dtype=np.uint16)

    masks = {
        "nonfinite": np.zeros((ny, nx), dtype=bool),
        "unstable": np.zeros((ny, nx), dtype=bool),
        "hot": np.zeros((ny, nx), dtype=bool),
        "cold": np.zeros((ny, nx), dtype=bool),
        "low_qe": np.zeros((ny, nx), dtype=bool),
        "high_response": np.zeros((ny, nx), dtype=bool),
        "spatial_outlier": np.zeros((ny, nx), dtype=bool),
        "adj_hot": np.zeros((ny, nx), dtype=bool),
        "adj_low_qe": np.zeros((ny, nx), dtype=bool),
        "adj_open": np.zeros((ny, nx), dtype=bool),
        "cluster": np.zeros((ny, nx), dtype=bool),
        "reference_pixel": ref_mask.copy(),
        "science_pixel": science_mask.copy(),
    }

    if flag_reference_pixels:
        add_bit(dq_map, ref_mask, "REFERENCE_PIXEL")

    # 0. Non-finite pixels
    # --------------------------------------------------
    nonfinite_any = np.any(~np.isfinite(stack), axis=0)
    nonfinite_any &= stats_mask

    masks["nonfinite"] = nonfinite_any
    add_bit(dq_map, nonfinite_any, "NONFINITE")

    print(f"Reference pixels: {np.sum(ref_mask)}")
    print(f"Science pixels:   {np.sum(science_mask)}")
    print(f"Any-frame non-finite science pixels: {np.sum(nonfinite_any)}")


    # 1. Median image
    stack_median = np.nanmedian(stack, axis=0)

    # 2. NEW temporal instability
    start = time.time()

    unstable, temporal_sigma_map, temporal_unstable_threshold = (
        detect_unstable_pixels_from_temporal_scatter(
            stack,
            stats_mask=stats_mask,
            temporal_sigma_thresh=temporal_sigma_thresh,
            remove_global=temporal_remove_global,
            remove_rows=temporal_remove_rows,
            remove_cols=temporal_remove_cols,
        )
    )

    masks["unstable"] = unstable
    add_bit(dq_map, unstable, "UNSTABLE")

    print(f"Unstable pixels: {np.sum(unstable)} ({time.time() - start:.2f}s)")

    # --------------------------------------------------
    # 3. Dark-specific hot/cold
    # --------------------------------------------------
    dark_level_image = None

    if mode == "dark":
        finite = np.isfinite(stack_median) & stats_mask
        #global dark level
        global_dark_med = np.nanmedian(stack_median[finite])
        #robust scatter of dark levels
        global_dark_sigma = robust_sigma_from_mad(stack_median[finite])
        #flags hot pixels
        hot = stack_median > (
            global_dark_med + dark_hot_sigma * global_dark_sigma
        )
        #flags cold pixels
        cold = stack_median < (
            global_dark_med - dark_cold_sigma * global_dark_sigma
        )

        hot &= finite
        cold &= finite

        masks["hot"] = hot
        masks["cold"] = cold

        add_bit(dq_map, hot, "HOT")
        add_bit(dq_map, cold, "COLD")

        dark_level_image = stack_median

        print(f"Dark global median: {global_dark_med:.6g}")
        print(f"Dark robust sigma:  {global_dark_sigma:.6g}")
        print(f"Hot pixels:         {np.sum(hot)}")
        print(f"Cold pixels:        {np.sum(cold)}")

    # --------------------------------------------------
    # 4. Flat-specific low-QE/high-response
    # --------------------------------------------------
    flat_response = None

    if mode == "flat":
        flat_med = stack_median.copy()
        finite = np.isfinite(flat_med) & stats_mask

        fill_value = np.nanmedian(flat_med[finite]) if np.any(finite) else 1.0

        flat_fill = flat_med.copy()
        flat_fill[~finite] = fill_value
        #normalize the flat with the defined kernal size
        if normalize_flats:
            illum = median_filter(
                flat_fill,
                size=flat_norm_kernel_size,
                mode="reflect",
            )
            #prevents division by 0
            illum[~np.isfinite(illum) | (illum <= 1e-6)] = 1e-6
            flat_response = flat_fill / illum
        else:
            norm = np.nanmedian(flat_fill[finite]) if np.any(finite) else 1.0
            if norm <= 0:
                norm = 1.0
            flat_response = flat_fill / norm
        #low and high QE mask cutoff
        low_qe = flat_response < flat_low_response_thresh
        high_response = flat_response > flat_high_response_thresh

        low_qe &= finite
        high_response &= finite

        masks["low_qe"] = low_qe
        masks["high_response"] = high_response

        add_bit(dq_map, low_qe, "LOW_QE")
        add_bit(dq_map, high_response, "HIGH_RESPONSE")

        print(f"Flat low-QE pixels:        {np.sum(low_qe)}")
        print(f"Flat high-response pixels: {np.sum(high_response)}")
        print(f"Flat response median:      {np.nanmedian(flat_response[finite]):.6g}")

    # --------------------------------------------------
    # 5. Spatial persistent outliers
    # --------------------------------------------------
    start = time.time()

    spatial_count = np.zeros((ny, nx), dtype=np.uint16)
    #loop over each frame
    for i in range(n_frames):
        frame = stack[i].copy()

        finite = np.isfinite(frame) & stats_mask
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
        #local median of the image
        local_med = median_filter(
            test_frame,
            size=spatial_kernel_size,
            mode="reflect",
        )

        abs_dev = np.abs(test_frame - local_med)
        #local robust scatter
        local_mad = median_filter(
            abs_dev,
            size=spatial_kernel_size,
            mode="reflect",
        )

        local_sigma = 1.4826 * local_mad

        finite_local_sigma = local_sigma[
            np.isfinite(local_sigma) & (local_sigma > 0) & stats_mask
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
        #spatial cutoff applied
        spatial_outlier = abs_dev > (spatial_sigma_thresh * local_sigma)
        spatial_outlier &= stats_mask

        spatial_count += spatial_outlier.astype(np.uint16)

    spatial_bpm = spatial_count >= min_frames_bad_spatial
    spatial_bpm &= stats_mask

    masks["spatial_outlier"] = spatial_bpm
    add_bit(dq_map, spatial_bpm, "SPATIAL_OUTLIER")

    print(f"Spatial persistent outliers: {np.sum(spatial_bpm)} ({time.time() - start:.2f}s)")

    # --------------------------------------------------
    # 6. Find neighbors of each catogery
    # --------------------------------------------------
    adj_hot = expand_neighbors(
        masks["hot"],
        science_mask=science_mask,
        radius=neighbor_radius,
    )

    adj_low_qe = expand_neighbors(
        masks["low_qe"],
        science_mask=science_mask,
        radius=neighbor_radius,
    )

    adj_open = expand_neighbors(
        masks["high_response"],
        science_mask=science_mask,
        radius=neighbor_radius,
    )

    masks["adj_hot"] = adj_hot
    masks["adj_low_qe"] = adj_low_qe
    masks["adj_open"] = adj_open

    add_bit(dq_map, adj_hot, "ADJ_HOT")
    add_bit(dq_map, adj_low_qe, "ADJ_LOW_QE")
    add_bit(dq_map, adj_open, "ADJ_OPEN")

    print(f"Adjacent-to-hot pixels:       {np.sum(adj_hot)}")
    print(f"Adjacent-to-low-QE pixels:    {np.sum(adj_low_qe)}")
    print(f"Adjacent-to-open-like pixels: {np.sum(adj_open)}")

    # --------------------------------------------------
    # 7. Cluster detection
    # --------------------------------------------------
    #combine all primary bad pixels
    core_defect_mask = (
        masks["nonfinite"]
        | masks["unstable"]
        | masks["hot"]
        | masks["cold"]
        | masks["low_qe"]
        | masks["high_response"]
        | masks["spatial_outlier"]
    )

    cluster_mask = detect_clusters(
        core_defect_mask,
        min_cluster_size=cluster_min_size,
        science_mask=science_mask,
    )

    masks["cluster"] = cluster_mask
    add_bit(dq_map, cluster_mask, "CLUSTER")

    print(f"Cluster pixels: {np.sum(cluster_mask)}")

    # --------------------------------------------------
    # 8. Final summary
    # --------------------------------------------------
    final_bpm = dq_map != 0

    print(f"\nSummary for {stack_name}:")
    for key, mask in masks.items():
        print(f"  {key:18s}: {np.sum(mask):8d}")

    print(f"  {'TOTAL':18s}: {np.sum(final_bpm):8d}")
    print(f"  Fraction flagged: {100.0 * np.sum(final_bpm) / final_bpm.size:.4f}%")

    products = {
        "stack_median": stack_median,
        "temporal_sigma_map": temporal_sigma_map,
        "temporal_unstable_threshold": np.array([temporal_unstable_threshold], dtype=np.float32),
        "spatial_count": spatial_count,
        "flat_response": flat_response,
        "dark_level_image": dark_level_image,
    }

    return final_bpm, dq_map, masks, products

############# R.matrix based .npz FILE bpm correction #########################
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



################ dynamic mask not implemented yet ##################################################
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
