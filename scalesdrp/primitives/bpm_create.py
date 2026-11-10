import time
import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter, binary_dilation, label
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation
import json

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
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return floor

    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    return max(1.4826 * mad, floor)


def add_bit(dq_map, mask, bit_name):
    dq_map[mask] |= DQ_BITS[bit_name]


def make_reference_pixel_mask(shape, ref_width=4):
    ny, nx = shape

    ref_mask = np.zeros((ny, nx), dtype=bool)
    ref_mask[:, :ref_width] = True
    ref_mask[:, nx-ref_width:] = True
    ref_mask[:ref_width, :] = True
    ref_mask[ny-ref_width:, :] = True

    science_mask = ~ref_mask
    return ref_mask, science_mask


def expand_neighbors(mask, science_mask=None, radius=1):
    if radius <= 0:
        expanded = mask.copy()
    else:
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        expanded = binary_dilation(mask, structure=structure)

    adjacent = expanded & (~mask)

    if science_mask is not None:
        adjacent &= science_mask

    return adjacent


def detect_clusters(mask, min_cluster_size=6, science_mask=None):
    work = mask.copy()

    if science_mask is not None:
        work &= science_mask

    structure = np.ones((3, 3), dtype=int)
    labels, nlab = label(work, structure=structure)

    cluster_mask = np.zeros_like(mask, dtype=bool)

    for lab in range(1, nlab + 1):
        comp = labels == lab
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


def remove_frame_common_mode(
    stack,
    stats_mask,
    remove_global=True,
    remove_rows=True,
    remove_cols=False,
):
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

    return clean


def detect_unstable_pixels_from_temporal_scatter(
    stack,
    stats_mask,
    temporal_sigma_thresh=10.0,
    temporal_sigma_abs_cut=None,
    mad_floor=1e-6,
    remove_global=True,
    remove_rows=True,
    remove_cols=False,
):
    """
    Detect unstable pixels from temporal scatter.

    If temporal_sigma_abs_cut is supplied, use:
        unstable = temporal_sigma_map > temporal_sigma_abs_cut

    Otherwise use:
        unstable = temporal_sigma_map >
                   median(temporal_sigma_map) + temporal_sigma_thresh * robust_sigma
    """
    clean_stack = remove_frame_common_mode(
        stack,
        stats_mask=stats_mask,
        remove_global=remove_global,
        remove_rows=remove_rows,
        remove_cols=remove_cols,
    )

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

    if temporal_sigma_abs_cut is not None:
        threshold = temporal_sigma_abs_cut
    else:
        population_median = np.nanmedian(temporal_sigma_map[valid])
        population_sigma = robust_sigma_from_mad(
            temporal_sigma_map[valid],
            floor=mad_floor,
        )
        threshold = population_median + temporal_sigma_thresh * population_sigma

    unstable = temporal_sigma_map > threshold
    unstable &= stats_mask

    print(f"Temporal unstable threshold: {threshold:.6g}")
    print(f"Unstable pixels:             {np.sum(unstable)}")

    return unstable, temporal_sigma_map, threshold


def row_correct_detection(image, valid, multiplicative=False, min_row_pixels=64):
    """Correct coherent whole-row offsets; never alter input calibration data.

    For multiplicative flat responses, divide by the row median. For
    additive dark residuals, subtract the row median. A row is left
    unchanged if insufficient finite science pixels exist.
    """
    out = np.asarray(image, dtype=np.float32).copy()
    row_levels = np.zeros(out.shape[0], dtype=np.float32)
    if multiplicative:
        row_levels[:] = 1.0
    for y in range(out.shape[0]):
        ok = valid[y] & np.isfinite(out[y])
        if np.count_nonzero(ok) < min_row_pixels:
            continue
        level = float(np.median(out[y, ok]))
        if multiplicative:
            if np.isfinite(level) and level > 0:
                out[y] /= level
                row_levels[y] = level
        else:
            if np.isfinite(level):
                out[y] -= level
                row_levels[y] = level
    return out, row_levels


def conservative_row_filter(original, corrected, valid, min_fraction=0.20):
    """Suppress only candidates in densely flagged rows cleared by correction.

    This deliberately errs toward retaining uncertain defects. The row
    fraction is a heuristic, not a proof of electronic pickup. Compare
    the removed-candidate FITS diagnostic before accepting the output.
    """
    row_n = np.maximum(np.sum(valid, axis=1), 1)
    fraction = np.sum(original & valid, axis=1) / row_n
    coherent_rows = fraction >= min_fraction
    removed = original & ~corrected & coherent_rows[:, None] & valid
    return original & ~removed, removed


def generate_bpm(
    image_stack,
    stack_name="Image Stack",
    mode="dark",

    # reference pixels
    ref_width=4,
    flag_reference_pixels=True,
    exclude_reference_from_stats=True,

    # temporal instability
    temporal_sigma_thresh=10.0,
    temporal_sigma_abs_cut=None,
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
    dark_hot_abs_cut=None,
    dark_cold_abs_cut=None,
    use_local_dark_residual=True,
    dark_residual_kernel_size=51,

    # flat response / QE
    normalize_flats=True,
    flat_norm_kernel_size=31,
    flat_low_response_thresh=0.3,
    flat_high_response_thresh=1.4,

    # neighbor and cluster logic
    neighbor_radius=0,
    cluster_min_size=6,

    # general
    mad_floor_fraction=0.05,
    row_correction=True,
    row_suppression_fraction=0.20,
):
    """
    Generate classified detector BPM from dark or flat slope-image stacks.

    Main changes in v5
    ------------------
    1. UNSTABLE can use an absolute temporal-sigma cutoff.
    2. HOT/COLD can use absolute thresholds.
    3. HOT/COLD can be measured on local-background-subtracted dark image.
       This helps avoid flagging broad dark structure as bad pixels.
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

    if dark_residual_kernel_size % 2 == 0:
        dark_residual_kernel_size += 1

    print(f"\n=== Generating classified BPM for: {stack_name} ===")
    print(f"Mode: {mode}")
    print(f"Stack shape: {stack.shape}")

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

    # --------------------------------------------------
    # 0. Non-finite pixels
    # --------------------------------------------------
    nonfinite_any = np.any(~np.isfinite(stack), axis=0)
    nonfinite_any &= stats_mask

    masks["nonfinite"] = nonfinite_any
    add_bit(dq_map, nonfinite_any, "NONFINITE")

    print(f"Reference pixels: {np.sum(ref_mask)}")
    print(f"Science pixels:   {np.sum(science_mask)}")
    print(f"Non-finite science pixels: {np.sum(nonfinite_any)}")

    # --------------------------------------------------
    # 1. Median image
    # --------------------------------------------------
    stack_median = np.nanmedian(stack, axis=0)

    # --------------------------------------------------
    # 2. Temporal instability
    # --------------------------------------------------
    start = time.time()

    unstable, temporal_sigma_map, temporal_unstable_threshold = (
        detect_unstable_pixels_from_temporal_scatter(
            stack,
            stats_mask=stats_mask,
            temporal_sigma_thresh=temporal_sigma_thresh,
            temporal_sigma_abs_cut=temporal_sigma_abs_cut,
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
    dark_residual_image = None
    dark_background = None
    hot_threshold = None
    cold_threshold = None

    row_removed = {}
    row_levels = {}

    if mode == "dark":
        finite = np.isfinite(stack_median) & stats_mask

        dark_level_image = stack_median.copy()

        if use_local_dark_residual:
            fill_value = np.nanmedian(stack_median[finite]) if np.any(finite) else 0.0
            dark_fill = stack_median.copy()
            dark_fill[~finite] = fill_value

            dark_background = median_filter(
                dark_fill,
                size=dark_residual_kernel_size,
                mode="reflect",
            )

            dark_test_image = stack_median - dark_background
            dark_residual_image = dark_test_image.copy()
        else:
            dark_test_image = stack_median.copy()

        finite_test = np.isfinite(dark_test_image) & stats_mask

        global_dark_med = np.nanmedian(dark_test_image[finite_test])
        global_dark_sigma = robust_sigma_from_mad(dark_test_image[finite_test])

        if dark_hot_abs_cut is not None:
            hot_threshold = dark_hot_abs_cut
        else:
            hot_threshold = global_dark_med + dark_hot_sigma * global_dark_sigma

        if dark_cold_abs_cut is not None:
            cold_threshold = dark_cold_abs_cut
        else:
            cold_threshold = global_dark_med - dark_cold_sigma * global_dark_sigma

        hot = dark_test_image > hot_threshold
        cold = dark_test_image < cold_threshold
        if row_correction:
            corrected, levels = row_correct_detection(dark_test_image, finite_test)
            row_levels["dark"] = levels
            hot_corr = corrected > hot_threshold
            cold_corr = corrected < cold_threshold
            hot, row_removed["hot"] = conservative_row_filter(
                hot, hot_corr, finite_test, row_suppression_fraction)
            cold, row_removed["cold"] = conservative_row_filter(
                cold, cold_corr, finite_test, row_suppression_fraction)

        hot &= finite_test
        cold &= finite_test

        masks["hot"] = hot
        masks["cold"] = cold

        add_bit(dq_map, hot, "HOT")
        add_bit(dq_map, cold, "COLD")

        print(f"Dark test median:   {global_dark_med:.6g}")
        print(f"Dark robust sigma:  {global_dark_sigma:.6g}")
        print(f"Hot threshold:      {hot_threshold:.6g}")
        print(f"Cold threshold:     {cold_threshold:.6g}")
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

        low_qe = flat_response < flat_low_response_thresh
        high_response = flat_response > flat_high_response_thresh
        if row_correction:
            corrected, levels = row_correct_detection(
                flat_response, finite, multiplicative=True)
            row_levels["flat"] = levels
            low_qe, row_removed["low_qe"] = conservative_row_filter(
                low_qe, corrected < flat_low_response_thresh,
                finite, row_suppression_fraction)
            high_response, row_removed["high_response"] = conservative_row_filter(
                high_response, corrected > flat_high_response_thresh,
                finite, row_suppression_fraction)

        low_qe &= finite
        high_response &= finite

        masks["low_qe"] = low_qe
        masks["high_response"] = high_response

        add_bit(dq_map, low_qe, "LOW_QE")
        add_bit(dq_map, high_response, "HIGH_RESPONSE")

        print(f"Flat low-QE threshold:       {flat_low_response_thresh}")
        print(f"Flat high-response threshold:{flat_high_response_thresh}")
        print(f"Flat low-QE pixels:          {np.sum(low_qe)}")
        print(f"Flat high-response pixels:   {np.sum(high_response)}")
        print(f"Flat response median:        {np.nanmedian(flat_response[finite]):.6g}")

    # --------------------------------------------------
    # 5. Spatial persistent outliers
    # --------------------------------------------------
    start = time.time()

    spatial_count = np.zeros((ny, nx), dtype=np.uint16)
    spatial_corrected_count = np.zeros((ny, nx), dtype=np.uint16)

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
        elif mode == "dark" and use_local_dark_residual:
            local_bg = median_filter(
                frame_fill,
                size=dark_residual_kernel_size,
                mode="reflect",
            )
            test_frame = frame_fill - local_bg
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

        spatial_outlier = abs_dev > (spatial_sigma_thresh * local_sigma)
        spatial_outlier &= stats_mask

        spatial_count += spatial_outlier.astype(np.uint16)
        if row_correction:
            corrected, _ = row_correct_detection(
                test_frame, finite,
                multiplicative=(mode == "flat" and normalize_flats))
            corrected_med = median_filter(corrected, size=spatial_kernel_size, mode="reflect")
            corrected_dev = np.abs(corrected - corrected_med)
            corrected_mad = median_filter(corrected_dev, size=spatial_kernel_size, mode="reflect")
            corrected_sigma = 1.4826 * corrected_mad
            corrected_sigma[~np.isfinite(corrected_sigma) | (corrected_sigma <= 0)] = local_sigma_floor
            corrected_outlier = (corrected_dev > spatial_sigma_thresh * corrected_sigma) & finite
            spatial_corrected_count += corrected_outlier.astype(np.uint16)

    spatial_bpm = spatial_count >= min_frames_bad_spatial
    spatial_bpm &= stats_mask
    if row_correction:
        corrected_bpm = (spatial_corrected_count >= min_frames_bad_spatial) & stats_mask
        spatial_bpm, row_removed["spatial_outlier"] = conservative_row_filter(
            spatial_bpm, corrected_bpm, stats_mask, row_suppression_fraction)

    masks["spatial_outlier"] = spatial_bpm
    add_bit(dq_map, spatial_bpm, "SPATIAL_OUTLIER")

    print(f"Spatial persistent outliers: {np.sum(spatial_bpm)} ({time.time() - start:.2f}s)")

    # --------------------------------------------------
    # 6. Neighbor expansion
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
    # Final BPM contains science defects only; retain reference bit in DQ.
    final_bpm = (dq_map != 0) & science_mask

    print(f"\nSummary for {stack_name}:")
    for key, mask in masks.items():
        print(f"  {key:18s}: {np.sum(mask):8d}")

    print(f"  {'TOTAL':18s}: {np.sum(final_bpm):8d}")
    print(f"  Fraction flagged (science only): {100.0 * np.sum(final_bpm) / np.sum(science_mask):.4f}%")

    products = {
        "stack_median": stack_median,
        "temporal_sigma_map": temporal_sigma_map,
        "temporal_unstable_threshold": np.array([temporal_unstable_threshold], dtype=np.float32),
        "spatial_count": spatial_count,
        "spatial_corrected_count": spatial_corrected_count if row_correction else None,
        **{f"row_removed_{k}": v.astype(np.uint8) for k, v in row_removed.items()},
        **{f"row_level_{k}": v for k, v in row_levels.items()},
        "flat_response": flat_response,
        "dark_level_image": dark_level_image,
        "dark_background": dark_background,
        "dark_residual_image": dark_residual_image,
        "dark_hot_threshold": np.array([hot_threshold], dtype=np.float32) if hot_threshold is not None else None,
        "dark_cold_threshold": np.array([cold_threshold], dtype=np.float32) if cold_threshold is not None else None,
    }

    return final_bpm, dq_map, masks, products


def create_bpm_header(
    camera,
    mclock,
    dark_files,
    flat_files,
    dark_params,
    flat_params,
    base_header=None,
):
    """
    Create a FITS header containing BPM calibration provenance.
    """

    hdr = (
        base_header.copy()
        if base_header is not None
        else fits.Header()
    )

    hdr["CALTYPE"] = ("BPM", "Calibration product type")
    hdr["CAMERA"] = (str(camera), "Detector camera")
    hdr["MCLOCK"] = (str(mclock), "Detector clock mode")

    hdr["NDARK"] = (len(dark_files), "Number of dark slopes")
    hdr["NFLAT"] = (len(flat_files), "Number of flat slopes")
    hdr["NCOMBINE"] = (
        len(dark_files) + len(flat_files),
        "Total input slope images"
    )

    # ----------------------------------------
    # Dark detection parameters
    # ----------------------------------------
    hdr["HIERARCH BPM DARK SPATSIG"] = (
        dark_params["spatial_sigma_thresh"],
        "Dark spatial sigma threshold"
    )
    hdr["HIERARCH BPM DARK SPATKERN"] = (
        dark_params["spatial_kernel_size"],
        "Dark spatial kernel size"
    )
    hdr["HIERARCH BPM DARK MINFRM"] = (
        dark_params["min_frames_bad_spatial"],
        "Minimum bad spatial frames"
    )

    # ----------------------------------------
    # Flat detection parameters
    # ----------------------------------------
    hdr["HIERARCH BPM FLAT TEMPSIG"] = (
        flat_params["temporal_sigma_thresh"],
        "Flat temporal sigma threshold"
    )
    hdr["HIERARCH BPM FLAT SPATSIG"] = (
        flat_params["spatial_sigma_thresh"],
        "Flat spatial sigma threshold"
    )
    hdr["HIERARCH BPM FLAT SPATKERN"] = (
        flat_params["spatial_kernel_size"],
        "Flat spatial kernel size"
    )
    hdr["HIERARCH BPM FLAT MINFRM"] = (
        flat_params["min_frames_bad_spatial"],
        "Minimum bad spatial frames"
    )
    hdr["HIERARCH BPM FLAT NORMKERN"] = (
        flat_params["flat_norm_kernel_size"],
        "Flat normalization kernel size"
    )

    # ----------------------------------------
    # Row correction
    # ----------------------------------------
    hdr["HIERARCH BPM ROWCORR"] = (
        bool(dark_params["row_correction"]),
        "Row correction enabled"
    )
    hdr["HIERARCH BPM ROWFRAC"] = (
        dark_params["row_suppression_fraction"],
        "Row suppression fraction"
    )

    # ----------------------------------------
    # Input file provenance
    # ----------------------------------------
    hdr.add_history("BPM generated from DARK and FLATLAMP slopes")

    hdr.add_history("DARK INPUT FILES:")
    for filename in dark_files:
        hdr.add_history(f"DARK: {os.path.basename(filename)}")

    hdr.add_history("FLAT INPUT FILES:")
    for filename in flat_files:
        hdr.add_history(f"FLAT: {os.path.basename(filename)}")

    return hdr

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


##################plotting ##########################################
def plot_bpm_diagnostics(
    dark_products,
    flat_products,
    dark_masks,
    flat_masks,
    output_dir,
    prefix="bpm",
    ref_width=4,
    max_hist_samples=400_000,
    bins=180,
    save_pdf=False,
    flat_low_response_thresh=0.3,
    flat_high_response_thresh=1.4,
    dark_min_frames_bad_spatial=4,
    flat_min_frames_bad_spatial=3,
):
    """Plot the distributions and actual cutoffs used by ``generate_bpm``.

    Parameters
    ----------
    dark_products, flat_products : dict
        Fourth outputs of ``generate_bpm`` for dark and flat stacks.
    dark_masks, flat_masks : dict
        Third outputs of ``generate_bpm`` for dark and flat stacks.
    output_dir : str or path-like
        Directory in which to save the figure and JSON statistics.
    prefix : str
        Filename prefix, e.g. ``bpm_IFS_fast06``.
    ref_width : int
        Exclude reference border from all distributions.
    max_hist_samples : int
        Maximum regularly sampled finite science pixels per distribution;
        category counts always use the complete masks.
    bins : int
        Histogram bin count.
    save_pdf : bool
        Save a PDF copy in addition to the PNG.

    Returns
    -------
    result : dict
        Keys: ``figure``, ``png_path``, ``pdf_path``, ``json_path``,
        ``statistics``. Caller may ``plt.close(result['figure'])``.

    Notes
    -----
    The dark histogram uses ``dark_residual_image`` if present, otherwise
    ``dark_level_image``; these are the images tested for HOT/COLD. The flat
    histogram uses ``flat_response`` tested for LOW_QE/HIGH_RESPONSE.
    Histograms show *pre-row-suppression* measurement distributions, whereas
    mask counts show the final flags after any row suppression.
    """
    import json
    import matplotlib.pyplot as plt

    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if not isinstance(prefix, str) or not prefix or os.path.basename(prefix) != prefix:
        raise ValueError("prefix must be a nonempty filename prefix, not a path")

    reference = np.asarray(dark_masks["science_pixel"], dtype=bool)
    flat_science = np.asarray(flat_masks["science_pixel"], dtype=bool)
    if reference.shape != flat_science.shape:
        raise ValueError("Dark and flat masks must have matching dimensions")
    science = reference & flat_science
    if not np.any(science):
        raise ValueError("No science pixels available for diagnostic distributions")

    def scalar(products, key):
        val = products.get(key)
        if val is None:
            return None
        a = np.asarray(val).ravel()
        if a.size == 0 or not np.isfinite(a[0]):
            return None
        return float(a[0])

    def values(products, key, mask=science):
        arr = products.get(key)
        if arr is None:
            return np.array([], dtype=float)
        arr = np.asarray(arr)
        if arr.shape != mask.shape:
            return np.array([], dtype=float)
        data = arr[mask]
        data = data[np.isfinite(data)]
        if data.size > max_hist_samples:
            step = int(np.ceil(data.size / max_hist_samples))
            data = data[::step]
        return data.astype(np.float64, copy=False)

    def draw_hist(ax, data, title, xlabel, cuts=(), logy=True):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Sampled science pixels")
        if data.size:
            # Quantile zoom makes the population visible despite rare outliers.
            lo, hi = np.percentile(data, [0.2, 99.8])
            finite_cuts = [float(v) for v, _, _ in cuts if v is not None and np.isfinite(v)]
            if finite_cuts:
                lo = min(lo, min(finite_cuts))
                hi = max(hi, max(finite_cuts))
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            pad = (hi - lo) * 0.04
            lo, hi = lo - pad, hi + pad
            ax.hist(data, bins=bins, range=(lo, hi), color="steelblue", alpha=0.8)
            ax.set_xlim(lo, hi)
            if logy:
                ax.set_yscale("log")
            for val, label, color in cuts:
                if val is not None and np.isfinite(val):
                    ax.axvline(val, color=color, linewidth=1.7,
                               linestyle="--", label=f"{label}: {val:.4g}")
            if finite_cuts:
                ax.legend(fontsize=8, loc="best")
            ax.text(0.02, 0.97, f"N sampled={data.size:,}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.grid(alpha=0.2)

    dark_test = "dark_residual_image" if dark_products.get("dark_residual_image") is not None else "dark_level_image"
    dark_hot = scalar(dark_products, "dark_hot_threshold")
    dark_cold = scalar(dark_products, "dark_cold_threshold")
    flat_low = float(flat_low_response_thresh)
    flat_high = float(flat_high_response_thresh)

    # Flat-response cutoffs are not stored in products; pass the actual
    # generate_bpm parameters to this function when they differ from defaults.
    fig, axes = plt.subplots(3, 2, figsize=(14, 15), constrained_layout=True)
    draw_hist(axes[0, 0], values(dark_products, dark_test),
              "Dark signal: HOT / COLD", "Dark slope residual (DN)",
              [(dark_cold, "Cold cutoff", "royalblue"),
               (dark_hot, "Hot cutoff", "crimson")])
    draw_hist(axes[0, 1], values(flat_products, "flat_response"),
              "Flat response: LOW QE / HIGH RESPONSE", "Normalized flat response",
              [(flat_low, "Low-QE cutoff", "crimson"),
               (flat_high, "High-response cutoff", "darkorange")])
    draw_hist(axes[1, 0], values(dark_products, "temporal_sigma_map"),
              "Dark temporal scatter", "Temporal MAD × 1.4826 (DN)",
              [(scalar(dark_products, "temporal_unstable_threshold"), "Unstable cutoff", "crimson")])
    draw_hist(axes[1, 1], values(flat_products, "temporal_sigma_map"),
              "Flat temporal scatter", "Temporal MAD × 1.4826 (DN)",
              [(scalar(flat_products, "temporal_unstable_threshold"), "Unstable cutoff", "crimson")])
    draw_hist(axes[2, 0], values(dark_products, "spatial_count"),
              "Dark spatial-outlier persistence", "Number of frames flagged",
              [(dark_min_frames_bad_spatial - 0.5, "Persistence cutoff", "crimson")], logy=True)
    draw_hist(axes[2, 1], values(flat_products, "spatial_count"),
              "Flat spatial-outlier persistence", "Number of frames flagged",
              [(flat_min_frames_bad_spatial - 0.5, "Persistence cutoff", "crimson")], logy=True)

    statistics = {
        "science_pixels": int(science.sum()),
        "dark": {}, "flat": {},
        "thresholds": {
            "dark_hot": dark_hot, "dark_cold": dark_cold,
            "flat_low_qe": flat_low, "flat_high_response": flat_high,
            "dark_temporal": scalar(dark_products, "temporal_unstable_threshold"),
            "flat_temporal": scalar(flat_products, "temporal_unstable_threshold"),
            "dark_spatial_min_frames": int(dark_min_frames_bad_spatial),
            "flat_spatial_min_frames": int(flat_min_frames_bad_spatial),
        },
    }
    for mode, masks, products in (("dark", dark_masks, dark_products),
                                  ("flat", flat_masks, flat_products)):
        statistics[mode]["flag_counts"] = {
            key: int(np.count_nonzero(np.asarray(val, dtype=bool) & science))
            for key, val in masks.items()
            if np.asarray(val).shape == science.shape and key != "science_pixel"
        }
        statistics[mode]["row_removed_counts"] = {
            key.removeprefix("row_removed_"): int(np.count_nonzero(np.asarray(val) & science))
            for key, val in products.items()
            if key.startswith("row_removed_") and val is not None
            and np.asarray(val).shape == science.shape
        }

    fig.suptitle(f"BPM detection diagnostics — {prefix}", fontsize=15)
    png_path = os.path.join(output_dir, f"{prefix}_diagnostics.png")
    json_path = os.path.join(output_dir, f"{prefix}_diagnostics.json")
    fig.savefig(png_path, dpi=160)
    pdf_path = None
    if save_pdf:
        pdf_path = os.path.join(output_dir, f"{prefix}_diagnostics.pdf")
        fig.savefig(pdf_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2, allow_nan=False)
    return {"figure": fig, "png_path": png_path, "pdf_path": pdf_path,
            "json_path": json_path, "statistics": statistics}



CORE = ('nonfinite','unstable','hot','cold','low_qe','high_response','spatial_outlier')
CATEGORY = ('nonfinite','unstable','hot','cold','low_qe','high_response',
            'spatial_outlier','adj_hot','adj_low_qe','adj_open','cluster')

def plot_bpm_extra_diagnostics(
    dark_masks, flat_masks, dark_products, flat_products,
    output_dir, prefix='bpm', dark_dq=None, flat_dq=None,
    dpi=200, save_pdf=False, marker_radius=2, edge_size=256,
):
    """Save four BPM diagnostic figures and JSON statistics.

    Parameters
    ----------
    dark_masks, flat_masks : dict
        Third outputs of generate_bpm() for dark and flat data.
    dark_products, flat_products : dict
        Fourth outputs of generate_bpm() for dark and flat data.
    output_dir, prefix : str
        Output directory and filename prefix.
    dark_dq, flat_dq : 2D arrays, optional
        Second outputs of generate_bpm(), used for exact final flags.
    dpi : int
        Saved PNG resolution.
    save_pdf : bool
        Save vector/raster PDF counterparts of each figure.
    marker_radius : int
        Display-only binary dilation radius in pixels on full-detector
        mask panels. Set 0 to show actual one-pixel defects.
    edge_size : int
        Side length in detector pixels of each corner cutout.

    Returns
    -------
    dict
        Keys: figures, paths, json_path, statistics.
        The caller should close figures after use.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not prefix or os.path.basename(prefix) != prefix:
        raise ValueError('prefix must be a filename prefix, not a path')
    if int(marker_radius) != marker_radius or marker_radius < 0:
        raise ValueError('marker_radius must be a nonnegative integer')
    if int(edge_size) != edge_size or edge_size < 8:
        raise ValueError('edge_size must be an integer >= 8')

    shape = np.asarray(dark_masks['science_pixel']).shape
    if len(shape) != 2 or np.asarray(flat_masks['science_pixel']).shape != shape:
        raise ValueError('Dark and flat science masks must be 2D and have the same shape')
    science = (np.asarray(dark_masks['science_pixel'], bool) &
               np.asarray(flat_masks['science_pixel'], bool))
    reference = ~science
    nscience = int(science.sum())
    if not nscience:
        raise ValueError('No science pixels')

    category_keys = (
        'hot', 'cold', 'unstable', 'spatial_outlier', 'cluster',
        'low_qe', 'high_response', 'adj_hot', 'adj_low_qe', 'adj_open',
    )

    def m(masks, name):
        value = masks.get(name)
        if value is None:
            return np.zeros(shape, dtype=bool)
        arr = np.asarray(value, dtype=bool)
        if arr.shape != shape:
            raise ValueError(f'Mask {name} has shape {arr.shape}; expected {shape}')
        return arr

    def p(products, name):
        value = products.get(name)
        return np.asarray(value) if value is not None and np.shape(value) == shape else None

    def derived_dq(masks, dq):
        if dq is not None:
            arr = np.asarray(dq)
            if arr.shape != shape:
                raise ValueError('DQ dimensions differ')
            return (arr != 0) & science
        return np.logical_or.reduce([m(masks, k) for k in category_keys]) & science

    dark_bad = derived_dq(dark_masks, dark_dq)
    flat_bad = derived_dq(flat_masks, flat_dq)
    union = dark_bad | flat_bad
    both = dark_bad & flat_bad
    dark_only = dark_bad & ~flat_bad
    flat_only = flat_bad & ~dark_bad

    # Display-only dilation. The statistics and returned data use ORIGINAL masks.
    def enlarged(mask, valid=None):
        raw = np.asarray(mask, bool)
        if marker_radius:
            raw = binary_dilation(raw, iterations=int(marker_radius))
        return raw if valid is None else (raw & valid)

    def show_image(ax, data, title, cmap='viridis', robust=True, extent=None):
        ax.set_title(title, fontsize=13)
        if data is None:
            ax.text(.5, .5, 'Not available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)
            return
        arr = np.asarray(data, dtype=float).copy()
        if extent is None:
            arr[~science] = np.nan
        finite = arr[np.isfinite(arr)]
        kwargs = {}
        if robust and finite.size:
            lo, hi = np.percentile(finite, [1, 99])
            if lo < hi:
                kwargs.update(vmin=lo, vmax=hi)
        im = ax.imshow(arr, origin='lower', interpolation='nearest',
                       cmap=cmap, extent=extent, **kwargs)
        ax.figure.colorbar(im, ax=ax, fraction=.046, pad=.025)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

    def show_mask(ax, mask, title, *, zoom=False):
        raw = np.asarray(mask, bool) & science
        visible = raw if zoom else enlarged(raw, science)
        img = np.full(shape, np.nan)
        img[science] = visible[science].astype(float)
        ax.imshow(img, origin='lower', interpolation='nearest',
                  cmap=ListedColormap(['#ffffff', '#141414']),
                  norm=BoundaryNorm([-.5, .5, 1.5], 2))
        ax.set_title(f'{title} ({int(raw.sum()):,})', fontsize=12)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        # Reference border is not displayed in final BPM diagnostic masks.
        yy, xx = np.where(science)
        ax.set_xlim(xx.min() - .5, xx.max() + .5)
        ax.set_ylim(yy.min() - .5, yy.max() + .5)

    def save(fig, suffix):
        path = os.path.join(output_dir, f'{prefix}_{suffix}.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        if save_pdf:
            fig.savefig(os.path.splitext(path)[0] + '.pdf', bbox_inches='tight')
        return path

    # Figure 1: overview, enlarged defects for visibility.
    fig1, a = plt.subplots(2, 3, figsize=(23, 15), constrained_layout=True)
    dark_signal = p(dark_products, 'dark_residual_image')
    if dark_signal is None:
        dark_signal = p(dark_products, 'dark_level_image')
    flat_response = p(flat_products, 'flat_response')
    show_image(a[0, 0], dark_signal, 'Dark residual / signal')
    show_image(a[0, 1], flat_response, 'Normalized flat response')
    show_image(a[0, 2], p(dark_products, 'temporal_sigma_map'), 'Dark temporal scatter')
    show_image(a[1, 0], p(flat_products, 'temporal_sigma_map'), 'Flat temporal scatter')
    show_mask(a[1, 1], union, f'Combined science BPM: {100 * union.sum() / nscience:.3f}%')

    # Avoid priority changes due to dilation: each category is enlarged separately,
    # and original overlaps retain priority. Enlarged pixels are for display only.
    classification = np.zeros(shape, dtype=np.uint8)
    classification[enlarged(dark_only, science)] = 1
    classification[enlarged(flat_only, science)] = 2
    classification[enlarged(both, science)] = 3
    classification[both] = 3
    # Reference border is excluded from the final BPM visualization.
    colors = ['#ffffff', '#e66101', '#1b9e77', '#6a51a3', '#ffffff']
    a[1, 2].imshow(classification, origin='lower', interpolation='nearest',
                   cmap=ListedColormap(colors),
                   norm=BoundaryNorm(np.arange(-.5, 5.5), 5))
    a[1, 2].set_title('Defect source: dark / flat / both', fontsize=13)
    a[1, 2].set_xlabel('Column')
    a[1, 2].set_ylabel('Row')
    yy, xx = np.where(science)
    a[1, 2].set_xlim(xx.min() - .5, xx.max() + .5)
    a[1, 2].set_ylim(yy.min() - .5, yy.max() + .5)
    a[1, 2].legend(handles=[Patch(facecolor=c, label=t) for c, t in zip(
        colors[:4], ['Good science', 'Dark only', 'Flat only', 'Both'])],
        loc='upper center', bbox_to_anchor=(.5, -.10), ncol=3, fontsize=10)
    fig1.suptitle(f'BPM detector-wide maps — {prefix}', fontsize=20)
    fig1.text(.5, .008, f'Defect markers enlarged by {marker_radius} display pixels; '
              'counts and statistics use original masks.', ha='center', fontsize=11)
    path1 = save(fig1, 'spatial_overview')

    # Figure 2: 13 category maps + large readable summary panels.
    fig2, ax = plt.subplots(4, 4, figsize=(25, 23), constrained_layout=True)
    axes = ax.ravel()
    category_maps = [(f'Dark {k}', m(dark_masks, k)) for k in
                     ('hot', 'cold', 'unstable', 'spatial_outlier', 'cluster')]
    category_maps += [(f'Flat {k}', m(flat_masks, k)) for k in
                      ('low_qe', 'high_response', 'unstable', 'spatial_outlier', 'cluster')]
    category_maps += [
        ('Dark adjacent hot', m(dark_masks, 'adj_hot')),
        ('Flat adjacent low QE', m(flat_masks, 'adj_low_qe')),
        ('Flat adjacent high response', m(flat_masks, 'adj_open')),
    ]
    for i, (label, mask) in enumerate(category_maps):
        show_mask(axes[i], mask, label)
    axes[13].axis('off')
    labels = [name for name, _ in category_maps]
    counts = [int((mask & science).sum()) for _, mask in category_maps]
    axes[14].barh(np.arange(len(labels)), counts, color='#3977a8')
    axes[14].set_yticks(np.arange(len(labels)), labels, fontsize=10)
    axes[14].invert_yaxis()
    axes[14].set_xlabel('Flagged science pixels')
    axes[14].set_title('Category counts (overlap allowed)', fontsize=13)
    axes[15].axis('off')
    axes[15].text(0, .98,
                  f'Science pixels: {nscience:,}\n'
                  f'Combined bad: {int(union.sum()):,} ({100 * union.sum() / nscience:.3f}%)\n'
                  f'Dark bad: {int(dark_bad.sum()):,}\n'
                  f'Flat bad: {int(flat_bad.sum()):,}\n'
                  f'Both: {int(both.sum()):,}\n'
                  f'Reference border: {int(reference.sum()):,}',
                  va='top', fontsize=15, transform=axes[15].transAxes)
    fig2.suptitle(f'BPM category maps — {prefix}', fontsize=20)
    path2 = save(fig2, 'categories')

    # Figure 3: row and column audits, with suppression metadata if available.
    fig3, a = plt.subplots(3, 2, figsize=(20, 17), constrained_layout=True)
    row_valid = np.maximum(science.sum(axis=1), 1)
    col_valid = np.maximum(science.sum(axis=0), 1)
    for axis, dimension, denom, xlab in (
        (a[0, 0], 1, row_valid, 'Row'),
        (a[0, 1], 0, col_valid, 'Column'),
    ):
        for label, mask in [('Dark', dark_bad), ('Flat', flat_bad), ('Combined', union)]:
            axis.plot(100 * mask.sum(axis=dimension) / denom, label=label, lw=1.6)
        axis.set(xlabel=xlab, ylabel='Bad science pixels [%]',
                 title=f'Flagged fraction by {xlab.lower()}')
        axis.legend(fontsize=11)
        axis.grid(alpha=.2)
    removed_dark = np.zeros(shape, bool)
    removed_flat = np.zeros(shape, bool)
    details = {}
    for mode, products, acc in [
        ('dark', dark_products, removed_dark),
        ('flat', flat_products, removed_flat),
    ]:
        for key, value in products.items():
            if key.startswith('row_removed_') and value is not None and np.shape(value) == shape:
                mask = np.asarray(value, bool) & science
                acc |= mask
                details[f'{mode}_{key}'] = int(mask.sum())
    show_mask(a[1, 0], removed_dark, 'Dark row-suppressed candidates')
    show_mask(a[1, 1], removed_flat, 'Flat row-suppressed candidates')
    a[2, 0].plot(removed_dark.sum(axis=1), label='Dark suppressed')
    a[2, 0].plot(removed_flat.sum(axis=1), label='Flat suppressed')
    a[2, 0].set(xlabel='Row', ylabel='Number of removed candidates',
                title='Row suppression concentration')
    a[2, 0].legend()
    if details:
        keys = list(details)
        a[2, 1].barh(np.arange(len(keys)), [details[k] for k in keys])
        a[2, 1].set_yticks(np.arange(len(keys)), keys, fontsize=10)
        a[2, 1].invert_yaxis()
        a[2, 1].set_xlabel('Removed candidates')
    else:
        a[2, 1].text(.5, .5, 'No row-suppression products available',
                     ha='center', va='center', transform=a[2, 1].transAxes,
                     fontsize=14)
    a[2, 1].set_title('Suppression breakdown')
    fig3.suptitle(f'BPM row/column audit — {prefix}', fontsize=20)
    path3 = save(fig3, 'row_suppression')

    # Figure 4: enlarged four-corner comparison, original BPM pixels (no dilation).
    h, w = shape
    sz = min(int(edge_size), h // 2, w // 2)
    if sz < 8:
        raise ValueError('Detector is too small for corner cutouts')
    corners = [
        ('Bottom left', slice(0, sz), slice(0, sz)),
        ('Bottom right', slice(0, sz), slice(w - sz, w)),
        ('Top left', slice(h - sz, h), slice(0, sz)),
        ('Top right', slice(h - sz, h), slice(w - sz, w)),
    ]
    fig4, a = plt.subplots(4, 3, figsize=(18, 24), constrained_layout=True)
    for i, (corner, ys, xs) in enumerate(corners):
        extent = (xs.start - .5, xs.stop - .5, ys.start - .5, ys.stop - .5)
        for j, (data, label) in enumerate([
            (dark_signal, 'Dark residual / signal'),
            (flat_response, 'Normalized flat response'),
        ]):
            if data is None:
                a[i, j].text(.5, .5, 'Not available', transform=a[i, j].transAxes,
                             ha='center', va='center')
            else:
                cut = np.asarray(data[ys, xs], float).copy()
                cut[~science[ys, xs]] = np.nan
                finite = cut[np.isfinite(cut)]
                kwargs = {}
                if finite.size:
                    vmin, vmax = np.percentile(finite, [1, 99])
                    if vmin < vmax:
                        kwargs = dict(vmin=vmin, vmax=vmax)
                im = a[i, j].imshow(cut, origin='lower', interpolation='nearest',
                                    extent=extent, cmap='viridis', **kwargs)
                fig4.colorbar(im, ax=a[i, j], fraction=.046, pad=.02)
            a[i, j].set_title(f'{corner}: {label}', fontsize=13)
        # Four distinct classes; no dilation so clusters are true pixel size.
        cut_class = np.zeros((sz, sz), dtype=np.uint8)
        cut_class[dark_only[ys, xs]] = 1
        cut_class[flat_only[ys, xs]] = 2
        cut_class[both[ys, xs]] = 3
        # Reference pixels remain unclassified in the plotted BPM.
        a[i, 2].imshow(cut_class, origin='lower', interpolation='nearest',
                       extent=extent, cmap=ListedColormap(colors),
                       norm=BoundaryNorm(np.arange(-.5, 5.5), 5))
        count = int(union[ys, xs].sum())
        valid = int(science[ys, xs].sum())
        frac = 100 * count / valid if valid else 0.0
        a[i, 2].set_title(f'{corner}: BPM ({count:,}; {frac:.2f}%)', fontsize=13)
        for j in range(3):
            a[i, j].set_xlabel('Column')
            a[i, j].set_ylabel('Row')
    fig4.legend(handles=[Patch(facecolor=c, label=t) for c, t in zip(
        colors[:4], ['Good science', 'Dark only', 'Flat only', 'Both'])],
        loc='lower center', bbox_to_anchor=(.5, -.008), ncol=4, fontsize=12)
    fig4.suptitle(f'BPM detector corners ({sz}×{sz} pixels) — {prefix}', fontsize=20)
    path4 = save(fig4, 'edge_zooms')

    stats = dict(
        science_pixels=nscience,
        reference_pixels=int(reference.sum()),
        dark_bad=int(dark_bad.sum()),
        flat_bad=int(flat_bad.sum()),
        combined_bad=int(union.sum()),
        combined_bad_fraction=float(union.sum() / nscience),
        dark_only=int(dark_only.sum()),
        flat_only=int(flat_only.sum()),
        both=int(both.sum()),
        category_counts=dict(zip(labels, counts)),
        row_suppression_counts=details,
        max_row_bad_fraction=float(np.max(union.sum(axis=1) / row_valid)),
        max_column_bad_fraction=float(np.max(union.sum(axis=0) / col_valid)),
        corner_statistics={
            name: dict(bad_pixels=int(union[ys, xs].sum()),
                       science_pixels=int(science[ys, xs].sum()),
                       bad_fraction=(float(union[ys, xs].sum() / science[ys, xs].sum())
                                     if science[ys, xs].any() else None))
            for name, ys, xs in corners
        },
        visualization=dict(marker_radius=int(marker_radius), edge_size=sz),
    )
    json_path = os.path.join(output_dir, f'{prefix}_extra_diagnostics.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    return dict(figures=[fig1, fig2, fig3, fig4],
                paths=[path1, path2, path3, path4],
                json_path=json_path, statistics=stats)












def plot_bpm_parameter_validation(
    dark_masks, flat_masks, dark_products, flat_products,
    output_dir, prefix='bpm',
    flat_low_response_thresh=0.3, flat_high_response_thresh=1.4,
    dark_min_frames_bad_spatial=4, flat_min_frames_bad_spatial=3,
    sensitivity_factors=(0.75, 0.9, 1.0, 1.1, 1.25),
    dpi=200, save_pdf=False,
):
    """Diagnostic plots for checking BPM parameter choices, without rerunning detection.

    Saves three large figures: (1) classification vs measurement, (2) threshold
    sensitivity and spatial persistence, (3) preprocessing/row-correction audit.
    Sensitivity curves recompute only direct threshold classifications on fixed
    stored products; they do NOT reproduce row suppression, neighbor expansion,
    clustering, or a complete generate_bpm() run. The saved final masks are
    shown alongside these candidate classifications for comparison.

    All plotted fractions and pixel counts exclude reference pixels.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not prefix or os.path.basename(prefix) != prefix:
        raise ValueError('prefix must be a filename prefix, not a path')
    science = (np.asarray(dark_masks['science_pixel'], bool) &
               np.asarray(flat_masks['science_pixel'], bool))
    shape = science.shape
    if science.ndim != 2 or not science.any():
        raise ValueError('No valid science-pixel region')
    nscience = int(science.sum())
    factors = np.asarray(sensitivity_factors, dtype=float)
    if factors.ndim != 1 or not len(factors) or np.any(~np.isfinite(factors)) or np.any(factors <= 0):
        raise ValueError('sensitivity_factors must contain positive finite values')

    def arr(products, key):
        v = products.get(key)
        return np.asarray(v, dtype=float) if v is not None and np.shape(v) == shape else None

    def scalar(products, key):
        v = products.get(key)
        if v is None:
            return None
        v = np.asarray(v).ravel()
        return float(v[0]) if len(v) and np.isfinite(v[0]) else None

    def mask(masks, key):
        v = masks.get(key)
        return np.asarray(v, bool) & science if v is not None and np.shape(v) == shape else np.zeros(shape, bool)

    def save(fig, suffix):
        path = os.path.join(output_dir, f'{prefix}_validation_{suffix}.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        if save_pdf:
            fig.savefig(path[:-4] + '.pdf', bbox_inches='tight')
        return path

    dark = arr(dark_products, 'dark_residual_image')
    if dark is None:
        dark = arr(dark_products, 'dark_level_image')
    flat = arr(flat_products, 'flat_response')
    datasets = [
        ('Dark hot', dark, mask(dark_masks, 'hot'), scalar(dark_products, 'dark_hot_threshold'), '>'),
        ('Dark cold', dark, mask(dark_masks, 'cold'), scalar(dark_products, 'dark_cold_threshold'), '<'),
        ('Flat low QE', flat, mask(flat_masks, 'low_qe'), float(flat_low_response_thresh), '<'),
        ('Flat high response', flat, mask(flat_masks, 'high_response'), float(flat_high_response_thresh), '>'),
        ('Dark unstable', arr(dark_products, 'temporal_sigma_map'), mask(dark_masks, 'unstable'),
         scalar(dark_products, 'temporal_unstable_threshold'), '>'),
        ('Flat unstable', arr(flat_products, 'temporal_sigma_map'), mask(flat_masks, 'unstable'),
         scalar(flat_products, 'temporal_unstable_threshold'), '>'),
    ]
    # Figure 1: verify that flagged pixels lie in anomalous measurement tails.
    fig1, axes = plt.subplots(3, 2, figsize=(20, 17), constrained_layout=True)
    for ax, (name, data, flagged, threshold, direction) in zip(axes.flat, datasets):
        ax.set_title(name, fontsize=15)
        if data is None:
            ax.text(.5, .5, 'Measurement unavailable', ha='center', va='center', transform=ax.transAxes)
            continue
        good = data[science & ~flagged & np.isfinite(data)]
        bad = data[science & flagged & np.isfinite(data)]
        allvals = data[science & np.isfinite(data)]
        if not allvals.size:
            continue
        lo, hi = np.percentile(allvals, [0.1, 99.9])
        if threshold is not None:
            lo, hi = min(lo, threshold), max(hi, threshold)
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        bins = np.linspace(lo, hi, 140)
        if good.size:
            ax.hist(good, bins=bins, histtype='step', lw=1.7, label=f'Retained ({good.size:,})')
        if bad.size:
            ax.hist(bad, bins=bins, histtype='step', lw=1.7, label=f'Flagged ({bad.size:,})')
        if threshold is not None:
            ax.axvline(threshold, color='crimson', ls='--', lw=2, label=f'Cutoff {direction} {threshold:.4g}')
        ax.set_yscale('log')
        ax.set_xlabel('Stored measurement (original units)')
        ax.set_ylabel('Science pixels')
        ax.legend(fontsize=10)
        ax.grid(alpha=.2)
    fig1.suptitle(f'BPM validation: retained vs flagged measurements — {prefix}', fontsize=21)
    path1 = save(fig1, 'classification')

    # Figure 2: sensitivity on fixed products + exact spatial persistence counts.
    fig2, axes = plt.subplots(4, 2, figsize=(20, 23), constrained_layout=True)
    sensitivity = {}
    for ax, (name, data, flagged, threshold, direction) in zip(axes.flat[:6], datasets):
        ax.set_title(name, fontsize=14)
        if data is None or threshold is None:
            ax.text(.5, .5, 'No threshold/measurement', transform=ax.transAxes, ha='center', va='center')
            continue
        # Vary the distance of the threshold from the robust population median.
        vals = data[science & np.isfinite(data)]
        center = float(np.median(vals)) if vals.size else 0.
        cuts = center + factors * (threshold - center)
        rates = []
        changed = []
        for cut in cuts:
            candidate = (data > cut if direction == '>' else data < cut) & science & np.isfinite(data)
            rates.append(100. * candidate.sum() / nscience)
            changed.append(int(np.count_nonzero(candidate ^ flagged)))
        ax.plot(factors, rates, 'o-', lw=2, label='Direct-threshold candidate')
        ax.axhline(100. * flagged.sum() / nscience, ls='--', color='gray',
                   label='Saved final category')
        ax.axvline(1, ls=':', color='crimson', lw=2)
        ax.set(xlabel='Threshold distance from population median (× configured)',
               ylabel='Flagged science pixels [%]')
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
        sensitivity[name] = dict(factors=factors.tolist(), cutoffs=cuts.tolist(),
                                 candidate_fractions_percent=rates, changed_vs_final=changed)
    for ax, (mode, products, masks, minimum) in zip(axes.flat[6:], [
        ('Dark', dark_products, dark_masks, dark_min_frames_bad_spatial),
        ('Flat', flat_products, flat_masks, flat_min_frames_bad_spatial),
    ]):
        counts = arr(products, 'spatial_count')
        ax.set_title(f'{mode} spatial persistence', fontsize=14)
        if counts is None:
            ax.text(.5, .5, 'Persistence counts unavailable', transform=ax.transAxes,
                    ha='center', va='center')
            continue
        c = counts[science & np.isfinite(counts)].astype(int)
        x, n = np.unique(c, return_counts=True)
        ax.bar(x, n, width=.8, label='Science-pixel persistence')
        ax.axvline(minimum - .5, color='crimson', ls='--', lw=2,
                   label=f'Minimum frames: {minimum}')
        ax.set(xlabel='Frames flagged as spatial outlier', ylabel='Science pixels')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
    fig2.suptitle(f'BPM validation: parameter sensitivity and persistence — {prefix}', fontsize=21)
    path2 = save(fig2, 'sensitivity')

    # Figure 3: show stored raw/corrected candidates and removed classifications.
    fig3, axes = plt.subplots(3, 3, figsize=(21, 19), constrained_layout=True)
    def show_map(ax, data, title, *, binary=False):
        ax.set_title(title, fontsize=14)
        if data is None:
            ax.text(.5, .5, 'Not stored', ha='center', va='center', transform=ax.transAxes)
            return
        display = np.asarray(data, float).copy()
        display[~science] = np.nan
        kw = {}
        if not binary:
            finite = display[np.isfinite(display)]
            if finite.size:
                vmin, vmax = np.percentile(finite, [1, 99])
                if vmin < vmax:
                    kw = dict(vmin=vmin, vmax=vmax)
        im = ax.imshow(display, origin='lower', interpolation='nearest',
                       cmap='gray_r' if binary else 'viridis', **kw)
        if not binary:
            fig3.colorbar(im, ax=ax, fraction=.046, pad=.02)
        ax.set(xlabel='Column', ylabel='Row')
    show_map(axes[0, 0], arr(dark_products, 'dark_level_image'), 'Dark median signal')
    show_map(axes[0, 1], arr(dark_products, 'dark_background'), 'Dark local background')
    show_map(axes[0, 2], dark, 'Dark detection residual')
    show_map(axes[1, 0], arr(flat_products, 'stack_median'), 'Flat median signal')
    show_map(axes[1, 1], flat, 'Flat normalized response')
    show_map(axes[1, 2], arr(flat_products, 'temporal_sigma_map'), 'Flat temporal scatter')
    for ax, (mode, products) in zip(axes[2, :2], [('Dark', dark_products), ('Flat', flat_products)]):
        removed = np.zeros(shape, bool)
        for key, value in products.items():
            if key.startswith('row_removed_') and value is not None and np.shape(value) == shape:
                removed |= np.asarray(value, bool) & science
        show_map(ax, removed, f'{mode}: row-suppressed candidates ({removed.sum():,})', binary=True)
    show_map(axes[2, 2], mask(dark_masks, 'spatial_outlier') | mask(flat_masks, 'spatial_outlier'),
             'Final spatial outliers (dark or flat)', binary=True)
    fig3.suptitle(f'BPM validation: preprocessing and suppression audit — {prefix}', fontsize=21)
    path3 = save(fig3, 'preprocessing')

    stats = dict(science_pixels=nscience, sensitivity=sensitivity,
                 note='Direct threshold sensitivity uses stored measurements only; not a full BPM recomputation.')
    json_path = os.path.join(output_dir, f'{prefix}_validation.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    return dict(figures=[fig1, fig2, fig3], paths=[path1, path2, path3],
                json_path=json_path, statistics=stats)
