import time
import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter, binary_dilation, label
from scipy import sparse

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
              "Dark signal: HOT / COLD", "Dark slope residual (input slope units)",
              [(dark_cold, "Cold cutoff", "royalblue"),
               (dark_hot, "Hot cutoff", "crimson")])
    draw_hist(axes[0, 1], values(flat_products, "flat_response"),
              "Flat response: LOW QE / HIGH RESPONSE", "Normalized flat response",
              [(flat_low, "Low-QE cutoff", "crimson"),
               (flat_high, "High-response cutoff", "darkorange")])
    draw_hist(axes[1, 0], values(dark_products, "temporal_sigma_map"),
              "Dark temporal scatter", "Temporal MAD × 1.4826 (slope units)",
              [(scalar(dark_products, "temporal_unstable_threshold"), "Unstable cutoff", "crimson")])
    draw_hist(axes[1, 1], values(flat_products, "temporal_sigma_map"),
              "Flat temporal scatter", "Temporal MAD × 1.4826 (slope units)",
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



















