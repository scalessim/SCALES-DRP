from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter


# ============================================================================
# Ramp-level data-quality bits
# ============================================================================

RAMP_DQ = {
    "BPM": np.uint16(1 << 0),
    "SATURATED": np.uint16(1 << 1),
    "SAT_NEIGHBOR": np.uint16(1 << 2),
}


def _validate_inputs(input_cube, saturation_map, bpm=None):
    """Validate and normalize the detector cube, saturation map, and BPM."""
    cube = np.asarray(input_cube)
    sat_map = np.asarray(saturation_map, dtype=np.float64)

    if cube.ndim != 3:
        raise ValueError("input_cube must have shape (Nreads, H, W)")

    n_reads, height, width = cube.shape
    detector_shape = (height, width)

    if sat_map.shape != detector_shape:
        raise ValueError(
            f"saturation_map has shape {sat_map.shape}; expected {detector_shape}"
        )

    if bpm is None:
        bpm_map = np.zeros(detector_shape, dtype=bool)
    else:
        bpm_map = np.asarray(bpm, dtype=bool)
        if bpm_map.shape != detector_shape:
            raise ValueError(
                f"bpm has shape {bpm_map.shape}; expected {detector_shape}"
            )

    return cube, sat_map, bpm_map


def _compute_saturation_timing(
    input_cube,
    saturation_map,
    *,
    neighbor_radius=1,
):
    """
    Compute direct-saturation and neighbor-propagated cutoff read maps.

    Notes
    -----
    ``n_reads`` is used internally as the sentinel value meaning
    "this condition never occurs in the ramp".

    Returns
    -------
    first_direct_sat : ndarray, int32, shape (H, W)
        First read at which each pixel itself reaches saturation.

    first_rejected : ndarray, int32, shape (H, W)
        First read rejected after including nearest-neighbor propagation.

    directly_saturated : ndarray, bool, shape (H, W)
        True where the pixel itself reaches saturation during the ramp.

    neighbor_affected : ndarray, bool, shape (H, W)
        True where a neighboring pixel causes rejection earlier than the
        pixel's own direct saturation time. This includes pixels that later
        saturate themselves and pixels that never directly saturate.

    neighbor_only : ndarray, bool, shape (H, W)
        True where a pixel never directly saturates but is rejected because
        one of its neighbors saturates.
    """
    cube = np.asarray(input_cube)
    sat_map = np.asarray(saturation_map, dtype=np.float64)

    if cube.ndim != 3:
        raise ValueError("input_cube must have shape (Nreads, H, W)")

    n_reads, height, width = cube.shape

    if sat_map.shape != (height, width):
        raise ValueError(
            f"saturation_map has shape {sat_map.shape}; "
            f"expected {(height, width)}"
        )

    neighbor_radius = int(neighbor_radius)
    if neighbor_radius < 0:
        raise ValueError("neighbor_radius must be >= 0")

    valid_saturation = np.isfinite(sat_map) & (sat_map > 0)

    # Threshold crossing for every read and pixel.
    threshold_crossing = (
        np.isfinite(cube)
        & valid_saturation[None, :, :]
        & (cube >= sat_map[None, :, :])
    )

    # Did the pixel ever cross its own saturation threshold?
    directly_saturated = np.any(threshold_crossing, axis=0)

    # First direct-saturation read. n_reads means "never saturated".
    first_direct_sat = np.full(
        (height, width),
        n_reads,
        dtype=np.int32,
    )

    first_direct_sat[directly_saturated] = np.argmax(
        threshold_crossing[:, directly_saturated],
        axis=0,
    ).astype(np.int32)

    # ------------------------------------------------------------------
    # Nearest-neighbor propagation using the 2-D timing map.
    # ------------------------------------------------------------------
    # If neighbor_radius=1, each pixel inherits the earliest saturation
    # read found in its surrounding 3x3 region. This is equivalent to
    # dilating every read plane, but avoids looping over all detector reads.
    # ------------------------------------------------------------------
    if neighbor_radius > 0:
        size = 2 * neighbor_radius + 1
        first_rejected = minimum_filter(
            first_direct_sat,
            size=size,
            mode="constant",
            cval=n_reads,
        ).astype(np.int32)
    else:
        first_rejected = first_direct_sat.copy()

    # A neighbor matters only if it moves the ramp cutoff earlier than the
    # pixel's own direct-saturation read.
    neighbor_affected = first_rejected < first_direct_sat

    # These pixels never directly saturate, but are rejected because a
    # nearest neighbor does.
    neighbor_only = neighbor_affected & ~directly_saturated

    return (
        first_direct_sat,
        first_rejected,
        directly_saturated,
        neighbor_affected,
        neighbor_only,
    )


def make_ramp_quality_mask(
    input_cube,
    saturation_map,
    *,
    bpm=None,
    neighbor_radius=1,
    print_summary=False,
):
    """
    Build the final detector DQ map and read-validity mask for ramp fitting.

    Parameters
    ----------
    input_cube : ndarray, shape (Nreads, H, W)
        Detector ramp. In the pipeline this should be the cube that is passed
        into ramp fitting (e.g. the linearity-corrected cube).

    saturation_map : ndarray, shape (H, W)
        Per-pixel saturation threshold in DN.

    bpm : ndarray, bool, shape (H, W), optional
        External detector bad-pixel mask. True means bad.

    neighbor_radius : int, optional
        Radius around each saturated pixel that should also be rejected.
        ``neighbor_radius=1`` means the saturated pixel plus its 8 nearest
        surrounding neighbors (3x3 region).

    print_summary : bool, optional
        Print compact saturation/masking diagnostics.

    Returns
    -------
    quality_map : ndarray, uint16, shape (H, W)
        Overall detector data-quality map.

        bit 0 : BPM
        bit 1 : pixel itself directly saturated during the ramp
        bit 2 : a saturated neighbor forced an earlier ramp cutoff

        Bits may be combined. For example, a pixel can be both SATURATED and
        SAT_NEIGHBOR when it eventually saturates itself but a neighbor
        saturated first.

    good_read_mask : ndarray, bool, shape (Nreads, H, W)
        Mask passed directly to the ramp fitter.

        True  = use this read
        False = reject this read

        Once the first rejection read is reached, that read and every later
        read are False. BPM pixels are False for every read.
    """
    cube, sat_map, bpm_map = _validate_inputs(
        input_cube,
        saturation_map,
        bpm=bpm,
    )

    n_reads, height, width = cube.shape

    (
        first_direct_sat,
        first_rejected,
        directly_saturated,
        neighbor_affected,
        neighbor_only,
    ) = _compute_saturation_timing(
        cube,
        sat_map,
        neighbor_radius=neighbor_radius,
    )

    # ------------------------------------------------------------------
    # 2-D detector DQ map
    # ------------------------------------------------------------------
    quality_map = np.zeros((height, width), dtype=np.uint16)

    quality_map[bpm_map] |= RAMP_DQ["BPM"]
    quality_map[directly_saturated] |= RAMP_DQ["SATURATED"]
    quality_map[neighbor_affected] |= RAMP_DQ["SAT_NEIGHBOR"]

    # ------------------------------------------------------------------
    # 3-D ramp-fitting mask
    # ------------------------------------------------------------------
    # Read r is good only when r is strictly earlier than first_rejected.
    read_index = np.arange(n_reads, dtype=np.int32)[:, None, None]
    good_read_mask = read_index < first_rejected[None, :, :]

    # Entire ramp is rejected for externally bad pixels.
    good_read_mask[:, bpm_map] = False

    if print_summary:
        affected = first_rejected < n_reads

        print("\nRamp saturation diagnostics")
        print("-" * 62)
        print(f"Total detector pixels                       : {height * width:,}")
        print(
            "Directly saturated pixels                 : "
            f"{np.count_nonzero(directly_saturated):,}"
        )
        print(
            "Pixels affected after neighbor propagation: "
            f"{np.count_nonzero(affected):,}"
        )
        print(
            "Neighbor-only pixels                      : "
            f"{np.count_nonzero(neighbor_only):,}"
        )
        print(
            "Ramps shortened by saturated neighbor     : "
            f"{np.count_nonzero(neighbor_affected):,}"
        )
        print(f"BPM pixels                                  : {np.count_nonzero(bpm_map):,}")

        both = neighbor_affected & directly_saturated
        if np.any(both):
            reads_lost = first_direct_sat[both] - first_rejected[both]
            print(
                "Median reads lost to neighbor            : "
                f"{np.median(reads_lost):.1f}"
            )
            print(
                "Maximum reads lost to neighbor           : "
                f"{np.max(reads_lost):d}"
            )

    return quality_map, good_read_mask


def plot_ramp_quality_flags(
    input_cube,
    saturation_map,
    quality_map,
    good_read_mask,
    pixels=None,
    *,
    neighbor_radius=1,
    read_times=None,
    figsize_maps=(18, 10),
    figsize_ramps=(15, 9),
    show=True,
):
    """
    Visualize direct saturation, nearest-neighbor masking, and ramp cutoffs.

    The detector-level figure shows:
      1. directly saturated pixels,
      2. neighbor-only pixels, colored by the first neighbor-induced cutoff,
      3. number of reads lost where a neighbor saturates earlier,
      4. first direct-saturation read,
      5. first rejected read after neighbor propagation,
      6. number of reads available to ramp fitting.

    Parameters
    ----------
    input_cube : ndarray, shape (Nreads, H, W)
    saturation_map : ndarray, shape (H, W)
    quality_map : ndarray, uint16, shape (H, W)
    good_read_mask : ndarray, bool, shape (Nreads, H, W)
    pixels : sequence of (row, col), optional
        Individual ramps to display.
    neighbor_radius : int, optional
        Must match the radius used to build ``good_read_mask``.
    read_times : ndarray, optional
        Read times; otherwise integer read number is used.
    show : bool, optional
        Call ``plt.show()`` when True.

    Returns
    -------
    fig_maps : matplotlib.figure.Figure
    fig_ramps : matplotlib.figure.Figure or None
    """
    cube, sat_map, _ = _validate_inputs(input_cube, saturation_map, bpm=None)
    qmap = np.asarray(quality_map, dtype=np.uint16)
    good = np.asarray(good_read_mask, dtype=bool)

    n_reads, height, width = cube.shape

    if qmap.shape != (height, width):
        raise ValueError(
            f"quality_map has shape {qmap.shape}; expected {(height, width)}"
        )

    if good.shape != cube.shape:
        raise ValueError("good_read_mask must have the same shape as input_cube")

    (
        first_direct,
        first_rejected,
        directly_saturated,
        neighbor_affected,
        neighbor_only,
    ) = _compute_saturation_timing(
        cube,
        sat_map,
        neighbor_radius=neighbor_radius,
    )

    # Decode the supplied DQ map as an independent consistency check.
    dq_saturated = (qmap & RAMP_DQ["SATURATED"]) != 0
    dq_neighbor = (qmap & RAMP_DQ["SAT_NEIGHBOR"]) != 0
    dq_bpm = (qmap & RAMP_DQ["BPM"]) != 0

    if not np.array_equal(dq_saturated, directly_saturated):
        print("Warning: SATURATED bits do not match saturation timing recomputation.")

    if not np.array_equal(dq_neighbor, neighbor_affected):
        print("Warning: SAT_NEIGHBOR bits do not match neighbor timing recomputation.")

    # ------------------------------------------------------------------
    # Plot-friendly timing maps
    # ------------------------------------------------------------------
    first_direct_plot = first_direct.astype(float)
    first_direct_plot[first_direct == n_reads] = np.nan

    first_rejected_plot = first_rejected.astype(float)
    first_rejected_plot[first_rejected == n_reads] = np.nan

    # Neighbor-only timing: show only pixels that never directly saturate but
    # are rejected because an adjacent pixel does.
    neighbor_only_read = np.full((height, width), np.nan, dtype=float)
    neighbor_only_read[neighbor_only] = first_rejected[neighbor_only]

    # Number of reads lost due to a neighbor. For pixels that never directly
    # saturate, there is no finite "own saturation" time, so they are left NaN
    # here and are shown separately in the neighbor-only timing panel.
    reads_lost = np.full((height, width), np.nan, dtype=float)
    both = neighbor_affected & directly_saturated
    reads_lost[both] = first_direct[both] - first_rejected[both]

    n_good = np.sum(good, axis=0, dtype=np.int32)

    # ==================================================================
    # Detector-level diagnostic figure
    # ==================================================================
    fig_maps, axes = plt.subplots(
        2,
        3,
        figsize=figsize_maps,
        constrained_layout=True,
    )
    axes = axes.ravel()

    axes[0].imshow(
        directly_saturated,
        origin="lower",
        interpolation="nearest",
    )
    axes[0].set_title(
        "Directly saturated pixels\n"
        f"N={np.count_nonzero(directly_saturated):,}"
    )

    im = axes[1].imshow(
        neighbor_only_read,
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=n_reads - 1,
    )
    axes[1].set_title(
        "Neighbor-only flags\n"
        f"N={np.count_nonzero(neighbor_only):,}"
    )
    fig_maps.colorbar(im, ax=axes[1], label="First rejected read")

    im = axes[2].imshow(
        reads_lost,
        origin="lower",
        interpolation="nearest",
    )
    axes[2].set_title(
        "Reads lost to saturated neighbor\n"
        f"N={np.count_nonzero(both):,}"
    )
    fig_maps.colorbar(im, ax=axes[2], label="Reads lost")

    im = axes[3].imshow(
        first_direct_plot,
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=n_reads - 1,
    )
    axes[3].set_title("First direct saturation read")
    fig_maps.colorbar(im, ax=axes[3], label="Read number")

    im = axes[4].imshow(
        first_rejected_plot,
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=n_reads - 1,
    )
    axes[4].set_title("First rejected read\nincluding saturated neighbors")
    fig_maps.colorbar(im, ax=axes[4], label="Read number")

    im = axes[5].imshow(
        n_good,
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=n_reads,
    )
    axes[5].set_title("Reads available for ramp fitting")
    fig_maps.colorbar(im, ax=axes[5], label="Number of good reads")

    for ax in axes:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig_maps.suptitle("Ramp-fitting saturation diagnostics", fontsize=15)

    # ==================================================================
    # Optional individual-ramp diagnostics
    # ==================================================================
    fig_ramps = None

    if pixels is not None:
        pixels = list(pixels)
        if not pixels:
            raise ValueError("pixels cannot be an empty sequence")

        if read_times is None:
            x = np.arange(n_reads)
            xlabel = "Read number"
        else:
            x = np.asarray(read_times)
            if x.shape != (n_reads,):
                raise ValueError(f"read_times must have shape {(n_reads,)}")
            xlabel = "Read time"

        n_pixels = len(pixels)
        ncols = min(3, n_pixels)
        nrows = int(np.ceil(n_pixels / ncols))

        fig_ramps, axarr = plt.subplots(
            nrows,
            ncols,
            figsize=figsize_ramps,
            squeeze=False,
            constrained_layout=True,
        )
        axes_flat = axarr.ravel()

        for ax, (row, col) in zip(axes_flat, pixels):
            if not (0 <= row < height and 0 <= col < width):
                raise IndexError(
                    f"Pixel {(row, col)} outside detector shape {(height, width)}"
                )

            ramp = cube[:, row, col]
            valid = good[:, row, col]
            finite = np.isfinite(ramp)

            used = valid & finite
            rejected = (~valid) & finite

            ax.plot(
                x[finite],
                ramp[finite],
                "o-",
                markersize=2,
                linewidth=1,
                alpha=0.35,
                label="Ramp",
            )
            ax.scatter(
                x[used],
                ramp[used],
                s=18,
                label="Used",
                zorder=4,
            )
            ax.scatter(
                x[rejected],
                ramp[rejected],
                s=30,
                marker="x",
                label="Rejected",
                zorder=5,
            )

            direct_read = int(first_direct[row, col])
            reject_read = int(first_rejected[row, col])

            if direct_read < n_reads:
                ax.axvline(
                    x[direct_read],
                    linestyle=":",
                    linewidth=1.5,
                    label="Own saturation",
                )

            if reject_read < n_reads:
                ax.axvline(
                    x[reject_read],
                    linestyle="--",
                    linewidth=1.5,
                    label="Fit cutoff",
                )

            flags = []
            q = qmap[row, col]
            if q & RAMP_DQ["BPM"]:
                flags.append("BPM")
            if q & RAMP_DQ["SATURATED"]:
                flags.append("SAT")
            if q & RAMP_DQ["SAT_NEIGHBOR"]:
                flags.append("SAT-NBR")

            flag_text = " | ".join(flags) if flags else "GOOD"

            ax.set_title(
                f"Pixel ({row}, {col})\n"
                f"{flag_text}, {np.count_nonzero(valid)}/{n_reads} valid reads"
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("DN")
            ax.grid(alpha=0.25)

        for ax in axes_flat[len(pixels):]:
            ax.axis("off")

        handles = []
        labels = []
        for ax in axes_flat[:len(pixels)]:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)

        fig_ramps.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(5, len(labels)),
        )
        fig_ramps.suptitle("Ramp-level masking diagnostics", fontsize=15)

    print("\nRamp-fitting saturation diagnostics")
    print("-" * 62)
    print(f"Total detector pixels               : {height * width:,}")
    print(f"Directly saturated                  : {np.count_nonzero(directly_saturated):,}")
    print(f"Neighbor-only pixels                : {np.count_nonzero(neighbor_only):,}")
    print(f"Ramps shortened by saturated nbr    : {np.count_nonzero(neighbor_affected):,}")
    print(f"BPM pixels                          : {np.count_nonzero(dq_bpm):,}")

    if np.any(both):
        print(f"Median reads lost to neighbor       : {np.nanmedian(reads_lost[both]):.1f}")
        print(f"Maximum reads lost to neighbor      : {int(np.nanmax(reads_lost[both]))}")

    if show:
        plt.show()

    return fig_maps, fig_ramps

