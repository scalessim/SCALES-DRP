#from __future__ import annotations
#from scalesdrp.core.scales_pkg_resources import get_resource_path
from pathlib import Path
import numpy as np
from astropy.io import fits
import os

def _evaluate_coefficient_cube(
    signal: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """
    Evaluate one ascending-power polynomial per detector pixel.

    Parameters
    ----------
    signal : ndarray, shape (N, H, W)
        Pedestal-subtracted measured signal.

    coefficients : ndarray, shape (Ncoeff, H, W)
        Polynomial coefficients in ascending order:

            corrected = c0 + c1*M + c2*M**2 + ...

    Returns
    -------
    corrected_signal : ndarray, shape (N, H, W)
    """
    signal = np.asarray(signal, dtype=np.float32)
    coefficients = np.asarray(coefficients, dtype=np.float64)

    if coefficients.ndim != 3:
        raise ValueError(
            "coefficients must have shape (Ncoeff, H, W)"
        )

    if signal.ndim != 3:
        raise ValueError(
            "signal must have shape (Nreads, H, W)"
        )

    if signal.shape[1:] != coefficients.shape[1:]:
        raise ValueError(
            "Signal and coefficient spatial shapes do not match: "
            f"{signal.shape[1:]} versus {coefficients.shape[1:]}"
        )

    # Horner evaluation for ascending coefficients:
    # Start from highest order and work downward.
    output = np.zeros(signal.shape, dtype=np.float64)

    for order in range(coefficients.shape[0] - 1, -1, -1):
        output *= signal
        output += coefficients[order][None, :, :]

    return output


def apply_brandt_linearity_reference(
    input_cube: np.ndarray,
    coefficient_file: str | Path,
    *,
    n_pedestal_reads: int = 2,
    pedestal_start_read: int = 0,
    saturation_fraction: float = 0.95,
    saturated_read_behavior: str = "raw",
    apply_only_successful: bool = False,
    return_aux: bool = False,
):
    """
    Apply a Brandt-style nonlinearity reference file to one detector ramp.

    The reference coefficients use ascending powers:

        corrected_signal =
            c0 + c1*M + c2*M**2 + ...

    where M is the pedestal-subtracted measured signal.

    Processing steps
    ----------------
    1. Estimate a per-pixel pedestal from the first pedestal reads.
    2. Subtract that pedestal from every read.
    3. Apply the per-pixel polynomial to pre-saturation signal.
    4. Add the pedestal back.
    5. Leave rejected/identity pixels unchanged.
    6. Leave saturated reads raw by default.

    Parameters
    ----------
    input_cube : ndarray, shape (Nreads, H, W)
        Input ramp in absolute measured DN.

    coefficient_file : str or pathlib.Path
        Reference FITS produced by write_linearity_reference_fits().

    n_pedestal_reads : int, optional
        Number of reads used to estimate the input-ramp pedestal.

    pedestal_start_read : int, optional
        First read included in the pedestal estimate.

    saturation_fraction : float, optional
        Fraction of the reference saturation level treated as usable.
        This should normally match the fraction used during coefficient
        generation.

    saturated_read_behavior : {"raw", "nan", "flat_last_valid"}
        Treatment of reads at or above the usable saturation signal.

        "raw"
            Preserve the original input values. Recommended when saturation
            is masked later during ramp fitting.

        "nan"
            Set saturated reads to NaN.

        "flat_last_valid"
            Replace saturated reads with the last valid corrected read.

    apply_only_successful : bool, optional
        If True, apply fitted coefficients only where SUCCESS is true.
        Other pixels remain exactly equal to the input.

        If False, all coefficient vectors are evaluated. This is normally
        safe because failed pixels in the supplied calibration code receive
        identity coefficients.

    return_aux : bool, optional
        If True, also return the pedestal, saturation mask, and applied mask.

    Returns
    -------
    corrected_cube : ndarray, shape (Nreads, H, W), float32

    If return_aux=True:
        corrected_cube, pedestal, saturation_mask, applied_mask
    """

    cube = np.asarray(input_cube, dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError(
            "input_cube must have shape (Nreads, H, W)"
        )

    n_reads, height, width = cube.shape

    n_pedestal_reads = int(n_pedestal_reads)
    pedestal_start_read = int(pedestal_start_read)

    if n_pedestal_reads < 1:
        raise ValueError("n_pedestal_reads must be at least 1")

    if pedestal_start_read < 0:
        raise ValueError("pedestal_start_read must be non-negative")

    pedestal_stop = min(
        pedestal_start_read + n_pedestal_reads,
        n_reads,
    )

    if pedestal_start_read >= pedestal_stop:
        raise ValueError(
            "No reads are available for pedestal estimation"
        )

    if not 0 < saturation_fraction <= 1:
        raise ValueError(
            "saturation_fraction must be in the interval (0, 1]"
        )

    if saturated_read_behavior not in {
        "raw",
        "nan",
        "flat_last_valid",
    }:
        raise ValueError(
            "saturated_read_behavior must be 'raw', 'nan', "
            "or 'flat_last_valid'"
        )

    coefficient_file = Path(coefficient_file)
    print("Coefficient file:", coefficient_file)
    print("Exists:", os.path.exists(coefficient_file))
    print("Absolute path:", os.path.abspath(coefficient_file))
    with fits.open(coefficient_file, memmap=False) as hdul:
        required = ("COEFF", "SATURATION")

        missing = [name for name in required if name not in hdul]
        if missing:
            raise KeyError(
                f"Missing required extensions in {coefficient_file}: "
                f"{missing}"
            )

        coefficients = np.asarray(
            hdul["COEFF"].data,
            dtype=np.float64,
        )

        saturation = np.asarray(
            hdul["SATURATION"].data,
            dtype=np.float64,
        )

        bpm = (
            np.asarray(hdul["BPM"].data, dtype=bool)
            if "BPM" in hdul
            else np.zeros((height, width), dtype=bool)
        )

        success = (
            np.asarray(hdul["SUCCESS"].data, dtype=bool)
            if "SUCCESS" in hdul
            else np.ones((height, width), dtype=bool)
        )

        coefficient_order = str(
            hdul[0].header.get("COEFORD", "ASCENDING")
        ).upper()

    if coefficient_order != "ASCENDING":
        raise ValueError(
            f"Unsupported coefficient ordering {coefficient_order!r}; "
            "this function expects ascending-power coefficients"
        )

    detector_shape = (height, width)

    if coefficients.ndim != 3:
        raise ValueError(
            "COEFF must have shape (Ncoeff, H, W)"
        )

    if coefficients.shape[1:] != detector_shape:
        raise ValueError(
            "COEFF spatial shape does not match input cube: "
            f"{coefficients.shape[1:]} versus {detector_shape}"
        )

    for name, array in {
        "SATURATION": saturation,
        "BPM": bpm,
        "SUCCESS": success,
    }.items():
        if array.shape != detector_shape:
            raise ValueError(
                f"{name} has shape {array.shape}; expected "
                f"{detector_shape}"
            )

    # ------------------------------------------------------------
    # Estimate the input-ramp pedestal
    # ------------------------------------------------------------
    pedestal_region = cube[
        pedestal_start_read:pedestal_stop
    ]

    with np.errstate(invalid="ignore"):
        pedestal = np.nanmedian(
            pedestal_region,
            axis=0,
        )

    pedestal_finite = np.isfinite(pedestal)

    # The coefficients were derived from pedestal-subtracted ramps.
    signal = (
        cube.astype(np.float64)
        - pedestal[None, :, :]
    )

    # Convert absolute saturation DN to pedestal-subtracted signal,
    # following the convention used during calibration.
    usable_saturation_signal = (
        saturation - pedestal
    ) * float(saturation_fraction)

    valid_saturation = (
        np.isfinite(usable_saturation_signal)
        & (usable_saturation_signal > 0)
    )

    saturation_mask = (
        ~np.isfinite(signal)
        | ~valid_saturation[None, :, :]
        | (
            signal
            >= usable_saturation_signal[None, :, :]
        )
    )

    coefficient_finite = np.all(
        np.isfinite(coefficients),
        axis=0,
    )

    applied_mask = (
        ~bpm
        & pedestal_finite
        & coefficient_finite
        & valid_saturation
    )

    if apply_only_successful:
        applied_mask &= success

    # ------------------------------------------------------------
    # Evaluate coefficients
    # ------------------------------------------------------------
    corrected_signal = _evaluate_coefficient_cube(
        signal,
        coefficients,
    )

    mapped_finite = np.all(
        (~np.isfinite(signal))
        | np.isfinite(corrected_signal),
        axis=0,
    )

    # All-or-nothing pixel policy:
    # if the polynomial produces a nonfinite value for a pixel,
    # leave the complete pixel ramp unchanged.
    applied_mask &= mapped_finite

    pre_saturation = (
        ~saturation_mask
        & applied_mask[None, :, :]
    )

    # Start from the original cube so every failed or skipped pixel
    # remains exactly unchanged.
    corrected_cube = cube.copy()

    corrected_absolute = (
        corrected_signal
        + pedestal[None, :, :]
    )

    corrected_cube[pre_saturation] = (
        corrected_absolute[pre_saturation]
        .astype(np.float32)
    )

    # ------------------------------------------------------------
    # Handle saturated reads
    # ------------------------------------------------------------
    saturated_applied = (
        saturation_mask
        & applied_mask[None, :, :]
    )

    if saturated_read_behavior == "raw":
        # Nothing to do: corrected_cube began as a copy of cube.
        pass

    elif saturated_read_behavior == "nan":
        corrected_cube[saturated_applied] = np.nan

    elif saturated_read_behavior == "flat_last_valid":
        for row, col in np.argwhere(applied_mask):
            valid_reads = np.flatnonzero(
                pre_saturation[:, row, col]
            )

            if valid_reads.size == 0:
                continue

            last_valid_read = int(valid_reads[-1])

            later_saturated = (
                np.arange(n_reads) > last_valid_read
            ) & saturation_mask[:, row, col]

            corrected_cube[
                later_saturated,
                row,
                col,
            ] = corrected_cube[
                last_valid_read,
                row,
                col,
            ]

    # Safety check: raw mode must not introduce new NaNs.
    if saturated_read_behavior == "raw":
        new_nonfinite = (
            ~np.isfinite(corrected_cube)
            & np.isfinite(cube)
        )

        if np.any(new_nonfinite):
            raise RuntimeError(
                "Linearity correction introduced new nonfinite values"
            )

    if return_aux:
        return (
            corrected_cube.astype(np.float32, copy=False),
            pedestal.astype(np.float32),
            saturation_mask,
            applied_mask,
        )

    return corrected_cube.astype(np.float32, copy=False)


import matplotlib.pyplot as plt
import numpy as np


def plot_six_linearity_corrected_pixels(
    raw_cube,
    corrected_cube,
    pixels,
    *,
    saturation_mask=None,
    read_times=None,
    figsize=(15, 9),
    savepath=None,
    dpi=200,
    show=True,
):
    """
    Plot raw and linearity-corrected ramps for six detector pixels.

    Parameters
    ----------
    raw_cube : ndarray, shape (Nreads, H, W)
        Original ramp cube.

    corrected_cube : ndarray, shape (Nreads, H, W)
        Linearity-corrected ramp cube.

    pixels : sequence of six (row, column) tuples
        Pixels to display.

    saturation_mask : ndarray, optional, shape (Nreads, H, W)
        Boolean mask. Saturated reads are shown with open markers.

    read_times : ndarray, optional, shape (Nreads,)
        Read times. If omitted, integer read number is used.

    figsize : tuple
        Figure size.

    savepath : str or pathlib.Path, optional
        Figure output path.

    dpi : int
        Saved figure resolution.

    show : bool
        Display the figure.

    Returns
    -------
    fig, axes
    """

    raw = np.asarray(raw_cube, dtype=np.float64)
    corrected = np.asarray(corrected_cube, dtype=np.float64)

    if raw.ndim != 3:
        raise ValueError(
            "raw_cube must have shape (Nreads, H, W)"
        )

    if corrected.shape != raw.shape:
        raise ValueError(
            "corrected_cube must have the same shape as raw_cube"
        )

    pixels = [tuple(pixel) for pixel in pixels]

    if len(pixels) != 6:
        raise ValueError(
            "Exactly six (row, column) pixels must be supplied"
        )

    n_reads, height, width = raw.shape

    if read_times is None:
        x = np.arange(n_reads, dtype=float)
        xlabel = "Read number"
    else:
        x = np.asarray(read_times, dtype=float)

        if x.shape != (n_reads,):
            raise ValueError(
                f"read_times must have shape {(n_reads,)}"
            )

        xlabel = "Read time"

    if saturation_mask is not None:
        saturation_mask = np.asarray(
            saturation_mask,
            dtype=bool,
        )

        if saturation_mask.shape != raw.shape:
            raise ValueError(
                "saturation_mask must have the same shape as raw_cube"
            )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=figsize,
        sharex=False,
        sharey=False,
    )

    axes_flat = axes.ravel()

    for ax, (row, col) in zip(axes_flat, pixels):
        if not (0 <= row < height and 0 <= col < width):
            raise IndexError(
                f"Pixel {(row, col)} is outside detector shape "
                f"{(height, width)}"
            )

        raw_ramp = raw[:, row, col]
        corrected_ramp = corrected[:, row, col]

        finite_raw = np.isfinite(raw_ramp)
        finite_corrected = np.isfinite(corrected_ramp)

        ax.plot(
            x[finite_raw],
            raw_ramp[finite_raw],
            "o-",
            color="black",
            markersize=3,
            linewidth=1.1,
            label="Raw",
        )

        ax.plot(
            x[finite_corrected],
            corrected_ramp[finite_corrected],
            "o-",
            color="red",
            markersize=3,
            linewidth=1.3,
            label="Corrected",
        )

        if saturation_mask is not None:
            saturated = saturation_mask[:, row, col]

            if np.any(saturated & finite_raw):
                ax.scatter(
                    x[saturated & finite_raw],
                    raw_ramp[saturated & finite_raw],
                    facecolors="none",
                    edgecolors="orange",
                    s=28,
                    linewidths=1.0,
                    label="Saturated",
                    zorder=5,
                )

        difference = corrected_ramp - raw_ramp
        finite_difference = np.isfinite(difference)

        if np.any(finite_difference):
            max_difference = np.max(
                np.abs(difference[finite_difference])
            )
        else:
            max_difference = np.nan

        ax.set_title(
            f"Pixel ({row}, {col})\n"
            rf"max $|L-M|$ = {max_difference:.2f} DN",
            fontsize=11,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Absolute detector count [DN]")
        ax.grid(alpha=0.25)

    handles, labels = axes_flat[0].get_legend_handles_labels()

    # Include any saturation label that may only occur in another panel.
    for ax in axes_flat[1:]:
        panel_handles, panel_labels = ax.get_legend_handles_labels()

        for handle, label in zip(panel_handles, panel_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.suptitle(
        "Representative detector ramps before and after "
        "linearity correction",
        fontsize=15,
        y=1.05,
    )

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(
            savepath,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig, axes










