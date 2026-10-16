"""Read-noise calibration from pre-corrected dark CDS images or ramp cubes.
No FITS products are written by this module.
"""
from pathlib import Path
import warnings
import numpy as np
from astropy.io import fits
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def robust_sigma_mad(x, axis=0):
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - med), axis=axis)


def subtract_common_offsets(image, n_channels=4, subtract_global=False,
                            subtract_amp=False, subtract_row=False,
                            ref_width=4, active_only=True):
    img = np.asarray(image, dtype=np.float64).copy()
    if img.ndim != 2 or n_channels < 1 or img.shape[1] % n_channels:
        raise ValueError('Expected a 2D image with width divisible by n_channels')
    ny, nx = img.shape
    if not 0 <= ref_width < min(ny, nx) // 2:
        raise ValueError('Invalid ref_width')
    valid = np.isfinite(img)
    if active_only and ref_width:
        valid[:ref_width] = False
        valid[-ref_width:] = False
        valid[:, :ref_width] = False
        valid[:, -ref_width:] = False
    if subtract_global and valid.any():
        img -= np.median(img[valid])
    if subtract_amp:
        for ch in range(n_channels):
            sl = slice(ch * (nx // n_channels), (ch + 1) * (nx // n_channels))
            m = valid[:, sl]
            if m.any():
                img[:, sl] -= np.median(img[:, sl][m])
    if subtract_row:
        for y in range(ny):
            if valid[y].any():
                img[y] -= np.median(img[y, valid[y]])
    return img


def sigma_clip_stack_no_nan(stack, sigma=5.0, min_good=2, fill_value=0.0):
    """Return finite sample std, retained sample count, and unreliable-pixel mask."""
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError('stack must have shape (N>=2, Y, X)')
    if sigma <= 0 or min_good < 2:
        raise ValueError('sigma must be positive and min_good >= 2')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(stack, axis=0)
        mad = robust_sigma_mad(stack, axis=0)
        fallback = np.nanstd(stack, axis=0, ddof=1)
        scale = np.where(np.isfinite(mad) & (mad > 0), mad, fallback)
        clip = (~np.isfinite(scale)) | (scale <= 0)
        good = np.isfinite(stack) & (clip[None] | (np.abs(stack - med[None]) <= sigma * scale[None]))
        n_good = good.sum(axis=0).astype(np.int32)
        std = np.nanstd(np.where(good, stack, np.nan), axis=0, ddof=1)
    bad = (n_good < min_good) | ~np.isfinite(std) | (std < 0)
    valid = ~bad
    replacement = float(np.median(std[valid])) if valid.any() else float(fill_value)
    if not np.isfinite(replacement) or replacement < 0:
        raise ValueError('fill_value must be finite and nonnegative')
    return np.where(bad, replacement, std), n_good, bad


def interpolate_bad_readnoise(map_e, bad_mask, *, kernel_size=3, max_iterations=8):
    """Replace invalid RN pixels using local valid neighbors, then nearest valid.

    BAD_RN remains True for every replaced pixel. Never use an originally
    invalid pixel as a donor in the local-median interpolation.
    """
    data = np.asarray(map_e, dtype=np.float64)
    bad = np.asarray(bad_mask, dtype=bool) | ~np.isfinite(data) | (data <= 0)
    if data.ndim != 2 or bad.shape != data.shape:
        raise ValueError('map_e and bad_mask must be matching 2D arrays')
    if kernel_size < 3 or kernel_size % 2 != 1 or max_iterations < 1:
        raise ValueError('kernel_size must be odd >=3; max_iterations >=1')
    valid = ~bad
    if not valid.any():
        raise ValueError('No positive, reliable RN pixels available for interpolation')
    # Nearest-neighbor seed guarantees a finite positive value in every hole.
    # Median filtering this seeded image yields a local, spatially robust fill.
    _, indices = distance_transform_edt(bad, return_indices=True)
    seeded = data.copy()
    seeded[bad] = data[tuple(indices[:, bad])]
    from scipy.ndimage import median_filter
    local = median_filter(seeded, size=kernel_size, mode='nearest')
    result = data.copy()
    result[bad] = local[bad]
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise RuntimeError('RN interpolation did not produce a finite positive map')
    return result, bad


def make_cds_from_file(fname, ext=0, read0=0, read1=1, already_cds=False):
    with fits.open(fname, memmap=True) as hdul:
        data = np.asarray(hdul[ext].data, dtype=np.float64)
    if already_cds:
        if data.ndim != 2:
            raise ValueError(f'{fname}: expected 2D CDS')
        return data
    if data.ndim != 3 or not (0 <= read0 < data.shape[0]) or not (0 <= read1 < data.shape[0]) or read0 == read1:
        raise ValueError(f'{fname}: invalid ramp dimensions or read indices')
    return data[read1] - data[read0]


def make_readnoise_header(source_header, *, gain, n_files, method, read0, read1,
                          already_cds, clip_sigma, min_good, n_channels,
                          ref_width, subtract_global, subtract_amp,
                          subtract_row, exclude_ref_pixels, fill_value,
                          n_bad, n_pairs, interpolation_kernel):
    """Copy detector/exposure metadata and append read-noise provenance."""
    hdr = source_header.copy() if source_header is not None else fits.Header()
    # Structural FITS cards are regenerated by the writer.
    for key in list(hdr):
        if key in ('SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'PCOUNT', 'GCOUNT') or (key.startswith('NAXIS') and key[5:].isdigit()):
            del hdr[key]
    hdr['CALTYPE'] = ('READNOISE', 'Calibration product')
    hdr['BUNIT'] = ('electron', 'Single-read RMS noise unit')
    hdr['RNMETHOD'] = (method, 'Noise estimation method')
    hdr['GAIN'] = (float(gain), 'Electrons per DN')
    hdr['NFILES'] = (int(n_files), 'Input dark exposures')
    hdr['NPAIRS'] = (int(n_pairs), 'Independent CDS difference pairs')
    hdr['READ0'] = (int(read0), 'First CDS read (zero based)')
    hdr['READ1'] = (int(read1), 'Second CDS read (zero based)')
    hdr['ALRCDS'] = (bool(already_cds), 'Input files already CDS')
    hdr['CLIPSIG'] = (float(clip_sigma), 'Sigma clip threshold')
    hdr['MINGOOD'] = (int(min_good), 'Minimum valid samples per pixel')
    hdr['NCHANS'] = (int(n_channels), 'Number of amplifier channels')
    hdr['REFWID'] = (int(ref_width), 'Reference pixel border width')
    hdr['SUBGLOB'] = bool(subtract_global)
    hdr['SUBAMP'] = bool(subtract_amp)
    hdr['SUBROW'] = bool(subtract_row)
    hdr['EXCLREF'] = bool(exclude_ref_pixels)
    hdr['FILLRN'] = (float(fill_value), 'Median valid RN in electrons (diagnostic)')
    hdr['RNINTERP'] = ('LOCALMED', 'Bad RN pixel interpolation')
    hdr['RNKERN'] = (int(interpolation_kernel), 'Local interpolation kernel width')
    hdr['NBADRN'] = (int(n_bad), 'Unreliable or excluded pixels')
    hdr.add_history('Read noise estimated from corrected dark CDS exposures.')
    hdr.add_history('BAD_RN=1 indicates interpolated or excluded pixels.')
    hdr.add_history('Nonpositive and nonfinite RN values replaced using neighbors.')
    hdr.add_history('pair_diff uses non-overlapping independent CDS pairs.')
    return hdr



def readnoise_from_cds_dark_exposures(
    rn_data,
    source_header=None,
    *,
    gain=1.0,
    read0=0,
    read1=1,
    n_channels=4,
    ref_width=4,
    subtract_global=True,
    subtract_amp=True,
    subtract_row=True,
    exclude_ref_pixels=False,
    clip_sigma=5.0,
    min_good=2,
    method="pair_diff",
    fill_bad_with="median",
    interpolation_kernel=3,
    interpolation_iterations=8,
):
    """
    Estimate single-read noise from a stack of corrected dark ramps.

    Parameters
    ----------
    rn_data : ndarray
        Input ramp stack with shape:
        (n_exposures, n_reads, ny, nx).

        Input data are assumed to have undergone ACN and
        1/f correction and to be expressed in DN.

    source_header : fits.Header, optional
        Header from the original dark exposure.

    gain : float
        Detector gain in electrons/DN.

    read0, read1 : int
        Zero-based read indices used to construct CDS images.

    n_channels : int
        Number of detector amplifier channels.

    ref_width : int
        Width of the reference-pixel border.

    subtract_global, subtract_amp, subtract_row : bool
        Apply additional common-offset subtraction to each
        CDS image before estimating noise.

    exclude_ref_pixels : bool
        Exclude reference pixels from the measured RN map
        and replace them through interpolation.

    clip_sigma : float
        Sigma-clipping threshold.

    min_good : int
        Minimum number of valid samples per pixel.

    method : {"pair_diff", "stack_std"}
        pair_diff:
            Construct non-overlapping CDS exposure pairs.
            RN = std(CDS_i - CDS_j) / 2.

        stack_std:
            RN = std(CDS exposures) / sqrt(2).

    fill_bad_with : {"median", float}
        Diagnostic fallback value recorded in the header.
        Actual invalid pixels are interpolated using neighbors.

    interpolation_kernel : int
        Odd local interpolation kernel size.

    interpolation_iterations : int
        Passed to the interpolation function.

    Returns
    -------
    rn_e : ndarray
        Single-read noise map in electrons.
        All values are finite and strictly positive.

    cds_noise_dn : ndarray
        Equivalent CDS noise map in DN.

    n_good : ndarray
        Number of valid samples used per pixel.

    bad_rn : ndarray
        Boolean mask identifying unreliable, excluded,
        or interpolated pixels.

    hdr : fits.Header
        FITS header containing calibration metadata.

    Notes
    -----
    Assumes independent, equally noisy reads and
    negligible dark-current shot noise.
    """

    # ---------------------------------------------------------
    # 1. Validate input data
    # ---------------------------------------------------------

    rn_data = np.asarray(rn_data)

    if rn_data.ndim != 4:
        raise ValueError(
            "rn_data must have shape "
            "(n_exposures, n_reads, ny, nx). "
            f"Received {rn_data.shape}."
        )

    n_files, n_reads, ny, nx = rn_data.shape

    if method not in ("pair_diff", "stack_std"):
        raise ValueError(
            "method must be 'pair_diff' or 'stack_std'."
        )

    if not np.isfinite(gain) or gain <= 0:
        raise ValueError(
            "gain must be finite and positive."
        )

    if not (0 <= read0 < n_reads):
        raise ValueError(
            f"Invalid read0={read0}; n_reads={n_reads}."
        )

    if not (0 <= read1 < n_reads):
        raise ValueError(
            f"Invalid read1={read1}; n_reads={n_reads}."
        )

    if read0 == read1:
        raise ValueError(
            "read0 and read1 must be different."
        )

    if read1 <= read0:
        raise ValueError(
            "read1 must be greater than read0."
        )

    if min_good < 2:
        raise ValueError(
            "min_good must be at least 2."
        )

    if not np.isfinite(clip_sigma) or clip_sigma <= 0:
        raise ValueError(
            "clip_sigma must be finite and positive."
        )

    if method == "pair_diff":
        minimum_files = max(4, 2 * min_good)
    else:
        minimum_files = max(2, min_good)

    if n_files < minimum_files:
        raise ValueError(
            f"Insufficient dark exposures for {method}. "
            f"Found {n_files}, require at least "
            f"{minimum_files}."
        )

    if not (
        isinstance(fill_bad_with, str)
        and fill_bad_with == "median"
    ):
        try:
            fill_value = float(fill_bad_with)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fill_bad_with must be 'median' "
                "or a finite nonnegative number."
            ) from exc

        if not np.isfinite(fill_value) or fill_value < 0:
            raise ValueError(
                "fill_bad_with must be nonnegative and finite."
            )

    # ---------------------------------------------------------
    # 2. Construct CDS images
    # ---------------------------------------------------------

    # Convert only the selected reads to float64.
    # Avoid converting the entire 4D ramp stack.

    cds_stack = (
        rn_data[:, read1, :, :].astype(np.float64)
        - rn_data[:, read0, :, :].astype(np.float64)
    )

    # ---------------------------------------------------------
    # 3. Remove residual common offsets
    # ---------------------------------------------------------

    for i in range(n_files):

        cds_stack[i] = subtract_common_offsets(
            cds_stack[i],
            n_channels=n_channels,
            subtract_global=subtract_global,
            subtract_amp=subtract_amp,
            subtract_row=subtract_row,
            ref_width=ref_width,
            active_only=True,
        )

    # ---------------------------------------------------------
    # 4. Construct noise-estimation samples
    # ---------------------------------------------------------

    if method == "pair_diff":

        # Independent, non-overlapping exposure pairs:
        # (0,1), (2,3), (4,5), ...

        n_pairs = n_files // 2

        samples = (
            cds_stack[1:2*n_pairs:2]
            - cds_stack[0:2*n_pairs:2]
        )

        # Var(CDS_i - CDS_j) = 4 * RN^2
        scale = 2.0

    else:

        samples = cds_stack
        n_pairs = 0

        # Var(CDS) = 2 * RN^2
        scale = np.sqrt(2.0)

    # ---------------------------------------------------------
    # 5. Sigma-clipped noise estimation
    # ---------------------------------------------------------

    noise_dn, n_good, bad_rn = (
        sigma_clip_stack_no_nan(
            samples,
            sigma=clip_sigma,
            min_good=min_good,
        )
    )

    # Convert measured noise to electrons
    raw_rn = noise_dn * gain / scale

    # ---------------------------------------------------------
    # 6. Identify unreliable and nonphysical pixels
    # ---------------------------------------------------------

    bad_rn = np.asarray(bad_rn, dtype=bool).copy()

    bad_rn |= (
        ~np.isfinite(raw_rn)
        | (raw_rn <= 0)
        | (n_good < min_good)
    )

    # Optionally exclude reference pixels
    if exclude_ref_pixels and ref_width > 0:

        bad_rn[:ref_width, :] = True
        bad_rn[-ref_width:, :] = True

        bad_rn[:, :ref_width] = True
        bad_rn[:, -ref_width:] = True

    # ---------------------------------------------------------
    # 7. Determine diagnostic median
    # ---------------------------------------------------------

    reliable = (
        ~bad_rn
        & np.isfinite(raw_rn)
        & (raw_rn > 0)
    )

    if not np.any(reliable):
        raise ValueError(
            "No reliable positive read-noise measurements "
            "available for interpolation."
        )

    if fill_bad_with == "median":

        fill_rn = float(
            np.median(raw_rn[reliable])
        )

    else:

        fill_rn = float(fill_bad_with)

    # ---------------------------------------------------------
    # 8. Interpolate unreliable pixels
    # ---------------------------------------------------------

    rn_e, bad_rn = interpolate_bad_readnoise(
        raw_rn,
        bad_rn,
        kernel_size=interpolation_kernel,
        max_iterations=interpolation_iterations,
    )

    # ---------------------------------------------------------
    # 9. Construct equivalent CDS noise map
    # ---------------------------------------------------------

    cds_noise_dn = (
        rn_e * np.sqrt(2.0) / gain
    )

    # ---------------------------------------------------------
    # 10. Final validation
    # ---------------------------------------------------------

    if not np.all(np.isfinite(rn_e)):
        raise RuntimeError(
            "Read-noise map contains NaNs or infinities "
            "after interpolation."
        )

    if np.any(rn_e <= 0):
        raise RuntimeError(
            "Read-noise map contains nonpositive values "
            "after interpolation."
        )

    if not np.all(np.isfinite(cds_noise_dn)):
        raise RuntimeError(
            "CDS noise map contains invalid values."
        )

    # ---------------------------------------------------------
    # 11. Create FITS header
    # ---------------------------------------------------------

    hdr = make_readnoise_header(
        source_header,
        gain=gain,
        n_files=n_files,
        method=method,
        read0=read0,
        read1=read1,
        already_cds=False,
        clip_sigma=clip_sigma,
        min_good=min_good,
        n_channels=n_channels,
        ref_width=ref_width,
        subtract_global=subtract_global,
        subtract_amp=subtract_amp,
        subtract_row=subtract_row,
        exclude_ref_pixels=exclude_ref_pixels,
        fill_value=fill_rn,
        n_bad=int(np.count_nonzero(bad_rn)),
        n_pairs=n_pairs,
        interpolation_kernel=interpolation_kernel,
    )

    # ---------------------------------------------------------
    # 12. Return calibration products
    # ---------------------------------------------------------

    return rn_e, cds_noise_dn, n_good, bad_rn, hdr


############plotting###########################################################

def plot_readnoise_diagnostics(
    rn,
    cds_noise,
    n_good,
    bad_rn,
    output_dir,
    *,
    filename="readnoise_diagnostics.png",
    title="SCALES Read-Noise Calibration",
    dpi=200,
    show=False,
):
    """
    Create and save a 2x2 read-noise diagnostic figure.

    Parameters
    ----------
    rn : ndarray
        2D single-read noise map in electrons.
    cds_noise : ndarray
        2D CDS noise map in DN. Used for consistency checks.
    n_good : ndarray
        2D number of valid samples per pixel.
    bad_rn : ndarray
        2D boolean mask; True indicates interpolated/excluded pixels.
    output_dir : str or Path
        Directory where the figure will be saved.
    filename : str
        Output figure filename.
    title : str
        Figure title.
    dpi : int
        Figure resolution.
    show : bool
        Display figure interactively.

    Returns
    -------
    Path
        Saved figure path.
    """

    rn = np.asarray(rn, dtype=float)
    cds_noise = np.asarray(cds_noise, dtype=float)
    n_good = np.asarray(n_good)
    bad_rn = np.asarray(bad_rn, dtype=bool)

    if rn.ndim != 2:
        raise ValueError("Read-noise map must be 2D.")

    if any(arr.shape != rn.shape for arr in
           (cds_noise, n_good, bad_rn)):
        raise ValueError("All input arrays must have identical shapes.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    savepath = output_dir / filename

    # Measured, physically valid pixels only
    valid = (
        np.isfinite(rn)
        & (rn > 0)
        & (~bad_rn)
        & (n_good >= 2)
    )

    if not np.any(valid):
        raise ValueError(
            "No valid measured read-noise pixels available."
        )

    values = rn[valid]

    median_rn = np.median(values)
    p5, p95 = np.percentile(values, [5, 95])
    vmin, vmax = np.percentile(values, [1, 99])

    if vmax <= vmin:
        vmax = vmin + max(abs(vmin) * 0.01, 1e-6)

    bad_fraction = 100.0 * np.mean(bad_rn)

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10),
        constrained_layout=True
    )

    fig.suptitle(title, fontsize=16, fontweight="bold")

    # --------------------------------------------------
    # Panel 1: Read-noise spatial map
    # --------------------------------------------------
    ax = axes[0, 0]

    im = ax.imshow(
        rn,
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    fig.colorbar(
        im, ax=ax,
        label="Single-read noise (e⁻)",
        fraction=0.046,
        pad=0.04,
    )

    ax.set_title("Read-Noise Map")
    ax.set_xlabel("Detector X (pixel)")
    ax.set_ylabel("Detector Y (pixel)")

    # --------------------------------------------------
    # Panel 2: Read-noise distribution
    # --------------------------------------------------
    ax = axes[0, 1]

    ax.hist(
        values,
        bins=100,
        range=(p5, p95),
        histtype="stepfilled",
        alpha=0.7,
        color="steelblue",
    )

    ax.axvline(
        median_rn, color="red", linestyle="--",
        linewidth=1.8,
        label=f"Median = {median_rn:.2f} e⁻"
    )

    ax.axvline(
        p5, color="black", linestyle=":",
        label=f"P5 = {p5:.2f} e⁻"
    )

    ax.axvline(
        p95, color="black", linestyle=":",
        label=f"P95 = {p95:.2f} e⁻"
    )

    ax.set_title("Read-Noise Distribution (Measured Pixels)")
    ax.set_xlabel("Single-read noise (e⁻)")
    ax.set_ylabel("Number of pixels")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # --------------------------------------------------
    # Panel 3: Number of valid samples
    # --------------------------------------------------
    ax = axes[1, 0]

    im = ax.imshow(
        n_good,
        origin="lower",
        cmap="cividis",
        interpolation="nearest",
        vmin=0,
        vmax=max(int(np.max(n_good)), 1),
    )

    fig.colorbar(
        im, ax=ax,
        label="Valid samples",
        fraction=0.046,
        pad=0.04,
    )

    ax.set_title(
        f"Valid Samples (Median = {np.median(n_good):.0f})"
    )
    ax.set_xlabel("Detector X (pixel)")
    ax.set_ylabel("Detector Y (pixel)")

    # --------------------------------------------------
    # Panel 4: Bad/interpolated pixel mask
    # --------------------------------------------------
    ax = axes[1, 1]

    cmap = ListedColormap(["#243746", "#e76f51"])

    im = ax.imshow(
        bad_rn.astype(int),
        origin="lower",
        cmap=cmap,
        vmin=-0.5,
        vmax=1.5,
        interpolation="nearest",
    )

    cbar = fig.colorbar(
        im, ax=ax,
        ticks=[0, 1],
        fraction=0.046,
        pad=0.04,
    )
    cbar.ax.set_yticklabels(["Valid", "Interpolated / Bad"])

    ax.set_title(
        f"Bad / Interpolated Pixels ({bad_fraction:.3f}%)"
    )
    ax.set_xlabel("Detector X (pixel)")
    ax.set_ylabel("Detector Y (pixel)")

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    fig.savefig(
        savepath,
        dpi=dpi,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)

    return savepath















