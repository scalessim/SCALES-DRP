import numpy as np
from astropy.io import fits
from pathlib import Path


########### readnoise estimation ##################################
#input is a set of ACN & 1/f corrected cube of reads

def robust_sigma_mad(x, axis=0):
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis)
    return 1.4826 * mad


def subtract_common_offsets(
    image,
    n_channels=4,
    subtract_global=True,
    subtract_amp=True,
    subtract_row=False,
    ref_width=4,
    active_only=True,
):
    img = image.astype(np.float64).copy()
    ny, nx = img.shape

    if active_only:
        valid = np.zeros_like(img, dtype=bool)
        valid[ref_width:ny-ref_width, ref_width:nx-ref_width] = True
        valid &= np.isfinite(img)
    else:
        valid = np.isfinite(img)

    if subtract_global and np.any(valid):
        img -= np.nanmedian(img[valid])

    if subtract_amp:
        ch_width = nx // n_channels

        for ch in range(n_channels):
            x1 = ch * ch_width
            x2 = (ch + 1) * ch_width if ch < n_channels - 1 else nx

            amp_mask = np.zeros_like(img, dtype=bool)
            amp_mask[:, x1:x2] = True
            amp_mask &= valid

            if np.any(amp_mask):
                img[:, x1:x2] -= np.nanmedian(img[amp_mask])

    if subtract_row:
        for y in range(ny):
            row_mask = valid[y, :]
            if np.any(row_mask):
                img[y, :] -= np.nanmedian(img[y, row_mask])

    return img


def sigma_clip_stack_no_nan(stack, sigma=5.0, min_good=2, fill_value=0.0):
    """
    Sigma-clipped std image with no NaNs in the returned std map.

    Returns
    -------
    std_filled : 2D array
        Finite sigma-clipped standard deviation map.

    n_good : 2D array
        Number of samples used per pixel.

    bad : 2D bool
        True where the value was weakly measured or filled.
    """

    stack = np.asarray(stack, dtype=np.float64)

    med = np.nanmedian(stack, axis=0)
    sig_mad = robust_sigma_mad(stack, axis=0)
    sig_std = np.nanstd(stack, axis=0, ddof=1)

    sig = sig_mad.copy()

    bad_sig = (~np.isfinite(sig)) | (sig <= 0)
    sig[bad_sig] = sig_std[bad_sig]

    bad_sig = (~np.isfinite(sig)) | (sig <= 0)

    good = np.isfinite(stack)

    good_clip = np.abs(stack - med[None, :, :]) < sigma * sig[None, :, :]

    # If sigma is invalid, do not reject by clipping; use all finite samples.
    good &= good_clip | bad_sig[None, :, :]

    clipped = np.where(good, stack, np.nan)

    n_good = np.sum(np.isfinite(clipped), axis=0)
    std = np.nanstd(clipped, axis=0, ddof=1)

    bad = (~np.isfinite(std)) | (std < 0) | (n_good < min_good)

    finite = np.isfinite(std) & (std >= 0)

    if np.any(finite):
        replacement = np.nanmedian(std[finite])
    else:
        replacement = fill_value

    std_filled = std.copy()
    std_filled[bad] = replacement
    std_filled[~np.isfinite(std_filled)] = replacement

    return std_filled, n_good, bad


def make_cds_from_file(
    fname,
    ext=0,
    read0=0,
    read1=1,
    already_cds=False,
):
    data = fits.getdata(fname, ext=ext).astype(np.float64)

    if already_cds:
        if data.ndim != 2:
            raise ValueError(f"{fname} should be a 2D CDS image.")
        return data

    if data.ndim != 3:
        raise ValueError(f"{fname} should be a 3D ramp cube.")

    if data.shape[0] <= max(read0, read1):
        raise ValueError(
            f"{fname} has only {data.shape[0]} reads, "
            f"but read0={read0}, read1={read1}."
        )

    return data[read1] - data[read0]


def readnoise_from_cds_dark_exposures(
    cds_files,
    gain=1.0,
    ext=0,
    read0=0,
    read1=1,
    already_cds=False,
    n_channels=4,
    ref_width=4,
    subtract_global=True,
    subtract_amp=True,
    subtract_row=False,
    exclude_ref_pixels=False,
    clip_sigma=5.0,
    min_good=2,
    method="pair_diff",
    fill_bad_with="median",
    output_file="readnoise_from_cds_dark_v3.fits",
):
    """
    Estimate single-read noise from corrected CDS dark exposures.

    This version guarantees no NaNs in the final readnoise map.

    method = "stack_std":
        RN = std(CDS_i) / sqrt(2)

    method = "pair_diff":
        RN = std(CDS_i - CDS_j) / 2

    Notes
    -----
    For pair_diff, the number of samples is Nfiles - 1.
    Therefore min_good should usually be small, e.g. 2 or 3.
    """

    cds_files = list(cds_files)

    if len(cds_files) < 2:
        raise ValueError("Need at least 2 dark files.")

    if method == "pair_diff" and len(cds_files) < 3:
        print("Warning: pair_diff has only one pair. Output will be weakly measured.")

    cds_images = []

    for fname in cds_files:
        cds = make_cds_from_file(
            fname,
            ext=ext,
            read0=read0,
            read1=read1,
            already_cds=already_cds,
        )

        cds = subtract_common_offsets(
            cds,
            n_channels=n_channels,
            subtract_global=subtract_global,
            subtract_amp=subtract_amp,
            subtract_row=subtract_row,
            ref_width=ref_width,
            active_only=True,
        )

        cds_images.append(cds)

    cds_stack = np.asarray(cds_images, dtype=np.float64)

    if method == "stack_std":
        cds_noise_dn, n_good, bad_rn = sigma_clip_stack_no_nan(
            cds_stack,
            sigma=clip_sigma,
            min_good=min_good,
        )

        readnoise_dn = cds_noise_dn / np.sqrt(2.0)

    elif method == "pair_diff":
        pair_diffs = []

        for i in range(len(cds_images) - 1):
            pair_diffs.append(cds_images[i + 1] - cds_images[i])

        pair_stack = np.asarray(pair_diffs, dtype=np.float64)

        pair_noise_dn, n_good, bad_rn = sigma_clip_stack_no_nan(
            pair_stack,
            sigma=clip_sigma,
            min_good=max(2, min_good),
        )

        # Difference of two CDS images:
        # Var(CDS_i - CDS_j) = 4 RN^2
        readnoise_dn = pair_noise_dn / 2.0

        # Equivalent CDS noise:
        # CDS noise = sqrt(2) * RN
        cds_noise_dn = readnoise_dn * np.sqrt(2.0)

    else:
        raise ValueError("method must be 'stack_std' or 'pair_diff'.")

    readnoise_e = readnoise_dn * gain

    finite = np.isfinite(readnoise_e) & (readnoise_e >= 0)

    if fill_bad_with == "median":
        if np.any(finite):
            fill_rn = np.nanmedian(readnoise_e[finite])
        else:
            fill_rn = 0.0
    elif isinstance(fill_bad_with, (int, float)):
        fill_rn = float(fill_bad_with)
    else:
        raise ValueError("fill_bad_with must be 'median' or a number.")

    bad_rn |= ~finite
    readnoise_e[~finite] = fill_rn

    cds_finite = np.isfinite(cds_noise_dn) & (cds_noise_dn >= 0)

    if np.any(cds_finite):
        fill_cds = np.nanmedian(cds_noise_dn[cds_finite])
    else:
        fill_cds = 0.0

    cds_noise_dn[~cds_finite] = fill_cds

    ny, nx = readnoise_e.shape

    if exclude_ref_pixels:
        bad_rn[:ref_width, :] = True
        bad_rn[-ref_width:, :] = True
        bad_rn[:, :ref_width] = True
        bad_rn[:, -ref_width:] = True

        # Still no NaNs in the primary map.
        readnoise_e[:ref_width, :] = fill_rn
        readnoise_e[-ref_width:, :] = fill_rn
        readnoise_e[:, :ref_width] = fill_rn
        readnoise_e[:, -ref_width:] = fill_rn

    hdr = fits.Header()
    hdr["BUNIT"] = "e-"
    hdr["RNMETHOD"] = method
    hdr["GAIN"] = gain
    hdr["NFILES"] = len(cds_files)
    hdr["READ0"] = read0
    hdr["READ1"] = read1
    hdr["ALRCDS"] = already_cds
    hdr["CLIPSIG"] = clip_sigma
    hdr["MINGOOD"] = min_good
    hdr["SUBGLOB"] = subtract_global
    hdr["SUBAMP"] = subtract_amp
    hdr["SUBROW"] = subtract_row
    hdr["EXCLREF"] = exclude_ref_pixels
    hdr["FILLRN"] = float(fill_rn)
    hdr["COMMENT"] = "Finite single-read noise map from corrected CDS dark exposures."
    hdr["COMMENT"] = "BAD_RN marks pixels that were filled or weakly measured."
    hdr["COMMENT"] = "stack_std: RN = std(CDS_i)/sqrt(2)."
    hdr["COMMENT"] = "pair_diff: RN = std(CDS_i-CDS_j)/2."

    hdul = fits.HDUList([
        fits.PrimaryHDU(readnoise_e.astype(np.float32), header=hdr),
        fits.ImageHDU(cds_noise_dn.astype(np.float32), name="CDS_NOISE_DN"),
        fits.ImageHDU(n_good.astype(np.int16), name="N_GOOD"),
        fits.ImageHDU(bad_rn.astype(np.uint8), name="BAD_RN"),
    ])

    hdul.writeto(output_file, overwrite=True)

    print("Wrote:", output_file)
    print("Readnoise median:", np.nanmedian(readnoise_e))
    print("Readnoise p5/p95:", np.nanpercentile(readnoise_e, [5, 95]))
    print("Bad/filled pixels:", np.sum(bad_rn), "/", bad_rn.size)
    print("Any NaN in RN map?", np.any(~np.isfinite(readnoise_e)))

    return readnoise_e, cds_noise_dn, n_good, bad_rn

#cds_dark_files = sorted(Path(path_ifs_dark).glob("*.fits"))

#rn_pair, cds_noise_pair, n_good_pair, bad_rn_pair = readnoise_from_cds_dark_exposures(
#    cds_dark_files,
#    gain=1.0,
#    ext=0,
#    read0=1,
#    read1=2,
#    already_cds=False,
#    subtract_global=False,
#    subtract_amp=True,
#    subtract_row=True,
#    method="pair_diff",
#    min_good=2,
#    exclude_ref_pixels=False,
#    output_file="readnoise_ifs_fast1.0_cd5.fits",
#)
###############################################################################













