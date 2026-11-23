"""
Measure the spot shift between a single monochromator frame and an existing
monochromator scan, and write shifted interpolated rectmats.

The single frame's L1 (made by StartCalibSA), minus the L1 of a background
frame taken with it (required; e.g. the monochromator at a wavelength
>~200 nm away), is compared with the scan's background-subtracted image at
the same wavelength. The shift is measured in two steps:

1. Cross-correlation of the two high-passed images, with the peak searched
   in a box around (rmat_shift_guess_dx, rmat_shift_guess_dy) just under
   half the lenslet pitch on each axis, so it cannot jump to a neighbouring
   lenslet.
2. Spot-by-spot refinement: every spot in the scan's Gaussian fit at that
   wavelength is centroided in both images with the same estimator, and the
   sigma-clipped median of the per-spot offsets is the final shift.
3. Optional (rmat_shift_affine = True): an affine fit (shift + scale +
   rotation + shear) to the per-spot offsets, for frames whose spot pattern
   also changed scale.

The shift (or the affine offset at each spot position) is then added to the
x and y centres of the scan's interpolated Gaussian spot parameters
({scmode}_gauss_spotpars_interpd.fits) and the OPT and C2 interpolated
rectmats are rebuilt from them, the same way ProcessMonochrom builds its
rectmat_xshift / rectmat_yshift rectmats.
"""
import os
import csv
import datetime

import numpy as np
import astropy.io.fits as pyfits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scalesdrp.primitives.ProcessMonochrom import ProcessMonochrom
from scalesdrp.primitives.scales_basic import fits_headers_to_dataframe
import scalesdrp.primitives.scales_basic as scbasic


# origin of the affine fit: the detector centre, so its constant terms are
# the shift there
AFFINE_X0 = AFFINE_Y0 = 1024.0


KNOWN_IFSMODES = ("LowRes-K", "LowRes-KL", "LowRes-KLM", "LowRes-Ls",
                  "LowRes-L", "LowRes-M", "MedRes-K", "MedRes-L", "MedRes-M")


def ifsmode_of(hdr, logger=None, name=""):
    """
    IFS mode of a header (or dataframe row): the IFSMODE keyword when it is a
    known mode, else derived from MODSLNAM / DSPRSNAM / FILTER with
    select_ifsmode. Warns when both are available and disagree.
    """
    raw = hdr.get("IFSMODE")
    raw = None if raw is None else str(raw).strip()
    from_kw = None
    if raw and raw.lower() not in ("nan", "none", "unknown"):
        from_kw = {m.upper(): m for m in KNOWN_IFSMODES}.get(
            raw.upper().replace("_", "-"))
        if from_kw is None and logger is not None:
            logger.warning(f"{name}: IFSMODE = {raw!r} is not a known mode; "
                           "using MODSLNAM/DSPRSNAM instead.")
    derived = scbasic.select_ifsmode(
        hdr.get("MODSLNAM"), hdr.get("DSPRSNAM"), hdr.get("FILTER"))
    if from_kw is not None:
        if derived is not None and derived != from_kw and logger is not None:
            logger.warning(
                f"{name}: IFSMODE = {from_kw} but MODSLNAM/DSPRSNAM give "
                f"{derived}; using IFSMODE.")
        return from_kw
    return derived


def _cfg(inst, name, default):
    """Config value, or default when the key is missing or None."""
    val = getattr(inst, name, None)
    return default if val is None else val


class RmatShift(ProcessMonochrom):
    """
    Measure the x/y spot shift of a monochromator frame relative to a scan,
    and write shifted OPT and C2 interpolated rectmats.
    """

    # ------------------------------------------------------------------
    # scan lookup
    # ------------------------------------------------------------------
    def find_matching_mcalunit(self, scan_dir, scmode, monowave, tol_nm):
        """
        Find the mcalunit image in scan_dir with the same IFS mode and the
        wavelength closest to monowave (nm). Raises if none is within tol_nm.
        """
        df = fits_headers_to_dataframe(scan_dir, pattern="*mcalunit.fits")
        # warn about IFSMODE/MODSLNAM disagreements once, not per file
        modes = df.apply(lambda row: ifsmode_of(row), axis=1)
        if "IFSMODE" in df.columns:
            known = {m.upper() for m in KNOWN_IFSMODES}
            raw = set(df["IFSMODE"].astype(str).str.strip())
            bad = sorted(v for v in raw - {"nan", "None", ""}
                         if v.upper().replace("_", "-") not in known)
            if bad:
                self.logger.warning(
                    f"Scan IFSMODE values {bad} were not used as given "
                    "(unknown mode names); fell back to MODSLNAM/DSPRSNAM.")
        df["IFSMODE"] = modes
        df = df[df["IFSMODE"] == scmode]
        if df.empty:
            raise RuntimeError(
                f"No {scmode} mcalunit images found in {scan_dir}.")

        waves = df["MONOWAVE"].astype(float)
        dl = (waves - monowave).abs()
        best = dl.idxmin()
        if dl[best] > tol_nm:
            nearest = waves.iloc[np.argsort(dl.values)[:3]].values
            raise RuntimeError(
                f"No {scmode} mcalunit within {tol_nm} nm of {monowave} nm in "
                f"{scan_dir}. Nearest scan wavelengths: "
                f"{', '.join(f'{w:.3f}' for w in nearest)} nm. "
                "Increase wave_match_tol_nm in the config to accept one.")
        return df.loc[best, "filename"], float(waves[best]), df.loc[best], df

    def scan_mean_image(self, scan_dir, df, scmode, cache_dir):
        """
        Mean of all the scan's mcalunit images for this mode: the background
        template ProcessMonochrom.monochrom_bksub subtracts. Cached in
        cache_dir as {scmode}_scan_meanim.fits, reused while the scan
        directory and file count match.
        """
        cache = os.path.join(cache_dir, f"{scmode}_scan_meanim.fits")
        scan_abs = os.path.abspath(scan_dir)
        if os.path.isfile(cache):
            h = pyfits.getheader(cache)
            if h.get("SCANDIR") == scan_abs[-60:] and h.get("NIMAGES") == len(df):
                self.logger.info(f"Using cached scan mean image {cache}")
                return pyfits.getdata(cache).astype(float)

        self.logger.info(f"Averaging {len(df)} scan mcalunit images")
        total = None
        count = None
        for name in df["filename"]:
            im = pyfits.getdata(os.path.join(scan_dir, name)).astype(float)
            good = np.isfinite(im)
            if total is None:
                total = np.zeros_like(im)
                count = np.zeros(im.shape, dtype=np.int32)
            total[good] += im[good]
            count += good
        with np.errstate(invalid="ignore", divide="ignore"):
            meanim = total / count
        hdr = pyfits.Header()
        hdr["SCANDIR"] = scan_abs[-60:]
        hdr["NIMAGES"] = (len(df), "mcalunit images averaged")
        hdr["IFSMODE"] = scmode
        pyfits.writeto(cache, meanim.astype(np.float32), hdr, overwrite=True)
        return meanim

    def calibrate_like_scan(self, im, meanim, destripe):
        """
        Same background removal ProcessMonochrom applies to the scan images:
        subtract the scan mean image scaled to this image's mean level, then
        (optionally) the masked row destripe.
        """
        cal = im - meanim * np.nanmean(im) / np.nanmean(meanim)
        cal = np.nan_to_num(cal, nan=0.0)
        if destripe:
            cal = self.masked_row_destripe(cal)[0]
        return cal

    def background_spot_separation(self, scan_dir, scmode, lams, monowave,
                                   bkg_waves):
        """
        Median distance (px) between the spots at monowave and those at the
        nearest background-frame wavelength, from the scan's spot fits.
        None when no background frame has a MONOWAVE (e.g. monochromator
        off), so there are no background spots.
        """
        waves = [w for w in bkg_waves if w is not None and np.isfinite(w)]
        if not waves:
            return None
        sp = pyfits.getdata(os.path.join(scan_dir, f"{scmode}_gauss_spotpars.fits"))
        lnm = np.asarray(lams, dtype=float) * 1000.0
        i0 = int(np.argmin(np.abs(lnm - monowave)))
        seps = []
        for w in waves:
            i1 = int(np.argmin(np.abs(lnm - w)))
            d = np.hypot(sp[i1][..., 1] - sp[i0][..., 1],
                         sp[i1][..., 2] - sp[i0][..., 2])
            sep = float(np.nanmedian(d))
            if not lnm.min() <= w <= lnm.max():
                # outside the scan: extrapolate linearly in wavelength
                sep *= abs(w - monowave) / max(abs(lnm[i1] - monowave), 1e-6)
            seps.append(sep)
        return min(seps)

    def load_background(self, bkg_frames, output_dir, hdr):
        """
        Mean of the background frames' L1 images, and their MONOWAVE values
        (None where missing).
        """
        ims, waves = [], []
        for b in bkg_frames:
            path = scbasic.get_l1_path_from_raw(b, output_dir)
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"Background L1 {path} not found; the background frame "
                    "must be reduced before the monochromator frame.")
            with pyfits.open(path, memmap=False) as hdul:
                ims.append(np.asarray(hdul[0].data, dtype=float))
                bh = hdul[0].header
            try:
                waves.append(float(bh["MONOWAVE"]))
            except (KeyError, TypeError, ValueError):
                waves.append(None)
            for key in ("EXPTIME", "MCLOCK", "IFSMODE", "MODSLNAM", "DSPRSNAM"):
                if str(bh.get(key)).strip() != str(hdr.get(key)).strip():
                    self.logger.warning(
                        f"Background {os.path.basename(path)} has {key}="
                        f"{bh.get(key)}, frame has {hdr.get(key)}.")
        im = np.nanmean(ims, axis=0) if len(ims) > 1 else ims[0]
        return im, waves

    @staticmethod
    def load_scan_calim(scan_dir, scmode, ll):
        """Slice ll of the scan's background-subtracted stack, if saved."""
        path = os.path.join(scan_dir, f"{scmode}_calim_stack.fits")
        if not os.path.isfile(path):
            return None
        with pyfits.open(path, memmap=True) as hdul:
            return np.array(hdul[0].section[ll], dtype=float)

    # ------------------------------------------------------------------
    # shift measurement
    # ------------------------------------------------------------------
    @staticmethod
    def highpass(im, sigma=8.0):
        """Remove smooth background; NaNs set to 0. Returns (image, noise)."""
        im = np.array(im, dtype=float)
        bad = ~np.isfinite(im)
        if bad.any():
            im[bad] = np.nanmedian(im)
        hp = im - gaussian_filter(im, sigma)
        hp[bad] = 0.0
        _, med, std = sigma_clipped_stats(hp[::4, ::4], sigma=3.0, maxiters=5)
        return hp - med, std

    @staticmethod
    def centroid(im, x, y, half, niter=3):
        """
        Iterative centre-of-light centroid in a (2*half+1)^2 stamp, with the
        stamp border median as background. Returns (x, y, peak).
        """
        ny, nx = im.shape
        peak = np.nan
        for _ in range(niter):
            if not (np.isfinite(x) and np.isfinite(y)):
                return np.nan, np.nan, np.nan
            xi, yi = int(round(x)), int(round(y))
            if xi - half < 0 or yi - half < 0 or xi + half >= nx or yi + half >= ny:
                return np.nan, np.nan, np.nan
            st = im[yi - half:yi + half + 1, xi - half:xi + half + 1]
            border = np.concatenate([st[0], st[-1], st[1:-1, 0], st[1:-1, -1]])
            w = st - np.median(border)
            peak = w.max()
            w = np.clip(w, 0, None)
            f = w.sum()
            if f <= 0:
                return np.nan, np.nan, np.nan
            yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
            x = xi + (w * xx).sum() / f
            y = yi + (w * yy).sum() / f
        return x, y, peak

    @staticmethod
    def lattice_half_pitch(spot_pars, margin=0.5):
        """
        Largest unambiguous shift per axis, (mx, my): half the lenslet lattice
        pitch measured from the spot fits at one wavelength, minus margin px.
        None if the pitch cannot be measured.
        """
        x, y = spot_pars[..., 1], spot_pars[..., 2]
        with np.errstate(invalid="ignore"):
            px = abs(np.nanmedian(np.diff(x, axis=1)))
            py = abs(np.nanmedian(np.diff(y, axis=0)))
        if not (np.isfinite(px) and np.isfinite(py)) or min(px, py) < 4:
            return None
        return 0.5 * px - margin, 0.5 * py - margin

    @staticmethod
    def xcorr_shift(ref, new, max_shift=(9.0, 9.0), guess=(0.0, 0.0)):
        """
        Shift (dx, dy) of new relative to ref from the peak of their FFT
        cross-correlation, searched only within a box of +-max_shift px
        around guess. max_shift is one number or (max_dx, max_dy).

        The spots at one wavelength sit on the lenslet lattice, so the
        cross-correlation has nearly equal peaks one lattice step apart
        (~18.9 px in x and ~20.4 px in y for LowRes); restricting the search
        to just under half the pitch on each axis keeps the peak on the right
        lenslet. Sub-pixel position from a parabola through the peak along
        each axis.
        """
        mx, my = (max_shift, max_shift) if np.isscalar(max_shift) else max_shift
        ny, nx = ref.shape
        cc = np.fft.irfft2(np.conj(np.fft.rfft2(ref)) * np.fft.rfft2(new),
                           s=ref.shape)
        cc = np.fft.fftshift(cc)
        cy, cx = ny // 2, nx // 2
        gx, gy = int(round(guess[0])), int(round(guess[1]))
        rx, ry = int(np.ceil(mx)), int(np.ceil(my))
        y0, x0 = cy + gy - ry, cx + gx - rx
        win = cc[y0:y0 + 2 * ry + 1, x0:x0 + 2 * rx + 1]
        yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
        win = np.where((abs(xx + gx - guess[0]) <= mx)
                       & (abs(yy + gy - guess[1]) <= my), win, -np.inf)
        iy, ix = np.unravel_index(np.argmax(win), win.shape)
        py, px = y0 + iy, x0 + ix

        def parab(m, c, p):
            den = m - 2 * c + p
            return 0.0 if den == 0 else 0.5 * (m - p) / den

        sx = parab(cc[py, px - 1], cc[py, px], cc[py, px + 1])
        sy = parab(cc[py - 1, px], cc[py, px], cc[py + 1, px])
        dx, dy = px - cx + sx, py - cy + sy
        on_edge = (abs(dx - guess[0]) > mx - 1) or (abs(dy - guess[1]) > my - 1)
        return float(dx), float(dy), on_edge

    def measure_spot_shift(self, ref_im, new_im, spot_pars, half=4,
                           snr_min=10.0, min_spots=20, max_shift=None,
                           guess=(0.0, 0.0), affine=False):
        """
        Measure the shift (new - ref) of the spots in new_im relative to
        ref_im. spot_pars is the (ny, nx, 6) Gaussian fit array of the scan
        at this wavelength (x centre in [..., 1], y centre in [..., 2]).

        Returns dx, dy and a dict of diagnostics. With affine=True (and
        enough spots) info['affine'] also holds an affine fit to the
        per-spot offsets (see fit_affine).
        """
        ref_hp, ref_noise = self.highpass(ref_im)
        new_hp, new_noise = self.highpass(new_im)

        # 1. global cross-correlation, clipped so hot pixels can't dominate
        if max_shift is None:
            max_shift = self.lattice_half_pitch(spot_pars) or (9.0, 9.0)
        if np.isscalar(max_shift):
            max_shift = (float(max_shift), float(max_shift))
        ref_c = np.clip(ref_hp, 0, np.percentile(ref_hp, 99.9))
        new_c = np.clip(new_hp, 0, np.percentile(new_hp, 99.9))
        dx_pc, dy_pc, on_edge = self.xcorr_shift(
            ref_c, new_c, max_shift=max_shift, guess=guess)
        self.logger.info(
            f"Cross-correlation shift: dx = {dx_pc:.3f}, dy = {dy_pc:.3f} px "
            f"(searched +-{max_shift[0]:.1f} x +-{max_shift[1]:.1f} px around "
            f"({guess[0]:.1f}, {guess[1]:.1f}))")
        if on_edge:
            self.logger.warning(
                "Cross-correlation peak is at the edge of the search box "
                f"(+-{max_shift[0]:.1f} x +-{max_shift[1]:.1f} px): the shift "
                "may be one lenslet off. Set rmat_shift_guess_dx/dy and rerun.")

        # 2. spot-by-spot refinement
        xs = spot_pars[..., 1].ravel()
        ys = spot_pars[..., 2].ravel()
        good = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[good], ys[good]

        rows = []
        for x0, y0 in zip(xs, ys):
            xr, yr, pr = self.centroid(ref_hp, x0, y0, half)
            if not np.isfinite(xr) or pr < snr_min * ref_noise:
                continue
            xn, yn, pn = self.centroid(new_hp, xr + dx_pc, yr + dy_pc, half)
            if not np.isfinite(xn) or pn < snr_min * new_noise:
                continue
            rows.append((xr, yr, xn - xr, yn - yr))
        rows = np.array(rows).reshape(-1, 4)
        n_spots = len(rows)

        info = dict(dx_pc=dx_pc, dy_pc=dy_pc, xcorr_on_edge=on_edge,
                    n_spots_model=int(good.sum()), n_spots=n_spots,
                    spots=rows, ref_noise=ref_noise, new_noise=new_noise)

        if n_spots < min_spots:
            self.logger.warning(
                f"Only {n_spots} spots passed the S/N cut (need {min_spots}); "
                "using the cross-correlation shift.")
            info.update(method="cross_correlation", dx_err=np.nan,
                        dy_err=np.nan, dx_std=np.nan, dy_std=np.nan)
            return dx_pc, dy_pc, info

        _, dx, dx_std = sigma_clipped_stats(rows[:, 2], sigma=3.0, maxiters=5)
        _, dy, dy_std = sigma_clipped_stats(rows[:, 3], sigma=3.0, maxiters=5)
        dx_err = 1.2533 * dx_std / np.sqrt(n_spots)
        dy_err = 1.2533 * dy_std / np.sqrt(n_spots)

        # linear variation of the shift across the field (px per 1000 px)
        clip = ((np.abs(rows[:, 2] - dx) < 3 * dx_std + 1e-6) &
                (np.abs(rows[:, 3] - dy) < 3 * dy_std + 1e-6))
        A = np.column_stack([np.ones(clip.sum()),
                             (rows[clip, 0] - 1024) / 1000.,
                             (rows[clip, 1] - 1024) / 1000.])
        gx = np.linalg.lstsq(A, rows[clip, 2], rcond=None)[0]
        gy = np.linalg.lstsq(A, rows[clip, 3], rcond=None)[0]

        if max(dx_std, dy_std) > 0.75:
            self.logger.warning(
                f"Per-spot shift scatter is large ({dx_std:.2f}, {dy_std:.2f} "
                "px; well-matched frames give ~0.2 px): the spots are not "
                "well matched and the shift is unreliable.")
        info.update(method="spot_centroids", dx_err=dx_err, dy_err=dy_err,
                    dx_std=dx_std, dy_std=dy_std,
                    dx_grad=(gx[1], gx[2]), dy_grad=(gy[1], gy[2]))

        if abs(dx - dx_pc) > 0.25 or abs(dy - dy_pc) > 0.25:
            self.logger.warning(
                f"Spot-centroid shift ({dx:.3f}, {dy:.3f}) differs from the "
                f"cross-correlation shift ({dx_pc:.3f}, {dy_pc:.3f}) by more "
                "than 0.25 px; check the diagnostic plot.")

        if affine:
            aff = self.fit_affine(rows)
            info["affine"] = aff
            self.logger.info(
                f"Affine fit ({aff['n_used']} spots): shift at "
                f"({AFFINE_X0:.0f}, {AFFINE_Y0:.0f}) = ({aff['coef'][0, 0]:.3f}, "
                f"{aff['coef'][1, 0]:.3f}) px, x scale {aff['scale_x']:.6f}, "
                f"y scale {aff['scale_y']:.6f}, rotation "
                f"{aff['rotation_deg'] * 3600:.1f} arcsec, shear "
                f"{aff['shear']:.2e}; residual scatter "
                f"({aff['dx_std']:.3f}, {aff['dy_std']:.3f}) px "
                f"(shift only: {dx_std:.3f}, {dy_std:.3f})")
        return float(dx), float(dy), info

    @staticmethod
    def fit_affine(rows, nsig=3.0, maxiters=5):
        """
        Sigma-clipped least-squares affine fit to per-spot offsets.

        rows: (N, 4) array of (x_ref, y_ref, dx, dy). The model is
            dx = a0 + a1 (x - X0) + a2 (y - Y0)
            dy = b0 + b1 (x - X0) + b2 (y - Y0)
        with (X0, Y0) the detector centre, so a0, b0 are the shift there.
        Returns a dict with coef = [[a0, a1, a2], [b0, b1, b2]], the
        residual scatter, and scale / rotation / shear terms for the log.
        """
        x = rows[:, 0] - AFFINE_X0
        y = rows[:, 1] - AFFINE_Y0
        A = np.column_stack([np.ones_like(x), x, y])
        use = np.ones(len(rows), bool)
        for _ in range(maxiters):
            ca = np.linalg.lstsq(A[use], rows[use, 2], rcond=None)[0]
            cb = np.linalg.lstsq(A[use], rows[use, 3], rcond=None)[0]
            rx, ry = rows[:, 2] - A @ ca, rows[:, 3] - A @ cb
            sx = 1.4826 * np.median(np.abs(rx[use] - np.median(rx[use])))
            sy = 1.4826 * np.median(np.abs(ry[use] - np.median(ry[use])))
            new_use = (np.abs(rx) < nsig * sx + 1e-6) & (np.abs(ry) < nsig * sy + 1e-6)
            if np.array_equal(new_use, use):
                break
            use = new_use
        coef = np.array([ca, cb])
        # x' = x + dx: Jacobian [[1 + a1, a2], [b1, 1 + b2]]
        return dict(coef=coef, n_used=int(use.sum()),
                    dx_std=float(np.std(rx[use])), dy_std=float(np.std(ry[use])),
                    scale_x=1 + ca[1], scale_y=1 + cb[2],
                    rotation_deg=float(np.degrees(0.5 * (cb[1] - ca[2]))),
                    shear=float(0.5 * (cb[1] + ca[2])))

    @staticmethod
    def apply_affine(x, y, coef):
        """Offsets (dx, dy) of the affine model coef at positions x, y."""
        xc, yc = x - AFFINE_X0, y - AFFINE_Y0
        return (coef[0, 0] + coef[0, 1] * xc + coef[0, 2] * yc,
                coef[1, 0] + coef[1, 1] * xc + coef[1, 2] * yc)

    # ------------------------------------------------------------------
    # shifted rectmats
    # ------------------------------------------------------------------
    # every rectmat ProcessMonochrom writes, and the shifted file names
    RECTMAT_KINDS = {
        "OPT_intp": "{m}_OPT_intp_rectmat_{t}.npz",   # Gaussian fits, 27 channels
        "C2_intp": "{m}_C2_intp_rectmat_{t}.npz",
        "AnK_intp": "{m}_AnK_intp_rectmat_{t}.npz",   # A&K ePSFs (LowRes)
        "OPT": "{m}_OPT_rectmat_{t}.npz",             # scan images, one channel
        "C2": "{m}_C2_rectmat_{t}.npz",               # per scan wavelength
    }

    def requested_rectmats(self, inst, scmode):
        """Rectmat kinds to write: rmat_shift_rectmats in the config ('all',
        or a list of RECTMAT_KINDS keys). The AnK and scan-image (OPT, C2)
        rectmats exist for LowRes only, as in ProcessMonochrom."""
        req = _cfg(inst, "rmat_shift_rectmats", "all")
        if isinstance(req, str):
            req = (list(self.RECTMAT_KINDS) if req.strip().lower() == "all"
                   else [s.strip() for s in req.strip("[]() ").split(",")])
        bad = [k for k in req if k not in self.RECTMAT_KINDS]
        if bad:
            raise RuntimeError(f"Unknown rmat_shift_rectmats {bad}; use 'all' or "
                               f"some of {list(self.RECTMAT_KINDS)}.")
        if not scmode.startswith("LowRes"):
            req = [k for k in req if k in ("OPT_intp", "C2_intp")]
        return req

    @staticmethod
    def shift_offsets(x, y, shift, coef):
        """(ox, oy) added to positions x, y: the affine model coef if given,
        else the constant shift (dx, dy)."""
        if coef is not None:
            return RmatShift.apply_affine(x, y, coef)
        return np.full_like(x, shift[0], dtype=float), np.full_like(y, shift[1], dtype=float)

    def write_shifted_rectmats(self, scan_dir, scmode, out_dir, tag, shift, coef,
                               kinds, fluxcut, nsub=10):
        """
        Write shifted versions of the scan's rectmats. Spot shapes come from
        the scan; only positions move (by `shift`, or by the affine model
        `coef` evaluated at each spot). Returns {kind: path} of what was
        written; kinds whose scan inputs are missing are skipped with a
        warning.
        """
        path = lambda k: os.path.join(out_dir, self.RECTMAT_KINDS[k].format(m=scmode, t=tag))
        scan = lambda name: os.path.join(scan_dir, name.format(m=scmode))
        written = {}

        def save(kind, rmat, cube_shape):
            scbasic.save_rectmat(path(kind), rmat, cube_shape=tuple(cube_shape))
            self.logger.info(f"wrote {path(kind)}")
            written[kind] = path(kind)

        # Gaussian interpolated rectmats: move the interpolated spot centres
        if {"OPT_intp", "C2_intp"} & set(kinds):
            arr = np.array(pyfits.getdata(scan("{m}_gauss_spotpars_interpd.fits")), dtype=float)
            ox, oy = self.shift_offsets(arr[..., 1], arr[..., 2], shift, coef)
            arr[..., 1] += ox
            arr[..., 2] += oy
            self.logger.info("making shifted interpolated (Gaussian) rectmats")
            opt = self.gen_QL_rectmat_interpd([np.zeros((2048, 2048))], arr,
                                              cut=fluxcut, method='optimal')
            # C2 has the same weights with rows and columns swapped
            # (gen_c2_rectmat_inds_interpd), i.e. the transpose
            if "OPT_intp" in kinds:
                save("OPT_intp", opt, arr.shape[:3])
            if "C2_intp" in kinds:
                save("C2_intp", opt.T.tocsr(), arr.shape[:3])

        # A&K interpolated rectmat: ProcessMonochrom.AnK_rectmat with the spot
        # fits moved. Positions enter AnK_rectmat only through a linear
        # (cubic-in-lambda) fit, so moving the fits is the same as its own
        # `shift` argument for a constant shift, and also works for affine.
        if "AnK_intp" in kinds:
            need = [scan(n) for n in ("{m}_ank_epsf.fits", "{m}_gauss_spotpars.fits",
                                      "{m}_lams.fits", "{m}_intp_lams.fits")]
            missing = [os.path.basename(p) for p in need if not os.path.isfile(p)]
            if missing:
                self.logger.warning(f"Not making the shifted AnK rectmat: {missing} "
                                    "not in the scan directory.")
            else:
                ank, lams_ank = self.read_AnK_lowres_psfs(need[0])
                fit = np.array(pyfits.getdata(need[1]), dtype=float)
                lams_fit = np.asarray(pyfits.getdata(need[2]), dtype=float)
                lams_out = np.asarray(pyfits.getdata(need[3]), dtype=float)
                ox, oy = self.shift_offsets(fit[..., 1], fit[..., 2], shift, coef)
                fit[..., 1] += ox
                fit[..., 2] += oy
                self.logger.info(f"making shifted AnK interpolated rectmat ({nsub} "
                                 "sub-wavelengths per channel)")
                rmat, _ = self.AnK_rectmat(ank, lams_ank, lams_out, fit, lams_fit, nsub=nsub)
                save("AnK_intp", rmat, (len(lams_out),) + fit.shape[1:3])

        # rectmats built from the scan images themselves (one channel per scan
        # wavelength): shift each background-subtracted scan image and cut the
        # spots from it at their moved boxes, as gen_QL_rectmat does
        if {"OPT", "C2"} & set(kinds):
            pos_p, cal_p = scan("{m}_posarr.fits"), scan("{m}_calim_stack.fits")
            missing = [os.path.basename(p) for p in (pos_p, cal_p) if not os.path.isfile(p)]
            if missing:
                self.logger.warning(f"Not making the shifted OPT/C2 (scan image) rectmats: "
                                    f"{missing} not in the scan directory.")
            else:
                posarr = np.array(pyfits.getdata(pos_p), dtype=float)
                self.logger.info("making shifted OPT/C2 rectmats from the scan images")
                opt = self.shifted_posarr_rectmat(cal_p, posarr, shift, coef, fluxcut)
                if "OPT" in kinds:
                    save("OPT", opt, posarr.shape[:3])
                if "C2" in kinds:
                    # gen_C2_rectmat uses the same weights, transposed
                    save("C2", opt.T.tocsr(), posarr.shape[:3])
        return written

    def shifted_posarr_rectmat(self, calim_path, posarr, shift, coef, cut):
        """
        OPT rectmat as ProcessMonochrom.gen_QL_rectmat(ims_cal, posarr),
        with every spot moved by the shift (or affine model).

        Each scan image is resampled (cubic spline) so its content moves by
        the shift; each spot's box from posarr moves by the shift rounded to
        whole pixels (clipped to the detector) and its weights are cut from
        the resampled image exactly as crop_sparse_vals does: pixels below
        cut * max set to 0, normalised to unit sum, only positive values
        kept. A spot whose box contains a NaN pixel is dropped, as before.
        """
        from scipy.ndimage import shift as nd_shift, map_coordinates
        nl, ny, nx = posarr.shape[:3]
        rows, cols, vals = [], [], []
        if coef is not None:
            yy, xx = np.mgrid[0:2048, 0:2048].astype(float)
            fox, foy = self.apply_affine(xx, yy, coef)
            src = [yy - foy, xx - fox]          # output pixel <- source position
            del yy, xx
        with pyfits.open(calim_path, memmap=True) as hdul:
            for ll in range(nl):
                img = np.array(hdul[0].section[ll], dtype=float)
                bad = ~np.isfinite(img)
                img[bad] = 0.0
                if coef is None:
                    w = nd_shift(img, (shift[1], shift[0]), order=3, mode='constant', cval=0.0)
                    wbad = nd_shift(bad.astype(float), (shift[1], shift[0]), order=1,
                                    mode='constant', cval=0.0) > 1e-3
                else:
                    w = map_coordinates(img, src, order=3, mode='constant', cval=0.0)
                    wbad = map_coordinates(bad.astype(float), src, order=1,
                                           mode='constant', cval=0.0) > 1e-3
                w[wbad] = np.nan

                p = posarr[ll].reshape(ny * nx, -1)
                lens = np.nonzero(np.isfinite(p[:, 0]))[0]
                if not len(lens):
                    continue
                xc, xs, xe, yc, ys, ye = (p[lens, i] for i in range(6))
                ox, oy = self.shift_offsets(xc, yc, shift, coef)
                ix, iy = np.round(ox).astype(int), np.round(oy).astype(int)
                xs = np.clip(xs.astype(int) + ix, 0, 2048)
                xe = np.clip(xe.astype(int) + ix, 0, 2048)
                ys = np.clip(ys.astype(int) + iy, 0, 2048)
                ye = np.clip(ye.astype(int) + iy, 0, 2048)
                hw = np.stack([ye - ys, xe - xs], 1)
                for h, wd in {tuple(s) for s in hw if s[0] > 0 and s[1] > 0}:
                    sel = (hw[:, 0] == h) & (hw[:, 1] == wd)
                    jy, jx = np.mgrid[0:h, 0:wd]
                    Y = ys[sel][:, None, None] + jy
                    X = xs[sel][:, None, None] + jx
                    st = w[Y, X]
                    mx = st.max(axis=(1, 2), keepdims=True)
                    st = np.where(st < cut * mx, 0.0, st)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        st = st / st.sum(axis=(1, 2), keepdims=True)
                    keep = st > 0
                    n = keep.sum(axis=(1, 2))
                    rows.append(np.repeat(lens[sel] + ll * ny * nx, n))
                    cols.append((Y * 2048 + X)[keep])
                    vals.append(st[keep])
        rows = np.concatenate(rows) if rows else np.zeros(0, int)
        cols = np.concatenate(cols) if cols else np.zeros(0, int)
        vals = np.concatenate(vals) if vals else np.zeros(0)
        from scipy import sparse as _sp
        return _sp.csr_matrix((vals, (rows, cols)), shape=(nl * ny * nx, 2048 * 2048))

    def plot_shift(self, info, dx, dy, title, path):
        rows = info["spots"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        if len(rows):
            for ax, col, val, lab in ((axes[0], 2, dx, "dx"), (axes[1], 3, dy, "dy")):
                d = rows[:, col]
                lo, hi = np.percentile(d, [0.5, 99.5])
                ax.hist(d, bins=60, range=(lo - 0.1, hi + 0.1), color="0.5")
                ax.axvline(val, color="C0", label=f"spots: {val:.3f}")
                ax.axvline(info["dx_pc" if col == 2 else "dy_pc"], color="C1",
                           ls="--", label="cross-corr.")
                ax.set_xlabel(f"{lab} (px, new - scan)")
                ax.legend()
            # binned map of the offsets left after the applied model
            if "affine" in info:
                mx, my = self.apply_affine(rows[:, 0], rows[:, 1],
                                           info["affine"]["coef"])
                label = "residual after affine fit (binned)"
            else:
                mx, my = dx, dy
                label = "residual after x/y shift (binned)"
            resx, resy = rows[:, 2] - mx, rows[:, 3] - my
            nb = 8
            edges = np.linspace(0, 2048, nb + 1)
            ix = np.clip(np.digitize(rows[:, 0], edges) - 1, 0, nb - 1)
            iy = np.clip(np.digitize(rows[:, 1], edges) - 1, 0, nb - 1)
            cen = 0.5 * (edges[1:] + edges[:-1])
            X, Y, U, V = [], [], [], []
            for i in range(nb):
                for j in range(nb):
                    m = (ix == i) & (iy == j)
                    if m.sum() >= 5:
                        X.append(cen[i]); Y.append(cen[j])
                        U.append(np.median(resx[m]))
                        V.append(np.median(resy[m]))
            q = axes[2].quiver(X, Y, U, V, angles="xy", scale_units="xy",
                               scale=0.1 / 100)
            axes[2].quiverkey(q, 0.8, 1.04, 0.1, "0.1 px", labelpos="E")
            axes[2].set_xlim(0, 2048); axes[2].set_ylim(0, 2048)
            axes[2].set_aspect("equal")
            axes[2].set_title(label, pad=20)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------
    def _perform(self):
        args = self.action.args
        inst = self.config.instrument
        output_dir = inst.output_directory
        plot_dir = inst.plot_directory
        out_dir = _cfg(inst, "rmat_out_directory", ".")
        scan_dir = _cfg(inst, "rmat_scan_dir", None)
        tol_nm = float(_cfg(inst, "wave_match_tol_nm", 0.5))
        snr_min = float(_cfg(inst, "rmat_shift_snr_min", 10.0))
        min_spots = int(_cfg(inst, "rmat_shift_min_spots", 20))
        measure_only = bool(_cfg(inst, "rmat_measure_only", False))
        fluxcut = float(_cfg(inst, "rectmat_fluxcut", 0.0))

        bkg_frames = _cfg(inst, "rmat_background_frames", None) or []
        if isinstance(bkg_frames, str):
            bkg_frames = [bkg_frames]
        bkg_names = {os.path.basename(b) for b in bkg_frames}
        if os.path.basename(args.name) in bkg_names:
            self.logger.info(f"{args.name} is a background frame; L1 only.")
            return self.action.args

        if scan_dir is None:
            raise RuntimeError("No monochromator scan directory given (-r).")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        # ---- the new frame's L1
        l1_path = scbasic.get_l1_path_from_raw(args.name, output_dir)
        if not os.path.isfile(l1_path):
            raise RuntimeError(f"L1 file {l1_path} was not created.")
        with pyfits.open(l1_path, memmap=False) as hdul:
            new_im = np.asarray(hdul[0].data, dtype=float)
            hdr = hdul[0].header.copy()
        stem = os.path.basename(l1_path).replace("_L1.fits", "")

        scmode = ifsmode_of(hdr, self.logger, stem)
        if scmode is None:
            raise RuntimeError(
                f"Could not determine the IFS mode of {stem} "
                f"(IFSMODE={hdr.get('IFSMODE')}, MODSLNAM={hdr.get('MODSLNAM')}, "
                f"DSPRSNAM={hdr.get('DSPRSNAM')}).")
        monowave = float(hdr["MONOWAVE"])
        self.scmode = scmode
        self.logger.info(
            f"{stem}: {scmode}, MONOWAVE = {monowave} nm, L1 = {l1_path}")

        # ---- matching scan products
        ref_name, ref_wave, ref_row, df_mode = self.find_matching_mcalunit(
            scan_dir, scmode, monowave, tol_nm)
        self.logger.info(
            f"Matched scan mcalunit {ref_name} ({ref_wave:.3f} nm)")
        if str(ref_row.get("MCLOCK")).strip() != str(hdr.get("MCLOCK")).strip():
            self.logger.warning(
                f"Detector clock differs: frame {hdr.get('MCLOCK')}, "
                f"scan {ref_row.get('MCLOCK')}.")
        lams = np.asarray(pyfits.getdata(
            os.path.join(scan_dir, f"{scmode}_lams.fits")), dtype=float)
        ll = int(np.argmin(np.abs(lams * 1000.0 - ref_wave)))
        if abs(lams[ll] * 1000.0 - ref_wave) > 0.01:
            self.logger.warning(
                f"{scmode}_lams.fits has no entry for {ref_wave:.3f} nm; "
                f"using the spot fits at {lams[ll] * 1000:.3f} nm.")
        spot_pars = pyfits.getdata(
            os.path.join(scan_dir, f"{scmode}_gauss_spotpars.fits"))[ll]

        # ---- background-subtracted spot image. The raw monochromator L1s are
        # dominated by the continuum traces and the spots are only faint
        # bumps on them. Differencing with a background frame taken at the
        # same time removes the traces exactly, whatever the frame's shift.
        # Template-free filters (running medians) and the scan's mean image
        # both failed on the 20261118 scans, so a background frame is required.
        if not bkg_frames:
            raise RuntimeError(
                "No background frame given. Pass one with -b (a frame taken "
                "with the frame, e.g. the monochromator at a wavelength "
                ">~200 nm away), or set rmat_background_frames in the config.")
        destripe = bool(_cfg(inst, "apply_destripe", True))
        bkg, bkg_waves = self.load_background(bkg_frames, output_dir, hdr)
        new_cal = np.nan_to_num(new_im - bkg, nan=0.0)
        if destripe:
            new_cal = self.masked_row_destripe(new_cal)[0]
        bkg_method = "background frame: " + ", ".join(
            os.path.basename(b) for b in bkg_frames)
        self.logger.info(f"Background removal: {bkg_method}")

        # a monochromator background leaves its own (negative) spots in the
        # difference; they must sit well away from the frame's spots
        min_sep = float(_cfg(inst, "rmat_bkg_min_sep_px", 7.0))
        bkg_sep = self.background_spot_separation(
            scan_dir, scmode, lams, monowave, bkg_waves)
        bkg_too_close = bkg_sep is not None and bkg_sep < min_sep
        if bkg_sep is not None:
            self.logger.info(
                f"Background spots are {bkg_sep:.1f} px from the frame's spots")
        if bkg_too_close:
            self.logger.warning(
                f"Background frame spots are only {bkg_sep:.1f} px from the "
                f"frame's spots (need >= {min_sep:.0f} px): its negative spots "
                "can bias the centroids. Use a background frame at a wavelength "
                "further away.")
        meanim = None

        # reference: the scan's own background-subtracted image at this
        # wavelength (what the spot fits were made from)
        ref_cal = self.load_scan_calim(scan_dir, scmode, ll)
        if ref_cal is None:
            self.logger.info(f"No {scmode}_calim_stack.fits in the scan; "
                             f"background-subtracting {ref_name} instead")
            if meanim is None:
                meanim = self.scan_mean_image(scan_dir, df_mode, scmode, out_dir)
            ref_im = pyfits.getdata(os.path.join(scan_dir, ref_name)).astype(float)
            ref_cal = self.calibrate_like_scan(ref_im, meanim, destripe)

        # ---- measure
        lowres = scmode.startswith("LowRes")
        half = 4 if lowres else 8
        # None -> just under half the lenslet pitch on each axis, measured from
        # the spot fits (LowRes ~18.9 x 20.4 px); else one number or [dx, dy]
        max_shift = _cfg(inst, "rmat_shift_max", None)
        if isinstance(max_shift, str):
            max_shift = (None if max_shift.strip() in ("", "None")
                         else [float(v) for v in
                               max_shift.strip("[]() ").split(",")])
        if max_shift is not None and not np.isscalar(max_shift):
            max_shift = tuple(float(v) for v in max_shift)
            if len(max_shift) == 1:
                max_shift = max_shift[0]
        guess = (float(_cfg(inst, "rmat_shift_guess_dx", 0.0)),
                 float(_cfg(inst, "rmat_shift_guess_dy", 0.0)))
        use_affine = _cfg(inst, "rmat_shift_affine", False)
        if isinstance(use_affine, str):
            use_affine = use_affine.strip().lower() in ("true", "1", "yes")
        dx, dy, info = self.measure_spot_shift(
            ref_cal, new_cal, spot_pars, half=half,
            snr_min=snr_min, min_spots=min_spots,
            max_shift=max_shift, guess=guess, affine=bool(use_affine))
        self.logger.info(
            f"Measured shift ({info['method']}, {info['n_spots']} spots): "
            f"dx = {dx:.3f} +/- {info['dx_err']:.3f}, "
            f"dy = {dy:.3f} +/- {info['dy_err']:.3f} px")
        aff = info.get("affine")
        if use_affine and aff is None:
            self.logger.warning(
                "rmat_shift_affine is set but there are too few spots for a "
                "fit; applying the x/y shift only.")

        plot_path = os.path.join(plot_dir, f"{stem}_rmat_shift.png")
        title = f"{stem} vs {ref_name}: dx = {dx:.3f}, dy = {dy:.3f} px"
        if aff is not None:
            title += (f"; affine: scale ({aff['scale_x']:.5f}, "
                      f"{aff['scale_y']:.5f}), rot "
                      f"{aff['rotation_deg'] * 3600:.0f}\"")
        self.plot_shift(info, dx, dy, title, plot_path)

        # ---- shifted rectmats
        if aff is not None:
            # name by the shift at the detector centre
            tag = (f"affine_dx{aff['coef'][0, 0]:.2f}"
                   f"_dy{aff['coef'][1, 0]:.2f}")
        else:
            tag = f"dx{dx:.2f}_dy{dy:.2f}"
        if measure_only:
            self.logger.info("rmat_measure_only is set; not writing rectmats.")
            written = {}
        else:
            kinds = self.requested_rectmats(inst, scmode)
            written = self.write_shifted_rectmats(
                scan_dir, scmode, out_dir, tag, (dx, dy),
                None if aff is None else aff["coef"], kinds, fluxcut,
                nsub=int(_cfg(inst, "ank_rectmat_nsub", 10)))
        opt_path = written.get("OPT_intp", "")
        c2_path = written.get("C2_intp", "")

        # ---- record
        scatter = (max(aff["dx_std"], aff["dy_std"]) if aff is not None
                   else max(info["dx_std"], info["dy_std"]))
        summary = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "frame": stem,
            "l1": l1_path,
            "scmode": scmode,
            "monowave_nm": monowave,
            "scan_dir": os.path.abspath(scan_dir),
            "scan_mcalunit": ref_name,
            "scan_monowave_nm": ref_wave,
            "quality": ("OK" if info["method"] == "spot_centroids"
                        and scatter <= 0.75
                        and not info["xcorr_on_edge"]
                        and not bkg_too_close else "CHECK"),
            "background": bkg_method,
            "background_spot_sep_px": (None if bkg_sep is None
                                       else round(bkg_sep, 2)),
            "method": info["method"],
            "model": "affine" if aff is not None else "shift",
            "dx": round(dx, 4), "dy": round(dy, 4),
            "dx_err": round(float(info["dx_err"]), 4),
            "dy_err": round(float(info["dy_err"]), 4),
            "dx_std": round(float(info["dx_std"]), 4),
            "dy_std": round(float(info["dy_std"]), 4),
            "dx_xcorr": round(info["dx_pc"], 4),
            "dy_xcorr": round(info["dy_pc"], 4),
            "n_spots": info["n_spots"],
            "opt_rectmat": opt_path,
            "c2_rectmat": c2_path,
            "plot": plot_path,
        }
        for kind, path in written.items():
            summary[f"rectmat_{kind}"] = path
        if "dx_grad" in info:
            summary["dx_grad_per1000px_x_y"] = tuple(round(float(g), 4) for g in info["dx_grad"])
            summary["dy_grad_per1000px_x_y"] = tuple(round(float(g), 4) for g in info["dy_grad"])
        if aff is not None:
            (a0, a1, a2), (b0, b1, b2) = aff["coef"]
            summary.update({
                "affine_origin_px": (AFFINE_X0, AFFINE_Y0),
                # dx = a0 + a1 (x - x0) + a2 (y - y0), same for dy with b
                "affine_a0": round(float(a0), 4), "affine_a1": float(a1),
                "affine_a2": float(a2),
                "affine_b0": round(float(b0), 4), "affine_b1": float(b1),
                "affine_b2": float(b2),
                "affine_scale_x": round(float(aff["scale_x"]), 7),
                "affine_scale_y": round(float(aff["scale_y"]), 7),
                "affine_rotation_arcsec": round(aff["rotation_deg"] * 3600, 2),
                "affine_shear": float(aff["shear"]),
                "affine_dx_std": round(aff["dx_std"], 4),
                "affine_dy_std": round(aff["dy_std"], 4),
                "affine_n_used": aff["n_used"],
            })

        with open(os.path.join(out_dir, f"{stem}_rmat_shift.txt"), "w") as f:
            for k, v in summary.items():
                f.write(f"{k} = {v}\n")

        log_path = os.path.join(out_dir, "rmat_shift_log.csv")
        cols = ["time", "frame", "scmode", "monowave_nm", "scan_mcalunit",
                "scan_monowave_nm", "background", "method", "model", "quality",
                "dx", "dy", "dx_err", "dy_err", "dx_std", "dy_std",
                "affine_a0", "affine_a1", "affine_a2",
                "affine_b0", "affine_b1", "affine_b2",
                "affine_dx_std", "affine_dy_std",
                "n_spots", "opt_rectmat", "c2_rectmat"]
        if os.path.isfile(log_path):
            # keep the columns of an existing log (older ones lack the
            # affine columns; the .txt summary always has everything)
            with open(log_path, newline="") as f:
                cols = next(csv.reader(f), cols)
            new_file = False
        else:
            new_file = True
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore",
                               restval="")
            if new_file:
                w.writeheader()
            w.writerow(summary)

        self.logger.info("+++++++++++ rectmat shift done +++++++++++")
        return self.action.args
