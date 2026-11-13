"""
Measure the spot shift between a single monochromator frame and an existing
monochromator scan, and write shifted interpolated rectmats.

The single frame's L1 (made by StartCalibSA) is compared with the master
calunit image at the same wavelength in the scan directory. The shift is
measured in two steps:

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
        df["IFSMODE"] = df.apply(
            lambda row: scbasic.select_ifsmode(
                row.get("MODSLNAM"), row.get("DSPRSNAM"), row.get("FILTER")),
            axis=1)
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

    def load_background(self, bkg_frames, output_dir, hdr):
        """Mean of the background frames' L1 images."""
        ims = []
        for b in bkg_frames:
            path = scbasic.get_l1_path_from_raw(b, output_dir)
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"Background L1 {path} not found; the background frame "
                    "must be reduced before the monochromator frame.")
            with pyfits.open(path, memmap=False) as hdul:
                ims.append(np.asarray(hdul[0].data, dtype=float))
                bh = hdul[0].header
            for key in ("EXPTIME", "MCLOCK", "MODSLNAM", "DSPRSNAM"):
                if str(bh.get(key)).strip() != str(hdr.get(key)).strip():
                    self.logger.warning(
                        f"Background {os.path.basename(path)} has {key}="
                        f"{bh.get(key)}, frame has {hdr.get(key)}.")
        return np.nanmean(ims, axis=0) if len(ims) > 1 else ims[0]

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

        scmode = scbasic.select_ifsmode(
            hdr.get("MODSLNAM"), hdr.get("DSPRSNAM"), hdr.get("FILTER"))
        if scmode is None:
            raise RuntimeError(
                f"Could not determine the IFS mode of {stem} "
                f"(MODSLNAM={hdr.get('MODSLNAM')}, DSPRSNAM={hdr.get('DSPRSNAM')}).")
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

        # ---- background-subtracted spot images. The raw monochromator L1s
        # are dominated by continuum; the spots only stand out after it is
        # removed.
        destripe = bool(_cfg(inst, "apply_destripe", True))
        if bkg_frames:
            # difference with the separately measured background frame(s)
            bkg = self.load_background(bkg_frames, output_dir, hdr)
            new_cal = np.nan_to_num(new_im - bkg, nan=0.0)
            if destripe:
                new_cal = self.masked_row_destripe(new_cal)[0]
            bkg_method = "background frame: " + ", ".join(
                os.path.basename(b) for b in bkg_frames)
            meanim = None
        else:
            # as ProcessMonochrom.monochrom_bksub: scaled scan-mean image
            meanim = self.scan_mean_image(scan_dir, df_mode, scmode, out_dir)
            new_cal = self.calibrate_like_scan(new_im, meanim, destripe)
            bkg_method = "scaled scan-mean image"
        self.logger.info(f"Background removal: {bkg_method}")

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
        opt_path = os.path.join(out_dir, f"{scmode}_OPT_intp_rectmat_{tag}.npz")
        c2_path = os.path.join(out_dir, f"{scmode}_C2_intp_rectmat_{tag}.npz")
        if measure_only:
            self.logger.info("rmat_measure_only is set; not writing rectmats.")
            opt_path = c2_path = ""
        else:
            interp_arr = np.array(pyfits.getdata(os.path.join(
                scan_dir, f"{scmode}_gauss_spotpars_interpd.fits")), dtype=float)
            if aff is not None:
                # offsets evaluated at each spot's own (scan) position
                ox, oy = self.apply_affine(interp_arr[..., 1],
                                           interp_arr[..., 2], aff["coef"])
                interp_arr[..., 1] += ox
                interp_arr[..., 2] += oy
            else:
                interp_arr[..., 1] += dx
                interp_arr[..., 2] += dy
            self.logger.info("making shifted interpolated optimal rectmat")
            opt_rmat = self.gen_QL_rectmat_interpd(
                [new_im], interp_arr, cut=fluxcut, method='optimal')
            # the C2 rectmat is built from the same weights with rows and
            # columns swapped (gen_c2_rectmat_inds_interpd), i.e. the transpose
            c2_rmat = opt_rmat.T.tocsr()
            cube_shape = interp_arr.shape[:3]
            scbasic.save_rectmat(opt_path, opt_rmat, cube_shape=cube_shape)
            scbasic.save_rectmat(c2_path, c2_rmat, cube_shape=cube_shape)
            self.logger.info(f"wrote {opt_path} and {c2_path}")

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
                        and not info["xcorr_on_edge"] else "CHECK"),
            "background": bkg_method,
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
