"""Four-panel wavelength/centroid diagnostics for SCALES rectmat calibration."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection


def plot_rectmat_centroid_diagnostics(
    ims_cal, posarr, centroids, lams, output_dir, filename,
    max_traces=100, image_index=None, dpi=180, show=False,
):
    """Plot detector spots, traced centroids, centroid counts and wavelength drift.

    Parameters
    ----------
    ims_cal : ndarray, shape (detector_y, detector_x) or (nwave, detector_y, detector_x)
        Only one frame is shown (panel 1), so this can be just that frame: the
        background-subtracted master at wavelength index image_index (the middle
        wavelength by default). A full stack also works; the frame is taken from it.
    posarr : ndarray, shape (nwave, lens_y, lens_x, >=6)
        Columns 0, 3 are predicted detector x/y; columns 1 and 4
        are x/y cutout start positions.
    centroids : ndarray, shape (nwave, lens_y, lens_x, 2)
        Local centroid coordinates within the corresponding cutouts.
    lams : array-like, shape (nwave,)
        Monochromator wavelengths (microns).
    output_dir : str or Path
        Directory for the PNG output.
    filename : str
        Output image name (e.g. 'MedRes-K_centroid_diagnostics.png').
    max_traces : int
        Maximum number of representative traces drawn in panel 2.
    image_index : int or None
        Wavelength index shown in panel 1; None selects the middle frame.
    dpi : int
        Output resolution.
    show : bool
        Display figure interactively if True; otherwise close it.

    Returns
    -------
    pathlib.Path
        Saved diagnostic figure path.
    """
    ims_cal = np.asarray(ims_cal)
    posarr = np.asarray(posarr)
    centroids = np.asarray(centroids)
    lams = np.asarray(lams, dtype=float).ravel()
    if ims_cal.ndim not in (2, 3) or centroids.ndim != 4 or posarr.ndim != 4:
        raise ValueError('Expected ims_cal (2D frame or 3D stack), posarr (4D), centroids (4D).')
    if centroids.shape[-1] != 2 or posarr.shape[-1] < 6:
        raise ValueError('Centroids require x/y; posarr requires cutout offsets.')
    if not (posarr.shape[0] == centroids.shape[0] == len(lams)):
        raise ValueError('Wavelength axis lengths must agree.')
    if ims_cal.ndim == 3 and ims_cal.shape[0] != len(lams):
        raise ValueError('ims_cal stack must have one frame per wavelength.')
    det_ny, det_nx = ims_cal.shape[-2:]
    if posarr.shape[1:3] != centroids.shape[1:3]:
        raise ValueError('Lenslet grid shapes must agree.')
    if len(lams) == 0:
        raise ValueError('No wavelengths provided.')

    # Convert local centroids to full-detector coordinates.
    x = posarr[..., 1] + centroids[..., 0]
    y = posarr[..., 4] + centroids[..., 1]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(lams[:, None, None])
    valid &= (x >= 0) & (x < det_nx) & (y >= 0) & (y < det_ny)
    x = np.where(valid, x, np.nan)
    y = np.where(valid, y, np.nan)
    order = np.argsort(lams)
    ls, xs, ys, vs = lams[order], x[order], y[order], valid[order]
    nw, ny, nx = vs.shape
    nvalid = vs.sum(axis=(1, 2))
    eligible = np.flatnonzero(vs.sum(axis=0).ravel() >= 2)
    if len(eligible) > max_traces:
        eligible = eligible[np.linspace(0, len(eligible)-1, max_traces, dtype=int)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel()
    idx = nw // 2 if image_index is None else int(image_index)
    if not 0 <= idx < nw:
        raise IndexError('image_index out of range.')
    image = ims_cal[idx] if ims_cal.ndim == 3 else ims_cal
    finite = image[np.isfinite(image)]
    if finite.size:
        lo, hi = np.percentile(finite, [5, 99.5])
        if hi <= lo:
            hi = lo + 1
        ax0.imshow(image, origin='lower', cmap='gray', vmin=lo, vmax=hi)
    else:
        ax0.imshow(np.zeros_like(image), origin='lower', cmap='gray')
    ax0.scatter(x[idx][valid[idx]], y[idx][valid[idx]], s=9,
                facecolors='none', edgecolors='cyan', linewidths=0.65)
    ax0.set(title=f'Detected centroids at {lams[idx]:.4f} µm (N={valid[idx].sum()})',
            xlabel='Detector x (pixel)', ylabel='Detector y (pixel)')

    norm = Normalize(vmin=float(np.nanmin(ls)), vmax=float(np.nanmax(ls)) + 1e-12)
    cmap = plt.get_cmap('viridis')
    for flat_id in eligible:
        iy, ix = np.unravel_index(flat_id, (ny, nx))
        good = vs[:, iy, ix]
        if good.sum() < 2:
            continue
        # Split at missing wavelengths rather than joining across gaps.
        xx, yy = xs[:, iy, ix], ys[:, iy, ix]
        segments, colors = [], []
        for j in range(nw - 1):
            if good[j] and good[j+1]:
                segments.append([[xx[j], yy[j]], [xx[j+1], yy[j+1]]])
                colors.append((ls[j] + ls[j+1]) / 2)
        if segments:
            ax1.add_collection(LineCollection(segments, cmap=cmap, norm=norm,
                                              array=np.asarray(colors), linewidths=0.9, alpha=0.75))
        ax1.scatter(xx[good], yy[good], c=ls[good], cmap=cmap, norm=norm,
                    s=4, linewidths=0)
    ax1.autoscale()
    ax1.set_aspect('equal', adjustable='box')
    ax1.set(title=f'Centroid traces ({len(eligible)} representative lenslets)',
            xlabel='Detector x (pixel)', ylabel='Detector y (pixel)')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, label='Wavelength (µm)')

    ax2.plot(ls, nvalid, 'o-', ms=4, lw=1.3)
    ax2.set(title='Valid centroids versus wavelength', xlabel='Wavelength (µm)',
            ylabel='Number of valid lenslets')
    ax2.grid(alpha=0.25)
    ax2.set_ylim(bottom=0)

    # Reference for each lenslet is its first valid wavelength.
    any_valid = vs.any(axis=0)
    first_idx = np.argmax(vs, axis=0)
    refx = np.take_along_axis(xs, first_idx[None, ...], axis=0)[0]
    refy = np.take_along_axis(ys, first_idx[None, ...], axis=0)[0]
    drift = np.hypot(xs - refx[None, ...], ys - refy[None, ...])
    drift = np.where(vs & any_valid[None, ...], drift, np.nan)
    median, p16, p84 = [], [], []
    for k in range(nw):
        vals = drift[k][np.isfinite(drift[k])]
        median.append(np.median(vals) if vals.size else np.nan)
        p16.append(np.percentile(vals, 16) if vals.size else np.nan)
        p84.append(np.percentile(vals, 84) if vals.size else np.nan)
    ax3.plot(ls, median, 'o-', ms=4, lw=1.3, label='Median displacement')
    ax3.fill_between(ls, p16, p84, alpha=0.25, label='16–84 percentile')
    ax3.set(title='Centroid displacement from first valid wavelength',
            xlabel='Wavelength (µm)', ylabel='Displacement (detector pixels)')
    ax3.grid(alpha=0.25)
    ax3.legend(loc='best', fontsize=9)

    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return path


def rectmat_trace_positions(rmat, cube_shape, npix=(2048, 2048)):
    """Detector position of every rectmat row: the centroid of its template.

    Rows are lensx + lensy*nx + l*nx*ny (gen_QL_rectmat[_interpd], AnK_rectmat),
    columns flattened detector pixels (y*npix_x + x). Returns x, y, each
    (nlam, ny, nx), NaN for rows without a template."""
    nlam, ny, nx = cube_shape
    rmat = rmat.tocsr()
    cols = np.arange(npix[0] * npix[1])
    tot = np.asarray(rmat.sum(axis=1)).ravel()
    with np.errstate(invalid='ignore', divide='ignore'):
        x = np.asarray(rmat @ (cols % npix[1]).astype(float)).ravel() / tot
        y = np.asarray(rmat @ (cols // npix[1]).astype(float)).ravel() / tot
    bad = ~(tot > 0)
    x[bad] = np.nan; y[bad] = np.nan
    return x.reshape(nlam, ny, nx), y.reshape(nlam, ny, nx)


def plot_rectmat_lenslet_map(rmat, cube_shape, lams, image, output_dir, filename,
                             zoom=None, title='', dpi=110, show=False):
    """Check how a rectmat is organised: which detector trace each lenslet (x, y) is.

    Positions come from the rectmat itself (template centroid of each row,
    rectmat_trace_positions), so this shows what an extraction with it does.
    Top row: every lenslet's trace on the detector (over `image`, e.g. one
    background-subtracted master), coloured by lenslet x index and by lenslet y
    index (smooth colour gradients across the field mean neighbouring traces are
    neighbouring lenslets), and a zoomed window with each trace labelled x,y.
    Bottom row: the lenslet grid coloured by the detector x and y of each
    lenslet's trace at the middle wavelength, and the number of wavelength
    channels with a template per lenslet.

    zoom: (x0, x1, y0, y1) detector window for the labelled panel; default a
        window around the median trace position holding a few dozen traces.
    """
    nlam, ny, nx = cube_shape
    lams = np.asarray(lams, float)
    X, Y = rectmat_trace_positions(rmat, cube_shape)
    lm = nlam // 2
    with np.errstate(invalid='ignore'):
        xm = np.nanmedian(X, axis=0); ym = np.nanmedian(Y, axis=0)
    nchan = np.isfinite(X).sum(axis=0)
    J, I = np.indices((ny, nx))
    have = nchan > 0
    image = np.asarray(image, float)

    if zoom is None:
        cx, cy = np.nanmedian(xm[have]), np.nanmedian(ym[have])
        #trace length and lenslet pitch set the window: about 6 lenslets across,
        #and the whole trace or 300 px along it, whichever is shorter
        span = np.nanmedian(np.hypot(np.nanmax(X, 0) - np.nanmin(X, 0),
                                     np.nanmax(Y, 0) - np.nanmin(Y, 0))[have])
        from scipy.spatial import cKDTree
        pts = np.column_stack([xm[have], ym[have]])
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        pitch = np.median(cKDTree(pts).query(pts, k=2)[0][:, 1])
        hx = max(5 * pitch, 60)
        hy = max(min(span / 2 + pitch, 150), 3 * pitch)
        zoom = (cx - hx, cx + hx, cy - hy, cy + hy)
    x0, x1, y0, y1 = zoom

    fig, ax = plt.subplots(2, 3, figsize=(22, 14))
    fin = image[np.isfinite(image)]
    lo, hi = (np.percentile(fin, [5, 99.5]) if fin.size else (0, 1))
    segs = []
    for j, i in zip(J[have], I[have]):
        ok = np.isfinite(X[:, j, i])
        if ok.sum() >= 2:
            segs.append((j, i, np.column_stack([X[ok, j, i], Y[ok, j, i]])))
    for a, idx, lab, cmap in [(ax[0, 0], 1, 'lenslet x', 'turbo'), (ax[0, 1], 0, 'lenslet y', 'turbo')]:
        a.imshow(image, origin='lower', cmap='gray', vmin=lo, vmax=hi)
        n = nx if idx == 1 else ny
        norm = Normalize(0, n - 1)
        lc = LineCollection([s for _, _, s in segs], cmap=cmap, norm=norm, linewidths=1.0)
        lc.set_array(np.array([(i if idx == 1 else j) for j, i, _ in segs], float))
        a.add_collection(lc)
        a.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='w', lw=1, ls='--'))
        a.set_xlim(0, image.shape[1]); a.set_ylim(0, image.shape[0])
        fig.colorbar(lc, ax=a, label=lab + ' index', fraction=0.046)
        a.set_title(f'{title} rectmat traces coloured by {lab} index ({len(segs)} lenslets)')
        a.set_xlabel('detector x (px)'); a.set_ylabel('detector y (px)')

    a = ax[0, 2]
    sub = image[max(int(y0), 0):max(int(y1), 0), max(int(x0), 0):max(int(x1), 0)]
    sfin = sub[np.isfinite(sub)]
    slo, shi = (np.percentile(sfin, [5, 99.8]) if sfin.size else (lo, hi))
    a.imshow(image, origin='lower', cmap='gray', vmin=slo, vmax=shi)
    for j, i, s in segs:
        inw = (s[:, 0] > x0) & (s[:, 0] < x1) & (s[:, 1] > y0) & (s[:, 1] < y1)
        if not inw.any():
            continue
        a.plot(s[:, 0], s[:, 1], '-', lw=0.8, color=plt.cm.turbo((i + 0.5) / nx))
        #label along the trace, staggered so neighbouring traces' labels don't overlap
        kk = np.nonzero(inw)[0]
        k = kk[int(len(kk) * (0.2 + 0.3 * ((i + j) % 3)))]
        k2 = kk[min(np.searchsorted(kk, k) + 1, len(kk) - 1)]
        ang = np.degrees(np.arctan2(s[k2, 1] - s[k, 1], s[k2, 0] - s[k, 0])) if k2 != k else 0.0
        ang = (ang + 90) % 180 - 90
        a.text(s[k, 0], s[k, 1], f'{i},{j}', fontsize=7, color='yellow', ha='center', va='center',
               rotation=ang, rotation_mode='anchor',
               bbox=dict(boxstyle='round,pad=0.1', fc='k', alpha=0.5, lw=0))
    a.set_xlim(x0, x1); a.set_ylim(y0, y1)
    a.set_title('zoom: traces labelled lenslet x,y (colour = lenslet x)')
    a.set_xlabel('detector x (px)'); a.set_ylabel('detector y (px)')

    xl = X[lm]; yl = Y[lm]
    for a, m, t in [(ax[1, 0], xl, f'detector x of each lenslet at {lams[lm]:.4f} um'),
                    (ax[1, 1], yl, f'detector y of each lenslet at {lams[lm]:.4f} um')]:
        im = a.imshow(m, origin='lower', cmap='viridis', interpolation='nearest')
        fig.colorbar(im, ax=a, label='px', fraction=0.046)
        a.set_title(t); a.set_xlabel('lenslet x'); a.set_ylabel('lenslet y')
    im = ax[1, 2].imshow(np.where(have, nchan, np.nan), origin='lower', cmap='magma',
                         interpolation='nearest', vmin=0, vmax=nlam)
    fig.colorbar(im, ax=ax[1, 2], label=f'channels with a template (of {nlam})', fraction=0.046)
    ax[1, 2].set_title('wavelength coverage per lenslet')
    ax[1, 2].set_xlabel('lenslet x'); ax[1, 2].set_ylabel('lenslet y')
    if ny * nx <= 600:
        #small grids (MedRes): print the detector x / y on the lenslet grid
        for a, m in [(ax[1, 0], xl), (ax[1, 1], yl)]:
            for j in range(ny):
                for i in range(nx):
                    if np.isfinite(m[j, i]):
                        a.text(i, j, f'{m[j, i]:.0f}', fontsize=5, ha='center', va='center', color='w')
    fig.suptitle(f'{title}: rectmat rows -> detector traces (row = lensx + lensy*nx + l*nx*ny, '
                 f'positions = template centroids)')
    fig.tight_layout()
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_spot_residual_diagnostics(sres, lams, output_dir, filename, title='', dpi=110,
                                   show=False):
    """Spot model residual diagnostics (ProcessMonochrom.spot_residuals).

    Row 1: residual RMS (within fit_radius px of each spot centre, % of the spot's
    data peak) versus wavelength for the Gaussian fits and the A&K ePSF (median and
    16-84% of spots); per-lenslet median maps for both; histogram of all spots.
    Rows 2-3: median over spots of data/peak (left) and of residual/peak at the
    stacked wavelengths, Gaussian (row 2) and A&K (row 3), on a common symmetric
    colour scale, so systematic shape errors (wings, asymmetry along the trace,
    centring offsets) show up as coherent patterns.
    """
    lams = np.asarray(lams, float)
    rg, ra = np.asarray(sres['rms_gauss']), np.asarray(sres['rms_ank'])
    has_ank = np.isfinite(ra).any()
    ns = len(sres['stack_lams'])
    fig = plt.figure(figsize=(5 * max(4, ns + 1), 14))
    gs = fig.add_gridspec(3, max(4, ns + 1))
    with np.errstate(invalid='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            pg = np.nanpercentile(rg.reshape(len(lams), -1), [16, 50, 84], axis=1)
            pa = np.nanpercentile(ra.reshape(len(lams), -1), [16, 50, 84], axis=1) if has_ank else None
            mg = np.nanmedian(rg, axis=0)
            ma = np.nanmedian(ra, axis=0) if has_ank else None

    a = fig.add_subplot(gs[0, 0])
    for p, lab, col in [(pg, 'Gaussian fit', 'C0'), (pa, 'A&K ePSF (as in the A&K rectmat)', 'C3')]:
        if p is None:
            continue
        a.fill_between(lams, p[0], p[2], color=col, alpha=0.25)
        a.plot(lams, p[1], '-', color=col, label=lab + ' (median, 16-84%)')
    a.set(xlabel='wavelength (um)', ylabel='residual RMS (% of spot peak)',
          title=f'{title} spot model residuals (r <= {sres["fit_radius"]} px)')
    a.set_ylim(bottom=0); a.legend(fontsize=8)

    allm = np.concatenate([m[np.isfinite(m)] for m in [mg, ma] if m is not None])
    vmax = np.percentile(allm, 98) if allm.size else 1
    for k, (m, t) in enumerate([(mg, 'median Gaussian residual per lenslet (%)'),
                                (ma, 'median A&K residual per lenslet (%)')]):
        a = fig.add_subplot(gs[0, 1 + k])
        if m is None:
            a.axis('off'); a.set_title('no A&K ePSFs'); continue
        im = a.imshow(m, origin='lower', cmap='magma', vmin=0, vmax=vmax, interpolation='nearest')
        a.set(title=t, xlabel='lenslet x', ylabel='lenslet y')
        fig.colorbar(im, ax=a, fraction=0.046)

    a = fig.add_subplot(gs[0, 3])
    vals = [v[np.isfinite(v)] for v in [rg, ra] if np.isfinite(v).any()]
    hi = np.percentile(np.concatenate(vals), 99.5) if vals else 1
    bins = np.linspace(0, hi, 60)
    a.hist(rg[np.isfinite(rg)], bins=bins, histtype='step', color='C0', label='Gaussian')
    if has_ank:
        a.hist(ra[np.isfinite(ra)], bins=bins, histtype='step', color='C3', label='A&K')
    a.set(xlabel='residual RMS (% of spot peak)', ylabel='spots', title='all spots')
    a.legend(fontsize=8)

    st = [sres['stack_gauss'], sres['stack_ank']]
    sv = np.concatenate([s[np.isfinite(s)] for s in st if np.isfinite(s).any()])
    lim = np.percentile(np.abs(sv), 99.5) if sv.size else 0.05
    for row, (s, name) in enumerate([(sres['stack_gauss'], 'Gaussian'), (sres['stack_ank'], 'A&K')], start=1):
        a = fig.add_subplot(gs[row, 0])
        dd = sres['stack_data'][ns // 2]
        if np.isfinite(dd).any():
            a.imshow(dd, origin='lower', cmap='magma', interpolation='nearest')
        a.set_title(f'median spot / peak, {sres["stack_lams"][ns // 2]:.4f} um', fontsize=10)
        for k in range(ns):
            a = fig.add_subplot(gs[row, 1 + k])
            if not np.isfinite(s[k]).any():
                a.axis('off'); a.set_title(f'{name}: none'); continue
            im = a.imshow(s[k], origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
            a.set_title(f'{name} median residual / peak, {sres["stack_lams"][k]:.4f} um', fontsize=10)
            fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f'{title}: spot residuals (crop pixels; each spot normalised by its data peak)')
    fig.tight_layout()
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return path
