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

    Row 1: residual RMS (within fit_radius px of each spot center, % of the spot's
    data peak) versus wavelength for the Gaussian fits and the A&K ePSF (median and
    16-84% of spots); per-lenslet median maps for both; histogram of all spots.
    Rows 2-3: median over spots of data/peak (left) and of residual/peak at the
    stacked wavelengths, Gaussian (row 2) and A&K (row 3), on a common symmetric
    colour scale, so systematic shape errors (wings, asymmetry along the trace,
    centering offsets) show up as coherent patterns.
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


#design band edges (um) per LowRes mode for plot_dispersion_check, from the SCALES
#model resolution curves (K-band, water ice, SEDs, L-band, CH4/PAH, M-band), plus
#extra reference wavelengths to mark (5.3 um: red end of the M-band light)
DISPERSION_CHECK_EDGES = {
    'LowRes-K': dict(edges=[1.95, 2.45]),
    'LowRes-KL': dict(edges=[2.0, 3.7]),
    'LowRes-KLM': dict(edges=[2.0, 5.0]),
    'LowRes-L': dict(edges=[2.9, 4.15]),
    'LowRes-Ls': dict(edges=[3.1, 3.5]),
    'LowRes-M': dict(edges=[4.5, 5.2], extra=[5.3]),
    #MedRes: the same K-, L- and M-band edges (5.2 um is the source's red limit)
    'MedRes-K': dict(edges=[1.95, 2.45]),
    'MedRes-L': dict(edges=[2.9, 4.15]),
    'MedRes-M': dict(edges=[4.5, 5.2]),
}


def _dispersion_panelA_medres(fig, gspec, cont, fit, lams, good, nwell, nolight, KEY, edges, cols,
                              predict, j0, i0, L0, nx):
    """Panel A of plot_dispersion_check for MedRes: a strip along the whole center
    trace (with its neighbouring traces either side) and zooms on its blue and red
    ends, with the spot centers, the predicted positions of the unused masters and
    the band-edge markers. Returns (axes, title)."""
    sub = gspec.subgridspec(1, 3, wspace=0.28, width_ratios=[0.8, 1, 1])
    g0 = good[:, j0, i0]
    xs, ys = fit[g0, j0, i0, 1], fit[g0, j0, i0, 2]
    keyw = [w for w, _, _ in KEY]
    lo_w = min([lams[g0].min()] + keyw); hi_w = max([lams[g0].max()] + keyw)
    p_lo, p_hi = predict(j0, i0, lo_w)[0], predict(j0, i0, hi_w)[0]
    H, W = cont.shape
    axes = []

    def show(a, xa, xb, ya, yb, aspect):
        xa, xb = int(max(0, xa)), int(min(W, xb)); ya, yb = int(max(0, ya)), int(min(H, yb))
        im = cont[ya:yb, xa:xb]
        if im.size == 0:
            return
        a.imshow(im, origin='lower', cmap='gray', aspect=aspect,
                 extent=(xa - .5, xb - .5, ya - .5, yb - .5),
                 vmin=np.nanpercentile(im, 25), vmax=np.nanpercentile(im, 99.5))
        a.set_xlim(xa - .5, xb - .5); a.set_ylim(ya - .5, yb - .5)

    #strip along the whole trace
    a = fig.add_subplot(sub[0, 0]); axes.append(a)
    xc = np.median(xs)
    show(a, xc - 45, xc + 45, min(ys.min(), p_lo[1], p_hi[1]) - 60,
         max(ys.max(), p_lo[1], p_hi[1]) + 60, 'auto')
    a.scatter(xs[::max(1, len(xs) // 300)] + 6, ys[::max(1, len(xs) // 300)], s=3,
              c=cols[g0][::max(1, len(xs) // 300)], edgecolors='none')
    for w, mk, col in KEY:
        q = predict(j0, i0, w)[0]
        a.scatter(q[0] + 6, q[1], s=60, marker=mk, facecolors=col, edgecolors='k', zorder=5)
    a.set(xlabel='detector x (px)', ylabel='detector y (px)')
    a.set_title(f'whole trace (spots drawn 6 px\nto the right)', fontsize=10)

    #zooms on the two ends
    for k, (w_spot, w_key, name) in enumerate([(lams[g0].min(), lo_w, 'blue'),
                                                (lams[g0].max(), hi_w, 'red')]):
        a = fig.add_subplot(sub[0, k + 1]); axes.append(a)
        c = 0.5 * (predict(j0, i0, w_spot)[0] + predict(j0, i0, w_key)[0])
        hy = max(70, 0.7 * abs(predict(j0, i0, w_key)[0][1] - predict(j0, i0, w_spot)[0][1]) + 40)
        show(a, c[0] - 0.5 * hy, c[0] + 0.5 * hy, c[1] - hy, c[1] + hy, 'equal')
        for di in (-1, 0, 1):                 #this lenslet and its neighbours in the slice
            j = j0 + di                       #(a slice runs along the first lenslet index)
            if not (0 <= j < nwell.shape[0]) or nwell[j, i0] < 10:
                continue
            g_ = good[:, j, i0]
            a.scatter(fit[g_, j, i0, 1], fit[g_, j, i0, 2], s=6, c=cols[g_], edgecolors='none', zorder=3)
            if len(nolight):
                q = predict(j, i0, nolight)
                a.scatter(q[:, 0], q[:, 1], s=14, facecolors='none', edgecolors='yellow',
                          linewidths=.8, zorder=3)
            for w, mk, col in KEY:
                q = predict(j, i0, w)[0]
                a.scatter(q[0], q[1], s=110, marker=mk, facecolors=col, edgecolors='k', linewidths=1.2, zorder=5)
                if di == 0:
                    a.annotate(f'{w:g} um', q, xytext=(12, 0), textcoords='offset points', color=col,
                               fontsize=10, fontweight='bold', va='center', ha='left', clip_on=True,
                               bbox=dict(boxstyle='round,pad=0.15', fc='k', ec='none', alpha=0.6))
            if di == 0:
                a.annotate(f'{w_spot:.3f}', predict(j, i0, w_spot)[0], xytext=(-9, 0),
                           textcoords='offset points', color='w', fontsize=8, ha='right', va='center')
        a.set_xlabel('detector x (px)')
        a.set_title(f'{name} end of the trace', fontsize=10)
        if k == 1:
            a.scatter([], [], s=9, c='magenta',
                      label=f'spot centers (Gaussian fits), {lams.min():.3f}-{lams.max():.3f} um')
            if len(nolight):
                a.scatter([], [], s=14, facecolors='none', edgecolors='yellow',
                          label=f'predicted: {len(nolight)} unused masters')
            for w, mk, col in KEY:
                a.scatter([], [], s=80, marker=mk, facecolors=col, edgecolors='k',
                          label=f'predicted {w:g} um' + (' (design band edge)' if w in edges else ''))
            a.legend(loc='upper right', bbox_to_anchor=(1.0, -0.09), fontsize=8, framealpha=.85)
    title = (f'A. background light (median of raw masters) along lenslet ({j0},{i0})\'s trace\n'
             f'with its neighbours in the slice in the end zooms')
    return axes, title


def plot_dispersion_check(cont, fit, lams, dxdl, dydl, good, all_waves, output_dir, filename,
                          edges=None, extra=None, label='', dpi=100, medres=False):
    """
    Dispersion check for a LowRes monochromator run: shows that the spot positions,
    the measured dispersion and the background light along the traces agree.

    A. Background light (median of the raw masters) along a line of traces (the
       center lenslet and its neighbours on the same line), with the Gaussian-fit spot
       centers, the predicted positions of the masters that were not used (no or faint
       monochromator light: mono_wave_range / mono_exclude_waves) and big markers at
       the design band edges (and any extra wavelengths).
    B. Center lenslet: distance along the track (arc length of the cubic x(lam), y(lam)
       fit) vs MONOWAVE, with a straight-line fit.
    C. All lenslets: straight-line distance between two spots / wavelength difference
       (no fit, no derivative).
    D. Local dispersion (derivative of each lenslet's cubic track fit, as used for R)
       vs wavelength.
    E. Background light along the trace (median over lenslets near the center
       lenslet), with the spot positions and the design band edges marked; the
       wavelength where the light drops at the red end and, where the light is
       roughly flat across the trace, the trace's half-max length and end wavelengths.

    Args:
        cont: (2048, 2048) median of the raw (not background-subtracted) masters
        fit: (nl, ny, nx, >=6) Gaussian spot fits [amp, x, y, ...]
        lams: (nl,) wavelengths of the masters used (um)
        dxdl, dydl, good: from ProcessMonochrom.trace_derivatives(fit, lams)
        all_waves: wavelengths of all masters of the mode, used or not (um)
        edges: design band edges (um); extra: more wavelengths to mark (um)
        label: mode / dataset name for the title
        medres: MedRes layout. Each trace spans most of the detector and there is no
            next trace on the same line, so A shows a strip along the whole center
            trace (with its neighbouring traces either side) and zooms on its two
            ends, and E runs along the whole trace.

    Returns (path, info): path of the saved figure and a dict with the measured
    numbers (median_disp, line_fit_disp, center lenslet, red-end wavelength, ...).
    """
    import warnings
    from scipy.ndimage import map_coordinates
    lams = np.asarray(lams, float)
    edges = list(edges or [])
    extra = list(extra or [])
    nl, ny, nx = fit.shape[:3]
    with np.errstate(invalid='ignore'):
        disp = np.hypot(dxdl, dydl)
    L0 = float(np.round(lams[nl // 2], 3))
    all_waves = np.sort(np.asarray(all_waves, float))
    nolight = np.array([w for w in all_waves if np.min(np.abs(lams - w)) > 1e-4])
    keystyle = [('o', 'lime'), ('s', 'orange'), ('^', 'red'), ('D', 'yellow'), ('v', 'cyan')]
    KEY = [(w, *keystyle[k % len(keystyle)]) for k, w in enumerate(edges + extra)]
    cols = plt.cm.cool((lams - lams.min()) / max(np.ptp(lams), 1e-9))

    def track(j, i, deg=3):
        g = good[:, j, i]
        px = np.polyfit(lams[g], fit[g, j, i, 1], deg); py = np.polyfit(lams[g], fit[g, j, i, 2], deg)
        ld = np.linspace(lams[g].min(), lams[g].max(), 8000 if medres else 2000)
        P = np.stack([np.polyval(px, ld), np.polyval(py, ld)], 1)
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        s -= np.interp(L0, ld, s)
        return g, px, py, ld, P, s

    def predict(j, i, w):
        #position of wavelength(s) w on lenslet (j, i)'s track (straight past its ends)
        g_, px_, py_, ld_, P_, s_ = track(j, i)
        lo_, hi_ = lams[g_].min(), lams[g_].max()
        w = np.atleast_1d(w).astype(float); out = np.zeros((len(w), 2))
        for k, x in enumerate(w):
            e = lo_ if x < lo_ else hi_ if x > hi_ else x
            p0 = np.array([np.polyval(px_, e), np.polyval(py_, e)])
            d0 = np.array([np.polyval(np.polyder(px_), e), np.polyval(np.polyder(py_), e)])
            out[k] = p0 + (x - e) * d0
        return out

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #center lenslet: well sampled and nearest the middle of the lenslet array
        nwell = good.sum(0)
        J, I = np.nonzero(nwell >= 0.9 * nl)
        if len(J) == 0:
            J, I = np.nonzero(nwell >= max(8, 0.5 * nl))
        if medres:
            #MedRes lenslet indices are not laid out like the detector: take the trace
            #nearest the detector center (median spot position)
            with np.errstate(invalid='ignore'):
                xm_ = np.nanmedian(np.where(good, fit[..., 1], np.nan), axis=0)[J, I]
                ym_ = np.nanmedian(np.where(good, fit[..., 2], np.nan), axis=0)[J, I]
            k0 = np.nanargmin((xm_ - cont.shape[1] / 2) ** 2 + (ym_ - cont.shape[0] / 2) ** 2)
        else:
            k0 = np.argmin((J - ny / 2) ** 2 + (I - nx / 2) ** 2)
        j0, i0 = int(J[k0]), int(I[k0])
        g, px, py, ld, P, s = track(j0, i0)
        tls = s[-1] - s[0]                                        #spot track length
        cen = lambda j, i: predict(j, i, L0)[0]
        lq1, lq3 = np.percentile(lams, [25, 75])
        u = predict(j0, i0, lq3)[0] - predict(j0, i0, lq1)[0]; u /= np.hypot(*u); v_ = np.array([-u[1], u[0]])
        #nearest neighbour lenslet whose trace lies on the same line (LowRes; a MedRes
        #trace spans the detector, so it has none)
        best = None
        for dj in (range(0, 7) if not medres else []):
            for di in range(-5, 6):
                if dj == 0 and di <= 0:
                    continue
                jj, ii = j0 + dj, i0 + di
                if not (0 <= jj < ny and 0 <= ii < nx) or nwell[jj, ii] < 10:
                    continue
                d = cen(jj, ii) - cen(j0, i0); pe = abs(d @ v_)
                if pe < 3.5 and 0.6 * tls < abs(d @ u) < 3 * tls and (best is None or pe < best[2]):
                    best = (dj, di, pe, abs(d @ u))
        if best is not None:
            dj, di = best[0], best[1]
            line = [(j0 + k * dj, i0 + k * di) for k in (-1, 0, 1)
                    if 0 <= j0 + k * dj < ny and 0 <= i0 + k * di < nx and nwell[j0 + k * dj, i0 + k * di] >= 10]
            along = best[3]
        else:
            dj = di = 0; line = [(j0, i0)]; along = 1.4 * tls
        if medres:
            along = 0.1 * tls          #sets the end-of-trace search windows in E

        fig = plt.figure(figsize=(28, 15))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1.25], wspace=0.22)
        # ---------------- A ----------------
        if medres:
            axA, A_title = _dispersion_panelA_medres(
                fig, gs[:, 0], cont, fit, lams, good, nwell, nolight, KEY, edges, cols,
                predict, j0, i0, L0, nx)
        else:
            subA = gs[:, 0].subgridspec(1, 2, wspace=0.04)
            A_title = (f'A. a line of traces around lenslet ({j0},{i0})\n' +
                       (f'neighbours on the line are ({dj:+d},{di:+d}) lenslets apart ({along:.0f} px along the trace)'
                        if best is not None else 'no neighbour found on the same line'))
            axA = []
            xc, yc = cen(j0, i0)
            hy = int(min(max(1.55 * along, 1.5 * tls), 150)); hx = int(0.46 * hy)
            x0 = int(np.clip(xc - hx, 0, cont.shape[1] - 2 * hx)); y0 = int(np.clip(yc - hy, 0, cont.shape[0] - 2 * hy))
            sub = cont[y0:y0 + 2 * hy, x0:x0 + 2 * hx]
            extA = (x0 - .5, x0 + 2 * hx - .5, y0 - .5, y0 + 2 * hy - .5)
            vlo, vhi = np.nanpercentile(sub, 25), np.nanpercentile(sub, 99.5)
            for k in range(2):
                a = fig.add_subplot(subA[0, k]); axA.append(a)
                a.imshow(sub, origin='lower', cmap='gray', extent=extA, vmin=vlo, vmax=vhi)
                a.set_xlim(x0, x0 + 2 * hx - 1); a.set_ylim(y0, y0 + 2 * hy - 1)
                a.set_xlabel('detector x (px)')
                if k == 0:
                    a.set_ylabel('detector y (px)')
                    a.set_title('background light (median of raw masters)', fontsize=10)
                    jl, il = line[0]; jh, ih = line[-1]
                    p_lo = predict(jl, il, lams.min() - 0.15 * np.ptp(lams))[0]
                    p_hi = predict(jh, ih, lams.max() + 0.15 * np.ptp(lams))[0]
                    a.plot([p_lo[0] + 6 * v_[0], p_hi[0] + 6 * v_[0]], [p_lo[1] + 6 * v_[1], p_hi[1] + 6 * v_[1]],
                           'c--', lw=.8, label='line of traces (drawn 6 px to the side)')
                    a.legend(loc='lower right', fontsize=7.5, framealpha=.85)
                    continue
                a.set_yticklabels([])
                a.set_title('same, with spot centers and predicted\npositions of the unused masters', fontsize=10)
                for (j, i) in line:
                    g_ = good[:, j, i]
                    a.scatter(fit[g_, j, i, 1], fit[g_, j, i, 2], s=6, c=cols[g_], edgecolors='none', zorder=3)
                    if len(nolight):
                        q = predict(j, i, nolight)
                        a.scatter(q[:, 0], q[:, 1], s=12, facecolors='none', edgecolors='yellow', linewidths=.8, zorder=3)
                    for w, mk, col in KEY:
                        q = predict(j, i, w)[0]
                        a.scatter(q[0], q[1], s=110, marker=mk, facecolors=col, edgecolors='k', linewidths=1.2, zorder=5)
                        if (j, i) == (j0, i0):
                            a.annotate(f'{w:g} um', q, xytext=(12, 0), textcoords='offset points', color=col,
                                       fontsize=10, fontweight='bold', va='center', ha='left',
                                       bbox=dict(boxstyle='round,pad=0.15', fc='k', ec='none', alpha=0.6))
                    if (j, i) == (j0, i0):
                        for w in (lams[g_].min(), lams[g_].max()):
                            a.annotate(f'{w:.3f}', predict(j, i, w)[0], xytext=(-9, 0), textcoords='offset points',
                                       color='w', fontsize=8, ha='right', va='center')
                    else:
                        a.annotate(f'lenslet ({j},{i})', predict(j, i, L0)[0], xytext=(-10, 0),
                                   textcoords='offset points', color='w', fontsize=8, va='center', ha='right')
                a.scatter([], [], s=9, c='magenta',
                          label=f'spot centers (Gaussian fits), {lams.min():.3f}-{lams.max():.3f} um')
                if len(nolight):
                    a.scatter([], [], s=12, facecolors='none', edgecolors='yellow',
                              label=f'predicted: {len(nolight)} unused masters\n(no or faint monochromator light)')
                for w, mk, col in KEY:
                    a.scatter([], [], s=80, marker=mk, facecolors=col, edgecolors='k',
                              label=f'predicted {w:g} um' + (' (design band edge)' if w in edges else ''))
                a.legend(loc='lower right', fontsize=7.5, framealpha=.85)

        # ---------------- B ----------------
        a = fig.add_subplot(gs[0, 1])
        sp = np.interp(lams[g], ld, s)
        a.plot(lams[g], sp, 'o', ms=4, label='spot centers (Gaussian fits)')
        c1 = np.polyfit(lams[g], sp, 1)
        a.plot(ld, np.polyval(c1, ld), 'k-', lw=1, label=f'straight-line fit: {c1[0]:.1f} px/um')
        a.set(xlabel='MONOWAVE (um)', ylabel=f'distance along track from {L0:g} um spot (px)',
              title=f'B. lenslet ({j0},{i0}): position along the track vs wavelength')
        a.legend(); a.grid(alpha=.3)

        # ---------------- C ----------------
        a = fig.add_subplot(gs[0, 2])
        ngood = good.reshape(nl, -1).sum(1); wl = np.nonzero(ngood >= 0.5 * ngood.max())[0]
        klo, khi = wl[int(round(0.1 * (len(wl) - 1)))], wl[int(round(0.9 * (len(wl) - 1)))]
        ok = good[klo] & good[khi]
        d_mean = (np.hypot(fit[khi, ..., 1] - fit[klo, ..., 1], fit[khi, ..., 2] - fit[klo, ..., 2])
                  / (lams[khi] - lams[klo]))
        dm = d_mean[ok]; mdm = float(np.median(dm))
        #range: the spread of the values (padded), at most +/-15% of the median
        q0, q1 = np.percentile(dm, [0.5, 99.5]); pad_ = max(0.5 * (q1 - q0), 0.002 * mdm)
        a.hist(dm, bins=np.linspace(max(0.85 * mdm, q0 - pad_), min(1.15 * mdm, q1 + pad_), 60),
               color='C0', alpha=.8)
        a.axvline(mdm, color='k', label=f'median {mdm:.1f} px/um (N={ok.sum()} lenslets)')
        a.set(xlabel='px/um', ylabel='lenslets',
              title=f'C. all lenslets: straight-line distance between the {lams[klo]:.3f} and '
                    f'{lams[khi]:.3f} um spots\n/ wavelength difference (no fit, no derivative)')
        a.legend()

        # ---------------- D ----------------
        a = fig.add_subplot(gs[1, 1])
        pc = np.nanpercentile(np.where(good, disp, np.nan).reshape(nl, -1), [16, 50, 84], axis=1)
        a.fill_between(lams, pc[0], pc[2], alpha=.3, label='16-84% over lenslets')
        a.plot(lams, pc[1], 'k-', label='median')
        lo_, hi_ = np.nanmin(pc[0]), np.nanmax(pc[2]); pad = 0.15 * (hi_ - lo_ + 1)
        a.set(xlabel='wavelength (um)', ylabel='local dispersion ds/dlambda (px/um)', ylim=(lo_ - pad, hi_ + pad),
              title="D. local dispersion used for R (derivative of each lenslet's cubic track fit)")
        a.legend()

        # ---------------- E ----------------
        a = fig.add_subplot(gs[1, 2])
        RE = 1.2 * along if best is not None else (0.62 * tls if medres else 1.3 * tls)
        ext = np.arange(-RE, RE + 1e-9, 0.25)
        profs, smark = [], []
        for j in range(max(0, j0 - 11), min(ny, j0 + 12)):
            for i in range(max(0, i0 - 11), min(nx, i0 + 12)):
                if nwell[j, i] < 0.85 * nl:
                    continue
                g2, px2, py2, ld2, P2, s2 = track(j, i)
                u0 = (P2[1] - P2[0]) / np.hypot(*(P2[1] - P2[0]))
                u1 = (P2[-1] - P2[-2]) / np.hypot(*(P2[-1] - P2[-2]))
                #points along the track at arc lengths ext, straight past its ends
                pts = np.stack([np.interp(ext, s2, P2[:, 0]), np.interp(ext, s2, P2[:, 1])], 1)
                lo_e, hi_e = ext < s2[0], ext > s2[-1]
                pts[lo_e] = P2[0] + (ext[lo_e] - s2[0])[:, None] * u0
                pts[hi_e] = P2[-1] + (ext[hi_e] - s2[-1])[:, None] * u1
                n = np.gradient(pts, axis=0); n /= np.hypot(*n.T)[:, None]; vv = np.stack([-n[:, 1], n[:, 0]], 1)
                pr = np.mean([map_coordinates(cont, [pts[:, 1] + t * vv[:, 1], pts[:, 0] + t * vv[:, 0]],
                                              order=1, cval=np.nan) for t in (-0.5, 0, 0.5)], axis=0)
                profs.append(pr)
                sm_ = np.full(nl, np.nan); sm_[g2] = np.interp(lams[g2], ld2, s2); smark.append(sm_)
        if not profs:                     #fall back to the center lenslet alone
            profs = [np.full(len(ext), np.nan)]; smark = [np.where(g, np.interp(lams, ld, s), np.nan)]
        profs = np.array(profs); med = np.nanmedian(profs, 0)
        sm = np.nanmedian(np.array(smark), 0)
        a.plot(ext, med, 'k-', lw=2,
               label=f'background light along the trace (median of {len(profs)} lenslets near the center)')
        base, top = np.nanpercentile(med, 2), np.nanmax(med)
        smok = np.isfinite(sm); sf, sl = sm[smok].min(), sm[smok].max()

        def s_at(w):
            if lams[smok][0] <= w <= lams[smok][-1]: return np.interp(w, lams[smok], sm[smok])
            if w > lams[smok][-1]: return sm[smok][-1] + (w - lams[smok][-1]) * pc[1][smok][-1]
            return sm[smok][0] - (lams[smok][0] - w) * pc[1][smok][0]

        def lam_at(e):
            if sf <= e <= sl: return np.interp(e, sm[smok], lams[smok])
            if e > sl: return lams[smok][-1] + (e - sl) / pc[1][smok][-1]
            return lams[smok][0] - (sf - e) / pc[1][smok][0]

        #this lenslet's own trace. Red end: where the background light drops half way from
        #its peak near the red end to the minimum after it. Blue end (and so the full
        #length): only where the light is roughly flat across the trace (the thermal
        #background rises steeply with wavelength in the broad modes, so there its blue
        #half-max point is not the end of the trace)
        TL = lam_lo = lam_hi = e_lo = e_hi = np.nan; flat = False
        wl_ = (ext > sf - 0.6 * along) & (ext < sf + 0.15 * (sl - sf))
        wr_ = (ext > sl - 0.15 * (sl - sf)) & (ext < sl + 0.6 * along)
        if wr_.any() and np.isfinite(med[wr_]).any():
            rmin = np.nonzero(wr_)[0][np.nanargmin(med[wr_])]
            inside = (ext > sf) & (ext < ext[rmin])
            if inside.any() and np.isfinite(med[inside]).any():
                kpk = np.nonzero(inside)[0][np.nanargmax(med[inside])]; peak = med[kpk]
                if (peak - med[rmin]) / abs(peak) > 0.10:
                    rh = 0.5 * (peak + med[rmin]); k = rmin
                    while k > kpk and med[k] < rh: k -= 1
                    e_hi = np.interp(rh, [med[k + 1], med[k]], [ext[k + 1], ext[k]]); lam_hi = lam_at(e_hi)
                    a.plot(e_hi, rh, 'o', color='C2', ms=8, zorder=6)
                    #flat enough for a blue end? light just inside the blue end of the spot
                    #range at least 60% of the way from the minimum to the peak
                    nearf = (ext > sf + 0.15 * (sl - sf)) & (ext < sf + 0.30 * (sl - sf))
                    flat = (((np.nanmedian(med[nearf]) - med[rmin]) > 0.6 * (peak - med[rmin]))
                            if nearf.any() else False)
                    if flat and wl_.any():
                        lmin = np.nonzero(wl_)[0][np.nanargmin(med[wl_])]
                        plat = np.nanmedian(med[(ext > sf + 0.2 * (sl - sf)) & (ext < sl - 0.2 * (sl - sf))])
                        lh = 0.5 * (plat + med[lmin]); k = lmin
                        while k < kpk and med[k] < lh: k += 1
                        e_lo = np.interp(lh, [med[k - 1], med[k]], [ext[k - 1], ext[k]])
                        lam_lo = lam_at(e_lo); TL = e_hi - e_lo
                        a.plot(e_lo, lh, 'o', color='C2', ms=8, zorder=6)
                        yarr = base + 0.75 * (plat - base)
                        a.annotate('', xy=(e_hi, yarr), xytext=(e_lo, yarr),
                                   arrowprops=dict(arrowstyle='<->', color='C2', lw=1.5))
                        a.text(0.5 * (e_lo + e_hi), yarr + 0.01 * (top - base),
                               f"this lenslet's trace: {TL:.1f} px at half max\n"
                               f'= {lam_lo:.2f} - {lam_hi:.2f} um with the measured dispersion',
                               color='C2', ha='center', va='bottom', fontsize=11)
                    else:
                        a.annotate(f'background light drops to half\nat {e_hi:.1f} px = {lam_hi:.2f} um\n'
                                   f'(measured dispersion)', (e_hi, rh), xytext=(-12, 0),
                                   textcoords='offset points', color='C2', ha='right', va='center',
                                   fontsize=11, zorder=7,
                                   bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
        span = top - base
        for k in range(0, nl, max(1, nl // 14)):
            if np.isfinite(sm[k]):
                a.axvline(sm[k], color=cols[k], lw=.8)
                a.text(sm[k], top + 0.03 * span, f'{lams[k]:.2f}', rotation=90, fontsize=8, ha='center', va='bottom')
        ks = sorted(KEY, key=lambda t: t[0]); xs_all = [s_at(w) for w, _, _ in ks]
        for n_, ((w, mk, col), xs_) in enumerate(zip(ks, xs_all)):
            a.axvline(xs_, color=col, lw=2.5, ls='--', zorder=1)
            a.plot(xs_, top + 0.42 * span, marker=mk, ms=12, mfc=col, mec='k', zorder=5, clip_on=False)
            close_l = n_ > 0 and abs(xs_ - xs_all[n_ - 1]) < 0.12 * RE
            close_r = n_ < len(ks) - 1 and abs(xs_all[n_ + 1] - xs_) < 0.12 * RE
            off_, ha_ = (0.012 * RE, 'left') if close_l else (-0.012 * RE, 'right') if close_r else (0, 'center')
            a.text(xs_ + off_, top + 0.48 * span, f'{w:g} um', color=col, fontsize=10, fontweight='bold',
                   va='bottom', ha=ha_)
        if best is not None:
            for xx in (-0.88 * RE, 0.88 * RE):
                a.text(xx, base + 0.3 * span, "next lenslet's\ntrace on the\nsame line", ha='center',
                       fontsize=9, color='grey')
        a.set(xlabel=f'distance along the trace from the {L0:g} um spot (px; extended straight past the end spots)',
              ylabel='median of raw masters (DN/s)', xlim=(-RE, RE),
              title='E. background light along the trace, with the monochromator spot positions marked (um)' +
                    ('' if flat else '\n(background light rises steeply with wavelength, so only the red end '
                                     'of the trace is marked)'))
        if np.isfinite(span) and span > 0:
            a.set_ylim(base - 0.05 * span, top + 0.62 * span)
        a.legend(loc='lower center')

        band = f' (design band {edges[0]:g}-{edges[1]:g} um)' if len(edges) >= 2 else ''
        #trace length and end wavelengths: from the background light at half max where it
        #is roughly flat across the trace; otherwise (thermal background rising with
        #wavelength) the first-to-last monochromator spot track, plus the red-end drop
        spot_lo, spot_hi = lams[g].min(), lams[g].max()
        if np.isfinite(TL):
            trace_str = (f'trace length {TL:.1f} px (background light at half max), '
                         f'{lam_lo:.3f} - {lam_hi:.3f} um')
        else:
            trace_str = (f'trace length {tls:.1f} px (first to last monochromator spot, lenslet '
                         f'({j0},{i0})), {spot_lo:.3f} - {spot_hi:.3f} um')
            if np.isfinite(lam_hi):
                trace_str += (f'; background light ends at {lam_hi:.3f} um '
                              f'(blue end not measurable)')
        fig.suptitle(f'{label} dispersion check: measured {mdm:.0f} px/um from the monochromator spot '
                     f'positions{band}\n{trace_str}', fontsize=14)
        fig.tight_layout()
        p0, p1 = axA[0].get_position(), axA[-1].get_position()
        fig.text(0.5 * (p0.x0 + p1.x1), p0.y1 + 0.045, A_title, ha='center', va='bottom', fontsize=11)
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        fig.savefig(path, dpi=dpi, bbox_inches='tight'); plt.close(fig)
        info = dict(center_lenslet=(j0, i0), track_len_px=float(tls), line_fit_disp=float(c1[0]),
                    median_disp=mdm, neighbour=(dj, di) if best is not None else None,
                    red_end_um=float(lam_hi), trace_len_px=float(TL), blue_end_um=float(lam_lo),
                    flat=bool(flat), n_unused=len(nolight),
                    edge_pos_px={float(w): float(s_at(w)) for w, _, _ in KEY})
    return path, info
