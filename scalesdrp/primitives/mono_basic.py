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
    ims_cal : ndarray, shape (nwave, detector_y, detector_x)
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
    if ims_cal.ndim != 3 or centroids.ndim != 4 or posarr.ndim != 4:
        raise ValueError('Expected ims_cal (3D), posarr (4D), centroids (4D).')
    if centroids.shape[-1] != 2 or posarr.shape[-1] < 6:
        raise ValueError('Centroids require x/y; posarr requires cutout offsets.')
    if not (ims_cal.shape[0] == posarr.shape[0] == centroids.shape[0] == len(lams)):
        raise ValueError('Wavelength axis lengths must agree.')
    if posarr.shape[1:3] != centroids.shape[1:3]:
        raise ValueError('Lenslet grid shapes must agree.')
    if len(lams) == 0:
        raise ValueError('No wavelengths provided.')

    # Convert local centroids to full-detector coordinates.
    x = posarr[..., 1] + centroids[..., 0]
    y = posarr[..., 4] + centroids[..., 1]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(lams[:, None, None])
    valid &= (x >= 0) & (x < ims_cal.shape[2]) & (y >= 0) & (y < ims_cal.shape[1])
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
    image = ims_cal[idx]
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
