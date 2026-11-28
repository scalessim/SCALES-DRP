"""Quality-control indicators for SCALES DRP products.

See QC_indicators_proposal.md (DRP_testing) for the indicator definitions.
"""
import numpy as np
import astropy.io.fits as pyfits


def _robust_rms(v, nsig=5.0):
    """RMS after clipping |v - median| > nsig * 1.4826 MAD. Returns (rms, outlier fraction)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    keep = np.abs(v - med) <= nsig * mad if mad > 0 else np.ones(v.size, bool)
    return float(np.sqrt(np.mean(v[keep] ** 2))), float(1.0 - keep.mean())


def wavelength_loo_residuals(lams, fitarr, deg=3, cov_frac=0.9, drop_ends=True):
    """Leave-one-wavelength-out test of the per-lenslet wavelength interpolation (QWAVRMS).

    Mirrors ProcessMonochrom.interp_gauss_spots(method='poly'): each lenslet with fits at
    more than cov_frac of the wavelengths has its spot x and y positions fit by a
    degree-`deg` polynomial in wavelength. Here each fitted wavelength is predicted from
    a fit to all the others, and the prediction error is split into the components along
    and across the local trace direction.

    The leave-one-out residuals are exact and computed without refitting, from the
    least-squares hat matrix: e_i = r_i / (1 - h_ii).

    Parameters
    ----------
    lams : (nlam,) monochromator wavelengths [um]
    fitarr : (nlam, ny, nx, >=3) Gaussian spot parameters [amp, x, y, ...]
    drop_ends : exclude the shortest and longest fitted wavelength of each lenslet from
        the summary. Leaving those out tests extrapolation, which the interpolated
        products never do.

    Returns
    -------
    dict with per-spot arrays 'along', 'cross' [px] and 'dlam' [um] (nlam, ny, nx; NaN
    where not tested), per-lenslet RMS maps 'rms_along_map', 'rms_cross_map' (ny, nx),
    and the summary 'cards': {key: (value, comment)} for the QC header.
    """
    lams = np.asarray(lams, float)
    fitarr = np.asarray(fitarr, float)
    nlam, ny, nx = fitarr.shape[:3]
    along = np.full((nlam, ny, nx), np.nan)
    cross = np.full((nlam, ny, nx), np.nan)
    dlam = np.full((nlam, ny, nx), np.nan)
    l0 = 0.5 * (lams.min() + lams.max())
    ls = 0.5 * (lams.max() - lams.min()) or 1.0
    nlens = 0
    for iy in range(ny):
        for ix in range(nx):
            amp = fitarr[:, iy, ix, 0]
            pos = fitarr[:, iy, ix, 1:3]
            if np.count_nonzero(np.isfinite(amp)) <= cov_frac * nlam:
                continue
            ok = np.isfinite(pos).all(axis=1)
            idx = np.where(ok)[0]
            if idx.size < deg + 3:
                continue
            t = (lams[idx] - l0) / ls
            V = np.vander(t, deg + 1)
            try:
                G = np.linalg.inv(V.T @ V)
            except np.linalg.LinAlgError:
                continue
            beta = G @ (V.T @ pos[idx])
            r = pos[idx] - V @ beta
            h = np.einsum('ij,jk,ik->i', V, G, V)
            e = r / (1.0 - h)[:, None]
            # local trace direction from the full fit: d(x, y)/d(lambda)
            dV = np.vander(t, deg + 1)[:, 1:] * np.arange(deg, 0, -1)
            dpos = (dV @ beta[:-1]) / ls  # px per um
            speed = np.hypot(dpos[:, 0], dpos[:, 1])
            with np.errstate(invalid='ignore', divide='ignore'):
                tx, tyy = dpos[:, 0] / speed, dpos[:, 1] / speed
            a = e[:, 0] * tx + e[:, 1] * tyy
            c = -e[:, 0] * tyy + e[:, 1] * tx
            sel = np.ones(idx.size, bool)
            if drop_ends:
                sel[[0, -1]] = False
            along[idx[sel], iy, ix] = a[sel]
            cross[idx[sel], iy, ix] = c[sel]
            with np.errstate(invalid='ignore', divide='ignore'):
                dlam[idx[sel], iy, ix] = a[sel] / speed[sel]
            nlens += 1

    with np.errstate(invalid='ignore'), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        rms_along_map = np.sqrt(np.nanmean(along ** 2, axis=0))
        rms_cross_map = np.sqrt(np.nanmean(cross ** 2, axis=0))

    rms_a, out_a = _robust_rms(along)
    rms_c, _ = _robust_rms(cross)
    rms_l, _ = _robust_rms(dlam)
    fa = np.abs(along[np.isfinite(along)])
    p90 = float(np.percentile(fa, 90)) if fa.size else np.nan
    cards = {
        'QWAVRMS': (rms_a, 'LOO wavelength interp resid, along trace [px]'),
        'QWAVP90': (p90, 'LOO |resid| along trace, 90th pct [px]'),
        'QWAVXRM': (rms_c, 'LOO wavelength interp resid, across trace [px]'),
        'QWAVNM': (rms_l * 1e3 if np.isfinite(rms_l) else np.nan,
                   'LOO wavelength interp resid [nm]'),
        'QWAVOUT': (out_a, 'Fraction of LOO resid clipped as >5 sigma'),
        'QWAVNLN': (int(nlens), 'Lenslets in the LOO wavelength test'),
        'QWAVDEG': (int(deg), 'Polynomial degree of wavelength interp'),
    }
    return dict(along=along, cross=cross, dlam=dlam, rms_along_map=rms_along_map,
                rms_cross_map=rms_cross_map, cards=cards)


def shift_residual_improvement(ref_im, new_im, xy, offsets, half, ref_noise=None,
                               new_noise=None):
    """Image-based test of a rectmat shift (RSIMPRV).

    For each spot, the stamp of new_im around its shifted position is modelled by the
    reference image (what the scan's rectmat was built from) in two ways: unshifted,
    i.e. what the scan's rectmat assumes, and moved by the fitted offsets, i.e. what the
    shifted rectmat assumes. Each model gets its own per-spot scale and offset, so lamp
    flux changes between the frames don't count. The pixels are the same for both.

    This uses every pixel of each stamp, not the spot fits the shift came from, so it
    is an independent check that the shift lines the reference spots up with the data.

    Parameters
    ----------
    ref_im, new_im : 2-D background-subtracted reference (scan) and new images
    xy : (N, 2) spot positions in the reference image [px]
    offsets : (N, 2) fitted offsets (new - reference) at those spots [px]
    half : stamp half-size [px]; the stamp is (2*half+1)^2
    ref_noise, new_noise : per-pixel noise of the two images, for the normalised residual

    Returns
    -------
    dict with per-spot 'rms_before', 'rms_after' and the QC 'cards'.
    """
    from scipy.ndimage import map_coordinates, spline_filter
    ref = np.nan_to_num(np.asarray(ref_im, float))
    new = np.nan_to_num(np.asarray(new_im, float))
    ny, nx = new.shape
    coeffs = spline_filter(ref, order=3)
    g = np.arange(-half, half + 1)
    gy, gx = np.meshgrid(g, g, indexing='ij')
    before, after, scales = [], [], []
    for (x0, y0), (ox, oy) in zip(np.asarray(xy, float), np.asarray(offsets, float)):
        cx, cy = int(round(x0 + ox)), int(round(y0 + oy))
        if cx - half < 0 or cy - half < 0 or cx + half >= nx or cy + half >= ny:
            continue
        yy, xx = gy + cy, gx + cx
        d = new[yy, xx].ravel()
        res = []
        for sx, sy in ((0.0, 0.0), (ox, oy)):
            m = map_coordinates(coeffs, [(yy - sy).ravel(), (xx - sx).ravel()],
                                order=3, prefilter=False, mode='nearest')
            A = np.column_stack([m, np.ones_like(m)])
            coef = np.linalg.lstsq(A, d, rcond=None)[0]
            res.append(np.sqrt(np.mean((d - A @ coef) ** 2)))
        before.append(res[0])
        after.append(res[1])
        scales.append(coef[0])
    before, after = np.array(before), np.array(after)
    n = before.size
    if n:
        b_med, a_med = float(np.median(before)), float(np.median(after))
        improve = a_med / b_med if b_med > 0 else np.nan
        frac_worse = float(np.mean(after > before))
    else:
        b_med = a_med = improve = frac_worse = np.nan
    noise = np.nan
    if ref_noise is not None and new_noise is not None and n:
        s = float(np.median(scales))
        noise = float(np.hypot(new_noise, s * ref_noise))
    cards = {
        'RSIMPRV': (improve, 'Median spot resid RMS, shifted / unshifted'),
        'RSRESB': (b_med, 'Median spot resid RMS, unshifted model'),
        'RSRESA': (a_med, 'Median spot resid RMS, shifted model'),
        'RSRESAN': (a_med / noise if np.isfinite(noise) and noise > 0 else np.nan,
                    'Shifted-model resid RMS / pixel noise'),
        'RSWORSE': (frac_worse, 'Fraction of spots worse after the shift'),
        'RSNIMPR': (int(n), 'Spots in the RSIMPRV test'),
    }
    return dict(rms_before=before, rms_after=after, cards=cards)


QC_VERSION = '1.0'

# rmat_shift thresholds (see QC_indicators_proposal.md, section B)
RS_SCAT_MAX = 0.75       # px, per-spot scatter about the model (hard)
RS_IMPRV_MAX = 1.05      # shifted / unshifted residual RMS (hard)
RS_ERR_WARN = 0.05       # px, shift uncertainty
RS_PCDIF_WARN = 0.25     # px, Gaussian-fit vs cross-correlation shift
RS_CENDF_WARN = 0.10     # px, Gaussian-fit vs centroid shift
RS_OUTFR_WARN = 0.10     # fraction of spots clipped
RS_GRAD_WARN = 0.10      # px per 1000 px, shift-only model
RS_SCALE_WARN = 1e-3     # |affine scale - 1|
RS_ROT_WARN = 0.02       # deg, affine rotation
RS_SHMAG_WARN = 0.30     # |shift| / lenslet pitch
RS_RESAN_WARN = 2.0      # shifted-model residual / pixel noise

RS_HARD = ['RSMETHOD', 'RSEDGE', 'RSSCAT', 'RSNSPOT', 'RSBKGOK', 'RSWAVDF', 'RSMCLK',
           'RSNKIND', 'RSIMPRV', 'QCPAR']
RS_WARN = ['RSDXERR', 'RSPCDIF', 'RSCENDF', 'RSOUTFR', 'RSGRAD', 'RSAFFIN', 'RSSHMAG',
           'RSRESAN']


def _f(v):
    """float, or nan for None / non-numeric."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def rmat_shift_qc(v):
    """QC cards and pass/fail bits for one rmat_shift result.

    v is a dict of the measured values (keys below; missing ones count as missing):
      method, xcorr_on_edge, scatter, n_spots, min_spots, neg_frac, neg_frac_max,
      ref_wave, monowave, tol_nm, mclock_frame, mclock_scan, bkg_mismatch (list of str),
      n_written, n_requested (None when no rectmats were requested, e.g. measure only),
      dx, dy, dx_err, dy_err, dx_pc, dy_pc, dx_centroid, dy_centroid, outlier_frac,
      dx_grad, dy_grad (each (gx, gy) or None), affine (dict or None), pitch ((px, py)
      or None), rs_cards (from shift_residual_improvement), parent_flag ('GOOD', 'BAD'
      or None).

    Rule: a check whose value is missing fails; a check that does not apply (the step
    was not run) is N/A and is not counted.

    Returns (cards, failed, warned): cards is {key: (value, comment)} in header order,
    failed and warned the lists of failed hard checks and raised warnings.
    """
    nan = np.nan
    fail, warn = {}, {}
    vals = {}

    def hard(key, value, bad, comment, na=False):
        vals[key] = (value, comment)
        if not na:
            fail[key] = bool(bad)

    def soft(key, bad):
        warn[key] = bool(bad)

    method = v.get('method')
    hard('RSMETHOD', str(method), method != 'gauss_fit',
         'Shift method (gauss_fit; xcorr = fallback)')
    edge = v.get('xcorr_on_edge')
    hard('RSEDGE', bool(edge) if edge is not None else 'N/A', edge is None or bool(edge),
         'Xcorr peak on search-box edge')
    scat = _f(v.get('scatter'))
    hard('RSSCAT', scat, not scat <= RS_SCAT_MAX, f'Per-spot scatter about model [px], max {RS_SCAT_MAX}')
    ns, nmin = v.get('n_spots'), v.get('min_spots', 20)
    hard('RSNSPOT', int(ns) if ns is not None else 'N/A', ns is None or ns < nmin,
         f'Spots passing S/N cut, min {nmin}')
    nf = v.get('neg_frac')
    nfmax = v.get('neg_frac_max', 0.1)
    vals['RSBKGNF'] = (_f(nf), 'Spots with a negative bkg spot nearby, frac')
    hard('RSBKGOK', (not nf > nfmax) if nf is not None else 'N/A', nf is not None and nf > nfmax,
         f'Background spot-free (RSBKGNF <= {nfmax})', na=nf is None)
    wdf = abs(_f(v.get('ref_wave')) - _f(v.get('monowave')))
    tol = v.get('tol_nm', 0.5)
    hard('RSWAVDF', wdf, not wdf <= tol, f'|scan - frame MONOWAVE| [nm], max {tol}')
    mf, ms = v.get('mclock_frame'), v.get('mclock_scan')
    mism = list(v.get('bkg_mismatch') or [])
    if mf is None or ms is None:
        mc_na, mc_bad = not mism, bool(mism)
    else:
        mc_na, mc_bad = False, str(mf).strip() != str(ms).strip() or bool(mism)
    hard('RSMCLK', ('N/A' if mc_na else (not mc_bad)), mc_bad,
         'MCLOCK/EXPTIME/mode match scan and bkg', na=mc_na)
    if mism:
        vals['RSMCLKX'] = (','.join(mism)[:68], 'Mismatched keys')
    nw, nr = v.get('n_written'), v.get('n_requested')
    if nr:
        fr = nw / nr
        hard('RSNKIND', fr, fr < 1, 'Rectmat kinds written / requested')
    else:
        hard('RSNKIND', 'N/A', False, 'Rectmat kinds written / requested', na=True)
    rs = v.get('rs_cards') or {}
    imp = _f(rs.get('RSIMPRV', (nan,))[0])
    hard('RSIMPRV', imp, not imp <= RS_IMPRV_MAX,
         f'Spot resid RMS shifted/unshifted, max {RS_IMPRV_MAX}')
    pf = v.get('parent_flag')
    hard('QCPAR', pf if pf else 'N/A', pf == 'BAD', 'QCFLAG of the parent scan calibration',
         na=not pf)

    # ---- values and warnings
    aff = v.get('affine')
    vals['RSMODEL'] = ('affine' if aff else 'shift', 'Shift model applied')
    if aff and aff.get('coef') is not None:
        sx, sy = _f(aff['coef'][0][0]), _f(aff['coef'][1][0])
    else:
        sx, sy = _f(v.get('dx')), _f(v.get('dy'))
    vals['RSDX'] = (sx, 'Applied x shift [px] (at det. center if affine)')
    vals['RSDY'] = (sy, 'Applied y shift [px] (at det. center if affine)')
    ex, ey = _f(v.get('dx_err')), _f(v.get('dy_err'))
    vals['RSDXERR'] = (ex, 'x shift uncertainty [px]')
    vals['RSDYERR'] = (ey, 'y shift uncertainty [px]')
    soft('RSDXERR', max(ex, ey) > RS_ERR_WARN)
    pc = max(abs(_f(v.get('dx')) - _f(v.get('dx_pc'))), abs(_f(v.get('dy')) - _f(v.get('dy_pc'))))
    vals['RSPCDIF'] = (pc, '|fit - xcorr shift| [px]')
    soft('RSPCDIF', pc > RS_PCDIF_WARN)
    cd = max(abs(_f(v.get('dx')) - _f(v.get('dx_centroid'))),
             abs(_f(v.get('dy')) - _f(v.get('dy_centroid'))))
    vals['RSCENDF'] = (cd, '|fit - centroid shift| [px]')
    soft('RSCENDF', cd > RS_CENDF_WARN)
    of = _f(v.get('outlier_frac'))
    vals['RSOUTFR'] = (of, 'Fraction of spots clipped as outliers')
    soft('RSOUTFR', of > RS_OUTFR_WARN)
    gx, gy = v.get('dx_grad'), v.get('dy_grad')
    if gx is not None and gy is not None:
        vals['RSGRADX'] = (float(np.hypot(*gx)), '|d(dx)/d(x,y)| [px per 1000 px]')
        vals['RSGRADY'] = (float(np.hypot(*gy)), '|d(dy)/d(x,y)| [px per 1000 px]')
        # a gradient only matters when the shift-only model was applied
        soft('RSGRAD', not aff and max(np.abs(gx).max(), np.abs(gy).max()) > RS_GRAD_WARN)
    if aff:
        vals['RSSCLX'] = (_f(aff.get('scale_x')), 'Affine x scale')
        vals['RSSCLY'] = (_f(aff.get('scale_y')), 'Affine y scale')
        vals['RSROT'] = (_f(aff.get('rotation_deg')), 'Affine rotation [deg]')
        vals['RSSHEAR'] = (_f(aff.get('shear')), 'Affine shear')
        soft('RSAFFIN', max(abs(_f(aff.get('scale_x')) - 1), abs(_f(aff.get('scale_y')) - 1))
             > RS_SCALE_WARN or abs(_f(aff.get('rotation_deg'))) > RS_ROT_WARN)
    pitch = v.get('pitch')
    if pitch:
        sm = max(abs(_f(v.get('dx'))) / pitch[0], abs(_f(v.get('dy'))) / pitch[1])
        vals['RSSHMAG'] = (sm, '|shift| / lenslet pitch')
        soft('RSSHMAG', sm > RS_SHMAG_WARN)
    for k in ('RSRESAN', 'RSRESB', 'RSRESA', 'RSWORSE', 'RSNIMPR'):
        if k in rs:
            vals[k] = rs[k]
    soft('RSRESAN', _f(rs.get('RSRESAN', (nan,))[0]) > RS_RESAN_WARN)

    # ---- combined flag
    fbits = sum(1 << i for i, k in enumerate(RS_HARD) if fail.get(k))
    wbits = sum(1 << i for i, k in enumerate(RS_WARN) if warn.get(k))
    failed = [k for k in RS_HARD if fail.get(k)]
    warned = [k for k in RS_WARN if warn.get(k)]
    cards = {
        'QCFLAG': ('BAD' if fbits else 'GOOD', 'Overall QC: BAD if any hard check failed'),
        'QCVER': (QC_VERSION, 'QC scheme version'),
        'QCBITS': (int(fbits), 'Failed hard checks (bit n = QCBn)'),
        'QCWBITS': (int(wbits), 'Raised warnings (bit n = QCWn)'),
        'QCNWARN': (len(warned), 'Number of warnings'),
        'QCNNA': (len([k for k in RS_HARD if k not in fail]), 'Hard checks not applicable'),
    }
    for i, k in enumerate(RS_HARD):
        cards[f'QCB{i}'] = (k, 'Hard check for bit %d' % i)
    for i, k in enumerate(RS_WARN):
        cards[f'QCW{i}'] = (k, 'Warning for bit %d' % i)
    cards.update(vals)
    return cards, failed, warned


def cards_to_header(cards):
    hdr = pyfits.Header()
    set_cards(hdr, cards)
    return hdr


def set_cards(hdr, cards):
    """Write {key: (value, comment)} to a FITS header; non-finite floats become 'N/A'."""
    for k, (v, c) in cards.items():
        if isinstance(v, np.generic):
            v = v.item()
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            v = 'N/A'
        if isinstance(v, float):
            v = float(f'{v:.6g}')
        hdr[k] = (v, c)
    return hdr


def update_qc_file(path, cards, images=None, ifsmode=None):
    """Create or update a QC summary file (e.g. {scmode}_calib_qc.fits).

    Cards are merged into the primary header, so steps can add their own indicators
    one at a time. images: {EXTNAME: 2-D array} written (replaced) as image extensions.
    """
    import os
    if os.path.exists(path):
        with pyfits.open(path) as hl:
            hdus = [h.copy() for h in hl]
    else:
        hdus = [pyfits.PrimaryHDU()]
        if ifsmode is not None:
            hdus[0].header['IFSMODE'] = ifsmode
    set_cards(hdus[0].header, cards)
    for name, arr in (images or {}).items():
        hdus = [h for h in hdus if h.name != name.upper()]
        hdus.append(pyfits.ImageHDU(np.asarray(arr, np.float32), name=name.upper()))
    pyfits.HDUList(hdus).writeto(path, overwrite=True)
    return path
