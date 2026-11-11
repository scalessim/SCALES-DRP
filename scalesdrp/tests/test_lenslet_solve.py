"""Tests for the per-lenslet least-squares solver (scalesdrp.core.lenslet_solve).

The reference is a frozen copy of the original serial loop from
scales_basic.solve_weighted_lsq_per_lenslet (main @ c9f5df9). The lsmr kernel
must reproduce it bit for bit, serially and in parallel.
"""
import json
import os
import types

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr

from scalesdrp.core import lenslet_block_cache as lbc
from scalesdrp.core import lenslet_solve as ls

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL_RECTMAT = os.path.join(REPO, "chi", "512_rectmat_1.npz")


# --------------------------------------------------------------------------
# Frozen reference: the original per-lenslet loop
# --------------------------------------------------------------------------

def reference_solve(R_matrix, data_vector, variance_vector, A_guess, flux_shape_3d,
                    bad_pixel_mask=None, fill_value=0.0, tolerance=1e-10, maxiter=100000):
    nlam, ny, nx = flux_shape_3d
    data_vector = np.asarray(data_vector, dtype=np.float64).ravel()
    variance = np.asarray(variance_vector, dtype=np.float64).ravel()
    good = np.isfinite(data_vector) & np.isfinite(variance) & (variance > 0)
    weights = np.zeros_like(variance)
    weights[good] = 1.0 / np.sqrt(variance[good])
    if bad_pixel_mask is not None:
        good &= ~np.asarray(bad_pixel_mask, dtype=bool).ravel()
    R_csc = R_matrix.tocsc()
    A_guess_cube = None
    if A_guess is not None:
        A_guess_cube = np.nan_to_num(np.asarray(A_guess, dtype=np.float64)
                                     ).reshape(flux_shape_3d)
    flux_cube = np.full(flux_shape_3d, fill_value, dtype=np.float64)
    chi2_per_dof = np.full((ny, nx), np.nan)
    n_dof = np.zeros((ny, nx), dtype=int)
    istop_arr = np.zeros((ny, nx), dtype=int)
    n_iters = np.zeros((ny, nx), dtype=int)
    failed = np.zeros((ny, nx), dtype=bool)
    lam_offsets = np.arange(nlam) * (nx * ny)
    for yy in range(ny):
        for xx in range(nx):
            cols = xx + yy * nx + lam_offsets
            R_cut = R_csc[:, cols]
            rows = np.unique(R_cut.indices)
            rows = rows[good[rows]]
            if len(rows) == 0:
                continue
            w = weights[rows]
            R_fit = sp.diags(w) @ R_cut.tocsr()[rows]
            d_fit = data_vector[rows] * w
            x0 = A_guess_cube[:, yy, xx] if A_guess_cube is not None else None
            x, istop, itn, *_ = lsmr(R_fit, d_fit, x0=x0, atol=tolerance,
                                     btol=tolerance, maxiter=maxiter)
            flux_cube[:, yy, xx] = x
            istop_arr[yy, xx] = istop
            n_iters[yy, xx] = itn
            dof = len(rows) - nlam
            n_dof[yy, xx] = dof
            if dof > 0:
                resid = R_fit @ x - d_fit
                chi2_per_dof[yy, xx] = np.sum(resid ** 2) / dof
    diag = dict(chi2_per_dof=chi2_per_dof, n_dof=n_dof, istop=istop_arr,
                n_iters=n_iters, failed=failed)
    return flux_cube.ravel(), diag


# --------------------------------------------------------------------------
# Synthetic problem
# --------------------------------------------------------------------------

NLAM, NY, NX = 12, 4, 5
SHAPE = (NLAM, NY, NX)
OFF_DETECTOR = (0, NX - 1)   # lenslet whose last N_OFF wavelengths have no pixels
N_OFF = 4


def make_rectmat(seed=0):
    """Banded spectra: lenslet (y, x), wavelength j covers a 3 x 4 pixel patch.

    Neighbouring lenslets share one pixel column (crosstalk); one lenslet has
    wavelengths that fall off the detector; one explicit zero is stored.
    """
    rng = np.random.default_rng(seed)
    W = NX * 3 + 1
    H = NY * (NLAM + 4)
    prof_t = np.array([0.25, 0.5, 0.25])
    prof_u = np.array([0.1, 0.4, 0.4, 0.1])
    r, c, v = [], [], []
    for y in range(NY):
        for x in range(NX):
            for j in range(NLAM):
                if (y, x) == OFF_DETECTOR and j >= NLAM - N_OFF:
                    continue
                col = x + y * NX + j * NX * NY
                for t in range(3):
                    for u in range(4):
                        r.append((y * (NLAM + 4) + j + t) * W + x * 3 + u)
                        c.append(col)
                        v.append(prof_t[t] * prof_u[u] * (1 + 0.1 * rng.normal()))
    R = sp.csr_matrix((v, (r, c)), shape=(H * W, NLAM * NY * NX))
    R = R.tocsc()
    R.data[5] = 0.0          # explicit stored zero: still counts as "touched"
    return R


def make_frame(R, seed=1):
    rng = np.random.default_rng(seed)
    x_true = rng.uniform(10, 100, R.shape[1])
    d0 = R @ x_true
    var = np.maximum(d0, 1) + 4.0
    data = d0 + rng.normal(0, np.sqrt(var))
    guess = x_true * (1 + 0.05 * rng.normal(size=x_true.size))
    return data, var, guess


@pytest.fixture(scope="module")
def problem():
    R = make_rectmat()
    data, var, guess = make_frame(R)
    return R, data, var, guess


def assert_same(f1, d1, f2, d2, keys=("chi2_per_dof", "n_dof", "istop", "n_iters", "failed")):
    assert np.array_equal(f1, f2)
    for k in keys:
        assert np.array_equal(d1[k], d2[k], equal_nan=True), k


# --------------------------------------------------------------------------
# Matching the original loop
# --------------------------------------------------------------------------

@pytest.mark.parametrize("use_guess", [True, False])
def test_lsmr_matches_original_loop(problem, use_guess):
    R, data, var, guess = problem
    g = guess if use_guess else None
    f_ref, d_ref = reference_solve(R, data, var, g, SHAPE)
    f, d = ls.solve_per_lenslet(R, data, var, g, SHAPE)
    assert_same(f, d, f_ref, d_ref)


def test_lsmr_matches_original_loop_with_bad_pixels(problem):
    R, data, var, guess = problem
    rng = np.random.default_rng(3)
    bpm = rng.random(R.shape[0]) < 0.1
    var = var.copy()
    var[rng.random(R.shape[0]) < 0.02] = np.nan      # also non-finite variance
    f_ref, d_ref = reference_solve(R, data, var, guess, SHAPE, bad_pixel_mask=bpm)
    f, d = ls.solve_per_lenslet(R, data, var, guess, SHAPE, bad_pixel_mask=bpm)
    assert_same(f, d, f_ref, d_ref)


@pytest.mark.skipif(not os.path.exists(REAL_RECTMAT), reason="test rectmat not present")
def test_lsmr_matches_original_loop_real_rectmat():
    R = sp.load_npz(REAL_RECTMAT)
    shape = (54, 27, 27)
    data, var, guess = make_frame(R)
    f_ref, d_ref = reference_solve(R, data, var, guess, shape)
    f, d = ls.solve_per_lenslet(R, data, var, guess, shape, n_workers=2)
    assert_same(f, d, f_ref, d_ref)


# --------------------------------------------------------------------------
# Chunking and parallelism don't change results
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["lsmr", "dense"])
def test_chunking_does_not_change_result(problem, method):
    R, data, var, guess = problem
    f0, d0 = ls.solve_per_lenslet(R, data, var, guess, SHAPE, method=method)
    for cs in (1, 3, 7, NY * NX):
        f, d = ls.solve_per_lenslet(R, data, var, guess, SHAPE, method=method,
                                    chunk_size=cs)
        assert_same(f, d, f0, d0, keys=("chi2_per_dof", "n_dof", "istop", "n_iters",
                                         "rank", "cond", "failed"))


def test_parallel_matches_serial_lsmr(problem):
    """lsmr is bit-identical whatever the worker count."""
    R, data, var, guess = problem
    f1, d1 = ls.solve_per_lenslet(R, data, var, guess, SHAPE)
    f2, d2 = ls.solve_per_lenslet(R, data, var, guess, SHAPE, n_workers=2, chunk_size=3)
    assert_same(f2, d2, f1, d1)
    assert d2["n_workers"] == 2


def test_parallel_matches_serial_dense(problem):
    """Dense agrees to round-off: workers run 1 BLAS thread, the parent may run
    several, and LAPACK's last bits depend on the thread count."""
    R, data, var, guess = problem
    f1, d1 = ls.solve_per_lenslet(R, data, var, guess, SHAPE, method="dense")
    f2, d2 = ls.solve_per_lenslet(R, data, var, guess, SHAPE, method="dense",
                                  n_workers=2, chunk_size=3)
    assert np.allclose(f2, f1, rtol=1e-12, atol=1e-12 * np.abs(f1).max())
    assert np.allclose(d2["chi2_per_dof"], d1["chi2_per_dof"], rtol=1e-10, equal_nan=True)
    for k in ("n_dof", "rank", "failed"):
        assert np.array_equal(d2[k], d1[k]), k


def test_parallel_from_disk_cache_matches_serial(problem, tmp_path):
    """Workers memory-map the blocks from the cache instead of receiving them."""
    R, data, var, guess = problem
    path = str(tmp_path / "rmat.npz")
    sp.save_npz(path, R)
    blocks = lbc.get_lenslet_blocks(R, SHAPE, rectmat_path=path, mode="disk",
                                    cache_dir=str(tmp_path / "cache"))
    assert blocks.cache_dir is not None
    f1, d1 = ls.solve_per_lenslet(R, data, var, guess, SHAPE)
    f2, d2 = ls.solve_per_lenslet(None, data, var, guess, SHAPE, n_workers=2,
                                  chunk_size=4, blocks=blocks)
    assert_same(f2, d2, f1, d1)


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

def test_dense_agrees_with_lsmr_in_noise_units(problem):
    """|dx| <= 1e-3 sigma per voxel, chi2/dof within 1e-8 relative.

    sigma is 1/sqrt(H_jj) as in calculate_error_flux_cube, which
    underestimates the true error, so this errs on the strict side.
    """
    R, data, var, guess = problem
    f_l, d_l = ls.solve_per_lenslet(R, data, var, guess, SHAPE, method="lsmr")
    f_d, d_d = ls.solve_per_lenslet(R, data, var, guess, SHAPE, method="dense")
    H = np.asarray(R.power(2).T @ (1.0 / var)).ravel()
    live = H > 0
    sigma = 1.0 / np.sqrt(H[live])
    assert np.max(np.abs(f_l - f_d)[live] / sigma) < 1e-3
    ok = np.isfinite(d_l["chi2_per_dof"])
    rel = np.abs(d_l["chi2_per_dof"][ok] - d_d["chi2_per_dof"][ok]) / d_l["chi2_per_dof"][ok]
    assert rel.max() < 1e-8
    assert np.array_equal(d_l["failed"], d_d["failed"])
    assert d_d["method"] == "dense" and d_l["method"] == "lsmr"


@pytest.mark.parametrize("method", ["lsmr", "dense"])
def test_unconstrained_wavelengths_stay_at_x0(problem, method):
    R, data, var, guess = problem
    f, d = ls.solve_per_lenslet(R, data, var, guess, SHAPE, method=method)
    y, x = OFF_DETECTOR
    cube, g = f.reshape(SHAPE), guess.reshape(SHAPE)
    assert np.array_equal(cube[NLAM - N_OFF:, y, x], g[NLAM - N_OFF:, y, x])
    if method == "dense":
        assert d["rank"][y, x] == NLAM - N_OFF
        assert d["rank"][1, 1] == NLAM


def test_dense_without_guess_zeroes_unconstrained(problem):
    R, data, var, _ = problem
    f, _ = ls.solve_per_lenslet(R, data, var, None, SHAPE, method="dense")
    y, x = OFF_DETECTOR
    assert np.all(f.reshape(SHAPE)[NLAM - N_OFF:, y, x] == 0)


@pytest.mark.parametrize("method", ["lsmr", "dense"])
def test_fully_masked_lenslet_gets_fill_value(problem, method):
    R, data, var, guess = problem
    blocks = ls.build_lenslet_blocks(R, SHAPE)
    i = 2 * NX + 2
    bpm = np.zeros(R.shape[0], dtype=bool)
    bpm[blocks.rows[blocks.row_ptr[i]:blocks.row_ptr[i + 1]]] = True
    f, d = ls.solve_per_lenslet(R, data, var, guess, SHAPE, method=method,
                                bad_pixel_mask=bpm, fill_value=-7.0)
    assert np.all(f.reshape(SHAPE)[:, 2, 2] == -7.0)
    assert d["n_dof"][2, 2] == 0 and np.isnan(d["chi2_per_dof"][2, 2])
    assert not d["failed"][2, 2]


def test_kernel_failure_is_recorded(problem, monkeypatch):
    R, data, var, guess = problem

    def boom(chunk, **kw):
        res = ls.ChunkResult.empty(chunk.start, chunk.stop, chunk.nlam)
        for k, *_ in ls._iter_lenslet_systems(chunk):
            res.status[k] = ls.STATUS_FAILED
            res.fail_msg.append((k, "RuntimeError: boom"))
        return res

    monkeypatch.setitem(ls.KERNELS, "boom", boom)
    f, d = ls.solve_per_lenslet(R, data, var, guess, SHAPE, method="boom", fill_value=3.0)
    assert d["failed"].all() and np.all(f == 3.0)
    assert d["fail_msg"][0, 0] == "RuntimeError: boom"


# --------------------------------------------------------------------------
# Block layout helpers
# --------------------------------------------------------------------------

def test_block_layouts_agree(problem):
    R, *_ = problem
    b = ls.build_lenslet_blocks(R, SHAPE)
    Rc = R.tocsc()
    for i in (0, 7, NX - 1):
        rows, A = b.block(i)
        cols = i + np.arange(NLAM) * NX * NY
        sub = Rc[:, cols]
        assert np.array_equal(rows, np.unique(sub.indices))
        assert np.array_equal(A.toarray(), sub.toarray()[rows])
    dense, n_rows = b.to_padded_dense(3, 9)
    bd = b.to_block_diagonal(3, 9).toarray()
    off = 0
    for k, i in enumerate(range(3, 9)):
        _, A = b.block(i)
        assert np.array_equal(dense[k, :n_rows[k]], A.toarray())
        assert not dense[k, n_rows[k]:].any()
        assert np.array_equal(bd[off:off + n_rows[k], k * NLAM:(k + 1) * NLAM], A.toarray())
        off += n_rows[k]


def test_wrong_shape_rejected(problem):
    R, data, var, guess = problem
    with pytest.raises(ValueError):
        ls.build_lenslet_blocks(R, (NLAM + 1, NY, NX))
    b = ls.build_lenslet_blocks(R, SHAPE)
    with pytest.raises(ValueError):
        ls.solve_per_lenslet(None, data, var, guess, (NLAM, NX, NY), blocks=b)


# --------------------------------------------------------------------------
# Settings parsing
# --------------------------------------------------------------------------

def test_resolve_n_workers():
    auto = max(1, (os.cpu_count() or 1) - 1)
    assert ls.resolve_n_workers("auto") == auto
    assert ls.resolve_n_workers(None) == auto
    assert ls.resolve_n_workers(" 'Auto' ") == auto
    assert ls.resolve_n_workers(3) == 3
    assert ls.resolve_n_workers("2") == 2
    for bad in (0, -1, "many"):
        with pytest.raises(ValueError):
            ls.resolve_n_workers(bad)


def test_get_kernel():
    assert ls.get_kernel("LSMR") is ls.solve_chunk_lsmr
    assert ls.get_kernel(" dense ") is ls.solve_chunk_dense
    with pytest.raises(ValueError):
        ls.get_kernel("cholesky")


def test_parse_cache_mode():
    assert lbc.parse_cache_mode(None) == "disk"
    assert lbc.parse_cache_mode(False) == "off"
    assert lbc.parse_cache_mode("'Memory'") == "memory"
    with pytest.raises(ValueError):
        lbc.parse_cache_mode("ssd")


# --------------------------------------------------------------------------
# Block cache
# --------------------------------------------------------------------------

@pytest.fixture
def rectmat_file(problem, tmp_path, monkeypatch):
    monkeypatch.setattr(lbc, "_mem_cache", {})
    monkeypatch.setattr(lbc, "_mmap_cache", {})
    R = problem[0]
    path = str(tmp_path / "rmat.npz")
    sp.save_npz(path, R)
    return R, path, str(tmp_path / "cache")


def _assert_blocks_equal(a, b):
    assert tuple(a.flux_shape) == tuple(b.flux_shape)
    for f in ls.LensletBlocks.ARRAY_FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        assert x.dtype == y.dtype, f
        assert np.array_equal(x, y), f


def test_disk_cache_round_trip(rectmat_file):
    R, path, cache_dir = rectmat_file
    built = ls.build_lenslet_blocks(R, SHAPE)
    b1 = lbc.get_lenslet_blocks(R, SHAPE, rectmat_path=path, mode="disk", cache_dir=cache_dir)
    _assert_blocks_equal(b1, built)
    assert isinstance(b1.vals, np.memmap)
    man = json.load(open(os.path.join(b1.cache_dir, "manifest.json")))
    assert man["rectmat_md5"] == lbc.file_md5(path)
    assert man["flux_shape"] == list(SHAPE)
    # same process: served from memory; R not needed
    assert lbc.get_lenslet_blocks(None, SHAPE, rectmat_path=path, mode="disk",
                                  cache_dir=cache_dir) is b1
    # new process (memory cleared): loaded from disk without rebuilding
    lbc._mem_cache.clear()
    b2 = lbc.get_lenslet_blocks(None, SHAPE, rectmat_path=path, mode="disk",
                                cache_dir=cache_dir)
    _assert_blocks_equal(b2, built)


def test_disk_cache_rebuilds_corrupt_cache(rectmat_file):
    R, path, cache_dir = rectmat_file
    b1 = lbc.get_lenslet_blocks(R, SHAPE, rectmat_path=path, mode="disk", cache_dir=cache_dir)
    cdir = b1.cache_dir
    np.save(os.path.join(cdir, "vals.npy"), np.zeros(3))      # truncated array
    lbc._mem_cache.clear()
    lbc._mmap_cache.clear()
    b2 = lbc.get_lenslet_blocks(R, SHAPE, rectmat_path=path, mode="disk", cache_dir=cache_dir)
    assert b2.cache_dir == cdir
    _assert_blocks_equal(b2, ls.build_lenslet_blocks(R, SHAPE))
    assert not [p for p in os.listdir(os.path.dirname(cdir)) if ".tmp-" in p or ".stale-" in p]


def test_disk_cache_key_follows_rectmat_content(rectmat_file):
    R, path, cache_dir = rectmat_file
    b1 = lbc.get_lenslet_blocks(R, SHAPE, rectmat_path=path, mode="disk", cache_dir=cache_dir)
    R2 = R.copy()
    R2.data[0] *= 2
    os.remove(path)
    sp.save_npz(path, R2)
    os.utime(path, ns=(1, 1))   # make sure the md5 memo sees a change
    b2 = lbc.get_lenslet_blocks(R2, SHAPE, rectmat_path=path, mode="disk", cache_dir=cache_dir)
    assert b2.cache_dir != b1.cache_dir
    assert b2.vals[0] != b1.vals[0] or not np.array_equal(b2.vals, b1.vals)
    assert len(os.listdir(os.path.join(cache_dir, "lenslet_blocks"))) == 2


def test_memory_and_off_modes_write_nothing(rectmat_file):
    R, path, cache_dir = rectmat_file
    m1 = lbc.get_lenslet_blocks(R, SHAPE, rectmat_path=path, mode="memory", cache_dir=cache_dir)
    assert lbc.get_lenslet_blocks(R, SHAPE, rectmat_path=path, mode="memory") is m1
    assert m1.cache_dir is None
    o1 = lbc.get_lenslet_blocks(R, SHAPE, rectmat_path=path, mode="off", cache_dir=cache_dir)
    o2 = lbc.get_lenslet_blocks(R, SHAPE, rectmat_path=path, mode="off", cache_dir=cache_dir)
    assert o1 is not o2
    assert not os.path.exists(cache_dir)


def test_disk_mode_needs_cache_dir(rectmat_file):
    R, path, _ = rectmat_file
    with pytest.raises(ValueError):
        lbc.get_lenslet_blocks(R, SHAPE, rectmat_path=path, mode="disk", cache_dir=None)


# --------------------------------------------------------------------------
# Config settings
# --------------------------------------------------------------------------

def test_chi2_solver_settings():
    chi2_solver_settings = ls.chi2_solver_settings
    cfg = types.SimpleNamespace(chi2_method_lowres="dense", chi2_method_medres="lsmr",
                                chi2_n_workers=3, chi2_block_cache="memory",
                                chi2_block_cache_dir=None, output_directory="out/")
    s = chi2_solver_settings(cfg, "LowRes-K")
    assert s == dict(method="dense", n_workers=3, cache_mode="memory",
                     cache_dir=os.path.join("out/", "cache"))
    assert chi2_solver_settings(cfg, "MedRes-L")["method"] == "lsmr"
    # all options missing -> defaults
    s = chi2_solver_settings(types.SimpleNamespace(), "LowRes-K")
    assert s["method"] == "lsmr" and s["cache_mode"] == "disk"
    assert s["n_workers"] == ls.resolve_n_workers("auto")
    with pytest.raises(ValueError):
        chi2_solver_settings(types.SimpleNamespace(chi2_method_lowres="qr"), "LowRes-K")
