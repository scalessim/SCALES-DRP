"""Per-lenslet weighted least squares, split into steps that can be reused.

This module holds the work behind
``scales_basic.solve_weighted_lsq_per_lenslet``. It imports only numpy and
scipy, so worker processes start quickly and a GPU backend can reuse the same
pieces.

The solve is split into four steps:

1. **Gather** (`build_lenslet_blocks`): reorder the rectification matrix R so
   each lenslet's ``nlam`` columns, and the detector rows they touch, are
   stored next to each other. Depends only on R, so the result can be cached
   per rectmat (see ``lenslet_block_cache``).
2. **Weight** (`prepare_problems`): attach one frame's data, weights,
   good-pixel mask and warm start. Vectorized; no Python loop per lenslet.
3. **Solve** (`KERNELS`): a kernel solves a contiguous range of lenslets (a
   `LensletChunk`) and returns a `ChunkResult`. All kernels share one
   signature, so a GPU kernel is one more entry in `KERNELS`.
4. **Assemble** (`assemble_results`): scatter chunk results back into the
   flux cube and the per-lenslet diagnostic maps.

`run_chunks` runs a kernel over chunks, serially or in a process pool.

Lenslet index ``i = y * nx + x``. Column ``c`` of R belongs to lenslet
``c % (nx * ny)`` at wavelength ``c // (nx * ny)``, matching
``flux_shape_3d = (nlam, ny, nx)`` in C order.
"""

import atexit
import contextlib
import dataclasses
import functools
import logging
import math
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Optional

import numpy as np
import scipy.sparse as sp

logger_default = logging.getLogger(__name__)

# Environment variables that cap BLAS/OpenMP threads. Worker processes get 1
# thread each, so N workers don't each start a full set of BLAS threads.
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

# Lenslet status codes in ChunkResult.status
STATUS_EMPTY = 0    # no usable detector pixels; flux left at fill_value
STATUS_SOLVED = 1
STATUS_FAILED = 2   # the kernel raised; see fail_msg


# --------------------------------------------------------------------------
# Step 1: gather
# --------------------------------------------------------------------------

@dataclasses.dataclass
class LensletBlocks:
    """R's nonzeros regrouped by lenslet (frame-independent).

    Arrays (n = number of lenslets, ny * nx):
        rows      (n_rows,)  int32   detector rows each lenslet touches,
                                     sorted, concatenated over lenslets
        row_ptr   (n+1,)     int64   lenslet i's rows are rows[row_ptr[i]:row_ptr[i+1]]
        vals      (nnz,)     float64 R values, grouped by lenslet, then row
        local_row (nnz,)     int32   row position within the lenslet's rows
        local_col (nnz,)     uint16  wavelength index 0..nlam-1
        nz_ptr    (n+1,)     int64   lenslet i's nonzeros are [nz_ptr[i]:nz_ptr[i+1]]

    ``cache_dir`` is set when the arrays are memory-mapped from an on-disk
    cache; worker processes then map the same files instead of receiving
    copies.
    """
    flux_shape: tuple
    rows: np.ndarray
    row_ptr: np.ndarray
    vals: np.ndarray
    local_row: np.ndarray
    local_col: np.ndarray
    nz_ptr: np.ndarray
    cache_dir: Optional[str] = None

    ARRAY_FIELDS = ("rows", "row_ptr", "vals", "local_row", "local_col", "nz_ptr")

    @property
    def nlam(self):
        return int(self.flux_shape[0])

    @property
    def n_lenslets(self):
        return int(self.flux_shape[1]) * int(self.flux_shape[2])

    @property
    def nbytes(self):
        return sum(getattr(self, f).nbytes for f in self.ARRAY_FIELDS)

    def block(self, i):
        """Return (detector_rows, sparse CSR block) for lenslet ``i`` (unweighted)."""
        r0, r1 = self.row_ptr[i], self.row_ptr[i + 1]
        z0, z1 = self.nz_ptr[i], self.nz_ptr[i + 1]
        A = sp.csr_matrix(
            (self.vals[z0:z1], (self.local_row[z0:z1], self.local_col[z0:z1])),
            shape=(int(r1 - r0), self.nlam))
        return np.asarray(self.rows[r0:r1]), A

    def to_padded_dense(self, start, stop):
        """Dense, zero-padded batch of lenslets ``start:stop`` (unweighted).

        For batched dense solvers (e.g. a GPU batched QR). Padding rows are
        all zero, so they don't change a least-squares solution.

        Returns:
            A (n, max_rows, nlam) float64, n_rows (n,) int64
        """
        n_rows = np.diff(self.row_ptr[start:stop + 1])
        max_rows = int(n_rows.max()) if len(n_rows) else 0
        out = np.zeros((stop - start, max_rows, self.nlam))
        for k, i in enumerate(range(start, stop)):
            z0, z1 = self.nz_ptr[i], self.nz_ptr[i + 1]
            np.add.at(out[k], (self.local_row[z0:z1], self.local_col[z0:z1]),
                      self.vals[z0:z1])
        return out, n_rows

    def to_block_diagonal(self, start, stop):
        """Lenslets ``start:stop`` as one block-diagonal sparse matrix (unweighted).

        Block k occupies rows ``row_ptr[start+k]-row_ptr[start]`` onward and
        columns ``k*nlam .. (k+1)*nlam``. For a single sparse solve over many
        lenslets.
        """
        z0, z1 = self.nz_ptr[start], self.nz_ptr[stop]
        lens = np.repeat(np.arange(stop - start), np.diff(self.nz_ptr[start:stop + 1]))
        row_off = self.row_ptr[start:stop] - self.row_ptr[start]
        r = row_off[lens] + self.local_row[z0:z1]
        c = lens * self.nlam + self.local_col[z0:z1]
        shape = (int(self.row_ptr[stop] - self.row_ptr[start]), (stop - start) * self.nlam)
        return sp.csr_matrix((self.vals[z0:z1], (r, c)), shape=shape)


def build_lenslet_blocks(R_matrix, flux_shape_3d):
    """Regroup R's nonzeros by lenslet (step 1).

    Explicitly stored zeros in R are kept, so the set of rows a lenslet
    "touches" is the same as in the original per-lenslet loop.
    """
    nlam, ny, nx = (int(n) for n in flux_shape_3d)
    n_lens = ny * nx
    n_pix, n_flux = R_matrix.shape
    if n_flux != nlam * n_lens:
        raise ValueError(f"R_matrix has {n_flux} columns but flux_shape_3d "
                         f"{tuple(flux_shape_3d)} implies {nlam * n_lens}.")
    if nlam > np.iinfo(np.uint16).max:
        raise ValueError(f"nlam = {nlam} does not fit the uint16 column index.")

    coo = R_matrix.tocsc().tocoo()   # column-major order: lam ascends within a row below
    col = coo.col.astype(np.int64)
    lens = col % n_lens
    lam = (col // n_lens).astype(np.uint16)
    row = coo.row.astype(np.int64)

    # Group by (lenslet, row); a stable sort keeps wavelengths ascending within a row.
    key = lens * n_pix + row
    order = np.argsort(key, kind="stable")
    key, lens, row, lam = key[order], lens[order], row[order], lam[order]
    vals = np.asarray(coo.data, dtype=np.float64)[order]
    del coo, col, order

    nz_ptr = np.zeros(n_lens + 1, dtype=np.int64)
    np.cumsum(np.bincount(lens, minlength=n_lens), out=nz_ptr[1:])

    first = np.ones(len(key), dtype=bool)        # first nonzero of each (lenslet, row)
    first[1:] = key[1:] != key[:-1]
    rows = row[first].astype(np.int32)
    row_ptr = np.zeros(n_lens + 1, dtype=np.int64)
    np.cumsum(np.bincount(lens[first], minlength=n_lens), out=row_ptr[1:])
    local_row = (np.cumsum(first) - 1 - row_ptr[lens]).astype(np.int32)

    return LensletBlocks(flux_shape=(nlam, ny, nx), rows=rows, row_ptr=row_ptr,
                         vals=vals, local_row=local_row, local_col=lam, nz_ptr=nz_ptr)


# --------------------------------------------------------------------------
# Step 2: weight
# --------------------------------------------------------------------------

@dataclasses.dataclass
class LensletProblems:
    """One frame's inputs, laid out to match ``blocks.rows`` (step 2 output).

    d_rows, w_rows, good_rows are the data, weight (1/sigma) and good-pixel
    flag of each entry of ``blocks.rows``. x0 is (n_lenslets, nlam) or None.
    """
    blocks: LensletBlocks
    d_rows: np.ndarray
    w_rows: np.ndarray
    good_rows: np.ndarray
    x0: Optional[np.ndarray]


def pixel_weights(data_vector, variance_vector, bad_pixel_mask=None):
    """Per-pixel data, weight (1/sigma) and good mask, as the solver uses them.

    Pixels with non-finite data or variance, variance <= 0, or flagged in
    bad_pixel_mask are not used.
    """
    data = np.asarray(data_vector, dtype=np.float64).ravel()
    variance = np.asarray(variance_vector, dtype=np.float64).ravel()
    good = np.isfinite(data) & np.isfinite(variance) & (variance > 0)
    weights = np.zeros_like(variance)
    weights[good] = 1.0 / np.sqrt(variance[good])
    if bad_pixel_mask is not None:
        good &= ~np.asarray(bad_pixel_mask, dtype=bool).ravel()
    return data, weights, good


def prepare_problems(blocks, data_vector, variance_vector, A_guess=None,
                     bad_pixel_mask=None):
    """Attach one frame to the blocks (step 2)."""
    data, weights, good = pixel_weights(data_vector, variance_vector, bad_pixel_mask)
    rows = blocks.rows
    x0 = None
    if A_guess is not None:
        nlam = blocks.nlam
        x0 = np.nan_to_num(np.asarray(A_guess, dtype=np.float64)).reshape(
            nlam, blocks.n_lenslets).T.copy()
    return LensletProblems(blocks=blocks, d_rows=data[rows], w_rows=weights[rows],
                           good_rows=good[rows], x0=x0)


@dataclasses.dataclass
class LensletChunk:
    """Lenslets ``start:stop`` with everything a kernel needs.

    Pointer arrays are rebased to start at 0. If the block arrays (vals,
    local_row, local_col) are None, ``cache_dir`` names an on-disk block cache
    to memory-map them from (see `LensletChunk.materialize`).
    """
    start: int
    stop: int
    nlam: int
    row_ptr: np.ndarray
    nz_ptr: np.ndarray
    d: np.ndarray
    w: np.ndarray
    good: np.ndarray
    x0: Optional[np.ndarray]
    vals: Optional[np.ndarray] = None
    local_row: Optional[np.ndarray] = None
    local_col: Optional[np.ndarray] = None
    cache_dir: Optional[str] = None
    nz_offset: int = 0   # global index of this chunk's first nonzero (for cache reads)

    def materialize(self):
        """Fill in the block arrays from the on-disk cache, if not present."""
        if self.vals is None:
            from scalesdrp.core.lenslet_block_cache import open_cached_arrays
            arrs = open_cached_arrays(self.cache_dir)
            z0 = self.nz_offset
            z1 = z0 + int(self.nz_ptr[-1])
            self.vals = arrs["vals"][z0:z1]
            self.local_row = arrs["local_row"][z0:z1]
            self.local_col = arrs["local_col"][z0:z1]
        return self


def make_chunks(problems, chunk_size, by_reference=False):
    """Split problems into contiguous LensletChunks.

    by_reference: leave the block arrays out and point at the blocks' on-disk
        cache instead, so sending a chunk to a worker copies only the frame
        data. Needs ``problems.blocks.cache_dir``.
    """
    b = problems.blocks
    n = b.n_lenslets
    chunk_size = max(1, int(chunk_size))
    if by_reference and b.cache_dir is None:
        raise ValueError("by_reference chunks need blocks loaded from a disk cache.")
    chunks = []
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        r0, r1 = int(b.row_ptr[s]), int(b.row_ptr[e])
        z0, z1 = int(b.nz_ptr[s]), int(b.nz_ptr[e])
        c = LensletChunk(
            start=s, stop=e, nlam=b.nlam,
            row_ptr=np.asarray(b.row_ptr[s:e + 1]) - r0,
            nz_ptr=np.asarray(b.nz_ptr[s:e + 1]) - z0,
            d=problems.d_rows[r0:r1], w=problems.w_rows[r0:r1],
            good=problems.good_rows[r0:r1],
            x0=None if problems.x0 is None else problems.x0[s:e],
            nz_offset=z0)
        if by_reference:
            c.cache_dir = b.cache_dir
        else:
            c.vals = b.vals[z0:z1]
            c.local_row = b.local_row[z0:z1]
            c.local_col = b.local_col[z0:z1]
        chunks.append(c)
    return chunks


def _iter_lenslet_systems(chunk):
    """Yield (k, A_weighted_csr, b_weighted, x0_or_None) for each usable lenslet.

    k is the index within the chunk. Lenslets with no usable pixels are skipped
    (their status stays STATUS_EMPTY). The weighted system matches the
    original loop: rows restricted to good pixels (sorted), each row of R
    scaled by its 1/sigma.
    """
    chunk.materialize()
    nlam = chunk.nlam
    for k in range(chunk.stop - chunk.start):
        r0, r1 = chunk.row_ptr[k], chunk.row_ptr[k + 1]
        keep = chunk.good[r0:r1]
        n_keep = int(np.count_nonzero(keep))
        if n_keep == 0:
            continue
        z0, z1 = chunk.nz_ptr[k], chunk.nz_ptr[k + 1]
        lr = np.asarray(chunk.local_row[z0:z1])
        lc = np.asarray(chunk.local_col[z0:z1])
        v = np.asarray(chunk.vals[z0:z1])
        if n_keep < r1 - r0:
            m = keep[lr]
            lr = (np.cumsum(keep) - 1)[lr[m]]
            lc, v = lc[m], v[m]
        w = chunk.w[r0:r1][keep]
        # Weight with diags(w) @ A, exactly as the original loop did. The
        # product's (unsorted) entry order sets the summation order inside
        # lsmr's mat-vecs; scaling A.data in place gives the same values but
        # different round-off, and so slightly different iteration counts.
        A = sp.diags(w) @ sp.csr_matrix((v, (lr, lc)), shape=(n_keep, nlam))
        b = chunk.d[r0:r1][keep] * w
        x0 = None if chunk.x0 is None else chunk.x0[k]
        yield k, A, b, x0


# --------------------------------------------------------------------------
# Step 3: solve kernels
# --------------------------------------------------------------------------

@dataclasses.dataclass
class ChunkResult:
    """Per-lenslet results for lenslets ``start:stop`` (arrays of length n).

    istop/n_iters are set by iterative kernels, rank/cond by direct ones;
    the other kernel leaves them at 0 / NaN.
    """
    start: int
    stop: int
    x: np.ndarray            # (n, nlam)
    status: np.ndarray       # STATUS_* codes
    chi2_per_dof: np.ndarray
    n_dof: np.ndarray
    istop: np.ndarray
    n_iters: np.ndarray
    rank: np.ndarray
    cond: np.ndarray
    fail_msg: list           # (k, message) pairs for failed lenslets

    @classmethod
    def empty(cls, start, stop, nlam):
        n = stop - start
        return cls(start=start, stop=stop, x=np.zeros((n, nlam)),
                   status=np.full(n, STATUS_EMPTY, dtype=np.int8),
                   chi2_per_dof=np.full(n, np.nan), n_dof=np.zeros(n, dtype=int),
                   istop=np.zeros(n, dtype=int), n_iters=np.zeros(n, dtype=int),
                   rank=np.zeros(n, dtype=int), cond=np.full(n, np.nan), fail_msg=[])


def _record_fit(res, k, A, b, x):
    res.x[k] = x
    res.status[k] = STATUS_SOLVED
    dof = A.shape[0] - A.shape[1]
    res.n_dof[k] = dof
    if dof > 0:
        resid = A @ x - b
        res.chi2_per_dof[k] = np.sum(resid ** 2) / dof


def solve_chunk_lsmr(chunk, tolerance=1e-10, maxiter=100000):
    """Iterative kernel: scipy lsmr per lenslet, warm-started from x0.

    Wavelengths with no detector pixels stay at x0 (lsmr never moves them).
    """
    from scipy.sparse.linalg import lsmr
    res = ChunkResult.empty(chunk.start, chunk.stop, chunk.nlam)
    for k, A, b, x0 in _iter_lenslet_systems(chunk):
        try:
            x, istop, itn, *_ = lsmr(A, b, x0=x0, atol=tolerance, btol=tolerance,
                                     maxiter=maxiter)
            res.istop[k] = istop
            res.n_iters[k] = itn
            _record_fit(res, k, A, b, x)
        except Exception as e:
            res.status[k] = STATUS_FAILED
            res.fail_msg.append((k, f"{type(e).__name__}: {e}"))
    return res


def solve_chunk_dense(chunk, rcond=None, **_ignored):
    """Direct kernel: dense SVD least squares (numpy lstsq) per lenslet.

    Solves for the correction from x0, ``x = x0 + lstsq(A, b - A x0)``, so the
    answer is the exact least-squares fit and, where the data don't constrain
    a direction (e.g. wavelengths with no pixels), x stays at x0, as with
    lsmr. Records rank and condition number. ``tolerance``/``maxiter`` are
    accepted and ignored so kernels can be swapped freely.
    """
    res = ChunkResult.empty(chunk.start, chunk.stop, chunk.nlam)
    for k, A, b, x0 in _iter_lenslet_systems(chunk):
        try:
            Ad = A.toarray()
            x0k = np.zeros(chunk.nlam) if x0 is None else x0
            dx, _, rank, sv = np.linalg.lstsq(Ad, b - Ad @ x0k, rcond=rcond)
            res.rank[k] = rank
            res.cond[k] = sv[0] / sv[-1] if len(sv) and sv[-1] > 0 else np.inf
            _record_fit(res, k, A, b, x0k + dx)
        except Exception as e:
            res.status[k] = STATUS_FAILED
            res.fail_msg.append((k, f"{type(e).__name__}: {e}"))
    return res


# Kernel registry: method name -> kernel(chunk, **kwargs) -> ChunkResult.
# A GPU kernel is added here with the same signature.
KERNELS = {
    "lsmr": solve_chunk_lsmr,
    "dense": solve_chunk_dense,
}


def get_kernel(method):
    try:
        return KERNELS[str(method).strip().lower()]
    except KeyError:
        raise ValueError(f"Unknown per-lenslet solver method {method!r}; "
                         f"expected one of {sorted(KERNELS)}.") from None


# --------------------------------------------------------------------------
# Running kernels
# --------------------------------------------------------------------------

def resolve_n_workers(n_workers):
    """Normalise a worker-count setting to an int >= 1.

    'auto' (or None) means ``os.cpu_count() - 1``, at least 1.
    """
    if n_workers is None or (isinstance(n_workers, str)
                             and n_workers.strip().strip("'\"").lower() == "auto"):
        return max(1, (os.cpu_count() or 1) - 1)
    try:
        n = int(n_workers)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid worker count {n_workers!r}; "
                         f"expected 'auto' or a positive integer.") from None
    if n < 1:
        raise ValueError(f"Worker count must be >= 1, got {n}.")
    return n


def default_chunk_size(n_lenslets, n_workers, chunks_per_worker=4):
    """A few chunks per worker, so uneven lenslets still balance across workers."""
    if n_workers <= 1:
        return max(1, n_lenslets)
    return max(1, math.ceil(n_lenslets / (n_workers * chunks_per_worker)))


def _run_kernel(method, kwargs, chunk):
    return get_kernel(method)(chunk, **kwargs)


@contextlib.contextmanager
def _single_threaded_children():
    """Make processes spawned inside this block start with 1 BLAS thread.

    The env vars are read when a child imports numpy, so they must be set
    when it is spawned; the parent's already-loaded BLAS is unaffected.
    """
    saved = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}
    os.environ.update({k: "1" for k in _THREAD_ENV_VARS})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_pool = None            # (n_workers, ProcessPoolExecutor), reused across frames
_pool_lock = threading.Lock()


def get_worker_pool(n_workers):
    """The process pool for ``n_workers``, created on first use and kept.

    Each 'spawn' worker re-imports the parent's __main__ (for the pipeline,
    the whole framework, ~seconds), so the pool is kept for the life of the
    process instead of being started per frame. Asking for a different size
    replaces it. ('fork' is unsafe here: the framework runs primitives in a
    thread.)
    """
    global _pool
    with _pool_lock:
        if _pool is not None and _pool[0] == n_workers:
            return _pool[1]
        _shutdown_pool_locked()
        ex = ProcessPoolExecutor(max_workers=n_workers,
                                 mp_context=multiprocessing.get_context("spawn"))
        _pool = (n_workers, ex)
        return ex


def _shutdown_pool_locked():
    global _pool
    if _pool is not None:
        _pool[1].shutdown(wait=True, cancel_futures=True)
        _pool = None


def shutdown_worker_pool():
    """Stop the worker processes (also done automatically at exit)."""
    with _pool_lock:
        _shutdown_pool_locked()


atexit.register(shutdown_worker_pool)


def run_chunks(method, chunks, n_workers=1, **kernel_kwargs):
    """Run kernel ``method`` over chunks; results come back in chunk order.

    n_workers == 1 runs in this process; otherwise the shared process pool
    from `get_worker_pool` is used.
    """
    n_workers = resolve_n_workers(n_workers)
    get_kernel(method)  # fail fast on a bad name
    if n_workers == 1 or len(chunks) <= 1:
        return [_run_kernel(method, kernel_kwargs, c) for c in chunks]
    ex = get_worker_pool(n_workers)
    task = functools.partial(_run_kernel, method, kernel_kwargs)
    try:
        # Workers are spawned during submit(), so only submission needs the
        # single-thread environment.
        with _single_threaded_children():
            futures = [ex.submit(task, c) for c in chunks]
        return [f.result() for f in futures]
    except BrokenProcessPool:
        shutdown_worker_pool()   # a worker died; start fresh next time
        raise


# --------------------------------------------------------------------------
# Step 4: assemble
# --------------------------------------------------------------------------

def assemble_results(results, flux_shape_3d, fill_value=0.0, method=None):
    """Scatter chunk results into the flux vector and diagnostic maps.

    Returns:
        flux_vector (N_fluxes,), diagnostics dict of (ny, nx) arrays with keys
        chi2_per_dof, n_dof, istop, n_iters, rank, cond, failed, fail_msg,
        plus 'method'.
    """
    nlam, ny, nx = (int(n) for n in flux_shape_3d)
    n = ny * nx
    x = np.full((n, nlam), fill_value, dtype=np.float64)
    maps = dict(chi2_per_dof=np.full(n, np.nan), n_dof=np.zeros(n, dtype=int),
                istop=np.zeros(n, dtype=int), n_iters=np.zeros(n, dtype=int),
                rank=np.zeros(n, dtype=int), cond=np.full(n, np.nan))
    failed = np.zeros(n, dtype=bool)
    fail_msg = np.empty(n, dtype=object)
    for r in results:
        s, e = r.start, r.stop
        solved = r.status == STATUS_SOLVED
        x[s:e][solved] = r.x[solved]
        for key in maps:
            maps[key][s:e] = getattr(r, key)
        failed[s:e] = r.status == STATUS_FAILED
        for k, msg in r.fail_msg:
            fail_msg[s + k] = msg
    flux_vector = x.T.reshape(-1)   # (n, nlam) -> (nlam, ny, nx) C order
    diag = {k: v.reshape(ny, nx) for k, v in maps.items()}
    diag["failed"] = failed.reshape(ny, nx)
    diag["fail_msg"] = fail_msg.reshape(ny, nx)
    diag["method"] = method
    return flux_vector, diag


# --------------------------------------------------------------------------
# Settings from scales.cfg
# --------------------------------------------------------------------------

def chi2_solver_settings(instrument_config, ifsmode):
    """Per-lenslet solver settings for a mode, from scales.cfg.

    Reads chi2_method_lowres / chi2_method_medres (default 'lsmr'),
    chi2_n_workers (default 'auto'), chi2_block_cache (default 'disk') and
    chi2_block_cache_dir (default <output_directory>/cache). Missing options
    fall back to the defaults.

    Returns:
        dict(method, n_workers, cache_mode, cache_dir)
    """
    from scalesdrp.core import lenslet_block_cache as lbc

    def opt(name, default):
        val = getattr(instrument_config, name, None)
        return default if val is None or val == '' else val

    if str(ifsmode).startswith('MedRes'):
        method = opt('chi2_method_medres', 'lsmr')
    else:
        method = opt('chi2_method_lowres', 'lsmr')
    get_kernel(method)
    cache_dir = opt('chi2_block_cache_dir', None)
    if cache_dir is None:
        cache_dir = os.path.join(opt('output_directory', 'redux/'), 'cache')
    return dict(method=str(method).strip().lower(),
                n_workers=resolve_n_workers(opt('chi2_n_workers', 'auto')),
                cache_mode=lbc.parse_cache_mode(opt('chi2_block_cache', 'disk')),
                cache_dir=cache_dir)


# --------------------------------------------------------------------------
# All four steps
# --------------------------------------------------------------------------

def solve_per_lenslet(R_matrix, data_vector, variance_vector, A_guess, flux_shape_3d,
                      bad_pixel_mask=None, fill_value=0.0, tolerance=1e-10,
                      maxiter=100000, method="lsmr", n_workers=1, chunk_size=None,
                      blocks=None):
    """Gather, weight, solve and assemble; see scales_basic.solve_weighted_lsq_per_lenslet.

    Returns:
        flux_vector (N_fluxes,), diagnostics dict (see assemble_results)
    """
    kernel = get_kernel(method)
    method = next(k for k, v in KERNELS.items() if v is kernel)
    n_workers = resolve_n_workers(n_workers)
    flux_shape_3d = tuple(int(n) for n in flux_shape_3d)
    if blocks is None:
        blocks = build_lenslet_blocks(R_matrix, flux_shape_3d)
    elif tuple(blocks.flux_shape) != flux_shape_3d:
        raise ValueError(f"blocks are for cube shape {tuple(blocks.flux_shape)}, "
                         f"not {flux_shape_3d}.")
    problems = prepare_problems(blocks, data_vector, variance_vector,
                                A_guess=A_guess, bad_pixel_mask=bad_pixel_mask)
    if chunk_size is None:
        chunk_size = default_chunk_size(blocks.n_lenslets, n_workers)
    # Parallel runs on a disk cache send only the frame data to workers;
    # they memory-map the blocks themselves.
    by_reference = n_workers > 1 and blocks.cache_dir is not None
    chunks = make_chunks(problems, chunk_size, by_reference=by_reference)
    results = run_chunks(method, chunks, n_workers=n_workers,
                         tolerance=tolerance, maxiter=maxiter)
    flux_vector, diag = assemble_results(results, flux_shape_3d,
                                         fill_value=fill_value, method=method)
    diag["n_workers"] = n_workers
    return flux_vector, diag
