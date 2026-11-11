"""Caching of per-lenslet blocks (`lenslet_solve.LensletBlocks`) across frames.

The blocks depend only on the rectification matrix, so they are built once
per rectmat and reused. Three modes:

* ``'off'``    - rebuild for every frame.
* ``'memory'`` - keep the blocks in RAM for the life of the process.
* ``'disk'``   - write them once as plain ``.npy`` files and memory-map them.
  The OS reads in only the pages in use and can drop them again, and worker
  processes map the same files instead of being sent copies.

On-disk layout::

    <cache_dir>/lenslet_blocks/<md5[:16]>_<nlam>x<ny>x<nx>_v<FORMAT_VERSION>/
        manifest.json   format version, rectmat name and full MD5, cube shape,
                        array dtypes and lengths, build time
        rows.npy row_ptr.npy vals.npy local_row.npy local_col.npy nz_ptr.npy

The key is the rectmat's content hash, so a rectmat regenerated under the same
name gets a new cache. A build is written to a temporary sibling directory and
renamed into place, so a crash or a concurrent run never leaves a half-written
cache that is later read. A cache whose manifest doesn't match (corrupt or
truncated) is replaced. Caches for other rectmats are never deleted
automatically; each build logs its path and size so stale ones are easy to
find.
"""

import datetime
import hashlib
import json
import logging
import os
import shutil
import tempfile

import numpy as np

from scalesdrp.core.lenslet_solve import LensletBlocks, build_lenslet_blocks

FORMAT_VERSION = 1
CACHE_MODES = ("off", "memory", "disk")
_SUBDIR = "lenslet_blocks"
_MANIFEST = "manifest.json"

_md5_memo = {}       # (abspath, size, mtime_ns) -> md5 hex
_mem_cache = {}      # (md5, flux_shape) -> LensletBlocks (in RAM or memory-mapped)
_mmap_cache = {}     # cache dir -> dict of memory-mapped arrays (worker side)


def parse_cache_mode(value):
    """Normalise a ``chi2_block_cache`` setting; None means 'disk'."""
    if value is None:
        return "disk"
    if value is False:
        return "off"
    mode = str(value).strip().strip("'\"").lower()
    if mode not in CACHE_MODES:
        raise ValueError(f"Invalid chi2_block_cache {value!r}; expected one of {CACHE_MODES}.")
    return mode


def file_md5(path, chunk_bytes=1 << 22):
    """MD5 of a file, remembered per (path, size, mtime) within this process."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _md5_memo:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk_bytes), b""):
                h.update(block)
        _md5_memo[key] = h.hexdigest()
    return _md5_memo[key]


def cache_path(cache_dir, md5, flux_shape):
    nlam, ny, nx = (int(n) for n in flux_shape)
    return os.path.join(cache_dir, _SUBDIR,
                        f"{md5[:16]}_{nlam}x{ny}x{nx}_v{FORMAT_VERSION}")


def _manifest_for(blocks, rectmat_path, md5):
    return {
        "format_version": FORMAT_VERSION,
        "rectmat": os.path.basename(rectmat_path),
        "rectmat_md5": md5,
        "flux_shape": [int(n) for n in blocks.flux_shape],
        "arrays": {f: {"dtype": getattr(blocks, f).dtype.str,
                       "length": int(getattr(blocks, f).shape[0])}
                   for f in LensletBlocks.ARRAY_FIELDS},
        "built": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def write_blocks(blocks, path, rectmat_path, md5):
    """Write blocks to ``path`` atomically (temp dir + rename)."""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=parent, prefix=os.path.basename(path) + ".tmp-")
    try:
        for f in LensletBlocks.ARRAY_FIELDS:
            np.save(os.path.join(tmp, f + ".npy"), np.ascontiguousarray(getattr(blocks, f)))
        with open(os.path.join(tmp, _MANIFEST), "w") as fh:   # written last
            json.dump(_manifest_for(blocks, rectmat_path, md5), fh, indent=2)
        try:
            os.rename(tmp, path)
        except OSError:
            if not os.path.isdir(path):   # not a lost race with another writer
                raise
            shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def open_cached_arrays(path):
    """Memory-map a cache directory's arrays (cached per process)."""
    if path not in _mmap_cache:
        _mmap_cache[path] = {f: np.load(os.path.join(path, f + ".npy"), mmap_mode="r")
                             for f in LensletBlocks.ARRAY_FIELDS}
    return _mmap_cache[path]


def load_blocks(path, md5, flux_shape):
    """Load a disk cache, or return None if it is missing or doesn't match."""
    try:
        with open(os.path.join(path, _MANIFEST)) as fh:
            man = json.load(fh)
        if (man.get("format_version") != FORMAT_VERSION
                or man.get("rectmat_md5") != md5
                or tuple(man.get("flux_shape", ())) != tuple(int(n) for n in flux_shape)):
            return None
        _mmap_cache.pop(path, None)
        arrs = open_cached_arrays(path)
        for f, spec in man["arrays"].items():
            a = arrs[f]
            if a.dtype.str != spec["dtype"] or a.shape[0] != spec["length"]:
                return None
        n_lens = int(flux_shape[1]) * int(flux_shape[2])
        if (len(arrs["row_ptr"]) != n_lens + 1 or len(arrs["nz_ptr"]) != n_lens + 1
                or arrs["row_ptr"][-1] != len(arrs["rows"])
                or arrs["nz_ptr"][-1] != len(arrs["vals"])):
            return None
    except (OSError, ValueError, KeyError, TypeError):
        _mmap_cache.pop(path, None)
        return None
    return LensletBlocks(flux_shape=tuple(int(n) for n in flux_shape),
                         cache_dir=path, **arrs)


def get_lenslet_blocks(R_matrix, flux_shape_3d, rectmat_path=None, mode="disk",
                       cache_dir=None, logger=None):
    """Return LensletBlocks for R, using the cache mode requested.

    Args:
        R_matrix: the rectmat (only read when the blocks have to be built).
        flux_shape_3d: (nlam, ny, nx).
        rectmat_path: the rectmat file; its MD5 is the cache key. Without it
            the blocks are built and not cached.
        mode: 'off', 'memory' or 'disk'.
        cache_dir: directory for 'disk' mode (required for it).
    """
    logger = logger or logging.getLogger(__name__)
    mode = parse_cache_mode(mode)
    shape = tuple(int(n) for n in flux_shape_3d)
    if mode == "off" or rectmat_path is None:
        return build_lenslet_blocks(R_matrix, shape)

    md5 = file_md5(rectmat_path)
    key = (md5, shape)
    blocks = _mem_cache.get(key)
    if blocks is not None and (mode == "memory" or blocks.cache_dir is not None):
        return blocks

    if mode == "memory":
        blocks = build_lenslet_blocks(R_matrix, shape)
        _mem_cache[key] = blocks
        logger.info(f"Built lenslet blocks for {os.path.basename(rectmat_path)} "
                    f"({blocks.nbytes / 1e9:.2f} GB, kept in memory).")
        return blocks

    if not cache_dir:
        raise ValueError("chi2_block_cache = 'disk' needs a cache directory.")
    path = cache_path(cache_dir, md5, shape)
    blocks = load_blocks(path, md5, shape)
    if blocks is None:
        if os.path.exists(path):
            logger.warning(f"Lenslet block cache {path} does not match "
                           f"{os.path.basename(rectmat_path)}; rebuilding it.")
            stale = path + ".stale-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            os.rename(path, stale)
            shutil.rmtree(stale, ignore_errors=True)
        built = build_lenslet_blocks(R_matrix, shape)
        write_blocks(built, path, rectmat_path, md5)
        blocks = load_blocks(path, md5, shape)
        if blocks is None:
            raise RuntimeError(f"Lenslet block cache {path} could not be read back.")
        logger.info(f"Wrote lenslet block cache {path} ({built.nbytes / 1e9:.2f} GB) "
                    f"for {os.path.basename(rectmat_path)} (md5 {md5}).")
    else:
        logger.info(f"Using lenslet block cache {path}.")
    _mem_cache[key] = blocks
    return blocks
