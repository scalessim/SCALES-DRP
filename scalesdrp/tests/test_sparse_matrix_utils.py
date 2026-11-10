import io

import numpy as np
import pytest
from astropy.io import fits
from scipy import sparse

from scalesdrp.core.sparse_matrix_utils import pack_matrix_fits, unpack_matrix_fits


def random_matrix(fmt="csr", dtype=np.float64, shape=(300, 500)):
    rng = np.random.default_rng(0)
    return sparse.random(*shape, density=0.02, format=fmt, dtype=dtype, random_state=rng)


def round_trip(matrix, header=None):
    """Pack, write to bytes, read back and unpack, as a file on disk would go."""
    buf = io.BytesIO()
    pack_matrix_fits(matrix, header=header).writeto(buf)
    buf.seek(0)
    with fits.open(buf) as hdul:
        return unpack_matrix_fits(hdul), hdul[0].header.copy()


def assert_same(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.format == b.format
    assert (a != b).nnz == 0


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_round_trip_is_exact(fmt, dtype):
    m = random_matrix(fmt, dtype)
    out, _ = round_trip(m)
    assert_same(out, m)


def test_round_trip_keeps_int64_indices():
    m = random_matrix("coo")
    m.row = m.row.astype(np.int64)
    m.col = m.col.astype(np.int64)
    out, _ = round_trip(m)
    assert_same(out, m)


def test_empty_matrix():
    m = sparse.csr_matrix((50, 60), dtype=np.float64)
    out, _ = round_trip(m)
    assert_same(out, m)


def test_header_is_copied_to_primary():
    hdr = fits.Header()
    hdr["STATEID"] = ("abc123", "instrument configuration")
    hdr["IFSMODE"] = "LowRes-K"
    hdr["HISTORY"] = "built from 40 monochromator frames"
    _, primary = round_trip(random_matrix(), header=hdr)
    assert primary["STATEID"] == "abc123"
    assert primary.comments["STATEID"] == "instrument configuration"
    assert primary["IFSMODE"] == "LowRes-K"
    assert "built from 40 monochromator frames" in str(primary["HISTORY"])


def test_image_header_structure_is_not_carried_over():
    # e.g. the header of a 2-D master calunit the matrix was built from
    image_hdr = fits.PrimaryHDU(np.zeros((4, 5), dtype=np.float32)).header
    image_hdr["IFSMODE"] = "LowRes-K"
    hdul = pack_matrix_fits(random_matrix(), header=image_hdr)
    primary = hdul[0].header
    assert primary["NAXIS"] == 0
    assert "NAXIS1" not in primary
    assert primary["IFSMODE"] == "LowRes-K"
    hdul.verify("exception")


def test_matrix_extension_records_shape_and_format():
    m = random_matrix("csc", np.float32, shape=(7, 9))
    hdr = pack_matrix_fits(m)["MATRIX_COO"].header
    assert (hdr["MSHAPE0"], hdr["MSHAPE1"]) == (7, 9)
    assert hdr["MDTYPE"] == "float32"
    assert hdr["MFORMAT"] == "csc"
    assert hdr["MNNZ"] == m.nnz


def test_unsupported_dtype_is_rejected():
    m = sparse.csr_matrix(np.eye(3, dtype=np.complex128))
    with pytest.raises(ValueError, match="value dtype"):
        pack_matrix_fits(m)
