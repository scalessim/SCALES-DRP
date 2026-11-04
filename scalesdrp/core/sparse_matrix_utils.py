"""
Store scipy.sparse matrices (rectification matrices, bad pixel matrices) as FITS.

KOA archives FITS files, not .npz, so a sparse matrix is written as a binary
table of its nonzero entries in COO form (X, Y, VALUE), plus the keywords
needed to rebuild it. Metadata describing the matrix (instrument state, IFS
mode, DRP version, ...) goes in the primary header.
"""
import numpy as np
from astropy.io import fits
from scipy import sparse

MATRIX_EXTNAME = "MATRIX_COO"

# FITS binary table formats for the dtypes we accept
_INDEX_FORMATS = {"int16": "I", "int32": "J", "int64": "K"}
_VALUE_FORMATS = {"int16": "I", "int32": "J", "int64": "K",
                  "float32": "E", "float64": "D"}


def pack_matrix_fits(matrix, header=None):
    """
    Pack a sparse matrix into a FITS HDUList.

    Parameters
    ----------
    matrix : scipy.sparse matrix or array
        Any sparse format (csr, csc, coo, ...). The format is recorded and
        restored by `unpack_matrix_fits`.
    header : astropy.io.fits.Header, optional
        Keywords to copy into the primary header, e.g. the header of the
        calibration frames the matrix was built from. Structural keywords
        (SIMPLE, BITPIX, NAXIS*, EXTEND, ...) are dropped.

    Returns
    -------
    astropy.io.fits.HDUList
        An empty primary HDU carrying `header`, and a binary table extension
        named MATRIX_COO with columns X (column index), Y (row index) and VALUE.
    """
    coo = matrix.tocoo()
    rows, cols, values = coo.row, coo.col, coo.data

    idx_format = _INDEX_FORMATS.get(str(rows.dtype))
    value_format = _VALUE_FORMATS.get(str(values.dtype))
    if idx_format is None:
        raise ValueError(f"Unsupported index dtype {rows.dtype}; "
                         f"expected one of {sorted(_INDEX_FORMATS)}")
    if value_format is None:
        raise ValueError(f"Unsupported value dtype {values.dtype}; "
                         f"expected one of {sorted(_VALUE_FORMATS)}")

    matrix_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="X", format=idx_format, array=cols),
        fits.Column(name="Y", format=idx_format, array=rows),
        fits.Column(name="VALUE", format=value_format, array=values),
    ], name=MATRIX_EXTNAME)

    hdr = matrix_hdu.header
    hdr["MSHAPE0"] = (coo.shape[0], "original matrix: number of rows")
    hdr["MSHAPE1"] = (coo.shape[1], "original matrix: number of columns")
    hdr["MDTYPE"] = (str(coo.dtype), "original matrix: value dtype")
    hdr["MFORMAT"] = (matrix.format, "original scipy.sparse format (csr/csc/coo/...)")
    hdr["MNNZ"] = (coo.nnz, "number of stored entries")
    hdr.add_comment("Sparse matrix stored in COO format.")
    hdr.add_comment("To reconstruct, place each VALUE at [Y (row), X (column)], 0 indexed.")

    primary_header = header.copy(strip=True) if header is not None else None
    primary_hdu = fits.PrimaryHDU(header=primary_header)

    return fits.HDUList([primary_hdu, matrix_hdu])


def unpack_matrix_fits(hdul):
    """
    Rebuild the sparse matrix written by `pack_matrix_fits`.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        The packed file. Its metadata is in ``hdul[0].header``.

    Returns
    -------
    scipy.sparse matrix
        Same shape, dtype and sparse format as the matrix that was packed.
    """
    matrix_hdu = hdul[MATRIX_EXTNAME]
    hdr = matrix_hdu.header
    shape = (hdr["MSHAPE0"], hdr["MSHAPE1"])
    dtype = np.dtype(hdr["MDTYPE"])
    fmt = hdr.get("MFORMAT", "coo")

    data = matrix_hdu.data
    # FITS stores big-endian; convert so scipy gets native-order arrays
    rows = np.asarray(data["Y"]).astype(data["Y"].dtype.newbyteorder("="))
    cols = np.asarray(data["X"]).astype(data["X"].dtype.newbyteorder("="))
    values = np.asarray(data["VALUE"]).astype(dtype)

    coo = sparse.coo_matrix((values, (rows, cols)), shape=shape, dtype=dtype)
    return coo.asformat(fmt)
