from __future__ import annotations

from typing import TypeAlias, Union

import anndata as ad
import numpy as np
from scipy import sparse

MatrixType: TypeAlias = Union[np.ndarray, sparse.sparray, sparse.spmatrix]


def is_integer_matrix(matrix: MatrixType, atol: float = 1e-12, rtol: float = 1e-12) -> bool:
    if isinstance(matrix, np.ndarray):
        data = matrix
    else:
        data = matrix.data  # pyright: ignore[reportAttributeAccessIssue]
    if np.issubdtype(data.dtype, np.integer):
        return True

    return bool(np.all(np.isclose(data, np.round(data), atol=atol, rtol=rtol)))


def get_adata_matrix_checked(
    adata: ad.AnnData,
    is_raw: bool = False,
    layer_name: str | None = None,
) -> MatrixType:
    if layer_name is None:
        if is_raw:
            if adata.raw is None:
                raise ValueError("AnnData raw matrix is not available")
            matrix = adata.raw.X
        else:
            matrix = adata.X
    else:
        matrix = adata.layers[layer_name]

    if not (
        isinstance(matrix, np.ndarray)
        or isinstance(matrix, sparse.spmatrix)
        or isinstance(matrix, sparse.sparray)
    ):
        if layer_name is None:
            if is_raw:
                raise ValueError(f"AnnData raw.X matrix is of invalid type '{type(matrix)}'")
            raise ValueError(f"AnnData X matrix is of invalid type '{type(matrix)}'")
        raise ValueError(
            f"AnnData object layer '{layer_name}' is of invalid type '{type(matrix)}'"
        )

    correct_type_matrix: MatrixType = matrix

    if correct_type_matrix.min() < 0:  # pyright: ignore[reportAttributeAccessIssue]
        raise ValueError("Values in the provided AnnData matrix cannot be negative")

    if is_raw and not is_integer_matrix(correct_type_matrix):
        raise ValueError("Values in the provided AnnData raw matrix must be integers")

    return correct_type_matrix
