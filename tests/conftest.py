from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ._ground_truth import (  # noqa: F401  # pylint: disable=unused-import
    TESTS_DIR,
    ground_truth,
    pytest_addoption,
    pytest_collection_finish,
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_generate_tests,
)


PBMC3K_H5AD = TESTS_DIR / "data" / "pbmc3k" / "pbmc3k.h5ad"


@pytest.fixture(scope="session")
def pbmc3k_h5ad_path() -> Path:
    if not PBMC3K_H5AD.exists():
        pytest.fail(f"Baseline test fixture not found: {PBMC3K_H5AD.relative_to(TESTS_DIR.parent)}")
    return PBMC3K_H5AD


@pytest.fixture
def pbmc3k_adata(pbmc3k_h5ad_path: Path) -> ad.AnnData:
    return ad.read_h5ad(pbmc3k_h5ad_path)


@pytest.fixture
def pbmc3k_dense_adata(pbmc3k_adata: ad.AnnData) -> ad.AnnData:
    return _copy_adata_with_matrix_format(pbmc3k_adata, _to_dense_matrix)


@pytest.fixture
def pbmc3k_sparse_adata(pbmc3k_adata: ad.AnnData) -> ad.AnnData:
    return _copy_adata_with_matrix_format(pbmc3k_adata, _to_sparse_matrix)


def _copy_adata_with_matrix_format(
    adata: ad.AnnData,
    convert: Callable[[Any], Any],
) -> ad.AnnData:
    copied = adata.copy()
    copied.X = convert(copied.X)
    for layer_name, layer in list(copied.layers.items()):
        copied.layers[layer_name] = convert(layer)
    return copied


def _to_dense_matrix(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _to_sparse_matrix(matrix: Any) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix))



def _build_synthetic_adata(*, as_sparse: bool = False) -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                ["A", "A", "A", "A", "B", "B", "B", "B"],
                categories=["B", "A"],
                ordered=True,
            ),
            "sample": pd.Categorical(
                ["s1", "s1", "s2", "s2", "s1", "s1", "s2", "s2"],
                categories=["s1", "s2"],
                ordered=True,
            ),
        },
        index=["A_s1_1", "A_s1_2", "A_s2_1", "A_s2_2", "B_s1_1", "B_s1_2", "B_s2_1", "B_s2_2"],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3", "g4", "g5", "g6", "g7"])
    matrix = np.array(
        [
            [6.0, 0.0, 1.0, 1.0, 5.0, 0.0, 4.0], # A, s1
            [5.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0], # A, s1
            [6.0, 0.0, 0.0, 1.0, 0.0, 5.0, 0.0], # A, s2
            [5.0, 0.0, 0.0, 1.0, 0.0, 4.0, 0.0], # A, s2
            [0.0, 6.0, 1.0, 0.0, 5.0, 0.0, 0.0], # B, s1
            [0.0, 5.0, 0.0, 0.0, 4.0, 0.0, 0.0], # B, s1
            [0.0, 6.0, 0.0, 0.0, 0.0, 5.0, 0.0], # B, s2
            [0.0, 5.0, 0.0, 0.0, 0.0, 4.0, 0.0], # B, s2
        ],
        dtype=float,
    )
    x = sparse.csr_matrix(matrix) if as_sparse else matrix
    counts = sparse.csr_matrix(matrix.astype(int)) if as_sparse else matrix.astype(int)
    return ad.AnnData(X=x, obs=obs, var=var, layers={"counts": counts})


@pytest.fixture
def synthetic_grouped_adata() -> ad.AnnData:
    return _build_synthetic_adata()


@pytest.fixture
def synthetic_sparse_adata() -> ad.AnnData:
    return _build_synthetic_adata(as_sparse=True)
