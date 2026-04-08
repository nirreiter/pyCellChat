from __future__ import annotations

import re

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from py_cellchat.database import CellChatDB
from ..test_util import assert_compare

from ..test_util import make_cellchat


def _filter_db() -> CellChatDB:
    interaction = pd.DataFrame(
        [
            {"ligand": "g1", "receptor": "g2", "annotation": "Secreted Signaling"},
            {"ligand": "g5", "receptor": "g2", "annotation": "Secreted Signaling"},
            {"ligand": "g7", "receptor": "g2", "annotation": "Secreted Signaling"},
        ],
        index=["g1_g2", "g5_g2", "g7_g2"],
    )
    gene_info = pd.DataFrame({"Symbol": ["g1", "g2", "g5", "g7"]})
    return CellChatDB(
        interaction=interaction,
        complex=pd.DataFrame(),
        cofactor=pd.DataFrame(),
        gene_info=gene_info,
    )


def _run_synthetic_pipeline(adata: ad.AnnData):
    cellchat = make_cellchat(adata, sample_column="sample")
    cellchat.db = _filter_db()
    cellchat.subset_data(features=["g1", "g2", "g5", "g7"])
    cellchat.lr = cellchat.db.interaction.copy()
    cellchat.compute_communication_probability()
    return cellchat


def _run_pbmc_pipeline(adata: ad.AnnData):
    cellchat = make_cellchat(adata)
    cellchat.load_database("human")
    cellchat.subset_data()
    cellchat.identify_over_expressed_genes()
    cellchat.identify_over_expressed_interactions()
    cellchat.compute_communication_probability()
    return cellchat


def _communication_signature(cellchat) -> list[tuple[str, str, str, float, float]]:
    assert cellchat.lr is not None
    groups = cellchat.idents.cat.categories.astype(str).tolist()
    lr_names = cellchat.lr.index.astype(str).tolist()
    prob = cellchat.net["prob"]
    pval = cellchat.net["pval"]

    signature: list[tuple[str, str, str, float, float]] = []
    for source_index, target_index, lr_index in np.argwhere(prob > 0):
        signature.append(
            (
                groups[int(source_index)],
                groups[int(target_index)],
                lr_names[int(lr_index)],
                float(prob[source_index, target_index, lr_index]),
                float(pval[source_index, target_index, lr_index]),
            )
        )
    return signature


def _communication_signature_from_ground_truth(
    payload: list[dict[str, object]],
) -> list[tuple[str, str, str, float, float]]:
    return [
        (
            str(row["source"]),
            str(row["target"]),
            str(row["lr"]),
            float(row["prob"]),
            float(row["pval"]),
        )
        for row in payload
    ]


def _rare_population_adata(*, as_sparse: bool = False) -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C"],
                categories=["A", "B", "C"],
                ordered=True,
            ),
            "sample": pd.Categorical(
                ["s1", "s1", "s2", "s2", "s1", "s1", "s2", "s2", "s1", "s1", "s2"],
                categories=["s1", "s2"],
                ordered=True,
            ),
        },
        index=[
            "A_s1_1",
            "A_s1_2",
            "A_s2_1",
            "A_s2_2",
            "B_s1_1",
            "B_s1_2",
            "B_s2_1",
            "B_s2_2",
            "C_s1_1",
            "C_s1_2",
            "C_s2_1",
        ],
    )
    var = pd.DataFrame(index=["g1", "g2", "g5", "g7"])
    matrix = np.array(
        [
            [6.0, 0.0, 1.0, 0.0],
            [5.0, 0.0, 1.0, 0.0],
            [6.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 6.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 6.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 6.0],
        ],
        dtype=float,
    )
    x = matrix if not as_sparse else pytest.importorskip("scipy.sparse").csr_matrix(matrix)
    counts = matrix.astype(int) if not as_sparse else pytest.importorskip("scipy.sparse").csr_matrix(matrix.astype(int))
    return ad.AnnData(X=x, obs=obs, var=var, layers={"counts": counts})


def _lr_index(cellchat, lr_name: str) -> int:
    assert cellchat.lr is not None
    return cellchat.lr.index.astype(str).tolist().index(lr_name)


@pytest.mark.synthetic
@pytest.mark.unit
@pytest.mark.parametrize(
    ("min_cells", "expected_prob"),
    [
        (3, np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)),
        (4, np.zeros((3, 3), dtype=float)),
    ],
)
def test_filter_communication_min_cells_non_filter_keep(
    min_cells: int,
    expected_prob: np.ndarray,
):
    adata = _rare_population_adata()
    cellchat = make_cellchat(adata, sample_column="sample")
    original_prob = np.ones((3, 3, 1), dtype=float)
    original_pval = np.zeros((3, 3, 1), dtype=float)
    cellchat.net = {
        "prob": original_prob.copy(),
        "pval": original_pval.copy(),
    }

    cellchat.filter_communication(min_cells=min_cells, non_filter_keep=True)

    np.testing.assert_allclose(cellchat.net["prob_non_filter"], original_prob)
    np.testing.assert_allclose(cellchat.net["pval_non_filter"], original_pval)
    np.testing.assert_allclose(cellchat.net["prob"][:, :, 0], expected_prob)
    np.testing.assert_allclose(cellchat.net["pval"], original_pval)


@pytest.mark.synthetic
@pytest.mark.unit
@pytest.mark.parametrize(
    ("net", "message"),
    [
        ({}, "CellChat.net must contain 'prob' and 'pval'"),
        ({"prob": np.zeros((2, 2), dtype=float), "pval": np.zeros((2, 2), dtype=float)}, "cellchat.net['prob'] must be a 3D array"),
        ({"prob": np.zeros((2, 2, 1), dtype=float), "pval": np.zeros((2, 2, 2), dtype=float)}, "cellchat.net['pval'] must match the shape"),
    ],
)
def test_filter_communication_invalid_net(
    synthetic_grouped_adata,
    net: dict[str, np.ndarray],
    message: str,
):
    cellchat = make_cellchat(synthetic_grouped_adata, sample_column="sample")
    cellchat.net = net

    with pytest.raises(ValueError, match=re.escape(message)):
        cellchat.filter_communication()


@pytest.mark.synthetic
@pytest.mark.unit
def test_filter_communication_min_samples_gt_samples(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata, sample_column="sample")
    cellchat.net = {
        "prob": np.zeros((2, 2, 1), dtype=float),
        "pval": np.ones((2, 2, 1), dtype=float),
    }

    with pytest.raises(ValueError, match="There are only 2 samples"):
        cellchat.filter_communication(min_samples=3)


@pytest.mark.synthetic
@pytest.mark.unit
def test_filter_communication_min_samples_dense_sparse(
    synthetic_grouped_adata,
    synthetic_sparse_adata,
):
    dense = _run_synthetic_pipeline(synthetic_grouped_adata)
    sparse = _run_synthetic_pipeline(synthetic_sparse_adata)

    dense.filter_communication(min_cells=1, min_samples=2)
    sparse.filter_communication(min_cells=1, min_samples=2)

    np.testing.assert_allclose(dense.net["prob"], sparse.net["prob"])
    np.testing.assert_allclose(dense.net["pval"], sparse.net["pval"])

    g1_index = _lr_index(dense, "g1_g2")
    g5_index = _lr_index(dense, "g5_g2")
    assert float(dense.net["prob"][1, 0, g1_index]) > 0.0
    assert float(dense.net["prob"][1, 0, g5_index]) == 0.0


@pytest.mark.synthetic
@pytest.mark.unit
def test_filter_communication_rare_keep_true():
    without_rare_keep = _run_synthetic_pipeline(_rare_population_adata())
    with_rare_keep = _run_synthetic_pipeline(_rare_population_adata())

    without_rare_keep.filter_communication(min_cells=1, min_samples=2, rare_keep=False)
    with_rare_keep.filter_communication(min_cells=1, min_samples=2, rare_keep=True)

    g7_index = _lr_index(with_rare_keep, "g7_g2")
    assert float(without_rare_keep.net["prob"][2, 1, g7_index]) == 0.0
    assert float(with_rare_keep.net["prob"][2, 1, g7_index]) > 0.0


@pytest.mark.pbmc3k
@pytest.mark.integration
@pytest.mark.r_script("r_scripts/generate_fc_ground_truth.R")
@pytest.mark.ground_truth("pbmc3k_benchmark/filter_communication/filter_communication_default.json")
def test_filter_communication_pbmc3k_default(
    pbmc3k_sparse_adata,
    ground_truth,
):
    cellchat = _run_pbmc_pipeline(pbmc3k_sparse_adata)
    cellchat.filter_communication()

    assert list(cellchat.net["prob"].shape) == ground_truth["prob_shape"]
    assert list(cellchat.net["pval"].shape) == ground_truth["pval_shape"]
    assert float(cellchat.net["prob"].sum()) == pytest.approx(ground_truth["prob_sum"])
    assert float(cellchat.net["pval"].sum()) == pytest.approx(ground_truth["pval_sum"])
    assert int(np.count_nonzero(cellchat.net["prob"])) == ground_truth["prob_nonzero"]
    assert_compare(
        _communication_signature(cellchat),
        _communication_signature_from_ground_truth(ground_truth["nonzero_communications"]),
        is_numeric=True,
    )


@pytest.mark.pbmc3k
@pytest.mark.integration
def test_filter_communication_pbmc3k_dense_sparse(
    pbmc3k_dense_adata,
    pbmc3k_sparse_adata,
):
    dense = _run_pbmc_pipeline(pbmc3k_dense_adata)
    sparse = _run_pbmc_pipeline(pbmc3k_sparse_adata)

    dense.filter_communication()
    sparse.filter_communication()

    assert list(dense.net["prob"].shape) == list(sparse.net["prob"].shape)
    assert list(dense.net["pval"].shape) == list(sparse.net["pval"].shape)
    np.testing.assert_allclose(dense.net["prob"], sparse.net["prob"])
    np.testing.assert_allclose(dense.net["pval"], sparse.net["pval"])
    assert_compare(_communication_signature(dense), _communication_signature(sparse), is_numeric=True)
