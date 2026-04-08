from __future__ import annotations

import re
from typing import cast

import numpy as np
import pandas as pd
import pytest

from py_cellchat.database import CellChatDB
from py_cellchat.modeling.statistics import tri_mean

from ..test_util import assert_compare, make_cellchat


_PBMC_SEED = 1


def _synthetic_lr_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ligand": "g1",
                "receptor": "g2",
                "agonist": "",
                "antagonist": "",
                "co_A_receptor": "",
                "co_I_receptor": "",
                "annotation": "Secreted Signaling",
            }
        ],
        index=["g1_g2"],
    )


def _synthetic_db() -> CellChatDB:
    return CellChatDB(
        interaction=pd.DataFrame(),
        complex=pd.DataFrame(),
        cofactor=pd.DataFrame(),
    )


def _base_synthetic_cellchat(adata):
    cellchat = make_cellchat(adata)
    cellchat.db = _synthetic_db()
    cellchat.subset_data(features=["g1", "g2"])
    cellchat.lr = _synthetic_lr_table()
    return cellchat


def _group_levels(cellchat) -> list[str]:
    return cellchat.idents.cat.categories.astype(str).tolist()


def _lr_names(cellchat) -> list[str]:
    assert cellchat.lr is not None
    return cellchat.lr.index.astype(str).tolist()


def _net_prob(cellchat) -> np.ndarray:
    assert cellchat.net is not None
    return cellchat.net["prob"]


def _net_pval(cellchat) -> np.ndarray:
    assert cellchat.net is not None
    return cellchat.net["pval"]


def _communication_signature(cellchat) -> list[tuple[str, str, str, float, float]]:
    groups = _group_levels(cellchat)
    lr_names = _lr_names(cellchat)
    prob = _net_prob(cellchat)
    pval = _net_pval(cellchat)

    entries: list[tuple[str, str, str, float, float]] = []
    for source_idx, target_idx, lr_idx in np.argwhere(prob > 0):
        entries.append(
            (
                groups[int(source_idx)],
                groups[int(target_idx)],
                lr_names[int(lr_idx)],
                float(prob[source_idx, target_idx, lr_idx]),
                float(pval[source_idx, target_idx, lr_idx]),
            )
        )
    return entries


def _communication_signature_from_ground_truth(
    payload: list[dict[str, object]],
) -> list[tuple[str, str, str, float, float]]:
    return [
        (
            str(row["source"]),
            str(row["target"]),
            str(row["lr"]),
            float(cast(int | float, row["prob"])),
            float(cast(int | float, row["pval"])),
        )
        for row in payload
    ]


def _make_synthetic_cellchat(adata, *, population_size: bool = False):
    cellchat = _base_synthetic_cellchat(adata)
    return cellchat.compute_communication_probability(
        population_size=population_size,
        nboot=5,
        seed_use=1,
    )


def _expected_synthetic_probability(*, population_size: bool) -> float:
    ligand_avg = tri_mean(np.array([1.0, 5.0 / 6.0, 1.0, 5.0 / 6.0], dtype=float))
    receptor_avg = tri_mean(np.array([1.0, 5.0 / 6.0, 1.0, 5.0 / 6.0], dtype=float))
    expected = ligand_avg * receptor_avg / (0.5 + ligand_avg * receptor_avg)
    if population_size:
        expected *= 0.25
    return float(expected)


def _expected_synthetic_pval() -> np.ndarray:
    return np.array(
        [
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )


def _run_pbmc_pipeline(
    adata,
    *,
    nboot: int | None = None,
    seed_use: int = _PBMC_SEED,
):
    cellchat = make_cellchat(adata)
    cellchat.load_database("human")
    cellchat.subset_data()
    cellchat.identify_over_expressed_genes(min_cells=10)
    cellchat.identify_over_expressed_interactions()
    if nboot is None:
        cellchat.compute_communication_probability(seed_use=seed_use)
    else:
        cellchat.compute_communication_probability(
            nboot=nboot,
            seed_use=seed_use,
        )
    return cellchat


@pytest.mark.synthetic
@pytest.mark.unit
def test_compute_communication_probability_synthetic_default(
    synthetic_grouped_adata,
):
    cellchat = _make_synthetic_cellchat(synthetic_grouped_adata)

    assert _group_levels(cellchat) == ["B", "A"]
    assert _lr_names(cellchat) == ["g1_g2"]

    expected_prob = np.array(
        [
            [0.0, 0.0],
            [_expected_synthetic_probability(population_size=False), 0.0],
        ],
        dtype=float,
    )
    expected_pval = np.array(
        [
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(_net_prob(cellchat)[:, :, 0], expected_prob)
    np.testing.assert_allclose(_net_pval(cellchat)[:, :, 0], expected_pval)


@pytest.mark.synthetic
@pytest.mark.unit
def test_compute_communication_probability_population_size(
    synthetic_grouped_adata,
):
    cellchat = _make_synthetic_cellchat(
        synthetic_grouped_adata,
        population_size=True,
    )

    expected_prob = np.array(
        [
            [0.0, 0.0],
            [_expected_synthetic_probability(population_size=True), 0.0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(_net_prob(cellchat)[:, :, 0], expected_prob)
    np.testing.assert_allclose(_net_pval(cellchat)[:, :, 0], _expected_synthetic_pval())


@pytest.mark.synthetic
@pytest.mark.unit
def test_compute_communication_probability_dense_sparse(
    synthetic_grouped_adata,
    synthetic_sparse_adata,
):
    dense = _make_synthetic_cellchat(synthetic_grouped_adata)
    sparse = _make_synthetic_cellchat(synthetic_sparse_adata)

    assert _group_levels(dense) == _group_levels(sparse)
    assert _lr_names(dense) == _lr_names(sparse)
    np.testing.assert_allclose(_net_prob(dense), _net_prob(sparse))
    np.testing.assert_allclose(_net_pval(dense), _net_pval(sparse))


@pytest.mark.synthetic
@pytest.mark.unit
def test_compute_communication_probability_empty_lr(
    synthetic_grouped_adata,
):
    cellchat = _base_synthetic_cellchat(synthetic_grouped_adata)
    assert cellchat.lr is not None
    cellchat.lr = cellchat.lr.iloc[0:0].copy()

    cellchat.compute_communication_probability(nboot=1, seed_use=1)

    assert list(_net_prob(cellchat).shape) == [2, 2, 0]
    assert list(_net_pval(cellchat).shape) == [2, 2, 0]


@pytest.mark.synthetic
@pytest.mark.unit
def test_compute_communication_probability_missing_db(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)
    cellchat.subset_data(features=["g1", "g2"])
    cellchat.lr = _synthetic_lr_table()

    with pytest.raises(ValueError, match="Must load a CellChatDB"):
        cellchat.compute_communication_probability(nboot=1, seed_use=1)


@pytest.mark.synthetic
@pytest.mark.unit
def test_compute_communication_probability_missing_subset_data(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)
    cellchat.db = _synthetic_db()
    cellchat.lr = _synthetic_lr_table()

    with pytest.raises(ValueError, match="Must run subset_data"):
        cellchat.compute_communication_probability(nboot=1, seed_use=1)


@pytest.mark.synthetic
@pytest.mark.unit
def test_compute_communication_probability_raw_use_false(
    synthetic_grouped_adata,
):
    cellchat = _base_synthetic_cellchat(synthetic_grouped_adata)

    with pytest.raises(NotImplementedError, match=re.escape("raw_use=False requires projected or smoothed data support, which is not implemented yet")):
        cellchat.compute_communication_probability(raw_use=False, nboot=1, seed_use=1)


@pytest.mark.synthetic
@pytest.mark.unit
@pytest.mark.parametrize("nboot", [0, -1])
def test_compute_communication_probability_nboot_invalid(
    synthetic_grouped_adata,
    nboot: int,
):
    cellchat = _base_synthetic_cellchat(synthetic_grouped_adata)

    with pytest.raises(ValueError, match="nboot must be a positive integer"):
        cellchat.compute_communication_probability(nboot=nboot, seed_use=1)


@pytest.mark.synthetic
@pytest.mark.unit
def test_compute_communication_probability_unused_group_levels(
    synthetic_grouped_adata,
):
    cellchat = _base_synthetic_cellchat(synthetic_grouped_adata)
    cellchat.adata.obs[cellchat.group_by_col] = pd.Series(
        cellchat.idents.astype(str),
        index=cellchat.adata.obs_names,
        name=cellchat.group_by_col,
        dtype=pd.CategoricalDtype(categories=["B", "A", "unused"], ordered=True),
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Please check unique(cellchat.idents) and ensure that the factor levels are correct. "
            "You may need to drop unused levels before running compute_communication_probability."
        ),
    ):
        cellchat.compute_communication_probability(nboot=1, seed_use=1)

# ══════════════════════════════════════════════════════════════════════════════
# pbmc3k
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.r_script("r_scripts/generate_cp_ground_truth.R")
@pytest.mark.ground_truth("pbmc3k_benchmark/compute_communication_probability/compute_communication_probability.json")
@pytest.mark.pbmc3k
@pytest.mark.integration
def test_compute_communication_probability_pbmc3k_default(
    pbmc3k_sparse_adata,
    ground_truth,
):
    cellchat = _run_pbmc_pipeline(pbmc3k_sparse_adata)

    assert_compare(_group_levels(cellchat), ground_truth["groups"])
    assert_compare(_lr_names(cellchat), ground_truth["lr_names"])
    assert list(_net_prob(cellchat).shape) == ground_truth["prob_shape"]
    assert list(_net_pval(cellchat).shape) == ground_truth["pval_shape"]
    assert float(_net_prob(cellchat).sum()) == pytest.approx(ground_truth["prob_sum"])
    assert int(np.count_nonzero(_net_prob(cellchat))) == ground_truth["prob_nonzero"]
    assert_compare(
        _communication_signature(cellchat),
        _communication_signature_from_ground_truth(ground_truth["nonzero_communications"]),
        is_numeric=True,
    )


@pytest.mark.pbmc3k
@pytest.mark.integration
def test_compute_communication_probability_pbmc3k_population_size(
    pbmc3k_sparse_adata,
):
    default = _run_pbmc_pipeline(pbmc3k_sparse_adata)
    scaled = _run_pbmc_pipeline(pbmc3k_sparse_adata)
    scaled.compute_communication_probability(
        population_size=True,
        seed_use=_PBMC_SEED,
    )

    groups = _group_levels(default)
    group_sizes = default.idents.value_counts(sort=False).reindex(groups).to_numpy(dtype=float)
    group_props = group_sizes / len(default.idents)
    expected_prob = np.einsum("ijk,ij->ijk", _net_prob(default), np.outer(group_props, group_props))

    np.testing.assert_allclose(_net_prob(scaled), expected_prob)
    np.testing.assert_allclose(_net_pval(scaled), _net_pval(default))



@pytest.mark.pbmc3k
@pytest.mark.integration
def test_compute_communication_probability_pbmc3k_dense_sparse(
    pbmc3k_dense_adata,
    pbmc3k_sparse_adata,
):
    dense = _run_pbmc_pipeline(pbmc3k_dense_adata)
    sparse = _run_pbmc_pipeline(pbmc3k_sparse_adata)

    assert_compare(_group_levels(dense), _group_levels(sparse))
    assert_compare(_lr_names(dense), _lr_names(sparse))
    np.testing.assert_allclose(_net_prob(dense), _net_prob(sparse))
    np.testing.assert_allclose(_net_pval(dense), _net_pval(sparse))
