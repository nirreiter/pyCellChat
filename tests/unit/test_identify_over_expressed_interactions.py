from __future__ import annotations

import anndata as ad
import pytest

from ..test_util import assert_compare, make_cellchat


pytestmark = [
    pytest.mark.r_script("r_scripts/generate_pbmc3k_ground_truths.R"),
    pytest.mark.pbmc3k,
    pytest.mark.integration,
]

PBMC3K_MARKER_PANEL = [
    "CD3D",
    "IL7R",
    "NKG7",
    "GNLY",
    "LYZ",
    "CST3",
    "FCER1A",
    "FCGR3A",
    "MS4A1",
    "CD79A",
    "PPBP",
]


@pytest.fixture
def pbmc3k_feature_panel(pbmc3k_adata: ad.AnnData) -> list[str]:
    available_features = set(pbmc3k_adata.var_names.astype(str))
    feature_panel = [feature for feature in PBMC3K_MARKER_PANEL if feature in available_features]
    if len(feature_panel) < 6:
        pytest.fail(
            "PBMC3K marker panel unexpectedly small; expected at least 6 available marker genes "
            f"but found {len(feature_panel)}"
        )
    return feature_panel


def _lr_sig_names(cellchat) -> list[str]:
    return cellchat.lr.index.tolist()


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_interactions_variable_both.json")
def test_identify_over_expressed_interactions_variable_both(
    pbmc3k_dense_adata,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_dense_adata)
    cellchat.load_database("human")
    cellchat.subset_data()
    cellchat.identify_over_expressed_genes(min_cells=10)
    cellchat.identify_over_expressed_interactions(variable_both=True)
    # print(_lr_sig_names(cellchat))
    # print(ground_truth["lr_sig_names"])
    assert_compare(_lr_sig_names(cellchat), ground_truth["lr_sig_names"])


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_interactions_variable_one.json")
def test_identify_over_expressed_interactions_variable_one(
    pbmc3k_dense_adata,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_dense_adata)
    cellchat.load_database("human")
    cellchat.subset_data()
    cellchat.identify_over_expressed_genes(min_cells=10)
    cellchat.identify_over_expressed_interactions(variable_both=False)

    assert_compare(_lr_sig_names(cellchat), ground_truth["lr_sig_names"])


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_interactions_variable_both.json")
def test_identify_over_expressed_interactions_inplace_false(
    pbmc3k_dense_adata,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_dense_adata)
    cellchat.load_database("human")
    cellchat.subset_data()
    cellchat.identify_over_expressed_genes(min_cells=10)
    result = cellchat.identify_over_expressed_interactions(inplace=False)

    assert result is not None
    assert cellchat.lr is None
    
    print(result)
    print()
    
    assert_compare(result.index.tolist(), ground_truth["lr_sig_names"])


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_interactions_explicit_features.json")
def test_identify_over_expressed_interactions_explicit_features(
    pbmc3k_dense_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_dense_adata)
    cellchat.load_database("human")
    cellchat.subset_data()
    cellchat.identify_over_expressed_interactions(features=pbmc3k_feature_panel)

    assert_compare(_lr_sig_names(cellchat), ground_truth["lr_sig_names"])
