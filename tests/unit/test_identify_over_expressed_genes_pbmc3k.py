from __future__ import annotations

import anndata as ad
import pytest

from py_cellchat.database import load_cellchat_db

from ..test_util import assert_compare, make_cellchat, selected_features_sorted, feature_values_sorted, feature_table, feature_table_from_ground_truth, with_condition_column


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


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes_no_de_marker_panel.json")
def test_identify_over_expressed_genes_no_de(
    pbmc3k_dense_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_dense_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    cellchat.identify_over_expressed_genes(
        do_differential_expression=False,
        min_cells=10,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes_de_marker_panel.json")
def test_identify_over_expressed_genes_de(
    pbmc3k_dense_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_dense_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    cellchat.identify_over_expressed_genes(threshold_p=1.0)

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]))


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes_de_marker_panel_threshold_logfc.json")
def test_identify_over_expressed_genes_de_threshold_logfc(
    pbmc3k_dense_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_dense_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    cellchat.identify_over_expressed_genes(
        threshold_logfc=1.0,
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]))


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes_de_marker_panel_only_pos_false.json")
def test_identify_over_expressed_genes_de_only_pos_false(
    pbmc3k_dense_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_dense_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    cellchat.identify_over_expressed_genes(
        only_pos=False,
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]))


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes_de_marker_panel_inplace_false.json")
def test_identify_over_expressed_genes_de_inplace_false(
    pbmc3k_dense_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_dense_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    returned_features = cellchat.identify_over_expressed_genes(
        inplace=False,
        threshold_p=1.0,
    )

    assert cellchat.selected_features is None
    assert cellchat.selected_features_df is None
    assert returned_features is not None

    observed = {
        "returned_features": feature_values_sorted(returned_features),
    }

    assert_compare(observed["returned_features"], ground_truth["returned_features"])


def test_identify_over_expressed_genes_no_de_dense_and_sparse_match(
    pbmc3k_dense_adata,
    pbmc3k_sparse_adata,
    pbmc3k_feature_panel,
):
    dense = make_cellchat(pbmc3k_dense_adata)
    sparse_cellchat = make_cellchat(pbmc3k_sparse_adata)

    dense.subset_data(features=pbmc3k_feature_panel)
    sparse_cellchat.subset_data(features=pbmc3k_feature_panel)

    dense.identify_over_expressed_genes(
        do_differential_expression=False,
        min_cells=10,
    )
    sparse_cellchat.identify_over_expressed_genes(
        do_differential_expression=False,
        min_cells=10,
    )

    assert_compare(selected_features_sorted(dense), selected_features_sorted(sparse_cellchat))


def test_identify_over_expressed_genes_de_dense_and_sparse_match(
    pbmc3k_dense_adata,
    pbmc3k_sparse_adata,
    pbmc3k_feature_panel,
):
    dense = make_cellchat(pbmc3k_dense_adata)
    sparse = make_cellchat(pbmc3k_sparse_adata)

    dense.subset_data(features=pbmc3k_feature_panel)
    sparse.subset_data(features=pbmc3k_feature_panel)

    dense.identify_over_expressed_genes(threshold_p=1.0)
    sparse.identify_over_expressed_genes(threshold_p=1.0)

    assert_compare(selected_features_sorted(dense), selected_features_sorted(sparse))
    assert_compare(feature_table(dense.selected_features_df), feature_table(sparse.selected_features_df))


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes_de_marker_panel_default_p.json")
def test_identify_over_expressed_genes_de_default_threshold_p(
    pbmc3k_dense_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    """threshold_p at the real default (0.05) — never exercised by any other fixture."""
    cellchat = make_cellchat(pbmc3k_dense_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    cellchat.identify_over_expressed_genes()

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]))


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes_de_marker_panel_threshold_percent.json")
def test_identify_over_expressed_genes_de_threshold_percent(
    pbmc3k_dense_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    """threshold_percent_expressing on real data with realistic zero-inflation."""
    cellchat = make_cellchat(pbmc3k_dense_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    cellchat.identify_over_expressed_genes(
        threshold_percent_expressing=10.0,
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]))


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes_de_full_gene_set.json")
def test_identify_over_expressed_genes_de_full_gene_set(
    pbmc3k_dense_adata,
    ground_truth,
):
    """Full gene set (no features= restriction): exercises chunked Mann-Whitney at scale."""
    cellchat = make_cellchat(pbmc3k_dense_adata)
    cellchat.db = load_cellchat_db("human")
    cellchat.subset_data()
    cellchat.identify_over_expressed_genes()

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]))


def test_identify_over_expressed_genes_condition_mode_runs_per_group(
    pbmc3k_dense_adata,
    pbmc3k_feature_panel,
):
    conditioned = with_condition_column(pbmc3k_dense_adata)
    cellchat = make_cellchat(conditioned, sample_column="condition")

    cellchat.subset_data(features=pbmc3k_feature_panel)
    cellchat.identify_over_expressed_genes(
        positive_samples=["cond_a"],
        only_pos=False,
        threshold_p=1.0,
    )

    assert cellchat.selected_features is not None
    assert cellchat.selected_features_df is not None
    observed_groups = set(cellchat.selected_features_df["group"].astype(str))
    expected_groups = set(conditioned.obs["cell_type"].astype(str).unique())
    assert observed_groups <= expected_groups


def test_identify_over_expressed_genes_condition_mode_can_ignore_groups(
    pbmc3k_dense_adata,
    pbmc3k_feature_panel,
):
    conditioned = with_condition_column(pbmc3k_dense_adata)
    cellchat = make_cellchat(conditioned, sample_column="condition")

    cellchat.subset_data(features=pbmc3k_feature_panel)
    cellchat.identify_over_expressed_genes(
        positive_samples=["cond_a"],
        ignore_groups_for_differential_expression=True,
        only_pos=False,
        threshold_p=1.0,
    )

    assert cellchat.selected_features is not None
    assert cellchat.selected_features_df is not None
    observed_groups = set(cellchat.selected_features_df["group"].astype(str))
    assert observed_groups <= {"cond_a"}
