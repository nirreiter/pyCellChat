from __future__ import annotations

from typing import Any, Sequence, cast

import numpy as np
import pandas as pd
import anndata as ad
import pytest

from py_cellchat import CellChat, identify_over_expressed_genes

from ..test_util import assert_compare, make_cellchat, with_condition_column

# ── helpers ────────────────────────────────────────────────────────────────────

def feature_values_sorted(values: pd.Series | pd.Index | np.ndarray | Sequence[str]) -> list[str]:
    if isinstance(values, pd.Series) or isinstance(values, pd.Index) or isinstance(values, np.ndarray):
        return sorted(values.astype(str).tolist())
    return sorted([str(value) for value in values])


def selected_features_sorted(cellchat: CellChat) -> list[str]:
    assert cellchat.selected_features is not None
    return sorted(cellchat.selected_features.astype(str).tolist())


def feature_table(feature_table: pd.DataFrame | None) -> list[tuple]:
	if feature_table is None or feature_table.empty:
		return []

	normalized = feature_table.loc[:, ["group", "feature", "logfc"]].copy()
	normalized["group"] = normalized["group"].astype(str)
	normalized["feature"] = normalized["feature"].astype(str)
	normalized["logfc"] = normalized["logfc"].astype(float).round(5) # 5 decimal places matches R settings
	normalized = normalized.sort_values(["group", "feature", "logfc"], kind="stable")
	return list(normalized.itertuples(index=False, name=None))

def feature_table_from_ground_truth(feature_table: list[dict[str, object]]) -> list[tuple]:
	return [(row["group"], row["feature"], row["logFC"]) for row in feature_table]


pytestmark = [
    pytest.mark.r_script("r_scripts/generate_iog_ground_truth.R"),
    pytest.mark.synthetic,
    pytest.mark.unit,
]

ALL_SYNTHETIC_FEATURES = ["g1", "g2", "g3", "g4", "g5", "g6", "g7"]

# ── synthetic ────────────────────────────────────────────────────────────────────

@pytest.mark.synthetic
def test_identify_over_expressed_genes_requires_subset_data_first(synthetic_grouped_adata):
    cellchat = make_cellchat(synthetic_grouped_adata)

    with pytest.raises(ValueError, match=r"Must call CellChat.subset_data\(\) first"):
        identify_over_expressed_genes(cellchat)


@pytest.mark.synthetic
def test_identify_over_expressed_genes_requires_positive_samples_when_ignoring_groups(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)
    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)

    with pytest.raises(ValueError, match="positive_samples is None"):
        identify_over_expressed_genes(cellchat, ignore_groups_for_de=True)


@pytest.mark.synthetic
def test_identify_over_expressed_genes_requires_sample_column_for_positive_samples(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)
    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cast(Any, cellchat).sample_col = None

    with pytest.raises(ValueError, match="No sample column"):
        identify_over_expressed_genes(cellchat, positive_samples=["s1"])


@pytest.mark.synthetic
def test_identify_over_expressed_genes_updates_python_state(synthetic_grouped_adata):
    cellchat = CellChat(synthetic_grouped_adata, group_by_column="cell_type")
    cellchat.subset_data()
    identify_over_expressed_genes(cellchat, do_differential_expression=False)

    assert cellchat.selected_features is not None
    assert cellchat.selected_features_df is None

    cellchat = CellChat(synthetic_grouped_adata, group_by_column="cell_type")
    cellchat.subset_data()
    identify_over_expressed_genes(cellchat, do_differential_expression=True)
    
    assert cellchat.selected_features is not None
    assert len(cellchat.selected_features) > 0
    assert cellchat.selected_features_df is not None
    assert len(cellchat.selected_features_df) > 0


@pytest.ground_truth_parameterize(  # pyright: ignore[reportAttributeAccessIssue]
    min_cells=("min.cells", [2, 3]),
)
@pytest.mark.synthetic
def test_identify_over_expressed_genes_no_de_synthetic(
    synthetic_grouped_adata,
    min_cells: int,
    ground_truth,
):
    cellchat = make_cellchat(synthetic_grouped_adata)
    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    identify_over_expressed_genes(cellchat, 
        do_differential_expression=False,
        min_cells=min_cells,
    )
    observed = {
        "selected_features": selected_features_sorted(cellchat),
    }

    assert_compare(observed["selected_features"], set(ground_truth["selected_features"]))


@pytest.ground_truth_parameterize(  # pyright: ignore[reportAttributeAccessIssue]
    threshold_percent_expressing=("thresh.pc", [0, 25, 50]),
    threshold_logfc=("thresh.fc", [0, 1, 2]),
    threshold_p=("thresh.p", [0, 0.05, 0.1]),
    only_pos=("only.pos", [True, False]),
)
@pytest.mark.synthetic
def test_identify_over_expressed_genes_de_synthetic(
    synthetic_grouped_adata,
    ground_truth,
    threshold_percent_expressing,
    threshold_logfc,
    threshold_p,
    only_pos,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    identify_over_expressed_genes(cellchat, 
        do_differential_expression=True, 
        threshold_percent_expressing=threshold_percent_expressing,
        threshold_logfc=threshold_logfc,
        threshold_p=threshold_p,
        only_pos=only_pos,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    # assert_compare(observed["selected_features"], set(ground_truth["selected_features"]))
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


@pytest.mark.ground_truth("synthetic/identify_over_expressed_genes/identify_over_expressed_genes_de_feature_subset.json")
def test_identify_over_expressed_genes_feature_subset(
    synthetic_grouped_adata,
    ground_truth,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    identify_over_expressed_genes(cellchat, 
        features=["g1", "g4", "g7", "missing"],
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], set(ground_truth["selected_features"]))
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


@pytest.mark.ground_truth("synthetic/identify_over_expressed_genes/identify_over_expressed_genes_de_inplace_false.json")
def test_identify_over_expressed_genes_inplace_false(
    synthetic_grouped_adata,
    ground_truth,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    returned_features = identify_over_expressed_genes(cellchat, 
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


def test_identify_over_expressed_genes_no_de_inplace_false_returns_features_only(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    returned_features = identify_over_expressed_genes(cellchat, 
        inplace=False,
        do_differential_expression=False,
        min_cells=2,
    )

    assert cellchat.selected_features is None
    assert cellchat.selected_features_df is None
    assert returned_features is not None
    assert_compare(feature_values_sorted(returned_features), ["g1", "g2", "g3", "g4", "g5", "g6"])


def test_identify_over_expressed_genes_accepts_index_features(synthetic_grouped_adata):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    identify_over_expressed_genes(cellchat, 
        features=pd.Index(["g1", "g4", "g7", "missing"]),
        threshold_p=1.0,
    )

    assert_compare(selected_features_sorted(cellchat), ["g1", "g4", "g7"])


def test_identify_over_expressed_genes_can_ignore_groups_for_condition_comparison(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=["g5", "g6"])
    identify_over_expressed_genes(cellchat, 
        positive_samples=["s1"],
        ignore_groups_for_de=True,
        threshold_p=1.0,
    )

    assert_compare(selected_features_sorted(cellchat), ["g5"])
    obs_ft = feature_table(cellchat.selected_features_df)
    expected_ft = [("s1", "g5", 4.62011)]
    assert_compare(obs_ft, expected_ft, is_numeric = True)


def test_identify_over_expressed_genes_dense_and_sparse_match(synthetic_grouped_adata, synthetic_sparse_adata):
    dense = make_cellchat(synthetic_grouped_adata)
    sparse = make_cellchat(synthetic_sparse_adata)

    dense.subset_data(features=ALL_SYNTHETIC_FEATURES)
    sparse.subset_data(features=ALL_SYNTHETIC_FEATURES)

    identify_over_expressed_genes(dense, threshold_p=1.0)
    identify_over_expressed_genes(sparse, threshold_p=1.0)

    assert_compare(selected_features_sorted(dense), selected_features_sorted(sparse))
    assert_compare(feature_table(dense.selected_features_df), feature_table(sparse.selected_features_df), is_numeric = True)



# ══════════════════════════════════════════════════════════════════════════════
# pbmc3k
# ══════════════════════════════════════════════════════════════════════════════

pytestmark = [
    pytest.mark.r_script("r_scripts/generate_iog_ground_truth.R"),
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


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes/identify_over_expressed_genes_no_de_marker_panel.json")
def test_identify_over_expressed_genes_no_de(
    pbmc3k_sparse_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_sparse_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    identify_over_expressed_genes(cellchat, 
        do_differential_expression=False,
        min_cells=10,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
    }

    assert_compare(observed["selected_features"], set(ground_truth["selected_features"]))


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes/identify_over_expressed_genes_de_marker_panel.json")
def test_identify_over_expressed_genes_de(
    pbmc3k_sparse_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_sparse_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    identify_over_expressed_genes(cellchat, threshold_p=1.0)

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], set(ground_truth["selected_features"]))
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes/identify_over_expressed_genes_de_marker_panel_threshold_logfc.json")
def test_identify_over_expressed_genes_de_threshold_logfc(
    pbmc3k_sparse_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_sparse_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    identify_over_expressed_genes(cellchat, 
        threshold_logfc=1.0,
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], set(ground_truth["selected_features"]))
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes/identify_over_expressed_genes_de_marker_panel_only_pos_false.json")
def test_identify_over_expressed_genes_de_only_pos_false(
    pbmc3k_sparse_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_sparse_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    identify_over_expressed_genes(cellchat, 
        only_pos=False,
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], set(ground_truth["selected_features"]))
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes/identify_over_expressed_genes_de_marker_panel_inplace_false.json")
def test_identify_over_expressed_genes_de_inplace_false(
    pbmc3k_sparse_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    cellchat = make_cellchat(pbmc3k_sparse_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    returned_features = identify_over_expressed_genes(cellchat, 
        inplace=False,
        threshold_p=1.0,
    )

    assert cellchat.selected_features is None
    assert cellchat.selected_features_df is None
    assert returned_features is not None

    observed = {
        "returned_features": feature_values_sorted(returned_features),
    }

    assert_compare(observed["returned_features"], set(ground_truth["returned_features"]))


def test_identify_over_expressed_genes_no_de_dense_and_sparse_match(
    pbmc3k_dense_adata,
    pbmc3k_sparse_adata,
    pbmc3k_feature_panel,
):
    dense_cellchat = make_cellchat(pbmc3k_dense_adata)
    sparse_cellchat = make_cellchat(pbmc3k_sparse_adata)

    dense_cellchat.subset_data(features=pbmc3k_feature_panel)
    sparse_cellchat.subset_data(features=pbmc3k_feature_panel)

    identify_over_expressed_genes(
        dense_cellchat,
        do_differential_expression=False,
        min_cells=10,
    )
    identify_over_expressed_genes(sparse_cellchat, 
        do_differential_expression=False,
        min_cells=10,
    )

    assert_compare(selected_features_sorted(dense_cellchat), selected_features_sorted(sparse_cellchat))


def test_identify_over_expressed_genes_de_dense_and_sparse_match(
    pbmc3k_dense_adata,
    pbmc3k_sparse_adata,
    pbmc3k_feature_panel,
):
    dense_cellchat = make_cellchat(pbmc3k_dense_adata)
    sparse_cellchat = make_cellchat(pbmc3k_sparse_adata)

    dense_cellchat.subset_data(features=pbmc3k_feature_panel)
    sparse_cellchat.subset_data(features=pbmc3k_feature_panel)

    identify_over_expressed_genes(dense_cellchat, threshold_p=1.0)
    identify_over_expressed_genes(sparse_cellchat, threshold_p=1.0)

    assert_compare(selected_features_sorted(dense_cellchat), selected_features_sorted(sparse_cellchat))
    assert_compare(feature_table(dense_cellchat.selected_features_df), feature_table(sparse_cellchat.selected_features_df), is_numeric = True)


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes/identify_over_expressed_genes_de_marker_panel_default_p.json")
def test_identify_over_expressed_genes_de_default_threshold_p(
    pbmc3k_sparse_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    """threshold_p at the real default (0.05) — never exercised by any other fixture."""
    cellchat = make_cellchat(pbmc3k_sparse_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    identify_over_expressed_genes(cellchat, )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], set(ground_truth["selected_features"]))
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes/identify_over_expressed_genes_de_marker_panel_threshold_percent.json")
def test_identify_over_expressed_genes_de_threshold_percent(
    pbmc3k_sparse_adata,
    pbmc3k_feature_panel,
    ground_truth,
):
    """threshold_percent_expressing on real data with realistic zero-inflation."""
    cellchat = make_cellchat(pbmc3k_sparse_adata)

    cellchat.subset_data(features=pbmc3k_feature_panel)
    identify_over_expressed_genes(cellchat, 
        threshold_percent_expressing=10.0,
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], set(ground_truth["selected_features"]))
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


@pytest.mark.ground_truth("pbmc3k_benchmark/identify_over_expressed_genes/identify_over_expressed_genes_de_full_gene_set.json")
def test_identify_over_expressed_genes_de_full_gene_set(
    pbmc3k_sparse_adata,
    ground_truth,
):
    """Full gene set (no features= restriction): exercises chunked Mann-Whitney at scale."""
    cellchat = make_cellchat(pbmc3k_sparse_adata)
    cellchat.load_database("human")
    cellchat.subset_data()
    identify_over_expressed_genes(cellchat, )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], set(ground_truth["selected_features"]))
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


def test_identify_over_expressed_genes_condition_mode_runs_per_group(
    pbmc3k_sparse_adata,
    pbmc3k_feature_panel,
):
    conditioned = with_condition_column(pbmc3k_sparse_adata)
    cellchat = make_cellchat(conditioned, sample_column="condition")

    cellchat.subset_data(features=pbmc3k_feature_panel)
    identify_over_expressed_genes(cellchat, 
        positive_samples=["cond_a"],
        only_pos=False,
        threshold_p=1.0,
    )

    assert cellchat.selected_features is not None
    assert cellchat.selected_features_df is not None
    observed_groups = set(cellchat.selected_features_df["group"].astype(str))
    expected_groups = set(conditioned.obs["cell_type"].astype(str).unique())
    assert observed_groups <= expected_groups
