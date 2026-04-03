from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from py_cellchat import CellChat

from ..test_util import assert_compare, make_cellchat, selected_features_sorted, feature_values_sorted, feature_table, feature_table_from_ground_truth


pytestmark = [
    # pytest.mark.r_script("r_scripts/generate_synthetic_ground_truths.R"),
    pytest.mark.synthetic,
    pytest.mark.unit,
]

ALL_SYNTHETIC_FEATURES = ["g1", "g2", "g3", "g4", "g5", "g6", "g7"]


def test_identify_over_expressed_genes_updates_python_state(synthetic_grouped_adata):
    cellchat = CellChat(synthetic_grouped_adata, group_by_column="cell_type")
    cellchat.subset_data()
    cellchat.identify_over_expressed_genes(do_differential_expression=False)

    assert cellchat.selected_features is not None
    assert cellchat.selected_features_df is None

    cellchat = CellChat(synthetic_grouped_adata, group_by_column="cell_type")
    cellchat.subset_data()
    cellchat.identify_over_expressed_genes(do_differential_expression=True)
    
    assert cellchat.selected_features is not None
    assert len(cellchat.selected_features) > 0
    assert cellchat.selected_features_df is not None
    assert len(cellchat.selected_features_df) > 0


# @pytest.mark.ground_truth(
#     "synthetic/identify_over_expressed_genes_no_de.json",
#     "synthetic/identify_over_expressed_genes_no_de_min_cells_3.json",
# )
def test_identify_over_expressed_genes_no_de(synthetic_grouped_adata):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cellchat.identify_over_expressed_genes(
        do_differential_expression=False,
        min_cells=2,
    )
    observed = {
        "selected_features": selected_features_sorted(cellchat),
    }

    assert_compare(observed["selected_features"], ["g1", "g2", "g3", "g4", "g5", "g6"])
    
    """min_cells at a second value (3) excludes g3 which is expressed in only 2 cells."""
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cellchat.identify_over_expressed_genes(
        do_differential_expression=False,
        min_cells=3,
    )
    observed = {
        "selected_features": selected_features_sorted(cellchat),
    }

    assert_compare(observed["selected_features"], ["g1", "g2", "g4", "g5", "g6"])


@pytest.mark.ground_truth("synthetic/identify_over_expressed_genes_de_p_1.0.json")
def test_identify_over_expressed_genes_de(
    synthetic_grouped_adata,
    ground_truth,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cellchat.identify_over_expressed_genes(threshold_p=1.0)

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


@pytest.mark.ground_truth("synthetic/identify_over_expressed_genes_de_feature_subset.json")
def test_identify_over_expressed_genes_feature_subset(
    synthetic_grouped_adata,
    ground_truth,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cellchat.identify_over_expressed_genes(
        features=["g1", "g4", "g7", "missing"],
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


@pytest.mark.ground_truth("synthetic/identify_over_expressed_genes_de_threshold_percent.json")
def test_identify_over_expressed_genes_threshold_percent(
    synthetic_grouped_adata,
    ground_truth,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cellchat.identify_over_expressed_genes(
        threshold_percent_expressing=30.0,
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


@pytest.mark.ground_truth("synthetic/identify_over_expressed_genes_de_threshold_logfc.json")
def test_identify_over_expressed_genes_threshold_logfc(
    synthetic_grouped_adata,
    ground_truth,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cellchat.identify_over_expressed_genes(
        threshold_logfc=2.0,
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


@pytest.mark.ground_truth("synthetic/identify_over_expressed_genes_de_threshold_p_zero.json")
def test_identify_over_expressed_genes_threshold_p_zero(
    synthetic_grouped_adata,
    ground_truth,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cellchat.identify_over_expressed_genes(threshold_p=0.0)

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


@pytest.mark.ground_truth("synthetic/identify_over_expressed_genes_de_only_pos_false.json")
def test_identify_over_expressed_genes_only_pos_false(
    synthetic_grouped_adata,
    ground_truth,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cellchat.identify_over_expressed_genes(
        only_pos=False,
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)


@pytest.mark.ground_truth("synthetic/identify_over_expressed_genes_de_inplace_false.json")
def test_identify_over_expressed_genes_inplace_false(
    synthetic_grouped_adata,
    ground_truth,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
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


def test_identify_over_expressed_genes_no_de_inplace_false_returns_features_only(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    returned_features = cellchat.identify_over_expressed_genes(
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
    cellchat.identify_over_expressed_genes(
        features=pd.Index(["g1", "g4", "g7", "missing"]),
        threshold_p=1.0,
    )

    assert_compare(selected_features_sorted(cellchat), ["g1", "g4", "g7"])


def test_identify_over_expressed_genes_positive_samples_filters_within_each_group(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=["g5", "g6"])
    cellchat.identify_over_expressed_genes(
        positive_samples=["s1"],
        threshold_p=1.0,
    )

    assert_compare(selected_features_sorted(cellchat), ["g5" ,"g5"])
    obs_ft = feature_table(cellchat.selected_features_df)
    expected_ft = [("A", "g5", 4.6201), ("B", "g5", 4.6201)]
    assert_compare(obs_ft, expected_ft, is_numeric = True)


def test_identify_over_expressed_genes_can_ignore_groups_for_condition_comparison(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=["g5", "g6"])
    cellchat.identify_over_expressed_genes(
        positive_samples=["s1"],
        ignore_groups_for_differential_expression=True,
        threshold_p=1.0,
    )

    assert_compare(selected_features_sorted(cellchat), ["g5"])
    obs_ft = feature_table(cellchat.selected_features_df)
    expected_ft = [("s1", "g5", 4.6201)]
    assert_compare(obs_ft, expected_ft, is_numeric = True)


def test_identify_over_expressed_genes_requires_subset_data_first(synthetic_grouped_adata):
    cellchat = make_cellchat(synthetic_grouped_adata)

    with pytest.raises(ValueError, match=r"Must call CellChat.subset_data\(\) first"):
        cellchat.identify_over_expressed_genes()


def test_identify_over_expressed_genes_requires_positive_samples_when_ignoring_groups(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)
    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)

    with pytest.raises(ValueError, match="positive_samples is None"):
        cellchat.identify_over_expressed_genes(
            ignore_groups_for_differential_expression=True,
        )


def test_identify_over_expressed_genes_positive_samples_inplace_false(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=["g5", "g6"])
    returned = cellchat.identify_over_expressed_genes(
        positive_samples=["s1"],
        inplace=False,
        threshold_p=1.0,
    )

    assert cellchat.selected_features is None
    assert cellchat.selected_features_df is None
    assert returned is not None
    assert len(returned) > 0


def test_identify_over_expressed_genes_ignore_groups_inplace_false(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=["g5", "g6"])
    returned = cellchat.identify_over_expressed_genes(
        positive_samples=["s1"],
        ignore_groups_for_differential_expression=True,
        inplace=False,
        threshold_p=1.0,
    )

    assert cellchat.selected_features is None
    assert cellchat.selected_features_df is None
    assert returned is not None


def test_identify_over_expressed_genes_feature_table_has_numeric_columns(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cellchat.identify_over_expressed_genes(threshold_p=1.0)

    df = cellchat.selected_features_df
    assert df is not None
    assert not df.empty
    assert pd.api.types.is_float_dtype(df["logfc"])
    assert pd.api.types.is_float_dtype(df["pvalue"])
    assert pd.api.types.is_float_dtype(df["padj"])
    assert ((df["pvalue"] >= 0) & (df["pvalue"] <= 1)).all()
    assert ((df["padj"] >= 0) & (df["padj"] <= 1)).all()
    assert np.isfinite(df["logfc"]).all()


def test_identify_over_expressed_genes_positive_samples_with_only_pos_false(
    synthetic_grouped_adata,
):
    """only_pos=False + positive_samples: down-regulated features (direction=-1) are included."""
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=["g5", "g6"])
    cellchat.identify_over_expressed_genes(
        positive_samples=["s1"],
        only_pos=False,
        threshold_p=1.0,
    )

    obs_ft = feature_table(cellchat.selected_features_df)
    expected_ft = [("A", "g5", 4.6201), ("A", "g6", -4.6201), ("B", "g5", 4.6201), ("B", "g6", -4.6201)]
    assert_compare(obs_ft, expected_ft, is_numeric = True)


def test_identify_over_expressed_genes_requires_sample_column_for_positive_samples(
    synthetic_grouped_adata,
):
    cellchat = make_cellchat(synthetic_grouped_adata)
    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cast(Any, cellchat).sample_col = None

    with pytest.raises(ValueError, match="No sample column"):
        cellchat.identify_over_expressed_genes(positive_samples=["s1"])


def test_identify_over_expressed_genes_dense_and_sparse_match(synthetic_grouped_adata, synthetic_sparse_adata):
    dense = make_cellchat(synthetic_grouped_adata)
    sparse = make_cellchat(synthetic_sparse_adata)

    dense.subset_data(features=ALL_SYNTHETIC_FEATURES)
    sparse.subset_data(features=ALL_SYNTHETIC_FEATURES)

    dense.identify_over_expressed_genes(threshold_p=1.0)
    sparse.identify_over_expressed_genes(threshold_p=1.0)

    assert_compare(selected_features_sorted(dense), selected_features_sorted(sparse))
    assert_compare(feature_table(dense.selected_features_df), feature_table(sparse.selected_features_df), is_numeric = True)


@pytest.mark.ground_truth("synthetic/identify_over_expressed_genes_de_threshold_logfc_half.json")
def test_identify_over_expressed_genes_threshold_logfc_half(
    synthetic_grouped_adata,
    ground_truth,
):
    """threshold_logfc at 0.5 (second value beyond the existing test at 2.0)."""
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cellchat.identify_over_expressed_genes(
        threshold_logfc=0.5,
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)



@pytest.mark.ground_truth("synthetic/identify_over_expressed_genes_de_threshold_p_half.json")
def test_identify_over_expressed_genes_threshold_p_half(
    synthetic_grouped_adata,
    ground_truth,
):
    """threshold_p at an intermediate value (0.5) between the tested extremes of 0.0 and 1.0."""
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cellchat.identify_over_expressed_genes(threshold_p=0.5)

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)



@pytest.mark.ground_truth("synthetic/identify_over_expressed_genes_de_threshold_pct_and_logfc.json")
def test_identify_over_expressed_genes_threshold_percent_and_logfc(
    synthetic_grouped_adata,
    ground_truth,
):
    """threshold_percent_expressing and threshold_logfc applied together (AND logic)."""
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cellchat.identify_over_expressed_genes(
        threshold_percent_expressing=30.0,
        threshold_logfc=0.5,
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)



@pytest.mark.ground_truth("synthetic/identify_over_expressed_genes_de_only_pos_false_logfc.json")
def test_identify_over_expressed_genes_only_pos_false_with_logfc(
    synthetic_grouped_adata,
    ground_truth,
):
    """only_pos=False combined with a non-zero threshold_logfc."""
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=ALL_SYNTHETIC_FEATURES)
    cellchat.identify_over_expressed_genes(
        only_pos=False,
        threshold_logfc=0.5,
        threshold_p=1.0,
    )

    observed = {
        "selected_features": selected_features_sorted(cellchat),
        "feature_table": feature_table(cellchat.selected_features_df),
    }

    assert_compare(observed["selected_features"], ground_truth["selected_features"])
    assert_compare(observed["feature_table"], feature_table_from_ground_truth(ground_truth["feature_table"]), is_numeric = True)
