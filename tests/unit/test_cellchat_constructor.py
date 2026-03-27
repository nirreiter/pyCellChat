from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from py_cellchat import CellChat, CellChatDB

from ..test_util import assert_compare, make_cellchat


pytestmark = [pytest.mark.unit]


@pytest.mark.synthetic
@pytest.mark.r_script("r_scripts/generate_synthetic_ground_truths.R")
@pytest.mark.ground_truth("synthetic/constructor_summary.json")
def test_cellchat_constructor_shape_and_categories(synthetic_grouped_adata, ground_truth):
    cellchat = make_cellchat(synthetic_grouped_adata)

    observed = {
        "n_cells": cellchat.adata.n_obs,
        "n_genes": cellchat.adata.n_vars,
        "group_categories": cellchat.idents.cat.categories.astype(str).tolist(),
        "sample_categories": cellchat.meta[cellchat.sample_col].cat.categories.astype(str).tolist(),
    }

    assert observed["n_cells"] == ground_truth["n_cells"]
    assert observed["n_genes"] == ground_truth["n_genes"]
    assert_compare(observed["group_categories"], ground_truth["group_categories"])
    assert_compare(observed["sample_categories"], ground_truth["sample_categories"])


def _object_state_snapshot(obj: object) -> dict[str, Any]:
    return {
        name: _snapshot_value(value)
        for name, value in vars(obj).items()
        if not name.startswith("_")
    }

def _snapshot_value(value: Any) -> Any:
    from numpy import ndarray
    from anndata import AnnData
    from pathlib import Path
    from scipy.sparse import issparse
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, AnnData):
        return {
            "type": "AnnData",
            "n_genes": int(value.n_vars),
            "n_cells": int(value.n_vars),
            "obs_columns": value.obs.columns.astype(str).tolist(),
            "var_columns": value.var.columns.astype(str).tolist(),
            "layers": sorted(str(name) for name in value.layers.keys()),
        }
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": value.columns.astype(str).tolist(),
        }
    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "length": int(value.shape[0]),
            "name": value.name,
        }
    if isinstance(value, ndarray):
        return {
            "type": "ndarray",
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
        }
    if issparse(value):
        return {
            "type": type(value).__name__,
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "nnz": int(value.nnz),
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": sorted(str(key) for key in value.keys()),
        }
    if isinstance(value, (list, tuple, set, pd.Index)):
        return {
            "type": type(value).__name__,
            "length": len(value),
        }
    return {"type": type(value).__name__}



@pytest.mark.pbmc3k
def test_constructor_uses_obs_backed_metadata(
    pbmc3k_adata,
    assert_object_state_keys,
):
    cellchat = CellChat(pbmc3k_adata, group_by_column="cell_type")
    state = _object_state_snapshot(cellchat)

    assert_object_state_keys(
        state,
        {
            "adata",
            "adata_signaling",
            "db",
            "experiment_type",
            "group_by_col",
            "images",
            "is_merged",
            "lr",
            "net",
            "netP",
            "options",
            "sample_col",
            "selected_features",
            "selected_features_df",
            "var_features",
        },
    )
    assert "meta" not in state
    assert "idents" not in state
    assert cellchat.meta is cellchat.adata.obs
    assert cellchat.idents.equals(cellchat.adata.obs[cellchat.group_by_col])
    assert cellchat.adata.n_obs == pbmc3k_adata.n_obs
    assert cellchat.adata.n_vars == pbmc3k_adata.n_vars
    assert state["adata"]["layers"] == ["counts"]
    assert cellchat.options == {"mode": "single", "datatype": "RNA"}
    assert cellchat.sample_col == "sample"
    assert isinstance(cellchat.meta[cellchat.sample_col].dtype, pd.CategoricalDtype)
    assert len(cellchat.meta[cellchat.sample_col].cat.categories) > 0
    assert isinstance(cellchat.idents.dtype, pd.CategoricalDtype)
    assert len(cellchat.idents.cat.categories) > 0


@pytest.mark.pbmc3k
def test_constructor_accepts_sparse_input(pbmc3k_sparse_adata):
    cellchat = CellChat(pbmc3k_sparse_adata, group_by_column="cell_type")
    assert cellchat.adata.n_obs == pbmc3k_sparse_adata.n_obs
    assert cellchat.options["mode"] == "single"


@pytest.mark.pbmc3k
def test_constructor_preserves_existing_category_order(pbmc3k_adata):
    adata = pbmc3k_adata.copy()
    ordered_categories = list(reversed(pd.unique(adata.obs["cell_type"].astype(str)).tolist()))
    adata.obs["ordered_groups"] = pd.Categorical(
        adata.obs["cell_type"].astype(str),
        categories=ordered_categories,
        ordered=True,
    )

    cellchat = CellChat(adata, group_by_column="ordered_groups")

    assert list(cellchat.idents.cat.categories) == ordered_categories
    assert list(cellchat.meta["ordered_groups"].cat.categories) == ordered_categories


@pytest.mark.pbmc3k
def test_constructor_prefers_samples_column_when_present(pbmc3k_adata):
    adata = pbmc3k_adata.copy()
    adata.obs["sample"] = pd.Categorical(["s1"] * adata.n_obs)

    cellchat = CellChat(adata, group_by_column="cell_type")

    assert cellchat.sample_col == "sample"
    assert list(cellchat.meta[cellchat.sample_col].cat.categories) == ["s1"]


def test_database_scaffold_export():
    db = CellChatDB()
    assert db.interaction_input.empty
    assert db.metadata == {}
