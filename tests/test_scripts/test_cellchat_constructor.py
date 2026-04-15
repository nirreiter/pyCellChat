from __future__ import annotations

import pandas as pd
import pytest

from py_cellchat import CellChat

from ..test_util import assert_compare, make_cellchat


@pytest.mark.synthetic
@pytest.mark.unit
@pytest.mark.r_script("r_scripts/generate_database_ground_truth.R")
@pytest.mark.ground_truth("synthetic/cellchat_constructor/constructor_summary.json")
def test_cellchat_constructor_shape_and_categories(synthetic_grouped_adata, ground_truth):
    cellchat = make_cellchat(synthetic_grouped_adata)

    observed = {
        "n_cells": cellchat.adata.n_obs,
        "n_genes": cellchat.adata.n_vars,
        "group_categories": cellchat.idents.cat.categories.astype(str).tolist(),
        "sample_categories": cellchat.adata.obs[cellchat.sample_col].cat.categories.astype(str).tolist(),
    }

    assert observed["n_cells"] == ground_truth["n_cells"]
    assert observed["n_genes"] == ground_truth["n_genes"]
    assert_compare(observed["group_categories"], ground_truth["group_categories"])
    assert_compare(observed["sample_categories"], ground_truth["sample_categories"])


@pytest.mark.pbmc3k
@pytest.mark.integration
def test_constructor_uses_obs_backed_metadata(
    pbmc3k_adata,
):
    cellchat = CellChat(pbmc3k_adata, group_by_column="cell_type")
    
    assert cellchat.adata.n_obs == pbmc3k_adata.n_obs
    assert cellchat.adata.n_vars == pbmc3k_adata.n_vars
    assert_compare(cellchat.idents.astype(str).to_numpy(), pbmc3k_adata.obs["cell_type"].astype(str).to_numpy())
    assert_compare(
        cellchat.idents.cat.categories.astype(str).to_numpy(),
        pbmc3k_adata.obs["cell_type"].cat.categories.astype(str).to_numpy(),
    )
    assert "counts" in cellchat.adata.layers
    assert cellchat.options == {"mode": "single", "datatype": "RNA"}
    assert cellchat.sample_col == "sample"
    assert isinstance(cellchat.adata.obs[cellchat.sample_col].dtype, pd.CategoricalDtype)
    assert len(cellchat.adata.obs[cellchat.sample_col].cat.categories) > 0
    assert isinstance(cellchat.idents, pd.Series)
    assert isinstance(cellchat.idents.dtype, pd.CategoricalDtype)
    assert len(cellchat.idents) > 0


@pytest.mark.pbmc3k
@pytest.mark.integration
def test_constructor_accepts_sparse_input(pbmc3k_sparse_adata):
    cellchat = CellChat(pbmc3k_sparse_adata, group_by_column="cell_type")
    assert cellchat.adata.n_obs == pbmc3k_sparse_adata.n_obs
    assert cellchat.options["mode"] == "single"


@pytest.mark.pbmc3k
@pytest.mark.integration
def test_constructor_prefers_samples_column_when_present(pbmc3k_adata):
    adata = pbmc3k_adata.copy()
    adata.obs["sample"] = pd.Categorical(["s1"] * adata.n_obs)

    cellchat = CellChat(adata, group_by_column="cell_type")

    assert cellchat.sample_col == "sample"
    assert list(cellchat.adata.obs[cellchat.sample_col].cat.categories) == ["s1"]
