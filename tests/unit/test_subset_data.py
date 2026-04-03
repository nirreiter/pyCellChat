from __future__ import annotations

import copy

import pytest
from scipy import sparse

from py_cellchat.database import extract_gene, load_cellchat_db, subset_db

from ..test_util import assert_compare, make_cellchat


pytestmark = [
    pytest.mark.r_script("r_scripts/generate_synthetic_ground_truths.R"),
    pytest.mark.unit,
]


@pytest.fixture(scope="module")
def human_db():
    """CellChatDB.human subsetted to the three protein-coding annotation categories."""
    return subset_db(load_cellchat_db("human"))


# ══════════════════════════════════════════════════════════════════════════════
# Existing parity test (explicit features, no DB) — synthetic adata
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.synthetic
@pytest.mark.ground_truth("synthetic/subset_data_explicit.json")
def test_subset_data(synthetic_grouped_adata, ground_truth):
    cellchat = make_cellchat(synthetic_grouped_adata)

    cellchat.subset_data(features=["g2", "missing", "g7"])
    assert cellchat.adata_signaling is not None
    observed = {
        "signaling_features": cellchat.adata_signaling.var_names.astype(str).tolist(),
    }

    assert_compare(observed["signaling_features"], ground_truth["signaling_features"])


# ══════════════════════════════════════════════════════════════════════════════
# DB-gene path
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.pbmc3k
def test_subset_data_db_genes_intersected(pbmc3k_adata, human_db):
    """subset_data() returns exactly the intersection of DB genes and adata genes."""
    cellchat = make_cellchat(pbmc3k_adata)
    cellchat.db = human_db

    cellchat.subset_data()

    adata_sig = cellchat.adata_signaling
    assert adata_sig is not None
    result_genes = set(adata_sig.var_names.astype(str))
    db_gene_set = set(extract_gene(human_db))
    adata_gene_set = set(pbmc3k_adata.var_names.astype(str))

    assert result_genes == db_gene_set & adata_gene_set
    assert len(result_genes) > 0  # pbmc3k shares genes with CellChatDB.human
    
    # pbmc3k has ~30k genes while CellChatDB has only a few hundred; result should be smaller.
    assert len(adata_sig.var_names) < len(pbmc3k_adata.var_names)


# ══════════════════════════════════════════════════════════════════════════════
# No DB — backward compat
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.synthetic
def test_subset_data_no_db_full_copy(synthetic_grouped_adata):
    # Backward compat: no DB → full AnnData copied into adata_signaling.
    cellchat = make_cellchat(synthetic_grouped_adata)
    assert cellchat.db is None

    cellchat.subset_data()

    assert cellchat.adata_signaling is not None
    result_genes = set(cellchat.adata_signaling.var_names.astype(str).tolist())
    assert result_genes == {"g1", "g2", "g3", "g4", "g5", "g6", "g7"}


# ══════════════════════════════════════════════════════════════════════════════
# Annotation sort side-effect
# ══════════════════════════════════════════════════════════════════════════════
# The adata gene names (g1-g7) do not match any CellChatDB gene symbols, so
# adata_signaling will be empty — but the sort side-effect on db.interaction
# happens before the gene intersection and is what these tests verify.

_CANONICAL_ORDER = [
    "Secreted Signaling",
    "ECM-Receptor",
    "Non-protein Signaling",
    "Cell-Cell Contact",
]


@pytest.mark.synthetic
def test_subset_data_annotation_sort_sideeffect(synthetic_grouped_adata, human_db):
    # Shuffle the interaction rows before attaching to ensure the sort is needed.
    # A new CellChatDB with a shuffled interaction is used so the
    # module-scoped human_db fixture is not mutated.
    db_copy = copy.copy(human_db)
    db_copy.interaction = human_db.interaction.sample(frac=1, random_state=42)

    cellchat = make_cellchat(synthetic_grouped_adata)
    cellchat.db = db_copy

    cellchat.subset_data()

    annotations = cellchat.db.interaction["annotation"].tolist()
    ranks = [_CANONICAL_ORDER.index(a) for a in annotations if a in _CANONICAL_ORDER]
    assert ranks == sorted(ranks)


@pytest.mark.synthetic
def test_subset_data_annotation_single_value_no_sort(synthetic_grouped_adata):
    # When only one annotation value is present the row order must not change.
    # Use the real DB filtered to a single category so this test never depends on
    # a hand-crafted synthetic DB.
    single_annotation_db = subset_db(load_cellchat_db("human"), search=["Secreted Signaling"])
    original_index = single_annotation_db.interaction.index.tolist()

    cellchat = make_cellchat(synthetic_grouped_adata)
    cellchat.db = single_annotation_db

    cellchat.subset_data()

    assert cellchat.db.interaction.index.tolist() == original_index


# ══════════════════════════════════════════════════════════════════════════════
# Explicit features path ignores DB
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.synthetic
def test_subset_data_features_ignores_db(synthetic_grouped_adata, human_db):
    # When features= is provided the DB gene set must not affect the result.
    cellchat = make_cellchat(synthetic_grouped_adata)
    cellchat.db = human_db

    cellchat.subset_data(features=["g5", "g6"])

    adata_sig = cellchat.adata_signaling
    assert adata_sig is not None
    assert set(adata_sig.var_names.astype(str)) == {"g5", "g6"}


# ══════════════════════════════════════════════════════════════════════════════
# Sparse input
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.pbmc3k
def test_subset_data_db_sparse_dense_parity(pbmc3k_adata, human_db):
    """Sparse and dense inputs produce identical signaling gene sets."""
    dense_adata = pbmc3k_adata.copy()
    sparse_adata = pbmc3k_adata.copy()
    sparse_adata.X = sparse.csr_matrix(sparse_adata.X)

    cellchat_dense = make_cellchat(dense_adata)
    cellchat_dense.db = human_db
    cellchat_dense.subset_data()

    cellchat_sparse = make_cellchat(sparse_adata)
    cellchat_sparse.db = human_db
    cellchat_sparse.subset_data()

    dense_sig = cellchat_dense.adata_signaling
    sparse_sig = cellchat_sparse.adata_signaling
    assert dense_sig is not None
    assert sparse_sig is not None
    assert set(dense_sig.var_names) == set(sparse_sig.var_names)
    assert sparse.issparse(sparse_sig.X)
