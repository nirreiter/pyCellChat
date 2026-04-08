from __future__ import annotations

import pandas as pd
import pytest

from py_cellchat.database import CellChatDB, load_cellchat_db, subset_db, extract_gene

from ..test_util import assert_compare

pytestmark = [pytest.mark.unit]


@pytest.fixture(scope="module")
def human_db():
    """Full CellChatDB.human loaded from bundled CSVs."""
    return load_cellchat_db("human")


# ── minimal DB used only for error-path tests ──────────────────────────────────
# Error tests verify ValueError conditions in the subset_db function itself,
# not behavior on real data, so a tiny synthetic DataFrame is appropriate here.

def _make_minimal_db() -> CellChatDB:
    return CellChatDB(interaction=pd.DataFrame({
        "ligand":     ["L1"],
        "receptor":   ["R1"],
        "annotation": ["Secreted Signaling"],
    }))


# ══════════════════════════════════════════════════════════════════════════════
# Default / annotation key
# ══════════════════════════════════════════════════════════════════════════════

def test_subset_db_annotation_default(human_db):
    result = subset_db(human_db)
    annotations = set(result.interaction["annotation"])
    assert "Non-protein Signaling" not in annotations
    assert annotations.issubset({"Secreted Signaling", "ECM-Receptor", "Cell-Cell Contact"})
    assert len(result.interaction) > 0


def test_subset_db_non_protein_true(human_db):
    result_default = subset_db(human_db)
    result_with_non_protein = subset_db(human_db, non_protein=True)
    # Including non-protein rows must produce at least as many interactions.
    assert len(result_with_non_protein.interaction) >= len(result_default.interaction)


def test_subset_db_custom_search(human_db):
    result = subset_db(human_db, search=["Secreted Signaling"])
    annotations = set(result.interaction["annotation"])
    assert annotations == {"Secreted Signaling"}
    assert len(result.interaction) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Custom key
# ══════════════════════════════════════════════════════════════════════════════

def test_subset_db_custom_key(human_db):
    # "CXCL" is a real pathway that exists in CellChatDB.human.
    result = subset_db(human_db, search=["CXCL"], key="pathway_name")
    assert set(result.interaction["pathway_name"]) == {"CXCL"}
    assert len(result.interaction) > 0


def test_subset_db_multi_key(human_db):
    result = subset_db(
        human_db,
        key=["annotation", "pathway_name"],
        search=[["Secreted Signaling"], ["CXCL"]],
    )
    assert set(result.interaction["annotation"]) == {"Secreted Signaling"}
    assert set(result.interaction["pathway_name"]) == {"CXCL"}


# ══════════════════════════════════════════════════════════════════════════════
# Immutability
# ══════════════════════════════════════════════════════════════════════════════

def test_subset_db_original_unchanged(human_db):
    original_len = len(human_db.interaction)
    subset_db(human_db)
    assert len(human_db.interaction) == original_len


# ══════════════════════════════════════════════════════════════════════════════
# Error paths — minimal DataFrame; testing error-handling code, not real data
# ══════════════════════════════════════════════════════════════════════════════

def test_subset_db_invalid_key_raises():
    db = _make_minimal_db()
    with pytest.raises(ValueError, match="not found in interaction"):
        subset_db(db, search=["foo"], key="nonexistent_column")


def test_subset_db_multi_key_search_not_list_raises():
    db = _make_minimal_db()
    with pytest.raises(ValueError, match="list of lists"):
        subset_db(
            db,
            key=["annotation", "ligand"],
            search=["Secreted Signaling", "L1"],  # flat list, not list-of-lists
        )


# ══════════════════════════════════════════════════════════════════════════════
# non_protein via explicit search
# ══════════════════════════════════════════════════════════════════════════════

def test_subset_db_non_protein_in_search_enables_flag(human_db):
    result = subset_db(
        human_db,
        search=["Secreted Signaling", "Non-protein Signaling"],
    )
    annotations = set(result.interaction["annotation"])
    assert "Secreted Signaling" in annotations
    assert "ECM-Receptor" not in annotations
    assert "Cell-Cell Contact" not in annotations
    # If the source DB contains Non-protein Signaling rows they must appear.
    if "Non-protein Signaling" in set(human_db.interaction["annotation"]):
        assert "Non-protein Signaling" in annotations


# ══════════════════════════════════════════════════════════════════════════════
# pbmc3k
# ══════════════════════════════════════════════════════════════════════════════


pytestmark = [
    pytest.mark.r_script("r_scripts/generate_database_ground_truth.R"),
    pytest.mark.pbmc3k,
    pytest.mark.integration,
]


@pytest.mark.ground_truth("pbmc3k_benchmark/subset_db/subset_db_default.json")
def test_subset_db_pbmc3k_default(pbmc3k_sparse_adata, ground_truth):
    result = subset_db(load_cellchat_db("human"))
    pbmc3k_genes = set(pbmc3k_sparse_adata.var_names.astype(str))

    observed = {
        "interaction_names": result.interaction.index.astype(str).tolist(),
        "pbmc3k_overlap_genes": sorted(set(extract_gene(result)) & pbmc3k_genes),
    }

    assert set(result.interaction["annotation"]) <= {
        "Secreted Signaling",
        "ECM-Receptor",
        "Cell-Cell Contact",
    }
    assert observed["pbmc3k_overlap_genes"]
    assert_compare(observed["interaction_names"], ground_truth["interaction_names"])
    assert_compare(observed["pbmc3k_overlap_genes"], ground_truth["pbmc3k_overlap_genes"])
