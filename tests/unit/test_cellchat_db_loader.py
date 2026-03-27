from __future__ import annotations

import pytest

from py_cellchat.database import CellChatDB, load_cellchat_db

pytestmark = [pytest.mark.unit]

# ══════════════════════════════════════════════════════════════════════════════
# Input validation — no CSV files needed
# ══════════════════════════════════════════════════════════════════════════════

def test_load_cellchat_db_invalid_species_errors():
    with pytest.raises(ValueError, match="narwhal"):
        CellChatDB.load("narwhal")


def test_load_cellchat_db_case_insensitive():
    # Should raise for unknown, not because of case — but "HUMAN" should be
    # treated as "human".  Only check no ValueError about species name.
    db = CellChatDB.load("HUMAN")
    assert db.metadata["species"] == "human"


# ══════════════════════════════════════════════════════════════════════════════
# Human DB structure
# ══════════════════════════════════════════════════════════════════════════════

def test_load_cellchat_db_human_structure():
    db = load_cellchat_db("human")
    
    assert db.metadata["species"] == "human"
    
    assert len(db.interaction_input) > 1000
    for col in ("ligand", "receptor", "annotation"):
        assert col in db.interaction_input.columns, f"Missing column: {col}"
    # Index should be string-like (interaction names), not a plain RangeIndex.
    assert db.interaction_input.index.dtype == object
    
    # complex index comes from the "index" column in the CSV; must be strings.
    assert db.complex_input.index.dtype == object
    # At least some entries exist.
    assert len(db.complex_input) > 0
    
    assert db.cofactor_input.index.dtype == object
    cofactor_gene_cols = [c for c in db.cofactor_input.columns if c.startswith("cofactor")]
    assert len(cofactor_gene_cols) > 0
    
    assert "Symbol" in db.gene_info.columns
    assert db.gene_info["Symbol"].dropna().str.len().gt(0).all()


# ══════════════════════════════════════════════════════════════════════════════
# Mouse DB structure
# ══════════════════════════════════════════════════════════════════════════════

def test_load_cellchat_db_mouse_structure():
    db = load_cellchat_db("mouse")
    
    assert db.metadata["species"] == "mouse"
    
    assert len(db.interaction_input) > 1000
    for col in ("ligand", "receptor", "annotation"):
        assert col in db.interaction_input.columns, f"Missing column: {col}"
    # Index should be string-like (interaction names), not a plain RangeIndex.
    assert db.interaction_input.index.dtype == object
    
    # complex index comes from the "index" column in the CSV; must be strings.
    assert db.complex_input.index.dtype == object
    # At least some entries exist.
    assert len(db.complex_input) > 0
    
    assert db.cofactor_input.index.dtype == object
    cofactor_gene_cols = [c for c in db.cofactor_input.columns if c.startswith("cofactor")]
    assert len(cofactor_gene_cols) > 0
    
    assert "Symbol" in db.gene_info.columns
    assert db.gene_info["Symbol"].dropna().str.len().gt(0).all()


# ══════════════════════════════════════════════════════════════════════════════
# Zebrafish DB structure
# ══════════════════════════════════════════════════════════════════════════════

def test_load_cellchat_db_zebrafish_structure():
    db = load_cellchat_db("zebrafish")
    
    assert db.metadata["species"] == "zebrafish"
    
    assert len(db.interaction_input) > 1000
    for col in ("ligand", "receptor", "annotation"):
        assert col in db.interaction_input.columns, f"Missing column: {col}"
    # Index should be string-like (interaction names), not a plain RangeIndex.
    assert db.interaction_input.index.dtype == object
    
    # complex index comes from the "index" column in the CSV; must be strings.
    assert db.complex_input.index.dtype == object
    # At least some entries exist.
    assert len(db.complex_input) > 0
    
    assert db.cofactor_input.index.dtype == object
    cofactor_gene_cols = [c for c in db.cofactor_input.columns if c.startswith("cofactor")]
    assert len(cofactor_gene_cols) > 0
    
    assert "Symbol" in db.gene_info.columns
    assert db.gene_info["Symbol"].dropna().str.len().gt(0).all()
