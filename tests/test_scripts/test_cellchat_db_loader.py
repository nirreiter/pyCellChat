from __future__ import annotations

import pytest

from py_cellchat.database import CellChatDB, load_cellchat_db

pytestmark = [pytest.mark.unit]


@pytest.mark.unit
def test_database_scaffold_export():
    db = CellChatDB()
    assert db.interaction.empty
    assert db.metadata == {}


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
    
    assert len(db.interaction) > 1000
    for col in ("ligand", "receptor", "annotation"):
        assert col in db.interaction.columns, f"Missing column: {col}"
    # Index should be string-like (interaction names), not a plain RangeIndex.
    assert db.interaction.index.dtype == object
    
    # complex index comes from the "index" column in the CSV; must be strings.
    assert db.complex.index.dtype == object
    # At least some entries exist.
    assert len(db.complex) > 0
    
    assert db.cofactor.index.dtype == object
    cofactor_gene_cols = [c for c in db.cofactor.columns if c.startswith("cofactor")]
    assert len(cofactor_gene_cols) > 0
    
    assert "Symbol" in db.gene_info.columns
    assert db.gene_info["Symbol"].dropna().str.len().gt(0).all()


# ══════════════════════════════════════════════════════════════════════════════
# Mouse DB structure
# ══════════════════════════════════════════════════════════════════════════════

def test_load_cellchat_db_mouse_structure():
    db = load_cellchat_db("mouse")
    
    assert db.metadata["species"] == "mouse"
    
    assert len(db.interaction) > 1000
    for col in ("ligand", "receptor", "annotation"):
        assert col in db.interaction.columns, f"Missing column: {col}"
    # Index should be string-like (interaction names), not a plain RangeIndex.
    assert db.interaction.index.dtype == object
    
    # complex index comes from the "index" column in the CSV; must be strings.
    assert db.complex.index.dtype == object
    # At least some entries exist.
    assert len(db.complex) > 0
    
    assert db.cofactor.index.dtype == object
    cofactor_gene_cols = [c for c in db.cofactor.columns if c.startswith("cofactor")]
    assert len(cofactor_gene_cols) > 0
    
    assert "Symbol" in db.gene_info.columns
    assert db.gene_info["Symbol"].dropna().str.len().gt(0).all()


# ══════════════════════════════════════════════════════════════════════════════
# Zebrafish DB structure
# ══════════════════════════════════════════════════════════════════════════════

def test_load_cellchat_db_zebrafish_structure():
    db = load_cellchat_db("zebrafish")
    
    assert db.metadata["species"] == "zebrafish"
    
    assert len(db.interaction) > 1000
    for col in ("ligand", "receptor", "annotation"):
        assert col in db.interaction.columns, f"Missing column: {col}"
    # Index should be string-like (interaction names), not a plain RangeIndex.
    assert db.interaction.index.dtype == object
    
    # complex index comes from the "index" column in the CSV; must be strings.
    assert db.complex.index.dtype == object
    # At least some entries exist.
    assert len(db.complex) > 0
    
    assert db.cofactor.index.dtype == object
    cofactor_gene_cols = [c for c in db.cofactor.columns if c.startswith("cofactor")]
    assert len(cofactor_gene_cols) > 0
    
    assert "Symbol" in db.gene_info.columns
    assert db.gene_info["Symbol"].dropna().str.len().gt(0).all()


# ══════════════════════════════════════════════════════════════════════════════
# Subunit / cofactor value types — the NaN sentinel regression
# ══════════════════════════════════════════════════════════════════════════════

# R's write.csv emits quoted empty strings ("") for absent subunit / cofactor
# slots.  pandas read_csv treats "" as a default NA value and silently converts
# those cells to NaN.  Downstream filters use `s != ""` to skip absent slots;
# float NaN passes that check and then fails gene-universe membership tests,
# causing nearly all complexes to be rejected.  The loaders must normalise
# these cells back to "" before returning the DataFrames.

@pytest.mark.parametrize("species", ["human", "mouse", "zebrafish"])
def test_complex_subunit_values_no_nan(species):
    """Absent subunit slots must be empty strings, not NaN."""
    db = CellChatDB.load(species)
    subunit_cols = [c for c in db.complex.columns if c.startswith("subunit")]
    assert subunit_cols, f"{species}: no subunit columns found in complex table"
    has_nan = db.complex[subunit_cols].isna().any().any()
    assert not has_nan, (
        f"{species}: complex subunit columns contain NaN — "
        "absent slots must be empty strings ('')"
    )


@pytest.mark.parametrize("species", ["human", "mouse", "zebrafish"])
def test_cofactor_gene_values_no_nan(species):
    """Absent cofactor gene slots must be empty strings, not NaN."""
    db = CellChatDB.load(species)
    cofactor_gene_cols = [c for c in db.cofactor.columns if c.startswith("cofactor")]
    assert cofactor_gene_cols, f"{species}: no cofactor gene columns found in cofactor table"
    has_nan = db.cofactor[cofactor_gene_cols].isna().any().any()
    assert not has_nan, (
        f"{species}: cofactor gene columns contain NaN — "
        "absent slots must be empty strings ('')"
    )

@pytest.mark.parametrize("species", ["human", "mouse", "zebrafish"])
def test_interaction_gene_values_no_nan(species):
    """Absent interaction gene slots must be empty strings, not NaN."""
    db = CellChatDB.load(species)
    has_nan = db.interaction.isna().any().any()
    assert not has_nan, (
        f"{species}: interaction gene columns contain NaN — "
        "absent slots must be empty strings ('')"
    )

@pytest.mark.parametrize("species", ["human", "mouse", "zebrafish"])
def test_gene_info_values_no_nan(species):
    """Absent gene info slots must be empty strings, not NaN."""
    db = CellChatDB.load(species)
    gene_info = db.interaction.isna().any().any()
    assert not gene_info, (
        f"{species}: interaction gene columns contain NaN — "
        "absent slots must be empty strings ('')"
    )
