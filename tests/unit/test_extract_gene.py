from __future__ import annotations

import pandas as pd
import pytest

from py_cellchat.database import CellChatDB
from py_cellchat.database.extract import extract_gene, extract_gene_subset

pytestmark = [pytest.mark.unit]


# ── helpers ────────────────────────────────────────────────────────────────────

def make_gene_info(*symbols: str) -> pd.DataFrame:
    return pd.DataFrame({"Symbol": list(symbols)})


def make_complex_input(entries: dict[str, list[str]]) -> pd.DataFrame:
    """Build a complex DataFrame indexed by complex name.

    ``entries`` maps complex name -> list of subunit gene symbols (empty
    string means absent subunit, matching how the R CSVs are stored).
    """
    max_cols = max(len(v) for v in entries.values()) if entries else 0
    records = {}
    for name, subunits in entries.items():
        padded = subunits + [""] * (max_cols - len(subunits))
        records[name] = {f"subunit_{i + 1}": s for i, s in enumerate(padded)}
    return pd.DataFrame.from_dict(records, orient="index")


def make_cofactor_input(entries: dict[str, list[str]]) -> pd.DataFrame:
    """Build a cofactor DataFrame indexed by cofactor name."""
    max_cols = max(len(v) for v in entries.values()) if entries else 0
    records = {}
    for name, genes in entries.items():
        padded = genes + [""] * (max_cols - len(genes))
        records[name] = {f"cofactor_{i + 1}": g for i, g in enumerate(padded)}
    return pd.DataFrame.from_dict(records, orient="index")


# ══════════════════════════════════════════════════════════════════════════════
# extract_gene_subset
# ══════════════════════════════════════════════════════════════════════════════

def test_extract_gene_subset_all_symbols():
    gene_info = make_gene_info("TGFB1", "EGFR", "MET")
    complex_input = make_complex_input({})
    result = extract_gene_subset(["TGFB1", "EGFR"], complex_input, gene_info)
    assert sorted(result) == ["EGFR", "TGFB1"]


def test_extract_gene_subset_complex_expansion():
    gene_info = make_gene_info("TGFB1", "TGFBR1", "TGFBR2")
    complex_input = make_complex_input({"TGFb_receptor_1_2": ["TGFBR1", "TGFBR2"]})
    result = extract_gene_subset(["TGFb_receptor_1_2"], complex_input, gene_info)
    assert sorted(result) == ["TGFBR1", "TGFBR2"]


def test_extract_gene_subset_empty_subunit_filtered():
    gene_info = make_gene_info("TGFBR1")
    complex_input = make_complex_input({"SPARSE_COMPLEX": ["TGFBR1", ""]})
    result = extract_gene_subset(["SPARSE_COMPLEX"], complex_input, gene_info)
    assert result == ["TGFBR1"]


def test_extract_gene_subset_mixed():
    gene_info = make_gene_info("TGFB1", "TGFBR1", "TGFBR2")
    complex_input = make_complex_input({"TGFb_receptor_1_2": ["TGFBR1", "TGFBR2"]})
    result = extract_gene_subset(["TGFB1", "TGFb_receptor_1_2"], complex_input, gene_info)
    assert sorted(result) == ["TGFB1", "TGFBR1", "TGFBR2"]


def test_extract_gene_subset_unknown_dropped():
    # "GHOST" is neither in gene_info["Symbol"] nor in complex_input.index.
    gene_info = make_gene_info("TGFB1")
    complex_input = make_complex_input({})
    result = extract_gene_subset(["TGFB1", "GHOST"], complex_input, gene_info)
    assert result == ["TGFB1"]


def test_extract_gene_subset_deduplication():
    gene_info = make_gene_info("TGFB1", "EGFR")
    complex_input = make_complex_input({})
    result = extract_gene_subset(["TGFB1", "TGFB1", "EGFR"], complex_input, gene_info)
    assert result.count("TGFB1") == 1
    assert sorted(result) == ["EGFR", "TGFB1"]


# ══════════════════════════════════════════════════════════════════════════════
# extract_gene
# ══════════════════════════════════════════════════════════════════════════════

def _make_db(
    ligands: list[str],
    receptors: list[str],
    *,
    agonists: list[str] | None = None,
    co_a: list[str] | None = None,
    complex_entries: dict[str, list[str]] | None = None,
    cofactor_entries: dict[str, list[str]] | None = None,
    gene_symbols: list[str] | None = None,
) -> CellChatDB:
    n = max(len(ligands), len(receptors))
    agonists = (list(agonists or []) + [""] * n)[:n]
    co_a = (list(co_a or []) + [""] * n)[:n]

    interaction = pd.DataFrame({
        "ligand": ligands,
        "receptor": receptors,
        "agonist": agonists,
        "co_A_receptor": co_a,
    })
    complex_input = make_complex_input(complex_entries or {})
    cofactor_input = make_cofactor_input(cofactor_entries or {})

    # Avoid treating complex identifiers as gene symbols when populating
    # gene_info. Only include ligands/receptors that are actual gene
    # symbols (i.e. not keys of the complex_input map).
    complex_keys = set((complex_entries or {}).keys())
    all_expected = (gene_symbols or []) + [
        g for v in (complex_entries or {}).values() for g in v if g
    ] + [
        g for v in (cofactor_entries or {}).values() for g in v if g
    ] + [g for g in (ligands + receptors) if g and g not in complex_keys]

    gene_info = make_gene_info(*dict.fromkeys(g for g in all_expected if g))

    return CellChatDB(
        interaction_input=interaction,
        complex_input=complex_input,
        cofactor_input=cofactor_input,
        gene_info=gene_info,
    )


def test_extract_gene_ligands_receptors():
    db = _make_db(["LG1", "LG2"], ["RCP1", "RCP2"])
    result = extract_gene(db)
    assert sorted(result) == ["LG1", "LG2", "RCP1", "RCP2"]


def test_extract_gene_complex_ligand():
    db = _make_db(
        ["COMPLEX_LR"],
        ["RCP1"],
        complex_entries={"COMPLEX_LR": ["SUBUNIT_A", "SUBUNIT_B"]},
        gene_symbols=["RCP1", "SUBUNIT_A", "SUBUNIT_B"],
    )
    result = extract_gene(db)
    assert "SUBUNIT_A" in result
    assert "SUBUNIT_B" in result
    assert "RCP1" in result
    assert "COMPLEX_LR" not in result


def test_extract_gene_cofactors():
    db = _make_db(
        ["LG1"],
        ["RCP1"],
        agonists=["COF_A"],
        cofactor_entries={"COF_A": ["COF_GENE_1", "COF_GENE_2"]},
        gene_symbols=["LG1", "RCP1", "COF_GENE_1", "COF_GENE_2"],
    )
    result = extract_gene(db)
    assert "COF_GENE_1" in result
    assert "COF_GENE_2" in result


def test_extract_gene_empty_cofactor_strings():
    # All cofactor cells are "" – no spurious genes should be added.
    db = _make_db(["LG1"], ["RCP1"])
    result = extract_gene(db)
    assert result == sorted(result) or set(result) == {"LG1", "RCP1"}
    assert "" not in result


def test_extract_gene_deduplication_across_lr_cofactor():
    # SHARED_GENE appears as a ligand AND as a cofactor subunit.
    db = _make_db(
        ["SHARED_GENE"],
        ["RCP1"],
        agonists=["COF_A"],
        cofactor_entries={"COF_A": ["SHARED_GENE"]},
        gene_symbols=["SHARED_GENE", "RCP1"],
    )
    result = extract_gene(db)
    assert result.count("SHARED_GENE") == 1
