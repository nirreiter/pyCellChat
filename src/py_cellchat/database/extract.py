from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .cellchat_db import CellChatDB


def extract_gene_subset(
    gene_set: list[str],
    complex_input: pd.DataFrame,
    gene_info: pd.DataFrame,
) -> list[str]:
    """Expand a set of gene / complex names to individual gene symbols.

    Mirrors ``extractGeneSubset`` in ``CellChat/R/database.R``.

    Parameters
    ----------
    gene_set:
        Ligand or receptor names that may be either official gene symbols or
        complex identifiers (row index of *complex_input*).
    complex_input:
        DataFrame indexed by complex name with columns ``subunit_1``,
        ``subunit_2``, … holding the constituent gene symbols.  Empty
        strings represent absent subunits.
    gene_info:
        DataFrame with at least a ``Symbol`` column listing official gene
        symbols.

    Returns
    -------
    list[str]
        Deduplicated list of individual gene symbols.
    """
    symbol_set = set(gene_info["Symbol"].dropna())

    # Split into known single symbols and complex identifiers.
    single_genes = [g for g in gene_set if g in symbol_set]
    complex_names = [g for g in gene_set if g not in symbol_set]

    # Expand complexes: keep only complex names that exist in the index.
    matched_complexes = complex_input.index.intersection(complex_names)
    subunit_cols = [c for c in complex_input.columns if c.startswith("subunit")]
    subunit_values: list[str] = []
    if len(matched_complexes) > 0:
        sub_df = complex_input.loc[matched_complexes, subunit_cols]
        subunit_values = [
            v for v in sub_df.values.flatten().tolist()
            if isinstance(v, str) and v != ""
        ]

    return list(dict.fromkeys(single_genes + subunit_values))


def extract_gene(db: CellChatDB) -> list[str]:
    """Return all individual gene symbols referenced in *db*.

    Covers ligand genes, receptor genes, and cofactor genes — fully
    expanding any complex names to their constituent subunits.

    Mirrors ``extractGene`` in ``CellChat/R/database.R``.

    Parameters
    ----------
    db:
        A :class:`~py_cellchat.database.CellChatDB` instance, typically
        already filtered by :func:`~py_cellchat.database.subset_db`.

    Returns
    -------
    list[str]
        Deduplicated list of gene symbols used by this database.
    """
    interaction = db.interaction
    complex_input = db.complex
    cofactor_input = db.cofactor
    gene_info = db.gene_info

    ligands = interaction["ligand"].dropna().unique().tolist()
    receptors = interaction["receptor"].dropna().unique().tolist()

    gene_l = extract_gene_subset(ligands, complex_input, gene_info)
    gene_r = extract_gene_subset(receptors, complex_input, gene_info)
    genes_lr = list(dict.fromkeys(gene_l + gene_r))

    # Expand cofactors (agonist, antagonist, co_A_receptor, co_I_receptor).
    cofactor_cols = ["agonist", "antagonist", "co_A_receptor", "co_I_receptor"]
    existing_cols = [c for c in cofactor_cols if c in interaction.columns]
    cofactor_names: list[str] = []
    for col in existing_cols:
        cofactor_names.extend(
            v for v in interaction[col].dropna().tolist()
            if isinstance(v, str) and v != ""
        )
    cofactor_names = list(dict.fromkeys(cofactor_names))

    matched_cofactors = cofactor_input.index.intersection(cofactor_names)
    cofactor_gene_cols = [c for c in cofactor_input.columns if c.startswith("cofactor")]
    genes_cofactor: list[str] = []
    if len(matched_cofactors) > 0:
        cof_df = cofactor_input.loc[matched_cofactors, cofactor_gene_cols]
        genes_cofactor = [
            v for v in cof_df.values.flatten().tolist()
            if isinstance(v, str) and v != ""
        ]
    genes_cofactor = list(dict.fromkeys(genes_cofactor))

    return list(dict.fromkeys(genes_lr + genes_cofactor))
