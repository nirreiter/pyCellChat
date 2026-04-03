from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import pandas as pd

if TYPE_CHECKING:
    from ..core.cellchat import CellChat


def identify_over_expressed_interactions(
    cellchat: CellChat,
    variable_both: bool = True,
    features: Iterable | None = None,
    inplace: bool = True,
):
    """Identify over-expressed ligand-receptor pairs within the used CellChatDB.

    Mirrors R ``identifyOverExpressedInteractions``.

    Mutation contract (``inplace=True``):
      - ``cellchat.lr`` is set to a DataFrame of the filtered
        interaction rows (same schema as ``cellchat.db.interaction``).

    Parameters
    ----------
    cellchat:
        CellChat object. ``subset_data`` must be run first.
        ``identify_over_expressed_genes`` must be run first if features is None.
    variable_both:
        If ``True`` (default), both the ligand and the receptor must be
        individually over-expressed (or belong to an over-expressed complex).
        If ``False``, only one partner needs to be over-expressed while both
        must still be present in the signaling gene universe.
    features:
        Override the over-expressed gene set. If ``None``, uses
        ``cellchat.selected_features`` from ``identify_over_expressed_genes``.
    inplace:
        If ``True`` (default), store the result in ``cellchat.lr``
        and return ``None``. If ``False``, return the filtered interaction
        DataFrame without mutating the object.
    """
    if cellchat.adata_signaling is None:
        raise ValueError("Must run subset_data on the cellchat object first")

    # --- Step 1: resolve gene universe and over-expressed feature set ----------
    # gene_use = every gene present in the signaling matrix (survived subsetData)
    gene_use: set[str] = set(cellchat.adata_signaling.var_names)

    if features is None:
        if cellchat.selected_features is None:
            raise ValueError(
                "Cannot run identify_over_expressed_interactions without first "
                "running identify_over_expressed_genes or providing a list of "
                "features to use."
            )
        features = cellchat.selected_features
    
    features = set(features)
    
    interaction_input = cellchat.db.interaction
    complex_input = cellchat.db.complex

    # --- Step 2: extract subunit columns from complex table -------------------
    subunit_cols = [c for c in complex_input.columns if "subunit" in c]
    complex_subunits = complex_input[subunit_cols]

    # --- Step 3: over-expressed complexes (complexSubunits.sig) ---------------
    # Keep complex if ≥1 subunit is over-expressed AND all subunits are in gene_use.
    def _is_sig_complex(row: pd.Series) -> bool:
        subunits = [s for s in row if s != ""]
        return (
            len(set(subunits) & features) > 0
            and all(s in gene_use for s in subunits)
        )

    sig_complex_mask = complex_subunits.apply(_is_sig_complex, axis=1)
    complex_subunits_sig = complex_subunits[sig_complex_mask]

    # --- Step 4: all usable complexes (complexSubunits.use) -------------------
    # Keep complex if ALL subunits are in gene_use (no over-expression requirement).
    def _is_use_complex(row: pd.Series) -> bool:
        subunits = [s for s in row if s != ""]
        return all(s in gene_use for s in subunits)

    use_complex_mask = complex_subunits.apply(_is_use_complex, axis=1)
    complex_subunits_use = complex_subunits[use_complex_mask]

    # --- Step 5: filter ligand-receptor pairs ---------------------------------
    # Ligand/receptor values in interaction are either single gene symbols
    # or complex names (row-index values in complex_input).
    sig_set = features | set(complex_subunits_sig.index)
    use_set = gene_use | set(complex_subunits_use.index)

    ligand = interaction_input["ligand"]
    receptor = interaction_input["receptor"]

    if variable_both:
        # Both ligand AND receptor must be over-expressed (sig gene or sig complex).
        lr_mask = ligand.isin(sig_set) & receptor.isin(sig_set)
    else:
        # Both must be present (gene_use or usable complex) AND
        # at least one must be over-expressed (sig gene or sig complex).
        present = ligand.isin(use_set) & receptor.isin(use_set)
        expressed = ligand.isin(sig_set) | receptor.isin(sig_set)
        lr_mask = present & expressed

    pair_lr_sig = interaction_input[lr_mask]

    # --- Step 6: store result and return --------------------------------------
    print(
        f"The number of highly variable ligand-receptor pairs used for "
        f"signaling inference is {len(pair_lr_sig)}"
    )

    if inplace:
        cellchat.lr = pair_lr_sig
        return
    return pair_lr_sig

    