from __future__ import annotations

import copy

from .cellchat_db import CellChatDB

_DEFAULT_PROTEIN_SEARCH = ["Secreted Signaling", "ECM-Receptor", "Cell-Cell Contact"]
_NON_PROTEIN_ANNOTATION = "Non-protein Signaling"


def subset_db(
    db: CellChatDB,
    search: list[str] | list[list[str]] | None = None,
    key: str | list[str] = "annotation",
    non_protein: bool = False,
) -> CellChatDB:
    """Return a copy of *db* with ``interaction`` filtered by *search*.

    Mirrors ``subsetDB`` in ``CellChat/R/database.R``.

    Parameters
    ----------
    db:
        Source :class:`CellChatDB` to filter.
    search:
        Values to keep in the *key* column(s).  When *key* is
        ``"annotation"`` and *search* is ``None``, defaults to the three
        standard protein-signaling categories; ``"Non-protein Signaling"``
        is added when *non_protein* is ``True``.
    key:
        Column name or list of column names in ``interaction`` to
        filter on.  When a list, *search* must be a matching list of
        value-lists (one per key).
    non_protein:
        When ``True``, include ``"Non-protein Signaling"`` interactions.
        Ignored when *key* is not ``"annotation"`` or *search* is
        explicitly provided.

    Returns
    -------
    CellChatDB
        A shallow copy of *db* with **only** ``interaction`` replaced
        by the filtered DataFrame; all other tables are shared references.
    """
    interaction = db.interaction.copy()

    single_key = isinstance(key, str)
    annotation_default = single_key and key == "annotation"

    # Resolve default search for annotation key.
    if search is None and annotation_default:
        search = list(_DEFAULT_PROTEIN_SEARCH)
        if non_protein:
            search = search + [_NON_PROTEIN_ANNOTATION]
    elif search is not None and annotation_default:
        # Caller passed non_protein via search; honour it for the strip below.
        if _NON_PROTEIN_ANNOTATION in search:
            non_protein = True

    if "Non-protein Signaling" in (search or []) or non_protein:
        pass  # keep all rows; filter below will handle it
    else:
        # Strip non-protein rows even when the caller provided a custom search
        # list that does not include the annotation key — matches R behaviour.
        if annotation_default and "annotation" in interaction.columns:
            interaction = interaction[
                interaction["annotation"] != _NON_PROTEIN_ANNOTATION
            ]

    # Apply column filter.
    if search is not None:
        if single_key:
            if key not in interaction.columns:
                raise ValueError(
                    f"Column '{key}' not found in interaction. "
                    f"Available columns: {list(interaction.columns)}"
                )
            interaction = interaction[interaction[key].isin(search)]
        else:
            # multi-key path: search must be a list of lists.
            if not isinstance(search, list) or not all(
                isinstance(s, list) for s in search
            ):
                raise ValueError(
                    "When 'key' is a list, 'search' must be a list of lists."
                )
            if len(key) != len(search):
                raise ValueError(
                    "'key' and 'search' must have the same length."
                )
            for k_i, s_i in zip(key, search):
                if k_i not in interaction.columns:
                    raise ValueError(
                        f"Column '{k_i}' not found in interaction."
                    )
                interaction = interaction[interaction[k_i].isin(s_i)]

    result = copy.copy(db)  # shallow copy — share complex/cofactor/gene_info
    result.interaction = interaction
    return result
