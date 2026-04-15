from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..core.cellchat import CellChat


_ANNOTATION_ORDER = [
    "Secreted Signaling",
    "ECM-Receptor",
    "Non-protein Signaling",
    "Cell-Cell Contact",
]


def compute_communication_probability_pathway(
    cellchat: CellChat | None = None,
    net: dict[str, Any] | None = None,
    pair_lr_use: pd.DataFrame | None = None,
    thresh: float = 0.05,
) -> CellChat | dict[str, Any]:
    resolved_net = _resolve_net(cellchat, net)
    prob = np.asarray(resolved_net["prob"], dtype=float)
    pval = np.asarray(resolved_net["pval"], dtype=float)

    if prob.ndim != 3:
        raise ValueError("cellchat.net['prob'] must be a 3D array")
    if pval.shape != prob.shape:
        raise ValueError("cellchat.net['pval'] must match the shape of cellchat.net['prob']")

    pair_lr = _resolve_pair_lr_use(cellchat, resolved_net, pair_lr_use)
    if len(pair_lr) != prob.shape[2]:
        raise ValueError(
            "The number of ligand-receptor pairs does not match the third dimension of "
            "cellchat.net['prob']"
        )
    if "pathway_name" not in pair_lr.columns:
        raise ValueError("pair_lr_use must contain a 'pathway_name' column")

    prob_masked = prob.copy()
    prob_masked[pval > thresh] = 0.0

    lr_labels = _resolve_lr_labels(pair_lr)
    lr_significant = lr_labels[np.sum(prob_masked, axis=(0, 1)) != 0.0].tolist()

    pathways = pd.unique(pair_lr["pathway_name"].astype(str)).tolist()
    prob_pathways = np.zeros((prob.shape[0], prob.shape[1], len(pathways)), dtype=float)
    for pathway_index, pathway in enumerate(pathways):
        lr_mask = pair_lr["pathway_name"].astype(str).to_numpy() == pathway
        if np.any(lr_mask):
            prob_pathways[:, :, pathway_index] = np.sum(prob_masked[:, :, lr_mask], axis=2)

    pathway_totals = np.sum(prob_pathways, axis=(0, 1))
    pathway_significant_mask = pathway_totals != 0.0
    pathway_significant = np.asarray(pathways, dtype=object)[pathway_significant_mask]
    prob_pathways_significant = prob_pathways[:, :, pathway_significant_mask]

    if pathway_significant.size > 0:
        sorted_index = np.argsort(-pathway_totals[pathway_significant_mask], kind="stable")
        pathway_significant = pathway_significant[sorted_index]
        prob_pathways_significant = prob_pathways_significant[:, :, sorted_index]

    net_pathway = {
        "pathways": pathway_significant.tolist(),
        "prob": prob_pathways_significant,
    }

    if cellchat is None:
        return net_pathway

    if cellchat.net is None:
        cellchat.net = {}
    cellchat.net["LRs"] = lr_significant
    cellchat.netP = net_pathway
    return cellchat


def _resolve_net(
    cellchat: CellChat | None,
    net: dict[str, Any] | None,
) -> dict[str, Any]:
    if net is not None:
        return net
    if cellchat is None or cellchat.net is None:
        raise ValueError(
            "Either provide `net` directly or run compute_communication_probability on a CellChat object first"
        )
    return cellchat.net


def _resolve_pair_lr_use(
    cellchat: CellChat | None,
    net: dict[str, Any],
    pair_lr_use: pd.DataFrame | None,
) -> pd.DataFrame:
    if pair_lr_use is not None:
        resolved = pair_lr_use
    elif "pair_lr_use" in net:
        resolved = net["pair_lr_use"]
    elif cellchat is not None and cellchat.lr is not None:
        resolved = cellchat.lr
    elif cellchat is not None and cellchat.db is not None:
        resolved = cellchat.db.interaction
    else:
        raise ValueError(
            "pair_lr_use was not provided and no ligand-receptor interaction table is available"
        )

    resolved = resolved.copy()
    if "annotation" in resolved.columns and resolved["annotation"].nunique() > 1:
        dtype = pd.CategoricalDtype(categories=_ANNOTATION_ORDER, ordered=True)
        resolved = resolved.assign(
            annotation=resolved["annotation"].astype(dtype)
        ).sort_values("annotation", kind="stable")
        resolved["annotation"] = resolved["annotation"].astype(str)
    return resolved


def _resolve_lr_labels(pair_lr_use: pd.DataFrame) -> np.ndarray:
    if pair_lr_use.index.nlevels == 1:
        labels = pair_lr_use.index.astype(str).to_numpy(copy=False)
        if not np.all(labels == ""):
            return labels
    for column_name in ("interaction_name", "interaction_name_2"):
        if column_name in pair_lr_use.columns:
            return pair_lr_use[column_name].astype(str).to_numpy(copy=False)
    return pair_lr_use.index.astype(str).to_numpy(copy=False)
