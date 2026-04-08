from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse

from ..core.matrix import get_adata_matrix_checked
from ..database import extract_gene_subset
from .expression import compute_expr_lr
from .statistics import build_group_average

if TYPE_CHECKING:
    from ..core.cellchat import CellChat


def filter_communication(
    cellchat: CellChat,
    min_cells: int = 10,
    min_samples: int | None = None,
    rare_keep: bool = False,
    non_filter_keep: bool = False,
) -> CellChat:
    net = _validate_net(cellchat.net)
    prob = np.asarray(net["prob"], dtype=float).copy()
    pval = np.asarray(net["pval"], dtype=float).copy()
    group = cellchat.idents
    group_levels = _levels(group)

    if prob.shape[:2] != (len(group_levels), len(group_levels)):
        raise ValueError(
            "cellchat.net['prob'] shape does not match the active CellChat identity levels"
        )

    if non_filter_keep:
        net["prob_non_filter"] = prob.copy()
        net["pval_non_filter"] = pval.copy()

    group_counts = group.value_counts(sort=False).reindex(group_levels, fill_value=0)
    global_excluded = np.flatnonzero(group_counts.to_numpy(dtype=int) <= min_cells)
    if global_excluded.size > 0:
        prob[global_excluded, :, :] = 0.0
        prob[:, global_excluded, :] = 0.0

    resolved_min_samples = 1 if min_samples is None else min_samples
    sample_info = cellchat.meta[cellchat.sample_col]
    sample_levels = _levels(sample_info)
    if min_samples is not None and resolved_min_samples > len(sample_levels):
        raise ValueError(
            f"There are only {len(sample_levels)} samples in the data. "
            "Please change the value of min_samples!"
        )

    if len(sample_levels) >= 2 and resolved_min_samples >= 2:
        prob = _filter_inconsistent_probabilities(
            cellchat=cellchat,
            prob=prob,
            min_cells=min_cells,
            min_samples=resolved_min_samples,
            rare_keep=rare_keep,
            group=group,
            group_levels=group_levels,
            sample_info=sample_info,
            sample_levels=sample_levels,
            global_excluded=global_excluded,
        )

    net["prob"] = prob
    net["pval"] = pval
    return cellchat


def _validate_net(net: dict[str, Any] | None) -> dict[str, Any]:
    if net is None or "prob" not in net or "pval" not in net:
        raise ValueError(
            "CellChat.net must contain 'prob' and 'pval' before running filter_communication"
        )

    prob = np.asarray(net["prob"])
    pval = np.asarray(net["pval"])
    if prob.ndim != 3:
        raise ValueError("cellchat.net['prob'] must be a 3D array")
    if pval.shape != prob.shape:
        raise ValueError("cellchat.net['pval'] must match the shape of cellchat.net['prob']")
    return net


def _filter_inconsistent_probabilities(
    *,
    cellchat: CellChat,
    prob: np.ndarray,
    min_cells: int,
    min_samples: int,
    rare_keep: bool,
    group: pd.Series,
    group_levels: Sequence[str],
    sample_info: pd.Series,
    sample_levels: Sequence[str],
    global_excluded: np.ndarray,
) -> np.ndarray:
    if cellchat.db is None:
        raise ValueError("Must load a CellChatDB before running filter_communication")
    if cellchat.adata_signaling is None:
        raise ValueError("Must run subset_data on the CellChat object first")

    parameters = cellchat.options.get("parameter", {})
    raw_use = bool(parameters.get("raw_use", True))
    if not raw_use:
        raise NotImplementedError(
            "raw_use=False requires projected or smoothed data support, which is not implemented yet"
        )

    pair_lr_use = _resolve_pair_lr_use(cellchat)
    if len(pair_lr_use) != prob.shape[2]:
        raise ValueError(
            "The number of ligand-receptor pairs does not match the third dimension of cellchat.net['prob']"
        )

    nonzero_lr_index = np.flatnonzero(np.sum(prob, axis=(0, 1)) != 0)
    if nonzero_lr_index.size == 0:
        return prob

    pair_lr_nonzero = pair_lr_use.iloc[nonzero_lr_index]
    gene_l = pair_lr_nonzero["ligand"].astype(str).tolist()
    gene_r = pair_lr_nonzero["receptor"].astype(str).tolist()
    gene_subset = extract_gene_subset(
        list(dict.fromkeys(gene_l + gene_r)),
        cellchat.db.complex,
        cellchat.db.gene_info,
    )

    matrix = get_adata_matrix_checked(cellchat.adata_signaling, False, None)
    data = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    data = np.asarray(data, dtype=float)
    data_max = float(np.max(data)) if data.size else 0.0
    data_use = data if data_max == 0.0 else data / data_max

    all_gene_names = cellchat.adata_signaling.var_names.astype(str).tolist()
    gene_mask = [gene_name in set(gene_subset) for gene_name in all_gene_names]
    data_use = data_use[:, gene_mask]
    gene_names = [gene_name for gene_name, keep in zip(all_gene_names, gene_mask, strict=False) if keep]

    mean_function = build_group_average(
        str(parameters.get("type_mean", "triMean")),
        trim=float(parameters.get("trim", 0.1)),
    )
    n_groups = len(group_levels)
    support_counts = np.zeros((n_groups, n_groups, len(pair_lr_nonzero)), dtype=int)
    sample_values = sample_info.astype(str).to_numpy()
    sample_rare = set()

    for sample_level in sample_levels:
        sample_mask = sample_values == str(sample_level)
        sample_counts = group.iloc[sample_mask].value_counts(sort=False).reindex(group_levels, fill_value=0)
        sample_excluded = np.flatnonzero(sample_counts.to_numpy(dtype=int) <= min_cells)
        sample_rare.update(sample_excluded.tolist())

        present_levels = [
            level
            for level, count in zip(group_levels, sample_counts.to_numpy(dtype=int), strict=False)
            if count > 0
        ]
        if not present_levels:
            continue

        data_use_avg_present = _aggregate_expression_by_group(
            data_use[sample_mask, :],
            gene_names,
            group.iloc[sample_mask],
            present_levels,
            mean_function,
        )
        data_use_avg = pd.DataFrame(
            0.0,
            index=gene_names,
            columns=list(group_levels),
        )
        data_use_avg.loc[:, present_levels] = data_use_avg_present.loc[:, present_levels]

        data_lavg = compute_expr_lr(gene_l, data_use_avg, cellchat.db.complex)
        data_ravg = compute_expr_lr(gene_r, data_use_avg, cellchat.db.complex)
        score_lr = np.einsum("lg,lh->ghl", data_lavg, data_ravg)
        if sample_excluded.size > 0:
            score_lr[sample_excluded, :, :] = 0.0
            score_lr[:, sample_excluded, :] = 0.0
        support_counts += (score_lr > 0).astype(int)

    sample_rare -= set(global_excluded.tolist())
    inconsistent = (support_counts > 0) & (support_counts < min_samples)
    inconsistent_lr = np.flatnonzero(np.any(inconsistent, axis=(0, 1)))
    for lr_position in inconsistent_lr.tolist():
        consistent = (support_counts[:, :, lr_position] >= min_samples).astype(int)
        if rare_keep and sample_rare:
            sample_rare_index = list(sample_rare)
            consistent[sample_rare_index, :] = 1
            consistent[:, sample_rare_index] = 1
        prob[:, :, nonzero_lr_index[lr_position]] *= consistent

    return prob


def _levels(values: pd.Series) -> list[str]:
    if isinstance(values.dtype, pd.CategoricalDtype):
        return values.cat.categories.astype(str).tolist()
    return pd.unique(values.astype(str)).tolist()


def _resolve_pair_lr_use(cellchat: CellChat) -> pd.DataFrame:
    pair_lr_use = cellchat.lr if cellchat.lr is not None else cellchat.db.interaction
    pair_lr_use = pair_lr_use.copy()
    if "annotation" in pair_lr_use.columns and pair_lr_use["annotation"].nunique() > 1:
        annotation_order = [
            "Secreted Signaling",
            "ECM-Receptor",
            "Non-protein Signaling",
            "Cell-Cell Contact",
        ]
        dtype = pd.CategoricalDtype(categories=annotation_order, ordered=True)
        pair_lr_use = pair_lr_use.assign(
            annotation=pair_lr_use["annotation"].astype(dtype)
        ).sort_values("annotation", kind="stable")
        pair_lr_use["annotation"] = pair_lr_use["annotation"].astype(str)
    return pair_lr_use


def _aggregate_expression_by_group(
    data_use: np.ndarray,
    gene_names: Sequence[str],
    group: pd.Series,
    group_levels: Sequence[str],
    mean_function,
) -> pd.DataFrame:
    group_values = group.astype(str).to_numpy()
    aggregated = np.zeros((len(gene_names), len(group_levels)), dtype=float)
    for index, level in enumerate(group_levels):
        group_mask = group_values == level
        group_matrix = data_use[group_mask, :]
        aggregated[:, index] = np.apply_along_axis(mean_function, 0, group_matrix)
    return pd.DataFrame(aggregated, index=gene_names, columns=list(group_levels))