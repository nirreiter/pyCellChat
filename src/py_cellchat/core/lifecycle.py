from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from .matrix import get_adata_matrix_checked


@dataclass(slots=True)
class CellChatState:
    adata: ad.AnnData
    group_by_col: str
    sample_col: str
    is_merged: bool
    experiment_type: str
    options: dict[str, Any]
    selected_features: np.ndarray | None = None
    selected_features_df: pd.DataFrame | None = None
    adata_signaling: ad.AnnData | None = None
    net: dict[str, Any] = field(default_factory=dict)
    netP: dict[str, Any] = field(default_factory=dict)
    images: dict[str, Any] = field(default_factory=dict)
    db: Any = None
    lr: dict[str, Any] = field(default_factory=dict)
    var_features: dict[str, Any] = field(default_factory=dict)


def create_cellchat_state(
    adata: ad.AnnData,
    experiment_type: str = "RNA",
    layer: str | None = None,
    counts_layer: str | None = "counts",
    group_by_column: str = "cluster",
    sample_column: str | None = None,
) -> CellChatState:
    """Create normalized CellChat state backed by AnnData.

    The Python object keeps per-cell metadata canonically in ``adata.obs``.
    ``group_by_col`` and ``sample_col`` identify the active categorical
    columns used by public APIs, rather than duplicating that state into
    independent mutable attributes.
    """
    if adata.isbacked:
        raise NotImplementedError(
            "Disk-backed adata not currently supported, please load data into memory"
        )
    if experiment_type != "RNA":
        raise NotImplementedError("Only single cell RNA experiments are currently supported")

    if layer is not None and layer not in adata.layers:
        raise ValueError(
            f"Layer '{layer}' was not found in the provided AnnData object ('adata.layers')"
        )
    if group_by_column not in adata.obs.columns:
        raise ValueError(
            "Groupby column "
            f"'{group_by_column}' was not found in the observation dataframe of the "
            "provided AnnData object ('adata.obs')"
        )
    if sample_column is not None and sample_column not in adata.obs.columns:
        raise ValueError(
            "Sample column "
            f"'{sample_column}' was not found in the observation dataframe of the "
            "provided AnnData object ('adata.obs')"
        )

    matrix = get_adata_matrix_checked(adata, False, layer)
    matrix_raw = get_adata_matrix_checked(adata, False, counts_layer)

    meta = adata.obs.copy()
    resolved_sample_column = _resolve_sample_column(meta, sample_column)
    meta[resolved_sample_column] = _coerce_factor_like(
        meta[resolved_sample_column],
        name=resolved_sample_column,
    )
    meta[group_by_column] = _coerce_factor_like(meta[group_by_column], name=group_by_column)

    normalized_adata = ad.AnnData(
        X=matrix,
        layers={"counts": matrix_raw},
        obs=meta,
        var=adata.var.copy(),
    )

    return CellChatState(
        adata=normalized_adata,
        group_by_col=group_by_column,
        sample_col=resolved_sample_column,
        is_merged=False,
        experiment_type=experiment_type,
        options={"mode": "single", "datatype": experiment_type},
        var_features={"features": None, "features_info": None},
    )


def _resolve_sample_column(meta: pd.DataFrame, sample_column: str | None) -> str:
    if sample_column is not None:
        return sample_column
    if "sample" in meta.columns:
        return "sample"
    meta["sample"] = "sample1"
    return "sample"


def _coerce_factor_like(values: pd.Series, name: str | None = None) -> pd.Series:
    if isinstance(values.dtype, pd.CategoricalDtype):
        present_values = set(values.astype(str))
        categories = [
            category
            for category in values.cat.categories.astype(str).tolist()
            if category in present_values
        ]
        dtype = pd.CategoricalDtype(categories=categories, ordered=values.cat.ordered)
        return pd.Series(values.astype(str).astype(dtype), index=values.index, name=name)

    series = values.astype("string")
    categories = pd.unique(series).tolist()
    dtype = pd.CategoricalDtype(categories=categories, ordered=True)
    return pd.Series(series.astype(dtype), index=values.index, name=name)
