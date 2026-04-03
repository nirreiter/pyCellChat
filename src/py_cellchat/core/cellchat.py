from __future__ import annotations

from typing import Any, cast
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
from ..preprocessing import identify_over_expressed_genes, identify_over_expressed_interactions

from ..database import CellChatDB, extract_gene, load_cellchat_db
from .matrix import get_adata_matrix_checked


class CellChat:
    adata: ad.AnnData
    group_by_col: str
    sample_col: str
    is_merged: bool
    experiment_type: str
    # options: dict[str, Any]
    selected_features: np.ndarray | None
    selected_features_df: pd.DataFrame | None
    adata_signaling: ad.AnnData | None
    # net: dict[str, Any]
    # netP: dict[str, Any]
    # images: dict[str, Any]
    db: CellChatDB
    lr: pd.DataFrame | None

    def __init__(
        self,
        adata: ad.AnnData,
        experiment_type: str = "RNA",
        layer: str | None = None,
        counts_layer: str | None = "counts",
        group_by_column: str = "cluster",
        sample_column: str | None = None,
    ):
        """Create a CellChat object from AnnData.

        The active grouping and sample metadata are stored canonically in
        ``adata.obs``. ``group_by_col`` and ``sample_col`` point to the
        columns used by downstream APIs. Compatibility accessors such as
        ``meta`` and ``idents`` are derived from that AnnData-backed state.
        """
        print("Creating a python CellChat object from an anndata object...")
        state = create_cellchat_state(
            adata=adata,
            experiment_type=experiment_type,
            layer=layer,
            counts_layer=counts_layer,
            group_by_column=group_by_column,
            sample_column=sample_column,
        )
        for field_name in state.__dataclass_fields__:
            setattr(self, field_name, getattr(state, field_name))

    @property
    def meta(self) -> pd.DataFrame:
        """Return the canonical per-cell metadata table.

        This is an AnnData-backed compatibility accessor over ``adata.obs``.
        """
        return cast(pd.DataFrame, self.adata.obs)

    @property
    def idents(self) -> pd.Series:
        """Return the active grouping series derived from ``adata.obs``."""
        return self.adata.obs[self.group_by_col]

    def load_database(self, species: str):
        self.db = load_cellchat_db(species)
        

    def subset_data(
        self,
        features: list[str] | None = None,
    ):
        # Annotation ordering side-effect: matches R's in-place sort of
        # object@DB$interaction before gene extraction.
        if self.db is not None:
            interaction = self.db.interaction
            if (
                "annotation" in interaction.columns
                and interaction["annotation"].nunique() > 1
            ):
                _ANNOTATION_ORDER = [
                    "Secreted Signaling",
                    "ECM-Receptor",
                    "Non-protein Signaling",
                    "Cell-Cell Contact",
                ]
                cat_type = pd.CategoricalDtype(
                    categories=_ANNOTATION_ORDER, ordered=True
                )
                self.db.interaction = interaction.assign(
                    annotation=interaction["annotation"].astype(cat_type)
                ).sort_values("annotation").assign(
                    annotation=lambda df: df["annotation"].astype(str)
                )

        if features is None:
            if self.db is not None:
                db_genes = extract_gene(self.db)
                selected = self.adata.var_names.intersection(db_genes)
                self.adata_signaling = self.adata[:, selected].copy()  # pyright: ignore[reportArgumentType]
            else:
                self.adata_signaling = self.adata.copy()
            return

        selected_features = self.adata.var_names.intersection(features)
        self.adata_signaling = self.adata[:, selected_features].copy()  # pyright: ignore[reportArgumentType]

    def identify_over_expressed_genes(
        self,
        inplace = True,
        min_cells: int = 10,
        only_pos: bool = True,
        features: list[str] | pd.Index[str] | None = None,
        threshold_percent_expressing: float = 0,
        threshold_logfc: float = 0,
        threshold_p: float = 0.05,
        do_differential_expression=True,
        positive_samples: list[str] | None = None,
        ignore_groups_for_differential_expression=False,
    ):
        return identify_over_expressed_genes(
            self,
            inplace = inplace, 
            min_cells = min_cells, 
            only_pos = only_pos,
            features = features,
            threshold_percent_expressing = threshold_percent_expressing,
            threshold_logfc = threshold_logfc,
            threshold_p = threshold_p,
            do_differential_expression = do_differential_expression,
            positive_samples = positive_samples,
            ignore_groups_for_differential_expression = ignore_groups_for_differential_expression,
        )
        

    def identify_over_expressed_interactions(
        self,
        variable_both: bool = True,
        features=None,
        inplace: bool = True,
    ):
        return identify_over_expressed_interactions(
            self,
            variable_both=variable_both,
            features=features,
            inplace=inplace,
        )

    def compute_communication_probability(self):
        raise NotImplementedError()

    def filter_communication(self):
        raise NotImplementedError()

    def compute_communication_probability_pathways(self):
        raise NotImplementedError()

    def aggregate_net(self):
        raise NotImplementedError()

    def network_analysis_compute_centrality(self):
        raise NotImplementedError()

    def __str__(self) -> str:
        result = ""
        if not self.is_merged:
            result += (
                "A python CellChat object created from a single "
                f"{self.experiment_type} dataset of {self.adata.n_vars} genes by "
                f"{self.adata.n_obs} cells"
            )
        else:
            raise NotImplementedError("Merged datasets are not currently supported")

        if self.experiment_type == "spatial":
            raise NotImplementedError("Spatial datasets are not currently supported")

        return result

    def lift_groups(self, new_groups: list[str]):
        raise NotImplementedError()

    def __getitem__(self):
        raise NotImplementedError()

    def __setitem__(self):
        raise NotImplementedError()

    def __delitem__(self):
        raise NotImplementedError()


def merge():
    raise NotImplementedError()


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
    lr: pd.DataFrame | None = None
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
