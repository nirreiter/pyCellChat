from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from ..database import CellChatDB, extract_gene, load_cellchat_db
from .matrix import get_adata_matrix_checked


class CellChat:
    adata: ad.AnnData
    group_by_col: str
    sample_col: str
    is_merged: bool
    options: dict[str, Any]
    selected_features: np.ndarray | None
    selected_features_df: pd.DataFrame | None
    adata_signaling: ad.AnnData | None
    net: dict[str, Any] | None
    netP: dict[str, Any] | None
    # images: dict[str, Any]
    db: CellChatDB | None
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
        ``idents`` are derived from that AnnData-backed state.
        """
        print("Creating a python CellChat object from an anndata object...")

        if adata.isbacked:
            raise NotImplementedError(
                "Disk-backed adata not currently supported, please load data into memory"
            )
        if not isinstance(adata.obs, pd.DataFrame):
            raise NotImplementedError(
                "Dataset2D type for adata.obs is not implemented yet"
            )
        if not isinstance(adata.var, pd.DataFrame):
            raise NotImplementedError(
                "Dataset2D type for adata.var is not implemented yet"
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
                f"Sample column '{sample_column}' was not found in the observation dataframe "
                "of the provided AnnData object ('adata.obs')"
            )

        matrix = get_adata_matrix_checked(adata, False, layer)
        matrix_raw = get_adata_matrix_checked(adata, False, counts_layer)
        
        obs = adata.obs.copy()
        if sample_column is None:
            if "sample" in adata.obs:
                sample_column = "sample"
            elif "samples" in adata.obs:
                sample_column = "samples"
            else:
                obs["sample"] = "sample1"
                obs["sample"] = obs["sample"].astype("category")
                sample_column = "sample"
        obs[group_by_column] = obs[group_by_column].astype("category")

        normalized_adata = ad.AnnData(
            X=matrix,
            layers={"counts": matrix_raw},
            obs=obs,
            var=adata.var.copy(),
        )
        
        self.adata = normalized_adata
        self.group_by_col = group_by_column
        self.sample_col = sample_column
        self.is_merged = False
        self.options = {"mode": "single", "datatype": experiment_type}
        self.selected_features = None
        self.selected_features_df = None
        self.adata_signaling = None
        self.net = None
        self.netP = None
        #self.images = None
        self.db = None
        self.lr = None

    @property
    def idents(self) -> pd.Series:
        """Return the active per-cell grouping derived from ``adata.obs``."""
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
                self.adata_signaling = self.adata[:, selected].copy()
            else:
                self.adata_signaling = self.adata.copy()
            return

        selected_features = self.adata.var_names.intersection(features)
        self.adata_signaling = self.adata[:, selected_features].copy()

    def aggregate_net(self):
        raise NotImplementedError()

    def network_analysis_compute_centrality(self):
        raise NotImplementedError()
    
    def lift_groups(self, new_groups: list[str]):
        raise NotImplementedError()

    def __str__(self) -> str:
        result = ""
        if not self.is_merged:
            result += (
                "A python CellChat object created from a single "
                f"{self.options['experiment_type']} dataset of {self.adata.n_vars} genes by "
                f"{self.adata.n_obs} cells"
            )
        else:
            raise NotImplementedError("Merged datasets are not currently supported")

        if self.options["experiment_type"] == "spatial":
            raise NotImplementedError("Spatial datasets are not currently supported")

        return result

    def __getitem__(self):
        raise NotImplementedError()

    def __setitem__(self):
        raise NotImplementedError()

    def __delitem__(self):
        raise NotImplementedError()


def merge():
    raise NotImplementedError()
