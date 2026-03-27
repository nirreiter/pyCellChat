from __future__ import annotations

from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats

from ..database.extract import extract_gene
from .lifecycle import create_cellchat_state
from .matrix import MatrixType


class CellChat:
    adata: ad.AnnData
    group_by_col: str
    sample_col: str
    is_merged: bool
    experiment_type: str
    options: dict[str, Any]
    selected_features: np.ndarray | None
    selected_features_df: pd.DataFrame | None
    adata_signaling: ad.AnnData | None
    net: dict[str, Any]
    netP: dict[str, Any]
    images: dict[str, Any]
    db: Any
    lr: dict[str, Any]

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

    def subset_data(
        self,
        features: list[str] | None = None,
    ):
        # Annotation ordering side-effect: matches R's in-place sort of
        # object@DB$interaction before gene extraction.
        if self.db is not None:
            interaction = self.db.interaction_input
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
                self.db.interaction_input = interaction.assign(
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
        inplace=True,
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
        if self.adata_signaling is None:
            raise ValueError("Must call CellChat.subset_data() first!")

        if features is None:
            feature_names = self.adata_signaling.var_names
        else:
            feature_names = self.adata_signaling.var_names.intersection(features)
        adata = self.adata_signaling[:, feature_names]  # pyright: ignore[reportArgumentType]

        if adata.X is None:
            raise ValueError("Anndata X cannot be None")
        X: MatrixType = cast(MatrixType, adata.X)
        if sparse.issparse(X):
            X = cast(MatrixType, sparse.csr_matrix(X))

        if not do_differential_expression:
            gene_mask, _cell_counts = sc.pp.filter_genes(
                self.adata_signaling,
                min_cells=min_cells,
                inplace=False,
            )  # pyright: ignore[reportGeneralTypeIssues]
            gene_mask = np.asarray(gene_mask, dtype=bool)
            if inplace:
                self.selected_features = feature_names.to_numpy()[gene_mask]
                return
            return pd.Index(feature_names.to_numpy()[gene_mask])

        if positive_samples is None:
            cells_in_positive = np.full(adata.n_obs, False, dtype=bool)
        else:
            if self.sample_col is None:
                raise ValueError(
                    "No sample column ('sample_col') set on adata object. Must set a sample "
                    "column when comparing conditions for DE."
                )

            samples = adata.obs[self.sample_col]
            cells_in_positive = samples.isin(positive_samples).to_numpy()

        if ignore_groups_for_differential_expression:
            if positive_samples is None:
                raise ValueError("Can't ignore groups for DE if positive_samples is None.")
            cell_masks = [
                (
                    ",".join(positive_samples),
                    cells_in_positive,
                    ~cells_in_positive,
                )
            ]
        else:
            groups = adata.obs[self.group_by_col]
            unique_groups = groups.unique()
            cell_masks = []
            for group_name in unique_groups:
                if positive_samples is None:
                    cells_in_case = (groups == group_name).to_numpy()
                    cells_in_control = (groups != group_name).to_numpy()
                else:
                    cells_in_case = ((groups == group_name) & cells_in_positive).to_numpy()
                    cells_in_control = ((groups == group_name) & ~cells_in_positive).to_numpy()
                cell_masks.append((group_name, cells_in_case, cells_in_control))

        kept_features = pd.DataFrame()
        X_view: Any = X
        
        # import time
        # from tqdm import tqdm
        for mask in cell_masks: #tqdm(cell_masks):
            # current_start = time.time()

            current_feature_mask = np.full(len(feature_names), True)
            group_name = mask[0]
            cells_in_case = mask[1]
            cells_in_control = mask[2]
            
            # print(cells_in_case)
            # print(X_view[cells_in_case, :])
            # print(cells_in_control)
            # print(X_view[cells_in_control, :])

            if threshold_percent_expressing > 0:
                percent_expressing_in_case = np.mean(X_view[cells_in_case, :] > 0, axis=0) * 100
                percent_expressing_in_control = (
                    np.mean(X_view[cells_in_control, :] > 0, axis=0) * 100
                )
                max_percent = np.maximum(
                    percent_expressing_in_case,
                    percent_expressing_in_control,
                )

                current_feature_mask &= max_percent > threshold_percent_expressing
                if np.sum(current_feature_mask) == 0:
                    print(
                        "No features passed the percent_expressing threshold for "
                        f"'{group_name}'"
                    )
                    continue

            # next_start = time.time()
            # print("Percent expression time:", next_start - current_start)
            # current_start = next_start

            if sparse.issparse(X):
                def lognorm_mean_function(value):
                    return np.log(np.mean(np.exp(value.toarray()), axis=0))
            else:
                def lognorm_mean_function(value):
                    return np.log(np.mean(np.exp(value), axis=0))

            avg_in_case = lognorm_mean_function(X_view[cells_in_case, :])
            avg_in_control = lognorm_mean_function(X_view[cells_in_control, :])
            # print(avg_in_case, avg_in_control)
            log_fold_change = avg_in_case - avg_in_control

            if threshold_logfc > 0:
                current_feature_mask &= np.abs(log_fold_change) > threshold_logfc
                if np.sum(current_feature_mask) == 0:
                    print(
                        "No features passed the log fold change threshold for "
                        f"'{group_name}'"
                    )
                    continue

            group_features = feature_names[current_feature_mask]
            log_fold_change = log_fold_change[current_feature_mask]

            # next_start = time.time()
            # print("Logfc time:", next_start - current_start)
            # current_start = next_start

            def sparse_mannwhitneyu(case_data, control_data, chunk_size=1000):
                n_genes = case_data.shape[1]
                n_case = case_data.shape[0]
                n_control = control_data.shape[0]

                case_buffer = np.empty((n_case, chunk_size), dtype=np.float32)
                control_buffer = np.empty((n_control, chunk_size), dtype=np.float32)

                pvalues = np.empty(n_genes)

                for start in range(0, n_genes, chunk_size):
                    end = min(start + chunk_size, n_genes)
                    actual_chunk_width = end - start

                    current_case_view = case_buffer[:, :actual_chunk_width]
                    current_control_view = control_buffer[:, :actual_chunk_width]

                    current_case_view[:] = case_data[:, start:end].toarray()
                    current_control_view[:] = control_data[:, start:end].toarray()

                    res = stats.mannwhitneyu(current_case_view, current_control_view, axis=0)
                    pvalues[start:end] = res.pvalue

                return pvalues

            case_data = cast(MatrixType, adata[cells_in_case, current_feature_mask].X)
            control_data = cast(MatrixType, adata[cells_in_control, current_feature_mask].X)
            if sparse.issparse(adata.X):
                pvalues = sparse_mannwhitneyu(
                    sparse.csc_matrix(case_data),
                    sparse.csc_matrix(control_data),
                )
            else:
                pvalues = stats.mannwhitneyu(case_data, control_data, axis=0).pvalue
            padj = stats.false_discovery_control(pvalues, method="bh")
            significant_feature_mask = pvalues < threshold_p

            if only_pos:
                significant_feature_mask &= log_fold_change > 0

            # next_start = time.time()
            # print("MannwhitneyU time:", next_start - current_start)
            # current_start = next_start

            kept_features = pd.concat(
                [
                    kept_features,
                    pd.DataFrame(
                        {
                            "group": group_name,
                            "feature": group_features[significant_feature_mask],
                            "logfc": log_fold_change[significant_feature_mask],
                            "pvalue": pvalues[significant_feature_mask],
                            "padj": padj[significant_feature_mask],
                        }
                    ),
                ]
            )

            # next_start = time.time()
            # print("Create df time:", next_start - current_start)

        if inplace:
            self.selected_features = kept_features["feature"]
            self.selected_features_df = kept_features
        else:
            return kept_features["feature"]

    def identify_over_expressed_interactions(self):
        raise NotImplementedError()

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
