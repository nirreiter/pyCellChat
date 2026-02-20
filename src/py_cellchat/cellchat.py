from __future__ import annotations
import numpy as np
from scipy import stats
import pandas as pd
import anndata as ad
import scanpy as sc


class CellChat:
    def __init__(self, adata: ad.AnnData, layer: str | None = None, group_by_column: str = "cluster", sample_column: str | None = None, experiment_type: str = "RNA"):
        print("Creating a python CellChat object from an anndata object...")
        if adata.isbacked:
            raise NotImplementedError("Disk-backed adata not currently supported, please load data into memory")
        if experiment_type != "RNA":
            raise NotImplementedError("Only single cell RNA experiments are currently supported")

        if layer is not None and layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' was not found in the provided AnnData object ('adata.layers')")
        if group_by_column not in adata.obs.columns:
            raise ValueError(f"Groupby column '{group_by_column}' was not found in the observation dataframe of the provided AnnData object ('adata.obs')")
        if sample_column is not None and sample_column not in adata.obs.columns:
            raise ValueError(f"Sample column '{sample_column}' was not found in the observation dataframe of the provided AnnData object ('adata.obs')")
        
        if layer is None:
            matrix = adata.X
            if not isinstance(matrix, np.ndarray):
                raise ValueError(f"AnnData object X matrix is of invalid type '{type(matrix)}'")
        else:
            matrix = adata.layers[layer]
            if not isinstance(matrix, np.ndarray):
                raise ValueError(f"AnnData object layer '{layer}' is of invalid type '{type(matrix)}'")
        matrix = matrix.copy()
        if matrix.min() < 0:
            raise ValueError("Values in the anndata matrix cannot be negative")
        
        self.adata = ad.AnnData(
            X = matrix, 
            obs = adata.obs, 
            var = adata.var
        )
        self.group_by_col = group_by_column
        self.sample_col = sample_column
        self.is_merged = False # TODO: Allow merging datasets
        self.experiment_type = experiment_type
        
        self.selected_features = None
        self.selected_features_df = None
    
    # Modeling methods
    
    def identify_over_expressed_genes(
        self, 
        inplace = True,
        min_cells: int = 10, 
        only_pos: bool = True, 
        features: list[str] | pd.Index[str] | None = None, 
        threshold_percent_expressing: float = 0, 
        threshold_logfc: float = 0, 
        threshold_p: float = 0.05, 
        do_differential_expression = True,
        positive_samples: list[str] | None = None,
        ignore_groups_for_differential_expression = False,
    ):
        if features is None:
            features = self.adata.var_names
        else:
            features = self.adata.var_names.intersection(features)
        adata = self.adata[:, features] # pyright: ignore[reportArgumentType]
        
        if adata.X is None:
            raise ValueError("Anndata X cannot be None")
        X = adata.X
        
        #* No DE, just filter genes by min_cells
        if not do_differential_expression:
            gene_mask, cell_counts = sc.pp.filter_genes(self.adata, min_cells=min_cells, inplace=False) # pyright: ignore[reportGeneralTypeIssues]
            if inplace:
                self.selected_features = features[gene_mask]
                return
            return features[gene_mask]
        
        #* Generate cell masks for chosen groups (case and control)
        if positive_samples is None:
            cells_in_positive = np.full(adata.n_obs, False)
        else:
            if self.sample_col is None:
                raise ValueError("No sample column ('sample_col') set on adata object. Must set a sample column when comparing conditions for DE.")
            
            samples = adata.obs[self.sample_col]
            cells_in_positive = samples.isin(positive_samples)
        
        if ignore_groups_for_differential_expression:
            if positive_samples is None:
                raise ValueError("Can't ignore groups for DE if positive_samples is None.")
            cell_masks = [(",".join(positive_samples), cells_in_positive, ~cells_in_positive)]
        else:
            groups = adata.obs[self.group_by_col]
            unique_groups = groups.unique()
            cell_masks = []
            for g in unique_groups:
                if positive_samples is None:
                    cells_in_case = (groups == g)
                    cells_in_control = (groups != g)
                else:
                    cells_in_case = (groups == g) & cells_in_positive
                    cells_in_control = (groups == g) & ~cells_in_positive
                cell_masks.append((g, cells_in_case, cells_in_control))
        
        #* Perform thresholding and DE for chosen groups
        kept_features = pd.DataFrame()
        for m in cell_masks:
            current_feature_mask = np.full(len(features), True)
            group_name = m[0]
            cells_in_case = m[1]
            cells_in_control = m[2]
            
            # percent expression threshold
            if threshold_percent_expressing > 0:
                percent_expressing_in_case = np.mean(X[cells_in_case, current_feature_mask] > 0, axis=0) * 100
                percent_expressing_in_control = np.mean(X[cells_in_control, current_feature_mask] > 0, axis=0) * 100
                max_percent = np.maximum(percent_expressing_in_case, percent_expressing_in_control)
                
                current_feature_mask &= max_percent > threshold_percent_expressing
                if np.sum(current_feature_mask) == 0:
                    print(f"No features passed the percent_expressing threshold for '{group_name}'")
                    continue
            
            # Always calculate logfoldchange to use in features dataframe, even if no threshold is used.
            # (since the data is already log-normalized, the difference is equivalent to a log fold change)
            def lognorm_mean_function(x): 
                np.log(np.mean(np.exp(x), axis=0))
            avg_in_case = lognorm_mean_function(X[cells_in_case, current_feature_mask])
            avg_in_control = lognorm_mean_function(X[cells_in_control, current_feature_mask])
            log_fold_change = avg_in_case - avg_in_control
            
            # average log fold change threshold 
            if threshold_logfc > 0:
                if only_pos:
                    current_feature_mask &= (log_fold_change > threshold_logfc)
                else:
                    current_feature_mask &= (np.abs(log_fold_change) > threshold_logfc)
                if np.sum(current_feature_mask) == 0:
                    print(f"No features passed the log fold change threshold for '{group_name}'")
                    continue
            
            # pvalue threshold
            case_data = adata[cells_in_case, current_feature_mask].X
            control_data = adata[cells_in_control, current_feature_mask].X
            pvalues = stats.mannwhitneyu(case_data, control_data).pvalue
            # TODO: Look into 'by' method
            padj = stats.false_discovery_control(pvalues, method='bh')
            
            significant_feature_mask = pvalues < threshold_p
            kept_features = pd.concat([kept_features, pd.DataFrame({
                "group": g,
                "feature": features[current_feature_mask][significant_feature_mask],
                "logfc": log_fold_change[significant_feature_mask],
                "pvalue": pvalues[significant_feature_mask],
                "padj": padj[significant_feature_mask],
            })])
        
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
    
    # Other methods

    def __str__(self) -> str:
        result = ""
        if not self.is_merged:
            result += f"A python CellChat object created from a single {self.experiment_type} dataset of {self.adata.n_vars} genes by {self.adata.n_obs} cells"
        else:
            raise NotImplementedError("Merged datasets are not currently supported")
            # result += f"A python Cellchat object merged from multiple {self.datatype} datasets, with {self.adata.n_vars} signaling genes by {self.adata.n_obs} cells"
        
        if self.experiment_type == "spatial":
            raise NotImplementedError("Spatial datasets are not currently supported")
            # result += f"\nThe input spatial locations are {self.image_coordinates}"
        
        return result
    
    def lift_groups(self, new_groups: list[str]):
        """
        Update the groups (usually cell types or clusters) used by the cellchat object.
        Groups will appear in the order provided by new_groups, so this function can be used to reorder the groups.
        This function only shows or hides labels and does not modify the underlying data. 
        Groups that have been removed are still present, just hidden. 
        Groups that are added and do not exist in the data are defined as empty.
        
        :param self: A python CellChat object
        :param new_groups: The groups which should be used by this dataset, in the desired order.
        :type new_groups: list
        """
        raise NotImplementedError()

    def __getitem__(self):
        raise NotImplementedError()
    
    def __setitem__(self):
        raise NotImplementedError()
    
    def __delitem__(self):
        raise NotImplementedError()
    

def merge():
    raise NotImplementedError()
