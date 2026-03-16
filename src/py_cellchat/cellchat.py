from __future__ import annotations
from typing import TypeAlias, Union
import numpy as np
from scipy import stats, sparse
import pandas as pd
import anndata as ad
import scanpy as sc

# from py_cellchat.cellchat_db import CellChatDB

MatrixType: TypeAlias = Union[np.ndarray, sparse.sparray, sparse.spmatrix]

def is_integer_matrix(matrix: MatrixType, atol: float = 1e-12, rtol: float = 1e-12) -> bool:
    if isinstance(matrix, np.ndarray):
        data = matrix
    else:
        data = matrix.data # pyright: ignore[reportAttributeAccessIssue]
    if np.issubdtype(data.dtype, np.integer):
        return True

    return bool(np.all(np.isclose(data, np.round(data), atol=atol, rtol=rtol)))


def get_adata_matrix_checked(
    adata: ad.AnnData,
    is_raw: bool = False,
    layer_name: str | None = None,
) -> MatrixType:
    
    if layer_name is None:
        if is_raw:
            matrix = adata.raw.X
        else:
            matrix = adata.X
    else:
        matrix = adata.layers[layer_name]
    
    if not (isinstance(matrix, np.ndarray) or isinstance(matrix, sparse.spmatrix) or isinstance(matrix, sparse.sparray)):
        if layer_name is None:
            if is_raw:
                raise ValueError(f"AnnData raw.X matrix is of invalid type '{type(matrix)}'")
            else:
                raise ValueError(f"AnnData X matrix is of invalid type '{type(matrix)}'")
        else:
            raise ValueError(f"AnnData object layer '{layer_name}' is of invalid type '{type(matrix)}'")
    
    correct_type_matrix: MatrixType = matrix
    
    if correct_type_matrix.min() < 0: # pyright: ignore[reportAttributeAccessIssue]
        raise ValueError("Values in the provided Anndata matrix cannot be negative")
    
    if is_raw and not is_integer_matrix(correct_type_matrix):
        raise ValueError("Values in the provided Anndata raw matrix must be integers")
    
    return correct_type_matrix


class CellChat:
    def __init__(
        self, 
        adata: ad.AnnData,
        experiment_type: str = "RNA",
        layer: str | None = None,
        counts_layer: str | None = "counts",
        group_by_column: str = "cluster", 
        sample_column: str | None = None, 
    ):
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
        
        matrix = get_adata_matrix_checked(adata, False, layer)
        matrix_raw = get_adata_matrix_checked(adata, False, counts_layer)

        self.adata = ad.AnnData(
            X = matrix,
            layers = {"counts": matrix_raw},
            obs = adata.obs,  # pyright: ignore[reportArgumentType] (Must be DataFrame since adata cannot be backed)
            var = adata.var,  # pyright: ignore[reportArgumentType] (Must be DataFrame since adata cannot be backed)
        )
        if sample_column is None:
            self.adata.obs["sample"] = "sample1"
            sample_column = "sample"
        
        self.group_by_col = group_by_column
        self.sample_col = sample_column
        self.is_merged = False # TODO: Allow merging datasets
        self.experiment_type = experiment_type
        
        self.selected_features = None
        self.selected_features_df = None
        
        self.adata_signaling = None
    
    # Modeling methods
    
    def subset_data(
        self,
        features: list[str] | None = None,
    ):
        #TODO: actually do subset data
        print("WARNING (TODO): not performing any subsetting!")
        self.adata_signaling = self.adata
    
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
        if self.adata_signaling is None:
            raise ValueError("Must call CellChat.subset_data() first!")
        
        if features is None:
            features = self.adata.var_names
        else:
            features = self.adata.var_names.intersection(features)
        adata = self.adata[:, features] # pyright: ignore[reportArgumentType]
        
        if adata.X is None:
            raise ValueError("Anndata X cannot be None")
        X = adata.X
        if sparse.issparse(X):
            X = X.tocsr()
        
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
            cell_masks = [(
                ",".join(positive_samples), 
                 cells_in_positive.to_numpy(), 
                 ~cells_in_positive.to_numpy()
            )]
        else:
            groups = adata.obs[self.group_by_col]
            unique_groups = groups.unique()
            cell_masks = []
            for g in unique_groups:
                if positive_samples is None:
                    cells_in_case = (groups == g).to_numpy()
                    cells_in_control = (groups != g).to_numpy()
                else:
                    cells_in_case = ((groups == g) & cells_in_positive).to_numpy()
                    cells_in_control = ((groups == g) & ~cells_in_positive).to_numpy()
                cell_masks.append((g, cells_in_case, cells_in_control))
        
        #* Perform thresholding and DE for chosen groups
        kept_features = pd.DataFrame()
        from tqdm import tqdm
        import time
        for m in tqdm(cell_masks):
            t = time.time()
            
            current_feature_mask = np.full(len(features), True)
            group_name = m[0]
            cells_in_case = m[1]
            cells_in_control = m[2]
            
            # percent expression threshold
            if threshold_percent_expressing > 0:
                percent_expressing_in_case = np.mean(X[cells_in_case, :] > 0, axis=0) * 100
                percent_expressing_in_control = np.mean(X[cells_in_control, :] > 0, axis=0) * 100
                max_percent = np.maximum(percent_expressing_in_case, percent_expressing_in_control)
                
                current_feature_mask &= max_percent > threshold_percent_expressing
                if np.sum(current_feature_mask) == 0:
                    print(f"No features passed the percent_expressing threshold for '{group_name}'")
                    continue
            
            t2 = time.time()
            print("Percent expression time:", t2 - t)
            t = t2
            
            # Always calculate logfoldchange to use in features dataframe, even if no threshold is used.
            # (since the data is already log-normalized, the difference is equivalent to a log fold change)
            if sparse.issparse(X):
                def lognorm_mean_function(x): 
                    return np.log(np.mean(np.exp(x.toarray()), axis=0))
            else:
                def lognorm_mean_function(x): 
                    return np.log(np.mean(np.exp(x), axis=0))
            
            avg_in_case = lognorm_mean_function(X[cells_in_case, :])
            avg_in_control = lognorm_mean_function(X[cells_in_control, :])
            log_fold_change = avg_in_case - avg_in_control
            
            # average log fold change threshold 
            if threshold_logfc > 0:
                current_feature_mask &= (np.abs(log_fold_change) > threshold_logfc)
                if np.sum(current_feature_mask) == 0:
                    print(f"No features passed the log fold change threshold for '{group_name}'")
                    continue
            
            features = features[current_feature_mask]
            log_fold_change = log_fold_change[current_feature_mask]
            
            t2 = time.time()
            print("Logfc time:", t2 - t)
            t = t2
            
            # pvalue threshold
            def sparse_mannwhitneyu(case_data, control_data, chunk_size=1000):
                n_genes = case_data.shape[1]
                n_case = case_data.shape[0]
                n_control = control_data.shape[0]
                
                # Pre-allocate buffers (The "Largest Possible" chunks)
                # We use float32 to match your expression data
                case_buffer = np.empty((n_case, chunk_size), dtype=np.float32)
                control_buffer = np.empty((n_control, chunk_size), dtype=np.float32)
                
                pvalues = np.empty(n_genes)

                for start in range(0, n_genes, chunk_size):
                    end = min(start + chunk_size, n_genes)
                    actual_chunk_width = end - start
                    
                    # Slicing the buffer if the last chunk is smaller than chunk_size
                    current_case_view = case_buffer[:, :actual_chunk_width]
                    current_control_view = control_buffer[:, :actual_chunk_width]
                    
                    # Fill the pre-allocated buffers with the sparse data
                    # .toarray() can take an 'out' argument in some scipy versions, 
                    # but the most robust way is:
                    current_case_view[:] = case_data[:, start:end].toarray()
                    current_control_view[:] = control_data[:, start:end].toarray()
                    
                    # Run MWU
                    res = stats.mannwhitneyu(current_case_view, current_control_view, axis=0)
                    pvalues[start:end] = res.pvalue
                    
                return pvalues
                        
            case_data = adata[cells_in_case, current_feature_mask].X
            control_data = adata[cells_in_control, current_feature_mask].X
            if sparse.issparse(adata.X):
                pvalues = sparse_mannwhitneyu(case_data.tocsc(), control_data.tocsc())
            else:
                pvalues = stats.mannwhitneyu(case_data, control_data, axis=0).pvalue
            # TODO: Look into method='by'
            padj = stats.false_discovery_control(pvalues, method='bh') 
            #? Why does CellChat use pvalue instead of padj for threshold
            significant_feature_mask = pvalues < threshold_p
            
            # only pos threshold
            if only_pos:
                significant_feature_mask &= log_fold_change > 0
            
            t2 = time.time()
            print("MannwhitneyU time:", t2 - t)
            t = t2
            
            # create the dataframe
            kept_features = pd.concat([kept_features, pd.DataFrame({
                "group": group_name,
                "feature": features[significant_feature_mask],
                "logfc": log_fold_change[significant_feature_mask],
                "pvalue": pvalues[significant_feature_mask],
                "padj": padj[significant_feature_mask],
            })])
            
            t2 = time.time()
            print("Create df time:", t2 - t)
            t = t2
        
        if inplace:
            self.selected_features = kept_features["feature"].to_numpy()
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
        Groups that are added and do not exist in the data are defined as empty.
        WARNING: This function will modify the data stored in the CellChat object.
        
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
