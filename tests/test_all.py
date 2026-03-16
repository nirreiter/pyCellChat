import anndata as ad
from py_cellchat import CellChat

def test_constructor():
    adata = ad.read_h5ad("tests/data/pbmc3k.h5ad")
    _ = CellChat(adata, group_by_column="cell_type")

def test_overexpressed_genes():
    adata = ad.read_h5ad("tests/data/pbmc3k.h5ad")
    cellchat = CellChat(adata, group_by_column="cell_type")
    cellchat.subset_data()
    cellchat.identify_over_expressed_genes()
    print(cellchat.selected_features_df)
