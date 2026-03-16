home <- path.expand("~")
py_bin <- file.path(home, "miniforge3", "envs", "pyCellChat", "bin", "python")

library(reticulate)
use_python(py_bin, required = TRUE)
cfg <- py_config()
print(cfg)

library(SingleCellExperiment)
library(CellChat)
library(anndataR)

sce <- read_h5ad("tests/data/pbmc3k.h5ad", as = "SingleCellExperiment")
assayNames(sce)

expr <- as.matrix(assay(sce, 'X'))
meta <- as.data.frame(colData(sce))

keep <- !is.na(meta$cell_type)
meta <- meta[keep, , drop = FALSE]
expr <- expr[, keep]
meta$cell_type <- factor(meta$cell_type)
meta$samples <- meta$sample

# Check dimensions BEFORE createCellChat
print(paste("expr dims:", nrow(expr), "x", ncol(expr)))
print(paste("meta dims:", nrow(meta), "x", ncol(meta)))
print(paste("expr colnames length:", length(colnames(expr))))
print(paste("meta rownames length:", length(rownames(meta))))

# Check if they match
print("First 5 expr colnames:")
print(head(colnames(expr)))
print("First 5 meta rownames:")
print(head(rownames(meta)))

# Check if they're the same
print(paste("colnames match rownames:", all(colnames(expr) == rownames(meta))))

db <- CellChatDB.human
cellchat <- createCellChat(object = as.matrix(expr), meta = meta, group.by = "cell_type")
cellchat@DB <- db

genes.db <- unique(c(
  cellchat@DB$interaction$ligand,
  unlist(strsplit(cellchat@DB$interaction$receptor, "_"))
))


sum(rownames(expr) %in% genes.db)           # <- should be thousands; if ~0, names don't match
sum(genes.db %in% rownames(expr))           # <- should be thousands; if ~0, names don't match
head(setdiff(genes.db, rownames(expr)), 20) # peek at what's missing
head(setdiff(rownames(expr), genes.db), 20) # peek at your naming style

cellchat <- subsetData(cellchat)
cellchat <- identifyOverExpressedGenes(cellchat)

over_expressed_info = cellchat@var.features$features.info
write.csv(over_expressed_info, file = "tests/data/benchmark/overexpressed_info.csv", row.names = FALSE)
