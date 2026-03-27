python_bin <- Sys.getenv("RETICULATE_PYTHON")
if (python_bin == "") {
  python_bin <- Sys.which("python")
}
if (python_bin != "") {
  library(reticulate)
  use_python(python_bin, required = TRUE)
}

library(jsonlite)
library(SingleCellExperiment)
library(CellChat)
library(anndataR)

ground_truth_dir <- file.path("data", "pbmc3k_benchmark")
dir.create(ground_truth_dir, recursive = TRUE, showWarnings = FALSE)

write_ground_truth <- function(name, payload) {
  write_json(
    payload,
    path = file.path(ground_truth_dir, name),
    auto_unbox = TRUE,
    pretty = TRUE,
    digits = 5,
  )
}

feature_signature <- function(df) {
  if (is.null(df) || nrow(df) == 0) {
    return(list())
  }

  normalized <- data.frame(
    group = as.character(df$clusters),
    feature = as.character(df$features),
    logFC = df$logFC,
    stringsAsFactors = FALSE
  )
  normalized <- normalized[order(normalized$group, normalized$feature, normalized$logFC), , drop = FALSE]

  lapply(seq_len(nrow(normalized)), function(i) {
    list(
      group = normalized$group[[i]],
      feature = normalized$feature[[i]],
      logFC = normalized$logFC[[i]]
    )
  })
}

run_de_case <- function(features, ...) {
  cellchat <- createCellChat(object = expr, meta = meta, group.by = "cell_type")
  cellchat <- subsetData(cellchat, features = features)
  identifyOverExpressedGenes(
    cellchat,
    features = features,
    do.fast = FALSE,
    return.object = TRUE,
    ...
  )
}

run_de_case_return <- function(features, ...) {
  cellchat <- createCellChat(object = expr, meta = meta, group.by = "cell_type")
  cellchat <- subsetData(cellchat, features = features)
  identifyOverExpressedGenes(
    cellchat,
    features = features,
    do.fast = FALSE,
    return.object = FALSE,
    ...
  )
}

sce <- read_h5ad("data/pbmc3k/pbmc3k.h5ad", as = "SingleCellExperiment")

expr <- as.matrix(assay(sce, 'X'))
meta <- as.data.frame(colData(sce))

keep <- !is.na(meta$cell_type)
meta <- meta[keep, , drop = FALSE]
expr <- expr[, keep]
meta$cell_type <- factor(meta$cell_type)
marker_panel <- intersect(
  c(
    "CD3D",
    "IL7R",
    "NKG7",
    "GNLY",
    "LYZ",
    "CST3",
    "FCER1A",
    "FCGR3A",
    "MS4A1",
    "CD79A",
    "PPBP"
  ),
  rownames(expr)
)

if (length(marker_panel) < 6) {
  stop("PBMC3K marker panel unexpectedly small; expected at least 6 genes")
}

expr <- as.matrix(expr)

no_de_cellchat <- createCellChat(object = expr, meta = meta, group.by = "cell_type")
no_de_cellchat <- subsetData(no_de_cellchat, features = marker_panel)
no_de_cellchat <- identifyOverExpressedGenes(
  no_de_cellchat,
  features = marker_panel,
  do.DE = FALSE,
  min.cells = 10,
  return.object = TRUE
)
write_ground_truth(
  "identify_over_expressed_genes_no_de_marker_panel.json",
  list(
    selected_features = as.list(no_de_cellchat@var.features$features)
  )
)

default_de_cellchat <- run_de_case(marker_panel, thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_marker_panel.json",
  list(
    selected_features = as.list(default_de_cellchat@var.features$features),
    feature_table = feature_signature(default_de_cellchat@var.features$features.info)
  )
)

threshold_logfc_cellchat <- run_de_case(marker_panel, thresh.fc = 1, thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_marker_panel_threshold_logfc.json",
  list(
    selected_features = as.list(threshold_logfc_cellchat@var.features$features),
    feature_table = feature_signature(threshold_logfc_cellchat@var.features$features.info)
  )
)

only_pos_false_cellchat <- run_de_case(marker_panel, only.pos = FALSE, thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_marker_panel_only_pos_false.json",
  list(
    selected_features = as.list(only_pos_false_cellchat@var.features$features),
    feature_table = feature_signature(only_pos_false_cellchat@var.features$features.info)
  )
)

returned_features <- run_de_case_return(marker_panel, thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_marker_panel_inplace_false.json",
  list(
    returned_features = as.list(as.character(returned_features$features))
  )
)

# ── New ground-truth fixtures ──────────────────────────────────────────────

# Default threshold_p=0.05: the real default is never exercised by any existing
# fixture, which all use thresh.p=0 or thresh.p=1.
default_p_cellchat <- run_de_case(marker_panel)
write_ground_truth(
  "identify_over_expressed_genes_de_marker_panel_default_p.json",
  list(
    selected_features = as.list(default_p_cellchat@var.features$features),
    feature_table = feature_signature(default_p_cellchat@var.features$features.info)
  )
)

# threshold_percent_expressing=10%: validates zero-inflation filtering on real
# scRNA-seq data, where dropout patterns differ from the synthetic fixture.
threshold_percent_cellchat <- run_de_case(marker_panel, thresh.pc = 0.10, thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_marker_panel_threshold_percent.json",
  list(
    selected_features = as.list(threshold_percent_cellchat@var.features$features),
    feature_table = feature_signature(threshold_percent_cellchat@var.features$features.info)
  )
)

# Full gene set in Cellchat DB
full_gene_cellchat <- createCellChat(object = expr, meta = meta, group.by = "cell_type")
full_gene_cellchat@DB <- CellChatDB.human
full_gene_cellchat <- subsetData(full_gene_cellchat)
full_gene_cellchat <- identifyOverExpressedGenes(
  full_gene_cellchat,
  do.fast = FALSE,
  return.object = TRUE
)
write_ground_truth(
  "identify_over_expressed_genes_de_full_gene_set.json",
  list(
    selected_features = as.list(full_gene_cellchat@var.features$features),
    feature_table = feature_signature(full_gene_cellchat@var.features$features.info)
  )
)
