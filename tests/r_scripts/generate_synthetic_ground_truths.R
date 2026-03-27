library(jsonlite)
library(CellChat)

ground_truth_dir <- file.path("data/synthetic")
dir.create(ground_truth_dir, recursive = TRUE, showWarnings = FALSE)

write_ground_truth <- function(name, payload) {
  write_json(
    payload,
    path = file.path(ground_truth_dir, name),
    auto_unbox = TRUE,
    pretty = TRUE
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

run_de_case <- function(...) {
  cellchat <- createCellChat(object = expr, meta = meta, group.by = "cell_type")
  cellchat <- subsetData(cellchat, features = rownames(expr))
  identifyOverExpressedGenes(
    cellchat,
    do.fast = FALSE,
    return.object = TRUE,
    ...
  )
}

run_de_case_return <- function(...) {
  cellchat <- createCellChat(object = expr, meta = meta, group.by = "cell_type")
  cellchat <- subsetData(cellchat, features = rownames(expr))
  identifyOverExpressedGenes(
    cellchat,
    do.fast = FALSE,
    return.object = FALSE,
    ...
  )
}

expr <- matrix(
  c(
    6, 5, 6, 5, 0, 0, 0, 0,
    0, 0, 0, 0, 6, 5, 6, 5,
    1, 0, 0, 0, 1, 0, 0, 0,
    1, 1, 1, 1, 0, 0, 0, 0,
    5, 4, 0, 0, 5, 4, 0, 0,
    0, 0, 5, 4, 0, 0, 5, 4,
    4, 0, 0, 0, 0, 0, 0, 0
  ),
  nrow = 7,
  byrow = TRUE,
  dimnames = list(
    c("g1", "g2", "g3", "g4", "g5", "g6", "g7"),
    c("A_s1_1", "A_s1_2", "A_s2_1", "A_s2_2", "B_s1_1", "B_s1_2", "B_s2_1", "B_s2_2")
  )
)

meta <- data.frame(
  cell_type = factor(c("A", "A", "A", "A", "B", "B", "B", "B"), levels = c("B", "A"), ordered = TRUE),
  samples = factor(c("s1", "s1", "s2", "s2", "s1", "s1", "s2", "s2"), levels = c("s1", "s2"), ordered = TRUE),
  row.names = c("A_s1_1", "A_s1_2", "A_s2_1", "A_s2_2", "B_s1_1", "B_s1_2", "B_s2_1", "B_s2_2")
)

cellchat <- createCellChat(object = expr, meta = meta, group.by = "cell_type")

write_ground_truth(
  "constructor_summary.json",
  list(
    n_genes = nrow(cellchat@data),
    n_cells = ncol(cellchat@data),
    group_categories = as.list(levels(cellchat@meta$cell_type)),
    sample_categories = as.list(levels(cellchat@meta$samples))
  )
)

subset_cellchat <- subsetData(cellchat, features = c("g2", "missing", "g7"))
write_ground_truth(
  "subset_data_explicit.json",
  list(
    signaling_features = as.list(rownames(subset_cellchat@data.signaling))
  )
)

de_cellchat <- subsetData(cellchat, features = rownames(expr))
de_cellchat <- identifyOverExpressedGenes(
  de_cellchat,
  do.DE = FALSE,
  min.cells = 2,
  return.object = TRUE
)
write_ground_truth(
  "identify_over_expressed_genes_no_de.json",
  list(
    selected_features = as.list(de_cellchat@var.features$features)
  )
)

default_de_cellchat <- run_de_case(thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_p_1.0.json",
  list(
    selected_features = as.list(default_de_cellchat@var.features$features),
    feature_table = feature_signature(default_de_cellchat@var.features$features.info)
  )
)

feature_subset_cellchat <- run_de_case(features = c("g1", "g4", "g7", "missing"), thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_feature_subset.json",
  list(
    selected_features = as.list(feature_subset_cellchat@var.features$features),
    feature_table = feature_signature(feature_subset_cellchat@var.features$features.info)
  )
)

threshold_percent_cellchat <- run_de_case(thresh.pc = 0.30, thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_threshold_percent.json",
  list(
    selected_features = as.list(threshold_percent_cellchat@var.features$features),
    feature_table = feature_signature(threshold_percent_cellchat@var.features$features.info)
  )
)

threshold_logfc_cellchat <- run_de_case(thresh.fc = 2, thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_threshold_logfc.json",
  list(
    selected_features = as.list(threshold_logfc_cellchat@var.features$features),
    feature_table = feature_signature(threshold_logfc_cellchat@var.features$features.info)
  )
)

threshold_p_zero_cellchat <- run_de_case(thresh.p = 0)
write_ground_truth(
  "identify_over_expressed_genes_de_threshold_p_zero.json",
  list(
    selected_features = as.list(threshold_p_zero_cellchat@var.features$features),
    feature_table = feature_signature(threshold_p_zero_cellchat@var.features$features.info)
  )
)

only_pos_false_cellchat <- run_de_case(only.pos = FALSE, thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_only_pos_false.json",
  list(
    selected_features = as.list(only_pos_false_cellchat@var.features$features),
    feature_table = feature_signature(only_pos_false_cellchat@var.features$features.info)
  )
)

returned_features <- run_de_case_return(thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_inplace_false.json",
  list(
    returned_features = as.list(as.character(returned_features$features))
  )
)

# min_cells=3: second value for the no-DE path (g3 has only 2 expressing cells
# and is therefore excluded, unlike min_cells=2 where it passes)
no_de_min_cells_3 <- subsetData(cellchat, features = rownames(expr))
no_de_min_cells_3 <- identifyOverExpressedGenes(
  no_de_min_cells_3,
  do.DE = FALSE,
  min.cells = 3,
  return.object = TRUE
)
write_ground_truth(
  "identify_over_expressed_genes_no_de_min_cells_3.json",
  list(
    selected_features = as.list(no_de_min_cells_3@var.features$features)
  )
)

# threshold_logfc=0.5: second logfc value (existing test uses 2.0)
threshold_logfc_half_cellchat <- run_de_case(thresh.fc = 0.5, thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_threshold_logfc_half.json",
  list(
    selected_features = as.list(threshold_logfc_half_cellchat@var.features$features),
    feature_table = feature_signature(threshold_logfc_half_cellchat@var.features$features.info)
  )
)

# threshold_p=0.5: intermediate p-value (existing tests use only 0.0 and 1.0)
threshold_p_half_cellchat <- run_de_case(thresh.p = 0.5)
write_ground_truth(
  "identify_over_expressed_genes_de_threshold_p_half.json",
  list(
    selected_features = as.list(threshold_p_half_cellchat@var.features$features),
    feature_table = feature_signature(threshold_p_half_cellchat@var.features$features.info)
  )
)

# threshold_percent_expressing=30% AND threshold_logfc=0.5 applied together
threshold_pct_and_logfc_cellchat <- run_de_case(thresh.pc = 0.30, thresh.fc = 0.5, thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_threshold_pct_and_logfc.json",
  list(
    selected_features = as.list(threshold_pct_and_logfc_cellchat@var.features$features),
    feature_table = feature_signature(threshold_pct_and_logfc_cellchat@var.features$features.info)
  )
)

# only_pos=FALSE with threshold_logfc=0.5: tests negative-direction filtering
only_pos_false_logfc_cellchat <- run_de_case(only.pos = FALSE, thresh.fc = 0.5, thresh.p = 1)
write_ground_truth(
  "identify_over_expressed_genes_de_only_pos_false_logfc.json",
  list(
    selected_features = as.list(only_pos_false_logfc_cellchat@var.features$features),
    feature_table = feature_signature(only_pos_false_logfc_cellchat@var.features$features.info)
  )
)
