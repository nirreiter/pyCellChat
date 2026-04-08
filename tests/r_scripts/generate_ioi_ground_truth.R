if (!exists("GROUND_TRUTH_BASE_RAN")) source("r_scripts/base_ground_truth.R")

writeLines("\nGenerating ground truths for: 'identify over expressed interactions'\n")

ioi_base_cellchat <- createCellChat(object = pbmc3k_expr, meta = pbmc3k_meta, group.by = "cell_type")
ioi_base_cellchat@DB <- CellChatDB.human
ioi_base_cellchat <- subsetData(ioi_base_cellchat)
ioi_base_cellchat <- identifyOverExpressedGenes(
  ioi_base_cellchat,
  min.cells = 10,
  do.fast = FALSE
)

ioi_variable_both_cellchat <- identifyOverExpressedInteractions(
  ioi_base_cellchat,
  variable.both = TRUE
)
write_ground_truth_pbmc(
  ground_truth_function_path(
    "identify_over_expressed_interactions",
    "identify_over_expressed_interactions_variable_both.json"
  ),
  list(
    lr_sig_names = as.list(rownames(ioi_variable_both_cellchat@LR$LRsig))
  )
)

ioi_variable_one_cellchat <- identifyOverExpressedInteractions(
  ioi_base_cellchat,
  variable.both = FALSE
)
write_ground_truth_pbmc(
  ground_truth_function_path(
    "identify_over_expressed_interactions",
    "identify_over_expressed_interactions_variable_one.json"
  ),
  list(
    lr_sig_names = as.list(rownames(ioi_variable_one_cellchat@LR$LRsig))
  )
)

ioi_explicit_cellchat <- createCellChat(object = pbmc3k_expr, meta = pbmc3k_meta, group.by = "cell_type")
ioi_explicit_cellchat@DB <- CellChatDB.human
ioi_explicit_cellchat <- subsetData(ioi_explicit_cellchat)
ioi_explicit_cellchat <- identifyOverExpressedInteractions(
  ioi_explicit_cellchat,
  features = pbmc3k_marker_panel
)
write_ground_truth_pbmc(
  ground_truth_function_path(
    "identify_over_expressed_interactions",
    "identify_over_expressed_interactions_explicit_features.json"
  ),
  list(
    lr_sig_names = as.list(rownames(ioi_explicit_cellchat@LR$LRsig))
  )
)
