if (!exists("GROUND_TRUTH_BASE_RAN")) source("r_scripts/base_ground_truth.R")

writeLines("\nGenerating ground truths for: 'identify over expressed genes'\n")

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

run_synthetic_de_case <- function(return.object = TRUE, ...) {
  cellchat <- make_synthetic_cellchat()
  identifyOverExpressedGenes(
    cellchat,
    do.fast = FALSE,
    return.object = return.object,
    ...
  )
}

run_pbmc_de_case <- function(features, return.object = TRUE, ...) {
  cellchat <- createCellChat(object = pbmc3k_expr, meta = pbmc3k_meta, group.by = "cell_type")
  cellchat@DB <- CellChatDB.human
  cellchat <- subsetData(cellchat, features = features)
  identifyOverExpressedGenes(
    cellchat,
    features = features,
    do.fast = FALSE,
    return.object = return.object,
    ...
  )
}

# ── identifyOverExpressedGenes: synthetic ───────────────────────────────────

synthetic_no_de_case <- function(case) {
  # print(paste("running synthetic no de case:", str(case)))
  params <- ground_truth_case_params(case)

  de_cellchat <- createCellChat(object = synthetic_expr, meta = synthetic_meta, group.by = "cell_type")
  de_cellchat <- subsetData(de_cellchat, features = rownames(synthetic_expr))
  de_cellchat <- identifyOverExpressedGenes(
    de_cellchat,
    do.DE = FALSE,
    min.cells = params$`min.cells`,
    return.object = TRUE
  )
  write_ground_truth_synthetic(
    ground_truth_case_output_relative_path(case),
    list(
      selected_features = as.list(de_cellchat@var.features$features)
    )
  )
}

result <- lapply(
  Filter(
    function(case) {
      grepl(
        "identify_over_expressed_genes_no_de_synthetic",
        normalize_ground_truth_path(case$nodeid[[1]])
      )
    },
    ground_truth_cases_for_script("r_scripts/generate_iog_ground_truth.R")
  ), 
  synthetic_no_de_case
)

synthetic_de_case <- function(case) {
  # print(paste("running synthetic de case:", str(case)))
  params <- ground_truth_case_params(case)
  thresh_pc <- params$`thresh.pc`
  if (!is.null(thresh_pc) && thresh_pc > 1) {
    thresh_pc <- thresh_pc / 100
  }

  de_cellchat <- createCellChat(object = synthetic_expr, meta = synthetic_meta, group.by = "cell_type")
  de_cellchat <- subsetData(de_cellchat, features = rownames(synthetic_expr))
  de_cellchat <- identifyOverExpressedGenes(
    de_cellchat,
    do.DE = TRUE,
    do.fast = FALSE,
    thresh.pc = thresh_pc,
    thresh.fc = params$`thresh.fc`,
    thresh.p = params$`thresh.p`,
    only.pos = params$`only.pos`,
    return.object = TRUE
  )
  write_ground_truth_synthetic(
    ground_truth_case_output_relative_path(case),
    list(
      selected_features = as.list(de_cellchat@var.features$features),
      feature_table = feature_signature(de_cellchat@var.features$features.info)
    )
  )
}

result <- lapply(
  Filter(
    function(case) {
      grepl(
        "identify_over_expressed_genes_de_synthetic",
        normalize_ground_truth_path(case$nodeid[[1]])
      )
    },
    ground_truth_cases_for_script("r_scripts/generate_iog_ground_truth.R")
  ), 
  synthetic_de_case
)


feature_subset_cellchat <- run_synthetic_de_case(
  features = c("g1", "g4", "g7", "missing"),
  thresh.p = 1
)
write_ground_truth_synthetic(
  ground_truth_function_path("identify_over_expressed_genes", "identify_over_expressed_genes_de_feature_subset.json"),
  list(
    selected_features = as.list(feature_subset_cellchat@var.features$features),
    feature_table = feature_signature(feature_subset_cellchat@var.features$features.info)
  )
)


returned_features <- run_synthetic_de_case(return.object = FALSE, thresh.p = 1)
write_ground_truth_synthetic(
  ground_truth_function_path("identify_over_expressed_genes", "identify_over_expressed_genes_de_inplace_false.json"),
  list(
    returned_features = as.list(as.character(returned_features$features))
  )
)


# ── identifyOverExpressedGenes: pbmc3k ──────────────────────────────────────

no_de_cellchat <- createCellChat(object = pbmc3k_expr, meta = pbmc3k_meta, group.by = "cell_type")
no_de_cellchat <- subsetData(no_de_cellchat, features = pbmc3k_marker_panel)
no_de_cellchat <- identifyOverExpressedGenes(
  no_de_cellchat,
  features = pbmc3k_marker_panel,
  do.DE = FALSE,
  min.cells = 10,
  return.object = TRUE
)
write_ground_truth_pbmc(
  ground_truth_function_path("identify_over_expressed_genes", "identify_over_expressed_genes_no_de_marker_panel.json"),
  list(
    selected_features = as.list(no_de_cellchat@var.features$features)
  )
)

default_de_cellchat <- run_pbmc_de_case(pbmc3k_marker_panel, thresh.p = 1)
write_ground_truth_pbmc(
  ground_truth_function_path("identify_over_expressed_genes", "identify_over_expressed_genes_de_marker_panel.json"),
  list(
    selected_features = as.list(default_de_cellchat@var.features$features),
    feature_table = feature_signature(default_de_cellchat@var.features$features.info)
  )
)

threshold_logfc_cellchat <- run_pbmc_de_case(pbmc3k_marker_panel, thresh.fc = 1, thresh.p = 1)
write_ground_truth_pbmc(
  ground_truth_function_path("identify_over_expressed_genes", "identify_over_expressed_genes_de_marker_panel_threshold_logfc.json"),
  list(
    selected_features = as.list(threshold_logfc_cellchat@var.features$features),
    feature_table = feature_signature(threshold_logfc_cellchat@var.features$features.info)
  )
)

only_pos_false_cellchat <- run_pbmc_de_case(pbmc3k_marker_panel, only.pos = FALSE, thresh.p = 1)
write_ground_truth_pbmc(
  ground_truth_function_path("identify_over_expressed_genes", "identify_over_expressed_genes_de_marker_panel_only_pos_false.json"),
  list(
    selected_features = as.list(only_pos_false_cellchat@var.features$features),
    feature_table = feature_signature(only_pos_false_cellchat@var.features$features.info)
  )
)

returned_features <- run_pbmc_de_case(pbmc3k_marker_panel, return.object = FALSE, thresh.p = 1)
write_ground_truth_pbmc(
  ground_truth_function_path("identify_over_expressed_genes", "identify_over_expressed_genes_de_marker_panel_inplace_false.json"),
  list(
    returned_features = as.list(as.character(returned_features$features))
  )
)

default_p_cellchat <- run_pbmc_de_case(pbmc3k_marker_panel)
write_ground_truth_pbmc(
  ground_truth_function_path("identify_over_expressed_genes", "identify_over_expressed_genes_de_marker_panel_default_p.json"),
  list(
    selected_features = as.list(default_p_cellchat@var.features$features),
    feature_table = feature_signature(default_p_cellchat@var.features$features.info)
  )
)

threshold_percent_cellchat <- run_pbmc_de_case(pbmc3k_marker_panel, thresh.pc = 0.10, thresh.p = 1)
write_ground_truth_pbmc(
  ground_truth_function_path("identify_over_expressed_genes", "identify_over_expressed_genes_de_marker_panel_threshold_percent.json"),
  list(
    selected_features = as.list(threshold_percent_cellchat@var.features$features),
    feature_table = feature_signature(threshold_percent_cellchat@var.features$features.info)
  )
)

full_gene_cellchat <- createCellChat(object = pbmc3k_expr, meta = pbmc3k_meta, group.by = "cell_type")
full_gene_cellchat@DB <- CellChatDB.human
full_gene_cellchat <- subsetData(full_gene_cellchat)
full_gene_cellchat <- identifyOverExpressedGenes(
  full_gene_cellchat,
  do.fast = FALSE,
  return.object = TRUE
)
write_ground_truth_pbmc(
  ground_truth_function_path("identify_over_expressed_genes", "identify_over_expressed_genes_de_full_gene_set.json"),
  list(
    selected_features = as.list(full_gene_cellchat@var.features$features),
    feature_table = feature_signature(full_gene_cellchat@var.features$features.info)
  )
)
