if (!exists("GROUND_TRUTH_BASE_RAN")) source("r_scripts/base_ground_truth.R")

writeLines("\nGenerating ground truths for: database functions\n")

## Use builders/writers from `base_ground_truth.R` (write_ground_truth_synthetic,
## write_ground_truth_pbmc, build_synthetic_fixture are provided there).

# Use synthetic fixture and writers from base

constructor_cellchat <- createCellChat(object = synthetic_expr, meta = synthetic_meta, group.by = "cell_type")
write_ground_truth_synthetic(
  ground_truth_function_path("cellchat_constructor", "constructor_summary.json"),
  list(
    n_genes = nrow(constructor_cellchat@data),
    n_cells = ncol(constructor_cellchat@data),
    group_categories = as.list(levels(constructor_cellchat@meta$cell_type)),
    sample_categories = as.list(levels(constructor_cellchat@meta$samples))
  )
)

# ── subsetData ──────────────────────────────────────────────────────────────

subset_cellchat <- subsetData(constructor_cellchat, features = c("g2", "missing", "g7"))
write_ground_truth_synthetic(
  ground_truth_function_path("subset_data", "subset_data_explicit.json"),
  list(
    signaling_features = as.list(rownames(subset_cellchat@data.signaling))
  )
)

# ── subsetDB ────────────────────────────────────────────────────────────────

pbmc_expr <- pbmc3k_expr

subset_db_default <- subsetDB(CellChatDB.human)
write_ground_truth_pbmc(
  ground_truth_function_path("subset_db", "subset_db_default.json"),
  list(
    interaction_names = as.list(rownames(subset_db_default$interaction)),
    pbmc3k_overlap_genes = as.list(sort(intersect(extractGene(subset_db_default), rownames(pbmc_expr))))
  )
)

# ── extractGene ─────────────────────────────────────────────────────────────

extract_gene_secreted_cxcl <- subsetDB(
  CellChatDB.human,
  key = c("annotation", "pathway_name"),
  search = list(c("Secreted Signaling"), c("CXCL"))
)
write_ground_truth_pbmc(
  ground_truth_function_path("extract_gene", "extract_gene_secreted_cxcl.json"),
  list(
    extracted_genes = as.list(sort(extractGene(extract_gene_secreted_cxcl))),
    pbmc3k_overlap_genes = as.list(sort(intersect(extractGene(extract_gene_secreted_cxcl), rownames(pbmc_expr))))
  )
)
