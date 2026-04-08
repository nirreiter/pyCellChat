if (!exists("GROUND_TRUTH_BASE_RAN")) source("r_scripts/base_ground_truth.R")

writeLines("\nGenerating ground truths for: 'compute communication probability'\n")

# Use factory to create a fresh CellChat from the canonical PBMC baseline
# (regenerated each time from `pbmc3k_expr` and `pbmc3k_meta`).

# ── computeCommunProb ───────────────────────────────────────────────────────

cellchat <- make_pbmc3k_cellchat()
cellchat <- identifyOverExpressedGenes(cellchat, do.fast = FALSE)
cellchat <- identifyOverExpressedInteractions(cellchat)

communication_cellchat <- computeCommunProb(cellchat)
write_ground_truth_pbmc(
  ground_truth_function_path(
    "compute_communication_probability",
    "compute_communication_probability.json"
  ),
  list(
    groups = as.list(dimnames(communication_cellchat@net$prob)[[1]]),
    lr_names = as.list(dimnames(communication_cellchat@net$prob)[[3]]),
    prob_shape = as.list(dim(communication_cellchat@net$prob)),
    pval_shape = as.list(dim(communication_cellchat@net$pval)),
    prob_sum = unname(sum(communication_cellchat@net$prob)),
    prob_nonzero = unname(sum(communication_cellchat@net$prob > 0)),
    nonzero_communications = communication_signature(communication_cellchat)
  )
)
