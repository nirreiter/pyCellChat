if (!exists("GROUND_TRUTH_BASE_RAN")) source("r_scripts/base_ground_truth.R")

writeLines("\nGenerating ground truths for: 'filter communication'\n")

# Use canonical PBMC baseline to build and filter communications

# ── filterCommunication ─────────────────────────────────────────────────────

cellchat <- make_pbmc3k_cellchat()
cellchat <- identifyOverExpressedGenes(cellchat, do.fast = FALSE)
cellchat <- identifyOverExpressedInteractions(cellchat)
cellchat <- computeCommunProb(cellchat)

filter_communication_cellchat <- filterCommunication(cellchat)

write_ground_truth_pbmc(
  ground_truth_function_path("filter_communication", "filter_communication_default.json"),
  list(
    groups = as.list(dimnames(filter_communication_cellchat@net$prob)[[1]]),
    lr_names = as.list(dimnames(filter_communication_cellchat@net$prob)[[3]]),
    prob_shape = as.list(dim(filter_communication_cellchat@net$prob)),
    pval_shape = as.list(dim(filter_communication_cellchat@net$pval)),
    prob_sum = unname(sum(filter_communication_cellchat@net$prob)),
    pval_sum = unname(sum(filter_communication_cellchat@net$pval)),
    prob_nonzero = unname(sum(filter_communication_cellchat@net$prob > 0)),
    nonzero_communications = communication_signature(filter_communication_cellchat)
  )
)
