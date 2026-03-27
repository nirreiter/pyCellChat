#!/usr/bin/env Rscript
# Export CellChatDB tables from R .rda files to CSV for Python bundling.
#
# Run from the pyCellChat repo root:
#   Rscript scripts/export_cellchat_db.R
#
# Writes four CSV files per species to:
#   src/py_cellchat/database/data/{species}/
#     interaction.csv
#     complex.csv
#     cofactor.csv
#     gene_info.csv
#
# Requirements: base R only (no CellChat package install needed; reads .rda
# files from the sibling CellChat/ directory).

CELLCHAT_DATA_DIR <- file.path("..", "CellChat", "data")
OUTPUT_BASE <- file.path("src", "py_cellchat", "database", "data")

# Map of Python species key -> .rda file name -> R object name
SPECIES <- list(
  human     = list(file = "CellChatDB.human.rda",     obj = "CellChatDB.human"),
  mouse     = list(file = "CellChatDB.mouse.rda",     obj = "CellChatDB.mouse"),
  zebrafish = list(file = "CellChatDB.zebrafish.rda", obj = "CellChatDB.zebrafish")
)

export_db <- function(species_key, rda_file, obj_name) {
  rda_path <- file.path(CELLCHAT_DATA_DIR, rda_file)
  if (!file.exists(rda_path)) {
    stop(paste("Cannot find .rda file:", rda_path))
  }

  env <- new.env(parent = emptyenv())
  load(rda_path, envir = env)
  db <- get(obj_name, envir = env)

  out_dir <- file.path(OUTPUT_BASE, species_key)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  # --- interaction ---
  # Row names are the interaction_name index used everywhere downstream.
  interaction <- db$interaction
  interaction <- cbind(interaction_name = rownames(interaction), interaction)
  rownames(interaction) <- NULL
  write.csv(interaction, file.path(out_dir, "interaction.csv"), row.names = FALSE)
  cat(sprintf("[%s] interaction: %d rows, %d cols\n",
              species_key, nrow(interaction), ncol(interaction)))

  # --- complex ---
  # Row names are the complex identifiers that extractGeneSubset matches by.
  complex <- db$complex
  complex <- cbind(index = rownames(complex), complex)
  rownames(complex) <- NULL
  write.csv(complex, file.path(out_dir, "complex.csv"), row.names = FALSE)
  cat(sprintf("[%s] complex:     %d rows, %d cols\n",
              species_key, nrow(complex), ncol(complex)))

  # --- cofactor ---
  # Row names are the cofactor identifiers matched by extractGene.
  cofactor <- db$cofactor
  cofactor <- cbind(index = rownames(cofactor), cofactor)
  rownames(cofactor) <- NULL
  write.csv(cofactor, file.path(out_dir, "cofactor.csv"), row.names = FALSE)
  cat(sprintf("[%s] cofactor:    %d rows, %d cols\n",
              species_key, nrow(cofactor), ncol(cofactor)))

  # --- geneInfo ---
  # Row names are not meaningful here; Symbol column is the key.
  gene_info <- db$geneInfo
  write.csv(gene_info, file.path(out_dir, "gene_info.csv"), row.names = FALSE)
  cat(sprintf("[%s] gene_info:   %d rows, %d cols\n",
              species_key, nrow(gene_info), ncol(gene_info)))
}

for (key in names(SPECIES)) {
  spec <- SPECIES[[key]]
  cat(sprintf("\nExporting %s...\n", key))
  export_db(key, spec$file, spec$obj)
}

cat("\nDone. CSV files written to:", OUTPUT_BASE, "\n")
