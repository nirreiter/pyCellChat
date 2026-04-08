## Base ground-truth bootstrap.
## This file is safe to `source()` multiple times; it will only perform
## expensive initialization once per R session. Generator scripts should do:
##
## if (!exists("GROUND_TRUTH_BASE_RAN")) source("r_scripts/base_ground_truth.R")
##
if (!exists("GROUND_TRUTH_BASE_RAN")) {
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

    synthetic_ground_truth_dir <- file.path("data", "synthetic")
    pbmc_ground_truth_dir <- file.path("data", "pbmc3k_benchmark")
    dir.create(synthetic_ground_truth_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(pbmc_ground_truth_dir, recursive = TRUE, showWarnings = FALSE)
    ground_truth_manifest_path <- file.path(getwd(), "_r_ground_truth_manifest.json")

    normalize_ground_truth_path <- function(path) {
        gsub("\\\\", "/", path)
    }

    load_ground_truth_manifest <- function() {
        if (!file.exists(ground_truth_manifest_path)) {
            return(list(cases = list()))
        }

        manifest <- read_json(ground_truth_manifest_path, simplifyVector = FALSE)
        if (is.null(manifest$cases)) {
            manifest$cases <- list()
        }
        manifest
    }

    ground_truth_cases_for_script <- function(script_path) {
        manifest <- load_ground_truth_manifest()
        normalized_script <- normalize_ground_truth_path(script_path)

        Filter(
            function(case) identical(normalize_ground_truth_path(case$r_script), normalized_script),
            manifest$cases
        )
    }

    ground_truth_case_params <- function(case) {
        if (is.null(case$params)) {
            return(list())
        }
        case$params
    }

    ground_truth_case_output_relative_path <- function(case) {
        if (is.null(case$ground_truth) || length(case$ground_truth) == 0) {
            stop("Ground-truth case is missing output names.")
        }

        output_path <- normalize_ground_truth_path(case$ground_truth[[1]])
        if (startsWith(output_path, "synthetic/")) {
            return(sub("^synthetic/", "", output_path))
        }
        if (startsWith(output_path, "pbmc3k_benchmark/")) {
            return(sub("^pbmc3k_benchmark/", "", output_path))
        }

        stop(
            paste(
                "Ground-truth output path must start with 'synthetic/' or",
                paste0("'pbmc3k_benchmark/'; got '", output_path, "'.")
            )
        )
    }

    ground_truth_case_output_name <- ground_truth_case_output_relative_path

    ground_truth_function_path <- function(function_name, file_name) {
        file.path(function_name, file_name)
    }

    # Generic writer: explicit dir + name
    write_ground_truth_file <- function(dir_path, name, payload) {
        output_path <- file.path(dir_path, name)
        dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
        write_json(
        payload,
        path = output_path,
        auto_unbox = TRUE,
        pretty = TRUE,
        digits = 5
        )
    }

    # Convenience writers for common targets used by generators
    write_ground_truth_pbmc <- function(name, payload) {
        write_ground_truth_file(pbmc_ground_truth_dir, name, payload)
    }
    write_ground_truth_synthetic <- function(name, payload) {
        write_ground_truth_file(synthetic_ground_truth_dir, name, payload)
    }

    # Load PBMC baseline expression/meta (kept in memory as pbmc3k_expr / pbmc3k_meta)
    pbmc3k_sce <- read_h5ad("data/pbmc3k/pbmc3k.h5ad", as = "SingleCellExperiment")
    pbmc3k_expr <- as.matrix(assay(pbmc3k_sce, "X"))
    pbmc3k_meta <- as.data.frame(colData(pbmc3k_sce))

    pbmc3k_keep <- !is.na(pbmc3k_meta$cell_type)
    pbmc3k_meta <- pbmc3k_meta[pbmc3k_keep, , drop = FALSE]
    pbmc3k_expr <- pbmc3k_expr[, pbmc3k_keep, drop = FALSE]
    pbmc3k_meta$cell_type <- factor(pbmc3k_meta$cell_type)

    pbmc3k_marker_panel <- intersect(
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
        rownames(pbmc3k_expr)
    )

    # Synthetic fixture used by multiple generators
    synthetic_expr <- matrix(
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

    synthetic_meta <- data.frame(
        cell_type = factor(c("A", "A", "A", "A", "B", "B", "B", "B"), levels = c("B", "A"), ordered = TRUE),
        samples = factor(c("s1", "s1", "s2", "s2", "s1", "s1", "s2", "s2"), levels = c("s1", "s2"), ordered = TRUE),
        row.names = c("A_s1_1", "A_s1_2", "A_s2_1", "A_s2_2", "B_s1_1", "B_s1_2", "B_s2_1", "B_s2_2")
    )

    # Factory: create a fresh CellChat object from the PBMC baseline for each generator
    make_pbmc3k_cellchat <- function() {
        cellchat <- createCellChat(object = pbmc3k_expr, meta = pbmc3k_meta, group.by = "cell_type")
        cellchat@DB <- CellChatDB.human
        cellchat <- subsetData(cellchat)
        cellchat
    }

    # Factory: create a CellChat object from the canonical synthetic fixture
    make_synthetic_cellchat <- function() {
        cellchat <- createCellChat(object = synthetic_expr, meta = synthetic_meta, group.by = "cell_type")
        # TODO: This shouldn't use a human db!
        cellchat@DB <- CellChatDB.human
        cellchat <- subsetData(cellchat, features = rownames(synthetic_expr))
        cellchat
    }

    # Helper: signature extractor for communication objects
    communication_signature <- function(cellchat) {
        prob <- cellchat@net$prob
        pval <- cellchat@net$pval
        indices <- which(prob > 0, arr.ind = TRUE)

        if (nrow(indices) == 0) {
            return(list())
        }

        groups <- dimnames(prob)[[1]]
        lr_names <- dimnames(prob)[[3]]
        signatures <- vector("list", nrow(indices))

        for (i in seq_len(nrow(indices))) {
            source_idx <- indices[i, 1]
            target_idx <- indices[i, 2]
            lr_idx <- indices[i, 3]
            signatures[[i]] <- list(
                source = groups[[source_idx]],
                target = groups[[target_idx]],
                lr = lr_names[[lr_idx]],
                prob = prob[source_idx, target_idx, lr_idx],
                pval = pval[source_idx, target_idx, lr_idx]
            )
        }

        signatures
    }

    # Mark base as initialized
    GROUND_TRUTH_BASE_RAN <- TRUE
}
