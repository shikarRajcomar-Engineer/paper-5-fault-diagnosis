# TEP faulty testing RData -> fault-specific CSV slices
# Run ONLY through 6.2_Convert_TEP_Testing_To_Parquet.py.
# R's full-data load may require several GB of RAM.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2) stop("Usage: Rscript extractor.R INPUT.RData SCRATCH_DIR")
source_file <- args[[1]]
scratch <- args[[2]]
if (!file.exists(source_file)) stop("Missing RData: ", source_file)
dir.create(scratch, recursive=TRUE, showWarnings=FALSE)
if (!requireNamespace("data.table", quietly=TRUE)) stop("Install R package data.table first")
cat("Loading RData once (large file): ", source_file, "\n", sep="")
flush.console()
env <- new.env(parent=globalenv())
loaded <- load(source_file, envir=env)
if (!("faulty_testing" %in% loaded)) stop("RData has no object 'faulty_testing': ", paste(loaded, collapse=", "))
d <- get("faulty_testing", envir=env)
rm(env)
invisible(gc())
required <- c("faultNumber", "simulationRun", "sample",
              paste0("xmeas_", 1:41), paste0("xmv_", 1:11))
if (!all(required %in% names(d))) stop("Required columns missing")
if (anyNA(d$faultNumber) || anyNA(d$simulationRun) || anyNA(d$sample)) stop("Missing identifiers")
cat("Loaded rows=", nrow(d), " cols=", ncol(d), "\n", sep="")
faults <- sort(unique(as.integer(d$faultNumber)))
cat("Fault identifiers: ", paste(faults, collapse=", "), "\n", sep="")
if (any(faults < 1L)) stop("Unexpected non-fault rows in faulty_testing; inspect source")
# Preserve all raw sensor values and identifiers without filtering by labels.
data.table::setDT(d)
for (id in faults) {
    out <- file.path(scratch, sprintf("fault_%02d.csv", id))
    ix <- which(d[["faultNumber"]] == id)
    cat(sprintf("Extracting fault %02d, %s rows ...\n", id, format(length(ix), big.mark=",")))
    tmp <- d[ix, ..required]
    data.table::fwrite(tmp, out)
    rm(tmp, ix)
    invisible(gc(verbose=FALSE))
    cat(sprintf("Saved %s (%0.1f MiB)\n", out, file.info(out)$size / 1048576))
}
cat("R extraction complete.\n")
