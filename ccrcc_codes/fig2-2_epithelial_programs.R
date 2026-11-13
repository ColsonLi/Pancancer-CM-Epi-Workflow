# Figure 2-2: epithelial subtype GO enrichment and selected-term dotplots

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(dplyr)
  library(ggplot2)
  library(scales)
  library(grid)
})

# Part 1: GO enrichment and per-subtype GO BP table export
setwd('/mnt/disk18t/data_disk18t/new_kirc_adata/cellsubtype_degs/')


read_and_filter_top1000 <- function(file){

  df <- read.csv(file, row.names = 1, check.names = FALSE)
  df_filtered <- df[df$logfoldchanges > 0.25 & df$pvals_adj < 0.05, ]
  syms <- rownames(df_filtered)[1:min(1000, nrow(df_filtered))]
  syms <- syms[!is.na(syms) & syms != ""]
  

  unique(syms)
}

syms_aldob <- read_and_filter_top1000('Epi_ALDOB_degs_epi.csv')
syms_aqp2  <- read_and_filter_top1000('Epi_AQP2_degs_epi.csv')
syms_ca12  <- read_and_filter_top1000('Epi_CA12_degs_epi.csv')
syms_ca9   <- read_and_filter_top1000('Epi_CA9_degs_epi.csv')
syms_gpx3  <- read_and_filter_top1000('Epi_GPX3_degs_epi.csv')
syms_jun   <- read_and_filter_top1000('Epi_JUN_degs_epi.csv')
syms_miox  <- read_and_filter_top1000('Epi_MIOX_degs_epi.csv')
syms_vim   <- read_and_filter_top1000('Epi_VIM_degs_epi.csv')


to_entrez <- function(symbols){
  bitr(symbols,
       fromType = "SYMBOL",
       toType   = "ENTREZID",
       OrgDb    = org.Hs.eg.db) |>
    dplyr::pull(ENTREZID) |>
    unique() |>
    as.character()
}

ent_aldob <- to_entrez(syms_aldob)
ent_aqp2  <- to_entrez(syms_aqp2)
ent_ca12  <- to_entrez(syms_ca12)
ent_ca9   <- to_entrez(syms_ca9)
ent_gpx3  <- to_entrez(syms_gpx3)
ent_jun   <- to_entrez(syms_jun)
ent_miox  <- to_entrez(syms_miox)
ent_vim   <- to_entrez(syms_vim)

run_go_all <- function(entrez_vec){
  enrichGO(gene          = entrez_vec,
           OrgDb         = org.Hs.eg.db,
           keyType       = "ENTREZID",
           ont           = "ALL",
           pAdjustMethod = "BH",
           pvalueCutoff  = 0.05,
           qvalueCutoff  = 0.05,
           readable      = TRUE)
}

ego_aldob <- run_go_all(ent_aldob)
ego_aqp2  <- run_go_all(ent_aqp2)
ego_ca12  <- run_go_all(ent_ca12)
ego_ca9   <- run_go_all(ent_ca9)
ego_gpx3  <- run_go_all(ent_gpx3)
ego_jun   <- run_go_all(ent_jun)
ego_miox  <- run_go_all(ent_miox)
ego_vim   <- run_go_all(ent_vim)


df_ALDOB <- as.data.frame(ego_aldob) |> mutate(Group = "Epi_ALDOB")
df_AQP2  <- as.data.frame(ego_aqp2)  |> mutate(Group = "Epi_AQP2")
df_CA12  <- as.data.frame(ego_ca12)  |> mutate(Group = "Epi_CA12")
df_CA9   <- as.data.frame(ego_ca9)   |> mutate(Group = "Epi_CA9")
df_GPX3  <- as.data.frame(ego_gpx3)  |> mutate(Group = "Epi_GPX3")
df_JUN   <- as.data.frame(ego_jun)   |> mutate(Group = "Epi_JUN")
df_MIOX  <- as.data.frame(ego_miox)  |> mutate(Group = "Epi_MIOX")
df_VIM   <- as.data.frame(ego_vim)   |> mutate(Group = "Epi_VIM")


go_df_all <- bind_rows(
  df_ALDOB, df_AQP2, df_CA12, df_CA9, 
  df_GPX3, df_JUN, df_MIOX, df_VIM
)


keep_cols <- c("ID","Description","ONTOLOGY","GeneRatio","Count",
               "pvalue","p.adjust","qvalue","geneID","Group")

for(g in unique(go_df_all$Group)){
  out <- go_df_all %>%

    dplyr::filter(Group == g, ONTOLOGY == "BP") %>% 
    dplyr::arrange(p.adjust) %>%
    dplyr::select(dplyr::all_of(intersect(keep_cols, names(go_df_all))))
  

  if(nrow(out) > 0){
    write.csv(out, paste0(as.character(g), "_GO_BP_group_20260405.csv"), row.names = FALSE)
  }
}

# Part 2: selected GO BP dotplots
setwd('/mnt/disk18t/data_disk18t/new_kirc_adata/gobp_filtered_top1000genes/')

df_MIOX <- read.csv('./Epi_MIOX_GO_BP_group_20260405.csv')
df_GPX3 <- read.csv('./Epi_GPX3_GO_BP_group_20260405.csv')
df_ALDOB <- read.csv('./Epi_ALDOB_GO_BP_group_20260405.csv')
df_CA9 <- read.csv('./Epi_CA9_GO_BP_group_20260405.csv')
df_JUN <- read.csv('./Epi_JUN_GO_BP_group_20260405.csv')
df_VIM <- read.csv('./Epi_VIM_GO_BP_group_20260405.csv')
df_AQP2 <- read.csv('./Epi_AQP2_GO_BP_group_20260405.csv')
df_CA12 <- read.csv('./Epi_CA12_GO_BP_group_20260405.csv')

go_df_all <- bind_rows(
  df_ALDOB, df_AQP2, df_CA12, df_CA9, 
  df_GPX3, df_JUN, df_MIOX, df_VIM
) |>

  mutate(Group = factor(Group, levels = c(
    "Epi_MIOX","Epi_GPX3","Epi_ALDOB", "Epi_CA9", 
     "Epi_JUN",  "Epi_VIM", "Epi_AQP2", "Epi_CA12"
  )))


groups_list <- list(
  Epi_ALDOB = c("GO:0034384","GO:0006520","GO:0006119","GO:0009636","GO:0006641"),
  Epi_AQP2  = c("GO:0050891","GO:0055078","GO:0072044","GO:0072073","GO:0031589"),
  Epi_CA12  = c("GO:1902600","GO:0006885","GO:0051452","GO:0072659","GO:0002181"),
  Epi_CA9   = c("GO:0001666","GO:0000280","GO:0007059","GO:0048002","GO:0007094"),
  Epi_GPX3  = c("GO:0045454","GO:0098754","GO:0044282","GO:0022904","GO:0042743"),
  Epi_JUN   = c("GO:0034976","GO:0097193","GO:0036293","GO:0043484","GO:0006986"),
  Epi_MIOX  = c("GO:1903825","GO:0003333","GO:0006006","GO:0170039","GO:1901293"),
  Epi_VIM   = c("GO:0090497","GO:0007015","GO:0030198","GO:0150117","GO:0001954") 
)


check_results <- lapply(names(groups_list), function(g) {
  selected_ids <- groups_list[[g]]

  existing_ids <- go_df_all$ID[go_df_all$Group == g]
  
  
  missing <- setdiff(selected_ids, existing_ids)
  
  if(length(missing) > 0) {
    message(paste0("⚠️ Group [", g, "] is missing IDs: ", paste(missing, collapse = ", ")))
  } else {
    message(paste0("✅ Group [", g, "] passed validation for all IDs!"))
  }
  return(missing)
})


go_ids <- c(
  "GO:1903825","GO:0003333","GO:0006006","GO:0170039","GO:1901293", ### Epi_MIOX includes subtype-specific terms
  "GO:0045454","GO:0098754","GO:0044282","GO:0022904","GO:0042743", ###Epi_GPX3
  "GO:0006520","GO:0006119","GO:0009636","GO:0006641","GO:0034384", ###Epi_ALDOB
  "GO:0001666","GO:0000280","GO:0007059","GO:0048002","GO:0007094", ###Epi_CA9
  "GO:0034976","GO:0097193","GO:0036293","GO:0043484","GO:0006986", ###Epi_JUN
  "GO:0090497","GO:0007015","GO:0030198","GO:0150117","GO:0001954",  ###Epi_VIM
  "GO:0050891","GO:0055078","GO:0072044","GO:0072073","GO:0031589", ###Epi_AQP2
  "GO:1902600","GO:0006885","GO:0051452","GO:0072659","GO:0002181" ###Epi_CA12
)

df_ALDOB_filtered <- df_ALDOB %>% filter(ID %in% go_ids)
df_AQP2_filtered  <- df_AQP2  %>% filter(ID %in% go_ids)
df_CA12_filtered  <- df_CA12  %>% filter(ID %in% go_ids)
df_CA9_filtered   <- df_CA9   %>% filter(ID %in% go_ids)
df_GPX3_filtered  <- df_GPX3  %>% filter(ID %in% go_ids)
df_JUN_filtered   <- df_JUN   %>% filter(ID %in% go_ids)
df_MIOX_filtered  <- df_MIOX  %>% filter(ID %in% go_ids)
df_VIM_filtered   <- df_VIM   %>% filter(ID %in% go_ids)

go_df_all <- bind_rows(
  df_ALDOB_filtered,
  df_AQP2_filtered,
  df_CA12_filtered,
  df_CA9_filtered,
  df_GPX3_filtered,
  df_JUN_filtered,
  df_MIOX_filtered,
  df_VIM_filtered
)

plot_df <- go_df_all %>%
  mutate(Group = factor(Group, levels = c(
    "Epi_MIOX", 
    "Epi_GPX3",
    "Epi_ALDOB", 
    "Epi_CA9", 
    "Epi_JUN", 
    "Epi_VIM",
    "Epi_AQP2", 
    "Epi_CA12"
  )))
plot_df2 <- plot_df %>%
  mutate(
    log10p = -log10(p.adjust),
    log10p_capped = squish(log10p, range = c(5, 20))
  )
p <- ggplot(plot_df2,
            aes(x = Group,
                y = Description,
                size = Count,
                colour = log10p_capped)) +
  geom_point() +
  scale_colour_gradient(
    low = "blue", high = "red",
    #limits = c(5, 20),
    oob = squish
  ) +
  scale_size(range = c(1.2, 5.5)) +  
  guides(
    colour = guide_colorbar(order = 1, barheight = unit(6, "cm")),
    size   = guide_legend(order = 2, override.aes = list(colour = "black"))
  ) +
  labs(
    title  = "GO Biological Process (Selected terms)",
    x = "Group", y = "GO Term",
    colour = "-log10(p.adjust)"
  ) +
  theme_bw(base_family = "Arial") +
  theme(
    plot.title  = element_text(size = 20, face = "bold", hjust = 0.5),
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 12, face = "bold", color = "black"),
    axis.text.y = element_text(size = 12, face = "bold", color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey85"),
    legend.title = element_text(size = 12, face = "bold", color = "black"),
    legend.text  = element_text(size = 10, color = "black"),
    legend.position = "right",
    legend.key.size = unit(1.0, "lines")
  )

print(p)


# Part 3: paired tumor Epi_JUN-versus-Epi_VIM edgeR source data for Figure 2
#
# Run this section after the corresponding Python source-data cell in
# fig2-1_epithelial_states_and_markers.ipynb has written the pseudobulk count matrix and library metadata.
# The paired design blocks patient-specific expression differences with
# ~ sample + state; Epi_VIM is the reference, so positive logFC denotes
# higher expression in Epi_JUN.

run_fig2_jun_vim_edger <- function() {
  suppressPackageStartupMessages(library(edgeR))

  work_dir <- file.path(
    "/mnt/disk18t/lr_xcy/riku/kirc_205",
    "review_revision_20260828_li/results",
    "tumor_epithelial_marker_publication_figure_repositioned",
    "source_data_work"
  )
  counts_path <- file.path(work_dir, "panel_b_pseudobulk_counts.csv.gz")
  metadata_path <- file.path(work_dir, "panel_b_pseudobulk_metadata.csv")
  results_path <- file.path(work_dir, "panel_b_edger_results.csv.gz")
  norm_path <- file.path(work_dir, "panel_b_edger_norm_factors.csv")

  counts_df <- read.csv(gzfile(counts_path), row.names = 1, check.names = FALSE)
  counts <- as.matrix(counts_df)
  storage.mode(counts) <- "numeric"
  metadata <- read.csv(metadata_path, check.names = FALSE, stringsAsFactors = FALSE)

  if (!identical(colnames(counts), metadata$library_id)) {
    stop("Pseudobulk count columns do not match metadata library_id order")
  }
  if (any(counts < 0) || any(abs(counts - round(counts)) > 1e-6)) {
    stop("Input pseudobulk matrix is not non-negative integer counts")
  }

  metadata$sample <- factor(metadata$sample)
  metadata$state <- relevel(factor(metadata$state), ref = "Epi_VIM")
  design <- model.matrix(~ sample + state, data = metadata)
  if (qr(design)$rank != ncol(design)) {
    stop("Paired sample + state design matrix is not full rank")
  }

  y <- DGEList(counts = round(counts))
  keep <- filterByExpr(y, design = design, min.count = 10)
  y <- y[keep, , keep.lib.sizes = FALSE]
  y <- calcNormFactors(y, method = "TMM")
  y <- estimateDisp(y, design = design, robust = TRUE)
  fit <- glmQLFit(y, design = design, robust = TRUE)
  coef_name <- "stateEpi_JUN"
  test <- glmQLFTest(fit, coef = which(colnames(design) == coef_name))
  tab <- topTags(test, n = Inf, sort.by = "none")$table
  tab$gene <- rownames(tab)
  tab <- tab[, c("gene", "logFC", "logCPM", "F", "PValue", "FDR")]
  tab$contrast <- "Epi_JUN_minus_Epi_VIM"
  tab$n_paired_tumors <- length(unique(metadata$sample))
  write.csv(tab, gzfile(results_path), row.names = FALSE, quote = FALSE)

  norm <- data.frame(
    library_id = colnames(y),
    library_size = y$samples$lib.size,
    norm_factor = y$samples$norm.factors,
    effective_library_size = y$samples$lib.size * y$samples$norm.factors,
    stringsAsFactors = FALSE
  )
  write.csv(norm, norm_path, row.names = FALSE, quote = FALSE)
  invisible(list(results = results_path, normalization = norm_path))
}

# Execute explicitly when regenerating Figure 2 source data:
# run_fig2_jun_vim_edger()


group_levels <- c(
  "Epi_MIOX",
  "Epi_GPX3",
  "Epi_ALDOB",
  "Epi_CA9",
  "Epi_JUN",
  "Epi_VIM",
  "Epi_AQP2",
  "Epi_CA12"
)


term_map <- bind_rows(lapply(names(groups_list), function(g) {
  data.frame(
    Group_owner = g,
    ID = groups_list[[g]],
    term_order = seq_along(groups_list[[g]]),
    stringsAsFactors = FALSE
  )
})) %>%
  mutate(Group_owner = factor(Group_owner, levels = group_levels))


plot_df <- go_df_all %>%
  mutate(Group = factor(Group, levels = group_levels)) %>%
  left_join(term_map, by = "ID") %>%
  mutate(
    term_key = paste(ID, Description, sep = " | ")
  )


term_levels <- term_map %>%
  left_join(
    plot_df %>% distinct(ID, Description, term_key),
    by = "ID"
  ) %>%
  arrange(Group_owner, term_order) %>%
  pull(term_key)

plot_df2 <- plot_df %>%
  mutate(
    term_key = factor(term_key, levels = rev(unique(term_levels))),
    log10p = -log10(p.adjust),
    log10p_capped = squish(log10p, range = c(2, 20))
  )

p <- ggplot(
  plot_df2,
  aes(
    x = Group,
    y = term_key,
    size = Count,
    colour = log10p_capped
  )
) +
  geom_point() +
  scale_y_discrete(
    labels = function(x) sub("^GO:[^|]+ \\| ", "", x)
  ) +
  scale_colour_gradient(
    low = "blue", high = "red",
    oob = squish
  ) +
  scale_size(range = c(1.2, 5.5)) +
  guides(
    colour = guide_colorbar(order = 1, barheight = unit(6, "cm")),
    size   = guide_legend(order = 2, override.aes = list(colour = "black"))
  ) +
  labs(
    title  = "GO Biological Process (Selected terms)",
    x = "Group",
    y = "GO Term",
    colour = "-log10(p.adjust)"
  ) +
  theme_bw(base_family = "Arial") +
  theme(
    plot.title  = element_text(size = 20, face = "bold", hjust = 0.5),
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 12, face = "bold", color = "black"),
    axis.text.y = element_text(size = 10, face = "bold", color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey85"),
    legend.title = element_text(size = 12, face = "bold", color = "black"),
    legend.text  = element_text(size = 10, color = "black"),
    legend.position = "right",
    legend.key.size = unit(1.0, "lines")
  )

print(p)

# Figure 2 panel H: save the final selected-term dotplot defined above.
# Keep both original setwd() paths and every original GO selection/plot parameter.
# Canvas dimensions match the uploaded standalone PDF (527 x 541 PDF points).
ggsave(
  filename = "redraw_gobp_8epi_1.pdf",
  plot = p, width = 527 / 72, height = 541 / 72, units = "in",
  device = cairo_pdf
)

# Additional Figure 2 Monocle3 analyses and panel outputs.
# Functions below are explicit entry points, not automatic reruns.
# Original scripts remain intact; their code is included here rather than sourced.

# Panels F/G: original Monocle3 calculation from the exported input bundle.
# Sources: leiden_detailed/monocle3/{run_epi_monocle3_root_miox.R,
# generate_epi_monocle3_pruned_ncenter300.R,
# redraw_monocle3_trajectory_pseudotime_ncenter300.R}.
# Keep the original two-stage fit: the graph used in the reported panels is
# the final pruned ncenter=300 graph, with close_loop=FALSE.
run_fig2_monocle3_from_inputs <- function(base_dir = "/mnt/disk18t/lr_xcy/riku/kirc_205/leiden_detailed/monocle3") {
  suppressPackageStartupMessages({
    library(monocle3)
    library(Matrix)
    library(ggplot2)
  })

  base_dir <- normalizePath(base_dir)
  input_dir <- file.path(base_dir, "inputs")
  out_dir <- file.path(base_dir, "outputs")
  fig_dir <- file.path(base_dir, "figures")
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

  matrix_path <- file.path(input_dir, "epi_6subtype_raw_allgenes_genes_by_cells.mtx")
  cell_meta_path <- file.path(input_dir, "cell_metadata.tsv")
  gene_meta_path <- file.path(input_dir, "gene_metadata.tsv")
  umap_path <- file.path(input_dir, "umap_coordinates.tsv")
  root_path <- file.path(input_dir, "root_cell.tsv")

  expr <- readMM(matrix_path)
  cell_meta <- read.delim(cell_meta_path, check.names = FALSE, stringsAsFactors = FALSE)
  gene_meta <- read.delim(gene_meta_path, check.names = FALSE, stringsAsFactors = FALSE)
  umap <- read.delim(umap_path, check.names = FALSE, stringsAsFactors = FALSE)
  root_info <- read.delim(root_path, check.names = FALSE, stringsAsFactors = FALSE)

  rownames(gene_meta) <- gene_meta$gene_id
  rownames(cell_meta) <- cell_meta$cell
  rownames(umap) <- umap$cell

  if (!all(colnames(expr) == NULL)) {
    colnames(expr) <- NULL
  }
  rownames(expr) <- gene_meta$gene_id
  colnames(expr) <- cell_meta$cell

  if (!identical(rownames(cell_meta), rownames(umap))) {
    stop("cell metadata and UMAP coordinates are not aligned")
  }
  if (!root_info$root_cell[1] %in% rownames(cell_meta)) {
    stop("root cell is not present in cell metadata")
  }

  cell_meta$cell_subtype <- factor(
    cell_meta$cell_subtype,
    levels = c("Epi_MIOX", "Epi_ALDOB", "Epi_GPX3", "Epi_CA9", "Epi_JUN", "Epi_VIM")
  )

  cds <- new_cell_data_set(
    expr,
    cell_metadata = cell_meta,
    gene_metadata = gene_meta
  )

  cds <- preprocess_cds(cds, num_dim = 50, norm_method = "none")
  reducedDims(cds)$UMAP <- as.matrix(umap[, c("UMAP_1", "UMAP_2")])

  cds <- cluster_cells(cds, reduction_method = "UMAP")
  cds <- learn_graph(cds, use_partition = FALSE)
  cds <- order_cells(cds, root_cells = root_info$root_cell[1])

  pseudotime_df <- data.frame(
    cell = colnames(cds),
    monocle3_pseudotime = as.numeric(pseudotime(cds)),
    cell_subtype = colData(cds)$cell_subtype,
    status = colData(cds)$status,
    sample = colData(cds)$sample,
    series = colData(cds)$series,
    stemness_score = colData(cds)$stemness_score,
      stringsAsFactors = FALSE
  )
  write.csv(pseudotime_df, file.path(out_dir, "epi_6subtype_monocle3_pseudotime.csv"), row.names = FALSE)

  summary_df <- aggregate(
    monocle3_pseudotime ~ cell_subtype,
    data = pseudotime_df,
    FUN = function(x) c(n = sum(is.finite(x)), mean = mean(x, na.rm = TRUE), median = median(x, na.rm = TRUE))
  )
  summary_out <- data.frame(
    cell_subtype = summary_df$cell_subtype,
    n = summary_df$monocle3_pseudotime[, "n"],
    mean = summary_df$monocle3_pseudotime[, "mean"],
    median = summary_df$monocle3_pseudotime[, "median"]
  )
  write.csv(summary_out, file.path(out_dir, "epi_6subtype_monocle3_pseudotime_by_subtype.csv"), row.names = FALSE)

  saveRDS(cds, file.path(out_dir, "epi_6subtype_monocle3_root_miox.cds.rds"))
  writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))

  suppressPackageStartupMessages({
    library(monocle3)
  })

  base_dir <- normalizePath(base_dir)
  input_dir <- file.path(base_dir, "inputs")
  out_dir <- file.path(base_dir, "outputs")
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  base_cds_path <- file.path(
    out_dir,
    "epi_6subtype_monocle3_root_miox.cds.rds"
  )
  root_path <- file.path(input_dir, "root_cell.tsv")

  cds <- readRDS(base_cds_path)
  root_info <- read.delim(
    root_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  root_cell <- root_info$root_cell[1]

  if (!root_cell %in% colnames(cds)) {
    stop("Root cell is not present in the Monocle3 CDS")
  }

  set.seed(42)
  cds <- learn_graph(
    cds,
    use_partition = FALSE,
    close_loop = FALSE,
    learn_graph_control = list(ncenter = 300)
  )
  cds <- order_cells(cds, root_cells = root_cell)

  saveRDS(
    cds,
    file.path(
      out_dir,
      "epi_6subtype_monocle3_root_miox_pruned_ncenter300.cds.rds"
    )
  )

  cat("root_cell", root_cell, "\n")
  cat("cells", ncol(cds), "genes", nrow(cds), "\n")
  cat(
    "principal_graph_nodes",
    igraph::vcount(principal_graph(cds)[["UMAP"]]),
    "principal_graph_edges",
    igraph::ecount(principal_graph(cds)[["UMAP"]]),
    "\n"
  )

  pseudotime_df <- as.data.frame(colData(cds))
  pseudotime_df$cell <- rownames(pseudotime_df)
  pseudotime_df$monocle3_pseudotime <- pseudotime(cds)
  pseudotime_df <- pseudotime_df[, c("cell", "cell_subtype", "sample", "status", "stemness_score", "monocle3_pseudotime")]
  write.csv(
    pseudotime_df,
    file.path(out_dir, "epi_6subtype_monocle3_pseudotime_pruned_ncenter300.csv"),
    row.names = FALSE
  )
  invisible(file.path(out_dir, "epi_6subtype_monocle3_root_miox_pruned_ncenter300.cds.rds"))
}

# Panel G: exact standalone trajectory and pseudotime plotting code.
# Reads the saved final ncenter=300 RDS; does not refit the trajectory.
run_fig2_monocle3_trajectory_panels <- function(base_dir = "/mnt/disk18t/lr_xcy/riku/kirc_205/leiden_detailed/monocle3") {
  suppressPackageStartupMessages({
    library(monocle3)
    library(ggplot2)
    library(ggrastr)
  })

  base_dir <- normalizePath(base_dir)
  out_dir <- file.path(base_dir, "outputs")
  fig_dir <- file.path(base_dir, "figures")

  cds <- readRDS(file.path(out_dir, "epi_6subtype_monocle3_root_miox_pruned_ncenter300.cds.rds"))

  color_path <- file.path(base_dir, "inputs", "adata_epi_cell_subtype_colors.tsv")
  cell_subtype_colors <- NULL
  if (file.exists(color_path)) {
    color_df <- read.delim(color_path, check.names = FALSE, stringsAsFactors = FALSE)
    cell_subtype_colors <- setNames(color_df$color, color_df$cell_subtype)
  }

  plot_and_save <- function(p, filename, width, height) {
    ggsave(file.path(fig_dir, paste0(filename, ".pdf")), p, width = width, height = height, device = cairo_pdf)
    ggsave(file.path(fig_dir, paste0(filename, ".svg")), p, width = width, height = height)
  }

  common_args <- list(
    label_cell_groups = FALSE,
    label_leaves = FALSE,
    label_branch_points = FALSE,
    label_roots = FALSE,
    show_trajectory_graph = TRUE,
    trajectory_graph_color = "black",
    trajectory_graph_segment_size = 0.6,
    cell_size = 0.3,
    cell_stroke = 0,
    alpha = 1
  )

  square_panel_theme <- theme(
    aspect.ratio = 1,
    plot.title = element_text(size = 10),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 8),
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 8)
  )

  p_subtype <- do.call(plot_cells, c(list(cds = cds, color_cells_by = "cell_subtype"), common_args)) +
    ggtitle("Monocle3 trajectory: ncenter=300") +
    square_panel_theme
  if (!is.null(cell_subtype_colors)) {
    p_subtype <- p_subtype + scale_color_manual(values = cell_subtype_colors)
  }
  p_subtype <- rasterise(p_subtype, layers = "Point", dpi = 600, dev = "ragg")
  plot_and_save(
    p_subtype,
    "epi_6subtype_monocle3_trajectory_cell_subtype_pruned_ncenter300_rasterdots_square_panel_larger_size0p3_alpha1",
    width = 5.2,
    height = 4.0
  )

  p_pt <- do.call(plot_cells, c(list(cds = cds, color_cells_by = "pseudotime"), common_args)) +
    ggtitle("Monocle3 pseudotime: ncenter=300") +
    square_panel_theme
  p_pt <- rasterise(p_pt, layers = "Point", dpi = 600, dev = "ragg")
  plot_and_save(
    p_pt,
    "epi_6subtype_monocle3_trajectory_pseudotime_pruned_ncenter300_rasterdots_square_panel_larger_size0p3_alpha1",
    width = 4.6,
    height = 4.0
  )
}

# Panel F: export the original saved pseudotimes, then draw the uploaded
# thin boxplot with exactly 35,987 tumour CA9/VIM/JUN cells.
# Sources: boxplot_existing_ncenter300_20260904/
# draw_existing_monocle3_pseudotime_boxplots.R and
# draw_monocle3_ncenter300_figure2b_matched_boxplot_thin.R.
# Neither the source-data export nor this plot refits Monocle3.
run_fig2_monocle3_boxplot <- function() {
  suppressPackageStartupMessages({
    library(monocle3)
    library(ggplot2)
  })

  root_dir <- "/mnt/disk18t/lr_xcy/riku/kirc_205"
  rds_path <- file.path(
    root_dir,
    "leiden_detailed/monocle3/outputs",
    "epi_6subtype_monocle3_root_miox_pruned_ncenter300.cds.rds"
  )
  out_dir <- file.path(
    root_dir,
    "leiden_detailed/monocle3/boxplot_existing_ncenter300_20260904"
  )
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  # Read the existing ordered object. No graph learning or pseudotime refitting occurs.
  cds <- readRDS(rds_path)
  plot_df <- as.data.frame(colData(cds))
  plot_df$cell_id <- rownames(plot_df)
  plot_df$monocle3_pseudotime <- as.numeric(pseudotime(cds))

  state_order <- c(
    "Epi_MIOX",
    "Epi_GPX3",
    "Epi_ALDOB",
    "Epi_CA9",
    "Epi_VIM",
    "Epi_JUN"
  )
  state_colors <- c(
    "Epi_MIOX" = "#e377c2",
    "Epi_GPX3" = "#b5bd61",
    "Epi_ALDOB" = "#aa40fc",
    "Epi_CA9" = "#d62728",
    "Epi_VIM" = "#17becf",
    "Epi_JUN" = "#279e68"
  )
  plot_df$cell_subtype <- factor(
    as.character(plot_df$cell_subtype),
    levels = state_order
  )

  source_columns <- c(
    "cell_id",
    "sample",
    "status",
    "cell_subtype",
    "monocle3_pseudotime"
  )
  write.csv(
    plot_df[, source_columns],
    file.path(out_dir, "existing_ncenter300_monocle3_pseudotime_source_data.csv"),
    row.names = FALSE
  )

  summarize_frame <- function(frame, population) {
    pieces <- lapply(state_order, function(state) {
      values <- frame$monocle3_pseudotime[frame$cell_subtype == state]
      data.frame(
        population = population,
        cell_subtype = state,
        n_cells = length(values),
        median = median(values),
        mean = mean(values),
        q25 = unname(quantile(values, 0.25)),
        q75 = unname(quantile(values, 0.75))
      )
    })
    do.call(rbind, pieces)
  }

  tumor_df <- plot_df[tolower(as.character(plot_df$status)) == "tumor", ]
  bridge_states <- c("Epi_CA9", "Epi_VIM", "Epi_JUN")
  bridge_df <- tumor_df[as.character(tumor_df$cell_subtype) %in% bridge_states, ]
  bridge_df$cell_subtype <- factor(
    as.character(bridge_df$cell_subtype),
    levels = bridge_states
  )
  summary_df <- rbind(
    summarize_frame(plot_df, "all"),
    summarize_frame(tumor_df, "tumor")
  )
  write.csv(
    summary_df,
    file.path(out_dir, "existing_ncenter300_monocle3_pseudotime_summary.csv"),
    row.names = FALSE
  )

  y_limits <- range(plot_df$monocle3_pseudotime, finite = TRUE)
  y_upper <- y_limits[2] * 1.02

  suppressPackageStartupMessages(library(ggplot2))

  root_dir <- "/mnt/disk18t/lr_xcy/riku/kirc_205"
  result_dir <- file.path(
    root_dir,
    "leiden_detailed/monocle3/boxplot_existing_ncenter300_20260904"
  )
  source_path <- file.path(
    result_dir,
    "existing_ncenter300_monocle3_pseudotime_source_data.csv"
  )
  output_path <- file.path(
    result_dir,
    "monocle3_ncenter300_pseudotime_boxplot_figure2b_matched_cells.pdf"
  )

  # Use the already exported pseudotime values; no trajectory refitting occurs.
  source_data <- read.csv(source_path, check.names = FALSE, stringsAsFactors = FALSE)
  state_order <- c("Epi_CA9", "Epi_VIM", "Epi_JUN")
  state_colors <- c(
    "Epi_CA9" = "#d62728",
    "Epi_VIM" = "#17becf",
    "Epi_JUN" = "#279e68"
  )

  plot_data <- source_data[
    tolower(source_data$status) == "tumor" &
      source_data$cell_subtype %in% state_order,
  ]
  plot_data$cell_subtype <- factor(plot_data$cell_subtype, levels = state_order)
  # Set adjacent state-centre spacing to two fifths of the initial one-unit gap.
  state_positions <- c(
    "Epi_CA9" = 1.6,
    "Epi_VIM" = 2,
    "Epi_JUN" = 2.4
  )
  plot_data$x_position <- unname(state_positions[as.character(plot_data$cell_subtype)])

  if (nrow(plot_data) != 35987L) {
    stop("Expected exactly 35,987 Figure 2B-matched tumour epithelial cells")
  }

  y_upper <- max(source_data$monocle3_pseudotime, na.rm = TRUE) * 1.02
  original_box_width <- 0.62
  thin_box_width <- original_box_width / 3

  plot_object <- ggplot(
    plot_data,
    aes(
      x = x_position,
      y = monocle3_pseudotime,
      fill = cell_subtype,
      group = cell_subtype
    )
  ) +
    geom_boxplot(
      width = thin_box_width,
      outlier.shape = NA,
      linewidth = 0.45,
      color = "#303840",
      alpha = 0.88
    ) +
    stat_summary(
      fun = median,
      geom = "point",
      shape = 21,
      size = 1.8,
      stroke = 0.45,
      fill = "white",
      color = "#20262C"
    ) +
    scale_fill_manual(values = state_colors, drop = FALSE) +
    scale_x_continuous(
      breaks = unname(state_positions),
      labels = state_order,
      limits = c(1.25, 2.75),
      expand = expansion(mult = 0)
    ) +
    coord_cartesian(ylim = c(0, y_upper)) +
    labs(
      title = "Monocle3 pseudotime in the CA9–VIM–JUN continuum",
      subtitle = paste0(
        "Exactly matched to Figure 2B; ",
        format(nrow(plot_data), big.mark = ",", scientific = FALSE),
        " tumour epithelial cells"
      ),
      x = NULL,
      y = "Monocle3 pseudotime"
    ) +
    theme_classic(base_size = 10) +
    theme(
      plot.title = element_text(size = 7.0, face = "bold", color = "#263238"),
      plot.subtitle = element_text(size = 6.3, color = "#66717B", margin = margin(b = 7)),
      axis.text.x = element_text(angle = 32, hjust = 1, vjust = 1, color = "#263238"),
      axis.text.y = element_text(color = "#263238"),
      axis.title.y = element_text(color = "#263238"),
      axis.line = element_line(color = "#303840", linewidth = 0.5),
      axis.ticks = element_line(color = "#303840", linewidth = 0.45),
      legend.position = "none",
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
    )

  # Adjacent state centres are two fifths as far apart as in the initial plot;
  # box width remains exactly one third of the initial 0.62. The narrower canvas
  # and tight x limits remove the excess side whitespace.
  ggsave(
    output_path,
    plot_object,
    width = 3.0,
    height = 4.1,
    device = cairo_pdf,
    bg = "white"
  )

  message(output_path)
  invisible(output_path)
}

# Execution order for panels F/G (from existing input files):
# run_fig2_monocle3_from_inputs()       # optional rebuild of the saved final RDS
# run_fig2_monocle3_trajectory_panels() # the two panel-G PDFs
# run_fig2_monocle3_boxplot()           # panel-F PDF and pseudotime source data
# The paired marker comparison uses run_fig2_jun_vim_edger() above.
# Input preparation is in fig2-1_epithelial_states_and_markers.ipynb.
# The uploaded fig2_6.pdf is an Illustrator assembly of the standalone panels.
