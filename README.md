# Distinct Epithelial States Associate with Contrasting Immune Niches in Clear Cell Renal Cell Carcinoma

Research overview and accompanying analysis code.

## Study overview

This study integrates single-cell, spatial, and clinical data to examine how epithelial states relate to immune and stromal environments in clear cell renal cell carcinoma (ccRCC). Recurrent combinations of non-epithelial populations are identified as **cross-cellular modules (CMs)**. The central comparison concerns a JUN-associated, stress-responsive epithelial state (**Epi_JUN**) and a VIM-associated, matrix-remodeling state (**Epi_VIM**), which share epithelial and mesenchymal features but show opposite associations with the lymphocyte-rich module **CM1**.

The analyses evaluate these relationships within tumor tissue, prioritize **SPP1** as a candidate ligand, and examine epithelial and immune expression programs in relation to prognosis and treatment outcomes. Independent colorectal cancer (CRC) and pancreatic ductal adenocarcinoma (PDAC) analyses assess JUN-like epithelial states and T/NK-rich modules. All analyses use existing human datasets; candidate signaling mechanisms require experimental validation.

## Data and study design

| Component | Data and analysis population | Purpose |
| --- | --- | --- |
| ccRCC single-cell atlas | Eight studies; 84 input samples; final annotated atlas of 401,648 cells from 83 samples: 61 tumor and 22 normal-like. | Resolve major lineages, epithelial states, and non-epithelial subtypes. |
| ccRCC module discovery | 82 eligible samples: 61 tumor and 21 normal-like; at least 50 non-epithelial cells per sample. | Identify recurrent microenvironmental compositions using balanced joint NMF. |
| Epithelial–CM associations | Nine epithelial states and 12 modules across 61 tumor samples. | Test all 108 state–module pairs with BH correction across that family. |
| Paired epithelial expression | 41 tumor samples with at least 20 cells in each of Epi_JUN and Epi_VIM. | Compare raw-count pseudobulks and assess paired ligand expression. |
| Spatial transcriptomics | 14 sections assembled; only the 12 tumor sections enter reported spatial association tests. | Map epithelial states and modules and evaluate local associations. |
| TCGA-KIRC | Overall survival; the primary clinically adjusted model includes 522 patients and 172 deaths. | Assess the prognostic association of relative JUN/VIM expression. |
| JAVELIN Renal 101 | Progression-free survival; 726 molecularly characterized patients and 358 events. | Examine treatment–module–epithelial-balance interactions for avelumab plus axitinib versus sunitinib. |
| CRC extension | Visualization atlas: 643,655 cells from 316 samples; focused association: 127 eligible tumor samples. | Match a JUN-like epithelial state and test its association with an independently fitted T/NK-rich module. |
| PDAC extension | Visualization atlas: 392,388 cells from 125 samples; focused association: 74 eligible tumor samples. | Perform the corresponding reference-matching and module-association analysis. |

“Normal-like” denotes the harmonized normal/non-tumor kidney comparison group, not a uniformly verified healthy-donor cohort. CRC/PDAC atlas totals describe visualization objects; epithelial reclustering, module fitting, and association testing use their respective cleaned and eligible populations. Sample and section identifiers are not uniformly linked to confirmed independent donors.

Dataset accessions, source publications, stage-specific inclusion, and sample annotations are described in the accompanying manuscript, SI Appendix, and Supplementary Table S1. Dataset S1 provides key figure statistics. Human Protein Atlas images supply qualitative protein-expression context for selected markers, not matched-patient validation.

## Analytical framework and figures

The analysis proceeds from an integrated atlas to epithelial-state characterization, module discovery, spatial evaluation, candidate signaling, clinical associations, and cross-cancer extension. The table below connects each part of the study to its principal code entry points.

| Figure | Research question and analysis | Main code |
| --- | --- | --- |
| **Fig. 1 — Integrated atlas** | Which major cell populations are represented across tumor and normal-like samples? Merge and quality-control datasets, integrate expression, and assemble reviewed subtype annotations. | [Atlas construction](fig1-1_atlas_construction.ipynb); [annotation and overview](fig1-2_atlas_annotation_and_overview.ipynb). |
| **Fig. 2 — Epithelial states** | How are epithelial states related, and what distinguishes Epi_JUN from Epi_VIM? Analyze sample-adjusted markers, paired pseudobulks, enrichment, transcriptional coordinates, and inferred trajectories. | [Epithelial states and markers](fig2-1_epithelial_states_and_markers.ipynb); [GO, edgeR, and Monocle3](fig2-2_epithelial_programs.R). |
| **Fig. 3 — Cross-cellular modules** | Which non-epithelial populations vary together, and how do their modules associate with epithelial states? Fit balanced joint NMF and test epithelial–CM associations. | [NMF and epithelial–module associations](fig3-1_balanced_joint_nmf_and_epithelial_module_associations.ipynb). |
| **Fig. 4 — Spatial relationships** | Do the sample-level associations recur locally within tumor tissue? Map expression references section by section, calculate module activity and JUN/VIM balance, and perform adjusted and translation-control analyses. | [Tangram mapping and spatial associations](fig4_tangram_mapping_and_spatial_associations.ipynb). |
| **Fig. 5 — Candidate signaling** | Which regulatory programs and candidate ligand–receptor routes accompany the JUN-associated niche? Combine regulon inference, paired ligand expression, CellChat predictions, and spatial hotspots. | [Regulon activity](fig5-1_pyscenic_regulon_activity.ipynb); [CellChat, SPP1, and spatial hotspots](fig5-2_cellchat_spp1_and_spatial_hotspots.ipynb). |
| **Fig. 6 — Clinical associations** | How do epithelial and immune programs relate to prognosis and relative treatment outcomes? Score bulk signatures and fit exploratory and clinically adjusted survival and interaction models. | [Clinical survival and treatment associations](fig6_s8-1_clinical_survival_and_treatment_associations.ipynb). |
| **Fig. 7 — CRC/PDAC extension** | Do independently defined JUN-like epithelial states also accompany T/NK-rich modules? Recluster cleaned tumor-source epithelial cells, match to the ccRCC reference, fit cancer-specific CMs, and test sample-level associations. | [Upstream cross-cancer analyses](fig7-1_cross_cancer_upstream_analysis.ipynb); [JUN-only figure generation](fig7-2_cross_cancer_jun_reference_and_tnk_associations.ipynb). |

Supplementary code covers integration and composition (S1), expression-based CNV and velocity/CellRank analyses (S2), shared stemness scoring and epithelial survival displays (S3), subtype characterization and complementary bulk deconvolution (S4), within-module subtype correlations (S5–S6), spatial replication (S7), and additional clinical analyses (S8).

CM identities are learned from single-cell subtype composition. Their numerical representations differ by analysis: sample-level NMF activity, loading-weighted spatial scores, and bulk signature enrichment are not a common absolute measurement scale. Likewise, spatial JUN/VIM fraction balance and clinical JUN/VIM expression balance describe different aspects of epithelial context.

## Code and repository structure

This directory collects **21 main entry files: 20 Jupyter notebooks and one R script**, together with the Fig. 7 upstream source archive. Figure-specific originals remain under the sibling directories `main_codes/fig1/`, `fig2/`, `fig3/`, `fig4/`, `fig5/`, `fig6_et_s8/`, `fig7/`, and the corresponding supplementary-figure directories. The collected copies do not synchronize automatically with those originals.

- [fig7_upstream_sources/README.md](fig7_upstream_sources/README.md): CRC/PDAC source provenance, input-production boundaries, and stage-specific execution instructions.
- [fig7_upstream_sources/SOURCE_MANIFEST.json](fig7_upstream_sources/SOURCE_MANIFEST.json): original paths, roles, and SHA256 checksums for 42 archived source and environment files.

The Fig. 7 code follows the current **JUN-only CRC/PDAC analysis**, corresponding to `Fig7_JUN_only_symmetric.pdf`. CRC uses the selected c08 state and CRCcal_CM08; PDAC uses c7 and calCM02. The extension is a single-cell reference-matching and sample-association analysis, not the former agent-workflow comparison or an extension of spatial and clinical testing to those cancers.

## Using the code

The collection includes both analysis reconstruction and plotting from saved outputs. Begin with the figure of interest and check the inputs and upstream dependencies specified in its notebook or script. Figure order is not executable notebook order: for example, final atlas assembly requires the lineage-specific annotation outputs, and several displays consume previously saved results.

Python analyses use combinations of Scanpy/rapids-singlecell, Harmony, infercnvpy, scVelo/veloVI, CellRank, Tangram, pySCENIC, AutoGeneS, GSEApy, and survival/statistical libraries. R analyses use edgeR, clusterProfiler, Monocle3, and CellChat. Different branches use different environments; recorded Fig. 7 environments are available for [CRC](fig7_upstream_sources/crc/calibrated_single_cluster_20261107/software_versions.json) and [PDAC](fig7_upstream_sources/pdac/calibrated_single_cluster_20261107/tables/software_versions.json). No single tested environment lockfile is supplied for the entire study.

Input data, fitted objects, reference databases, and figure PDFs are **not bundled in this code directory**. Paths retain their original local settings. Check those paths, the working directory, required resources, and output locations before execution. The Fig. 7 upstream entry starts from existing local expression/AnnData objects and defaults to a read-only/dry-run configuration; it is not a FASTQ-to-figure pipeline.

Some reconstruction stages overwrite existing outputs. In particular, the configured NMF rebuild removes its existing output directory before recreating it, and the velocity notebook defaults to rebuilding objects from loom inputs. Check the relevant execution switches and preserve any required results before rerunning. This documentation update does not represent a new end-to-end execution of the scientific analyses.

## Citation

When using these analyses, cite the accompanying manuscript, **“Distinct Epithelial States Associate with Contrasting Immune Niches in Clear Cell Renal Cell Carcinoma,”** together with the relevant original dataset and method publications listed in the manuscript and SI Appendix.

Documentation updated: 14 November 2026.
