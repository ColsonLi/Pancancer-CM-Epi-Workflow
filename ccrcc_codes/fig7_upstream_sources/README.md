# Fig. 7 上游源码

对应当前 `Fig7_JUN_only_symmetric.pdf`。入口为上一层的 [fig7-2_cross_cancer_upstream_analysis.ipynb](../fig7-2_cross_cancer_upstream_analysis.ipynb)，绘图仍在 `fig7-1_cross_cancer_jun_reference_and_tnk_associations.ipynb`。

本目录收录 42 份原始脚本/环境记录，内容与来源逐字节一致；文件来源、用途、SHA256 和大小见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。[runner.py](runner.py) 是新加的执行入口和源表适配层，不冒充原始科学代码。整理时只读取原分析目录，未运行拟合、改写原始结果或重新导出图。

## 当前主分析

| 癌种 | 原始分支 | 当前上皮 | 当前 CM | 主图关联 |
| --- | --- | --- | --- | --- |
| CRC | `crc_codex/calibrated_single_cluster_20261107/` | HVG3000 / Leiden 0.4，15 群中 9 群满足支持条件；选 c08 | 14 模块中的 CRCcal_CM08，65 非上皮亚群 | 127 个肿瘤样本；JUN fraction 对 raw NNLS CM activity |
| PDAC | `pdac_codex/calibrated_single_cluster_20261107/` | A_HVG3000_sample_r0.4，12 群中 9 群满足支持条件；选 c7 | 11 模块中的 calCM02，54 非上皮亚群 | 74 个肿瘤样本；JUN fraction 对 raw NNLS CM activity |

两个癌种的主关联均要求每样本至少 30 个保留上皮细胞、50 个保留非上皮细胞。主图显示 nominal、双侧 Spearman P，不用上游完整敏感性家族的 q 替代。上皮分母为该样本全部保留的肿瘤来源上皮细胞，不限 JUN/VIM 两群。

## 起点与输入生产关系

这里的“上游”从已有 h5ad/表达对象和 metadata 开始，不包含原始测序下载、比对或公共 atlas 原作者的全部预处理。数据未复制进来，原路径保持不变。外部项目真实位置为 `/mnt/my_storage/data/lr_riku/validation_4_kirc_agents/`；原 `/home/lr_xcy/lr_xcy/riku/lr_riku/validation_4_kirc_agents/` 别名在当前机器解析到同一位置。

CRC：

1. 原 `epi-cm-core-workflow` 最终全基因表达/注释 h5ad，是清理与重新分析的起点。
2. `cm_epi_redesign/01_prepare_clean_data.py` 产生 reviewed clean mask。
3. `full_atlas_reintegration_20261107/01_full_atlas.py` 产生背景 atlas 和粗谱系；`02_lineage_reclustering.py` 的非上皮分支产生新的表达型 T/NK 群。配套 `full_common.py`、`expression_engine.py` 已收录。
4. `epi_expression_reintegration/02_freeze_ccrcc_reference.py` 产生 ccRCC 参考所需表达与 metadata，读取既有 ccRCC 上皮及 DEG 表。
5. 当前 calibrated 的 `01_prepare` → `02_CM` → `03_expression_clustering` → `04_reference_and_similarity` → `05_singlecell` 是 Fig. 7 的真正主链。

CRC atlas 绘图还读取既有 `all_cell_final_labels.csv.gz` 的背景注释。其历史生产代码 `full_atlas_reintegration_20261107/03_new_CM_discovery.py`、`04_new_EPI_CM_associations.py` 和上皮匹配辅助源码已附，用于追溯该背景表；不自动执行，不将那个历史 CM/上皮版本拿来替代 calibrated 结果。`02_lineage_reclustering.py` 的可选旧上皮函数仍按原路径加载历史匹配脚本；当前 notebook 只调用非上皮分支。

当前 CRC prepare 脚本还保留 `cm_epi_redesign/spatial/panel_genes.csv` 的历史依赖；生成它的 `04_spatial.py` 也已附。最终 JUN-only 图不读取空间结果，也不执行该脚本的空间检验分支。

PDAC：

1. 原 `epi-cm-core-workflow` 各谱系全基因 h5ad 及样本 metadata 是起点；UCSC 样本研究/患者信息来自 `external_atlas_for_cm-epi_validation/PDAC/UCSC/metadata_samples.csv`。
2. `cm_epi_redesign/scripts/02_epithelial_sweep.py` 产生当前复用的 `cleaned_tumor_ductal.h5ad`（55,357 个细胞）和清理/历史对照表。该脚本包含旧程序评分/扫参，已有输入时不必运行，旧标签不会成为当前 c7。
3. calibrated 的 `00_ccRCC_reference`、`00_cleanup_evidence` → `01_cm_calibration` → `02_expression_sweep` → `03_reference_matching` → `04_three_modalities:singlecell` 是主链。
4. 背景 atlas 的 inventory、整合/谱系处理、review/refit、标签汇总和 GPU helper 原码也已附在 `full_atlas_reintegration_20261107/scripts/`。当前图读取已有背景坐标，再叠加 calibrated 的标签，不用历史背景 CM 替代 calCM02。历史背景脚本中的 GPU 子进程调用仍保留原文件路径；不在本 notebook 自动重建整套背景分支。

因此，包含了主分析及其前置对象的生产源码，但不是“对空目录执行一遍就从 FASTQ 自动生成全部对象”的独立软件包。背景分支需要其原始输入、软件环境和按依赖顺序执行的历史代码。

## 执行与导出

- 导入 `runner.py` 和调用 `preflight()` 只读检查，不启动科学分析。notebook 默认 `RUN_ANALYSES = False`。
- 显式设为 `True` 后，各单元执行本目录的原脚本副本；`__file__` 保持原位置含义，因此原输入、输出路径和算法参数不变。CRC/PDAC 的同名 `common.py` 模块按阶段隔离。
- CRC 保留原线程限制、seed、扫参范围和缓存行为；原 CM 脚本遇到已有拟合表时可能复用缓存。本次没有清理缓存，也未假装执行了一次全新拟合。
- PDAC 仅提取并执行 `04_three_modalities.py` 的原始 `singlecell(pair, cm)` 函数体，不导入 `clinical_eval`，不运行混合脚本的 main、clinical 或 spatial。
- `export_pdac_singlecell_display_inputs()` 单独提取原 finalization/report 的单细胞表和 normal-reference 标签生成逻辑，避免依赖旧临床/空间报告。
- 两个 `render_*_joint_atlas_fig7style.py` 仅调用 `prepare_data()`，由已保存的真实背景坐标生成逐细胞绘图表，不拟合新 UMAP、不提前画图。
- `export_figure_inputs()` 调用原相似性源表提取脚本，再按原规则整理当前绘图所需字段、窗口和 loading。它是明确标注的导出适配层，不进行生物学参数选择；共享表中已有的其他癌种行予以保留。原完整 `prepare_source_data.py` 作为来源备查，不在主入口运行其旧 ccRCC/balance 分支。
- 统计核对后，另运行 `fig7-1` 绘图。原始顶层 PDF 不被这次代码整理覆盖。

运行前备份需要保留的结果，沿用各分支的环境记录。真正执行会写入原外部项目目录，可能覆盖同名文件；本次整理不代表已授权或已完成一次重跑。
