from pathlib import Path
import importlib.util, sys
import numpy as np, pandas as pd, h5py
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parent
WF=ROOT.parent/'epi-cm-core-workflow'
ATLAS=WF/'h5ad/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata/adata_anno_cellsubtype.h5ad'
EPI=WF/'h5ad/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/epithelial/adata_epi.h5ad'
XEN=WF/'h5ad/04-spatial-validation-optional/01-input-audit/adata_xenium_unprocessed.h5ad'
FIG7=Path('/mnt/disk18t/lr_xcy/riku/kirc_205/final/fig7_extension_20261004/CRC')

PROGRAMS={
 'JUN':['JUN','FOS','JUNB','FOSB','KLF6','ZBTB20','ATF3','EGR1','PDK4','IRF1','DUSP1','IER2','BTG2','ZFP36'],
 'JUN_nonIEG':['KLF6','ZBTB20','PDK4','IRF1','BTG2','GADD45B','KLF4','DDIT4'],
 'VIM':['VIM','LGALS1','LGALS3','S100A4','S100A6','NUPR1','TGFBI','FN1','SPARC','ITGA5','EMP3','SERPINE1'],
 'Hypoxia':['CA9','CA12','HIF1A','VEGFA','SLC2A1','BNIP3','NDRG1','LDHA','ENO1'],
 'Proliferative':['MKI67','TOP2A','UBE2C','CENPF','PCNA','MCM2','MCM5','STMN1','BIRC5','CDK1'],
 'WNT_Stem':['LGR5','ASCL2','OLFM4','AXIN2','SMOC2','SOX9','RNF43','MYC','EPHB2','CD44'],
 'Goblet_Secretory':['MUC2','TFF3','SPINK4','REG4','CLCA1','AGR2','FCGBP','ZG16','SPDEF'],
 'Epithelial':['EPCAM','KRT8','KRT18','KRT19','CDH1','KRT20','CEACAM5','CEACAM6','VIL1','CDX2','PHGR1'],
 'Leukocyte':['PTPRC','LST1','TYROBP','FCER1G','CD3D','CD3E','CD247','NKG7','MS4A1'],
 'Myeloid':['LST1','TYROBP','FCER1G','CTSS','AIF1','CSF1R','CD68','C1QA','C1QB'],
 'Stromal':['COL1A1','COL1A2','COL3A1','DCN','LUM','PDGFRA','COL6A1'],
 'Endothelial':['PECAM1','VWF','KDR','EMCN','CDH5','RAMP2'],
 'Tcell':['CD3D','CD3E','TRAC','CD247','LCK','CD2'],
 'Bcell':['CD79A','CD79B','MS4A1','CD19','CD37','CD74'],
 'Plasma':['MZB1','JCHAIN','SDC1','XBP1','DERL3','TNFRSF17'],
 'Mast':['TPSAB1','TPSB2','CPA3','KIT','MS4A2','HPGDS'],
 'Schwann':['SOX10','S100B','PLP1','MPZ','SLC1A3'],
 'T_NK':['NKG7','GNLY','GZMB','GZMK','PRF1','CD3D','CD3E','CD8A','CCL5','CTSW','KLRD1','TOX','CXCL13'],
 'T_GZMB':['GZMB','PRF1','GZMH','CTSW','CCL5','FGFBP2'],
 'T_GZMK':['GZMK','CXCR3','CCL4','EOMES'],
 'T_TOX_CXCL13':['TOX','CXCL13','PDCD1','HAVCR2','LAG3','TIGIT','ENTPD1'],
 'T_NKcyto':['GNLY','KLRD1','KLRF1','KLRC1','NKG7','FCGR3A'],
 'T_Naive':['CCR7','IL7R','TCF7','LEF1','LTB'],
 'T_Treg':['FOXP3','IL2RA','CTLA4','IKZF2'],
 'Myeloid_Stromal':['C1QA','C1QB','CD163','APOE','SPP1','LST1','COL1A1','COL1A2','DCN','LUM','FN1','SPARC'],
}

def read_col(node):
 if isinstance(node,h5py.Group):
  cats=read_col(node['categories']); codes=node['codes'][:]
  return np.asarray([cats[i] if i>=0 else '' for i in codes])
 if h5py.check_string_dtype(node.dtype):return node.asstr()[:]
 return node[:]

def read_frame(group,cols=None):
 return pd.DataFrame({k:read_col(group[k]) for k in (cols or list(group.attrs['column-order']))},index=read_col(group[group.attrs['_index']]))

def load_csr(group):
 return sparse.csr_matrix((group['data'][:],group['indices'][:],group['indptr'][:]),shape=tuple(group.attrs['shape']))

def bh(p):
 p=np.asarray(p,dtype=float);q=np.full_like(p,np.nan);v=np.isfinite(p)
 if v.any():q[v]=multipletests(p[v],method='fdr_bh')[1]
 return q

def corr(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);ok=np.isfinite(x)&np.isfinite(y)
 if ok.sum()<5 or np.std(x[ok])==0 or np.std(y[ok])==0:return np.nan,np.nan
 r,p=stats.spearmanr(x[ok],y[ok]);return float(r),float(p)

def zscore(x,axis=0):
 x=np.asarray(x);return (x-np.nanmean(x,axis=axis,keepdims=True))/np.maximum(np.nanstd(x,axis=axis,keepdims=True),1e-5)

def gene_scores(x,genes,programs=PROGRAMS):
 idx={g:i for i,g in enumerate(genes)};used=sorted(set(g for v in programs.values() for g in v if g in idx))
 mat=x[:,[idx[g] for g in used]]
 if sparse.issparse(mat):mat=mat.toarray()
 zz=zscore(np.asarray(mat,dtype=np.float32));lookup={g:i for i,g in enumerate(used)}
 raw={};scores={}
 for k,gs in programs.items():
  ids=[lookup[g] for g in gs if g in lookup]
  scores[k]=zz[:,ids].mean(1) if ids else np.zeros(x.shape[0])
  raw[k]=mat[:,ids].mean(1) if ids else np.zeros(x.shape[0])
 return pd.DataFrame(scores),pd.DataFrame(raw)

def signed_wilcoxon(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return float(stats.wilcoxon(x,alternative='two-sided',method='auto').pvalue) if len(x)>0 and np.any(x!=0) else 1.
