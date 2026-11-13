from pathlib import Path
import sys, json, gc, os
import numpy as np,pandas as pd,h5py
from scipy import sparse,stats
ROOT=Path(__file__).resolve().parent
OLD_EPI=ROOT.parent/'epi_expression_reintegration'
sys.path.insert(0,str(OLD_EPI))
from analysis_common import read_col,read_frame,load_csr,subset_csr,pseudobulk,donor_ids,savejson,GENE_FILTER,balance_profiles
ATLAS=ROOT.parent/'epi-cm-core-workflow/h5ad/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata/adata_anno_cellsubtype.h5ad'
WF=ROOT.parent/'epi-cm-core-workflow'
REF=ROOT.parent.parent/'shared_clinical_and_signatures/reference'
LINEAGES={
 'Epithelial':['EPCAM','KRT8','KRT18','KRT19','CDH1','CEACAM5','VIL1','PHGR1'],
 'T_NK':['CD3D','CD3E','TRAC','CD247','NKG7','GNLY','KLRD1'],
 'Myeloid':['LST1','TYROBP','FCER1G','CSF1R','AIF1','C1QA'],
 'B':['MS4A1','CD79A','CD79B','CD19','BANK1'],
 'Plasma':['MZB1','JCHAIN','DERL3','TNFRSF17','SDC1'],
 'Stromal':['COL1A1','COL1A2','COL3A1','DCN','LUM','PDGFRA'],
 'Pericyte':['RGS5','PDGFRB','CSPG4','MCAM','NOTCH3'],
 'Endothelial':['PECAM1','VWF','KDR','EMCN','CDH5'],
 'Mast':['TPSAB1','TPSB2','CPA3','KIT','MS4A2'],
 'Schwann':['SOX10','S100B','PLP1','MPZ','SLC1A3']}
def load_original():
 with h5py.File(ATLAS) as f:
  cols=['sample','series','study_id','patient_id','status','original_barcode','sample_id','source_h5ad','source_row_index','platform','cell_type','cell_subtype','doublet_score','predicted_doublet','n_genes_by_counts','total_counts','pct_counts_MT']
  obs=read_frame(f['obs'],[c for c in cols if c in f['obs']]);genes=read_col(f['raw/var'][f['raw/var'].attrs['_index']]);x=load_csr(f['raw/X']).astype('float32')
 obs['donor_id_v1']=donor_ids(obs);obs['study_source_v1']=obs.series.astype(str);obs['original_cell_subtype']=obs.cell_subtype.astype(str)
 return x,obs,genes
def load_metadata():return pd.read_csv(ROOT/'all_cell_labels.csv.gz',index_col=0,low_memory=False)
