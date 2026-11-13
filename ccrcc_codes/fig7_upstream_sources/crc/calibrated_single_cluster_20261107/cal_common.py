from pathlib import Path
import os, sys, json, gc, warnings, time
import numpy as np
import pandas as pd
import h5py
from scipy import sparse, stats
from scipy.spatial.distance import cdist
from scipy.optimize import nnls, linear_sum_assignment
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
CRC = ROOT.parent
OLD = CRC/'cm_epi_redesign'
FA = CRC/'full_atlas_reintegration_20261107'
ER = CRC/'epi_expression_reintegration/reference'
SHARED = CRC.parent/'shared_clinical_and_signatures'
sys.path.insert(0, str(CRC/'epi_expression_reintegration'))
from analysis_common import read_col, read_frame, load_csr, pseudobulk, donor_ids, savejson, GENE_FILTER
ATLAS = CRC/'epi-cm-core-workflow/h5ad/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata/adata_anno_cellsubtype.h5ad'
PYTHON = '/mnt/disk18t/lr_xcy/riku/go_bp/pathway_analysis/cellsubtype_degs/.micromamba/envs/codex_lr_scanpy311/bin/python'
SEED = 20261107
CORE = {
 'Epi_JUN': ['JUN','JUNB','FOS','FOSB','ATF3','KLF6','ZBTB20','PDK4','IRF1','DUSP1','IER2','ZFP36'],
 'Epi_VIM': ['VIM','S100A4','S100A6','LGALS1','LGALS3','NUPR1','KRT17','S100A2','TGFBI','LAMC2','ITGA6','KRT14']}

def bh(p):
 p=np.asarray(p,float);out=np.full(p.shape,np.nan);ix=np.flatnonzero(np.isfinite(p));order=ix[np.argsort(p[ix])]
 if len(order):out[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
 return out

def balanced_profile(pb, pm, label='cluster_id'):
 # Equal donors within each study, then equal studies. Pseudobulks already
 # combine repeated samples of the same biological donor and cluster.
 f=pd.DataFrame(pb);f['study_id']=pm.study_id.to_numpy();f[label]=pm[label].to_numpy()
 return f.groupby([label,'study_id'],observed=True).mean(numeric_only=True).groupby(level=0).mean()

def load_epi():
 return sparse.load_npz(ROOT/'data/tumor_epithelial_log1p.npz'),pd.read_csv(ROOT/'data/tumor_epithelial_metadata.csv.gz',index_col=0),pd.read_csv(ROOT/'data/genes.csv').gene.to_numpy()

def safe_spearman(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y)
 if ok.sum()<5 or np.std(x[ok])==0 or np.std(y[ok])==0:return np.nan,np.nan
 r,p=stats.spearmanr(x[ok],y[ok]);return float(r),float(p)

def partial_rank(x,y,groups):
 x=stats.rankdata(x);y=stats.rankdata(y);d=pd.get_dummies(np.asarray(groups),dtype=float).to_numpy();d=np.c_[np.ones(len(x)),d]
 rank=np.linalg.matrix_rank(d);xr=x-d@np.linalg.lstsq(d,x,rcond=None)[0];yr=y-d@np.linalg.lstsq(d,y,rcond=None)[0]
 if np.std(xr)<1e-12 or np.std(yr)<1e-12:return np.nan,np.nan,len(x)-rank-1
 r=float(np.corrcoef(xr,yr)[0,1]);df=len(x)-rank-1;p=float(2*stats.t.sf(abs(r)*np.sqrt(df/max(1-r*r,1e-12)),df))
 return r,p,df

def cos(a,b):
 return np.asarray(a)@np.asarray(b).T/np.maximum(np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=1)[None,:],1e-12)
