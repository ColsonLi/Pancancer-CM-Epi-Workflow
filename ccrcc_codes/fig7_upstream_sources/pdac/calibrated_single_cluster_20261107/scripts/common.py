"""Fixed definitions for the PDAC single-expression-cluster calibration."""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[_k]='8'
from pathlib import Path
import sys, json, time, warnings, gc, h5py
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy import sparse, stats
from anndata.io import read_elem
import anndata as ad
from statsmodels.stats.multitest import multipletests
B=Path(__file__).resolve().parents[1]; P=B.parent; C=P/'epi-cm-core-workflow'
OLD=P/'cm_epi_redesign'; FULL=P/'full_atlas_reintegration_20261107'
SH=P.parent/'shared_clinical_and_signatures'
PY=C/'uv_envs/main/.venv/bin/python'
KIRC=Path('/mnt/disk18t/lr_xcy/riku/kirc_205/leiden_detailed/adata_epi.h5ad')
SEED=42
PRIMARY='A_HVG3000_sample_r0.4'
for _d in ['tables','objects','labels','figures','logs']:(B/_d).mkdir(parents=True,exist_ok=True)
PANELS={
 'JUN':['JUN','JUNB','FOS','FOSB','ATF3','KLF6','ZBTB20','PDK4','IRF1','DUSP1','IER2','ZFP36'],
 'VIM':['VIM','S100A4','S100A6','LGALS1','LGALS3','NUPR1','KRT17','S100A2','TGFBI','LAMC2','ITGA6','KRT14'],
 'Acinar':['GP2','PRSS1','PRSS2','REG1A','REG1B','CPA1','CPA2','CELA3A','CELA3B','CLPS','PNLIP','CTRB1','CTRB2','CPB1','PLA2G1B'],
 'Ductal':['EPCAM','KRT7','KRT8','KRT18','KRT19','MUC1','TACSTD2'],
 'NormalDuctal':['FXYD2','CFTR','SLC4A4','CLDN10','ANXA4','SERPING1','SOD3'],
 'MalignantDuctal':['S100P','CEACAM6','AGR2','TSPAN1','KRT17','LAMC2','LGALS3'],
 'Immune':['PTPRC','CD3D','CD3E','LST1','TYROBP','NKG7','MS4A1'],
 'TNK':['CD3D','CD3E','TRAC','CD2','NKG7','KLRD1','GNLY'],
 'Myeloid':['LST1','TYROBP','FCER1G','LYZ','AIF1','CTSS','CSF1R'],
 'B':['MS4A1','CD79A','CD79B','CD37','BANK1','CD19','MZB1','JCHAIN','SDC1','XBP1'],
 'Stromal':['COL1A1','COL1A2','DCN','LUM','COL3A1','PDGFRA','RGS5','PDGFRB','CSPG4','ACTA2','TAGLN'],
 'Endothelial':['PECAM1','VWF','EMCN','CDH5','KDR','RAMP2','PLVAP'],
 'Mast':['TPSAB1','TPSB2','CPA3','KIT','MS4A2','HPGDS'],
 'CM1':['CD3D','CD3E','CD3G','CD2','TRAC','LCK','IL7R','GZMK','CCL5','NKG7','GNLY','GZMB','KLRD1','TOX'],
}
DIGESTIVE=set(PANELS['Acinar']+['CEL','CELA2A','CELA2B','CTRC','CTRL','PNLIPRP1','PNLIPRP2','CTRB','AMY2A','AMY2B','REG3A','REG3G','REG4'])
KIDNEY=set('UMOD SLC12A1 SLC12A3 SLC34A1 SLC5A2 SLC22A6 SLC22A8 SLC3A1 SLC7A9 SLC17A1 SLC17A3 SLC22A2 SLC22A12 SLC13A1 SLC13A3 LRP2 CUBN AQP1 AQP2 ALDOB MIOX CLCNKA CLCNKB KCNJ1 NPHS1 NPHS2 PODXL SLC4A1 ATP6V1G3 FXYD2 CA9 NDUFA4L2 PAX2 PAX8 ENPEP DPEP1 GATM GLYAT ACSM2A ACSM2B AGXT AGXT2 BHMT BHMT2 HPD FABP1 FMO1 FMO2 FMO3 FMO4 FMO5 PDZK1 PDZK1IP1 SLC6A19 SLC6A18 SLC5A10 SLC5A12 KCNJ16 KCNJ10 AQP3 AQP6 ATP6V0D2 ATP6V1B1'.split())
FLAGGED=['B_CD3D','B_PRSS1','B_S100A6','Endo_ATP1B1','Endo_CXCR4','Endo_DCN','Endo_MYL9','Endo_PRSS1','Endo_TYROBP','Mast_B2M','Mast_CD79A','Mast_HLA-DRA','Mast_MUC1','Mast_PRSS1','Mye_CLPS','Mye_IL32','Mye_KRT19','Str_KRT19','Str_PRSS1','Str_SRGN','T_PRSS1']
def js(x,path):
    def cv(z):
        if isinstance(z,np.generic):z=z.item()
        if isinstance(z,float) and not np.isfinite(z):return None
        if isinstance(z,dict):return {str(k):cv(v) for k,v in z.items()}
        if isinstance(z,(list,tuple)):return [cv(v) for v in z]
        if isinstance(z,Path):return str(z)
        return z
    (B/path).write_text(json.dumps(cv(x),indent=2,default=str,allow_nan=False)+'\n')
def fdr(p):
    p=np.asarray(p,float);q=np.full(p.shape,np.nan);ok=np.isfinite(p)
    if ok.any():q[ok]=multipletests(p[ok],method='fdr_bh')[1]
    return q
def load_expr(path,raw=False):
    with h5py.File(path,'r') as f:
        o=read_elem(f['obs']);v=read_elem(f['raw/var' if raw else 'var']);x=read_elem(f['raw/X' if raw else 'X'])
    return sparse.csr_matrix(x,dtype=np.float32),o,v
def enrich(o):
    o=o.copy()
    for k in ['sample','series','patient_id','status','cell_subtype']:
        if k in o:o[k]=o[k].astype(str)
    u=pd.read_csv(P.parent/'external_atlas_for_cm-epi_validation/PDAC/UCSC/metadata_samples.csv').set_index('Sample_ID')
    o['study']=o.series
    ix=o.series.eq('UCSC');o.loc[ix,'study']=o.loc[ix,'sample'].map(u.Publication).fillna('UCSC_unknown')
    o['donor']=o.study.astype(str)+'::'+o.patient_id.astype(str)
    o['donor_key_basis']=np.where(o.patient_id.eq(o['sample']),'sample proxy; linkage unavailable','provided patient ID')
    return o
def means(x,genes):
    genes=pd.Index(genes)
    return pd.DataFrame({k:np.asarray(x[:,genes.get_indexer([g for g in gs if g in genes])].mean(axis=1)).ravel() for k,gs in PANELS.items()})
def groupmeans(x,lab):
    lab=pd.Categorical(lab)
    mat=sparse.csr_matrix((np.ones(len(lab),dtype=np.float32),(lab.codes,np.arange(len(lab)))),shape=(len(lab.categories),len(lab)))
    n=np.asarray(mat.sum(axis=1)).ravel();z=mat@x
    return (z.toarray() if sparse.issparse(z) else np.asarray(z))/n[:,None],lab.categories,n
def donor_profiles(x,o,label,genes,min_cells=5):
    keys=o[label].astype(str)+'||'+o.donor.astype(str);z,lev,n=groupmeans(x,keys)
    parts=pd.Series(lev).str.split('||',regex=False,expand=True);ok=n>=min_cells
    idx=pd.MultiIndex.from_arrays([parts.loc[ok,0],parts.loc[ok,1]],names=['cluster','donor'])
    return pd.DataFrame(z[ok],index=idx,columns=genes),pd.DataFrame({'cluster':parts.loc[ok,0].values,'donor':parts.loc[ok,1].values,'n_cells':n[ok]})
def correlate(x,y,study=None):
    x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y)
    x=x[ok];y=y[ok];n=len(x)
    if n<6 or np.std(x)==0 or np.std(y)==0:return np.nan,np.nan,n
    if study is None:r,p=stats.spearmanr(x,y);return float(r),float(p),n
    d=pd.get_dummies(np.asarray(study)[ok],drop_first=True,dtype=float).to_numpy();d=np.column_stack([np.ones(n),d])
    z=stats.rankdata(np.column_stack([x,y]),axis=0);z-=d@np.linalg.lstsq(d,z,rcond=None)[0]
    r=np.corrcoef(z.T)[0,1];df=n-d.shape[1]-1
    return float(r),float(2*stats.t.sf(abs(r)*np.sqrt(df/max(1e-15,1-r*r)),df)),n
