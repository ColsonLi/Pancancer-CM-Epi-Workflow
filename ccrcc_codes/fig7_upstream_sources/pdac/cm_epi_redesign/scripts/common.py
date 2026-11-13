"""Shared fixed programs and utilities; all output remains under cm_epi_redesign."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ.setdefault(key, '8')
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
CORE = ROOT/'epi-cm-core-workflow'
for d in ['tables','objects','figures','logs']:
    (BASE/d).mkdir(parents=True, exist_ok=True)
PANELS = {
    'JUN': ['JUN','JUNB','FOS','FOSB','ATF3','KLF6','ZBTB20','PDK4','IRF1','DUSP1','IER2','ZFP36'],
    'VIM': ['VIM','S100A4','S100A6','LGALS1','LGALS3','NUPR1','KRT17','S100A2','TGFBI','LAMC2','ITGA6','KRT14'],
    'Hypoxia': ['CA9','VEGFA','SLC2A1','BNIP3','NDRG1','LDHA','PGK1','ENO1'],
    'Classical': ['GATA6','HNF4A','HNF1A','FOXA2','TFF1','TFF2','AGR2','MUC5AC','CEACAM6','S100P'],
    'Acinar': ['GP2','PRSS1','PRSS2','REG1A','REG1B','CPA1','CPA2','CELA3A','CELA3B','CLPS','PNLIP','CTRB1','CTRB2','CPB1','PLA2G1B'],
    'NormalDuctal': ['FXYD2','CFTR','SLC4A4','CLDN10','ANXA4','SERPING1','SOD3'],
    'Ductal': ['EPCAM','KRT7','KRT8','KRT18','KRT19','MUC1','TACSTD2'],
    'MalignantDuctal': ['S100P','CEACAM6','AGR2','TSPAN1','KRT17','LAMC2','LGALS3'],
    'Immune': ['PTPRC','CD3D','CD3E','LST1','TYROBP','NKG7','MS4A1'],
    'CM1': ['CD3D','CD3E','CD3G','CD2','TRAC','LCK','IL7R','GZMK','CCL5','NKG7','GNLY','GZMB','KLRD1','TOX'],
    'CM10': ['COL1A1','COL1A2','COL3A1','DCN','LUM','CTHRC1','COL11A1','FAP','LST1','C1QA','C1QB','TYROBP','SPP1'],
}
DIGESTIVE = set(PANELS['Acinar'] + ['CEL','CELA2A','CELA2B','CTRC','CTRL','PNLIPRP1','PNLIPRP2','CTRB','AMY2A','AMY2B','REG3A','REG3G','REG4','CTRB2'])
STRICT_DUCTAL = ['Epi_'+x for x in ['S100A6','S100A11','STMN1','IL32','LGALS4','CTSE']]
FLAGGED = ['B_CD3D','B_PRSS1','B_S100A6','Endo_ATP1B1','Endo_CXCR4','Endo_DCN','Endo_MYL9','Endo_PRSS1','Endo_TYROBP','Mast_B2M','Mast_CD79A','Mast_HLA-DRA','Mast_MUC1','Mast_PRSS1','Mye_CLPS','Mye_IL32','Mye_KRT19','Str_KRT19','Str_PRSS1','Str_SRGN','T_PRSS1']

def save_json(x, name):
    def clean(z):
        if isinstance(z,np.generic):z=z.item()
        if isinstance(z,float) and not np.isfinite(z):return None
        if isinstance(z,dict):return {str(k):clean(v) for k,v in z.items()}
        if isinstance(z,(list,tuple)):return [clean(v) for v in z]
        if isinstance(z,Path):return str(z)
        return z
    (BASE/name).write_text(json.dumps(clean(x),indent=2,allow_nan=False,default=str)+'\n')

def fdr(p):
    p=np.asarray(p,float); q=np.full(p.shape,np.nan); ok=np.isfinite(p)
    if ok.any(): q[ok]=multipletests(p[ok],method='fdr_bh')[1]
    return q

def means(X, genes, panels=PANELS):
    genes=pd.Index(genes)
    return pd.DataFrame({name:np.asarray(X[:,genes.get_indexer([g for g in panel if g in genes])].mean(axis=1)).ravel() for name,panel in panels.items()})

def rho(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<4 or np.std(x[ok])==0 or np.std(y[ok])==0:return np.nan,np.nan,int(ok.sum())
    r,p=stats.spearmanr(x[ok],y[ok]); return float(r),float(p),int(ok.sum())

def partial(x,y,c):
    a=np.column_stack([x,y,c]); a=a[np.isfinite(a).all(axis=1)]
    if len(a)<10:return np.nan,np.nan,len(a)
    a=stats.rankdata(a,axis=0); design=np.column_stack([np.ones(len(a)),a[:,2:]])
    res=a[:,:2]-design@np.linalg.lstsq(design,a[:,:2],rcond=None)[0]
    r=np.corrcoef(res.T)[0,1]; df=len(a)-design.shape[1]-1
    p=2*stats.t.sf(abs(r)*np.sqrt(df/max(1e-15,1-r*r)),df)
    return r,p,len(a)

def read_raw(path):
    import h5py
    from anndata.io import read_elem
    with h5py.File(path,'r') as f:
        o=read_elem(f['obs']); v=read_elem(f['raw/var']); x=read_elem(f['raw/X'])
    return sparse.csr_matrix(x,dtype=np.float32), o, v

def spatial_manifest():
    p=CORE/'tables/04-spatial-validation-optional/03-tangram-reference/spatial_sample_scope_manifest.csv'
    m=pd.read_csv(p)
    m['output_h5ad']=[str(CORE/'h5ad/04-spatial-validation-optional/02-build-spatial-h5ad'/Path(p).name) for p in m.output_h5ad]
    return m
