import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']: os.environ[k]='8'
from pathlib import Path
import json, time, gc, h5py, warnings
import numpy as np
import pandas as pd
from scipy import sparse, stats
from anndata.io import read_elem
import anndata as ad
import scanpy as sc
from statsmodels.stats.multitest import multipletests
B=Path(__file__).resolve().parents[1]; P=B.parent; C=P/'epi-cm-core-workflow'
V=P.parent; SEED=20261107
sc.settings.n_jobs=8
MARKERS={
 'Ductal':['EPCAM','KRT7','KRT8','KRT18','KRT19','MUC1','TACSTD2'],
 'Acinar':['CPA1','CPA2','PRSS1','PRSS2','CELA3A','CELA3B','CTRB1','CLPS','PNLIP'],
 'Endocrine':['CHGA','CHGB','SCGN','ISL1','PCSK1','INS','GCG','SST'],
 'TNK':['CD3D','CD3E','TRAC','CD2','NKG7','KLRD1','GNLY'],
 'Myeloid':['LST1','TYROBP','FCER1G','LYZ','AIF1','CTSS','CSF1R'],
 'B':['MS4A1','CD79A','CD79B','CD37','BANK1','CD19'],
 'Plasma':['MZB1','JCHAIN','SDC1','XBP1','DERL3','TNFRSF17'],
 'Stromal':['COL1A1','COL1A2','DCN','LUM','COL3A1','PDGFRA'],
 'Pericyte':['RGS5','PDGFRB','CSPG4','NOTCH3','MCAM','ACTA2','TAGLN'],
 'Endothelial':['PECAM1','VWF','EMCN','CDH5','KDR','RAMP2','PLVAP'],
 'Mast':['TPSAB1','TPSB2','CPA3','KIT','MS4A2','HPGDS'],
 'NormalDuctal':['CFTR','FXYD2','SLC4A4','CLDN10','ANXA4'],
 'TumorDuctal':['S100P','CEACAM6','AGR2','TSPAN1','KRT17','LAMC2'],
 'Cycling':['MKI67','TOP2A','UBE2C','CENPF','STMN1'],
 'JUN':['JUN','JUNB','FOS','FOSB','ATF3','DUSP1','ZFP36'],
 'EMT':['VIM','S100A4','LGALS1','TGFBI','LAMC2','ZEB1'],
 'T_naive':['CCR7','TCF7','LEF1','IL7R','LTB'],
 'T_reg':['FOXP3','IL2RA','CTLA4','IKZF2','TIGIT'],
 'T_cytotoxic':['CD8A','CD8B','CCL5','GZMK','GZMB','PRF1'],
 'NK':['GNLY','KLRD1','NKG7','FCGR3A','XCL1'],
 'T_exhausted':['PDCD1','LAG3','HAVCR2','TOX','CXCL13'],
 'IFN':['ISG15','IFIT1','IFIT3','MX1','IFI6'],
 'Mono':['FCN1','VCAN','S100A8','S100A9','CTSD'],
 'Macrophage':['C1QA','C1QB','APOE','CD68','MRC1'],
 'SPP1_Mac':['SPP1','APOC1','GPNMB','TREM2'],
 'cDC':['FCER1A','CD1C','CLEC10A','CLEC9A','CST3'],
 'pDC':['GZMB','JCHAIN','TCF4','IL3RA','CLEC4C'],
 'Neutrophil':['CSF3R','FCGR3B','CXCR2','S100A8','S100A9'],
 'CAF':['CTHRC1','FAP','COL11A1','POSTN','INHBA'],
 'Fibroblast':['PI16','COL14A1','C7','CFD','DPT'],
 'Lymphatic':['PROX1','PDPN','FLT4','CCL21','LYVE1'],
 'Venous_Endo':['ACKR1','SELP','VWF'],
 'Capillary_Endo':['CA4','RGCC','EMCN','KDR'],
}
def js(x,path):
    (B/path).write_text(json.dumps(x,indent=2,default=lambda z:z.item() if isinstance(z,np.generic) else str(z))+'\n')
def qvals(p):
    p=np.asarray(p,float);q=np.full(p.shape,np.nan);ok=np.isfinite(p)
    if ok.any():q[ok]=multipletests(p[ok],method='fdr_bh')[1]
    return q
def load_expr(path,raw=False):
    with h5py.File(path,'r') as f:
        o=read_elem(f['obs']);v=read_elem(f['raw/var' if raw else 'var']);x=read_elem(f['raw/X' if raw else 'X'])
    return sparse.csr_matrix(x,dtype=np.float32),o,v
def profile(x,genes):
    genes=pd.Index(genes)
    return pd.DataFrame({k:np.asarray(x[:,genes.get_indexer([g for g in gs if g in genes])].mean(axis=1)).ravel() for k,gs in MARKERS.items()})
def groupmeans(x,lab):
    lab=pd.Categorical(lab);mat=sparse.csr_matrix((np.ones(len(lab),dtype=np.float32),(lab.codes,np.arange(len(lab)))),shape=(len(lab.categories),len(lab)))
    n=np.asarray(mat.sum(axis=1)).ravel();return (mat@x).toarray()/n[:,None],lab.categories
def donor_profiles(x,obs,label,genes):
    rows=[]; keys=[]
    for (cl,donor),ix in obs.groupby([label,'donor'],observed=True).indices.items():
        if len(ix)>=5:
            rows.append(np.asarray(x[ix].mean(axis=0)).ravel());keys.append((str(cl),str(donor),len(ix)))
    df=pd.DataFrame(rows,columns=genes,index=pd.MultiIndex.from_tuples([(a,b) for a,b,n in keys],names=['cluster','donor']))
    return df,pd.DataFrame(keys,columns=['cluster','donor','n_cells'])
def save_embedding(emb,obs,path,uns=None):
    a=ad.AnnData(sparse.csr_matrix((len(obs),0)),obs=obs.copy())
    for k,z in emb.items():a.obsm[k]=z
    if uns:a.uns.update(uns)
    a.write_h5ad(B/path,compression='gzip')
def markers(x,obs,var,label,prefix):
    a=ad.AnnData(x,obs=obs[[label]].copy(),var=var.copy());a.uns['log1p']={'base':None}
    sc.tl.rank_genes_groups(a,label,method='t-test',n_genes=60,use_raw=False)
    dd=sc.get.rank_genes_groups_df(a,None);dd.to_csv(B/f'tables/{prefix}_markers.csv',index=False)
    return dd
