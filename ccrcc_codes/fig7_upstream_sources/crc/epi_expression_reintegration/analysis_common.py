from pathlib import Path
import json, gc, numpy as np, pandas as pd, h5py
from scipy import sparse
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'cm_epi_redesign'
ATLAS=ROOT.parent/'epi-cm-core-workflow/h5ad/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata/adata_anno_cellsubtype.h5ad'
KIRC=Path('/mnt/disk18t/lr_xcy/riku/kirc_205/leiden_detailed/adata_epi.h5ad')
PAIR=KIRC.parent/'epi_vim_vs_jun_pairwise_deg/results'
V=ROOT.parent.parent
GENE_FILTER=r'^(MT-|RPL[0-9]|RPS[0-9]|RPLP[0-9]|RPSA$)'
def read_col(node):
 if isinstance(node,h5py.Group):
  cats=read_col(node['categories']); codes=node['codes'][:];return np.where(codes>=0,cats[np.maximum(codes,0)],'')
 return node.asstr()[:] if h5py.check_string_dtype(node.dtype) else node[:]
def read_frame(g,cols=None):
 return pd.DataFrame({k:read_col(g[k]) for k in (cols or list(g.attrs['column-order']))},index=read_col(g[g.attrs['_index']]))
def load_csr(g):
 return sparse.csr_matrix((g['data'][:],g['indices'][:],g['indptr'][:]),shape=tuple(g.attrs['shape']))
def subset_csr(g,rows):
 # Read only selected CSR rows, merging nearby rows to reduce compressed I/O.
 rows=np.asarray(rows); ip=g['indptr'][:]; breaks=np.r_[0,np.where(np.diff(rows)>64)[0]+1,len(rows)]
 parts=[]
 for b,e in zip(breaks[:-1],breaks[1:]):
  lo,hi=rows[b],rows[e-1]+1; st,en=ip[lo],ip[hi]
  block=sparse.csr_matrix((g['data'][st:en],g['indices'][st:en],ip[lo:hi+1]-st),shape=(hi-lo,int(g.attrs['shape'][1])))
  parts.append(block[rows[b:e]-lo])
 return sparse.vstack(parts,format='csr')
def savejson(path,obj):
 Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def donor_ids(obs):
 # Patient IDs can repeat between studies. Preserve IDs but namespace them.
 p=obs['patient_id'].fillna('').astype(str) if 'patient_id' in obs else obs['sample'].astype(str)
 bad=p.str.lower().isin(['','nan','none','unknown','na'])
 p=p.mask(bad,obs['sample'].astype(str))
 st=obs['study_id'] if 'study_id' in obs else obs['series']
 return st.astype(str)+'::'+p
def load_full():
 return sparse.load_npz(ROOT/'crc_retained_log1p.npz'),pd.read_csv(ROOT/'crc_retained_metadata.csv.gz',index_col=0,low_memory=False),pd.read_csv(ROOT/'crc_genes.csv').gene.to_numpy()
def pseudobulk(x,obs,groupcols,min_cells=1):
 keys=obs[groupcols].astype(str).agg('|||'.join,axis=1)
 codes,unique=pd.factorize(keys,sort=True); n=np.bincount(codes)
 agg=sparse.csr_matrix((np.ones(len(obs),dtype='float32'),(codes,np.arange(len(obs)))),shape=(len(unique),len(obs)))
 means=(agg@x).toarray()/n[:,None]
 meta=pd.DataFrame([k.split('|||') for k in unique],columns=groupcols);meta['n_cells']=n
 keep=n>=min_cells
 return means[keep].astype('float32'),meta.loc[keep].reset_index(drop=True)
def balance_profiles(profiles,meta,label='cluster'):
 # Profiles already average cells within donor; average donors then studies.
 tmp=pd.DataFrame(profiles);tmp['study']=meta['study_id'].to_numpy();tmp['label']=meta[label].to_numpy()
 by=tmp.groupby(['label','study'],observed=True).mean(numeric_only=True)
 return by.groupby(level=0).mean(),by
