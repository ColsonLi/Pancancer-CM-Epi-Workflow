import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']: os.environ[k]='8'
from pathlib import Path
import json, h5py, numpy as np, pandas as pd
from anndata.io import read_elem
B=Path(__file__).resolve().parents[1]; P=B.parent; C=P/'epi-cm-core-workflow'
for d in ['scripts','tables','objects','figures','logs']: (B/d).mkdir(exist_ok=True)
rows=[]
paths=list((C/'h5ad').rglob('*.h5ad'))+[P/'cm_epi_redesign/objects/cleaned_tumor_ductal.h5ad',P/'cm_epi_redesign/objects/adata_cleaned_tumor_ductal_best_states.h5ad',Path('/mnt/disk18t/lr_xcy/riku/kirc_205/leiden_detailed/adata_epi.h5ad')]
for p in paths:
    if 'spatial' in str(p): continue
    try:
        with h5py.File(p,'r') as f:
            o=read_elem(f['obs']);v=read_elem(f['var'])
            rr={'path':str(p),'n_cells':len(o),'n_genes':len(v),'obs_columns':';'.join(o.columns),'obsm':';'.join(f.get('obsm',{}).keys()),'raw_genes':len(read_elem(f['raw/var'])) if 'raw' in f else 0,'uns':';'.join(f.get('uns',{}).keys())}
            for col in o:
                if any(s in col.lower() for s in ['cluster','leiden','hvg3000','best','state']):
                    rr[col+'_unique']=o[col].nunique()
            rows.append(rr)
            if 'adata_anno_cellsubtype' in p.name:
                o.to_csv(B/'tables/original_atlas_obs.csv.gz');v.to_csv(B/'tables/original_atlas_var.csv')
                print('ATLAS',o.shape,rr,flush=True)
                for col in ['series','sample','status','patient_id','cell_type','cell_subtype','broad_lineage']: 
                    if col in o: print(col,o[col].value_counts().head(30).to_dict(),flush=True)
            if 'kirc_205' in str(p):
                o.to_csv(B/'tables/kirc_reference_obs.csv.gz');v.to_csv(B/'tables/kirc_reference_var.csv')
                print('KIRC',rr,flush=True)
    except Exception as e: rows.append({'path':str(p),'error':str(e)})
pd.DataFrame(rows).to_csv(B/'tables/checkpoint_inventory.csv',index=False)
m=pd.read_csv(C/'tables/01-celltype_integration_clustering/01-input-metadata-audit/included/metadata.csv');m.to_csv(B/'tables/original_included_manifest.csv',index=False)
print('MANIFEST',m.shape,m.columns.tolist(),flush=True)
print(m.groupby(['dataset','status']).size().to_string(),flush=True)
