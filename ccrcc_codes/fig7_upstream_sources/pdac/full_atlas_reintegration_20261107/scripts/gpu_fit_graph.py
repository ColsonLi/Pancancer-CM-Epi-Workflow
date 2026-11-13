import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='8'
import sys,time,json,warnings
import anndata as ad
import rapids_singlecell as rsc
from pathlib import Path
inp,out,name=sys.argv[1:];a=ad.read_h5ad(inp);start=time.time()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    rsc.pp.harmony_integrate(a,key='sample',basis='X_pca',adjusted_basis='X_harmony',max_iter_harmony=50,n_clusters=min(100,max(10,a.n_obs//100)),random_state=20261107,verbose=True)
report={'backend':'rapids_singlecell '+rsc.__version__+' harmony2 GPU','max_iter_harmony':50,'elapsed_seconds':time.time()-start,'warnings':[str(w.message) for w in caught],'converged':not any('did not converge' in str(w.message).lower() for w in caught)}
rsc.pp.neighbors(a,n_neighbors=30,n_pcs=40,use_rep='X_harmony',random_state=20261107)
for res in ([.4,.6,.8] if name=='epithelial' else [.5]):
    key=f'leiden_r{res:.1f}';rsc.tl.leiden(a,resolution=res,key_added=key,random_state=20261107,n_iterations=100);a.obs[key]=name+'_20261107r_c'+a.obs[key].astype(str)
a.obs['cluster']=a.obs['leiden_r0.6' if name=='epithelial' else 'leiden_r0.5']
rsc.tl.umap(a,random_state=20261107,maxiter=200)
a.write_h5ad(out,compression='gzip');p=Path(out).parents[1]/'tables'/f'{name}_harmony_backend.json';p.write_text(json.dumps(report,indent=2));print('GPU graph done',name,time.time()-start,flush=True)
