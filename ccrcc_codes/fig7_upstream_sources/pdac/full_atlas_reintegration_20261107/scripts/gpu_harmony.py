import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='8'
import sys,json,warnings,time
import anndata as ad
import rapids_singlecell as rsc
import numpy as np
inp,out,log=sys.argv[1:];a=ad.read_h5ad(inp);start=time.time()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    rsc.pp.harmony_integrate(a,key='sample',basis='X_pca',adjusted_basis='X_harmony',max_iter_harmony=50,n_clusters=min(100,max(10,a.n_obs//100)),random_state=20261107,verbose=True)
np.save(out,a.obsm['X_harmony'].astype(np.float32))
open(log,'w').write(json.dumps({'backend':'rapids_singlecell '+rsc.__version__+' harmony2 GPU','max_iter_harmony':50,'elapsed_seconds':time.time()-start,'warnings':[str(w.message) for w in caught],'converged':not any('did not converge' in str(w.message).lower() for w in caught)},indent=2))
