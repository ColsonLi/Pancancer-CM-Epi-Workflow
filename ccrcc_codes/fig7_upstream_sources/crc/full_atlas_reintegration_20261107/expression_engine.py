from full_common import *
import anndata as ad,scanpy as sc,harmonypy as hm
from pynndescent import NNDescent
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
sc.settings.n_jobs=min(8,int(os.environ.get('NUMBA_NUM_THREADS','8')));sc.settings.verbosity=2
def fitting_rows(obs,cap,stratify_major=False):
 if len(obs)<=cap:return np.arange(len(obs))
 rng=np.random.default_rng(42)
 keys=['sample','cell_type'] if stratify_major else ['sample']
 groups=obs.groupby(keys,observed=True).indices
 quota=max(1,int(np.ceil(cap/len(groups))));chunks=[]
 for ix in groups.values():chunks.append(rng.choice(ix,min(len(ix),quota),replace=False))
 fit=np.unique(np.concatenate(chunks))
 if len(fit)>cap:fit=np.sort(rng.choice(fit,cap,replace=False))
 elif len(fit)<cap:
  remain=np.setdiff1d(np.arange(len(obs)),fit);fit=np.sort(np.r_[fit,rng.choice(remain,min(cap-len(fit),len(remain)),False)])
 return fit
def integrate(x,obs,genes,out,prefix,cap,global_scope=False,sensitivities=False):
 out=Path(out);out.mkdir(exist_ok=True,parents=True)
 fit=fitting_rows(obs,cap,stratify_major=global_scope)
 print(prefix,'cells',len(obs),'fit',len(fit),flush=True)
 eligible=~pd.Series(genes).str.match(GENE_FILTER).to_numpy()
 a=ad.AnnData(x[fit][:,eligible].copy(),obs=obs.iloc[fit].copy(),var=pd.DataFrame(index=genes[eligible]));a.uns['log1p']={'base':None}
 ns=a.obs['sample'].value_counts();hfit=a.obs['sample'].map(ns).ge(50).to_numpy()
 if hfit.sum()<1000:hfit=np.ones(len(a),dtype=bool)
 hv=a[hfit].copy();sc.pp.highly_variable_genes(hv,flavor='seurat',n_top_genes=min(3000,a.n_vars-1),batch_key='sample' if a.obs['sample'].nunique()>1 else None);a.var=hv.var.copy();del hv
 a.var.to_csv(out/'HVG_statistics.csv',index_label='gene');a=a[:,a.var.highly_variable].copy()
 logx=a.X.copy();sc.pp.scale(a,max_value=10);pca_center=np.asarray(a.X.mean(axis=0)).ravel();npcs=min(40,a.n_obs-1,a.n_vars-1);sc.tl.pca(a,n_comps=npcs,random_state=42,svd_solver='arpack')
 scale_mean=a.var['mean'].to_numpy();scale_sd=a.var['std'].to_numpy();loadings=a.varm['PCs'].copy();a.X=logx
 ho=hm.run_harmony(a.obsm['X_pca'],a.obs,['study_source_v1'],theta=2,lamb=1,max_iter_harmony=20,random_state=42,device='cpu')
 corr=np.asarray(ho.Z_corr);corr=corr if corr.shape==a.obsm['X_pca'].shape else corr.T;a.obsm['X_pca_harmony']=corr.astype('float32');del ho
 sc.pp.neighbors(a,n_neighbors=min(30,len(a)-1),use_rep='X_pca_harmony',random_state=42)
 configs=[(.6,42)]+([(.4,42),(.8,42),(.6,1729)] if sensitivities else [])
 for resolution,seed in configs:
  key=f'leiden_r{resolution:g}_s{seed}';sc.tl.leiden(a,resolution=resolution,key_added=key,random_state=seed,flavor='igraph',directed=False,n_iterations=4)
  a.obs[key]=[f'{prefix}_r{int(resolution*10):02d}_{int(c):02d}'+('' if seed==42 else f'_s{seed}') for c in a.obs[key]]
 a.obs['cluster_v1']=a.obs['leiden_r0.6_s42'];sc.tl.umap(a,random_state=42,maxiter=200)
 np.save(out/'fit_row_indices.npy',fit)
 a.write_h5ad(out/'fitted_expression_reference.h5ad',compression='gzip')
 # Standard PCA out-of-sample transform, in chunks, using frozen fit scaling.
 gi=pd.Index(genes).get_indexer(a.var_names);pc=np.empty((len(obs),npcs),dtype='float32')
 for start in range(0,len(obs),8192):
  end=min(start+8192,len(obs));block=x[start:end,gi].toarray();block=(block-scale_mean)/np.maximum(scale_sd,1e-8);np.clip(block,-10,10,out=block);pc[start:end]=(block-pca_center)@loadings
 pc[fit]=a.obsm['X_pca']
 # Propagate the local Harmony displacement from neighboring fitted cells in
 # uncorrected PCA, never using a state/program score as the mapping manifold.
 print(prefix,'mapping all cells',flush=True)
 index=NNDescent(a.obsm['X_pca'].astype('float32'),n_neighbors=30,random_state=42,n_jobs=sc.settings.n_jobs);index.prepare()
 nei,dist=index.query(pc,k=15);weights=1/np.maximum(dist,1e-3);weights/=weights.sum(1,keepdims=True)
 delta=a.obsm['X_pca_harmony']-a.obsm['X_pca'];corrected=pc+np.einsum('nk,nkd->nd',weights,delta[nei]);corrected[fit]=a.obsm['X_pca_harmony'];del index,delta,nei,dist,weights;gc.collect()
 index=NNDescent(a.obsm['X_pca_harmony'].astype('float32'),n_neighbors=30,random_state=42,n_jobs=sc.settings.n_jobs);index.prepare()
 nei,dist=index.query(corrected,k=30);w=1/np.maximum(dist,1e-3);w/=w.sum(1,keepdims=True)
 labels=obs.copy();labels['in_expression_fit_reference']=False;labels.iloc[fit,labels.columns.get_loc('in_expression_fit_reference')]=True
 for resolution,seed in configs:
  key=f'leiden_r{resolution:g}_s{seed}';categories=sorted(a.obs[key].unique());codes=pd.Categorical(a.obs[key],categories=categories).codes
  votes=np.zeros((len(obs),len(categories)),dtype='float32')
  for j in range(30):np.add.at(votes,(np.arange(len(obs)),codes[nei[:,j]]),w[:,j])
  labels[key]=np.asarray(categories)[votes.argmax(1)];labels.iloc[fit,labels.columns.get_loc(key)]=a.obs[key].to_numpy()
  if resolution==.6 and seed==42:labels['mapping_vote_confidence']=votes.max(1)
 labels['cluster_v1']=labels['leiden_r0.6_s42']
 um=np.einsum('nk,nkd->nd',w,a.obsm['X_umap'][nei]);um[fit]=a.obsm['X_umap']
 np.savez_compressed(out/'all_cell_representations.npz',X_pca=pc,X_pca_harmony=corrected,X_umap=um,HVGs=a.var_names.to_numpy(),PCs=loadings,scaling_mean=scale_mean,scaling_sd=scale_sd,pca_center=pca_center)
 np.savez_compressed(out/'all_to_fitted_mapping_neighbors.npz',fitted_row_indices=fit,neighbor_fit_indices=nei,distances=dist,weights=w)
 labels.to_csv(out/'per_cell_labels.csv.gz',index_label='cell_id')
 # Reusable full-cell compact object; source expression and exact row/gene
 # identities are referenced, not duplicated as a misleading count layer.
 full=ad.AnnData(obs=labels.copy(),var=pd.DataFrame(index=genes));full.obsm['X_pca']=pc;full.obsm['X_pca_harmony']=corrected;full.obsm['X_umap']=um
 full.uns['expression_source']={'h5ad':str(ATLAS),'matrix':'raw/X','gene_identity':'raw/var index','subset_by_obs_names':True,'scale':'normalized natural-log1p; no second normalization'}
 full.uns['integration_method']={'fit_n':len(fit),'all_n':len(obs),'fit_cap':cap,'n_HVGs':a.n_vars,'n_PCs':npcs,'harmony_batch':'series','harmony_fit':'balanced expression reference','all_cell_mapping':'PCA transform; local Harmony displacement; 30-NN weighted cluster vote; fitted UMAP barycentric projection','fitted_graph':str(out/'fitted_expression_reference.h5ad'),'mapping_graph':str(out/'all_to_fitted_mapping_neighbors.npz'),'seed':42,'Leiden_iterations':4}
 full.write_h5ad(out/'all_cells_expression_integrated.h5ad',compression='gzip')
 pd.crosstab([labels['sample'],labels.study_id,labels.donor_id_v1,labels.status],labels.cluster_v1).to_csv(out/'sample_donor_cluster_counts.csv')
 # Donor-balanced top markers (annotation only), using the complete gene universe.
 pb,pm=pseudobulk(x,labels,['donor_id_v1','study_id','cluster_v1'],min_cells=10)
 prof=balance_profiles(pb,pm,label='cluster_v1')[0];prof.columns=genes;prof.to_csv(out/'balanced_cluster_profiles_log1p.csv.gz',index_label='cluster')
 background=prof.mean(0);marker=[]
 for cl,row in prof.iterrows():
  delta=row-background
  eligible=(row>=.15)&~pd.Series(genes,index=genes).str.match(GENE_FILTER)
  ranked=delta.loc[eligible].sort_values(ascending=False).head(50)
  for rank,(g,v) in enumerate(ranked.items(),1):marker.append({'cluster':cl,'rank':rank,'gene':g,'delta_vs_equal_cluster_background':v,'balanced_mean_log1p':row[g]})
 pd.DataFrame(marker).to_csv(out/'top50_expression_markers.csv',index=False)
 support=labels.groupby('cluster_v1').agg(n_cells=('sample','size'),n_samples=('sample','nunique'),n_patients=('donor_id_v1','nunique'),n_studies=('study_id','nunique'),tumor_fraction=('status',lambda z:z.eq('tumor').mean()))
 support['top5_markers']=pd.DataFrame(marker).groupby('cluster').head(5).groupby('cluster').gene.agg(';'.join);support.to_csv(out/'cluster_support_markers.csv')
 savejson(out/'completion.json',{'complete':True,'all_cells':len(obs),'fitted_cells':len(fit),'clusters':labels.cluster_v1.nunique(),'samples':labels['sample'].nunique(),'studies':labels.study_id.nunique(),'source':str(ATLAS),'mapping_explicit':True})
 print(prefix,'COMPLETE',len(obs),'cells',labels.cluster_v1.nunique(),'clusters',flush=True)
 return labels,prof
