from cal_common import *
import anndata as ad
import scanpy as sc
import harmonypy as hm
from sklearn.metrics import silhouette_score, adjusted_rand_score

def main():
 sc.settings.n_jobs=min(6,int(os.environ.get('NUMBA_NUM_THREADS','6')));sc.settings.verbosity=2
 d=ROOT/'epithelial';d.mkdir(exist_ok=True)
 x,obs,genes=load_epi();eligible=~pd.Series(genes).str.match(GENE_FILTER).to_numpy()
 a=ad.AnnData(x[:,eligible].copy(),obs=obs[['sample_id','study_id','series','donor_id']].copy(),var=pd.DataFrame(index=genes[eligible]));a.uns['log1p']={'base':None}
 sc.pp.highly_variable_genes(a,flavor='seurat',n_top_genes=3000,batch_key='study_id')
 hv=a.var.copy();hv.to_csv(d/'all_gene_HVG_statistics.csv')
 ranking=hv.sort_values(['highly_variable_nbatches','dispersions_norm'],ascending=False).index.tolist()
 savejson(d/'configuration_specification.json',{'primary_configuration':'hvg3000_r0.4','primary_choice_before_similarity':'3000 genes preserve more state information; resolution .4 is the middle tested granularity, avoiding preselected score programs or maximum rank/association searches. All five resolutions retained for sensitivity.','HVG_sizes':[2000,3000],'resolutions':[.2,.3,.4,.5,.6],'batch':'study_id','batch_rationale':'protocol/study effects corrected; sample-level correction would remove genuine between-patient state differences needed in downstream cooccurrence','n_PCs':40,'neighbors':30,'seed':42,'scope':'ALL cleaned tumor-source epithelial cells; no fitted-subset label propagation','dimension_reduction_exclusions':GENE_FILTER,'DEG_and_markers':'all source genes retained for annotation/DE; dimension-reduction exclusions do not delete genes from source matrix','candidate_gate':'>=500 cells AND >=15 tumor samples','source_expression':'raw normalized natural-log1p; no program-score manifold, no reclustering of program scores'})
 qc=[];alllabels=pd.DataFrame(index=obs.index)
 for n in [2000,3000]:
  out=d/f'hvg{n}';out.mkdir(exist_ok=True);gs=ranking[:n];ix=pd.Index(genes).get_indexer(gs)
  z=ad.AnnData(x[:,ix].copy(),obs=obs[['sample_id','study_id','series','donor_id']].copy(),var=pd.DataFrame(index=gs))
  # Sparse standard-deviation scaling with exact centered sparse PCA; do not
  # densify the full expression matrix and do not use score-space clustering.
  sc.pp.scale(z,zero_center=False);sc.tl.pca(z,n_comps=40,zero_center=True,svd_solver='arpack',random_state=42)
  print('PCA completed',n,flush=True)
  ho=hm.run_harmony(z.obsm['X_pca'],z.obs,['study_id'],theta=2,lamb=1,max_iter_harmony=30,max_iter_kmeans=20,random_state=42,device='cpu')
  corr=np.asarray(ho.Z_corr);corr=corr if corr.shape==z.obsm['X_pca'].shape else corr.T;z.obsm['X_pca_harmony']=corr.astype('float32');del ho
  sc.pp.neighbors(z,n_neighbors=30,use_rep='X_pca_harmony',random_state=42)
  pd.Series(gs,name='gene').to_csv(out/'HVGs.csv',index=False)
  for res in [.2,.3,.4,.5,.6]:
   name=f'hvg{n}_r{res:g}';key=f'leiden_r{res:g}';sc.tl.leiden(z,resolution=res,key_added=key,random_state=42,flavor='igraph',directed=False,n_iterations=4)
   labs=z.obs[key].astype(str).map(lambda c:f'CRCcalQ2_{name}_c{int(c):02d}');alllabels[name]=labs.to_numpy()
   tab=obs[['sample_id','study_id','donor_id']].copy();tab['cluster_id']=labs.to_numpy();tab.to_csv(d/f'labels_{name}.csv.gz',index_label='cell_id')
   support=tab.groupby('cluster_id').agg(n_cells=('sample_id','size'),n_samples=('sample_id','nunique'),n_patients=('donor_id','nunique'),n_studies=('study_id','nunique'))
   support['candidate_gate']=support.n_cells.ge(500)&support.n_samples.ge(15);support.to_csv(d/f'support_{name}.csv')
   # A random capped silhouette is a QC description, not optimized for the
   # JUN/VIM winner. Same seed/sample cap for every tested configuration.
   sil=silhouette_score(corr,labs,sample_size=min(7000,len(obs)),random_state=42)
   cross=pd.crosstab(labs.to_numpy(),obs.study_id.to_numpy());p=cross.div(cross.sum(1),axis=0);ent=-(p*np.log(p.clip(lower=1e-12))).sum(1)
   qc.append(dict(configuration=name,n_HVG=n,resolution=res,n_clusters=len(support),n_candidate_clusters=int(support.candidate_gate.sum()),fraction_cells_in_candidates=float(support.loc[support.candidate_gate,'n_cells'].sum()/len(obs)),silhouette_40_HarmonyPCs=float(sil),median_study_entropy=float(ent.median()),PCA_explained_fraction=float(z.uns['pca']['variance_ratio'].sum()),primary=name=='hvg3000_r0.4'))
   print('Leiden',name,qc[-1],flush=True)
  sc.tl.umap(z,random_state=42,maxiter=300)
  np.savez_compressed(out/'representations.npz',X_pca=z.obsm['X_pca'],X_pca_harmony=z.obsm['X_pca_harmony'],X_umap=z.obsm['X_umap'],PCs=z.varm['PCs'],HVGs=np.array(gs),scaling_sd=z.var['std'].to_numpy(),variance_ratio=z.uns['pca']['variance_ratio'])
  compact=ad.AnnData(obs=z.obs.copy());compact.obsm['X_pca']=z.obsm['X_pca'];compact.obsm['X_pca_harmony']=z.obsm['X_pca_harmony'];compact.obsm['X_umap']=z.obsm['X_umap'];compact.obsp['connectivities']=z.obsp['connectivities'];compact.obsp['distances']=z.obsp['distances'];compact.uns['expression_source']=str(ROOT/'data/tumor_epithelial_log1p.npz');compact.write_h5ad(out/'all_cells_graph_and_labels.h5ad',compression='gzip')
  del z;gc.collect()
 pd.DataFrame(qc).to_csv(d/'all_configuration_QC.csv',index=False);alllabels.to_csv(d/'all_configuration_per_cell_labels.csv.gz',index_label='cell_id')
 ari=[]
 for i,c1 in enumerate(alllabels):
  for c2 in alllabels.columns[i+1:]:ari.append(dict(configuration_1=c1,configuration_2=c2,ARI=adjusted_rand_score(alllabels[c1],alllabels[c2])))
 pd.DataFrame(ari).to_csv(d/'cross_configuration_ARI.csv',index=False)
 savejson(d/'clustering_completion.json',{'complete':True,'n_cells':len(obs),'configurations':alllabels.columns.tolist(),'primary_configuration':'hvg3000_r0.4','all_cells_directly_clustered':True,'label_paths':[str(d/f'labels_{name}.csv.gz') for name in alllabels]})

if __name__=='__main__':
 with threadpool_limits(limits=6):main()
