from common import *
import harmonypy as hm
import subprocess
from sklearn.metrics import adjusted_rand_score

def freeze_reference():
    root=Path('/mnt/disk18t/lr_xcy/riku/kirc_205/leiden_detailed')
    genes=[];frames=[]
    for state,fn in [('Epi_JUN','Epi_JUN_degs_vs_Epi_VIM.csv'),('Epi_VIM','Epi_VIM_degs_vs_Epi_JUN.csv')]:
        d=pd.read_csv(root/'epi_vim_vs_jun_pairwise_deg/results'/fn)
        d=d[(d.logfoldchanges>0)&(d.pvals_adj<.05)].drop_duplicates('gene').head(100)
        frames.append(d);genes.extend(d.gene)
    genes=sorted(set(genes));pd.concat(frames).to_csv(B/'tables/frozen_ccRCC_top100_per_state_DEGs.csv',index=False)
    with h5py.File(root/'adata_epi.h5ad','r') as f:
        obs=read_elem(f['obs']);v=read_elem(f['raw/var']); xx=read_elem(f['raw/X'])
    have=[g for g in genes if g in v.index];xx=sparse.csr_matrix(xx)[:,v.index.get_indexer(have)]
    obs['donor']=obs['series'].astype(str)+'::'+obs['sample'].astype(str)
    prof,support=donor_profiles(xx,obs,'cell_subtype',have)
    cent=prof.groupby(level='cluster').mean()
    # All reference states and donors define the same fixed gene coordinate system.
    mu=prof.mean(axis=0);sd=prof.std(axis=0).clip(lower=.1)
    cent.to_csv(B/'tables/frozen_ccRCC_donor_balanced_centroids.csv')
    prof.to_csv(B/'tables/frozen_ccRCC_donor_profiles.csv')
    support.to_csv(B/'tables/frozen_ccRCC_donor_support.csv',index=False)
    pd.DataFrame({'gene':have,'mean':mu,'sd':sd}).to_csv(B/'tables/frozen_ccRCC_gene_scaling.csv',index=False)
    js({'source':str(root/'adata_epi.h5ad'),'reference_states':obs.cell_subtype.value_counts().to_dict(),'features':'first100 original-order positive adjusted-P<.05 pairwise DEGs per original state; union','n_union':len(genes),'n_in_reference':len(have),'metric':'cosine of donor-balanced normalized-log centroids gene-z standardized by ccRCC-only all-state donor means/SD, SD floor .1; identical scaling for both states and every PDAC cluster','frozen_before_pdac_partition':True,'donor_identity':'series::sample (reference has no independent patient key)'},'tables/reference_contract.json')
    del xx;gc.collect()

def integrate(x,obs,var,name,nhvg=3000,res=.6,umap=True):
    start=time.time(); print('INTEGRATE',name,x.shape,flush=True)
    a=ad.AnnData(x,obs=obs.copy(),var=var.copy());a.uns['log1p']={'base':None}
    sc.pp.highly_variable_genes(a,n_top_genes=nhvg+300,flavor='seurat',batch_key='series')
    # Retain digestive, endocrine, HLA and immune programs; remove only ubiquitous
    # mitochondrial/ribosomal technical features from the dimension-reduction fit.
    eligible=~a.var_names.str.match(r'^(MT-|RPL|RPS)')
    idx=np.where(eligible&a.var.highly_variable.to_numpy())[0]
    idx=idx[np.argsort(-np.nan_to_num(a.var.dispersions_norm.to_numpy()[idx],nan=-999))[:nhvg]]
    a.var['selected_HVG']=False;a.var.iloc[idx,a.var.columns.get_loc('selected_HVG')]=True
    a.var.to_csv(B/f'tables/{name}_HVGs.csv')
    w=a[:,idx].copy();sc.pp.scale(w,max_value=10)
    sc.tl.pca(w,n_comps=40,svd_solver='randomized',random_state=SEED)
    # Finite 100-cluster fit keeps the correction bounded; every cell participates.
    pca=ad.AnnData(sparse.csr_matrix((len(obs),0)),obs=w.obs[['sample']].copy());pca.obsm['X_pca']=w.obsm['X_pca']
    pca.write_h5ad(B/f'objects/{name}_pca_for_harmony.h5ad')
    subprocess.run([str(C/'uv_envs/rapids/.venv/bin/python'),str(B/'scripts/gpu_harmony.py'),str(B/f'objects/{name}_pca_for_harmony.h5ad'),str(B/f'objects/{name}_harmony.npy'),str(B/f'tables/{name}_harmony_backend.json')],check=True)
    z=np.load(B/f'objects/{name}_harmony.npy')
    w.obsm['X_harmony']=z.astype(np.float32)
    js({'n_cells':len(obs),'n_hvg':len(idx),'n_pcs':40,'n_neighbors':30,'harmony_batch':'sample','no_status_regression':True,'backend':'RAPIDS harmony2 GPU; independent PCA fit CPU','max_iter_harmony':50,'elapsed_to_harmony_sec':time.time()-start},f'tables/{name}_integration.json')
    del pca;gc.collect()
    sc.pp.neighbors(w,n_neighbors=30,n_pcs=40,use_rep='X_harmony',random_state=SEED)
    for r in ([.4,.6,.8] if name=='epithelial' else [res]):
        key=f'leiden_r{r:.1f}';sc.tl.leiden(w,resolution=r,key_added=key,flavor='igraph',n_iterations=2,directed=False,random_state=SEED)
        w.obs[key]=name+'_20261107_c'+w.obs[key].astype(str)
    key=f'leiden_r{res:.1f}';w.obs['cluster']=w.obs[key].astype('category')
    if umap:sc.tl.umap(w,random_state=SEED,maxiter=150 if len(obs)>30000 else 250)
    # Expression saved in native normalized-log space, not scaled values.
    w.X=x[:,idx].copy();w.var=a.var.iloc[idx].copy();w.obs.to_csv(B/f'tables/{name}_cell_labels.csv.gz')
    w.uns['source_expression_h5ad']=str(C/'h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_normalized_log_all_genes.h5ad')
    w.uns['expression_role']='native normalized_total1e4_log1p, no additional normalization; all genes accessed by saved cell IDs in source object'
    w.write_h5ad(B/f'objects/{name}_integrated_hvg.h5ad',compression='gzip')
    dd=markers(x,w.obs,var,'cluster',name)
    mp=profile(x,var.index);mp.index=w.obs_names
    mp.groupby(w.obs.cluster,observed=True).mean().to_csv(B/f'tables/{name}_multimarker_cluster_profiles.csv')
    out=[]
    for cl,ix in w.obs.groupby('cluster',observed=True).indices.items():
        oo=w.obs.iloc[ix];dp=oo.donor.value_counts(normalize=True)
        out.append({'cluster':cl,'n_cells':len(ix),'n_samples':oo['sample'].nunique(),'n_donors':oo.donor.nunique(),'n_studies':oo.series.nunique(),'tumor_fraction':oo.status.eq('tumor').mean(),'max_donor_fraction':dp.max(),'median_n_genes':oo.n_genes_by_counts.median(),'median_mt':oo.pct_counts_MT.median(),'top15_markers':';'.join(dd[dd.group==cl].names.head(15))})
    pd.DataFrame(out).to_csv(B/f'tables/{name}_cluster_summary.csv',index=False)
    w.obs.groupby(['cluster','donor','series','sample','status'],observed=True).size().rename('n_cells').reset_index().to_csv(B/f'tables/{name}_cluster_sample_counts.csv',index=False)
    if name=='epithelial':
        stab=[]
        for k in ['leiden_r0.4','leiden_r0.8']:
            ct=pd.crosstab(w.obs.cluster,w.obs[k]);ct.to_csv(B/f'tables/epithelial_main_vs_{k}.csv')
            stab.append({'comparison':k,'ARI':adjusted_rand_score(w.obs.cluster,w.obs[k]),'main_n_clusters':w.obs.cluster.nunique(),'sensitivity_n_clusters':w.obs[k].nunique()})
        pd.DataFrame(stab).to_csv(B/'tables/epithelial_resolution_stability.csv',index=False)
    del w,a,mp;gc.collect();print('DONE',name,time.time()-start,flush=True)

def main():
    freeze_reference()
    fpath=C/'h5ad/01-celltype_integration_clustering/06-broad-annotation/adata_anno.h5ad'
    x,o,v=load_expr(fpath,raw=True)
    with h5py.File(fpath,'r') as f:
        emb={k:read_elem(f['obsm'][k]) for k in f['obsm']}
        hvg=read_elem(f['var'])
    # Recover pre-filter full atlas. These original broad labels have documented
    # multi-marker review and a converged all-cell sample-Harmony embedding.
    o['donor']=o.series.astype(str)+'::'+o.patient_id.astype(str)
    o['original_broad_lineage']=o.cell_type.astype(str)
    mapping={'Epithelial Cells':'Epithelial','T/NK Cells':'TNK','Myeloid Cells':'Myeloid','B Cells':'B','Stromal Cells':'Stromal','Endothelial Cells':'Endothelial','Mast Cells':'Mast'}
    o['broad_lineage']=o.cell_type.map(mapping).astype(str)
    prev=pd.read_csv(B/'tables/original_atlas_obs.csv.gz',index_col=0)
    o['retained_in_319042_atlas']=o.index.isin(prev.index)
    o['legacy_subtype']=prev.cell_subtype.reindex(o.index).astype(str).fillna('not_previously_retained')
    manifest=pd.read_csv(B/'tables/original_included_manifest.csv'); scm=manifest[manifest.modality.astype(str).str.contains('single|scRNA',case=False,regex=True)]
    observed=set(o['sample']);scman=manifest[manifest.modality.eq('single_cell')]
    allmanifest=set(np.where(scman.dataset.eq('UCSC'),scman.sample_id,scman.gsm))
    print('FULL',x.shape,'samples',len(observed),'manifest missing',allmanifest-observed,flush=True)
    sm=o.groupby(['series','sample','patient_id','donor','status'],observed=True).agg(n_cells=('sample','size'),n_previously_retained=('retained_in_319042_atlas','sum'),median_genes=('n_genes_by_counts','median'),median_mt=('pct_counts_MT','median')).reset_index()
    sm.to_csv(B/'tables/full_atlas_sample_inventory.csv',index=False)
    mp=profile(x,v.index);mp.index=o.index
    mp.groupby(o.leiden_res0p4,observed=True).mean().to_csv(B/'tables/full_atlas_existing_cluster_multimarker_profiles.csv')
    mp.groupby(o.broad_lineage,observed=True).mean().to_csv(B/'tables/full_atlas_broad_multimarker_profiles.csv')
    genes=sorted(set(sum(MARKERS.values(),[]))&set(v.index))
    gm,cl=groupmeans(x[:,v.index.get_indexer(genes)],o.leiden_res0p4)
    pd.DataFrame(gm,index=cl,columns=genes).to_csv(B/'tables/full_atlas_existing_cluster_marker_gene_means.csv')
    hvg.to_csv(B/'tables/full_atlas_archived_HVGs.csv')
    save_embedding(emb,o,'objects/full_atlas_verified_embeddings.h5ad',{'source_full_atlas':str(fpath),'source_expression':str(C/'h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_normalized_log_all_genes.h5ad'),'reuse':'full 392388-cell archived HVG/PCA/sample-Harmony/neighbors/Leiden/UMAP, before broad marker-rank filtering; sample Harmony converged at maxiter50; no disease-status regression'})
    o.to_csv(B/'tables/full_atlas_cell_labels.csv.gz')
    o.groupby(['sample','broad_lineage'],observed=True).size().unstack(fill_value=0).to_csv(B/'tables/full_atlas_sample_lineage_counts.csv')
    vals=x.data[::max(1,len(x.data)//1000000)]
    js({'n_cells':len(o),'n_samples':o['sample'].nunique(),'n_donors':o.donor.nunique(),'n_genes':len(v),'unique_gene_names':v.index.is_unique,'unique_cell_ids':o.index.is_unique,'tumor_normal_counts':o.status.value_counts().to_dict(),'lineage_counts':o.broad_lineage.value_counts().to_dict(),'restored_after_label_filter':int((~o.retained_in_319042_atlas).sum()),'manifest_singlecell_samples':len(allmanifest),'manifest_missing':list(allmanifest-observed),'normalized_log_noninteger_fraction':float((vals!=np.floor(vals)).mean()),'expression_min':float(vals.min()),'expression_max':float(vals.max()),'double_normalized':False,'all_cell_projection':False,'full_atlas_reuse':'archived all-cell fit, no cell threshold or epithelial restriction; numeric-QC and source-QC reused; later label-agreement filter reversed'},'tables/full_atlas_contract.json')
    del emb,mp,prev;gc.collect()
    for lineage in ['Epithelial','TNK','Myeloid','B','Stromal','Endothelial','Mast']:
        name='epithelial' if lineage=='Epithelial' else lineage.lower()
        if (B/f'objects/{name}_integrated_hvg.h5ad').exists(): print('EXISTS',name,flush=True);continue
        take=o.broad_lineage.eq(lineage).to_numpy();integrate(x[take],o.loc[take].copy(),v,name,3000,res=.6 if lineage=='Epithelial' else .5)

if __name__=='__main__':main()
