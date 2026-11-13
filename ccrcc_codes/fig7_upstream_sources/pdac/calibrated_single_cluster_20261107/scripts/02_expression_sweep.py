from common import *
import scanpy as sc
import harmonypy as hm
from sklearn.metrics import adjusted_rand_score
import importlib.metadata
sc.settings.n_jobs=8

def main():
    ep=ad.read_h5ad(OLD/'objects/cleaned_tumor_ductal.h5ad');ep.obs=enrich(ep.obs)
    assert ep.n_obs==55357 and ep.obs.status.eq('tumor').all()
    # This is precisely the archived conservative tumor-ductal cell population.
    # No target-state program or CM is used for dimensionality reduction.
    ep.obs[['sample','series','study','patient_id','donor','donor_key_basis','status','cell_subtype']].to_csv(B/'tables/epithelial_input_cells.csv.gz',index_label='cell_id')
    cleanup=pd.read_csv(OLD/'tables/epithelial_cleanup_cells.csv.gz',index_col=0)
    cleanup.to_csv(B/'tables/epithelial_cleanup_and_normal_reference.csv.gz',index_label='cell_id')
    cleanup.groupby(['status','cell_subtype']).agg(n=('retained_primary','size'),retained_tumor_ductal=('retained_primary','sum')).to_csv(B/'tables/epithelial_population_flow.csv')
    exclude=DIGESTIVE|set(PANELS['Immune'])|set(PANELS['CM1'])
    configs=[];alllabels=pd.DataFrame(index=ep.obs_names)
    for nh in [2000,3000,4000]:
        t=time.time();sc.pp.highly_variable_genes(ep,n_top_genes=nh+len(exclude),flavor='seurat',batch_key='series')
        use=ep.var.highly_variable.to_numpy()&~ep.var_names.isin(exclude)&~ep.var_names.str.match(r'^(MT-|RPL|RPS)')
        ix=np.where(use)[0];ix=ix[np.argsort(-np.nan_to_num(ep.var.dispersions_norm.to_numpy()[ix],nan=-999))[:nh]]
        pd.DataFrame({'gene':ep.var_names[ix]}).to_csv(B/f'tables/A_HVG{nh}_genes.csv',index=False)
        w=ep[:,ix].copy();sc.pp.scale(w,max_value=10);sc.tl.pca(w,n_comps=40,svd_solver='arpack',random_state=42)
        # Original sample-level Harmony 0.0.10, identical arguments and PC count.
        harm=hm.run_harmony(w.obsm['X_pca'],w.obs,'sample',max_iter_harmony=30,random_state=42,verbose=True)
        z=np.asarray(harm.Z_corr);z=z if z.shape[0]==w.n_obs else z.T
        w.obsm['X_harmony']=z.astype(np.float32)
        sc.pp.neighbors(w,n_neighbors=20,n_pcs=30,use_rep='X_harmony',random_state=42)
        for res in [.3,.4,.5,.6,.8]:
            config=f'A_HVG{nh}_sample_r{res}'
            sc.tl.leiden(w,resolution=res,key_added='cluster',flavor='igraph',n_iterations=2,directed=False,random_state=42)
            cl=w.obs.cluster.astype(str).to_numpy();alllabels[config]=cl
            lab=ep.obs[['sample','study','donor']].copy();lab['cluster_id']=['c'+x for x in cl]
            lab.rename(columns={'sample':'sample_id'}).to_csv(B/f'labels/{config}.csv.gz',index_label='cell_id')
            configs.append({'configuration':config,'n_cells':len(cl),'n_clusters':len(np.unique(cl)),'HVG_requested':nh,'HVG_actual':len(ix),'PCA':40,'Harmony':'sample','neighbors':20,'neighbor_PCs':30,'resolution':res,'seed':42,'Harmony_iterations':len(harm.objective_harmony)-1,'labels_path':str(B/f'labels/{config}.csv.gz')})
            print('SAVED',configs[-1],flush=True)
            if config==PRIMARY:
                sc.tl.umap(w,random_state=42)
                saved=ad.AnnData(sparse.csr_matrix((w.n_obs,0)),obs=ep.obs.copy())
                saved.obs['cluster_id']=['c'+x for x in cl]
                for key in ['X_pca','X_harmony','X_umap']:saved.obsm[key]=w.obsm[key].copy()
                saved.uns['expression_source']=str(OLD/'objects/cleaned_tumor_ductal.h5ad')
                saved.write_h5ad(B/'objects/primary_epithelial_embeddings.h5ad',compression='gzip')
        np.savez_compressed(B/f'objects/HVG{nh}_harmony_graph.npz',PCA=w.obsm['X_pca'],Harmony=z,connectivities_data=w.obsp['connectivities'].data,connectivities_indices=w.obsp['connectivities'].indices,connectivities_indptr=w.obsp['connectivities'].indptr)
        print('HVG done',nh,round(time.time()-t,1),flush=True)
        del w,harm,z;gc.collect()
    pd.DataFrame(configs).to_csv(B/'tables/epithelial_configuration_manifest.csv',index=False)
    alllabels.to_csv(B/'labels/all_configurations_wide.csv.gz',index_label='cell_id')
    # Historical cluster membership was not archived, but counts, programs,
    # sample metrics and endpoint labels were. Test all independent summaries.
    old=pd.read_csv(OLD/'tables/A_cluster_program_annotations.csv');old=old[old.epi_scheme.eq(PRIMARY)].copy();old.cluster=old.cluster.astype(str)
    new=alllabels[PRIMARY];newcounts=new.value_counts();comp=old.set_index('cluster')[['n','n_samples','axis']].copy();comp['replay_n']=newcounts
    comp['replay_n_samples']=[ep.obs.loc[new.eq(c),'sample'].nunique() for c in comp.index]
    prog=pd.read_csv(OLD/'tables/epithelial_program_scores_all.csv.gz',index_col=0).loc[ep.obs_names]
    axis=prog.JUN-prog.VIM;comp['replay_axis']=[axis[new.eq(c)].mean() for c in comp.index]
    comp['count_exact']=comp.n.eq(comp.replay_n);comp['sample_count_exact']=comp.n_samples.eq(comp.replay_n_samples);comp['axis_abs_difference']=(comp.axis-comp.replay_axis).abs()
    comp.to_csv(B/'tables/historical_13cluster_summary_replay.csv')
    # Endpoint agreement can establish partition compatibility, not certify
    # unavailable old cell-by-cluster labels.
    st=old.set_index('cluster').state;newstate=new.map(st)
    archived=pd.read_csv(OLD/'tables/epithelial_scheme_cell_labels.csv.gz',index_col=0)[PRIMARY].reindex(new.index)
    jaccards=[]
    for config in alllabels:
        jaccards.append({'configuration':config,'ARI_to_primary':adjusted_rand_score(new,alllabels[config])})
    pd.DataFrame(jaccards).to_csv(B/'tables/epithelial_partition_ARI.csv',index=False)
    js({'primary_configuration':PRIMARY,'primary_fixed_reason':'historical expression-only replay anchor, fixed before similarity or association; other configurations sensitivity, not winner shopping','n_cells':ep.n_obs,'n_samples':ep.obs['sample'].nunique(),'n_donors':ep.obs.donor.nunique(),'n_studies':ep.obs.study.nunique(),'input':'identical saved cleaned_tumor_ductal population from cm_epi_redesign; conservative old ductal labels, tumor source, Acinar<Ductal and Ductal>Immune; excludes normal acinar/normal-dominated FXYD2 population','normal_reference_table':'tables/epithelial_cleanup_and_normal_reference.csv.gz','historical_n_clusters':len(old),'replay_n_clusters':new.nunique(),'all_archived_counts_exact':bool(comp.count_exact.all()),'max_axis_difference':comp.axis_abs_difference.max(),'historical_endpoint_cell_agreement':float(newstate.eq(archived).mean()),'exact_historical_cell_by_cluster_archive_available':False,'historical_identity_claim':'deterministic configuration replay with complete numerical summary and endpoint comparison; original c9/c6 cell-by-cluster archive absent, so cannot independently certify exact original membership','versions':{p:importlib.metadata.version(p) for p in ['scanpy','harmonypy','anndata','numpy','igraph','scikit-learn']}},'tables/epithelial_clustering_contract.json')
    print('EPITHELIAL COMPLETE',flush=True)
if __name__=='__main__':main()
