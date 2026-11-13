from common import *
import shutil,subprocess
from sklearn.metrics import adjusted_rand_score

# Fixed multi-marker evidence, defined after the expression-only broad-lineage
# review and BEFORE any cross-cancer similarity or CM association calculation.
# Digestive enzymes NEVER reassign a non-epithelial cell.
PANELS={
 'Ductal':['EPCAM','KRT7','KRT19','MUC1','TACSTD2'],
 'TNK':['CD3D','CD3E','TRAC','CD2','NKG7','KLRD1','GNLY'],
 'Myeloid':['LST1','TYROBP','FCER1G','AIF1','CTSS','CSF1R'],
 'B':['MS4A1','CD79A','CD79B','BANK1','CD19'],
 'Plasma':['MZB1','JCHAIN','SDC1','DERL3','TNFRSF17'],
 'Stromal':['COL1A1','COL1A2','DCN','LUM','COL3A1','PDGFRA'],
 'Pericyte':['RGS5','PDGFRB','CSPG4','NOTCH3','MCAM','ACTA2','TAGLN'],
 'Endothelial':['PECAM1','VWF','EMCN','CDH5','KDR','RAMP2','PLVAP'],
 'Mast':['TPSAB1','TPSB2','CPA3','KIT','MS4A2','HPGDS']}

def review(x,o,v):
    groups=['Epithelial','TNK','Myeloid','B','Stromal','Endothelial','Mast'];scores={};npos={}
    for name,gs in PANELS.items():
        xx=x[:,v.index.get_indexer([g for g in gs if g in v.index])]
        scores[name]=np.asarray(xx.mean(axis=1)).ravel();npos[name]=np.asarray((xx>0).sum(axis=1)).ravel()
    S=np.column_stack([scores['Ductal'],scores['TNK'],scores['Myeloid'],np.maximum(scores['B'],scores['Plasma']),np.maximum(scores['Stromal'],scores['Pericyte']),scores['Endothelial'],scores['Mast']])
    N=np.column_stack([npos['Ductal'],npos['TNK'],npos['Myeloid'],np.maximum(npos['B'],npos['Plasma']),np.maximum(npos['Stromal'],npos['Pericyte']),npos['Endothelial'],npos['Mast']])
    original=o.broad_lineage.astype(str).to_numpy();oldidx=np.array([groups.index(g) for g in original]);ix=np.arange(len(o));new=original.copy();reason=np.full(len(o),'retained archived broad identity',dtype=object)
    strong=(S>=1.0)&(N>=3)
    # Acinar/endocrine identity protects genuine epithelial specialization from
    # reassignment. Enzymes in other original lineages do not trigger any action.
    ac=x[:,v.index.get_indexer(MARKERS['Acinar'])];en=x[:,v.index.get_indexer(MARKERS['Endocrine'])]
    protected=original=='Epithelial';protected &= ((np.asarray(ac.mean(axis=1)).ravel()>=1)&(np.asarray((ac>0).sum(axis=1)).ravel()>=3))|((np.asarray(en.mean(axis=1)).ravel()>=1)&(np.asarray((en>0).sum(axis=1)).ravel()>=3))
    best=S.argmax(axis=1);ownweak=(N[ix,oldidx]<=1)&(S[ix,oldidx]<.3)
    change=(best!=oldidx)&strong[ix,best]&ownweak&~protected
    new[change]=np.asarray(groups)[best[change]];reason[change]='reassigned: >=3 target-specific markers, mean log-expression >=1; <=1 original-lineage marker and original mean <0.3'
    # Detect convincing incompatible dual identity, not a single marker, HLA or
    # VIM expression. Do not force such cells into either valid lineage.
    rank=np.argsort(-S,axis=1);first,second=rank[:,0],rank[:,1]
    dual=strong[ix,first]&strong[ix,second]&(S[ix,first]<3*S[ix,second])
    # For original acinar/endocrine, dual signal still requires ductal plus a
    # specific non-epi lineage or two specific non-epi programs. Enzymes alone
    # never contribute to the dual-lineage test.
    new[dual]='Unresolved_mixed';reason[dual]='unresolved mixed: >=3 markers and mean log-expression >=1 for two incompatible lineages; strongest/second ratio <3'
    o['broad_lineage_before_coexpression_review']=original;o['broad_lineage']=new;o['lineage_review_decision']=reason
    evidence=pd.DataFrame({'original_lineage':original,'reviewed_lineage':new,'decision':reason},index=o.index)
    for j,g in enumerate(groups):evidence[g+'_mean']=S[:,j];evidence[g+'_positive_markers']=N[:,j]
    evidence['epithelial_acinar_endocrine_protected']=protected
    evidence.to_csv(B/'tables/all_cell_lineage_coexpression_evidence.csv.gz')
    pd.crosstab(evidence.original_lineage,evidence.reviewed_lineage).to_csv(B/'tables/lineage_review_transition_counts.csv')
    pd.DataFrame({'lineage':new,'decision':reason,'series':o.series.to_numpy(),'status':o.status.to_numpy()}).groupby(['lineage','decision','series','status']).size().rename('n_cells').reset_index().to_csv(B/'tables/lineage_review_decision_summary.csv',index=False)
    js({'panels':PANELS,'rule':'target reassignment: >=3 expressed markers and mean normalized-log >=1; old identity <=1 marker and mean <0.3; original acinar/endocrine protected; mixed: two strong incompatible programs and score ratio <3','HLA_VIM_digestive_not_reassignment_markers':True,'all_cells_remain_in_full_atlas':True,'unresolved_mixed_not_in_primary_epi_or_NMF':True,'counts':pd.Series(new).value_counts().to_dict(),'decision_counts':pd.Series(reason).value_counts().to_dict(),'defined_before_reference_winners_and_CM_associations':True},'tables/lineage_coexpression_review_contract.json')
    return o

def fit(x,obs,var,name):
    if (B/f'objects/{name}_integrated_hvg.h5ad').exists():return
    start=time.time();print('REFIT',name,x.shape,flush=True)
    a=ad.AnnData(x,obs=obs.copy(),var=var.copy());a.uns['log1p']={'base':None}
    sc.pp.highly_variable_genes(a,n_top_genes=3300,flavor='seurat',batch_key='series')
    idx=np.flatnonzero(a.var.highly_variable.to_numpy()&~a.var_names.str.match(r'^(MT-|RPL|RPS)'));idx=idx[np.argsort(-np.nan_to_num(a.var.dispersions_norm.to_numpy()[idx],nan=-999))[:3000]]
    a.var['selected_HVG']=False;a.var.iloc[idx,a.var.columns.get_loc('selected_HVG')]=True;a.var.to_csv(B/f'tables/{name}_HVGs.csv')
    w=a[:,idx].copy();sc.pp.scale(w,max_value=10);w.var.to_csv(B/f'tables/{name}_scale_parameters.csv');sc.tl.pca(w,n_comps=40,svd_solver='randomized',random_state=SEED)
    tmp=ad.AnnData(sparse.csr_matrix((len(obs),0)),obs=w.obs[['sample']].copy());tmp.obsm['X_pca']=w.obsm['X_pca'];tmp.write_h5ad(B/f'objects/{name}_pca_for_harmony.h5ad')
    subprocess.run([str(C/'uv_envs/rapids/.venv/bin/python'),str(B/'scripts/gpu_fit_graph.py'),str(B/f'objects/{name}_pca_for_harmony.h5ad'),str(B/f'objects/{name}_gpu_graph.h5ad'),name],check=True)
    gp=ad.read_h5ad(B/f'objects/{name}_gpu_graph.h5ad')
    for key in gp.obsm:w.obsm[key]=gp.obsm[key]
    w.obsp=gp.obsp.copy();w.uns.update(gp.uns)
    for col in gp.obs:
        if col!='sample':w.obs[col]=gp.obs[col].to_numpy()
    w.obs['cluster']=w.obs.cluster.astype('category');w.X=x[:,idx].copy()
    w.uns['source_expression_h5ad']=str(C/'h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_normalized_log_all_genes.h5ad');w.uns['expression_role']='native normalized_total1e4_log1p, no additional normalization; all genes indexed by saved source cell IDs'
    w.write_h5ad(B/f'objects/{name}_integrated_hvg.h5ad',compression='gzip');w.obs.to_csv(B/f'tables/{name}_cell_labels.csv.gz')
    dd=markers(x,w.obs,var,'cluster',name);mp=profile(x,var.index);mp.index=w.obs_names;mp.groupby(w.obs.cluster,observed=True).mean().to_csv(B/f'tables/{name}_multimarker_cluster_profiles.csv')
    out=[]
    for cl,ii in w.obs.groupby('cluster',observed=True).indices.items():
        oo=w.obs.iloc[ii];out.append({'cluster':cl,'n_cells':len(ii),'n_samples':oo['sample'].nunique(),'n_donors':oo.donor.nunique(),'n_studies':oo.series.nunique(),'tumor_fraction':oo.status.eq('tumor').mean(),'max_donor_fraction':oo.donor.value_counts(normalize=True).max(),'median_n_genes':oo.n_genes_by_counts.median(),'median_mt':oo.pct_counts_MT.median(),'top15_markers':';'.join(dd[dd.group==cl].names.head(15))})
    pd.DataFrame(out).to_csv(B/f'tables/{name}_cluster_summary.csv',index=False)
    w.obs.groupby(['cluster','donor','series','sample','status'],observed=True).size().rename('n_cells').reset_index().to_csv(B/f'tables/{name}_cluster_sample_counts.csv',index=False)
    if name=='epithelial':
        stab=[]
        for key in ['leiden_r0.4','leiden_r0.8']:
            pd.crosstab(w.obs.cluster,w.obs[key]).to_csv(B/f'tables/epithelial_main_vs_{key}.csv');stab.append({'comparison':key,'ARI':adjusted_rand_score(w.obs.cluster,w.obs[key]),'main_n_clusters':w.obs.cluster.nunique(),'sensitivity_n_clusters':w.obs[key].nunique()})
        pd.DataFrame(stab).to_csv(B/'tables/epithelial_resolution_stability.csv',index=False)
    js({'n_cells':len(obs),'n_hvg':len(idx),'n_pcs':40,'n_neighbors':30,'harmony_batch':'sample','no_status_regression':True,'backend':'CPU HVG/scaled PCA, RAPIDS harmony2 + neighbors + Leiden + UMAP','max_iter_harmony':50,'elapsed_sec':time.time()-start,'after_lineage_coexpression_review':True},f'tables/{name}_integration.json')
    print('REFIT DONE',name,time.time()-start,flush=True);del a,w,tmp,gp;gc.collect()

def main():
    marker=B/'tables/lineage_coexpression_review_contract.json'
    # First-pass objects are retained, never confused with reviewed results.
    if not marker.exists():
        for folder in ['objects','tables']:
            dst=B/folder/'pre_coexpression_review';dst.mkdir(exist_ok=True)
            for name in ['epithelial','tnk','myeloid','b','stromal','endothelial','mast']:
                for p in list((B/folder).glob(name+'_*')):
                    if p.is_file():shutil.move(str(p),str(dst/p.name))
    x,obs,var=load_expr(C/'h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_normalized_log_all_genes.h5ad')
    o=pd.read_csv(B/'tables/full_atlas_cell_labels.csv.gz',index_col=0,low_memory=False).reindex(obs.index)
    if marker.exists():
        review_saved=pd.read_csv(B/'tables/reviewed_full_atlas_cell_labels.csv.gz',index_col=0,low_memory=False);o=review_saved.reindex(obs.index)
    else:
        o=review(x,o,var);o.to_csv(B/'tables/reviewed_full_atlas_cell_labels.csv.gz')
    # Update the all-cell label table while keeping archived broad identity.
    o.to_csv(B/'tables/full_atlas_cell_labels.csv.gz')
    for lineage in ['Epithelial','TNK','Myeloid','B','Stromal','Endothelial','Mast']:
        name='epithelial' if lineage=='Epithelial' else lineage.lower();take=o.broad_lineage.eq(lineage).to_numpy();fit(x[take],o.loc[take].copy(),var,name)
    un=o[o.broad_lineage.eq('Unresolved_mixed')].copy();un['cluster']='unresolved_mixed_20261107';un['new_subtype']=un.cluster;un['marker_annotation']='incompatible dual-lineage coexpression; excluded from primary EPI/NMF';un.to_csv(B/'tables/unresolved_mixed_cell_labels.csv.gz')

if __name__=='__main__':main()
