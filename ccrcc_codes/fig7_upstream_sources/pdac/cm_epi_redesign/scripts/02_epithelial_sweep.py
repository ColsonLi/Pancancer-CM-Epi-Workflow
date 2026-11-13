#!/usr/bin/env python3
from common import *
import anndata as ad
import scanpy as sc
import harmonypy as hm
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import gc, warnings

def main():
    x,obs,var=read_raw(CORE/'h5ad/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/epithelial/adata_epi.h5ad')
    for c in ['sample','series','status','patient_id','cell_subtype']:obs[c]=obs[c].astype(str)
    rawscore=means(x,var.index);rawscore.index=obs.index
    obs=obs.join(rawscore.add_prefix('raw_'))
    # Conservative identity filter: old trusted ductal, tumor source, acinar low,
    # and ductal signal stronger than aggregate immune signal. No CNV claim.
    identity=(obs.raw_Acinar<obs.raw_Ductal)&(obs.raw_Ductal>obs.raw_Immune)
    primary=obs.status.eq('tumor')&obs.cell_subtype.isin(STRICT_DUCTAL)&identity
    extended=obs.status.eq('tumor')&obs.cell_subtype.isin(STRICT_DUCTAL+['Epi_ANXA4','Epi_FXYD2'])&identity
    extended &= obs.cell_subtype.isin(STRICT_DUCTAL)|(obs.raw_MalignantDuctal>obs.raw_NormalDuctal)
    flow=obs[['sample','series','status','patient_id','cell_subtype']].copy()
    flow['primary_ductal']=primary;flow['extended_ductal']=extended;flow['identity_pass']=identity
    flow.groupby(['sample','series','status','patient_id','cell_subtype']).agg(n_cells=('primary_ductal','size'),n_primary=('primary_ductal','sum'),n_extended=('extended_ductal','sum')).reset_index().to_csv(BASE/'tables/epithelial_cleanup_sample_subtype.csv',index=False)
    obs['retained_primary']=primary;obs['retained_extended']=extended
    obs.to_csv(BASE/'tables/epithelial_cleanup_cells.csv.gz',index_label='cell')
    # Score gene-wise standardized normalized-log values; fit means/SD in primary
    # tumor ductal only, then apply identically to extended cells.
    fullgenes=var.index.astype(str);prog_genes=sorted(set(sum(PANELS.values(),[]))&set(fullgenes))
    gx=x[:,fullgenes.get_indexer(prog_genes)].toarray();mu=gx[primary].mean(axis=0);sd=gx[primary].std(axis=0);sd=np.maximum(sd,.1)
    scores=pd.DataFrame({p:((gx[:,pd.Index(prog_genes).get_indexer([g for g in gs if g in prog_genes])]-mu[pd.Index(prog_genes).get_indexer([g for g in gs if g in prog_genes])])/sd[pd.Index(prog_genes).get_indexer([g for g in gs if g in prog_genes])]).mean(axis=1) for p,gs in PANELS.items()},index=obs.index)
    pd.DataFrame({'gene':prog_genes,'mean':mu,'sd':sd}).to_csv(BASE/'tables/program_gene_standardization.csv',index=False)
    scores.to_csv(BASE/'tables/epithelial_program_scores_all.csv.gz',index_label='cell')
    ep=ad.AnnData(x[primary.to_numpy()].copy(),obs=obs.loc[primary].copy(),var=var.copy());ep.uns['log1p']={'base':None}
    score=scores.loc[ep.obs_names]; axis=(score.JUN-score.VIM).to_numpy()
    labels=pd.DataFrame(index=ep.obs_names);manifest=[];clusters=[]
    refs=[];gene_sigs=[];sample_rows=[]

    def record(scheme, state, family, details, cells=None, expr=None, scs=None):
        oo=ep.obs if cells is None else cells
        xx=ep.X if expr is None else expr
        ss=score if scs is None else scs
        state=np.asarray(state,dtype=str)
        if cells is None: labels[scheme]=state
        else: pd.DataFrame({'state':state},index=oo.index).to_csv(BASE/f'tables/{scheme}_extended_cell_labels.csv.gz',index_label='cell')
        manifest.append({'epi_scheme':scheme,'family':family,'n_cells':len(oo),'n_JUN':int((state=='JUN').sum()),'n_VIM':int((state=='VIM').sum()),'details':details})
        for sm,ix in oo.groupby('sample',observed=True).indices.items():
            n=len(ix);nj=int((state[ix]=='JUN').sum());nv=int((state[ix]=='VIM').sum())
            sample_rows.append({'epi_scheme':scheme,'sample':sm,'n_epi':n,'JUN_n':nj,'VIM_n':nv,'JUN_frac':nj/n,'VIM_frac':nv/n,'balance':np.log((nj+.5)/(nv+.5)),'continuous_balance':float((ss.iloc[ix].JUN-ss.iloc[ix].VIM).mean())})
        # Pseudobulk signature contrast estimated with samples as equal units,
        # not cells. Export every gene effect; no survival-based signature tuning.
        means_by=[]
        for st in ['JUN','VIM','Other']:
            mask=state==st
            if mask.sum()==0:continue
            arr=np.asarray(xx[mask].mean(axis=0)).ravel()
            refs.append((scheme+'__'+st,arr,int(mask.sum())))
        # Equal sample JUN-vs-VIM contrast where both >=10 cells.
        for sm,ix in oo.groupby('sample',observed=True).indices.items():
            j=ix[state[ix]=='JUN'];v=ix[state[ix]=='VIM']
            if len(j)>=10 and len(v)>=10:
                means_by.append(np.asarray(xx[j].mean(axis=0)-xx[v].mean(axis=0)).ravel())
        if len(means_by)>=3:
            d=np.vstack(means_by); effect=d.mean(axis=0); t,p=stats.ttest_1samp(d,0,axis=0);q=fdr(p)
            eligible=~fullgenes.isin(DIGESTIVE)&~fullgenes.str.match(r'^(MT-|RPL|RPS)')
            # Core-panels plus top 30 sample-balanced contrast markers per sign.
            for direction,panel in [(1,'JUN'),(-1,'VIM')]:
                rank=np.argsort(-effect*direction)
                picks=[i for i in rank if eligible[i] and direction*effect[i]>.1][:30]
                for i in picks:gene_sigs.append({'epi_scheme':scheme,'state':panel,'gene':fullgenes[i],'effect_JUN_minus_VIM':effect[i],'p':p[i],'q':q[i],'n_paired_samples':len(d)})
        print(scheme,manifest[-1]['n_JUN'],manifest[-1]['n_VIM'],flush=True)

    # B: explicit archetypes, ordered axis extremes, K-means on five programs,
    # and non-negative matrix factorization of program activities.
    names=['JUN','VIM','Hypoxia','Classical','Acinar']
    P=score[names].to_numpy();arg=np.asarray(names)[np.argmax(P,axis=1)]
    record('B_program_argmax',np.where(np.isin(arg,['JUN','VIM']),arg,'Other'),'B','argmax of five fixed gene-z programs')
    for quant in [1/3,.2]:
        lo,hi=np.quantile(axis,[quant,1-quant]);s=np.where(axis>=hi,'JUN',np.where(axis<=lo,'VIM','Other'))
        record(f'B_axis_q{int(quant*100)}',s,'B',f'fixed program JUN-minus-VIM extremes; quantile={quant}')
    for k in [4,6,8]:
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(P)
        cent=np.array([axis[km.labels_==j].mean() for j in range(k)])
        st=np.where(km.labels_==cent.argmax(),'JUN',np.where(km.labels_==cent.argmin(),'VIM','Other'))
        record(f'B_archetype_k{k}',st,'B',f'KMeans of five fixed gene-z scores, k={k}; extremes oriented only by JUN-VIM program')
    # NMF on nonnegative raw program gene means, no shift-induced pseudo-data.
    R=obs.loc[primary,['raw_'+n for n in names]].to_numpy();R/=np.maximum(R.std(axis=0),.1)
    for k in [4,5]:
        nm=NMF(n_components=k,init='nndsvda',random_state=42,max_iter=1000)
        with warnings.catch_warnings():warnings.simplefilter('ignore');W=nm.fit_transform(R)
        H=nm.components_;align=H[:,0]/np.maximum(H.sum(axis=1),1e-9)-H[:,1]/np.maximum(H.sum(axis=1),1e-9)
        comp=W.argmax(axis=1);st=np.where(comp==align.argmax(),'JUN',np.where(comp==align.argmin(),'VIM','Other'))
        pd.DataFrame(H,columns=names).to_csv(BASE/f'tables/B_programNMF_K{k}_H.csv',index_label='component')
        record(f'B_programNMF_K{k}',st,'B','NMF on five nonnegative program means; component alignment by fixed programs only')
    # Extended tumor ductal sensitivity; requires malignant-like > normal-ductal
    # only for old ANXA4/FXYD2, never admits normal-source cells.
    es=scores.loc[extended];ea=(es.JUN-es.VIM).to_numpy();lo,hi=np.quantile(axis,[1/3,2/3])
    record('B_extended_axis_q33',np.where(ea>=hi,'JUN',np.where(ea<=lo,'VIM','Other')),'B','extended tumor ductal; same fixed primary thresholds',obs.loc[extended],x[extended.to_numpy()],es)
    # A: digestive/HLA/lineage ambient exclusions from HVGs. No target module is
    # used for HVG selection or cluster annotation.
    exclude=DIGESTIVE|set(PANELS['Immune'])|set(PANELS['CM1'])
    for n_hvg in [1500,3000]:
        sc.pp.highly_variable_genes(ep,n_top_genes=n_hvg+len(exclude),flavor='seurat',batch_key='series')
        use=ep.var.highly_variable.to_numpy()&~ep.var_names.isin(exclude)&~ep.var_names.str.match(r'^(MT-|RPL|RPS)')
        # Exactly n_hvg highest normalized-dispersion genes among eligible HVGs.
        idx=np.where(use)[0];idx=idx[np.argsort(-np.nan_to_num(ep.var.dispersions_norm.to_numpy()[idx],nan=-999))[:n_hvg]]
        work=ep[:,idx].copy();sc.pp.scale(work,max_value=10);sc.tl.pca(work,n_comps=40,svd_solver='arpack',random_state=42)
        pd.DataFrame({'gene':work.var_names}).to_csv(BASE/f'tables/A_HVG{n_hvg}_genes.csv',index=False)
        for batch in ['series','sample']:
            harm=hm.run_harmony(work.obsm['X_pca'],work.obs,batch,max_iter_harmony=30,random_state=42,verbose=False)
            Z=np.asarray(harm.Z_corr)
            if Z.shape[0]!=work.n_obs:Z=Z.T
            work.obsm['X_harmony']=Z.astype(np.float32)
            sc.pp.neighbors(work,n_neighbors=20,n_pcs=30,use_rep='X_harmony',random_state=42)
            for resolution in [.2,.4,.8,1.2]:
                sc.tl.leiden(work,resolution=resolution,key_added='cluster',flavor='igraph',n_iterations=2,directed=False,random_state=42)
                cl=work.obs.cluster.astype(str).to_numpy();summ=[]
                for j in np.unique(cl):
                    mask=cl==j;part=score.loc[mask];summ.append({'cluster':j,'n':int(mask.sum()),'n_samples':work.obs.loc[mask,'sample'].nunique(),'axis':float(axis[mask].mean()),**{f'program_{p}':float(part[p].mean()) for p in names}})
                tab=pd.DataFrame(summ);eligible=tab[(tab.n>=100)&(tab.n_samples>=3)].sort_values('axis');nend=max(1,int(np.ceil(len(eligible)*.25)))
                jset=eligible.tail(nend).cluster.tolist();vset=eligible.head(nend).cluster.tolist();s=np.where(np.isin(cl,jset),'JUN',np.where(np.isin(cl,vset),'VIM','Other'))
                scheme=f'A_HVG{n_hvg}_{batch}_r{resolution}'
                tab['epi_scheme']=scheme;tab['state']=np.where(tab.cluster.isin(jset),'JUN',np.where(tab.cluster.isin(vset),'VIM','Other'));clusters.append(tab)
                record(scheme,s,'A',f'HVG={n_hvg}; Harmony={batch}; Leiden={resolution}; upper/lower 25% eligible cluster-centroid JUN-VIM axis')
                if n_hvg==3000 and batch=='series' and resolution==.4:
                    ep.obsm['X_pca']=work.obsm['X_pca'];ep.obsm['X_harmony']=Z
                    ep.obs['A_HVG3000_series_r0.4_cluster']=cl
            del harm,Z
        del work;gc.collect()
    labels.to_csv(BASE/'tables/epithelial_scheme_cell_labels.csv.gz',index_label='cell')
    pd.DataFrame(manifest).to_csv(BASE/'tables/epithelial_scheme_manifest.csv',index=False)
    pd.DataFrame(sample_rows).to_csv(BASE/'tables/epithelial_scheme_sample_metrics.csv',index=False)
    pd.concat(clusters,ignore_index=True).to_csv(BASE/'tables/A_cluster_program_annotations.csv',index=False)
    pd.DataFrame(gene_sigs).to_csv(BASE/'tables/epithelial_discovered_signatures.csv',index=False)
    # All schemes' state references in one small normalized-log object.
    rf=ad.AnnData(np.vstack([r[1] for r in refs]),obs=pd.DataFrame({'n_cells':[r[2] for r in refs]},index=[r[0] for r in refs]),var=pd.DataFrame(index=fullgenes))
    rf.write_h5ad(BASE/'objects/epithelial_scheme_references.h5ad',compression='gzip')
    ep.obs=ep.obs.join(score.add_prefix('program_'));ep.write_h5ad(BASE/'objects/cleaned_tumor_ductal.h5ad',compression='gzip')
    save_json({'n_original_epi':len(obs),'n_primary':int(primary.sum()),'n_extended':int(extended.sum()),'n_primary_samples':ep.obs['sample'].nunique(),'n_schemes':len(manifest),'strict_old_labels':STRICT_DUCTAL,'malignancy':'tumor-associated ductal; no CNV confirmation; normal-source/acinar/normal-ductal old labels excluded from primary','minimum_sample_epi_primary':50,'minimum_sample_epi_sensitivity':25},'tables/epithelial_summary.json')

if __name__=='__main__':main()
