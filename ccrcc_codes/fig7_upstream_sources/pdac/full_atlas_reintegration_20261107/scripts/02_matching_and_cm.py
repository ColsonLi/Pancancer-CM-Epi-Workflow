from common import *
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import nnls,linear_sum_assignment
import re

def enrich_metadata(o):
    u=pd.read_csv(V/'external_atlas_for_cm-epi_validation/PDAC/UCSC/metadata_samples.csv').set_index('Sample_ID')
    o['study']=o.series.astype(str)
    use=o.series.eq('UCSC');o.loc[use,'study']=o.loc[use,'sample'].map(u.Publication).fillna('UCSC_unknown')
    o['donor']=o.study.astype(str)+'::'+o.patient_id.astype(str)
    o['donor_key_basis']=np.where(o.patient_id.astype(str).eq(o['sample'].astype(str)),'sample-level proxy (true patient link unavailable in archived manifest)','provided patient identifier')
    return o

def annotate():
    o=enrich_metadata(pd.read_csv(B/'tables/full_atlas_cell_labels.csv.gz',index_col=0))
    manifest=pd.read_csv(B/'tables/original_included_manifest.csv');sm=manifest[manifest.modality.eq('single_cell')].copy()
    sm['atlas_sample']=np.where(sm.dataset.eq('UCSC'),sm.sample_id,sm.gsm)
    present=set(o['sample']);sm['present_in_rebuilt_atlas']=sm.atlas_sample.isin(present);sm.to_csv(B/'tables/manifest_to_atlas_sample_crosswalk.csv',index=False)
    con=json.loads((B/'tables/full_atlas_contract.json').read_text());con['manifest_missing']=sm.loc[~sm.present_in_rebuilt_atlas,'atlas_sample'].tolist();con['manifest_singlecell_samples']=len(sm);con['n_donors']=o.donor.nunique();con['n_underlying_studies']=o.study.nunique();js(con,'tables/full_atlas_contract.json')
    con['lineage_counts']=o.broad_lineage.value_counts().to_dict();con['unresolved_mixed_cells']=int(o.broad_lineage.eq('Unresolved_mixed').sum());con['lineage_annotation']='archived multi-marker annotation plus conservative fixed per-cell multi-marker coexpression reassignment/mixed resolution before reference matching';js(con,'tables/full_atlas_contract.json')
    allnew=[];anns=[]
    subtype_panels={'tnk':['T_naive','T_reg','T_cytotoxic','NK','T_exhausted','IFN','Cycling'],
      'myeloid':['Mono','Macrophage','SPP1_Mac','cDC','pDC','Neutrophil','IFN','Cycling'],
      'b':['B','Plasma','IFN','Cycling'],'stromal':['Stromal','Pericyte','CAF','Fibroblast','Cycling'],
      'endothelial':['Endothelial','Lymphatic','Venous_Endo','Capillary_Endo','Cycling'],'mast':['Mast','IFN','Cycling'],
      'epithelial':['Ductal','Acinar','Endocrine','NormalDuctal','TumorDuctal','Cycling']}
    for name in ['epithelial','tnk','myeloid','b','stromal','endothelial','mast']:
        oo=enrich_metadata(pd.read_csv(B/f'tables/{name}_cell_labels.csv.gz',index_col=0))
        mp=pd.read_csv(B/f'tables/{name}_multimarker_cluster_profiles.csv',index_col=0)
        summary=pd.read_csv(B/f'tables/{name}_cluster_summary.csv').set_index('cluster')
        # Multi-program labels are descriptive, never a deletion rule. All
        # digestive-positive non-epi cells remain, and admixture is explicit.
        for cl,row in mp.iterrows():
            ss=row[subtype_panels[name]].sort_values(ascending=False)
            desc='/'.join(ss.index[:2]);core={'tnk':'TNK','myeloid':'Myeloid','b':'B','stromal':'Stromal','endothelial':'Endothelial','mast':'Mast'}.get(name,'Ductal')
            cc=oo[oo.cluster.eq(cl)]
            rr=summary.loc[cl].to_dict();rr.update(cluster=cl,lineage=name,n_donors=cc.donor.nunique(),n_studies=cc.study.nunique(),marker_annotation=desc,acinar_warning=bool(row.Acinar>row[core]),ductal_warning=bool(name!='epithelial' and row.Ductal>row[core]),retained=True,exclusion_reason='none: digestive markers alone are insufficient for exclusion')
            anns.append(rr)
            oo.loc[oo.cluster.eq(cl),'marker_annotation']=desc
        oo.to_csv(B/f'tables/{name}_cell_labels.csv.gz')
        oo['new_subtype']=oo.cluster
        allnew.append(oo)
    unresolved=enrich_metadata(pd.read_csv(B/'tables/unresolved_mixed_cell_labels.csv.gz',index_col=0,low_memory=False))
    allnew.append(unresolved)
    no=pd.concat(allnew).reindex(o.index)
    for c in ['cluster','new_subtype','marker_annotation']:o[c]=no[c]
    for c in no.columns:
        if c.startswith('leiden_r'):o[c]=no[c]
    assert o.cluster.notna().all()
    o.to_csv(B/'tables/full_atlas_new_labels.csv.gz')
    full=ad.read_h5ad(B/'objects/full_atlas_verified_embeddings.h5ad');full.obs=o.reindex(full.obs_names).copy();full.uns['primary_labels']='reviewed broad lineage, resolved expression reclusters; incompatible dual-lineage cells retained as Unresolved_mixed; full_atlas_new_labels table authoritative';full.write_h5ad(B/'objects/full_atlas_verified_embeddings.h5ad',compression='gzip');del full;gc.collect()
    pd.DataFrame(anns).to_csv(B/'tables/new_subtype_annotations.csv',index=False)
    o.groupby(['study','series','sample','patient_id','donor','status'],observed=True).size().rename('n_cells').reset_index().to_csv(B/'tables/included_sample_study_patient_counts.csv',index=False)
    o.groupby(['broad_lineage','new_subtype','donor','study','sample','status'],observed=True).size().rename('n_cells').reset_index().to_csv(B/'tables/subtype_donor_sample_support.csv',index=False)
    return o

def reference_matching(o):
    eo=o[o.broad_lineage.eq('Epithelial')].copy()
    source=C/'h5ad/01-celltype_integration_clustering/04-integration-harmony/adata_normalized_log_all_genes.h5ad'
    # Backed sparse column/row selection avoids materializing full expression.
    a=ad.read_h5ad(source,backed='r');scaling=pd.read_csv(B/'tables/frozen_ccRCC_gene_scaling.csv',index_col=0)
    genes=scaling.index.intersection(a.var_names,sort=False);pos=a.obs_names.get_indexer(eo.index)
    coverage=pd.read_csv(B/'tables/frozen_ccRCC_top100_per_state_DEGs.csv');coverage['present_in_PDAC_common_gene_matrix']=coverage.gene.isin(genes);coverage.to_csv(B/'tables/ccRCC_reference_gene_coverage_in_PDAC.csv',index=False)
    x=sparse.csr_matrix(a.X[pos,:])[:,a.var_names.get_indexer(genes)]
    a.file.close()
    dp,support=donor_profiles(x,eo,'cluster',genes);dp.to_csv(B/'tables/epithelial_reference_feature_donor_profiles.csv')
    support.to_csv(B/'tables/epithelial_reference_feature_donor_support.csv',index=False)
    cp=dp.groupby(level='cluster').mean();ref=pd.read_csv(B/'tables/frozen_ccRCC_donor_balanced_centroids.csv',index_col=0)[genes]
    mu=scaling.loc[genes,'mean'];sd=scaling.loc[genes,'sd']
    Z=cp.sub(mu).div(sd);R=ref.sub(mu).div(sd)
    sims=cosine_similarity(Z,R.loc[['Epi_JUN','Epi_VIM']]);out=pd.DataFrame(sims,index=cp.index,columns=['cosine_JUN','cosine_VIM'])
    for state in ['JUN','VIM']:
        out[f'rank_{state}']=out[f'cosine_{state}'].rank(ascending=False,method='min').astype(int)
        out[f'spearman_{state}']=[stats.spearmanr(z,R.loc['Epi_'+state]).statistic for z in Z.to_numpy()]
    out['JUN_minus_VIM_descriptive']=out.cosine_JUN-out.cosine_VIM
    out['n_profile_donors']=support.groupby('cluster').donor.nunique()
    out['single_profile_donor_warning']=out.n_profile_donors.lt(2)
    ann=pd.read_csv(B/'tables/new_subtype_annotations.csv').set_index('cluster');out=out.join(ann)
    chosen={state:out[f'cosine_{state}'].idxmax() for state in ['JUN','VIM']}
    # 300 donor-resampled PDAC means; ref coordinate system remains frozen.
    rng=np.random.default_rng(SEED);win={s:{c:0 for c in cp.index} for s in ['JUN','VIM']};bvals=[]
    for b in range(300):
        zs=[]
        for cl in cp.index:
            pp=dp.loc[cl].to_numpy();ii=rng.integers(0,len(pp),len(pp));zs.append((pp[ii].mean(axis=0)-mu.to_numpy())/sd.to_numpy())
        bs=cosine_similarity(np.array(zs),R.loc[['Epi_JUN','Epi_VIM']])
        for j,s in enumerate(['JUN','VIM']):
            win[s][cp.index[bs[:,j].argmax()]]+=1
            for i,c in enumerate(cp.index):bvals.append({'bootstrap':b,'cluster':c,'state':s,'similarity':bs[i,j]})
    bd=pd.DataFrame(bvals)
    for s in ['JUN','VIM']:
        out[f'donor_bootstrap_winner_fraction_{s}']=[win[s][c]/300 for c in out.index]
        ci=bd[bd.state.eq(s)].groupby('cluster').similarity.quantile([.025,.975]).unstack()
        out[f'cosine_{s}_CI_low']=ci[.025];out[f'cosine_{s}_CI_high']=ci[.975]
        out.loc[out.n_profile_donors.lt(2),[f'cosine_{s}_CI_low',f'cosine_{s}_CI_high']]=np.nan
    lso=[]
    for study in eo.study.unique():
        sub=eo[eo.study.ne(study)];sdp,_=donor_profiles(x[eo.study.ne(study).to_numpy()],sub,'cluster',genes)
        c=sdp.groupby(level=0).mean();z=c.sub(mu).div(sd);ss=cosine_similarity(z,R.loc[['Epi_JUN','Epi_VIM']])
        for j,state in enumerate(['JUN','VIM']):lso.append({'omitted_study':study,'state':state,'winner':c.index[ss[:,j].argmax()],'similarity':ss[:,j].max()})
    pd.DataFrame(lso).to_csv(B/'tables/epithelial_leave_one_study_out_matches.csv',index=False)
    sensitivity=[]
    for key in ['leiden_r0.4','leiden_r0.8']:
        sdprof,_=donor_profiles(x,eo,key,genes);cc=sdprof.groupby(level=0).mean();zz=cc.sub(mu).div(sd);sims=cosine_similarity(zz,R.loc[['Epi_JUN','Epi_VIM']])
        for j,state in enumerate(['JUN','VIM']):
            winner=cc.index[sims[:,j].argmax()];maincells=eo.cluster.eq(chosen[state]);senscells=eo[key].eq(winner);jaccard=(maincells&senscells).sum()/(maincells|senscells).sum()
            for i,cl in enumerate(cc.index):sensitivity.append({'partition':key,'state':state,'cluster':cl,'cosine':sims[i,j],'is_independent_argmax':cl==winner,'winning_cluster_main_winner_jaccard':jaccard if cl==winner else np.nan})
    pd.DataFrame(sensitivity).to_csv(B/'tables/epithelial_resolution_reference_matching_sensitivity.csv',index=False)
    out.to_csv(B/'tables/all_epithelial_reference_similarities.csv')
    bd.to_csv(B/'tables/epithelial_reference_similarity_donor_bootstrap.csv.gz',index=False)
    chosen.update(n_reference_features=len(scaling),n_covered=len(genes),coverage=len(genes)/len(scaling),same_winner=chosen['JUN']==chosen['VIM'],metric='uniform fixed-reference gene-z cosine, independent argmax; not JUN-minus-VIM extrema',bootstrap='300 donor resamples; conditional on fixed ccRCC reference and clustering')
    chosen['winner_QC']={s:{'n_cells':int(out.loc[chosen[s],'n_cells']),'n_profile_donors':int(out.loc[chosen[s],'n_profile_donors']),'median_mt':float(out.loc[chosen[s],'median_mt']),'max_donor_fraction':float(out.loc[chosen[s],'max_donor_fraction']),'warning':'numerical nearest only; not a validated conserved malignant state; inspect QC, coverage, donor and resolution stability'} for s in ['JUN','VIM']}
    js(chosen,'tables/selected_single_epithelial_clusters.json');print('MATCHES',chosen,flush=True)

def fit_nmf(VV,weights,k,seed=SEED,maxiter=5000):
    mod=NMF(n_components=k,init='nndsvda' if seed==SEED else 'random',random_state=seed,max_iter=maxiter,tol=1e-5)
    with warnings.catch_warnings():warnings.simplefilter('ignore');w=mod.fit_transform(VV*np.sqrt(weights)[:,None])
    h=mod.components_; scale=np.maximum(np.linalg.norm(h,axis=1),1e-12);h=h/scale[:,None]
    return h,float(mod.reconstruction_err_**2),mod.n_iter_

def cm_discovery(o):
    non=o[~o.broad_lineage.isin(['Epithelial','Unresolved_mixed'])].copy();counts=pd.crosstab(non['sample'],non.new_subtype)
    counts=counts.reindex(o['sample'].unique(),fill_value=0)
    meta=o[['sample','study','series','status','donor','patient_id','donor_key_basis']].drop_duplicates().set_index('sample').reindex(counts.index)
    meta['n_non_epi']=counts.sum(axis=1);meta['fit_eligible']=meta.n_non_epi.ge(50)
    freq=counts.div(counts.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
    counts.to_csv(B/'tables/non_epithelial_sample_subtype_counts.csv');freq.to_csv(B/'tables/non_epithelial_sample_subtype_frequencies.csv');meta.to_csv(B/'tables/cm_sample_eligibility.csv')
    f=freq.loc[meta.fit_eligible];mm=meta.loc[f.index]
    lo=f.min();scale=(f.max()-lo).replace(0,1);v=((f-lo)/scale).clip(lower=0).to_numpy();va=((freq-lo)/scale).clip(lower=0).to_numpy()
    pd.DataFrame({'minimum':lo,'range':scale}).to_csv(B/'tables/cm_minmax_scaling.csv')
    weights=mm.status.map({s:.5/mm.status.eq(s).sum() for s in mm.status.unique()}).to_numpy();weights/=weights.mean()
    rows=[];seedfits=[];Hs={};fits={}
    for k in range(2,min(20,len(f)-2,f.shape[1])+1):
        hs=[]
        for seed in [SEED,17,93]:
            h,err,ni=fit_nmf(v,weights,k,seed);hs.append((h,err,ni,seed));seedfits.append({'K':k,'seed':seed,'weighted_SSE':err,'n_iter':ni,'converged':ni<5000,'max_iter':5000,'tolerance':1e-5})
        h,err,ni,seed=min(hs,key=lambda z:z[1]);sim=[]
        for hh,_,_,_ in hs:
            s=cosine_similarity(h,hh);a,b=linear_sum_assignment(-s);sim.append(s[a,b].mean())
        Hs[k]=h;fits[k]=hs
        pd.DataFrame(h,columns=f.columns).to_csv(B/f'tables/nmf_screen_K{k}_H.csv',index=False)
        rows.append({'K':k,'weighted_SSE':err,'explained_uncentered':1-err/np.sum(v*v*weights[:,None]),'n_iter':ni,'converged':ni<5000,'best_seed':seed,'seed_matched_cosine':np.mean(sim)})
        print('NMF SCREEN',rows[-1],flush=True)
    pd.DataFrame(seedfits).to_csv(B/'tables/nmf_all_seed_convergence_diagnostics.csv',index=False)
    screen=pd.DataFrame(rows);xr=(screen.K-screen.K.min())/(screen.K.max()-screen.K.min());yr=(screen.weighted_SSE-screen.weighted_SSE.min())/(screen.weighted_SSE.max()-screen.weighted_SSE.min());screen['elbow_distance_below_chord']=1-xr-yr
    elbow=int(screen.loc[screen.elbow_distance_below_chord.idxmax(),'K'])
    shortlist=sorted(set([max(2,elbow-2),elbow,min(int(screen.K.max()),elbow+2)]));rng=np.random.default_rng(SEED)
    sr=[];allst=[]
    for k in shortlist:
        h=Hs[k]
        # Patient block resampling preserves paired samples; stratify by study.
        for b in range(30):
            ix=[]
            for study,g in mm.groupby('study'):
                ds=g.donor.unique()
                for donor in rng.choice(ds,len(ds),replace=True):ix.extend(np.flatnonzero(mm.donor.eq(donor).to_numpy()))
            ix=np.array(ix);ss=mm.iloc[ix].status;wt=ss.map({s:.5/ss.eq(s).sum() for s in ss.unique()}).to_numpy();wt/=wt.mean()
            hb,err,ni=fit_nmf(v[ix],wt,k,SEED+b+1,maxiter=3000)
            sim=cosine_similarity(h,hb);a,z=linear_sum_assignment(-sim);vals=sim[a,z]
            allst.extend([{'K':k,'replicate':b,'kind':'patient_bootstrap','module':int(i),'matched_cosine':float(s),'converged':ni<3000} for i,s in zip(a,vals)])
        # Actual underlying publications, not only UCSC vs GEO container.
        for study in mm.study.unique():
            ix=mm.study.ne(study).to_numpy();ss=mm.loc[ix,'status'];wt=ss.map({s:.5/ss.eq(s).sum() for s in ss.unique()}).to_numpy();wt/=wt.mean()
            hb,_,ni=fit_nmf(v[ix],wt,k,maxiter=3000);sim=cosine_similarity(h,hb);a,z=linear_sum_assignment(-sim)
            allst.extend([{'K':k,'replicate':study,'kind':'leave_study_out','module':int(i),'matched_cosine':float(s),'converged':ni<3000} for i,s in zip(a,sim[a,z])])
        z=pd.DataFrame([r for r in allst if r['K']==k]);sr.append({'K':k,'patient_bootstrap_mean':z.loc[z.kind.eq('patient_bootstrap'),'matched_cosine'].mean(),'patient_bootstrap_q10':z.loc[z.kind.eq('patient_bootstrap'),'matched_cosine'].quantile(.1),'study_leaveout_mean':z.loc[z.kind.eq('leave_study_out'),'matched_cosine'].mean(),'study_leaveout_q10':z.loc[z.kind.eq('leave_study_out'),'matched_cosine'].quantile(.1)})
        print('NMF STABILITY',sr[-1],flush=True)
    stab=pd.DataFrame(sr);stab.to_csv(B/'tables/nmf_shortlist_stability.csv',index=False);pd.DataFrame(allst).to_csv(B/'tables/nmf_module_resampling_stability.csv',index=False)
    screen=screen.merge(stab,on='K',how='left');screen.to_csv(B/'tables/nmf_rank_screen.csv',index=False)
    acceptable=stab[(stab.patient_bootstrap_mean>=.80)&(stab.study_leaveout_mean>=.75)]
    selected=int(min(acceptable.K,key=lambda k:abs(k-elbow))) if len(acceptable) else elbow
    h=Hs[selected];W=np.vstack([nnls(h.T,row,maxiter=100*selected)[0] for row in va]);F=h/h.sum(axis=1,keepdims=True)
    names=[f'PDAC_CM_20261107_{i+1:02d}' for i in range(selected)]
    pd.DataFrame(h,index=names,columns=f.columns).to_csv(B/'tables/selected_nmf_H.csv');pd.DataFrame(F,index=names,columns=f.columns).to_csv(B/'tables/selected_nmf_H_fraction.csv');pd.DataFrame(W,index=freq.index,columns=names).to_csv(B/'tables/selected_nmf_W_unweighted_NNLS.csv')
    # Biological alignment frozen before epithelial-CM association testing.
    lineage=non[['new_subtype','broad_lineage']].drop_duplicates().set_index('new_subtype').broad_lineage.reindex(f.columns)
    lineages=['B','Endothelial','Mast','Myeloid','Stromal','TNK']
    comp=pd.DataFrame({l:F[:,lineage.eq(l)].sum(axis=1) for l in lineages},index=names)
    comp.to_csv(B/'tables/selected_cm_lineage_loading_composition.csv')
    refs=pd.read_csv(V/'shared_clinical_and_signatures/reference/all_CM_lineage_loading_shares.csv');refs.lineage=refs.lineage.replace({'T_NK':'TNK','pDC':'Myeloid'});rf=refs.pivot_table(index='CM',columns='lineage',values='loading_fraction',aggfunc='sum').reindex(columns=lineages,fill_value=0)
    sim=cosine_similarity(comp,rf.loc[['joint_01_sharedCM','joint_10_tCM']]);match=pd.DataFrame(sim,index=names,columns=['CM1_lineage_cosine','CM10_lineage_cosine'])
    # No arbitrary purity bonuses, association information, or forced homology.
    chosen={'CM1':match.CM1_lineage_cosine.idxmax(),'CM10':match.CM10_lineage_cosine.idxmax()}
    mods=[]
    for i,n in enumerate(names):
        val=W[:,i];tm=meta.status.eq('tumor');nm=meta.status.eq('normal-like');r,p=stats.mannwhitneyu(val[tm],val[nm],alternative='two-sided')
        dominant=comp.loc[n].idxmax();top=f.columns[np.argsort(-F[i])[:8]].tolist()
        mods.append({'module':n,'dominant_lineage':dominant,'top_subtypes':';'.join(top),'tumor_median_W':float(np.median(val[tm])),'normal_median_W':float(np.median(val[nm])),'tumor_normal_MW_p':p,'tumor_normal_label':'tumor-enriched' if np.median(val[tm])>np.median(val[nm]) else 'normal-enriched','matched_CM1':n==chosen['CM1'],'matched_CM10':n==chosen['CM10'],**match.loc[n].to_dict()})
    pd.DataFrame(mods).to_csv(B/'tables/selected_cm_module_annotations.csv',index=False)
    match.to_csv(B/'tables/all_cm_reference_composition_similarity.csv')
    js({'selected_K':selected,'elbow_K':elbow,'shortlisted_K':shortlist,'rank_rule':'maximum normalized reconstruction-error distance below end-point chord; shortlist elbow +/-2; retain nearest elbow rank with mean patient bootstrap >=.80 and mean leave-publication-out >=.75, else report unstable elbow','scaling':'non-epithelial subtype frequency column min-max, fit-eligible sample min/range','balance':'equal total tumor/normal squared-error weight; row sqrt weight; no epithelial/outcome inputs','projection':'unweighted NNLS onto row-L2 normalized H; primary >=50 non-epi; projections of lower-cell samples saved but not tested','n_fit_samples':len(f),'n_all_samples':len(freq),'n_subtypes':f.shape[1],'n_non_epi_cells':len(non),'no_subtype_deletions':True,'selected_matches':chosen,'matching':'uniform cosine to complete ccRCC broad-lineage H-fraction composition; local TNK modules may be split; not a verified subtype-level homolog','CM1_close_candidates':match.sort_values('CM1_lineage_cosine',ascending=False).head(4).to_dict('index'),'stability':stab[stab.K.eq(selected)].to_dict('records')},'tables/selected_cm_contract.json')
    print('CM SELECTED',selected,chosen,flush=True)

if __name__=='__main__':
    oo=annotate();reference_matching(oo);cm_discovery(oo)
