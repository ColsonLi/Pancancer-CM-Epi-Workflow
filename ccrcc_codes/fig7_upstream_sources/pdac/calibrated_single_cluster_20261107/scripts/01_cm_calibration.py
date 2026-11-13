from common import *
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import nnls,linear_sum_assignment

def fit(v,weights,k,seed,maxiter=5000):
    model=NMF(n_components=k,init='nndsvda' if seed==42 else 'random',random_state=seed,max_iter=maxiter,tol=1e-5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');w=model.fit_transform(v*np.sqrt(weights)[:,None])
    h=model.components_;h/=np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
    return h,float(model.reconstruction_err_**2),model.n_iter_,not any('Maximum number' in str(z.message) for z in caught)
def weights(status):
    z=pd.Series(np.asarray(status));w=z.map({s:.5/z.eq(s).sum() for s in z.unique()}).to_numpy();return w/w.mean()
def match(h,hh):
    sim=cosine_similarity(h,hh);a,b=linear_sum_assignment(-sim);return a,sim[a,b]

def main():
    evidence=pd.read_csv(B/'tables/original_non_epi_cleanup_expression_evidence.csv')
    # Fixed quantitative multi-enzyme rule, applied to every subtype. A lone
    # gene in a subtype name is NOT evidence. Restore unsupported old deletions.
    own=np.array([r[r.lineage] for _,r in evidence.iterrows()])
    removed=evidence.acinar_3markers_fraction.ge(.95)&evidence.Acinar.ge(1)&evidence.Acinar.ge(.5*own)
    excluded=set(evidence.loc[removed,'cell_subtype'])
    evidence['retained']=~removed
    evidence['exclusion_evidence']=np.where(removed,'multi-enzyme contamination: >=95% cells with >=3 acinar markers, mean acinar log-expression >=1 and >=50% of own-lineage panel',np.where(evidence.archived_removed,'restored: own-lineage evidence retained; previous name-based/off-lineage deletion not sufficiently supported','retained cleaned assignment'))
    evidence.to_csv(B/'tables/non_epi_cleanup_decisions_with_evidence.csv',index=False)
    # The evidence-based review already supplies the requested granularity:
    # 39 previously retained + 15 unsupported exclusions restored = 54 nodes.
    retained=evidence[evidence.retained].copy();assert len(retained)==54
    assignments=[];centroids=[];support_rows=[];genes=None
    names={'t_nk':'TNK','myeloid':'Myeloid','b':'B','stromal':'Stromal','endothelial':'Endothelial','mast':'Mast'}
    for lineage,own in names.items():
        p=next((C/f'h5ad/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/{lineage}').glob('*.h5ad'))
        x,o,v=load_expr(p,raw=True);o=enrich(o)
        keep=~o.cell_subtype.isin(excluded);x=x[keep.to_numpy()];o=o.loc[keep].copy()
        o['non_epi_subtype']=o.cell_subtype
        o['lineage']=own;assignments.append(o[['sample','study','series','patient_id','donor','donor_key_basis','status','cell_subtype','non_epi_subtype','lineage']])
        dp,sup=donor_profiles(x,o,'non_epi_subtype',v.index)
        centroids.append(dp.groupby(level='cluster').mean());support_rows.append(sup)
        genes=v.index
        print('CM SUBTYPES',lineage,len(o),o.non_epi_subtype.nunique(),flush=True)
        del x,dp;gc.collect()
    non=pd.concat(assignments);non.to_csv(B/'labels/non_epithelial_54_subtypes.csv.gz',index_label='cell_id')
    cent=pd.concat(centroids);cent.to_csv(B/'tables/non_epi_54_donor_balanced_gene_centroids.csv.gz')
    pd.concat(support_rows).to_csv(B/'tables/non_epi_54_donor_profile_support.csv',index=False)
    markerrows=[]
    eligible=~genes.str.match(r'^(MT-|RPL|RPS)')&~genes.isin(DIGESTIVE)
    for cl in cent.index:
        lineage=non.loc[non.non_epi_subtype.eq(cl),'lineage'].iloc[0]
        others=non.loc[non.lineage.eq(lineage),'non_epi_subtype'].unique();others=[z for z in others if z!=cl]
        delta=cent.loc[cl]-cent.loc[others].mean()
        for rank,g in enumerate(delta[eligible].sort_values(ascending=False).head(20).index,1):markerrows.append({'subtype':cl,'lineage':lineage,'gene':g,'rank':rank,'donor_balanced_within_lineage_difference':delta[g]})
    pd.DataFrame(markerrows).to_csv(B/'tables/non_epi_54_expression_markers.csv',index=False)
    with h5py.File(C/'h5ad/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata/adata_anno_cellsubtype.h5ad','r') as f:allobs=enrich(read_elem(f['obs']))
    counts=pd.crosstab(non['sample'],non.non_epi_subtype).reindex(allobs['sample'].unique(),fill_value=0)
    assert counts.shape[1]==54
    meta=allobs[['sample','study','series','patient_id','donor','donor_key_basis','status']].drop_duplicates().set_index('sample').reindex(counts.index)
    meta['n_non_epi']=counts.sum(axis=1);meta['fit_eligible']=meta.n_non_epi.ge(50)
    freq=counts.div(counts.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
    counts.to_csv(B/'tables/non_epi_54_sample_counts.csv');freq.to_csv(B/'tables/non_epi_54_sample_frequencies.csv');meta.to_csv(B/'tables/non_epi_sample_eligibility.csv')
    f=freq.loc[meta.fit_eligible];mm=meta.loc[f.index];minimum=f.min();rngscale=(f.max()-minimum).replace(0,1)
    v=((f-minimum)/rngscale).clip(lower=0).to_numpy();va=((freq-minimum)/rngscale).clip(lower=0).to_numpy();wt=weights(mm.status)
    pd.DataFrame({'minimum':minimum,'range':rngscale}).to_csv(B/'tables/cm_column_minmax_parameters.csv')
    pd.DataFrame({'sample':mm.index,'status':mm.status,'fitting_weight':wt}).to_csv(B/'tables/cm_equal_tumor_normal_weights.csv',index=False)
    screen=[];seedrows=[];stability=[];Hs={}
    rng=np.random.default_rng(42)
    for k in range(10,15):
        fits=[]
        for seed in [42,17,93]:
            h,err,ni,conv=fit(v,wt,k,seed)
            fits.append((h,err,ni,conv,seed));seedrows.append({'K':k,'seed':seed,'weighted_SSE':err,'iterations':ni,'converged':conv,'max_iter':5000,'tol':1e-5})
        h,err,ni,conv,seed=min(fits,key=lambda z:z[1]);Hs[k]=h
        pd.DataFrame(h,columns=f.columns).to_csv(B/f'tables/cm_K{k}_H.csv',index_label='component')
        seedstab=np.mean([match(h,hh)[1].mean() for hh,_,_,_,_ in fits])
        for b in range(50):
            ix=[]
            for study,g in mm.groupby('study'):
                ds=g.donor.unique()
                for donor in rng.choice(ds,len(ds),replace=True):ix.extend(np.flatnonzero(mm.donor.eq(donor).to_numpy()))
            ix=np.asarray(ix);hh,ee,nn,cc=fit(v[ix],weights(mm.iloc[ix].status),k,1000+b,maxiter=3000)
            a,vals=match(h,hh)
            stability.extend({'K':k,'kind':'patient_bootstrap','replicate':str(b),'component':int(i),'matched_cosine':float(s),'refit_converged':cc,'refit_iterations':nn} for i,s in zip(a,vals))
        for study in mm.study.unique():
            ix=mm.study.ne(study).to_numpy();hh,ee,nn,cc=fit(v[ix],weights(mm.loc[ix,'status']),k,42,maxiter=3000)
            a,vals=match(h,hh)
            stability.extend({'K':k,'kind':'leave_study_out','replicate':study,'component':int(i),'matched_cosine':float(s),'refit_converged':cc,'refit_iterations':nn} for i,s in zip(a,vals))
        st=pd.DataFrame([z for z in stability if z['K']==k]);bs=st[st.kind.eq('patient_bootstrap')].matched_cosine;ls=st[st.kind.eq('leave_study_out')].matched_cosine
        screen.append({'K':k,'weighted_SSE':err,'relative_weighted_Frobenius_error':np.sqrt(err/np.sum(v*v*wt[:,None])),'explained_uncentered':1-err/np.sum(v*v*wt[:,None]),'best_seed':seed,'iterations':ni,'converged':conv,'seed_stability':seedstab,'patient_bootstrap_mean':bs.mean(),'patient_bootstrap_q10':bs.quantile(.1),'study_leaveout_mean':ls.mean(),'study_leaveout_q10':ls.quantile(.1),'resampling_converged_fraction':st.refit_converged.mean()})
        pd.DataFrame(stability).to_csv(B/'tables/cm_module_bootstrap_and_study_stability.csv',index=False)
        print('CM SCREEN',screen[-1],flush=True)
    ss=pd.DataFrame(screen);xx=(ss.K-10)/4;yy=(ss.weighted_SSE-ss.weighted_SSE.min())/(ss.weighted_SSE.max()-ss.weighted_SSE.min());ss['reconstruction_elbow_distance']=1-xx-yy
    elbow=int(ss.loc[ss.reconstruction_elbow_distance.idxmax(),'K'])
    ss['stable_eligible']=ss.converged&(ss.patient_bootstrap_mean>=.8)&(ss.study_leaveout_mean>=.8)&(ss.patient_bootstrap_q10>=.5)&(ss.study_leaveout_q10>=.5)
    candidates=ss.loc[ss.stable_eligible,'K'].tolist();selected=min(candidates,key=lambda k:(abs(k-elbow),k)) if candidates else elbow
    ss['selected']=ss.K.eq(selected);ss.to_csv(B/'tables/cm_K10_14_reconstruction_stability_screen.csv',index=False)
    pd.DataFrame(seedrows).to_csv(B/'tables/cm_all_seed_convergence.csv',index=False)
    h=Hs[selected];w=np.vstack([nnls(h.T,row,maxiter=100*selected)[0] for row in va]);frac=h/h.sum(axis=1,keepdims=True)
    modules=[f'calCM{i+1:02d}' for i in range(selected)]
    H=pd.DataFrame(h,index=modules,columns=f.columns);F=pd.DataFrame(frac,index=modules,columns=f.columns);W=pd.DataFrame(w,index=freq.index,columns=modules)
    H.to_csv(B/'tables/cm_selected_H.csv');F.to_csv(B/'tables/cm_selected_H_fraction.csv');W.to_csv(B/'tables/cm_selected_W_unweighted_NNLS.csv')
    li=non[['non_epi_subtype','lineage']].drop_duplicates().set_index('non_epi_subtype').lineage.reindex(f.columns)
    comp=pd.DataFrame({l:frac[:,li.eq(l)].sum(axis=1) for l in names.values()},index=modules)
    comp['TNK_nodes_ge5pct']=(F.loc[:,li.eq('TNK')]>=.05).sum(axis=1)
    cohesive=comp.TNK_nodes_ge5pct.ge(3)
    cm1=comp.loc[cohesive,'TNK'].idxmax() if cohesive.any() else comp.TNK.idxmax()
    comp['is_chosen_CM1']=comp.index==cm1;comp.to_csv(B/'tables/cm_selected_lineage_composition.csv')
    chosenst=pd.DataFrame(stability);chosenst=chosenst[(chosenst.K==selected)&(chosenst.component==modules.index(cm1))]
    chosenst.to_csv(B/'tables/cm1_specific_bootstrap_study_stability.csv',index=False)
    js({'n_original_non_epi_cells':int(evidence.n_cells.sum()),'removed_cells':int(evidence.loc[~evidence.retained,'n_cells'].sum()),'removed_subtypes':sorted(excluded),'cleaned_non_epi_cells':len(non),'previously_retained_parent_subtypes':39,'restored_previous_exclusions':sorted(set(FLAGGED)-excluded),'non_epi_subtypes':54,'granularity_rule':'preserve expression-annotated original subtypes; audit old39 cleanup: six multi-enzyme contaminant groups excluded, 15 prior exclusions restored for insufficient off-lineage evidence; no fragmentation or module-guided merge','n_samples_all':len(meta),'n_fit_samples':len(f),'n_fit_tumor':int(mm.status.eq('tumor').sum()),'n_fit_normal':int(mm.status.eq('normal-like').sum()),'K_screen':[10,11,12,13,14],'selected_K':selected,'reconstruction_elbow_K':elbow,'rank_selection_rule':'normalized reconstruction elbow in K10..14; nearest stable rank (mean patient/study matched cosine >=.8 and 10th percentiles >=.5; converged); ties lower K; no epithelial/clinical inputs','any_stable_rank':bool(candidates),'fit':'joint NMF on column-minmax non-epi frequencies; row sqrt weights with equal total tumor/normal weight; no regularization; L2-normalized H; unweighted NNLS W projection','CM1':cm1,'CM1_TNK_loading_fraction':comp.loc[cm1,'TNK'],'CM1_TNK_nodes_ge5pct':int(comp.loc[cm1,'TNK_nodes_ge5pct']),'CM1_alignment_rule':'highest T/NK H fraction among modules with >=3 T/NK nodes carrying >=5% load each (else highest T/NK); no epithelial or outcome selection','CM1_stability':chosenst.groupby('kind').matched_cosine.agg(['mean','median','min']).to_dict('index'),'top_CM1_nodes':F.loc[cm1].nlargest(10).to_dict(),'patient_identity_limitation':'UCSC absent true patient links uses provided sample-level proxy; study leaveout additionally reported'},'tables/cm_calibration_contract.json')
    print('CM COMPLETE',selected,cm1,comp.loc[cm1].to_dict(),flush=True)
if __name__=='__main__':main()
