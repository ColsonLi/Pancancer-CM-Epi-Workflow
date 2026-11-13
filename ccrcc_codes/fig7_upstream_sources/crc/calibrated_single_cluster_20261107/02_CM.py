from cal_common import *
from sklearn.decomposition import NMF

def weights(meta,variant):
 w=np.zeros(len(meta))
 for status,g in meta.groupby('status'):
  ids=meta.index.get_indexer(g.index)
  if 'studybal' not in variant:w[ids]=.5/len(g)
  else:
   for _,sg in g.groupby('study_id'):w[meta.index.get_indexer(sg.index)]=.5/g.study_id.nunique()/len(sg)
 return w/w.mean()

def fit(v,meta,k,variant,seed=42):
 w=weights(meta.reset_index(drop=True),variant);a=v*np.sqrt(w[:,None])
 model=NMF(n_components=k,init='nndsvda' if seed==42 else 'random',random_state=seed,solver='cd',max_iter=3500,tol=1e-5,alpha_H=.001 if 'legacy' in variant else 0,l1_ratio=.1 if 'legacy' in variant else 0)
 with warnings.catch_warnings(record=True) as caught:model.fit_transform(a)
 h=model.components_;proj=np.array([nnls(h.T,row)[0] for row in a]);err=np.square(a-proj@h).sum()
 return h,dict(n_iter=model.n_iter_,converged=model.n_iter_<model.max_iter,weighted_SSE=err,weighted_explained_fraction=1-err/np.square(a).sum(),warning_count=len(caught))

def main():
 d=ROOT/'CM';d.mkdir(exist_ok=True)
 counts=pd.read_csv(ROOT/'data/non_epithelial_sample_subtype_counts.csv',index_col=0);meta=pd.read_csv(ROOT/'data/sample_counts.csv',index_col=0)
 freq=counts.div(counts.sum(1).replace(0,np.nan),axis=0).fillna(0);eligible=meta.n_non_epi>=50
 lo=freq.loc[eligible].min(0);hi=freq.loc[eligible].max(0);span=(hi-lo);vary=span>1e-10
 scaled=((freq.loc[:,vary]-lo[vary])/span[vary]).clip(0,1);v=scaled.loc[eligible].to_numpy();m=meta.loc[eligible].copy().reset_index()
 freq.to_csv(d/'sample_non_epi_subtype_fractions.csv');scaled.to_csv(d/'sample_column_minmax_matrix.csv');pd.DataFrame({'min':lo,'max':hi,'span':span,'used':vary}).to_csv(d/'scaling.csv')
 variants=['legacy_minmax_studybal','legacy_minmax_statusbal','standard_minmax_studybal','standard_minmax_statusbal']
 savejson(d/'selection_specification.json',{'K':[12,13,14,15],'variants':variants,'fit_eligibility':'at least 50 non-epithelial cells, all tumor and normal samples','weights':'equal total tumor/normal, equal studies within status for studybal; normalized mean weight=1','legacy_regularization':'alpha_H=.001, l1_ratio=.1; clean C_K14_minmax_studybal equivalent','standard_regularization':'unpenalized','rank_selection':'require full and >=95% resample convergence, median stability >=.75; among fits within 95% of best weighted explained fraction choose maximum minimum(patient q10,study q10) across ALL matched components; ties <.005 select coarsest K then study-balanced. No reference/EPI/outcome enters rank selection.','resampling':'30 patient-within-study/status bootstraps plus 10 80%-study subsamples per fit; frozen full-data minmax scaling','projection':'unweighted scipy NNLS on row-L1-normalized H; raw W primary, closed W sensitivity'})
 fits={};screen=[];stability=[];rng=np.random.default_rng(SEED)
 if (d/'all_fits_convergence_reconstruction.csv').exists() and (d/'patient_study_module_stability.csv').exists():
  screen=pd.read_csv(d/'all_fits_convergence_reconstruction.csv').to_dict('records');stability=pd.read_csv(d/'patient_study_module_stability.csv').to_dict('records')
  for name in pd.DataFrame(screen).model.unique():fits[name]=pd.read_csv(d/name/'H_native.csv',index_col=0).to_numpy()
  print('Reusing completed numerical fits/resamples; finishing CM annotation only',flush=True)
 for variant in variants:
  for k in [12,13,14,15]:
   name=f'K{k:02d}_{variant}';out=d/name;out.mkdir(exist_ok=True)
   if name in fits:continue
   two=[fit(v,m,k,variant,s) for s in [42,143]];h,diag=min(two,key=lambda z:z[1]['weighted_SSE']);fits[name]=h
   for j,(hs,ds) in enumerate(two):screen.append(dict(model=name,K=k,variant=variant,seed=[42,143][j],selected_seed=bool(h is hs),**ds))
   pd.DataFrame(h,columns=scaled.columns).to_csv(out/'H_native.csv',index_label='component_index')
   pd.DataFrame({'sample_id':m.sample_id,'status':m.status,'study_id':m.study_id,'fit_weight':weights(m,variant)}).to_csv(out/'sample_fit_weights.csv',index=False)
   print('fit',name,diag,flush=True)
   if variant=='standard_minmax_statusbal':rng=np.random.default_rng(SEED+400+k)
   for b in range(40):
    if b<30:
     ix=[]
     for _,g in m.groupby(['study_id','status']):
      donors=g.donor_id_v1.unique()
      for donor in rng.choice(donors,len(donors),replace=True):ix.extend(g.index[g.donor_id_v1.eq(donor)])
     kind='patient'
    else:
     studies=m.study_id.unique();chosen=rng.choice(studies,max(2,int(.8*len(studies))),False);ix=m.index[m.study_id.isin(chosen)].tolist();kind='study'
    hb,db=fit(v[ix],m.iloc[ix].reset_index(drop=True),k,variant)
    sim=cos(h,hb);rr,cc=linear_sum_assignment(-sim)
    for r,c in zip(rr,cc):stability.append(dict(model=name,K=k,variant=variant,replicate=b,resampling=kind,module_index=int(r),cosine=float(sim[r,c]),converged=db['converged'],n_iter=db['n_iter']))
   print('resampling finished',name,flush=True)
 screen=pd.DataFrame(screen);screen.to_csv(d/'all_fits_convergence_reconstruction.csv',index=False)
 st=pd.DataFrame(stability);st.to_csv(d/'patient_study_module_stability.csv',index=False)
 summ=st.groupby('model').agg(median_cosine=('cosine','median'),q10_cosine=('cosine',lambda z:z.quantile(.1)),resample_convergence=('converged','mean'))
 qs=st.groupby(['model','resampling']).cosine.quantile(.1).unstack();summ=summ.join(qs.add_suffix('_q10'));summ['min_resampling_q10']=summ[['patient_q10','study_q10']].min(1)
 selected=screen.loc[screen.selected_seed].set_index('model').join(summ)
 valid=selected.loc[selected.converged&(selected.resample_convergence>=.95)&(selected.median_cosine>=.75)]
 if len(valid)==0:valid=selected.loc[selected.converged]
 threshold=.95*valid.weighted_explained_fraction.max();short=valid.loc[valid.weighted_explained_fraction>=threshold]
 best=short.min_resampling_q10.max();close=short.loc[short.min_resampling_q10>=best-.005].sort_values(['K','variant'])
 name=close.index[0];selected['within_reconstruction_band']=selected.weighted_explained_fraction>=threshold;selected['chosen']=selected.index==name;selected.to_csv(d/'rank_selection.csv')
 h=fits[name];h=h/np.maximum(h.sum(1,keepdims=True),1e-12);names=[f'CRCcal_CM{j+1:02d}' for j in range(len(h))]
 H=pd.DataFrame(h,index=names,columns=scaled.columns);W=np.array([nnls(h.T,row)[0] for row in scaled.to_numpy()]);closed=W/np.maximum(W.sum(1,keepdims=True),1e-12)
 H.to_csv(d/'H_selected.csv',index_label='CM');pd.DataFrame(W,index=meta.index,columns=names).to_csv(d/'W_unweighted_NNLS_raw.csv');pd.DataFrame(closed,index=meta.index,columns=names).to_csv(d/'W_unweighted_NNLS_closed.csv')
 anno=pd.read_csv(ROOT/'data/non_epithelial_subtype_annotations.csv',index_col=0);lineages=anno.lineage.reindex(H.columns)
 lc=H.T.groupby(lineages).sum().T;lc.to_csv(d/'CM_lineage_H_loading_shares.csv')
 # CM1 correspondence is evaluated only AFTER K selection; use canonical
 # ccRCC marker mixtures and the same top50 rank-weighted CRC subtype mixture.
 markers=pd.read_csv(ROOT/'data/non_epithelial_top50_markers.csv')
 orig=pd.read_csv(SHARED/'reference/cm_cell_subtype_fraction_top50deg.csv');loads=pd.read_csv(SHARED/'reference/CM1_CM10_complete_subtype_loadings.csv');lookup=orig.drop_duplicates('cell_subtype').set_index('cell_subtype').top50_gene.to_dict()
 gs=sorted(set(g.strip() for s in lookup.values() for g in str(s).split(',') if not pd.Series([g.strip()]).str.match(GENE_FILTER).iloc[0]));gi={g:i for i,g in enumerate(gs)}
 sub=np.zeros((len(H.columns),len(gs)))
 for i,c in enumerate(H.columns):
  for r in markers.loc[markers.cluster.eq(c)].itertuples():
   if r.gene in gi:sub[i,gi[r.gene]]=1/np.sqrt(r.rank)
  sub[i]/=max(sub[i].sum(),1e-12)
 refs=[];refnames=sorted(loads.CM.unique())
 for cm in refnames:
  vec=np.zeros(len(gs))
  for r in loads.loc[loads.CM.eq(cm)].itertuples():
   genes=[g.strip() for g in str(lookup.get(r.cell_subtype,'')).split(',') if g.strip() in gi];w=1/np.sqrt(np.arange(1,len(genes)+1));w/=max(w.sum(),1e-12)
   for g,z in zip(genes,w):vec[gi[g]]+=r.loading_fraction*z
  refs.append(vec)
 profiles=h@sub;sim=cos(profiles,np.array(refs));matching=[]
 for j,cc in enumerate(refnames):
  ranks=stats.rankdata(-sim[:,j],method='min')
  for i,cm in enumerate(names):matching.append(dict(CM=cm,reference_CM=cc,cosine=sim[i,j],rank=ranks[i],T_NK_H_share=lc.loc[cm,'T_NK']))
 pd.DataFrame(matching).to_csv(d/'all_CM_reference_correspondence.csv',index=False)
 reference_cm1='joint_01_sharedCM';jref=refnames.index(reference_cm1)
 canonical_lineages=pd.read_csv(SHARED/'reference/all_CM_lineage_loading_shares.csv');canonical_T=float(canonical_lineages.loc[canonical_lineages.CM.eq(reference_cm1)&canonical_lineages.lineage.eq('T_NK'),'loading_fraction'].iloc[0])
 cohesive=lc.T_NK.to_numpy()>=canonical_T
 unrestricted_maximum=names[int(np.argmax(sim[:,jref]))]
 # The requested counterpart is a cohesive T/NK-rich CM, not a generic
 # marker-overlap maximum driven by mixed B/T identity. Use the canonical
 # reference's own T/NK share as the composition requirement, without any EPI
 # outcome. Preserve unrestricted marker maxima in the full matching table.
 qualified=np.flatnonzero(cohesive)
 cm1=names[int(qualified[np.argmax(sim[qualified,jref])])] if len(qualified) else names[int(lc.T_NK.to_numpy().argmax())]
 pd.DataFrame(profiles,index=names,columns=gs).to_csv(d/'module_marker_mixture_weights.csv.gz')
 chosenst=st.loc[st.model.eq(name)].copy();chosenst['CM']=np.array(names)[chosenst.module_index];chosenst.groupby('CM').cosine.agg(['median',lambda z:z.quantile(.1)]).rename(columns={'<lambda_0>':'q10'}).to_csv(d/'selected_module_stability.csv')
 comp=H.loc[cm1].sort_values(ascending=False).rename('H_share').to_frame().join(anno);comp.index.name='subtype_id';comp.to_csv(d/'CM1_composition.csv')
 # Explicit expression evidence for the T/NK states specified by the task.
 pr=pd.read_csv(ROOT/'data/non_epithelial_balanced_profiles.csv.gz',index_col=0);tg=[g for g in ['CD3D','CD3E','GZMK','GNLY','TOX','CXCL13','KLRD1','NKG7'] if g in pr]
 comp.join(pr[tg]).to_csv(d/'CM1_composition_marker_evidence.csv')
 manifest={'model':name,'K':int(selected.loc[name,'K']),'CM1_like':cm1,'canonical_reference_CM1':reference_cm1,'canonical_CM1_T_NK_share':canonical_T,'CM1_module_selection':'require T/NK H share >= canonical CM1 own T/NK share, then maximum canonical CM1 marker cosine; if none qualify, maximum T/NK share flagged. Composition selection after K but before ANY EPI association.','CM1_composition_requirement_passed':bool(cohesive[names.index(cm1)]),'unrestricted_marker_maximum_not_used':unrestricted_maximum,'CM1_reference_marker_cosine':float(sim[names.index(cm1),jref]),'CM1_T_NK_H_share':float(lc.loc[cm1,'T_NK']),'n_non_epithelial_cells':int(counts.to_numpy().sum()),'n_subtypes':counts.shape[1],'n_fit_samples':int(eligible.sum()),'fit_status_counts':meta.loc[eligible].status.value_counts().to_dict(),'n_fit_patients':meta.loc[eligible].donor_id_v1.nunique(),'n_fit_studies':meta.loc[eligible].study_id.nunique(),'diagnostics':selected.loc[name].to_dict(),'primary_activity':'unweighted NNLS raw on row-L1 H','selection_before_EPI_associations':True}
 savejson(d/'selected_CM.json',manifest);print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=4):main()
