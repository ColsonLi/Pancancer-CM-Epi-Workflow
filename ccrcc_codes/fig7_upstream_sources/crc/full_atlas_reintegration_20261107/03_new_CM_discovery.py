from full_common import *
from sklearn.decomposition import NMF
from scipy.optimize import nnls,linear_sum_assignment
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import warnings,time
OUT=ROOT/'CM';OUT.mkdir(exist_ok=True)
def fit_nmf(m,metadata,k,seed=42,max_iter=2000):
 status=metadata.status.to_numpy();weights=np.zeros(len(metadata))
 for s in np.unique(status):weights[status==s]=len(metadata)/(len(np.unique(status))*np.sum(status==s))
 model=NMF(n_components=k,init='nndsvda',random_state=seed,solver='cd',max_iter=max_iter,tol=1e-4)
 with warnings.catch_warnings(record=True) as caught:
  w=model.fit_transform(m*np.sqrt(weights[:,None]))
 h=model.components_;reconstruction=w@h;err=np.sum((m*np.sqrt(weights[:,None])-reconstruction)**2)
 diagnostic={'K':k,'n_iter':model.n_iter_,'max_iter':max_iter,'converged':model.n_iter_<max_iter,'weighted_SSE':err,'weighted_explained_fraction':1-err/np.maximum(np.sum((m*np.sqrt(weights[:,None]))**2),1e-12),'warning_count':len(caught)}
 return h,diagnostic
def aligned(a,b):
 sim=1-cdist(a,b,metric='cosine');rr,cc=linear_sum_assignment(-np.nan_to_num(sim));return sim[rr,cc]
def main():
 obs=load_metadata();parts=[];support=[];markers=[]
 for path in sorted((ROOT/'lineages').iterdir()):
  if not path.is_dir() or not (path/'completion.json').exists():continue
  ll=pd.read_csv(path/'per_cell_labels.csv.gz',index_col=0,low_memory=False);ll['new_subtype_v1']=ll.cluster_v1;ll['new_lineage_v1']=path.name;parts.append(ll[['sample','status','series','study_id','donor_id_v1','patient_id','new_subtype_v1','new_lineage_v1']])
  sp=pd.read_csv(path/'new_subtype_annotation.csv');support.append(sp)
  mm=pd.read_csv(path/'top50_expression_markers.csv');mm['lineage']=path.name;markers.append(mm)
 non=pd.concat(parts);assert non.index.is_unique
 assert len(non)==obs.broad_lineage_v1.ne('Epithelial').sum()
 non.to_csv(OUT/'all_non_epithelial_new_subtype_labels.csv.gz',index_label='cell_id')
 annotation=pd.concat(support,ignore_index=True);annotation.to_csv(OUT/'new_subtype_annotation.csv',index=False)
 markers=pd.concat(markers,ignore_index=True);markers.to_csv(OUT/'new_subtype_marker_evidence.csv',index=False)
 counts=pd.crosstab(non['sample'],non.new_subtype_v1)
 metadata=obs[['sample','series','study_id','patient_id','donor_id_v1','status']].drop_duplicates('sample').set_index('sample')
 counts=counts.reindex(metadata.index,fill_value=0);totals=counts.sum(1)
 freq=counts.div(totals.replace(0,np.nan),axis=0).fillna(0)
 metadata['n_non_epithelial']=totals;metadata['eligible_CM_fit']=totals>=50
 metadata['primary_fit_weight']=0.
 for status,group in metadata.loc[metadata.eligible_CM_fit].groupby('status'):
  metadata.loc[group.index,'primary_fit_weight']=metadata.eligible_CM_fit.sum()/(2*len(group))
 metadata.to_csv(OUT/'all_sample_CM_eligibility.csv',index_label='sample');counts.to_csv(OUT/'sample_subtype_counts.csv',index_label='sample');freq.to_csv(OUT/'sample_subtype_frequencies_within_non_epi.csv',index_label='sample')
 eligible=metadata.eligible_CM_fit.to_numpy();raw=freq.to_numpy(dtype='float64');lo=raw[eligible].min(0);hi=raw[eligible].max(0);span=hi-lo;vary=span>1e-10
 scaled=np.clip((raw[:,vary]-lo[vary])/span[vary],0,1)
 subtype=freq.columns.to_numpy()[vary];meta=metadata.iloc[np.where(eligible)[0]].copy().reset_index()
 train=scaled[eligible]
 pd.DataFrame({'subtype_id':freq.columns,'min':lo,'max':hi,'range':span,'used':vary}).to_csv(OUT/'column_minmax_scaling.csv',index=False)
 pd.DataFrame(scaled,index=metadata.index,columns=subtype).to_csv(OUT/'sample_subtype_minmax_matrix.csv',index_label='sample')
 screens=[];Hs={}
 for k in range(2,min(20,len(subtype)-1,len(train)-1)+1):
  h,d=fit_nmf(train,meta,k)
  if not d['converged']:h,d=fit_nmf(train,meta,k,max_iter=6000)
  Hs[k]=h;screens.append(d);print('screen',d,flush=True)
 screen=pd.DataFrame(screens);screen.to_csv(OUT/'K2_20_actual_convergence_reconstruction.csv',index=False)
 shortlist=[k for k in [4,8,12,16,20] if k in Hs];resamples=[];rng=np.random.default_rng(20261107)
 for k in shortlist:
  for b in range(40):
   if b<30:
    ix=[]
    for (st,status),g in meta.groupby(['study_id','status']):
     ds=g.donor_id_v1.unique()
     for d in rng.choice(ds,len(ds),replace=True):ix.extend(g.index[g.donor_id_v1.eq(d)])
    kind='patient_bootstrap_within_study_status'
   else:
    studies=meta.study_id.unique();chosen=rng.choice(studies,max(2,int(.8*len(studies))),replace=False);ix=meta.index[meta.study_id.isin(chosen)].tolist();kind='80percent_study_subsampling'
   ix=np.asarray(ix);h,d=fit_nmf(train[ix],meta.iloc[ix],k,seed=42,max_iter=2000)
   sims=aligned(Hs[k],h)
   for j,s in enumerate(sims):resamples.append({'K':k,'replicate':b,'resampling':kind,'aligned_module_index':j,'cosine_H':s,'fit_converged':d['converged'],'fit_n_iter':d['n_iter'],'n_fit_rows':len(ix)})
  print('stability K',k,flush=True)
 resamples=pd.DataFrame(resamples);resamples.to_csv(OUT/'rank_patient_study_resampling_stability.csv',index=False)
 stability=resamples.groupby('K').agg(median_cosine=('cosine_H','median'),q10_cosine=('cosine_H',lambda v:v.quantile(.1)),convergence_fraction=('fit_converged','mean'))
 selection=screen.set_index('K').join(stability)
 valid=selection.loc[selection.index.isin(shortlist)&selection.converged&(selection.median_cosine>=.75)&(selection.weighted_explained_fraction>=.85)]
 if len(valid):chosen=int(valid.index.min());rule='smallest shortlisted converged K with >=85% weighted reconstruction and median resampling cosine>=0.75'
 else:
  valid=selection.loc[selection.index.isin(shortlist)&selection.converged&(selection.median_cosine>=.75)]
  if len(valid):
   threshold=.95*valid.weighted_explained_fraction.max();chosen=int(valid.loc[valid.weighted_explained_fraction>=threshold].index.min());rule='smallest stable converged shortlisted K within 95% of best stable weighted explained fraction'
  else:chosen=int(selection.loc[selection.index.isin(shortlist)].median_cosine.idxmax());rule='no stable converged candidate met threshold; maximum median cosine, ambiguity flagged'
 selection.to_csv(OUT/'rank_selection_data_only.csv');h=Hs[chosen];h=h/np.maximum(h.sum(1,keepdims=True),1e-12)
 module_stability=resamples.loc[resamples.K.eq(chosen)].groupby('aligned_module_index').agg(median_cosine=('cosine_H','median'),q10_cosine=('cosine_H',lambda v:v.quantile(.1)),n_resamples=('cosine_H','size'))
 module_stability.index=[f'CRCfa1_CM{i+1:02d}' for i in module_stability.index];module_stability.to_csv(OUT/'selected_K_module_resampling_stability.csv',index_label='CM')
 activities=np.asarray([nnls(h.T,row)[0] for row in scaled]);closed=activities/np.maximum(activities.sum(1,keepdims=True),1e-12)
 names=[f'CRCfa1_CM{j+1:02d}' for j in range(chosen)]
 pd.DataFrame(h,index=names,columns=subtype).to_csv(OUT/'H_minmax_basis.csv',index_label='CM')
 pd.DataFrame(activities,index=metadata.index,columns=names).to_csv(OUT/'W_NNLS_unweighted_raw.csv',index_label='sample');pd.DataFrame(closed,index=metadata.index,columns=names).to_csv(OUT/'W_NNLS_unweighted_closed.csv',index_label='sample')
 physical=h*span[vary][None,:];physical/=np.maximum(physical.sum(1,keepdims=True),1e-12)
 composition=pd.DataFrame(physical,index=names,columns=subtype);composition.to_csv(OUT/'H_subtype_fraction_basis_composition.csv',index_label='CM')
 subtype_lineage=non[['new_subtype_v1','new_lineage_v1']].drop_duplicates().set_index('new_subtype_v1').new_lineage_v1
 lineage_comp=composition.T.groupby(subtype_lineage).sum().T;lineage_comp.to_csv(OUT/'CM_lineage_compositions.csv',index_label='CM')
 classification=[]
 for j,cm in enumerate(names):
  tm=closed[eligible&metadata.status.eq('tumor').to_numpy(),j].mean();nm=closed[eligible&metadata.status.eq('normal').to_numpy(),j].mean();ratio=(tm+1e-8)/(nm+1e-8)
  classification.append({'CM':cm,'mean_tumor_activity':tm,'mean_normal_activity':nm,'tumor_normal_ratio':ratio,'category':'tumor-associated' if ratio>1.5 else ('normal-associated' if ratio<2/3 else 'shared'),'top5_subtypes':';'.join(composition.loc[cm].nlargest(5).index)})
 pd.DataFrame(classification).to_csv(OUT/'module_tumor_shared_normal_classification.csv',index=False)
 h_basis=pd.DataFrame(h,index=names,columns=subtype)
 lineage_h=h_basis.T.groupby(subtype_lineage).sum().T
 lineage_h.to_csv(OUT/'CM_lineage_compositions_minmax_loading_basis.csv',index_label='CM')
 # Original ccRCC loading_fraction is normalized NMF H, not cell-count
 # composition. Match on the same minmax-H basis; keep back-transformed
 # fraction-basis contributions above as a separately named output.
 match_CM(h_basis,markers,lineage_h)
 manifest={'complete':True,'non_epithelial_cells':len(non),'samples_total':len(metadata),'samples_CM_eligible':int(eligible.sum()),'eligible_status_counts':metadata.loc[eligible].status.value_counts().to_dict(),'n_subtypes':len(freq.columns),'n_variable_subtypes':len(subtype),'K_selected':chosen,'selection_rule':rule,'selected_diagnostics':selection.loc[chosen].to_dict(),'fit_weights':'equal total tumor/normal; no epithelial or outcome use','projection':'scipy.optimize.nnls unweighted on frozen column-minmax matrix; raw and closed outputs','scope':'all tumor and normal samples; >=50 non-epithelial for fitting only','frozen_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),'epithelial_outcomes_read':False}
 savejson(OUT/'frozen_CM_selection.json',manifest);print(json.dumps(manifest,indent=2),flush=True)
def match_CM(composition,markers,lineage_comp):
 original=pd.read_csv(REF/'cm_cell_subtype_fraction_top50deg.csv');loads=pd.read_csv(REF/'CM1_CM10_complete_subtype_loadings.csv')
 refCM=sorted(loads.CM.unique());lookup=original.drop_duplicates('cell_subtype').set_index('cell_subtype').top50_gene.to_dict()
 gene_set=set()
 for v in lookup.values():gene_set.update(g.strip() for g in str(v).split(','))
 genes=sorted(g for g in gene_set if not pd.Series([g]).str.match(GENE_FILTER).iloc[0]);gi={g:i for i,g in enumerate(genes)}
 new=np.zeros((len(composition.columns),len(genes)))
 for i,c in enumerate(composition.columns):
  tab=markers.loc[markers.cluster.eq(c)].sort_values('rank')
  for _,r in tab.iterrows():
   if r.gene in gi:new[i,gi[r.gene]]=1/np.sqrt(r['rank'])
  if new[i].sum():new[i]/=new[i].sum()
 ref_profiles=[];ref_lineage=[]
 lineage_map={'B':'B','T':'T_NK','NK':'T_NK','Myeloid':'Myeloid','Mye':'Myeloid','Stromal':'Stromal','S':'Stromal','Endothelial':'Endothelial','Endo':'Endothelial','Mast':'Mast','Plasma':'Plasma','Schwann':'Schwann'}
 for cm in refCM:
  vec=np.zeros(len(genes));lc={lin:0. for lin in lineage_comp.columns}
  for _,r in loads.loc[loads.CM.eq(cm)].iterrows():
   gs=[g.strip() for g in str(lookup.get(r.cell_subtype,'')).split(',') if g.strip() in gi]
   weights=np.asarray([1/np.sqrt(j+1) for j in range(len(gs))]);weights/=max(weights.sum(),1e-12)
   for g,w in zip(gs,weights):vec[gi[g]]+=r.loading_fraction*w
   lin=lineage_map.get(r.lineage,lineage_map.get(r.cell_subtype.split('_')[0],r.lineage))
   if lin in lc:lc[lin]+=r.loading_fraction
  ref_profiles.append(vec);ref_lineage.append([lc[lin] for lin in lineage_comp.columns])
 new_cm=composition.to_numpy()@new
 sim=1-cdist(new_cm,np.asarray(ref_profiles),metric='cosine');lsim=1-cdist(lineage_comp.to_numpy(),np.asarray(ref_lineage),metric='cosine')
 rows=[];maxima={}
 for j,cc in enumerate(refCM):
  rank=stats.rankdata(-sim[:,j],method='min');maximum=composition.index[np.nanargmax(sim[:,j])];maxima[cc]=maximum
  for i,c in enumerate(composition.index):rows.append({'new_CM':c,'ccRCC_reference_CM':cc,'marker_correspondence_cosine':sim[i,j],'lineage_composition_cosine':lsim[i,j],'within_CRC_rank':rank[i],'is_independent_maximum':c==maximum,'n_fixed_ccRCC_marker_genes':len(genes)})
 pd.DataFrame(rows).to_csv(OUT/'all_CM_ccRCC_reference_matching.csv',index=False)
 pd.DataFrame(new_cm,index=composition.index,columns=genes).to_csv(OUT/'new_CM_marker_correspondence_profiles.csv.gz')
 pd.DataFrame(ref_profiles,index=refCM,columns=genes).to_csv(OUT/'ccRCC_fixed_CM_marker_profiles.csv.gz')
 savejson(OUT/'frozen_CM_reference_matches.json',{'independent_reference_maxima':maxima,'metric':'cosine of new-subtype top50 rank-weighted marker mixtures vs original ccRCC CM subtype-marker mixtures; genes fixed by original ccRCC','composition_basis':'row-normalized minmax NMF H for both cancers; not expanded-CM cell-count fractions; back-transformed fraction-basis contributions reported separately','lineage_composition':'reported separately, not used to select a more favorable association','CRC_epithelial_outcomes_used':False,'homology_limit':'relative composition/marker correspondence; splitting/merging possible, not proof of identical CM'})
if __name__=='__main__':
 with threadpool_limits(limits=4):main()
