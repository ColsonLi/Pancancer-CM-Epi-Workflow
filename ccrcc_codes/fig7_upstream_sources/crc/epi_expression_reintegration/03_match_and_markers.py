from analysis_common import *
import argparse
from scipy.spatial.distance import cdist
from scipy import stats
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
REF=ROOT/'reference'
TARGETS=['Epi_JUN','Epi_VIM']
def score_profiles(profiles,genes,kind='top100',metric='cosine',kidney=False):
 marks=pd.read_csv(REF/f'frozen_{kind}_per_state.csv')
 avail=set(genes); gs=sorted(set(marks.gene)&avail)
 if kidney:
  filt=pd.read_csv(REF/'kidney_specificity_filter.csv').set_index('gene').iloc[:,0]
  gs=[g for g in gs if not bool(filt.get(g,False))]
 ref=pd.read_csv(REF/'ccrcc_donor_balanced_state_profiles_log1p.csv.gz',index_col=0)
 scale=pd.read_csv(REF/'ccrcc_only_gene_scaling.csv').set_index('gene')
 ids=pd.Index(genes).get_indexer(gs)
 xx=profiles[:,ids]; yy=ref.loc[TARGETS,gs].to_numpy()
 if metric=='cosine':
  mu=scale.loc[gs,'reference_mean'].to_numpy();sd=scale.loc[gs,'reference_sd'].to_numpy()
  sim=1-cdist((xx-mu)/sd,(yy-mu)/sd,metric='cosine')
 else:
  sim=np.asarray([[stats.spearmanr(row,y).statistic for y in yy] for row in xx])
 return sim,gs
def balanced(pb,pm,label='cluster_v1'):
 return balance_profiles(pb,pm,label=label)[0]
def enrich_labels(scope):
 labels=pd.read_csv(ROOT/scope/'per_cell_labels.csv.gz',index_col=0,low_memory=False)
 source=pd.read_csv(ROOT/'crc_retained_metadata.csv.gz',index_col=0,low_memory=False)
 for c in ['original_barcode','donor_id_v1','source_h5ad','source_row_index','sample_id','platform','status_original']:
  if c in source:labels[c]=source.loc[labels.index,c]
 return labels
def main(scope):
 out=ROOT/scope
 x,obs,genes=load_full();labels=enrich_labels(scope)
 if scope!='primary_all':x=x[obs.index.get_indexer(labels.index)];obs=obs.loc[labels.index]
 obs=labels.copy()
 print(scope,'full-expression donor pseudobulks',x.shape,flush=True)
 pb,pm=pseudobulk(x,obs,['donor_id_v1','study_id','cluster_v1'],min_cells=10)
 # The complete full-expression pseudobulk object is reusable and auditable.
 np.savez_compressed(out/'donor_cluster_pseudobulk_log1p.npz',expression=pb,genes=genes)
 pm.to_csv(out/'donor_cluster_pseudobulk_metadata.csv',index=False)
 profiles=balanced(pb,pm)
 profiles.columns=genes
 profiles.to_csv(out/'cluster_donor_study_balanced_profiles_log1p.csv.gz',index_label='cluster')
 sim,gs=score_profiles(profiles.to_numpy(),genes)
 coverage=[]
 marks=pd.read_csv(REF/'frozen_top100_per_state.csv')
 for st in TARGETS:
  selected=set(marks.loc[marks.state.eq(st),'gene']);coverage.append({'state':st,'reference_gene_count':len(selected),'n_shared':len(selected&set(genes)),'coverage':len(selected&set(genes))/len(selected)})
 pd.DataFrame(coverage).to_csv(out/'reference_gene_coverage.csv',index=False)
 result=pd.DataFrame(sim,index=profiles.index,columns=['JUN_similarity','VIM_similarity'])
 for st in ['JUN','VIM']:result[st+'_rank']=result[st+'_similarity'].rank(ascending=False,method='min').astype(int)
 result['JUN_minus_VIM']=result.JUN_similarity-result.VIM_similarity
 info=obs.groupby('cluster_v1').agg(n_cells=('sample','size'),n_samples=('sample','nunique'),n_donors=('donor_id_v1','nunique'),n_studies=('study_id','nunique'),n_technical_sources=('series','nunique'),tumor_fraction=('status',lambda z:z.eq('tumor').mean()),normal_fraction=('status',lambda z:z.eq('normal').mean()))
 result=result.join(info)
 result['n_donors_ge10']=pm.groupby('cluster_v1').donor_id_v1.nunique()
 result['n_studies_ge10']=pm.groupby('cluster_v1').study_id.nunique()
 result['reference_union_genes_n']=len(gs)
 result['JUN_reference_coverage']=coverage[0]['coverage'];result['VIM_reference_coverage']=coverage[1]['coverage']
 # Independent argmax, frozen before looking at CM outcomes.
 selection={'JUN_cluster':result.JUN_similarity.idxmax(),'VIM_cluster':result.VIM_similarity.idxmax(),'same_cluster_wins_both':result.JUN_similarity.idxmax()==result.VIM_similarity.idxmax(),'metric':'cosine of ccRCC-gene-standardized donor/study-balanced mean log1p on fixed reference markers','selected_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),'cm_outcomes_read':False}
 savejson(out/'frozen_cluster_selection.json',selection)
 result.to_csv(out/'all_cluster_reference_similarity.csv',index_label='cluster')
 print('FROZEN',selection,flush=True)
 # Prespecified metric/signature sensitivities, never used to replace maxima.
 sens=[]
 for name,kind,metric,kidney in [('same_genes_spearman','top100','spearman',False),('broader_top300_cosine','top300','cosine',False),('kidney_filtered_cosine','top100','cosine',True)]:
  scores,sgenes=score_profiles(profiles.to_numpy(),genes,kind=kind,metric=metric,kidney=kidney)
  for j,st in enumerate(['JUN','VIM']):
   ranks=stats.rankdata(-scores[:,j],method='min')
   for i,c in enumerate(profiles.index):sens.append({'configuration':name,'cluster':c,'reference_state':st,'similarity':scores[i,j],'rank':int(ranks[i]),'n_genes':len(sgenes),'is_maximum':scores[i,j]==scores[:,j].max()})
 pd.DataFrame(sens).to_csv(out/'reference_metric_sensitivities.csv',index=False)
 # Separate source status interpretation, not malignant-vs-benign calls.
 stat=[]
 for status in ['tumor','normal']:
  k=obs.status.eq(status).to_numpy()
  if not k.any():continue
  sp,sm=pseudobulk(x[k],obs.loc[k],['donor_id_v1','study_id','cluster_v1'],min_cells=10)
  pp=balanced(sp,sm);ss,_=score_profiles(pp.to_numpy(),genes)
  for i,c in enumerate(pp.index):stat.append({'status':status,'cluster':c,'JUN_similarity':ss[i,0],'VIM_similarity':ss[i,1],'n_cells':int((obs.loc[k,'cluster_v1']==c).sum()),'n_donor_cluster_profiles':int(sm.cluster_v1.eq(c).sum())})
 pd.DataFrame(stat).to_csv(out/'status_separate_reference_similarity.csv',index=False)
 # Bootstrap donors within biological study, shared draw across all clusters.
 rng=np.random.default_rng(20261106); nboot=200; bsim=[]
 donor_study=obs[['study_id','donor_id_v1']].drop_duplicates()
 pm2=pm.copy();pm2['donor_draw_index']=pd.Index(donor_study.donor_id_v1).get_indexer(pm.donor_id_v1)
 ids=pd.Index(genes).get_indexer(gs);small=pb[:,ids]
 clusters=profiles.index.tolist();ncl=len(clusters)
 for b in range(nboot):
  mult=np.zeros(len(donor_study),dtype=int)
  for study,ix in donor_study.groupby('study_id').indices.items():
   draw=rng.choice(ix,len(ix),replace=True);mult+=np.bincount(draw,minlength=len(mult))
  weight=mult[pm2.donor_draw_index.to_numpy()]
  sums=np.zeros((ncl,len(gs)));ns=np.zeros(ncl)
  for (cl,study),ix in pm.groupby(['cluster_v1','study_id']).indices.items():
   w=weight[ix]
   if w.sum()>0:
    ci=clusters.index(cl);sums[ci]+=np.average(small[ix],axis=0,weights=w);ns[ci]+=1
  bp=sums/np.maximum(ns[:,None],1)
  bs,_=score_profiles(bp,gs)
  bs[ns==0]=np.nan
  bsim.append(bs)
 boot=np.asarray(bsim)
 np.savez_compressed(out/'donor_bootstrap_similarity.npz',similarities=boot,clusters=np.asarray(clusters),states=np.asarray(TARGETS),seed=20261106)
 for j,st in enumerate(['JUN','VIM']):
  result[st+'_ci_low']=np.nanquantile(boot[:,:,j],.025,axis=0);result[st+'_ci_high']=np.nanquantile(boot[:,:,j],.975,axis=0)
  wins=np.nanargmax(boot[:,:,j],axis=1)
  result[st+'_bootstrap_win_probability']=[np.mean(wins==i) for i in range(ncl)]
  def finite_ranks(v):
   r=np.full(len(v),np.nan,dtype=float);ok=np.isfinite(v);r[ok]=stats.rankdata(-v[ok],method='min');return r
  ranks=np.apply_along_axis(finite_ranks,1,boot[:,:,j])
  result[st+'_bootstrap_profile_available_fraction']=np.isfinite(boot[:,:,j]).mean(0)
  result[st+'_rank_ci_low']=np.nanquantile(ranks,.025,axis=0);result[st+'_rank_ci_high']=np.nanquantile(ranks,.975,axis=0)
 result.to_csv(out/'all_cluster_reference_similarity.csv',index_label='cluster')
 print(scope,'balanced markers',flush=True)
 # Full-gene donor contrasts: target vs all other retained epithelial cells in
 # the same donor; average donor effects within study then studies equally.
 dx=x.copy();dx.data=np.ones_like(dx.data)
 det,dm=pseudobulk(dx,obs,['donor_id_v1','study_id','cluster_v1'],min_cells=10);del dx
 assert pm.equals(dm)
 dp,dmeta=pseudobulk(x,obs,['donor_id_v1','study_id'],min_cells=1)
 donor_lookup={r.donor_id_v1:i for i,r in dmeta.iterrows()}
 marker=[]
 detprof=balanced(det,pm);detprof.columns=genes;detprof.to_csv(out/'cluster_balanced_detection.csv.gz',index_label='cluster')
 for cl in clusters:
  ii=np.where(pm.cluster_v1.eq(cl))[0];di=np.array([donor_lookup[d] for d in pm.iloc[ii].donor_id_v1]);n=pm.iloc[ii].n_cells.to_numpy();total=dmeta.iloc[di].n_cells.to_numpy();ok=total-n>=10
  ii=ii[ok];di=di[ok];n=n[ok];total=total[ok]
  if len(ii)<2:continue
  rest=(dp[di]*total[:,None]-pb[ii]*n[:,None])/(total-n)[:,None]
  contrast=pb[ii]-rest
  study=pd.DataFrame(contrast).groupby(pm.iloc[ii].study_id.to_numpy()).mean()
  delta=study.mean(0).to_numpy()
  pval=stats.ttest_1samp(contrast,0,axis=0,nan_policy='omit').pvalue
  pval=np.nan_to_num(pval,nan=1.)
  q=multipletests(pval,method='fdr_bh')[1]
  tab=pd.DataFrame({'cluster':cl,'gene':genes,'delta_log1p_donor_study_balanced':delta,'mean_log1p':profiles.loc[cl].to_numpy(),'detection':detprof.loc[cl].to_numpy(),'p_donor_contrast':pval,'q_within_cluster':q,'n_donors_with_rest':len(ii),'n_studies_with_rest':len(study)})
  tab['eligible_marker']=(tab.delta_log1p_donor_study_balanced>=.15)&(tab.detection>=.10)&~tab.gene.str.match(GENE_FILTER)
  tab=tab.sort_values(['eligible_marker','delta_log1p_donor_study_balanced'],ascending=False)
  tab['marker_rank']=np.where(tab.eligible_marker,np.cumsum(tab.eligible_marker),np.nan)
  marker.append(tab)
 markers=pd.concat(marker,ignore_index=True)
 markers.to_csv(out/'all_gene_patient_study_balanced_markers.csv.gz',index=False)
 top=markers.loc[markers.eligible_marker&markers.marker_rank.le(50)];top.to_csv(out/'top50_balanced_markers.csv',index=False)
 topgenes=top.groupby('cluster').head(5).groupby('cluster').gene.agg(';'.join)
 result['top5_balanced_markers']=topgenes;result.to_csv(out/'all_cluster_reference_similarity.csv',index_label='cluster')
 lineages={'epithelial':['EPCAM','KRT8','KRT18','KRT19','CDH1','VIL1'],'immune':['PTPRC','CD3D','CD3E','LST1','TYROBP'],'stromal':['COL1A1','COL1A2','DCN','LUM'],'endothelial':['PECAM1','VWF','KDR'],'colonocyte':['CA1','CA2','CA7','SLC26A3','AQP8'],'goblet':['MUC2','TFF3','SPINK4','REG4'],'cycling':['MKI67','TOP2A','UBE2C'],'AP1':['JUN','FOS','JUNB','ATF3'],'mesenchymal':['VIM','FN1','LGALS1','S100A4']}
 qc=[]
 for cl in clusters:
  for lineage,gset in lineages.items():
   gset=[g for g in gset if g in profiles]
   if gset:qc.append({'cluster':cl,'program_annotation_only':lineage,'genes':';'.join(gset),'mean_log1p':profiles.loc[cl,gset].mean(),'mean_detection':detprof.loc[cl,gset].mean()})
 pd.DataFrame(qc).to_csv(out/'broad_lineage_QC_annotation_only.csv',index=False)
 # Compare old program-manifold identities only AFTER all selections freeze.
 legacy=OLD/'deep_optimization/all_state_cell_weights.npz'
 if scope=='primary_all' and legacy.exists():
  oldobs=pd.read_csv(OLD/'clean_epithelial_metadata.csv.gz',index_col=0);oldobs=oldobs.loc[oldobs.status.eq('tumor')]
  weights=np.load(legacy);keys=[k for k in weights.files if k.startswith('leiden__immune_programs__r0.6__c')]
  if keys:
   oldlab=np.full(len(oldobs),'unassigned',dtype=object)
   for k in keys:oldlab[np.asarray(weights[k])>.5]=k.split('__')[-1]
   oldobs['old_program_cluster']=oldlab
   joint=obs.join(oldobs[['old_program_cluster']],how='inner')
   pd.crosstab(joint.cluster_v1,joint.old_program_cluster).to_csv(out/'new_vs_old_program_cluster_crosswalk_tumor.csv')
   pd.crosstab(joint.cluster_v1,joint.old_program_cluster,normalize='index').to_csv(out/'new_vs_old_program_cluster_rowfractions_tumor.csv')
 print(result.to_string(),flush=True)
 savejson(out/'matching_completion.json',{'complete':True,'n_clusters':len(result),'bootstrap_replicates':nboot,'bootstrap_reference':'fixed; uncertainty conditional on original ccRCC reference','selection':selection,'scope':scope})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scope',default='primary_all');a=p.parse_args()
 with threadpool_limits(limits=4):main(a.scope)
