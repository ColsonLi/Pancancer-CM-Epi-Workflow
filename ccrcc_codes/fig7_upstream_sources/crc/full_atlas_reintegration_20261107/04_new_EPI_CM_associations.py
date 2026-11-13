from full_common import *
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from threadpoolctl import threadpool_limits
OUT=ROOT/'associations';OUT.mkdir(exist_ok=True)
def residual_ranks(values,study):
 ranks=np.column_stack([stats.rankdata(values[:,j]) for j in range(values.shape[1])]);frame=pd.DataFrame(ranks);return ranks-frame.groupby(np.asarray(study)).transform('mean').to_numpy()
def family(df,features,modules,level,status,threshold):
 x=df[features].to_numpy();y=df[modules].to_numpy();st=df.study_id.to_numpy();rx=residual_ranks(x,st);ry=residual_ranks(y,st)
 dummies=pd.get_dummies(df.study_id,drop_first=True,dtype=float).to_numpy();dfree=max(1,len(df)-len(np.unique(st))-1)
 rows=[]
 for i,feat in enumerate(features):
  for j,cm in enumerate(modules):
   variable=np.std(rx[:,i])>1e-10 and np.std(ry[:,j])>1e-10
   if variable:
    raw=stats.spearmanr(x[:,i],y[:,j]);r=float(stats.pearsonr(rx[:,i],ry[:,j]).statistic)
    if level=='sample':
     design=sm.add_constant(np.column_stack([stats.rankdata(x[:,i]),dummies]),has_constant='add')
     fit=sm.OLS(stats.rankdata(y[:,j]),design).fit(cov_type='cluster',cov_kwds={'groups':df.donor_id_v1.to_numpy()});p=float(fit.pvalues[1])
    else:p=float(2*stats.t.sf(abs(r)*np.sqrt(dfree/max(1-r*r,1e-12)),dfree))
    rawr,rawp=raw.statistic,raw.pvalue
   else:r,p,rawr,rawp=np.nan,1.,np.nan,1.
   rows.append({'feature':feat,'CM':cm,'level':level,'status':status,'epi_threshold':threshold,'n_rows':len(df),'n_patients':df.donor_id_v1.nunique(),'n_studies':len(np.unique(st)),'raw_spearman_rho':rawr,'raw_spearman_p':rawp,'study_adjusted_rank_r':r,'study_adjusted_p':p,'df_adjusted':dfree,'n_zero_feature':int(np.sum(x[:,i]==0)),'constant_within_study':not variable})
 result=pd.DataFrame(rows);result['q_full_family_BH']=multipletests(result.study_adjusted_p,method='fdr_bh')[1];result['family_size']=len(result);return result
def main():
 obs=load_metadata();epi=pd.read_csv(ROOT/'epithelial/primary_all/per_cell_labels.csv.gz',index_col=0,low_memory=False)
 sel=json.loads((ROOT/'epithelial/primary_all/frozen_cluster_selection.json').read_text())
 cmfreeze=json.loads((ROOT/'CM/frozen_CM_selection.json').read_text());assert cmfreeze['epithelial_outcomes_read'] is False
 cm=pd.read_csv(ROOT/'CM/W_NNLS_unweighted_closed.csv',index_col=0);modules=cm.columns.tolist()
 cmeta=pd.read_csv(ROOT/'CM/all_sample_CM_eligibility.csv',index_col=0)
 counts=pd.crosstab(epi['sample'],epi.cluster_v1).reindex(cmeta.index,fill_value=0);features=counts.columns.tolist();fraction=counts.div(counts.sum(1).replace(0,np.nan),axis=0).fillna(0)
 df=cmeta.copy();df['n_epithelial']=counts.sum(1);df=df.join(fraction).join(cm)
 df['log_fraction_ratio_JUN_VIM']=np.log((df[sel['JUN_cluster']]+.005)/(df[sel['VIM_cluster']]+.005))
 df['eligible_association_epi30_non50']=(df.n_epithelial>=30)&(df.n_non_epithelial>=50)
 df['eligible_association_epi50_non50']=(df.n_epithelial>=50)&(df.n_non_epithelial>=50)
 df.to_csv(OUT/'all_sample_new_EPI_CM_inputs.csv',index_label='sample');counts.to_csv(OUT/'new_epithelial_sample_cluster_counts.csv',index_label='sample');fraction.to_csv(OUT/'new_epithelial_sample_cluster_fractions.csv',index_label='sample')
 allres=[];contrasts=[];eligible_counts=[];within=[]
 for threshold in [30,50]:
  for status in ['tumor','normal']:
   use=df.loc[(df.status==status)&(df.n_epithelial>=threshold)&(df.n_non_epithelial>=50)].copy()
   eligible_counts.append({'status':status,'epi_threshold':threshold,'samples':len(use),'patients':use.donor_id_v1.nunique(),'studies':use.study_id.nunique()})
   use.to_csv(OUT/f'eligible_{status}_epi{threshold}_sample_inputs.csv',index_label='sample')
   if len(use)<5:continue
   patient=use.groupby(['study_id','donor_id_v1'])[[*features,*modules,'log_fraction_ratio_JUN_VIM']].mean().reset_index()
   patient['n_samples']=use.groupby(['study_id','donor_id_v1']).size().to_numpy()
   patient.to_csv(OUT/f'{status}_epi{threshold}_patient_aggregated_inputs.csv',index=False)
   for level,values in [('sample',use.reset_index()),('patient',patient)]:
    allres.append(family(values,features,modules,level,status,threshold));contrasts.append(family(values,['log_fraction_ratio_JUN_VIM'],modules,level,status,threshold))
   if threshold==30:
    for study,g in patient.groupby('study_id'):
     for f in [sel['JUN_cluster'],sel['VIM_cluster'],'log_fraction_ratio_JUN_VIM']:
      for module in modules:
       r,p=stats.spearmanr(g[f],g[module]) if len(g)>=4 and g[f].nunique()>1 and g[module].nunique()>1 else (np.nan,np.nan)
       within.append({'status':status,'study_id':study,'feature':f,'CM':module,'n_patients':len(g),'rho':r,'p':p})
 result=pd.concat(allres,ignore_index=True);result.to_csv(OUT/'ALL_cluster_ALL_module_complete_associations.csv',index=False)
 contrast=pd.concat(contrasts,ignore_index=True);contrast.to_csv(OUT/'prespecified_JUN_VIM_fraction_ratio_associations.csv',index=False)
 selected=result.loc[result.feature.isin([sel['JUN_cluster'],sel['VIM_cluster']])].copy();selected.to_csv(OUT/'selected_reference_maxima_associations_all_modules.csv',index=False)
 pd.DataFrame(within).to_csv(OUT/'selected_features_within_study_patient_associations.csv',index=False);pd.DataFrame(eligible_counts).to_csv(OUT/'association_eligibility_counts.csv',index=False)
 # Historical comparator is explicitly not the new primary CM.
 legacy=pd.read_csv(ROOT.parent/'cm_epi_redesign/all_cm_sample_activities.csv',index_col=0)['C_K14_minmax_studybal__CM1like']
 comparisons=[]
 for module in modules:
  joint=pd.concat([cm[module],legacy],axis=1).dropna();r,p=stats.spearmanr(joint.iloc[:,0],joint.iloc[:,1]);comparisons.append({'new_CM':module,'historical_comparator':'C_K14_minmax_studybal__CM1like','n_samples':len(joint),'rho':r,'p_descriptive':p,'used_to_select_new_CM':False})
 pd.DataFrame(comparisons).to_csv(OUT/'historical_K14_comparator_only.csv',index=False)
 # One exact cross-modal population mapping: no unassigned cells.
 non=pd.read_csv(ROOT/'CM/all_non_epithelial_new_subtype_labels.csv.gz',index_col=0,low_memory=False)
 final=obs.copy();final['atlas_cluster_v1']=final.cluster_v1;final['new_subtype_v1']=''
 final.loc[epi.index,'new_subtype_v1']=epi.cluster_v1.to_numpy();final.loc[non.index,'new_subtype_v1']=non.new_subtype_v1.to_numpy();assert final.new_subtype_v1.ne('').all()
 final['in_atlas_expression_fit_reference']=final.in_expression_fit_reference
 final['in_lineage_expression_fit_reference']=False
 final.loc[epi.index,'in_lineage_expression_fit_reference']=epi.in_expression_fit_reference.to_numpy()
 for path in (ROOT/'lineages').iterdir():
  if path.is_dir() and (path/'per_cell_labels.csv.gz').exists():
   ll=pd.read_csv(path/'per_cell_labels.csv.gz',index_col=0,usecols=['cell_id','in_expression_fit_reference'])
   final.loc[ll.index,'in_lineage_expression_fit_reference']=ll.in_expression_fit_reference.to_numpy()
 final['JUN_reference_maximum_v1']=final.new_subtype_v1.eq(sel['JUN_cluster']);final['VIM_reference_maximum_v1']=final.new_subtype_v1.eq(sel['VIM_cluster'])
 final.to_csv(ROOT/'all_cell_final_labels.csv.gz',index_label='cell_id')
 import anndata as ad
 atlas=ad.read_h5ad(ROOT/'atlas/all_cells_expression_integrated.h5ad');assert atlas.obs_names.equals(final.index);atlas.obs=final
 atlas.uns['final_population_map_v1']={'per_cell_table':str(ROOT/'all_cell_final_labels.csv.gz'),'epithelial_reference_selection':str(ROOT/'epithelial/primary_all/frozen_cluster_selection.json'),'non_epithelial_subtypes':str(ROOT/'CM/new_subtype_annotation.csv'),'new_CM_activities':str(ROOT/'CM/W_NNLS_unweighted_closed.csv'),'JUN_cluster':sel['JUN_cluster'],'VIM_cluster':sel['VIM_cluster']}
 atlas.write_h5ad(ROOT/'atlas/all_cells_expression_integrated.h5ad',compression='gzip')
 savejson(OUT/'association_completion.json',{'complete':True,'primary':'patient aggregation; tumor epi>=30/non_epi>=50; study-adjusted rank correlation','sample_analysis':'patient-cluster robust rank OLS with study fixed effects','full_family_BH':'all new epithelial clusters × all chosen-rank CM modules, separately by status/coverage/level','ratio_family':'one prespecified log fraction ratio × all modules, separately corrected','epithelial_selection':sel,'new_CM_K':cmfreeze['K_selected'],'all_pairs_saved':len(result),'fraction_denominator':'all new-atlas epithelial cells within each sample','ratio_pseudocount':.005,'spatial_clinical':'not evaluated; old results not reused'})
 print(pd.DataFrame(eligible_counts).to_string(index=False),flush=True);print(selected.query('status=="tumor" and level=="patient" and epi_threshold==30').to_string(index=False),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=4):main()
