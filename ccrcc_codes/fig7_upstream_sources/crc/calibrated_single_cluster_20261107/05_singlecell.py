from cal_common import *

def main():
 d=ROOT/'singlecell';d.mkdir(exist_ok=True)
 sel=json.loads((ROOT/'selected_pair.json').read_text());cm=json.loads((ROOT/'CM/selected_CM.json').read_text());pair=sel['pair'];c1=cm['CM1_like']
 labels=pd.read_csv(sel['exact_labels'],index_col=0);metadata=pd.read_csv(ROOT/'data/sample_counts.csv',index_col=0);counts=pd.read_csv(ROOT/'data/non_epithelial_sample_subtype_counts.csv',index_col=0)
 epi=pd.crosstab(labels.sample_id,labels.cluster_id);metadata['n_epi']=epi.sum(1).reindex(metadata.index).fillna(0).astype(int)
 for state in ['Epi_JUN','Epi_VIM']:metadata['n_'+state]=epi[pair[state]].reindex(metadata.index).fillna(0).astype(int);metadata['f_'+state]=metadata['n_'+state]/metadata.n_epi.replace(0,np.nan)
 metadata['balance']=np.log((metadata.f_Epi_JUN+.005)/(metadata.f_Epi_VIM+.005))
 raw=pd.read_csv(ROOT/'CM/W_unweighted_NNLS_raw.csv',index_col=0);closed=pd.read_csv(ROOT/'CM/W_unweighted_NNLS_closed.csv',index_col=0);metadata['CM1_raw']=raw[c1];metadata['CM1_closed']=closed[c1]
 metadata.to_csv(d/'exact_pair_sample_fractions_and_CM.csv',index_label='sample_id')
 H=pd.read_csv(ROOT/'CM/H_selected.csv',index_col=0);scale=pd.read_csv(ROOT/'CM/scaling.csv',index_col=0).reindex(H.columns)
 rows=[];units=[];boot=[];loso=[];rng=np.random.default_rng(SEED)
 for threshold in [30,50]:
  use=metadata.status.eq('tumor')&metadata.n_epi.ge(threshold)&metadata.n_non_epi.ge(50);m=metadata.loc[use].copy()
  # Aggregate actual exact-cluster counts and actual non-epi subtype counts,
  # then independently project the pooled patient composition by NNLS.
  p=m.groupby(['donor_id_v1','study_id'],as_index=False).agg(n_samples=('n_epi','size'),n_epi=('n_epi','sum'),n_non_epi=('n_non_epi','sum'),n_Epi_JUN=('n_Epi_JUN','sum'),n_Epi_VIM=('n_Epi_VIM','sum'))
  p['f_Epi_JUN']=p.n_Epi_JUN/p.n_epi;p['f_Epi_VIM']=p.n_Epi_VIM/p.n_epi;p['balance']=np.log((p.f_Epi_JUN+.005)/(p.f_Epi_VIM+.005))
  pc=counts.loc[m.index].groupby(m.donor_id_v1).sum().reindex(p.donor_id_v1);pf=pc.div(pc.sum(1),axis=0).reindex(columns=H.columns)
  v=((pf-scale['min'])/scale.span).clip(0,1).to_numpy();w=np.array([nnls(H.to_numpy().T,row)[0] for row in v]);p['CM1_raw']=w[:,H.index.get_loc(c1)];p['CM1_closed']=w[:,H.index.get_loc(c1)]/np.maximum(w.sum(1),1e-12);p=p.set_index('donor_id_v1',drop=False)
  m['unit_id']=m.index;m['analysis_unit']='sample';p['unit_id']=p.donor_id_v1;p['analysis_unit']='patient';p['min_epi']=threshold;m['min_epi']=threshold;units.extend([m,p])
  for unit,t in [('sample',m),('patient',p)]:
   for activity in ['CM1_raw','CM1_closed']:
    for feat in ['f_Epi_JUN','f_Epi_VIM','balance']:
     r,pv=safe_spearman(t[feat],t[activity]);rr,pp,df=partial_rank(t[feat],t[activity],t.study_id)
     context=dict(unit=unit,min_epi=threshold,CM_activity=activity,feature=feat,n=len(t),n_patients=t.donor_id_v1.nunique(),n_studies=t.study_id.nunique(),cluster_JUN=pair['Epi_JUN'],cluster_VIM=pair['Epi_VIM'],CM=c1)
     rows.extend([{**context,'method':'raw_Spearman','effect':r,'p':pv,'df':len(t)-2},{**context,'method':'study_partial_rank','effect':rr,'p':pp,'df':df}])
     if threshold==30 and activity=='CM1_raw':
      for study in sorted(t.study_id.unique()):
       tt=t.loc[t.study_id.ne(study)];lr,lp,ldf=partial_rank(tt[feat],tt[activity],tt.study_id);loso.append(dict(**context,omitted_study=study,partial_rank_r=lr,p=lp,n_remaining=len(tt)))
      for b in range(500):
       ids=[]
       for _,g in t.groupby('study_id'):
        ds=g.donor_id_v1.unique()
        for donor in rng.choice(ds,len(ds),True):ids.extend(np.flatnonzero(t.index.isin(g.index[g.donor_id_v1.eq(donor)])))
       tt=t.iloc[ids];br,bp,_=partial_rank(tt[feat],tt[activity],tt.study_id);boot.append(dict(**context,replicate=b,partial_rank_r=br))
 result=pd.DataFrame(rows);result['q_BH_all_48_tests']=bh(result.p)
 primary=result.unit.eq('sample')&result.min_epi.eq(30)&result.CM_activity.eq('CM1_raw');result['q_BH_primary_6_tests']=np.nan;result.loc[primary,'q_BH_primary_6_tests']=bh(result.loc[primary,'p'])
 result.to_csv(d/'ALL_exact_pair_associations.csv',index=False);pd.concat(units).to_csv(d/'all_sample_patient_analysis_inputs.csv.gz',index_label='source_index');pd.DataFrame(boot).to_csv(d/'primary_threshold_donor_bootstrap.csv.gz',index=False);pd.DataFrame(loso).to_csv(d/'primary_threshold_leave_study_out.csv',index=False)
 bs=pd.DataFrame(boot).groupby(['unit','feature']).partial_rank_r.agg([lambda z:z.quantile(.025),lambda z:z.quantile(.975)]);bs.columns=['CI_low','CI_high'];bs.to_csv(d/'study_partial_rank_donor_bootstrap_CI.csv')
 # Independently recompute one association using explicit Pearson covariance
 # of within-study residualized rank vectors and the appropriate residual df.
 t=metadata.loc[metadata.status.eq('tumor')&metadata.n_epi.ge(30)&metadata.n_non_epi.ge(50)];xr=t.balance.rank(method='average');yr=t.CM1_raw.rank(method='average');xr=xr-xr.groupby(t.study_id).transform('mean');yr=yr-yr.groupby(t.study_id).transform('mean');independent=float(np.dot(xr,yr)/np.sqrt(np.dot(xr,xr)*np.dot(yr,yr)));reported=float(result.loc[primary&result.feature.eq('balance')&result.method.eq('study_partial_rank'),'effect'].iloc[0]);check={'independently_recomputed_association':'sample primary balance vs raw unweighted NNLS CM1, study-adjusted rank residual covariance','recomputed_r':independent,'table_r':reported,'absolute_error':abs(independent-reported),'passed':abs(independent-reported)<1e-10};savejson(d/'independent_association_check.json',check)
 savejson(d/'methods.json',{'exact_pair':pair,'CM1':c1,'primary_samples':'tumor samples >=30 cleaned tumor epithelial and >=50 cleaned non-epithelial cells','sensitivity':'>=50 epithelial; same >=50 non-epi','denominator':'all retained tumor-source epithelial cells in that sample, not the two-state subtotal','balance':'log((fJUN+.005)/(fVIM+.005))','CM_primary':'unweighted raw NNLS usage on row-L1 normalized H; no sqrt-weight in projection','partial':'Pearson of global ranks residualized on study dummy variables; t df=n-rank(study design)-1','patient_aggregation':'sum exact epithelial-cluster counts over individually eligible samples; pool non-epi subtype counts, apply frozen minmax then unweighted NNLS; one row per namespaced donor','multiplicity':'one 48-test family=3 features x2 thresholds x2 units x2 CM activity scales x2 methods; primary six-test family also shown; bootstrap/LOSO descriptive, no selection','n_primary_samples':int(primary.sum()/6*len(t)) if False else len(t),'n_primary_patients':t.donor_id_v1.nunique(),'independent_recompute_passed':check['passed']})
 print(result.loc[primary].to_string(index=False),flush=True);print(result.loc[result.unit.eq('patient')&result.min_epi.eq(30)&result.CM_activity.eq('CM1_raw')].to_string(index=False),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=4):main()
