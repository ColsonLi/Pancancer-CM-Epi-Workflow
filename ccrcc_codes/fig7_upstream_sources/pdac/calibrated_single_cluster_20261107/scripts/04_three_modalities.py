from common import *
from scipy.optimize import nnls
import itertools
sys.path.insert(0,str(SH))
import clinical_eval as ce

def singlecell(pair,cm):
    lab=pd.read_csv(B/f'labels/{PRIMARY}.csv.gz',index_col=0)
    meta=pd.read_csv(B/'tables/non_epi_sample_eligibility.csv',index_col=0)
    nc=pd.read_csv(B/'tables/non_epi_54_sample_counts.csv',index_col=0)
    W=pd.read_csv(B/'tables/cm_selected_W_unweighted_NNLS.csv',index_col=0)
    H=pd.read_csv(B/'tables/cm_selected_H.csv',index_col=0);scale=pd.read_csv(B/'tables/cm_column_minmax_parameters.csv',index_col=0)
    ec=pd.crosstab(lab.sample_id,lab.cluster_id).reindex(meta.index,fill_value=0)
    ef=ec.div(ec.sum(axis=1).replace(0,np.nan),axis=0)
    ec.to_csv(B/'tables/singlecell_all_cluster_sample_counts.csv');ef.to_csv(B/'tables/singlecell_all_cluster_sample_fractions.csv')
    meta['n_epi']=ec.sum(axis=1);meta['JUN_fraction']=ef[pair['JUN']];meta['VIM_fraction']=ef[pair['VIM']]
    meta['balance']=np.log((meta.JUN_fraction+.005)/(meta.VIM_fraction+.005));meta['CM1_W']=W[cm['CM1']]
    meta['primary_eligible']=meta.status.eq('tumor')&meta.n_epi.ge(30)&meta.n_non_epi.ge(50)
    meta.to_csv(B/'tables/singlecell_exact_pair_sample_metrics.csv')
    res=[];primary=[]
    for unit in ['sample','patient']:
        if unit=='sample':mc=meta.copy();ee=ec;ww=W[cm['CM1']]
        else:
            use=meta.status.eq('tumor');md=meta.loc[use]
            ee=ec.loc[use].groupby(md.donor).sum();nn=nc.loc[use].groupby(md.donor).sum()
            mc=md.groupby('donor').agg(study=('study','first'),status=('status','first'),n_samples=('status','size'),donor_key_basis=('donor_key_basis','first'))
            mc['n_epi']=ee.sum(axis=1);mc['n_non_epi']=nn.sum(axis=1)
            freq=nn.div(nn.sum(axis=1),axis=0);v=((freq-scale.minimum)/scale['range']).clip(lower=0).fillna(0)
            ww=pd.Series([nnls(H.to_numpy().T,z,maxiter=100*len(H))[0][H.index.get_loc(cm['CM1'])] for z in v.to_numpy()],index=nn.index)
        f=ee.div(ee.sum(axis=1).replace(0,np.nan),axis=0)
        metrics=pd.DataFrame({'JUN_fraction':f[pair['JUN']],'VIM_fraction':f[pair['VIM']],'balance':np.log((f[pair['JUN']]+.005)/(f[pair['VIM']]+.005))})
        if unit=='patient':
            pd.concat([mc,metrics,ww.rename('CM1_W')],axis=1).to_csv(B/'tables/singlecell_exact_pair_patient_aggregates.csv')
        for threshold in [30,50]:
            use=mc.status.eq('tumor')&mc.n_epi.ge(threshold)&mc.n_non_epi.ge(50);ix=mc.index[use]
            for adj in ['none','study']:
                for metric in metrics:
                    r,p,n=correlate(metrics.loc[ix,metric],ww.loc[ix],None if adj=='none' else mc.loc[ix,'study'])
                    row={'unit':unit,'min_epi':threshold,'min_non_epi':50,'context':'all_tumor','adjustment':adj,'metric':metric,'rho':r,'p':p,'n':n,'JUN_cluster':pair['JUN'],'VIM_cluster':pair['VIM'],'CM1':cm['CM1']}
                    res.append(row)
            for study in mc.loc[ix,'study'].unique():
                ii=ix[mc.loc[ix,'study'].eq(study)]
                for metric in metrics:
                    r,p,n=correlate(metrics.loc[ii,metric],ww.loc[ii]);res.append({'unit':unit,'min_epi':threshold,'min_non_epi':50,'context':study,'adjustment':'none','metric':metric,'rho':r,'p':p,'n':n,'JUN_cluster':pair['JUN'],'VIM_cluster':pair['VIM'],'CM1':cm['CM1']})
    out=pd.DataFrame(res);out['q_BH_within_unit_threshold_context']=np.nan
    for _,ix in out.groupby(['unit','min_epi','context']).groups.items():out.loc[ix,'q_BH_within_unit_threshold_context']=fdr(out.loc[ix,'p'])
    out['q_BH_all_primary_and_sensitivity']=np.nan;ix=out.context.eq('all_tumor');out.loc[ix,'q_BH_all_primary_and_sensitivity']=fdr(out.loc[ix,'p']);out['q_BH_all_saved_tests']=fdr(out.p)
    out.to_csv(B/'tables/singlecell_all_exact_pair_associations.csv',index=False)
    out[out.context.eq('all_tumor')].to_csv(B/'tables/singlecell_exact_pair_primary_sensitivity.csv',index=False)
    js({'fraction_denominator':'all 55,357 cleaned tumor-associated ductal epithelial cells in the frozen clustering input, not a JUN/VIM pool','primary':'tumor sample with >=30 cleaned epithelial and >=50 cleaned non-epithelial cells','sensitivity':'>=50 epithelial; count-aggregated tumor-only patient-key profiles, fixed scaling/H and fresh unweighted NNLS W','balance':'ln((f_single_JUN+0.005)/(f_single_VIM+0.005))','adjustment':'partial Spearman: residuals of both ranks on intercept and study dummy variables; t test with covariate-adjusted degrees of freedom','FDR':'primary family = three metrics x raw/adjusted = six tests at each unit/threshold; 24 primary+sensitivity tests also jointly adjusted; within-study exploratory families separate; global-all-saved-tests q additionally saved','n_primary_samples':int(meta.primary_eligible.sum()),'primary_n_epi_cells':int(meta.loc[meta.primary_eligible,'n_epi'].sum()),'primary_n_non_epi_cells':int(meta.loc[meta.primary_eligible,'n_non_epi'].sum()),'pair':{s:pair[s] for s in ['JUN','VIM']},'CM1':cm['CM1']},'tables/singlecell_contract.json')
    print('SINGLE CELL',out[(out.context=='all_tumor')&(out.unit=='sample')&(out.min_epi==30)].to_string(index=False),flush=True)
    return meta

def transfer_signatures(pair,cm):
    sig=json.loads((B/'tables/exact_selected_pair_signatures.json').read_text())
    cent=pd.read_csv(B/'tables/non_epi_54_donor_balanced_gene_centroids.csv.gz',index_col=0)
    F=pd.read_csv(B/'tables/cm_selected_H_fraction.csv',index_col=0).loc[cm['CM1']]
    cent=cent.loc[F.index];target=F.to_numpy()@cent.to_numpy();rest=(1-F).to_numpy()@cent.to_numpy()/(len(F)-1)
    effect=pd.Series(target-rest,index=cent.columns);exclude=set(sum(sig.values(),[]))|DIGESTIVE
    eligible=~effect.index.str.match(r'^(MT-|RPL|RPS)')&~effect.index.isin(exclude)
    genes=effect[eligible&(effect>.1)].sort_values(ascending=False).head(50).index.tolist();sig['CM1']=genes
    pd.DataFrame({'gene':effect.index,'CM1_H_weighted_minus_complement_effect':effect,'eligible':eligible}).to_csv(B/'tables/CM1_exact_current_module_transfer_gene_effects.csv',index=False)
    js(sig,'tables/frozen_exact_pair_and_current_CM1_transfer_signatures.json')
    return sig

def clinical(sig,pair):
    ex,cl=ce.load_cohort('PAAD',SH)
    scores,cov=ce.score_signatures(ex,{'single_balance':{'up':sig['single_JUN'],'down':sig['single_VIM']}},standardize_scores=True)
    scores.to_csv(B/'tables/clinical_exact_pair_scores.csv');cov.to_csv(B/'tables/clinical_signature_gene_coverage.csv',index=False);cl.to_csv(B/'tables/clinical_prepared_PAAD_source.csv')
    ductal=cl.histological_type.astype(str).str.strip().eq('Pancreas-Adenocarcinoma Ductal Type')
    results=[];kms=[];curves=[];risk=[];used=[]
    for name,mask in [('full_primary',pd.Series(True,index=cl.index)),('confirmed_ductal',ductal)]:
        out=ce.evaluate_scores(scores,cl.loc[mask],cohort=name,endpoints=['OS','PFI'],standardize_scores=False,output_dir=B/f'tables/clinical_{name}')
        results.append(out['cox']);kms.append(out['km_tests']);curves.append(out['km_curves'])
        for endpoint in ['OS','PFI']:
            data=ce._endpoint_frame(cl.loc[mask],scores,endpoint)
            for split in ['median','quartiles']:
                lo=scores.single_balance.quantile(.5 if split=='median' else .25);hi=lo if split=='median' else scores.single_balance.quantile(.75)
                d=data.copy() if split=='median' else data[(data.single_balance<=lo)|(data.single_balance>=hi)].copy()
                d['group']=np.where(d.single_balance>hi if split=='median' else d.single_balance>=hi,'high','low')
                for group,g in d.groupby('group'):
                    for t in [0,365,730,1095,1460,1825]:risk.append({'cohort':name,'endpoint':endpoint,'split':split,'group':group,'time_days':t,'n_at_risk':int(g.time.ge(t).sum()),'n_events_before':int(g.loc[g.time.lt(t),'event'].sum())})
            data['cohort']=name;data['endpoint']=endpoint;data['complete_age_sex_stage']=data.age.notna()&data.sex.isin(['MALE','FEMALE'])&data.stage.isin(['I','II','III','IV'])
            used.append(data)
    cox=pd.concat(results,ignore_index=True);cox['q_BH_across_two_cohorts_two_endpoints']=np.nan
    for model,ix in cox.groupby('model').groups.items():cox.loc[ix,'q_BH_across_two_cohorts_two_endpoints']=fdr(cox.loc[ix,'p'])
    km=pd.concat(kms,ignore_index=True);km['q_BH_across_cohorts_endpoints']=np.nan
    for split,ix in km.groupby('split').groups.items():km.loc[ix,'q_BH_across_cohorts_endpoints']=fdr(km.loc[ix,'p'])
    cox.to_csv(B/'tables/clinical_exact_pair_Cox_OS_PFI.csv',index=False);km.to_csv(B/'tables/clinical_exact_pair_KM_logrank.csv',index=False);pd.concat(curves).to_csv(B/'tables/clinical_exact_pair_KM_curves.csv',index=False);pd.DataFrame(risk).to_csv(B/'tables/clinical_exact_pair_KM_numbers_at_risk.csv',index=False);pd.concat(used).to_csv(B/'tables/clinical_exact_endpoint_model_source.csv')
    js({'n_full_molecular_primary':len(cl),'n_confirmed_ductal_molecular':int(ductal.sum()),'histology_rule':'TCGA-CDR histological_type exactly Pancreas-Adenocarcinoma Ductal Type','signature_source':'only saved DEGs of the exact selected single JUN cluster versus exact selected single VIM cluster, not pooled/target-vs-rest signatures','cluster_pair':{s:pair[s] for s in ['JUN','VIM']},'scoring':'shared clinical_eval.score_signatures signed mean within-full-PAAD gene-z; up-arm JUN minus down-arm VIM; final full-cohort SD; same frozen full-cohort scale for ductal sensitivity','Cox':'clinical_eval.evaluate_scores, age + recorded sex + categorical stage, complete cases; HR per full-cohort score SD; OS and TCGA-CDR PFI','KM':'shared evaluator median and outer quartiles fixed on all full molecular cohort scores, unchanged for ductal subset; no optimized cutpoint; explicit number-at-risk source table','BH':'four adjusted OS/PFI x full/ductal balance tests jointly; unadjusted separate; KM four cohort-endpoint tests within split'},'tables/clinical_contract.json')
    print('CLINICAL',cox[cox.model.eq('adjusted')].to_string(index=False),flush=True)

def spatial(sig,pair,meta):
    manifest=pd.read_csv(C/'tables/04-spatial-validation-optional/03-tangram-reference/spatial_sample_scope_manifest.csv')
    manifest['actual_used_h5ad_path']=[str(C/'h5ad/04-spatial-validation-optional/02-build-spatial-h5ad'/Path(p).name) for p in manifest.output_h5ad]
    manifest.to_csv(B/'tables/spatial_14section_source_manifest.csv',index=False)
    rows=[];coverage=[];qc=[];rng=np.random.default_rng(20261107)
    stromal=['COL1A1','COL1A2','COL3A1','DCN','LUM','FAP','COL11A1']
    for rr in manifest.itertuples():
        path=C/'h5ad/04-spatial-validation-optional/02-build-spatial-h5ad'/Path(rr.output_h5ad).name
        a=ad.read_h5ad(path);genes=sorted(set(sum(sig.values(),[])+PANELS['Ductal']+stromal)&set(a.var_names));gi=pd.Index(genes)
        x=a[:,genes].X;x=x.toarray() if sparse.issparse(x) else np.asarray(x)
        z=(x-x.mean(axis=0))/np.maximum(x.std(axis=0),.1)
        scs={name:np.mean(z[:,gi.get_indexer([g for g in gs if g in gi])],axis=1) for name,gs in sig.items()}
        for name,gs in sig.items():coverage.append({'section':rr.sample,'signature':name,'n_requested':len(gs),'n_present':sum(g in gi for g in gs),'coverage':sum(g in gi for g in gs)/len(gs)})
        scs['balance']=scs['single_JUN']-scs['single_VIM']
        epi=np.mean(z[:,gi.get_indexer([g for g in PANELS['Ductal'] if g in gi])],axis=1);stroma=np.mean(z[:,gi.get_indexer([g for g in stromal if g in gi])],axis=1)
        total=a.obs.total_counts.to_numpy() if 'total_counts' in a.obs else np.asarray(a.layers['counts'].sum(axis=1)).ravel()
        spots=pd.DataFrame(scs,index=a.obs_names);spots['epithelial_proxy']=epi;spots['stromal_proxy']=stroma;spots['total_counts']=total;spots['section']=rr.sample;spots['patient']=rr.patient_id
        if 'spatial' in a.obsm:spots[['x','y']]=a.obsm['spatial'][:,:2]
        spots.to_csv(B/f'tables/spatial_exact_pair_spot_scores_{rr.sample}.csv.gz',index_label='spot_id')
        for quality in ['counts_ge500','all_source_tissue_spots']:
            keep=total>=500 if quality=='counts_ge500' else np.ones(a.n_obs,bool)
            qc.append({'section':rr.sample,'patient':rr.patient_id,'quality':quality,'n_source_spots':a.n_obs,'n_used_spots':int(keep.sum())})
            covs=stats.rankdata(np.column_stack([epi[keep],stroma[keep],np.log1p(total[keep])]),axis=0);design=np.column_stack([np.ones(keep.sum()),covs])
            for metric in ['single_JUN','single_VIM','balance']:
                for adj in ['none','epithelial_stromal_library']:
                    yy=stats.rankdata(np.column_stack([scs[metric][keep],scs['CM1'][keep]]),axis=0)
                    if adj!='none':yy-=design@np.linalg.lstsq(design,yy,rcond=None)[0]
                    r=np.corrcoef(yy.T)[0,1]
                    rows.append({'section':rr.sample,'patient':rr.patient_id,'quality':quality,'metric':metric,'adjustment':adj,'rho':r,'n_spots':int(keep.sum()),'JUN_cluster':pair['JUN'],'VIM_cluster':pair['VIM']})
        print('SPATIAL SECTION',rr.sample,a.n_obs,flush=True)
    pd.DataFrame(coverage).to_csv(B/'tables/spatial_exact_pair_signature_coverage.csv',index=False);pd.DataFrame(qc).to_csv(B/'tables/spatial_spot_QC.csv',index=False)
    sec=pd.DataFrame(rows);sec.to_csv(B/'tables/spatial_exact_pair_section_associations.csv',index=False)
    pp=sec.groupby(['patient','quality','metric','adjustment']).rho.apply(lambda z:np.tanh(np.arctanh(np.clip(z,-.999999,.999999)).mean())).reset_index()
    pp.to_csv(B/'tables/spatial_exact_pair_patient_effects.csv',index=False)
    overlap=set(meta.loc[meta.series.eq('GSE278688'),'patient_id'].astype(str))&set(pp.patient.astype(str))
    tests=[]
    for scope,use in [('all12_patients',np.ones(len(pp),bool)),('without_recorded_sc_overlap',~pp.patient.isin(overlap))]:
        for key,g in pp.loc[use].groupby(['quality','metric','adjustment']):
            vals=np.arctanh(np.clip(g.rho.to_numpy(),-.999999,.999999));n=len(vals)
            flips=np.array(list(itertools.product([-1,1],repeat=n)))
            p=np.mean(np.abs((flips*vals).mean(axis=1))>=abs(vals.mean())-1e-12)
            boot=np.tanh(vals[rng.integers(0,n,size=(10000,n))].mean(axis=1))
            tests.append({'scope':scope,'quality':key[0],'metric':key[1],'adjustment':key[2],'patient_equal_weight_r':np.tanh(vals.mean()),'CI_low':np.quantile(boot,.025),'CI_high':np.quantile(boot,.975),'patient_signflip_p':p,'n_patients':n,'n_sections':sec[sec.patient.isin(g.patient)].section.nunique(),'positive_patients':int((vals>0).sum()),'n_spots':int(sec[(sec.quality==key[0])&sec.patient.isin(g.patient)].drop_duplicates('section').n_spots.sum())})
    out=pd.DataFrame(tests);out['q_BH_six_tests_within_scope_quality']=np.nan
    for _,ix in out.groupby(['scope','quality']).groups.items():out.loc[ix,'q_BH_six_tests_within_scope_quality']=fdr(out.loc[ix,'patient_signflip_p'])
    out.to_csv(B/'tables/spatial_exact_pair_patient_signflip_inference.csv',index=False)
    js({'method':'new direct expression scoring of the exact single-cluster donor-paired DEG signatures; no historical pooled mapping, no Tangram atom approximation','score':'mean within-section per-gene normalized-log z score; JUN minus VIM arm means; CM1 current H-weighted non-epithelial differential signature; 50 genes per arm if >=50 eligible','primary_QC':'source filtered in-tissue spots with >=500 UMIs; all source tissue spots sensitivity; all14 sections retained','adjustment':'partial rank correlation controlling fixed ductal epithelial proxy, stromal proxy, log library size','inference':'within-section local balance-CM1 partial Spearman; within-patient mean Fisher z across sections; equal patient weight across12 patients; exact two-sided patient sign-flip (2^12), 10000 patient bootstraps','FDR':'six metric x adjustment tests per scope/spot-QC; no spot-level p-values','recorded_sc_overlap':sorted(overlap),'limitation':'expression proxies in mixed spots, not calibrated epithelial fractions; no spatial-autocorrelation-independent spot inference; sign-flip assumes symmetric independent patient effects'},'tables/spatial_contract.json')
    print('SPATIAL META',out[(out.scope=='all12_patients')&(out.quality=='counts_ge500')&(out.metric=='balance')].to_string(index=False),flush=True)

def main():
    pair=json.loads((B/'tables/selected_single_cluster_pair.json').read_text());cm=json.loads((B/'tables/cm_calibration_contract.json').read_text())
    if not pair['valid_pair']:
        js({'reason':'same cluster wins both states; no distinct pair or pair-derived DEG clinical signature is fabricated','pair':pair},'tables/no_valid_distinct_pair.json');return
    meta=singlecell(pair,cm);sig=transfer_signatures(pair,cm);clinical(sig,pair)
    print('SINGLE-CELL AND CLINICAL COMPLETE; spatial is user-descoped and existing exploratory outputs are not recomputed.',flush=True)
if __name__=='__main__':main()
