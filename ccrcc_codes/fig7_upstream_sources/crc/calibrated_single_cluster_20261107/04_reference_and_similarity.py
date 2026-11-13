from cal_common import *

def donor_mean(pb,pm,label):
 out=balanced_profile(pb,pm,label=label);return out

def resampled_centroids(pb,pm,labels,rng,donor='donor_id',label='cluster_id'):
 mult={}
 for study,g in pm.groupby('study_id'):
  ds=g[donor].unique();draw=rng.choice(ds,len(ds),True);u,n=np.unique(draw,return_counts=True);mult.update(dict(zip([(study,s) for s in u],n)))
 w=np.array([mult.get((s,d),0) for s,d in zip(pm.study_id,pm[donor])],float)
 # Cache an exact donor-to-(cluster,study) summation matrix. This replaces
 # repeated pandas groupbys only; donor draws and equal-study means are
 # numerically identical to the explicit loop implementation.
 key=(tuple(labels),donor,label)
 cache=pm.attrs.setdefault('_resampling_aggregation',{})
 if key not in cache:
  groups=[(k,ids) for k,ids in pm.groupby([label,'study_id'],sort=True).indices.items() if k[0] in labels]
  rr=np.concatenate([np.full(len(ids),i) for i,(_,ids) in enumerate(groups)]);cc=np.concatenate([ids for _,ids in groups]);D=sparse.csr_matrix((np.ones(len(cc)),(rr,cc)),shape=(len(groups),len(pm)));group_labels=np.array([labels.index(k[0]) for k,_ in groups]);cache[key]=(D,group_labels)
 D,group_labels=cache[key];den=np.asarray(D@w).ravel();valid=den>0;means=(D@(pb*w[:,None]))[valid]/den[valid,None];out=np.zeros((len(labels),pb.shape[1]));np.add.at(out,group_labels[valid],means);n=np.bincount(group_labels[valid],minlength=len(labels));out=out/np.maximum(n[:,None],1);out[n==0]=np.nan
 return out

def main():
 d=ROOT/'similarity';d.mkdir(exist_ok=True)
 pbfile=np.load(ER/'ccrcc_pseudobulk_log1p.npz',allow_pickle=True);rpb=pbfile['expression'];rdet=pbfile['detection'];rg=pbfile['genes'].astype(str)
 rm=pd.read_csv(ER/'ccrcc_pseudobulk_metadata.csv');ro=pd.read_csv(ER/'original_ccrcc_cell_labels.csv.gz',index_col=0);lookup=ro.drop_duplicates('sample').set_index('sample').series
 rm['study_id']=rm.donor_proxy.map(lookup);rm['donor_id']=rm.donor_proxy;rm['cluster_id']=rm.cell_subtype
 rp=balanced_profile(rpb,rm);rp.columns=rg;det=balanced_profile(rdet,rm);det.columns=rg
 second=balanced_profile(rpb.astype(float)**2,rm);second.columns=rg;mu=rp.mean(0);sd=np.sqrt((second.mean(0)-mu**2).clip(lower=0)).clip(lower=.05)
 pd.DataFrame({'mean':mu,'sd':sd}).to_csv(d/'ccRCC_only_gene_scaling.csv',index_label='gene');rp.to_csv(d/'ccRCC_balanced_state_centroids.csv.gz',index_label='state')
 kidney=pd.read_csv(ER/'kidney_specificity_filter.csv',index_col=0).iloc[:,0].astype(bool)
 pairroot=Path('/mnt/disk18t/lr_xcy/riku/kirc_205/leiden_detailed/epi_vim_vs_jun_pairwise_deg/results')
 panels={'archived_core12x2':CORE};rows=[]
 paired={}
 for state,other in [('Epi_JUN','Epi_VIM'),('Epi_VIM','Epi_JUN')]:
  t=pd.read_csv(pairroot/f'{state}_degs_vs_{other}.csv');t['positive_significant_source']=t.logfoldchanges.gt(0)&t.pvals_adj.lt(.05);t['ribosomal_mito']=t.gene.str.match(GENE_FILTER);t['kidney_background']=t.gene.map(kidney).fillna(False)
  t['donor_balanced_delta']=t.gene.map(rp.loc[state]-rp.loc[other]);t['target_detection']=t.gene.map(det.loc[state]);t['qualified']=t.positive_significant_source&~t.ribosomal_mito&~t.kidney_background&t.donor_balanced_delta.ge(.1)&t.target_detection.ge(.1)
  paired[state]=t.loc[t.qualified,'gene'].drop_duplicates().head(100).tolist();t['selected_signature']=t.gene.isin(paired[state]);t['state']=state;rows.append(t)
 pd.concat(rows).to_csv(d/'paired_reference_DEG_filter_audit.csv',index=False);panels['paired_conserved_nonribo_nonkidney_top100']=paired
 savejson(d/'reference_signatures.json',panels)
 savejson(d/'reference_specification.json',{'source_ccRCC':str(ER/'reference_manifest.json'),'core_source':'/mnt/disk18t/lr_xcy/riku/kirc_205/final/fig7_positive_redesign_20261106/prepare_unified_panel_b.py lines17-20 (read-only archived genes; not old CRC assignments/scores)','primary_definition':'archived_core12x2','paired_definition':'original paired ccRCC DEG source order; positive adjusted p<.05; exclude mitochondrial/ribosomal prefixes and ccRCC normal-kidney background; require positive >=.1 donor/study-balanced log1p delta and >=.1 detection, first100 per arm; no CRC, spatial or outcome-dependent gene choice','metric':'uniform cosine on the shared union of both state programs after ccRCC-only gene standardization; same centroids/scaling/metric for every cluster','profiles':'donor balanced within study, then equal study mean; every observed CRC donor/cluster retained (minimum1) so ALL clusters have a centroid; ccRCC preexisting minimum20 per donor/state; exact-pair clinical DEG pseudobulks require10 cells','bootstrap':'200 independent donor-within-study resamples of CRC AND ccRCC reference centroids; scaling frozen; all clusters/all configurations','LOSO':'each CRC study omitted in turn with reference frozen; all clusters and both definitions','gate':'>=500 cells AND >=15 tumor samples; report unconditional and eligible ranks separately','primary_configuration':'hvg3000_r0.4; frozen before reference matching'})
 x,obs,cg=load_epi();alllabels=pd.read_csv(ROOT/'epithelial/all_configuration_per_cell_labels.csv.gz',index_col=0);assert obs.index.equals(alllabels.index)
 union=sorted(set(g for p in panels.values() for gs in p.values() for g in gs)&set(cg)&set(rg));ci=pd.Index(cg).get_indexer(union);ri=pd.Index(rg).get_indexer(union);sx=x[:,ci]
 np.savez_compressed(d/'reference_pseudobulk_used.npz',expression=rpb[:,ri],genes=np.array(union));rm.to_csv(d/'reference_pseudobulk_metadata.csv',index=False)
 coverage=[]
 for panel,p in panels.items():
  for state,gs in p.items():coverage.append(dict(definition=panel,state=state,n_requested=len(gs),n_present_CRC=sum(g in cg for g in gs),n_present_panel=sum(g in pd.read_csv(ROOT/'data/shared_panel_genes.csv').gene.tolist() for g in gs),missing_CRC=';'.join(g for g in gs if g not in cg)))
 pd.DataFrame(coverage).to_csv(d/'signature_coverage.csv',index=False)
 simrows=[];bootrows=[];losorows=[];markerrows=[];rng=np.random.default_rng(SEED)
 for config in alllabels:
  out=d/config;out.mkdir(exist_ok=True);o=obs[['sample_id','study_id','donor_id']].copy();o['cluster_id']=alllabels[config].to_numpy()
  pb,pm=pseudobulk(sx,o,['donor_id','study_id','cluster_id'],min_cells=1);prof=balanced_profile(pb,pm);prof.columns=union;labels=prof.index.tolist()
  prof.to_csv(out/'balanced_centroids_reference_genes.csv',index_label='cluster_id');np.savez_compressed(out/'donor_cluster_pseudobulk.npz',expression=pb,genes=np.array(union));pm.to_csv(out/'donor_cluster_pseudobulk_metadata.csv',index=False)
  support=pd.read_csv(ROOT/f'epithelial/support_{config}.csv',index_col=0).reindex(labels);gate=support.candidate_gate.to_numpy(bool)
  fp,fm=pseudobulk(x,o,['donor_id','study_id','cluster_id'],min_cells=1);fullprof=balanced_profile(fp,fm);fullprof.columns=cg;fullprof.to_csv(out/'all_cluster_fullgene_balanced_centroids.csv.gz',index_label='cluster_id')
  for cl,row in fullprof.iterrows():
   delta=row-fullprof.drop(cl).mean(0);valid=row.ge(.15)&~pd.Series(cg,index=cg).str.match(GENE_FILTER)
   for rank,(g,v) in enumerate(delta.loc[valid].sort_values(ascending=False).head(20).items(),1):markerrows.append(dict(configuration=config,cluster_id=cl,rank=rank,gene=g,delta_log1p=v,mean_target=row[g]))
  del fp,fm,fullprof;gc.collect()
  base={};boot={}
  for panel,p in panels.items():
   gs=sorted(set(g for v in p.values() for g in v)&set(union));ii=pd.Index(union).get_indexer(gs);m=mu.loc[gs].to_numpy();s=sd.loc[gs].to_numpy();ref=rp.loc[['Epi_JUN','Epi_VIM'],gs].to_numpy();sim=cos((prof[gs].to_numpy()-m)/s,(ref-m)/s);base[panel]=(ii,m,s);boot[panel]=[]
   for j,state in enumerate(['Epi_JUN','Epi_VIM']):
    ranks=stats.rankdata(-sim[:,j],method='min');erank=np.full(len(labels),np.nan);erank[gate]=stats.rankdata(-sim[gate,j],method='min')
    for i,cl in enumerate(labels):simrows.append(dict(configuration=config,definition=panel,cluster_id=cl,reference_state=state,similarity=sim[i,j],rank_all=int(ranks[i]),rank_candidates=erank[i],is_maximum_all=ranks[i]==1,is_maximum_candidate=erank[i]==1,n_shared_union=len(gs),**support.loc[cl].to_dict()))
  rsub=rpb[:,ri];refstates=['Epi_JUN','Epi_VIM']
  for b in range(200):
   cp=resampled_centroids(pb,pm,labels,rng);rr=resampled_centroids(rsub,rm,refstates,rng)
   for panel,(ii,m,s) in base.items():
    sim=cos((cp[:,ii]-m)/s,(rr[:,ii]-m)/s);boot[panel].append(sim)
    for j,state in enumerate(refstates):
     ranks=stats.rankdata(-sim[:,j],method='min',nan_policy='omit');erank=np.full(len(labels),np.nan);erank[gate]=stats.rankdata(-sim[gate,j],method='min',nan_policy='omit')
     for i,cl in enumerate(labels):bootrows.append(dict(configuration=config,definition=panel,replicate=b,cluster_id=cl,reference_state=state,similarity=sim[i,j],rank_all=ranks[i],rank_candidates=erank[i],winner_candidate=erank[i]==1))
  for study in sorted(pm.study_id.unique()):
   keep=pm.study_id.ne(study);p=balanced_profile(pb[keep],pm.loc[keep].reset_index(drop=True)).reindex(labels);p.columns=union
   for panel,(ii,m,s) in base.items():
    gs=np.array(union)[ii];ref=rp.loc[refstates,gs].to_numpy();sim=cos((p.iloc[:,ii].to_numpy()-m)/s,(ref-m)/s)
    for j,state in enumerate(refstates):
     ranks=stats.rankdata(-sim[:,j],method='min',nan_policy='omit');erank=np.full(len(labels),np.nan);erank[gate]=stats.rankdata(-sim[gate,j],method='min',nan_policy='omit')
     for i,cl in enumerate(labels):losorows.append(dict(configuration=config,definition=panel,omitted_study=study,cluster_id=cl,reference_state=state,similarity=sim[i,j],rank_all=ranks[i],rank_candidates=erank[i]))
  print('all-cluster matching + uncertainty',config,len(labels),flush=True)
 sim=pd.DataFrame(simrows);bt=pd.DataFrame(bootrows);ls=pd.DataFrame(losorows);bt.to_csv(d/'all_configuration_bootstrap_draws.csv.gz',index=False);ls.to_csv(d/'all_configuration_leave_study_out.csv',index=False)
 keys=['configuration','definition','cluster_id','reference_state']
 summ=bt.groupby(keys).agg(bootstrap_similarity_low=('similarity',lambda z:z.quantile(.025)),bootstrap_similarity_high=('similarity',lambda z:z.quantile(.975)),bootstrap_rank_low=('rank_candidates',lambda z:z.quantile(.025)),bootstrap_rank_high=('rank_candidates',lambda z:z.quantile(.975)),bootstrap_win_fraction=('winner_candidate','mean')).reset_index()
 lsum=ls.groupby(keys).agg(LOSO_similarity_min=('similarity','min'),LOSO_similarity_max=('similarity','max'),LOSO_rank_min=('rank_candidates','min'),LOSO_rank_max=('rank_candidates','max'),LOSO_win_fraction=('rank_candidates',lambda z:np.mean(z==1))).reset_index()
 sim=sim.merge(summ,on=keys).merge(lsum,on=keys)
 markers=pd.DataFrame(markerrows);markers.to_csv(d/'all_configuration_expression_top20_markers.csv',index=False)
 markertext=markers.loc[markers['rank'].le(5)].groupby(['configuration','cluster_id']).gene.agg(';'.join).rename('top5_expression_markers').reset_index();sim=sim.merge(markertext,on=['configuration','cluster_id'],how='left');sim.to_csv(d/'ALL_CLUSTERS_ALL_CONFIGURATIONS_similarity_ranks.csv',index=False)
 config='hvg3000_r0.4';p=sim.loc[sim.configuration.eq(config)&sim.definition.eq('archived_core12x2')&sim.is_maximum_candidate]
 pair={r.reference_state:r.cluster_id for r in p.itertuples()};savejson(ROOT/'selected_pair.json',{'configuration':config,'primary_reference':'archived_core12x2','pair':pair,'same_cluster_wins_both':len(set(pair.values()))<2,'selection':'independent #1 cosine to each reference among prespecified supported candidates; no pooling; default expression configuration frozen before similarities/associations','exact_labels':str(ROOT/f'epithelial/labels_{config}.csv.gz'),'all_similarity_table':str(d/'ALL_CLUSTERS_ALL_CONFIGURATIONS_similarity_ranks.csv')})
 print('SELECTED SINGLE PAIR',pair,flush=True)
 q=pd.read_csv(ROOT/'data/epithelial_per_cell_coexpression_QC.csv.gz',index_col=0,low_memory=False).loc[obs.index];q['cluster_id']=alllabels[config].to_numpy();q.groupby('cluster_id').agg(n_cells=('sample','size'),n_samples=('sample','nunique'),n_studies=('study_id','nunique'),mean_epithelial_markers=('epithelial_detected','mean'),min_epithelial_markers=('epithelial_detected','min'),mean_epithelial_log1p=('epithelial_mean','mean'),mean_immune_log1p=('immune_mean','mean'),mean_plasma_log1p=('plasma_mean','mean'),immune_mixed=('exclude_immune_coexpression','sum'),plasma_mixed=('exclude_epithelial_plasma_coexpression','sum'),prior_mixed=('exclude_prior_reviewed_crosslineage','sum'),insufficient_epithelial=('exclude_insufficient_epithelial_evidence','sum')).to_csv(ROOT/'epithelial/final_cluster_coexpression_QC.csv')
 # Independent implementation: loop the selected cells by donor and study,
 # calculate centroids directly from the original sparse expression, then
 # manually dot/norm against the reference (not a reused cdist result).
 cl=pair['Epi_JUN'];gs=sorted(set(g for v in CORE.values() for g in v)&set(cg)&set(rg));ix=pd.Index(cg).get_indexer(gs);keep=alllabels[config].eq(cl);z=x[keep.to_numpy()][:,ix];mcheck=obs.loc[keep,['donor_id','study_id']].reset_index(drop=True);studymeans=[]
 for study,g in mcheck.groupby('study_id'):
  donors=[]
  for _,dg in g.groupby('donor_id'):
   if len(dg)>=1:donors.append(np.asarray(z[dg.index].mean(0)).ravel())
  if donors:studymeans.append(np.mean(donors,axis=0))
 cp=np.mean(studymeans,axis=0);az=(cp-mu.loc[gs].to_numpy())/sd.loc[gs].to_numpy();bz=(rp.loc['Epi_JUN',gs].to_numpy()-mu.loc[gs].to_numpy())/sd.loc[gs].to_numpy();ind=float(np.dot(az,bz)/(np.linalg.norm(az)*np.linalg.norm(bz)));reported=float(p.loc[p.reference_state.eq('Epi_JUN'),'similarity'].iloc[0]);savejson(d/'independent_similarity_check.json',{'cluster':cl,'definition':'archived_core12x2','recomputed_cosine':ind,'table_cosine':reported,'absolute_error':abs(ind-reported),'passed':abs(ind-reported)<1e-5})
 # Exact-pair donor-pseudobulk paired DEG direction is the ONLY clinical arm
 # source. No anchors or original pooled-state genes are appended.
 o=obs[['sample_id','study_id','donor_id']].copy();o['cluster_id']=alllabels[config].to_numpy();keep=o.cluster_id.isin(pair.values()).to_numpy();pb,pm=pseudobulk(x[keep],o.loc[keep],['donor_id','study_id','cluster_id'],min_cells=10);pr=balanced_profile(pb,pm);pr.columns=cg
 pr.to_csv(d/'selected_pair_fullgene_centroids.csv.gz',index_label='cluster_id');np.savez_compressed(d/'selected_pair_fullgene_donor_pseudobulk.npz',expression=pb,genes=cg);pm.to_csv(d/'selected_pair_fullgene_donor_pseudobulk_metadata.csv',index=False)
 dx=x[keep].copy();dx.data=np.ones_like(dx.data);detpb,dm=pseudobulk(dx,o.loc[keep],['donor_id','study_id','cluster_id'],min_cells=10);de=balanced_profile(detpb,dm);de.columns=cg
 all_degs=[];sigs={}
 for state,other in [('Epi_JUN','Epi_VIM'),('Epi_VIM','Epi_JUN')]:
  cl=pair[state];oc=pair[other];a=pb[pm.cluster_id.eq(cl)];b=pb[pm.cluster_id.eq(oc)];delta=pr.loc[cl]-pr.loc[oc];sdg=np.std(np.r_[a,b],axis=0,ddof=1).clip(.05)
  # Donor-level Welch test is descriptive (study structure is preserved in
  # balancing/uncertainty); genes ordered by balanced standardized delta.
  tt=stats.ttest_ind(a,b,axis=0,equal_var=False);q=bh(tt.pvalue)
  t=pd.DataFrame({'state':state,'cluster_id':cl,'other_cluster_id':oc,'gene':cg,'mean_target':pr.loc[cl].to_numpy(),'mean_other':pr.loc[oc].to_numpy(),'delta_log1p':delta.to_numpy(),'standardized_delta':delta.to_numpy()/sdg,'detection_target':de.loc[cl].to_numpy(),'detection_other':de.loc[oc].to_numpy(),'p_donor_Welch_descriptive':tt.pvalue,'q_donor_Welch_descriptive':q})
  t['eligible']=t.delta_log1p.ge(.1)&t.detection_target.ge(.1)&t.q_donor_Welch_descriptive.lt(.05)&~t.gene.str.match(GENE_FILTER)&~t.gene.str.match(r'^(IGH|IGK|IGL)')
  t=t.sort_values(['eligible','standardized_delta','gene'],ascending=[False,False,True]);t['positive_rank']=np.where(t.eligible,np.cumsum(t.eligible),np.nan);sigs[state]=t.loc[t.eligible,'gene'].head(50).tolist();all_degs.append(t)
 pd.concat(all_degs).to_csv(d/'selected_exact_pair_DEGs_all_genes.csv.gz',index=False);savejson(ROOT/'clinical_signature_exact_pair.json',{'balance':{'up':sigs['Epi_JUN'],'down':sigs['Epi_VIM']}})
 print(p[['reference_state','cluster_id','similarity','rank_all','rank_candidates','bootstrap_win_fraction','LOSO_win_fraction']].to_string(index=False),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=4):main()
