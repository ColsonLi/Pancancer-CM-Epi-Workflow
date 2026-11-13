from analysis_common import *
from scipy.spatial.distance import cdist
OUT=ROOT/'reference';OUT.mkdir(exist_ok=True)
TARGETS=['Epi_JUN','Epi_VIM']
def main():
 with h5py.File(KIRC) as f:
  obs=read_frame(f['obs'],['sample','series','status','cell_subtype'])
  genes=read_col(f['raw/var'][f['raw/var'].attrs['_index']]);x=load_csr(f['raw/X']).astype('float32')
  assert x.shape==(len(obs),len(genes)) and len(np.unique(genes))==len(genes)
 obs['donor_proxy']=obs['sample'].astype(str)
 obs.to_csv(OUT/'original_ccrcc_cell_labels.csv.gz',index_label='cell_id')
 pb,pm=pseudobulk(x,obs,['donor_proxy','cell_subtype'],min_cells=20)
 dx=x.copy();dx.data=np.ones_like(dx.data);det,dm=pseudobulk(dx,obs,['donor_proxy','cell_subtype'],min_cells=20);del dx
 assert pm.equals(dm)
 prof=pd.DataFrame(pb,columns=genes).groupby(pm.cell_subtype).mean()
 detection=pd.DataFrame(det,columns=genes).groupby(pm.cell_subtype).mean()
 mu=prof.mean(0).to_numpy()
 second=pd.DataFrame(pb.astype('float64')**2,columns=genes).groupby(pm.cell_subtype).mean().mean(0).to_numpy()
 sd=np.maximum(np.sqrt(np.maximum(second-mu**2,0)),.05)
 scale=pd.DataFrame({'gene':genes,'reference_mean':mu,'reference_sd':sd})
 scale.to_csv(OUT/'ccrcc_only_gene_scaling.csv',index=False)
 prof.to_csv(OUT/'ccrcc_donor_balanced_state_profiles_log1p.csv.gz',index_label='state')
 detection.to_csv(OUT/'ccrcc_donor_balanced_state_detection.csv.gz',index_label='state')
 np.savez_compressed(OUT/'ccrcc_pseudobulk_log1p.npz',expression=pb,detection=det,genes=genes)
 pm.to_csv(OUT/'ccrcc_pseudobulk_metadata.csv',index=False)
 summary=obs.groupby('cell_subtype').agg(n_cells=('sample','size'),n_samples=('sample','nunique'),n_series=('series','nunique'),tumor_fraction=('status',lambda z:z.eq('tumor').mean()))
 summary['n_pseudobulks_ge20']=pm.cell_subtype.value_counts();summary.to_csv(OUT/'ccrcc_original_state_inventory.csv')
 markers=[]
 for st in TARGETS:
  delta=prof.loc[st].to_numpy()-prof.drop(st).mean(0).to_numpy()
  m=pd.DataFrame({'state':st,'gene':genes,'mean_log1p_target':prof.loc[st].to_numpy(),'mean_log1p_other_states':prof.drop(st).mean(0).to_numpy(),'delta_log1p':delta,'standardized_delta':delta/sd,'target_detection':detection.loc[st].to_numpy()})
  m['eligible']=(m.delta_log1p>=.1)&(m.target_detection>=.1)&~m.gene.str.match(GENE_FILTER)
  m=m.sort_values(['eligible','standardized_delta','gene'],ascending=[False,False,True])
  m['reference_only_rank']=np.where(m.eligible,np.cumsum(m.eligible),np.nan)
  markers.append(m)
 markers=pd.concat(markers,ignore_index=True)
 markers.to_csv(OUT/'ccrcc_only_marker_ranking_all_genes.csv.gz',index=False)
 top100=markers.loc[markers.eligible&markers.reference_only_rank.le(100)].copy()
 top300=markers.loc[markers.eligible&markers.reference_only_rank.le(300)].copy()
 assert set(top100.state)==set(TARGETS)
 top100.to_csv(OUT/'frozen_top100_per_state.csv',index=False);top300.to_csv(OUT/'frozen_top300_per_state.csv',index=False)
 normal=[s for s in prof.index if any(g in s for g in ['ALDOB','AQP2','GPX3','MIOX'])]
 kidney=(prof.loc[normal].mean(0)-prof.loc[TARGETS].max(0)>=.25) if normal else pd.Series(False,index=genes)
 kidney.rename('kidney_specific_reference_only').to_csv(OUT/'kidney_specificity_filter.csv',index_label='gene')
 # Verify published original DEG direction against the actual original labels.
 audit=[]
 for st in TARGETS:
  for typ,path in [('one_vs_rest',V/'shared_clinical_and_signatures/reference/epithelial_degs'/f'{st}_degs_epi.csv'),('pairwise',PAIR/f'{st}_degs_vs_{TARGETS[1-TARGETS.index(st)]}.csv')]:
   tab=pd.read_csv(path); t=tab.loc[tab.logfoldchanges.gt(0)].head(100)
   for _,r in t.iterrows():
    g=r.gene
    if g not in prof:continue
    other=prof.drop(st)[g].mean() if typ=='one_vs_rest' else prof.loc[TARGETS[1-TARGETS.index(st)],g]
    audit.append({'state':st,'source_type':typ,'source_file':str(path),'gene':g,'source_logfoldchange':r.logfoldchanges,'donor_balanced_delta_log1p':prof.loc[st,g]-other,'direction_agrees':prof.loc[st,g]>other,'ribosomal_or_mito':bool(pd.Series([g]).str.match(GENE_FILTER).iloc[0])})
 pd.DataFrame(audit).to_csv(OUT/'original_DEG_direction_audit.csv',index=False)
 # Canonical gene audit uses all cells plus donor-balanced profiles, no replacement.
 canonical=[g for g in ['JUN','FOS','JUNB','KLF6','ZBTB20','VIM','EPCAM','KRT8','KRT18','CDH1','PTPRC','COL1A1','FN1','NDUFA4L2','CA9','CA12','MIOX','ALDOB'] if g in prof]
 prof[canonical].to_csv(OUT/'canonical_gene_expression_by_original_state.csv')
 # Self-match is a reconstruction check, never independent validation.
 common=sorted(set(top100.gene)); idx=pd.Index(genes).get_indexer(common)
 z=(prof[common].to_numpy()-mu[idx])/sd[idx]
 sim=1-cdist(z,z,metric='cosine')
 pd.DataFrame(sim,index=prof.index,columns=prof.index).to_csv(OUT/'reference_self_similarity_not_validation.csv')
 rng=np.random.default_rng(42);v=x[np.sort(rng.choice(len(obs),min(500,len(obs)),False))].copy();v.data=np.expm1(v.data)
 manifest={'source':str(KIRC),'scale':'original raw.X natural-log1p; X is scaled and is not used','n_cells':len(obs),'n_genes':len(genes),'n_states':len(prof),'state_labels':prof.index.tolist(),'donor_proxy':'sample; patient ID unavailable in original epithelial object','pseudobulk_min_cells':20,'top100_counts':top100.groupby('state').size().to_dict(),'top100_union_n':len(set(top100.gene)),'top300_union_n':len(set(top300.gene)),'normal_renal_labels_for_sensitivity':normal,'kidney_specific_n':int(kidney.sum()),'gene_sd_floor':.05,'expm1_row_sum_quantiles':np.quantile(np.asarray(v.sum(1)).ravel(),[0,.25,.5,.75,1]).tolist(),'crc_cluster_identity_read':False,'rank_conditioned_signatures_used':False,'marker_scarcity_rule':'If fewer than 100/300 eligible genes exist, retain all without relaxing delta/detection filters. VIM has few donor-balanced positive one-versus-rest markers. These are independently reconstructed, not the prohibited rank-selected signatures.','freeze_time_utc':pd.Timestamp.now(tz='UTC').isoformat()}
 savejson(OUT/'reference_manifest.json',manifest)
 print(json.dumps(manifest,indent=2),flush=True)
 print('Top20 markers',top100.groupby('state').head(20)[['state','gene','delta_log1p','standardized_delta']].to_string(index=False),flush=True)
 print('Original DEG audit',pd.DataFrame(audit).groupby(['state','source_type']).agg(n=('gene','size'),agree=('direction_agrees','mean'),generic=('ribosomal_or_mito','sum')).to_string(),flush=True)
if __name__=='__main__':main()
