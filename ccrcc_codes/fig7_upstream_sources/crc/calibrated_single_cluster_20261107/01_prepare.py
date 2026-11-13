from cal_common import *

def main():
 d=ROOT/'data';d.mkdir(parents=True,exist_ok=True)
 old=pd.read_csv(OLD/'clean_cell_metadata.csv.gz',index_col=0,low_memory=False)
 obs=pd.read_csv(FA/'all_cell_labels.csv.gz',index_col=0,low_memory=False)
 assert old.index.equals(obs.index)
 with h5py.File(ATLAS) as f:
  genes=read_col(f['raw/var'][f['raw/var'].attrs['_index']]);x=load_csr(f['raw/X']).astype('float32')
 pd.Series(genes,name='gene').to_csv(d/'genes.csv',index=False)
 print('original expression',x.shape,x.nnz,flush=True)
 programs={
  'epithelial':['EPCAM','KRT8','KRT18','KRT19','CDH1','CEACAM5','PHGR1','VIL1'],
  'immune':['PTPRC','CD3D','CD3E','TRAC','LST1','TYROBP','FCER1G','MS4A1','CD79A'],
  'plasma':['MZB1','JCHAIN','DERL3','TNFRSF17'],
  'goblet':['MUC2','TFF3','CLCA1'],
  'enteroendocrine':['CHGA','CHGB','NEUROD1','INSM1'],
  'Paneth':['LYZ','DEFA5','DEFA6'],
  'tuft':['POU2F3','TRPM5','DCLK1','AVIL'],
  'T_NK':['CD3D','CD3E','TRAC','CD247','CD2','NKG7','GNLY','GZMK','TOX','CXCL13','KLRD1']}
 used=sorted(set(g for v in programs.values() for g in v if g in genes))
 small=x[:,pd.Index(genes).get_indexer(used)].toarray()
 qc=pd.DataFrame(index=obs.index)
 for tag,gs in programs.items():
  a=small[:,[used.index(g) for g in gs if g in used]];qc[tag+'_mean']=a.mean(1);qc[tag+'_detected']=(a>0).sum(1)
 qc['PTPRC_log1p']=small[:,used.index('PTPRC')]
 epi=obs.broad_lineage_v1.eq('Epithelial').to_numpy()
 # Coexpression, not VIM/EMT expression, identifies mixed profiles. PTPRC
 # alone is insufficient: it must co-occur with another immune marker.
 immune_mix=(qc.epithelial_detected>=2)&(qc.PTPRC_log1p>0)&(qc.immune_detected>=2)&(qc.immune_mean>=.25)
 immune_mix|=(qc.epithelial_detected>=2)&(qc.immune_detected>=3)&(qc.immune_mean>.6*qc.epithelial_mean.clip(lower=.1))
 plasma_mix=(qc.epithelial_detected>=2)&(qc.plasma_detected>=2)&(qc.plasma_mean>=.5)&(qc.plasma_mean>.5*qc.epithelial_mean.clip(lower=.1))
 qc['exclude_immune_coexpression']=immune_mix&epi
 qc['exclude_epithelial_plasma_coexpression']=plasma_mix&epi
 qc['exclude_prior_reviewed_crosslineage']=epi&~old.keep_clean.to_numpy()
 qc['exclude_insufficient_epithelial_evidence']=epi&((qc.epithelial_detected<2).to_numpy()|(qc.epithelial_mean<.25).to_numpy())
 # The full-atlas broad label is not itself proof of epithelial identity.
 # Reapply the existing explicit mixed-cluster review and require two real
 # epithelial markers; otherwise plasma-only/immune-only low-complexity rows
 # evade a doublet rule that itself requires epithelial coexpression.
 clean=epi&~immune_mix.to_numpy()&~plasma_mix.to_numpy()&old.keep_clean.to_numpy()&~qc.exclude_insufficient_epithelial_evidence.to_numpy()
 # Normal samples never enter tumor HVG/PCA. In tumor-source tissue, remove
 # coexpression-supported specialized mucosa only in normal-dominant source
 # subtypes (normal/tumor compartment frequency ratio >5). Goblet-like tumor
 # states not in those subtypes are retained, avoiding blanket MUC2 removal.
 tab=pd.crosstab(obs.loc[clean,'cell_subtype'],obs.loc[clean,'status']);freq=tab.div(tab.sum(0),axis=1)
 nd=(freq.tumor/freq.normal.replace(0,np.nan)).lt(.2)
 tab['tumor_normal_frequency_ratio']=freq.tumor/freq.normal.replace(0,np.nan);tab['normal_dominant']=nd
 tab.to_csv(d/'source_epithelial_subtype_status_enrichment.csv')
 specialized=np.full(len(obs),'unspecialized',dtype=object)
 for tag in ['goblet','enteroendocrine','Paneth','tuft']:
  keep=(qc[tag+'_detected']>=2)&(qc[tag+'_mean']>=.5);specialized[keep]=tag
 qc['specialized_mucosal_lineage']=specialized
 normal=clean&obs.status.eq('normal').to_numpy()
 normal_like=clean&obs.status.eq('tumor').to_numpy()&obs.cell_subtype.isin(nd.index[nd]).to_numpy()&(specialized!='unspecialized')
 qc['exclude_tumor_source_specialized_normal_dominant']=normal_like
 qc['included_tumor_expression']=clean&obs.status.eq('tumor').to_numpy()&~normal_like
 qc['normal_reference']=normal
 qc=obs[['sample','sample_id','study_id','series','donor_id_v1','patient_id','status','cell_subtype','broad_lineage_v1']].join(qc)
 qc.loc[epi].to_csv(d/'epithelial_per_cell_coexpression_QC.csv.gz',index_label='cell_id')
 qc.loc[epi].groupby(['status','cell_subtype']).agg(n_cells=('sample','size'),immune_mixed=('exclude_immune_coexpression','sum'),plasma_mixed=('exclude_epithelial_plasma_coexpression','sum'),specialized_normal=('exclude_tumor_source_specialized_normal_dominant','sum'),retained_tumor=('included_tumor_expression','sum')).to_csv(d/'epithelial_exclusion_summary.csv')
 eo=obs.loc[qc.included_tumor_expression].copy();eo['sample_id']=eo['sample'];eo['donor_id']=eo.donor_id_v1
 sparse.save_npz(d/'tumor_epithelial_log1p.npz',x[qc.included_tumor_expression.to_numpy()],compressed=False)
 eo.to_csv(d/'tumor_epithelial_metadata.csv.gz',index_label='cell_id')
 nr=qc.loc[normal].copy();nr.to_csv(d/'normal_epithelial_reference_labels.csv.gz',index_label='cell_id')
 pb,pm=pseudobulk(x[normal],obs.loc[normal].assign(specialized=specialized[normal]),['donor_id_v1','study_id','specialized'],min_cells=10)
 np.savez_compressed(d/'normal_reference_pseudobulk.npz',expression=pb,genes=genes);pm.to_csv(d/'normal_reference_pseudobulk_metadata.csv',index=False)
 # The prior 60 cleaned subtypes are expression-supported source labels,
 # except the six program-archetype T labels. Replace those six with the
 # already saved 11 direct-expression T/NK clusters; do NOT reuse score labels.
 t=pd.read_csv(FA/'lineages/T_NK/per_cell_labels.csv.gz',index_col=0,low_memory=False)
 nonmask=old.keep_clean&old.lineage.ne('Epi')&obs.broad_lineage_v1.ne('Epithelial')
 inconsistent=nonmask&old.lineage.eq('T')&~old.index.isin(t.index)
 nonmask&=~inconsistent
 no=obs.loc[nonmask,['sample','sample_id','series','study_id','patient_id','donor_id_v1','status']].copy();no['subtype_id']=old.loc[no.index,'cell_subtype'];no['lineage']=old.loc[no.index,'lineage']
 common=no.index.intersection(t.index);no.loc[common,'subtype_id']=t.loc[common,'cluster_v1'];no.loc[common,'lineage']='T_NK'
 no['sample_id']=no['sample'];no['donor_id']=no.donor_id_v1
 no.to_csv(d/'non_epithelial_per_cell_labels.csv.gz',index_label='cell_id')
 counts=pd.crosstab(no['sample'],no.subtype_id)
 meta=obs[['sample','series','study_id','patient_id','donor_id_v1','status']].drop_duplicates('sample').set_index('sample');counts=counts.reindex(meta.index,fill_value=0)
 meta['n_non_epi']=counts.sum(1);meta['n_tumor_epi']=eo.groupby('sample').size().reindex(meta.index).fillna(0).astype(int)
 meta.to_csv(d/'sample_counts.csv',index_label='sample_id');counts.to_csv(d/'non_epithelial_sample_subtype_counts.csv',index_label='sample_id')
 populations=[]
 for n in [30,50]:
  t=meta.loc[meta.status.eq('tumor')&meta.n_tumor_epi.ge(n)&meta.n_non_epi.ge(50)];populations.append(dict(min_epithelial=n,min_non_epithelial=50,n_samples=len(t),n_patients=t.donor_id_v1.nunique(),n_studies=t.study_id.nunique(),n_epithelial_cells=int(t.n_tumor_epi.sum()),n_non_epithelial_cells=int(t.n_non_epi.sum())))
 pd.DataFrame(populations).to_csv(d/'primary_threshold_population_counts.csv',index=False)
 # Donor-balanced markers support annotation and CM composition, not EPI
 # clustering. Retain all genes in these profiles and in DE testing.
 nx=x[nonmask.to_numpy()];pb,pm=pseudobulk(nx,no,['donor_id','study_id','subtype_id'],min_cells=10)
 prof=balanced_profile(pb,pm,label='subtype_id');prof.columns=genes;prof.to_csv(d/'non_epithelial_balanced_profiles.csv.gz',index_label='subtype_id')
 bg=prof.mean(0);markers=[]
 for sub,row in prof.iterrows():
  delta=row-bg;ok=(row>=.15)&~pd.Series(genes,index=genes).str.match(GENE_FILTER)
  for rank,(g,v) in enumerate(delta.loc[ok].sort_values(ascending=False).head(50).items(),1):markers.append(dict(cluster=sub,gene=g,rank=rank,delta=v,mean_log1p=row[g]))
 pd.DataFrame(markers).to_csv(d/'non_epithelial_top50_markers.csv',index=False)
 anno=no.groupby('subtype_id').agg(n_cells=('sample','size'),n_samples=('sample','nunique'),lineage=('lineage','first'))
 anno['top10_markers']=pd.DataFrame(markers).groupby('cluster').head(10).groupby('cluster').gene.agg(';'.join)
 anno.to_csv(d/'non_epithelial_subtype_annotations.csv')
 # Save the measured-panel expression once for later held-out exact-label
 # classification, without using old epithelial assignments or old scores.
 panel=pd.read_csv(OLD/'spatial/panel_genes.csv').gene.tolist();shared=sorted(set(panel)&set(genes));ix=pd.Index(genes).get_indexer(shared)
 for tag,mask in [('tumor_epi',qc.included_tumor_expression.to_numpy()),('non_epi',nonmask.to_numpy())]:
  z=x[mask][:,ix].copy();z.data=np.expm1(z.data);s=np.asarray(z.sum(1)).ravel();z=sparse.diags(1e4/np.maximum(s,1e-8))@z;z=z.tocsr();z.data=np.log1p(z.data)
  sparse.save_npz(d/(tag+'_panel_log1p.npz'),z,compressed=False)
 pd.Series(shared,name='gene').to_csv(d/'shared_panel_genes.csv',index=False)
 summary={'original_cells':len(obs),'epithelial_broad_cells':int(epi.sum()),'immune_mixed_removed':int((epi&immune_mix.to_numpy()).sum()),'plasma_mixed_removed':int((epi&plasma_mix.to_numpy()).sum()),'overlap_mixed':int((epi&immune_mix.to_numpy()&plasma_mix.to_numpy()).sum()),'prior_reviewed_crosslineage_removed':int(qc.exclude_prior_reviewed_crosslineage.sum()),'insufficient_epithelial_evidence_removed':int(qc.exclude_insufficient_epithelial_evidence.sum()),'normal_reference_cells':int(normal.sum()),'tumor_source_specialized_normal_dominant_removed':int(normal_like.sum()),'tumor_epithelial_cells':len(eo),'tumor_epithelial_samples':eo['sample'].nunique(),'tumor_epithelial_patients':eo.donor_id.nunique(),'tumor_epithelial_studies':eo.study_id.nunique(),'clean_non_epithelial_cells':len(no),'non_epithelial_subtypes':no.subtype_id.nunique(),'inconsistent_T_lineage_excluded':int(inconsistent.sum()),'non_epithelial_source':'prior cell-level/named cross-lineage cleaning; original expression subtype labels, with all six score-archetype T labels replaced by 11 saved direct-expression T/NK clusters','expression_scale':'source raw.X normalized natural-log1p; not integer counts; not renormalized for expression graph','HVG_scope':'clean tumor-source epithelial only; every normal epithelial cell excluded from HVG/PCA; normal-dominant specialized mucosa coexpression excluded in tumor-source samples','normal_specialized_counts':pd.Series(specialized[normal]).value_counts().to_dict(),'epithelial_QC_revision':'pre-downstream expression QC review: remove previously reviewed mixed source clusters and cells with <2 epithelial markers or epithelial mean<.25; HVG sizes/resolutions/reference genes/scaling/CM rank unchanged. Initial graph and similarity diagnostics archived in pre_QC_review, never used for downstream endpoints.','prior_outputs_modified':False}
 savejson(d/'preparation_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=6):main()
