from full_common import *
from expression_engine import integrate
from threadpoolctl import threadpool_limits
def main():
 x,obs,genes=load_original()
 obs.to_csv(ROOT/'original_cell_metadata.csv.gz',index_label='cell_id');pd.Series(genes,name='gene').to_csv(ROOT/'gene_order.csv',index=False)
 # Preserve reliable previous QC as review flags, without name-based deletion.
 old=pd.read_csv(ROOT.parent/'cm_epi_redesign/clean_cell_metadata.csv.gz',index_col=0,low_memory=False)
 for c in [c for c in old if c.startswith('excluded_') or c=='keep_clean']:obs['prior_'+c]=old.loc[obs.index,c]
 obs['included_full_atlas_v1']=True
 manifests=[]
 for name in ['01-celltype_integration_clustering/01-select-samples/02_eligible_samples_for_merge.csv','01-celltype_integration_clustering/02-merge-metadata/selected_sample_manifest.csv','01-celltype_integration_clustering/02-merge-metadata/merged_sample_manifest.csv']:
  path=WF/'tables'/name;table=pd.read_csv(path,low_memory=False);manifests.append({'path':str(path),'n_rows':len(table),'columns':table.columns.tolist()})
 audit={'input':str(ATLAS),'input_cells':len(obs),'retained_cells':len(obs),'samples':obs['sample'].nunique(),'patients':obs.donor_id_v1.nunique(),'biological_studies':obs.study_id.nunique(),'technical_sources':obs.series.nunique(),'status_cells':obs.status.value_counts().to_dict(),'status_samples':obs[['sample','status']].drop_duplicates().status.value_counts().to_dict(),'n_genes':len(genes),'nnz':int(x.nnz),'scale':'normalized natural-log1p raw.X, not counts; no second normalization','new_cell_exclusions':0,'sample_exclusions_for_low_epi':0,'eligibility_manifests':manifests}
 savejson(ROOT/'input_scope_audit.json',audit);print(json.dumps(audit,indent=2),flush=True)
 obs.groupby(['sample','study_id','series','patient_id','status'],dropna=False).size().rename('n_original_retained_cells').to_csv(ROOT/'all_sample_inclusion_counts.csv')
 labels,prof=integrate(x,obs,genes,ROOT/'atlas','CRCfa1A',cap=60000,global_scope=True)
 # Expression annotation after unsupervised fit, with original crosswalk.
 mean=prof.mean(0);sd=prof.std(0).clip(lower=.1);scores={}
 evidence=[]
 for lin,gs in LINEAGES.items():
  gs=[g for g in gs if g in prof];scores[lin]=((prof[gs]-mean[gs])/sd[gs]).mean(1)
  for c in prof.index:evidence.append({'atlas_cluster':c,'lineage':lin,'canonical_genes':';'.join(gs),'mean_log1p':prof.loc[c,gs].mean(),'n_canonical_mean_ge025':int(prof.loc[c,gs].ge(.25).sum()),'gene_z_score':scores[lin].loc[c]})
 scores=pd.DataFrame(scores);scores.to_csv(ROOT/'atlas/canonical_lineage_z_scores.csv');pd.DataFrame(evidence).to_csv(ROOT/'atlas/canonical_multi_marker_evidence.csv',index=False)
 original={'Epithelial Cells':'Epithelial','T Cells':'T_NK','NK Cells':'T_NK','Cycling Immune Cells':'T_NK','Myeloid Cells':'Myeloid','B Cells':'B','Plasma Cells':'Plasma','Stromal Cells':'Stromal','Endothelial Cells':'Endothelial','Mast Cells':'Mast','Schwann Cells':'Schwann'}
 labels['original_major_lineage']=labels.cell_type.map(original).fillna(labels.cell_type)
 cross=pd.crosstab(labels.cluster_v1,labels.original_major_lineage);cross.to_csv(ROOT/'atlas/original_major_lineage_crosswalk.csv')
 annotation=[]
 for c in scores.index:
  s=scores.loc[c].sort_values(ascending=False);best=s.index[0];original_max=cross.loc[c].idxmax();margin=s.iloc[0]-s.iloc[1]
  n=int(pd.DataFrame(evidence).query('atlas_cluster==@c and lineage==@best').n_canonical_mean_ge025.iloc[0])
  chosen=best if (margin>=.35 and n>=2) else original_max
  annotation.append({'atlas_cluster':c,'assigned_lineage':chosen,'canonical_best':best,'canonical_margin':margin,'original_majority':original_max,'original_majority_fraction':cross.loc[c].max()/cross.loc[c].sum(),'rule':'canonical best if margin>=0.35 and >=2 core genes mean>=0.25, else original-majority crosswalk'})
 annotation=pd.DataFrame(annotation);annotation.to_csv(ROOT/'atlas/broad_lineage_annotation.csv',index=False)
 mapping=annotation.set_index('atlas_cluster').assigned_lineage
 labels['broad_lineage_v1']=labels.cluster_v1.map(mapping)
 labels['broad_lineage_changed_from_original']=labels.broad_lineage_v1.ne(labels.original_major_lineage)
 labels.to_csv(ROOT/'all_cell_labels.csv.gz',index_label='cell_id');labels.to_csv(ROOT/'atlas/per_cell_labels.csv.gz',index_label='cell_id')
 labels.groupby('broad_lineage_v1').agg(n_cells=('sample','size'),n_samples=('sample','nunique'),n_patients=('donor_id_v1','nunique'),n_studies=('study_id','nunique'),tumor_fraction=('status',lambda z:z.eq('tumor').mean())).to_csv(ROOT/'broad_lineage_counts.csv')
 print(labels.broad_lineage_v1.value_counts().to_string(),flush=True)
 # Update full reusable object with the explicit annotation mapping.
 import anndata as ad
 a=ad.read_h5ad(ROOT/'atlas/all_cells_expression_integrated.h5ad');a.obs=labels;a.write_h5ad(ROOT/'atlas/all_cells_expression_integrated.h5ad',compression='gzip')
 savejson(ROOT/'full_atlas_completion.json',{'complete':True,'n_cells':len(labels),'n_samples':labels['sample'].nunique(),'n_lineages':labels.broad_lineage_v1.nunique()})
if __name__=='__main__':
 with threadpool_limits(limits=8):main()
