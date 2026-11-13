#!/usr/bin/env python3
"""Read original CRC atlas; save cleaned data and expression-supported QC, originals untouched."""
from common import *
import json, gc

def main():
 with h5py.File(ATLAS,'r') as f:
  obs=read_frame(f['obs'],['sample','series','study_id','patient_id','status','cell_subtype','cell_type','doublet_score','n_genes_by_counts','total_counts'])
  genes=read_col(f['raw/var'][f['raw/var'].attrs['_index']])
  print('Loading raw normalized log1p matrix',flush=True)
  x=load_csr(f['raw/X'])
 print('Loaded',x.shape,'nnz',x.nnz,flush=True)
 scores,raw=gene_scores(x,genes);scores.index=raw.index=obs.index
 with h5py.File(XEN,'r') as f:panel=set(read_col(f['var'][f['var'].attrs['_index']]))
 used=sorted(set(g for gs in PROGRAMS.values() for g in gs if g in genes)|set(genes).intersection(panel))
 idx={g:i for i,g in enumerate(genes)}
 small=x[:,[idx[g] for g in used]].toarray()
 # Expression and detection QC at the original cluster level.
 exp=pd.DataFrame(small,index=obs.index,columns=used).groupby(obs.cell_subtype).mean()
 det=pd.DataFrame(small>0,index=obs.index,columns=used).groupby(obs.cell_subtype).mean()
 exp.to_csv(ROOT/'qc_subtype_mean_log_expression.csv');det.to_csv(ROOT/'qc_subtype_detection_fraction.csv')
 raw.groupby(obs.cell_subtype).mean().to_csv(ROOT/'qc_subtype_program_raw_means.csv')
 epi=obs.cell_subtype.str.startswith('Epi_')
 lineage=obs.cell_subtype.str.split('_').str[0]
 own=lineage.map({'Epi':'Epithelial','T':'Tcell','B':'Bcell','Plasma':'Plasma','Mye':'Myeloid','Endo':'Endothelial','Stromal':'Stromal','Mast':'Mast','Schwann':'Schwann','Cycling':'Leukocyte'})
 ownraw=np.zeros(len(obs))
 for k in own.unique():
  ii=own.eq(k).to_numpy();ownraw[ii]=raw[k].to_numpy()[ii]
 # Named cross-lineage clusters are excluded conservatively, never reclassified.
 bad_cluster=obs.cell_subtype.str.endswith(('_PHGR1','_PIGR','_KRT8'))|obs.cell_subtype.isin(['Epi_PTPRC','B_CD3D','Mye_CD3D','Mast_CD3D','Plasma_CD3D','Schwann_TYROBP'])
 epi_core=['EPCAM','KRT8','KRT18','KRT19','CDH1','CEACAM5','PHGR1','VIL1']
 immune_core=['PTPRC','LST1','TYROBP','FCER1G','CD3D','CD3E','MS4A1']
 epi_detect=(small[:,[used.index(g) for g in epi_core if g in used]]>0).sum(1)
 immune_detect=(small[:,[used.index(g) for g in immune_core if g in used]]>0).sum(1)
 bad_epi=epi.to_numpy()&(immune_detect>=3)&(raw.Leukocyte.to_numpy()>0.6*np.maximum(raw.Epithelial.to_numpy(),0.1))
 bad_non=(~epi.to_numpy())&(epi_detect>=3)&(raw.Epithelial.to_numpy()>np.maximum(0.8*ownraw,0.35))
 bad_other=(~epi.to_numpy())&(~lineage.isin(['T','Mye','Cycling']).to_numpy())&(raw.Tcell.to_numpy()>np.maximum(1.2*ownraw,0.7))
 obs['excluded_named_crosslineage_cluster']=bad_cluster
 obs['excluded_cell_immune_in_epi']=bad_epi
 obs['excluded_cell_epi_in_nonepi']=bad_non
 obs['excluded_cell_T_in_other']=bad_other
 obs['keep_clean']=~(bad_cluster.to_numpy()|bad_epi|bad_non|bad_other)
 obs['lineage']=lineage
 obs['n_epi_markers_detected']=epi_detect;obs['n_immune_markers_detected']=immune_detect
 for k in ['Epithelial','Leukocyte','Stromal','Tcell','T_NK','Myeloid']:obs['raw_'+k]=raw[k]
 # T/NK refinement on a program manifold, based only on T/NK-intrinsic expression.
 tmask=obs.keep_clean&lineage.eq('T')
 tcols=['T_GZMB','T_GZMK','T_TOX_CXCL13','T_NKcyto','T_Naive','T_Treg']
 ts=scores.loc[tmask,tcols].to_numpy();tlabel=np.asarray(tcols)[np.argmax(ts,axis=1)]
 obs['cm_subtype']=obs.cell_subtype
 obs.loc[tmask,'cm_subtype']=tlabel
 obs.to_csv(ROOT/'clean_cell_metadata.csv.gz',index_label='cell_id')
 scores.to_csv(ROOT/'cell_program_scores.csv.gz',index_label='cell_id')
 raw.to_csv(ROOT/'cell_program_rawmeans.csv.gz',index_label='cell_id')
 qc=obs.groupby('cell_subtype').agg(n=('sample','size'),n_kept=('keep_clean','sum'),named_exclusion=('excluded_named_crosslineage_cluster','max'),immune_in_epi=('excluded_cell_immune_in_epi','sum'),epi_in_non_epi=('excluded_cell_epi_in_nonepi','sum'),T_in_other=('excluded_cell_T_in_other','sum'))
 qc.to_csv(ROOT/'qc_excluded_cells_by_subtype.csv')
 epimask=epi&obs.keep_clean
 eobs=obs.loc[epimask].copy();ex=x[epimask.to_numpy()]
 sparse.save_npz(ROOT/'clean_epithelial_log1p.npz',ex,compressed=False)
 pd.Series(genes,name='gene').to_csv(ROOT/'raw_gene_names.csv',index=False)
 eobs.to_csv(ROOT/'clean_epithelial_metadata.csv.gz',index_label='cell_id')
 # Tumor enrichment means enrichment over compartment-normalized normal epithelium,
 # not tumor status alone. No claim of CNV-proven malignancy is made.
 counts=pd.crosstab(eobs.cell_subtype,eobs.status)
 freq=counts/counts.sum(axis=0)
 counts['tumor_over_normal_frequency']=freq['tumor']/freq['normal'].replace(0,np.nan)
 counts['enriched_for_A']=counts.tumor_over_normal_frequency.ge(0.8)&counts.tumor.ge(100)
 counts['exclude_normal_dominant']=counts.tumor_over_normal_frequency.lt(0.2)
 counts.to_csv(ROOT/'clean_epithelial_tumor_enrichment.csv')
 # Pseudobulk-derived subtype profiles for independent CM module signatures.
 good=obs.keep_clean&~epi
 cleanmeans=pd.DataFrame(small[good],index=obs.index[good],columns=used).groupby(obs.loc[good,'cm_subtype']).mean()
 cleanmeans.to_csv(ROOT/'clean_cm_subtype_marker_means.csv')
 px=small[:,[used.index(g) for g in sorted(panel.intersection(genes))]]
 px=np.expm1(px);px=np.log1p(px/np.maximum(px.sum(1,keepdims=True),1e-6)*1e4)
 pd.DataFrame(px[good],index=obs.index[good],columns=sorted(panel.intersection(genes))).groupby(obs.loc[good,'cm_subtype']).mean().to_csv(ROOT/'clean_cm_subtype_panel_log1p_means.csv')
 pd.DataFrame(px[epimask],index=eobs.index,columns=sorted(panel.intersection(genes))).to_pickle(ROOT/'clean_epithelial_panel_log1p.pkl')
 cohort=obs[['sample','series','study_id','patient_id','status']].drop_duplicates('sample').set_index('sample')
 ec=obs[obs.keep_clean&epi].groupby('sample').size();nc=obs[obs.keep_clean&~epi].groupby('sample').size()
 cohort['n_epi']=ec.reindex(cohort.index).fillna(0).astype(int);cohort['n_non_epi']=nc.reindex(cohort.index).fillna(0).astype(int)
 for threshold in [1,20,50]:cohort[f'eligible_epi{threshold}_non50']=(cohort.n_epi>=threshold)&(cohort.n_non_epi>=50)
 cohort.to_csv(ROOT/'sample_eligibility_clean.csv')
 summary={'n_atlas_cells':len(obs),'n_clean_cells':int(obs.keep_clean.sum()),'n_excluded':int((~obs.keep_clean).sum()),'n_clean_epi':len(eobs),'tumor_samples_eligible':{str(t):int((cohort.status.eq('tumor')&cohort[f'eligible_epi{t}_non50']).sum()) for t in [1,20,50]},'original_epi_minimum':1,'clean_epi_QC':'exclude Epi_PTPRC and cells expressing >=3 immune core markers with immune raw mean >0.6 x epithelial mean; EMT genes alone never trigger exclusion','non_epi_QC':'exclude PHGR1/PIGR/KRT8-named clusters; CD3D in B/myeloid/mast/plasma; Schwann_TYROBP; cell-level epithelial coexpression and T-in-other flags','malignancy_limitation':'tumor-enriched epithelial proxies, not inferCNV-proven malignant states','matrix':'raw.X is normalized log1p, not counts; no second normalization'}
 (ROOT/'cleaning_summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True);print(counts.to_string(),flush=True)

if __name__=='__main__':main()
