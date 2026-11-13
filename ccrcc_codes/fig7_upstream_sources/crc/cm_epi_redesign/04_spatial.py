#!/usr/bin/env python3
"""Direct Xenium projection with cancer-cell-restricted epithelial scores and local non-epi CMs.

The original Tangram mapping is not reused. Bins, not individual cells, are the
local units; nine patients, not 17 dependent sections, are inferential units.
"""
from common import *
import json,argparse,gc
from scipy.special import softmax
from scipy.optimize import nnls

def prepare():
 d=ROOT/'spatial';d.mkdir(exist_ok=True)
 scope=pd.read_csv(FIG7/'spatial_region_scope.csv')
 with h5py.File(XEN,'r') as f:
  obs=read_frame(f['obs'],['name','patient_id','tissue_region','celltype','celltype_sub'])
  ids=obs.name.isin(scope.section).to_numpy()
  coords=f['obsm/spatial'][:][ids]
  genes=read_col(f['var'][f['var'].attrs['_index']])
  x=load_csr(f['X'])[ids]
 # Xenium X is log1p normalized to 135, NOT raw counts. Undo log1p,
 # re-normalize the common panel to 10k, and re-log to match the SC panel reference.
 x.data=np.expm1(x.data);s=np.asarray(x.sum(1)).ravel();x=sparse.diags(1e4/np.maximum(s,1e-6))@x;x=x.tocsr();x.data=np.log1p(x.data).astype(np.float32)
 sparse.save_npz(d/'xenium_17regions_panel_log1p.npz',x,compressed=False)
 o=obs.loc[ids].copy();o['spatial_x']=coords[:,0];o['spatial_y']=coords[:,1]
 o.to_csv(d/'xenium_17regions_metadata.csv.gz',index_label='cell_id')
 pd.Series(genes,name='gene').to_csv(d/'panel_genes.csv',index=False)
 counts=pd.crosstab(o.name,o.celltype);counts.to_csv(d/'section_celltype_counts.csv')
 coverage=[]
 for k,gs in PROGRAMS.items():
  found=[g for g in gs if g in genes];coverage.append({'program':k,'n_requested':len(gs),'n_panel':len(found),'genes':';'.join(found)})
 pd.DataFrame(coverage).to_csv(d/'fixed_program_panel_coverage.csv',index=False)
 print('Spatial prepared',x.shape,counts.sum(0).to_dict(),flush=True)

def project():
 d=ROOT/'spatial';d.mkdir(exist_ok=True)
 obs=pd.read_csv(d/'xenium_17regions_metadata.csv.gz',index_col=0)
 x=sparse.load_npz(d/'xenium_17regions_panel_log1p.npz')
 genes=pd.read_csv(d/'panel_genes.csv').gene.to_numpy();geneidx={g:i for i,g in enumerate(genes)}
 eo=pd.read_csv(ROOT/'clean_epithelial_metadata.csv.gz',index_col=0)
 epanel=pd.read_pickle(ROOT/'clean_epithelial_panel_log1p.pkl')
 assignments=np.load(ROOT/'epithelial_state_assignments.npz')
 names=assignments.files
 tumor=eo.status.eq('tumor').to_numpy()
 # Standardization is learned on tumor epithelium/reference and on annotated
 # Cancer cells/target, never on local immune activity or patient outcomes.
 malignant=obs.celltype.eq('Cancer cell').to_numpy();em=epanel.loc[tumor].mean().to_numpy();es=epanel.loc[tumor].std().clip(lower=.2).to_numpy()
 gcommon=epanel.columns
 sx=x[malignant][:,[geneidx[g] for g in gcommon]]
 tm=np.asarray(sx.mean(0)).ravel();ts=np.sqrt(np.maximum(np.asarray(sx.power(2).mean(0)).ravel()-tm**2,.04))
 # A zero in the target cannot be used to distinguish unmeasured VIM markers.
 forbid=set(PROGRAMS['Leukocyte']+PROGRAMS['T_NK']+PROGRAMS['Myeloid']+PROGRAMS['Bcell']+PROGRAMS['Plasma']+PROGRAMS['Mast'])
 forbid|={'ACTB','ARPC3','ARPC5','HLA-B','C1QBP','TUBA1B','TUBB'}
 allow=np.asarray([g not in forbid for g in gcommon])
 stateprobs={};coverage=[];protorows=[]
 for name in names:
  a=assignments[name];pro=[];tags=[]
  for flag,tag in [(0,'other'),(1,'JUN'),(2,'VIM')]:
   if (a==flag).sum()<10:continue
   profile=epanel.loc[a==flag].mean().to_numpy();pro.append((profile-em)/es);tags.append(tag)
   row=dict(zip(gcommon,profile));row.update({'scheme':name,'state':tag});protorows.append(row)
  pro=np.asarray(pro)
  # Use 20 strongest transferable epithelial-intrinsic state-discriminating genes.
  contrast=np.ptp(pro,axis=0)*allow
  ids=np.argsort(-contrast)[:20];ids=ids[contrast[ids]>.05]
  p=pro[:,ids]
  if len(ids)<3:continue
  # Cosine-scaled nearest centroid logits, with equal priors and no local fitting.
  weights=p/ts[ids][None,:]/np.sqrt(len(ids))
  offsets=-(weights*tm[ids][None,:]).sum(1)-.5*(p*p).sum(1)/np.sqrt(len(ids))
  logits=sx[:,ids]@weights.T+offsets
  probs=softmax(logits,axis=1).astype(np.float32)
  stateprobs[name]=pd.DataFrame(probs,columns=tags)
  coverage.append({'scheme':name,'n_panel_genes':len(ids),'panel_genes':';'.join(gcommon[ids]),'n_prototype_states':len(tags),'JUN_mean_probability':float(probs[:,tags.index('JUN')].mean()),'VIM_mean_probability':float(probs[:,tags.index('VIM')].mean()),'projection':'equal-prior panel centroid softmax; expression restricted to annotated Cancer cells'})
 pd.DataFrame(coverage).to_csv(d/'epithelial_projection_coverage.csv',index=False)
 pd.DataFrame(protorows).to_csv(ROOT/'epithelial_state_panel_prototypes.csv',index=False)
 # Non-epithelial subtype projection is constrained to original Xenium broad labels.
 ref=pd.read_csv(ROOT/'clean_cm_subtype_panel_log1p_means.csv',index_col=0)
 freq=pd.read_csv(ROOT/'clean_non_epi_subtype_frequency.csv',index_col=0)
 subtypes=freq.columns.tolist();subidx={s:i for i,s in enumerate(subtypes)}
 non=~obs.celltype.isin(['Cancer cell','Epithelial cell']).to_numpy()
 non_ids=np.flatnonzero(non);global_to_non=np.full(len(obs),-1,dtype=np.int32);global_to_non[non_ids]=np.arange(len(non_ids))
 # Hard broad-lineage mapping avoids claiming immune/stromal expression is epithelial.
 ctmap={'T cell':['T_','Cycling_NKG7','Cycling_S100A4'], 'B cell':['B_','Cycling_TCL1A'],
        'Plasma cell':['Plasma_','Cycling_MZB1'],'Macrophage':['Mye_'],'Dendritic cell':['Mye_','Cycling_HLA-DRA'],
        'Neutrophil':['Mye_'],'Fibroblast':['Stromal_'],'Smooth muscle cell':['Stromal_'],
        'Endothelial cell':['Endo_'],'Mast cell':['Mast_'],'Schwann cell':['Schwann_']}
 # Soft phenotype proportions inside each broad lineage; retain uncertainty rather than nearest-label calls.
 probs_non=np.zeros((len(non_ids),len(subtypes)),dtype=np.float32);subq=[]
 for ct,prefixes in ctmap.items():
  ii=np.flatnonzero(obs.celltype.eq(ct).to_numpy());ii=ii[non[ii]]
  if len(ii)==0:continue
  candidates=[s for s in subtypes if any(s.startswith(p) for p in prefixes) and s in ref.index]
  if not candidates:continue
  rg=ref.loc[candidates];common=[g for g in rg.columns if g in geneidx]
  rg=rg[common]
  v=rg.var(0).sort_values(ascending=False)
  # Every T/NK archetype has measured discriminating markers on this panel.
  top=v.index[:30].tolist();gi=[geneidx[g] for g in top]
  rr=rg[top].to_numpy();rr=zscore(rr,axis=0)
  xx=x[ii][:,gi];mu=np.asarray(xx.mean(0)).ravel();sd=np.sqrt(np.maximum(np.asarray(xx.power(2).mean(0)).ravel()-mu**2,.04))
  weight=rr/sd[None,:]/np.sqrt(len(top));offset=-(weight*mu[None,:]).sum(1)-.5*(rr*rr).sum(1)/np.sqrt(len(top))
  pr=softmax(xx@weight.T+offset,axis=1).astype(np.float32)
  outids=global_to_non[ii]
  probs_non[np.ix_(outids,[subidx[s] for s in candidates])]=pr
  subq.append({'celltype':ct,'n':len(ii),'n_subtypes':len(candidates),'median_max_probability':float(np.median(pr.max(1))),'genes':';'.join(top)})
 pd.DataFrame(subq).to_csv(d/'non_epi_projection_quality.csv',index=False)
 cmmeta=pd.read_csv(ROOT/'cm_scheme_metadata.csv')
 # All frozen CM variants are evaluated, not only the best SC model.
 cms={}
 for r in cmmeta.itertuples():
  h=pd.read_csv(ROOT/'cm_models'/f'{r.cm_scheme}_H.csv',index_col=0)
  scale=pd.read_csv(ROOT/'cm_models'/f'{r.cm_scheme}_scaling.csv',index_col=0).reindex(subtypes)
  cms[r.cm_scheme]=(h.to_numpy(),scale,r.CM1like_component,r.CM10like_component,h.index.tolist())
 rows=[];paired=[];bin_manifest=[]
 # 100um primary; 50um and 150um are pre-specified scale sensitivity analyses.
 for size in [50,100,150]:
  for section in obs.name.unique():
   si=np.flatnonzero(obs.name.eq(section).to_numpy());so=obs.iloc[si]
   xy=so[['spatial_x','spatial_y']].to_numpy();binxy=np.floor((xy-xy.min(0))/size).astype(int)
   _,codes=np.unique(binxy,axis=0,return_inverse=True);nb=codes.max()+1
   binmat=sparse.csr_matrix((np.ones(len(si)),(codes,np.arange(len(si)))),shape=(nb,len(si)))
   cancer_local=malignant[si];non_local=non[si]
   nc=np.asarray(binmat@cancer_local.astype(float)).ravel();nn=np.asarray(binmat@non_local.astype(float)).ravel()
   keep=(nc>=20)&(nn>=20)
   if keep.sum()<20:
    print('Few bins',section,size,keep.sum(),flush=True)
   if keep.sum()<5:continue
   # Sparse membership matrix into compact cancer/non-cancer arrays.
   cancer_glob=si[cancer_local];malidx=np.searchsorted(np.flatnonzero(malignant),cancer_glob)
   bce=binmat[:,np.flatnonzero(cancer_local)]
   bn=binmat[:,np.flatnonzero(non_local)]
   f=bn@probs_non[global_to_non[si[non_local]]]
   f=f/np.maximum(f.sum(1,keepdims=True),1e-9)
   out=pd.DataFrame({'n_cancer':nc[keep],'n_non_epi':nn[keep]})
   tcount=binmat@so.celltype.eq('T cell').to_numpy().astype(float)
   mscount=binmat@so.celltype.isin(['Macrophage','Dendritic cell','Neutrophil','Fibroblast','Smooth muscle cell']).to_numpy().astype(float)
   localcms={'annotated_T_fraction':np.asarray(tcount/np.maximum(nn,1))[keep], 'annotated_myeloid_stromal_fraction':np.asarray(mscount/np.maximum(nn,1))[keep]}
   for name,(h,scale,tcm,mcm,cols) in cms.items():
    v=np.clip((f[keep]-scale.lo.to_numpy())/scale.scale.to_numpy(),0,1)
    # Multiplicative NNLS projection, shared for all bins; no re-fitting of H.
    w=np.maximum(v@h.T,1e-6);hh=h@h.T
    for _ in range(100):w*=np.maximum(v@h.T,1e-12)/np.maximum(w@hh,1e-12)
    w/=np.maximum(w.sum(1,keepdims=True),1e-9)
    localcms[name+'__CM1like']=w[:,cols.index(tcm)]
    localcms[name+'__CM10like']=w[:,cols.index(mcm)]
   for name,pr in stateprobs.items():
    pp=bce@pr.iloc[malidx].to_numpy();pp/=np.maximum(nc[:,None],1)
    jun=pp[keep,pr.columns.get_loc('JUN')];vim=pp[keep,pr.columns.get_loc('VIM')]
    bal=np.log((jun+.01)/(vim+.01))
    out[name+'__JUN']=jun;out[name+'__VIM']=vim;out[name+'__balance']=bal
    for cm,y in localcms.items():
     for feature,xx in [('JUN',jun),('VIM',vim),('balance',bal)]:
      r,p=corr(xx,y)
      low=xx<=np.quantile(xx,1/3);high=xx>=np.quantile(xx,2/3)
      yl=float(np.median(y[low]));yh=float(np.median(y[high]))
      rows.append({'scheme':name,'cm':cm,'bin_um':size,'section':section,'patient':so.patient_id.iloc[0],'region':so.tissue_region.iloc[0],'feature':feature,'n_bins':int(keep.sum()),'rho':r,'p_descriptive_bin':p,'activity_high':yh,'activity_low':yl,'paired_difference':yh-yl})
   for cm,y in localcms.items():out[cm]=y
   p=d/f'bins_{size}um_{section}.csv.gz';out.to_csv(p,index=False)
   bin_manifest.append({'section':section,'patient':so.patient_id.iloc[0],'bin_um':size,'n_bins_total':nb,'n_bins_eligible':int(keep.sum()),'file':str(p)})
   print('Spatial',size,section,'bins',keep.sum(),flush=True)
   del binmat,pp,out;gc.collect()
 reg=pd.DataFrame(rows);reg.to_csv(d/'section_correlations_all.csv',index=False)
 patient=reg.groupby(['scheme','cm','bin_um','feature','patient'],as_index=False).agg(mean_region_rho=('rho','mean'),mean_high=('activity_high','mean'),mean_low=('activity_low','mean'),paired_difference=('paired_difference','mean'),n_regions=('section','nunique'))
 patient.to_csv(d/'patient_correlations_and_pairs_all.csv',index=False)
 summary=[]
 for (name,cm,size,feat),g in patient.groupby(['scheme','cm','bin_um','feature']):
  summary.append({'scheme':name,'cm':cm,'bin_um':size,'feature':feat,'n_patients':len(g),'median_patient_rho':g.mean_region_rho.median(),'mean_patient_rho':g.mean_region_rho.mean(),'n_positive':int((g.mean_region_rho>0).sum()),'n_negative':int((g.mean_region_rho<0).sum()),'p_patient_rho_wilcoxon':signed_wilcoxon(g.mean_region_rho),'paired_activity_median_difference':g.paired_difference.median(),'p_paired_high_vs_low':signed_wilcoxon(g.paired_difference)})
 summary=pd.DataFrame(summary);summary['q_patient_rho_all']=bh(summary.p_patient_rho_wilcoxon);summary['q_paired_all']=bh(summary.p_paired_high_vs_low)
 summary['q_patient_rho_within_scale']=summary.groupby('bin_um').p_patient_rho_wilcoxon.transform(lambda v:bh(v))
 summary['q_paired_within_scale']=summary.groupby('bin_um').p_paired_high_vs_low.transform(lambda v:bh(v))
 summary['q_patient_rho_within_scale_feature']=summary.groupby(['bin_um','feature']).p_patient_rho_wilcoxon.transform(lambda v:bh(v))
 target=(summary.bin_um==100)&(summary.feature=='balance')&summary.cm.str.endswith('__CM1like')
 summary.loc[target,'q_primary_balance_CM1family']=bh(summary.loc[target,'p_patient_rho_wilcoxon'])
 summary.to_csv(d/'patient_summary_all.csv',index=False)
 pd.DataFrame(bin_manifest).to_csv(d/'bin_manifest.csv',index=False)
 (d/'methods.json').write_text(json.dumps({'source':'original Xenium log1p normalized panel, independent of saved Tangram','scope':'17 regions / 9 patients; annotated Cancer cells only for epithelial projection','primary_scale_um':100,'sensitivity_scales_um':[50,150],'bin_thresholds':{'Cancer_cells':20,'non_epithelial_cells':20},'projection':'SC-panel-normalized expression; equal-prior nearest-centroid softmax from frozen epithelial states. Non-epi subtype projection constrained by Xenium broad labels. NMF H fixed, 100-step nonnegative multiplicative projection.','inference':'Within-section Spearman on bins; average section rho per patient; two-sided Wilcoxon on nine patients. Paired median CM in top/bottom epithelial-feature thirds per section, equal-region patient average then paired Wilcoxon.','limitations':['Xenium panel lacks VIM, LGALS1/3, S100A4/6, NUPR1, TGFBI; matrix state is an indirect partial-panel projection.','Binary epithelial schemes mechanically force JUN/VIM opposition and are not independent opposing-state evidence.','All variants tested; all-variant FDR exported; not an independent scheme-selection cohort.','Bin-level p-values are descriptive because neighboring bins share tissue architecture.']},indent=2))
 print(summary[(summary.bin_um==100)&(summary.feature=='balance')].sort_values('median_patient_rho',ascending=False).head(15).to_string(index=False),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
 if a.prepare:prepare()
 else:project()
