from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import nnls
import importlib.metadata

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'savefig.dpi':180,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def md(df):
    d=df.copy()
    def fmt(z):
        if isinstance(z,(float,np.floating)):return f'{z:.5g}' if np.isfinite(z) else 'NA'
        return str(z)
    return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(z) for z in r)+' |' for r in d.itertuples(index=False,name=None))
def save(fig,name):
    fig.savefig(B/f'figures/{name}.png',bbox_inches='tight',facecolor='white');fig.savefig(B/f'figures/{name}.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)

def verify(pair,cm):
    # Independent implementation: no shared donor_profiles/correlate helpers,
    # and no saved cluster centroid or saved sample fraction as the input.
    ep=ad.read_h5ad(OLD/'objects/cleaned_tumor_ductal.h5ad')
    lab=pd.read_csv(B/f'labels/{PRIMARY}.csv.gz',index_col=0).reindex(ep.obs_names)
    mdta=pd.read_csv(B/'tables/epithelial_input_cells.csv.gz',index_col=0).reindex(ep.obs_names)
    defs=json.loads((B/'tables/ccRCC_reference_definitions.json').read_text());gs=sorted(set(defs['archived_core12']['JUN']+defs['archived_core12']['VIM'])&set(ep.var_names))
    subset=lab.cluster_id.eq(pair['JUN']).to_numpy();allx=ep.X[:,ep.var_names.get_indexer(gs)]
    donorvals=[]
    for donor in mdta.loc[subset,'donor'].unique():
        ix=np.flatnonzero(subset&mdta.donor.eq(donor).to_numpy())
        if len(ix)>=5:donorvals.append(np.asarray(allx[ix].mean(axis=0)).ravel())
    target=np.mean(donorvals,axis=0)
    with h5py.File(KIRC,'r') as f:
        oo=read_elem(f['obs']);vv=read_elem(f['raw/var']);xx=sparse.csr_matrix(read_elem(f['raw/X']))[:,vv.index.get_indexer(gs)]
    donors=oo.series.astype(str)+'::'+oo['sample'].astype(str);rr=[]
    for donor in donors[oo.cell_subtype.eq('Epi_JUN')].unique():
        ix=np.flatnonzero(oo.cell_subtype.eq('Epi_JUN').to_numpy()&donors.eq(donor).to_numpy())
        if len(ix)>=5:rr.append(np.asarray(xx[ix].mean(axis=0)).ravel())
    reference=np.mean(rr,axis=0);scale=pd.read_csv(B/'tables/ccRCC_frozen_gene_standardization.csv',index_col=0).loc[gs]
    zt=(target-scale['mean'].to_numpy())/scale.sd.to_numpy();zr=(reference-scale['mean'].to_numpy())/scale.sd.to_numpy();sim=np.dot(zt,zr)/(np.linalg.norm(zt)*np.linalg.norm(zr))
    tab=pd.read_csv(B/'tables/primary_all_cluster_Sim_JUN_Sim_VIM_ranks.csv').set_index('cluster_id');saved=tab.loc[pair['JUN'],'Sim_JUN']
    # Rebuild fractions from the exact cell IDs and unweighted NNLS from H.
    meta=pd.read_csv(B/'tables/non_epi_sample_eligibility.csv',index_col=0)
    totals=lab.groupby('sample_id').size();jn=lab[lab.cluster_id.eq(pair['JUN'])].groupby('sample_id').size();vn=lab[lab.cluster_id.eq(pair['VIM'])].groupby('sample_id').size()
    counts=pd.read_csv(B/'tables/non_epi_54_sample_counts.csv',index_col=0);H=pd.read_csv(B/'tables/cm_selected_H.csv',index_col=0);ss=pd.read_csv(B/'tables/cm_column_minmax_parameters.csv',index_col=0)
    balance=[];activity=[]
    for sample in meta.index:
        n=int(totals.get(sample,0));nn=int(counts.loc[sample].sum())
        if meta.loc[sample,'status']!='tumor' or n<30 or nn<50:continue
        balance.append(np.log((jn.get(sample,0)/n+.005)/(vn.get(sample,0)/n+.005)))
        freq=counts.loc[sample].to_numpy()/nn;vec=np.maximum(0,(freq-ss.minimum.to_numpy())/ss['range'].to_numpy())
        activity.append(nnls(H.to_numpy().T,vec,maxiter=100*len(H))[0][H.index.get_loc(cm['CM1'])])
    rx=stats.rankdata(balance);ry=stats.rankdata(activity);r=np.dot(rx-rx.mean(),ry-ry.mean())/(np.linalg.norm(rx-rx.mean())*np.linalg.norm(ry-ry.mean()))
    p=2*stats.t.sf(abs(r)*np.sqrt((len(rx)-2)/(1-r*r)),len(rx)-2)
    assoc=pd.read_csv(B/'tables/singlecell_exact_pair_primary_sensitivity.csv');a=assoc[(assoc.unit=='sample')&(assoc.min_epi==30)&assoc.adjustment.eq('none')&assoc.metric.eq('balance')].iloc[0]
    out=pd.DataFrame([{'check':'independent raw-cell donor-mean JUN similarity','saved':saved,'independently_recomputed':sim,'absolute_difference':abs(saved-sim),'n_units':len(donorvals)},{'check':'independent raw-cell fractions + fresh NNLS Spearman balance','saved':a.rho,'independently_recomputed':r,'absolute_difference':abs(a.rho-r),'n_units':len(rx)},{'check':'independent Spearman p','saved':a.p,'independently_recomputed':p,'absolute_difference':abs(a.p-p),'n_units':len(rx)}])
    out.to_csv(B/'tables/independent_numerical_crosschecks.csv',index=False)
    assert abs(saved-sim)<2e-6 and abs(a.rho-r)<1e-10
    js({'crosschecks_passed':True,'max_abs_difference':out.absolute_difference.max(),'exact_pair':{s:pair[s] for s in ['JUN','VIM']},'all15_label_files_same_cell_set':all(pd.read_csv(B/f'labels/{c}.csv.gz',usecols=['cell_id']).cell_id.equals(pd.Series(ep.obs_names,name='cell_id')) for c in pd.read_csv(B/'tables/epithelial_configuration_manifest.csv').configuration),'original_files_modified':False},'tables/verified_output_summary.json')
    del xx,ep;gc.collect()

def overview(pair,cm):
    emb=ad.read_h5ad(B/'objects/primary_epithelial_embeddings.h5ad');sim=pd.read_csv(B/'tables/primary_all_cluster_Sim_JUN_Sim_VIM_ranks.csv')
    clusters=sorted(emb.obs.cluster_id.unique(),key=lambda z:int(z[1:]));colors={c:plt.get_cmap('tab20')(i%20) for i,c in enumerate(clusters)}
    fig,axs=plt.subplots(3,2,figsize=(16,16),layout='constrained',gridspec_kw={'height_ratios':[1.1,1,1]})
    ax=axs[0,0];xy=emb.obsm['X_umap'];rng=np.random.default_rng(42);order=rng.permutation(len(xy))
    ax.scatter(xy[order,0],xy[order,1],c=[colors[c] for c in emb.obs.cluster_id.iloc[order]],s=1.2,alpha=.65,rasterized=True,linewidths=0)
    for c in clusters:
        ix=emb.obs.cluster_id.eq(c).to_numpy();pos=np.median(xy[ix],axis=0)
        ax.text(*pos,c,fontsize=9,ha='center',va='center',bbox={'boxstyle':'round,pad=.15','fc':'white','alpha':.85,'ec':'black' if c in [pair['JUN'],pair['VIM']] else 'none','lw':1.2})
    ax.set(title=f'A  Direct expression clusters · {len(xy):,} tumor-ductal cells',xlabel='UMAP1',ylabel='UMAP2');ax.set_xticks([]);ax.set_yticks([])
    ax.text(.01,.01,f'JUN match {pair["JUN"]} · VIM match {pair["VIM"]}',transform=ax.transAxes,fontsize=10,bbox={'fc':'white','alpha':.8,'ec':'none'})
    ax=axs[0,1];ss=sim.set_index('cluster_id').reindex(clusters);mat=ss[['Sim_JUN','Sim_VIM']].to_numpy();im=ax.imshow(mat,cmap='coolwarm',vmin=-1,vmax=1,aspect='auto')
    ax.set_yticks(np.arange(len(ss)),[f'{c}  (n={int(ss.loc[c,"n_cells"]):,})'+(' †' if not ss.loc[c,'coverage_eligible'] else '') for c in ss.index],fontsize=9);ax.set_xticks([0,1],['Sim_JUN','Sim_VIM']);ax.set_title('B  Uniform ccRCC gene-z cosine · all clusters');ax.set_xlabel('★ eligible #1; † fails ≥500 cells or ≥10 tumor samples',fontsize=9)
    for i,c in enumerate(ss.index):
        for j,st in enumerate(['JUN','VIM']):
            text=f'{mat[i,j]:.3f}' + ('  ★ #1' if c==pair[st] else '')
            ax.text(j,i,text,ha='center',va='center',fontsize=9,color='white' if abs(mat[i,j])>.65 else 'black')
    fig.colorbar(im,ax=ax,shrink=.65,pad=.025,label='cosine');ax.spines[['top','right','bottom','left']].set_visible(False)
    ax=axs[1,0];F=pd.read_csv(B/'tables/cm_selected_H_fraction.csv',index_col=0).loc[cm['CM1']].nlargest(12).sort_values()
    ax.barh(F.index,F.values,color=['#326a9d' if z.startswith(('T_','NK_')) else '#b5bac1' for z in F.index]);ax.set(xlabel='Fraction of module H loading',title=f'C  {cm["CM1"]} composition · K={cm["selected_K"]}')
    ax.text(.99,.02,f'T/NK loading {100*cm["CM1_TNK_loading_fraction"]:.1f}%\n(below the 80% target)',transform=ax.transAxes,ha='right',va='bottom',fontsize=10,bbox={'fc':'white','alpha':.9,'ec':'none'});ax.legend(handles=[Patch(color='#326a9d',label='T/NK'),Patch(color='#b5bac1',label='Other lineage')],loc='center right',bbox_to_anchor=(1,.55),frameon=False,fontsize=9)
    ax=axs[1,1];mt=pd.read_csv(B/'tables/singlecell_exact_pair_sample_metrics.csv',index_col=0);mt=mt[mt.primary_eligible];st=sorted(mt.study.unique());pal={s:plt.get_cmap('tab10')(i) for i,s in enumerate(st)}
    for study,g in mt.groupby('study'):ax.scatter(g.balance,g.CM1_W,s=30,c=[pal[study]],label=str(study),alpha=.8,edgecolors='white',linewidths=.3)
    aa=pd.read_csv(B/'tables/singlecell_exact_pair_primary_sensitivity.csv');aa=aa[(aa.unit=='sample')&(aa.min_epi==30)&aa.metric.eq('balance')].set_index('adjustment');raw=aa.loc['none'];adj=aa.loc['study']
    ax.set(title=f'D  Single-cell balance vs CM1 · n={int(raw.n)} samples',xlabel='ln[(single JUN fraction + .005)/(single VIM fraction + .005)]',ylabel=f'{cm["CM1"]} unweighted NNLS activity')
    ax.text(.02,.98,f'Raw ρ={raw.rho:.3f}, p={raw.p:.3g}, q={raw.q_BH_within_unit_threshold_context:.3g}\nStudy-adjusted ρ={adj.rho:.3f}, p={adj.p:.3g}',transform=ax.transAxes,va='top',fontsize=10,bbox={'fc':'white','alpha':.85,'ec':'none'})
    ax.legend(fontsize=7,ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.20))
    ax=axs[2,0];pp=pd.read_csv(B/'tables/spatial_exact_pair_patient_effects.csv');pp=pp[(pp.quality=='counts_ge500')&pp.metric.eq('balance')&pp.adjustment.eq('epithelial_stromal_library')].sort_values('patient');sp=pd.read_csv(B/'tables/spatial_exact_pair_patient_signflip_inference.csv');sp=sp[(sp.scope=='all12_patients')&(sp.quality=='counts_ge500')&sp.metric.eq('balance')&sp.adjustment.eq('epithelial_stromal_library')].iloc[0]
    yy=np.arange(len(pp));ax.scatter(pp.rho,yy,c='#326a9d',s=35);ax.errorbar(sp.patient_equal_weight_r,len(pp)+.4,xerr=[[sp.patient_equal_weight_r-sp.CI_low],[sp.CI_high-sp.patient_equal_weight_r]],fmt='D',color='black',capsize=3)
    ax.set_yticks([*yy,len(pp)+.4],[*pp.patient,'Patient-equal pooled']);ax.axvline(0,c='grey',ls='--',lw=1);ax.invert_yaxis();ax.set(xlabel='Local balance–CM1 partial Spearman / Fisher-z mean')
    ax.set_title(f'E  Spatial · exact pair · 14 sections / 12 patients\nPooled r={sp.patient_equal_weight_r:.3f} [{sp.CI_low:.3f}, {sp.CI_high:.3f}]; patient p={sp.patient_signflip_p:.3g}, q={sp.q_BH_six_tests_within_scope_quality:.3g}',fontsize=11)
    ax=axs[2,1];co=pd.read_csv(B/'tables/clinical_exact_pair_Cox_OS_PFI.csv');co=co[co.model.eq('adjusted')].sort_values(['cohort','endpoint'],ascending=[False,True]);yy=np.arange(len(co))
    ax.errorbar(co.HR,yy,xerr=np.array([co.HR-co.CI_low,co.CI_high-co.HR]),fmt='o',color='#97515b',capsize=4)
    labels=[f'{"Full PAAD" if r.cohort=="full_primary" else "Ductal"} · {r.endpoint}\nn={int(r.n)}, events={int(r.events)}' for r in co.itertuples()]
    ax.set_yticks(yy,labels);ax.set_xscale('log');ax.axvline(1,c='grey',ls='--',lw=1);ax.invert_yaxis();ax.set(xlabel='Adjusted hazard ratio per expression-balance SD (95% CI)',title='F  TCGA-PAAD · age/sex/categorical stage adjusted')
    maxci=co.CI_high.max();ax.set_xlim(min(.45,co.CI_low.min()*.8),maxci*1.65)
    for y,r in zip(yy,co.itertuples()):ax.text(maxci*1.08,y,f'p={r.p:.3g}\nq={r.q_BH_across_two_cohorts_two_endpoints:.3g}',va='center',fontsize=9)
    ax.set_ylim(len(co)-.3,-.7)
    fig.suptitle(f'PDAC numerical nearest-reference pair: JUN {pair["JUN"]} / VIM {pair["VIM"]}',fontsize=18)
    save(fig,'01_single_cluster_calibration_overview')

def kmfigure():
    curves=pd.read_csv(B/'tables/clinical_exact_pair_KM_curves.csv');risk=pd.read_csv(B/'tables/clinical_exact_pair_KM_numbers_at_risk.csv');tests=pd.read_csv(B/'tables/clinical_exact_pair_KM_logrank.csv')
    fig=plt.figure(figsize=(13,11),layout='constrained');grid=fig.add_gridspec(2,2)
    for idx,(cohort,endpoint) in enumerate(itertools.product(['full_primary','confirmed_ductal'],['OS','PFI'])):
        inner=grid[idx//2,idx%2].subgridspec(2,1,height_ratios=[4,1],hspace=.08);ax=fig.add_subplot(inner[0]);rax=fig.add_subplot(inner[1]);test=tests[(tests.cohort==cohort)&tests.endpoint.eq(endpoint)&tests.split.eq('median')].iloc[0]
        sub=curves[(curves.cohort==cohort)&curves.endpoint.eq(endpoint)&curves.split.eq('median')]
        for group,color in [('low','#326a9d'),('high','#af5260')]:
            d=sub[sub.group==group];xx=np.r_[0,d.time_days.to_numpy()/365.25];yy=np.r_[1,d.survival.to_numpy()];ax.step(xx,yy,where='post',label=group,color=color,lw=2)
        ax.set(xlim=(0,5),ylim=(0,1.03),ylabel='Survival probability',xlabel='Years',title=f'{"Full primary PAAD" if cohort=="full_primary" else "Confirmed ductal histology"} · {endpoint}')
        ax.text(.98,.95,f'Log-rank p={test.p:.3g}\nq={test.q_BH_across_cohorts_endpoints:.3g}\nn={int(test.n)}, events={int(test.events)}',transform=ax.transAxes,ha='right',va='top',fontsize=10);ax.legend(title='Single-cluster expression balance',frameon=False,fontsize=9,loc='lower left')
        times=[0,365,730,1095,1460,1825];rows=[]
        for group in ['low','high']:
            d=risk[(risk.cohort==cohort)&risk.endpoint.eq(endpoint)&risk.split.eq('median')&risk.group.eq(group)].set_index('time_days');rows.append([int(d.loc[t,'n_at_risk']) for t in times])
        rax.axis('off');tb=rax.table(cellText=rows,rowLabels=['Low','High'],colLabels=['0','1','2','3','4','5 yr'],loc='center',cellLoc='center');tb.auto_set_font_size(False);tb.set_fontsize(10);tb.scale(1,1.1)
        rax.set_title('Number at risk',fontsize=10,pad=2)
    fig.suptitle('Exact single-cluster DEG balance · fixed full-cohort median split',fontsize=16);save(fig,'02_clinical_KM_with_numbers_at_risk')

def report(pair,cm):
    ec=json.loads((B/'tables/epithelial_clustering_contract.json').read_text());rc=json.loads((B/'tables/ccRCC_reference_contract.json').read_text());sccon=json.loads((B/'tables/singlecell_contract.json').read_text());clcon=json.loads((B/'tables/clinical_contract.json').read_text())
    sim=pd.read_csv(B/'tables/primary_all_cluster_Sim_JUN_Sim_VIM_ranks.csv');ss=pd.read_csv(B/'tables/cm_K10_14_reconstruction_stability_screen.csv');assoc=pd.read_csv(B/'tables/singlecell_exact_pair_primary_sensitivity.csv');sp=pd.read_csv(B/'tables/spatial_exact_pair_patient_signflip_inference.csv');co=pd.read_csv(B/'tables/clinical_exact_pair_Cox_OS_PFI.csv');km=pd.read_csv(B/'tables/clinical_exact_pair_KM_logrank.csv');cov=pd.read_csv(B/'tables/clinical_signature_gene_coverage.csv');check=pd.read_csv(B/'tables/independent_numerical_crosschecks.csv')
    refdiag=pd.read_csv(B/'tables/ccRCC_reference_state_separation_diagnostics.csv')
    cmref=pd.read_csv(B/'tables/cm_all_modules_original_ccRCC_CM1_composition_sensitivity.csv')
    bs=pd.read_csv(B/'tables/similarity_reference_and_target_donor_bootstrap.csv.gz');ls=pd.read_csv(B/'tables/similarity_leave_study_out_all_clusters.csv');rls=pd.read_csv(B/'tables/similarity_leave_ccRCC_reference_study_out_all_clusters.csv');stability=[]
    for definition,s in pair['all_reference_definition_selections'].items():
        for state in ['JUN','VIM']:
            c=s[state];bb=bs[(bs.configuration==PRIMARY)&bs.reference_definition.eq(definition)&bs.state.eq(state)&bs.cluster_id.eq(c)];ll=ls[(ls.configuration==PRIMARY)&ls.reference_definition.eq(definition)&ls.state.eq(state)]
            rll=rls[(rls.configuration==PRIMARY)&rls.reference_definition.eq(definition)&rls.state.eq(state)]
            stability.append({'definition':definition,'state':state,'winner':c,'bootstrap_winner_fraction':bb.winner.mean(),'PDAC_LSO_winner_fraction':ll[ll.winner].cluster_id.eq(c).mean(),'ccRCC_LSO_winner_fraction':rll[rll.winner].cluster_id.eq(c).mean(),'bootstrap_similarity_CI_low':bb.similarity.quantile(.025),'bootstrap_similarity_CI_high':bb.similarity.quantile(.975)})
    stable=pd.DataFrame(stability);stable.to_csv(B/'tables/primary_reference_definition_winner_stability.csv',index=False)
    primaryassoc=assoc[(assoc.unit=='sample')&(assoc.min_epi==30)];spprimary=sp[(sp.scope=='all12_patients')&sp.quality.eq('counts_ge500')]
    br=primaryassoc[primaryassoc.metric.eq('balance')&primaryassoc.adjustment.eq('none')].iloc[0];ba=primaryassoc[primaryassoc.metric.eq('balance')&primaryassoc.adjustment.eq('study')].iloc[0]
    sb=spprimary[spprimary.metric.eq('balance')&spprimary.adjustment.eq('epithelial_stromal_library')].iloc[0]
    coxsummary=co[co.model.eq('adjusted')].set_index(['cohort','endpoint']);osfull=coxsummary.loc[('full_primary','OS')];pfifull=coxsummary.loc[('full_primary','PFI')];osduct=coxsummary.loc[('confirmed_ductal','OS')];pfiduct=coxsummary.loc[('confirmed_ductal','PFI')]
    flow=pd.read_csv(B/'tables/epithelial_population_flow.csv');normal=flow[flow.cell_subtype.isin(['Epi_GP2','Epi_PRSS1','Epi_CELA3B','Epi_FXYD2'])].pivot_table(index='cell_subtype',columns='status',values='n',fill_value=0).reset_index()
    normal['normal_source_fraction']=normal['normal-like']/(normal['normal-like']+normal['tumor']);normal.to_csv(B/'tables/normal_epithelial_reference_population_counts.csv',index=False)
    cleanup=pd.read_csv(B/'tables/epithelial_cleanup_and_normal_reference.csv.gz',index_col=0,low_memory=False);normalcells=cleanup[cleanup.cell_subtype.isin(normal.cell_subtype)].copy();normalcells['reference_population']=np.where(normalcells.cell_subtype.eq('Epi_FXYD2'),'normal_dominated_FXYD2','acinar_reference');normalcells.rename(columns={'sample':'sample_id'}).to_csv(B/'labels/normal_epithelial_reference.csv.gz',index_label='cell_id')
    hp=B/'tables/historical_original_environment_replay_contract.json'
    if hp.exists():
        hist=json.loads(hp.read_text());hs=pd.read_csv(B/'tables/historical_original_environment_all_cluster_similarities.csv')
        histtext=f'''### Separate controlled replay in the archived original software environment

The primary analysis above uses the current Python3.12 / NumPy2.3.5 / pandas2.3 / scikit-learn1.9 environment. A **separate historical sensitivity** reran the identical saved ordered HVG3000 genes, PCA/Harmony/neighbors/Leiden parameters and cell order in the archived original Python3.14 / NumPy2.4.5 / pandas3.0.3 / scikit-learn1.8 environment. **The primary 12-cluster partition and all its downstream c7/c2 results were not replaced.**

Original-environment replay: **{hist['n_clusters']} clusters**, archived {hist['archived_n_clusters']}; all literal-ID cluster cell counts exact: **{hist['all_archived_cluster_counts_exact']}**; even the label-permutation-invariant sorted cluster-size multisets agree: **{hist['sorted_cluster_size_multisets_exact']}**; maximum archived program-axis difference **{hist['max_axis_abs_difference']:.6g}**; literal-ID mapped archived pooled endpoint agreement **{hist['archived_pooled_endpoint_agreement']:.6%}**. Exact newly replayed per-cell memberships (including any c9/c6 in that partition) are now saved at **`labels/historical_original_env_A_HVG3000_sample_r0.4.csv.gz`**. These numerical checks {'reproduce' if hist['all_archived_cluster_counts_exact'] and hist['n_clusters']==hist['archived_n_clusters'] else 'do not reproduce'} the archived 13-cluster summaries. **Matching the cluster count 13 alone does not recover historical c9/c6.** The saved original PCA is exactly identical to both recomputed PCAs after axis alignment (RMSE0; `tables/historical_PCA_replay_comparison.json`), so the remaining stochastic/numerical graph/Harmony differences are not resolved by restoring that checkpoint. Independent original cell-by-cluster authentication remains impossible because that original file was not saved.

Historical-environment core-reference similarities and separate eligible ranks (descriptive sensitivity only; no historical pooled effect is transferred):

{md(hs[hs.reference_definition.eq('archived_core12')][['cluster_id','n_cells','n_tumor_samples','coverage_eligible','Sim_JUN','rank_JUN_eligible','Sim_VIM','rank_VIM_eligible']])}

Source: `tables/historical_original_environment_replay_contract.json`, `historical_original_environment_cluster_summary_comparison.csv`, `historical_original_environment_all_cluster_similarities.csv`; graph/PCA/Harmony: `objects/historical_original_env_HVG3000_harmony_graph.npz`. Thus a new primary c9 is not historical c9, and neither is a pooled JUN endpoint.
'''
    else:
        histtext='### Separate original-environment historical replay\n\nA bounded original-software replay is still running; it is not being substituted for the frozen new primary 12-cluster partition or its c7/c2 downstream endpoints.\n'
    marker='\n'.join(f'- **{state} `{PRIMARY}/{pair[state]}`**: '+', '.join(pair['marker_identity'][state])+'.' for state in ['JUN','VIM'])
    text=f'''# PDAC calibrated expression-derived single-cluster pair

## Outcome

**A robust conserved JUN/VIM homolog pair is not established.** The frozen primary expression partition selects **one JUN-nearest cluster `{pair['JUN']}` and one VIM-nearest cluster `{pair['VIM']}`**, independently ranked #1 among coverage-eligible clusters using the archived 12+12 reference definition. However, the VIM maximum is **negative cosine ({pair['similarities']['VIM']['Sim_VIM']:.5f})**: it is the *least dissimilar* eligible cluster, not positive similarity to a conserved VIM state. The independent paired-ccRCC DEG definition selects `{pair['all_reference_definition_selections']['paired_ccRCC_DEG50']['JUN']}` for JUN, not `{pair['JUN']}`. These exact primary cell memberships, and only their pairwise DEGs, nevertheless carry every downstream result below as a transparent numerical calibration. No three-cluster pooling, relabeling old endpoints, signature retuning, or survival-based choice was used.

**Important qualification:** numerical nearest-reference winners are not automatically a robust biological homolog pair. Reference-definition, donor-bootstrap and leave-study-out results below must be read alongside the maxima. The selected CM is T/NK-rich (**{100*cm['CM1_TNK_loading_fraction']:.2f}%**) but **does not reach the desired 80% T/NK purity**. Negative/opposite findings remain unchanged.

**The earlier pooled three-modality positive pattern is not reproduced by the exact c7/c2 pair:**

- Single cell, n={int(br.n)}: balance raw rho={br.rho:.5f}, p={br.p:.6g}, q={br.q_BH_within_unit_threshold_context:.6g}; study-adjusted rho={ba.rho:.5f}, p={ba.p:.6g}, q={ba.q_BH_within_unit_threshold_context:.6g} (not significant after study adjustment).
- Spatial, 14 sections/12 patients: adjusted balance r={sb.patient_equal_weight_r:.5f}, 95% CI [{sb.CI_low:.5f}, {sb.CI_high:.5f}], exact patient sign-flip p={sb.patient_signflip_p:.6g}, q={sb.q_BH_six_tests_within_scope_quality:.6g} (null).
- Adjusted OS: full-primary HR={osfull.HR:.5f}, p={osfull.p:.6g}; ductal HR={osduct.HR:.5f}, p={osduct.p:.6g} (both null).
- Adjusted PFI: full-primary HR={pfifull.HR:.5f}, p={pfifull.p:.6g}, q={pfifull.q_BH_across_two_cohorts_two_endpoints:.6g}; ductal HR={pfiduct.HR:.5f}, p={pfiduct.p:.6g}, q={pfiduct.q_BH_across_two_cohorts_two_endpoints:.6g}. The full-PAAD PFI association is retained, but does not establish OS or ductal-histology replication.

## Populations and input provenance

- Source-QC and historical label-agreement-filtered atlas: 319,042 cells; this calibration uses the archived cleaned ductal population, not the separate 392,388-cell full-atlas reannotation. All **125 tumor/normal-like samples** remain in the non-epithelial inventory.
- Original non-epithelial: **{cm['n_original_non_epi_cells']:,} cells**. Quantitative multi-enzyme review excludes **{cm['removed_cells']:,} cells in six subtypes**: {', '.join(cm['removed_subtypes'])}. Each has >=95% of cells expressing >=3 acinar markers, mean acinar log-expression >=1 and >=50% of its own-lineage panel. This is evidence of contamination/mixed expression, not raw-droplet ambient correction.
- Fifteen previous exclusions were restored because lineage coexpression remains strong and their names alone do not justify deletion. The resulting **{cm['cleaned_non_epi_cells']:,} cells / {cm['non_epi_subtypes']} subtypes** meet the requested granularity without forced subdivision or CM-guided merging. `tables/non_epi_cleanup_decisions_with_evidence.csv` contains all 60 original subtype decisions and quantitative evidence. Exact labels: `labels/non_epithelial_54_subtypes.csv.gz`.
- NMF fit: **{cm['n_fit_samples']} samples ({cm['n_fit_tumor']} tumor, {cm['n_fit_normal']} normal-like)** with >=50 retained non-epithelial cells. All-sample unweighted NNLS projections are saved; lower-cell samples are not tested.
- Epithelial clustering: **{ec['n_cells']:,} cleaned tumor-associated ductal cells**, {ec['n_samples']} tumor samples, {ec['n_donors']} publication-namespaced donor keys, {ec['n_studies']} underlying studies. This conservatively reuses the historical strict ductal population: tumor source; original six ductal labels; acinar mean < ductal mean; ductal mean > immune mean. Acinar GP2/PRSS1/REG1A, endocrine, normal-dominated FXYD2, and other excluded/ambiguous epithelial cells (including ANXA4) are separately recorded in `tables/epithelial_cleanup_and_normal_reference.csv.gz` and `tables/epithelial_population_flow.csv`. ANXA4 itself is not claimed normal-dominated: 920/1295 cells are tumor-source, and its exclusion is a limitation of the strict historical replay population. Tumor-associated is not inferCNV-confirmed malignant.
- Primary association: **{sccon['n_primary_samples']} tumor samples**, containing {sccon['primary_n_epi_cells']:,} epithelial and {sccon['primary_n_non_epi_cells']:,} non-epithelial cells, after >=30 epithelial / >=50 non-epithelial gates. >=50 epithelial and patient-key aggregation are sensitivities.
- Original ccRCC reference: {rc['n_original_cells']:,} epithelial cells, including original Epi_JUN {rc['n_original_JUN']:,} and Epi_VIM {rc['n_original_VIM']:,}; {rc['n_paired_donors_for_DEG']} paired sample/donor keys for paired DEG derivation. Native normalized-log `raw.X` and original labels only.

Separate excluded acinar / normal-dominated FXYD2 reference populations (exact IDs in `labels/normal_epithelial_reference.csv.gz`; tumor-source acinar-like cells are not assumed malignant):

{md(normal)}

## 1. Balanced joint NMF

Column min-max scaling is fit only on eligible sample subtype frequencies; sqrt row weights give equal total tumor/normal squared-error weight. Unpenalized joint NMF, three seeds, tolerance 1e-5, max5000. H rows are L2-normalized; W is reprojected by **unweighted NNLS**, not read from weighted fit coefficients.

K selection uses reconstruction plus patient/study stability **only**, before epithelial matching/association or clinical evaluation. The reconstruction elbow within K10–14 is K={cm['reconstruction_elbow_K']}; the nearest rank satisfying the prespecified mean >=0.8 and 10th-percentile >=0.5 resampling thresholds is **K={cm['selected_K']}**. Fifty study-stratified patient-key block bootstraps and all leave-publication-out fits are reported for **every K**, not only the winning rank.

{md(ss[['K','weighted_SSE','explained_uncentered','converged','iterations','patient_bootstrap_mean','patient_bootstrap_q10','study_leaveout_mean','study_leaveout_q10','selected']])}

CM1-like alignment is composition-only: highest T/NK loading among cohesive modules with >=3 T/NK nodes each >=5% H loading. **`{cm['CM1']}`**, T/NK fraction **{cm['CM1_TNK_loading_fraction']:.5f}**, has {cm['CM1_TNK_nodes_ge5pct']} such nodes. Leading loadings: {', '.join(f'{g} {v:.3f}' for g,v in list(cm['top_CM1_nodes'].items())[:8])}. Module-specific bootstrap/leave-study-out statistics: `{json.dumps(cm['CM1_stability'])}`. This is a stable broad T/NK ecology, not an >=80%-pure or GNLY/GZMB/TOX-specific proven ccRCC homolog. The close alternate T/NK component is retained in the full H/composition tables; it is not substituted after viewing associations.

For context, the complete original ccRCC CM1 H has **70.55% T/NK loading**, itself below80%. A separate descriptive broad-lineage reference comparison (not the selection rule) shows both calCM02 and calCM09 resemble it. calCM09 is actually the closest on that alternate six-lineage cosine, so a unique CM1 homolog is **not** established; the predeclared cohesive-max-T/NK module is retained unchanged:

{md(cmref[cmref.TNK_loading_fraction.ge(.5)])}

Outputs: `tables/cm_selected_H.csv`, `cm_selected_H_fraction.csv`, `cm_selected_W_unweighted_NNLS.csv`, `cm_column_minmax_parameters.csv`, `cm_equal_tumor_normal_weights.csv`, `cm_all_seed_convergence.csv`, `cm_module_bootstrap_and_study_stability.csv`, `cm1_specific_bootstrap_study_stability.csv`.

## 2. Direct epithelial expression clustering and historical replay

All **15 configurations** (HVG2000/3000/4000 × Leiden0.3/0.4/0.5/0.6/0.8) use direct normalized-log expression -> series-aware Seurat-dispersion HVGs -> scaled/capped expression -> PCA40 -> **sample-level Harmony** -> neighbors20 on 30 Harmony PCs -> Leiden (igraph, 2 iterations, seed42). Sample correction reproduces the original configuration and limits sample/library effects; sample/bio-confounding remains a limitation. Digestive/immune-lineage ambient genes and MT/RPL/RPS are excluded from dimension reduction exactly as in the archived expression pipeline; no reference scores, CM values or outcomes enter clustering.

**Exact per-cell labels for every configuration** (`cell_id`, `sample_id`, `cluster_id`, study, donor): `labels/A_HVG{{2000,3000,4000}}_sample_r{{0.3,0.4,0.5,0.6,0.8}}.csv.gz`. Complete explicit paths and cluster counts: `tables/epithelial_configuration_manifest.csv`. Wide cross-configuration table: `labels/all_configurations_wide.csv.gz`. Primary labels: **`labels/{PRIMARY}.csv.gz`**. Coordinates: `objects/primary_epithelial_embeddings.h5ad`; normalized-log expression remains in the immutable `cm_epi_redesign/objects/cleaned_tumor_ductal.h5ad`, indexed by these exact IDs.

The primary **{PRIMARY}** anchor was fixed before reference matching, not selected for associations. Archived summary cluster count **{ec['historical_n_clusters']}**; current-environment replay cluster count **{ec['replay_n_clusters']}**; all per-ID archived cluster cell counts exactly match: **{ec['all_archived_counts_exact']}**; maximum archived axis-centroid difference **{ec['max_axis_difference']:.6g}**; *literal numeric-ID* historical pooled endpoint agreement **{ec['historical_endpoint_cell_agreement']:.6%}** (not label-permutation invariant). The original sample-Harmony c9/c6 *individual cluster membership file was not archived*. Thus the current primary 12-cluster replay does not reproduce the archived13 summary, and **exact historical cell-by-cell c9/c6 identity cannot be independently certified**. No old pooled endpoint is called a single c9/c6 result. Comparison source: `tables/historical_13cluster_summary_replay.csv`; cross-partition ARIs: `tables/epithelial_partition_ARI.csv`; ordered HVG list equality: `tables/historical_HVG_feature_replay_comparison.json` (3000/3000, identical order).

{histtext}

## 3. Uniform reference similarity, coverage gate and exact single winners

- Primary definition: unmodified archived 12 JUN + 12 VIM core genes, used to compare *actual original ccRCC Epi_JUN and Epi_VIM expression centroids* (not idealized binary program vectors).
- Independent sensitivity: paired original ccRCC JUN-minus-VIM donor DEG signature, q<0.05, signed normalized-log effect >=0.1, top50 by signed paired t per arm after mitochondrial/ribosomal, hemoglobin, fixed kidney-background and digestive filters. Arm sizes {rc['DEG_arm_sizes']}. Gene selection uses no PDAC rankings or downstream outcomes. The archived panel provenance is the earlier PDAC workflow's fixed canonical core; its kidney-only origin is not independently documented, whereas all expression centroids and the paired-DEG derivation are ccRCC-only.
- One uniform metric for all clusters and both states: **cosine in frozen ccRCC gene-z coordinates** (donor-state profile SD floor0.1), equal donors within each centroid, >=5 cells/profile. No state-specific rescaling or weights. Per-gene availability is recorded in `tables/reference_gene_coverage.csv`.
- Candidate coverage gate: **>=500 cells and >=10 tumor samples**. Ineligible clusters retain scores/all-cluster ranks but cannot win. No 45-cell single-donor cluster can be selected.
- Two-source uncertainty: 500 primary / 200 sensitivity-configuration block bootstraps jointly resample ccRCC reference and PDAC donor keys; frozen scaling and partition. PDAC leave-study-out recomputes target profiles and coverage gate; a separate ccRCC-reference leave-study-out recomputes the original reference centroids with features/scaling fixed. Configuration sweep is a sensitivity analysis, never a best-effect search.

Reference-state separation (native VIM expression is almost equal between the original labels, and the original VIM label does not show a higher archived VIM-core mean than JUN; the literal measured reference is retained rather than replaced by an idealized program):

{md(refdiag)}

### Full primary all-cluster table (separate within-PDAC ranks)

Ranks below are among coverage-eligible PDAC clusters; all-cluster ranks, both-state bootstrap CIs and winner probabilities are also present in `tables/primary_all_cluster_Sim_JUN_Sim_VIM_ranks.csv`.

{md(sim[['cluster_id','n_cells','n_tumor_samples','coverage_eligible','Sim_JUN','rank_JUN_eligible','Sim_VIM','rank_VIM_eligible']])}

### Selected pair, marker identity and stability

{marker}

{md(stable)}

The reference definitions {'agree on the exact pair' if all(s['JUN']==pair['JUN'] and s['VIM']==pair['VIM'] for s in pair['all_reference_definition_selections'].values()) else '**do not agree on the exact pair**; no signature/definition was swapped to force agreement'}. Winner probabilities below ~0.6 or inconsistent leave-study-out maxima indicate unstable ranks, not a confident conserved homolog. All 15 configurations × two definitions × every cluster are in `tables/all_configurations_all_cluster_similarities.csv`; full bootstrap and leave-study-out records are in `similarity_reference_and_target_donor_bootstrap.csv.gz` and `similarity_leave_study_out_all_clusters.csv`. The downstream pair remains the frozen primary maxima regardless of this sensitivity.

Within a fixed reference definition, high winner probability can coexist with a poor absolute match. Here the primary VIM winner has negative cosine and its interval crosses zero; its S100A11/S100A6/LCN2/MMP7 epithelial marker identity is not strong evidence of an EMT/VIM state. The primary JUN winner is an AP-1/heat-shock/immediate-early stress program. Thus c7/c2 results should be described as **exact numerical nearest-reference cluster proxies**, not validated conserved malignant Epi_JUN/Epi_VIM homologs.

Exact-pair transfer signatures use **only `{pair['JUN']}` versus `{pair['VIM']}`**, {pair.get('transfer_DEG_paired_donors')} paired donor keys with >=10 cells/cluster. Top50 signed paired-t genes with q<0.05 and effect>=0.1, excluding MT/RPL/RPS and digestive genes. Actual arm sizes: {pair.get('transfer_signature_sizes')}. All-gene evidence: `tables/exact_selected_pair_donor_paired_DEGs.csv`; membership-derived donor expression: `exact_selected_pair_donor_expression_profiles.csv.gz`; genes: `exact_selected_pair_signatures.json`. No target-vs-rest or old pooled signatures replace this contrast.

## 4a. Single-cell validation, exact pair only

Fraction denominator: **all cleaned tumor-associated ductal epithelial cells** in the frozen partition; numerator is one cluster only. Balance = ln((fJUN+0.005)/(fVIM+0.005)). Raw Spearman and partial rank correlation residualized on study dummies; two-sided tests with covariate-adjusted degrees of freedom. Patient-key aggregation combines tumor samples' counts, then recomputes non-epithelial frequencies and fixed-H unweighted NNLS W.

Primary family BH q spans three endpoints × raw/study-adjusted = six tests; sensitivities additionally have a joint24-test q, and a global-all-saved-tests q is saved. No post-selection all-cluster correlation screen was used.

{md(primaryassoc[['metric','adjustment','n','rho','p','q_BH_within_unit_threshold_context','q_BH_all_primary_and_sensitivity']])}

### Patient aggregation and >=50 epithelial sensitivity

{md(assoc.loc[~((assoc.unit=='sample')&(assoc.min_epi==30)),['unit','min_epi','metric','adjustment','n','rho','p','q_BH_within_unit_threshold_context']])}

Source tables: `tables/singlecell_all_cluster_sample_counts.csv`, `singlecell_all_cluster_sample_fractions.csv`, `singlecell_exact_pair_sample_metrics.csv`, `singlecell_exact_pair_patient_aggregates.csv`, `singlecell_all_exact_pair_associations.csv` (includes all within-study analyses).

## 4b. Spatial validation, exact new signatures only

All **14 Visium sections / 12 explicit patients** are scored directly using the exact pair-derived DEG arms, not historical pooled Tangram maps. X is archived total-count1e4/log1p expression; gene-z is within section, identical mean-gene scoring for each arm. CM1 signature is derived from the selected current H-weighted non-epithelial expression profile versus its complement, with the pair genes excluded to prevent direct gene overlap.

Primary in-tissue spots require >=500 UMIs; all-source-spots sensitivity is also saved. Adjustment controls fixed ductal epithelial proxy, stromal proxy and log library size via rank residuals. Section-local effects are averaged in Fisher-z space within patients and then equally over patients. P-values are exact patient sign-flips, **not spot-level independent tests**; CIs are 10,000 patient bootstraps. BH family is six metrics × adjustments per QC/scope.

{md(spprimary[['metric','adjustment','n_patients','n_sections','n_spots','patient_equal_weight_r','CI_low','CI_high','patient_signflip_p','q_BH_six_tests_within_scope_quality','positive_patients']])}

All-spots and seven-patient no-recorded-single-cell-overlap sensitivities remain in `tables/spatial_exact_pair_patient_signflip_inference.csv`. Source data: `spatial_14section_source_manifest.csv`, per-spot `spatial_exact_pair_spot_scores_GSM*.csv.gz`, `spatial_exact_pair_section_associations.csv`, `spatial_exact_pair_patient_effects.csv`, `spatial_exact_pair_signature_coverage.csv`, `spatial_spot_QC.csv`. Direct scores are mixed-spot expression proxies, **not calibrated mapped cell fractions**.

## 4c. TCGA clinical validation, exact pairwise-DEG balance only

Shared **`shared_clinical_and_signatures/clinical_eval.py`** is used unchanged. Full primary molecular n={clcon['n_full_molecular_primary']}; exact TCGA-CDR ductal histology molecular n={clcon['n_confirmed_ductal_molecular']}. Complete-case Cox n/events are below. Scoring is mean within-full-PAAD gene-z JUN arm minus VIM arm, then one full-cohort score SD; the same frozen scaling/cutpoint is used for ductal sensitivity. This expression proxy is not a recovered cell fraction. OS and TCGA-CDR PFI (not clinical-trial PFS) both tested; age/recorded sex/categorical stage adjusted, Efron ties, no outcome-optimized cutpoint. BH jointly covers the four cohort×endpoint adjusted balance tests.

{md(co.loc[co.model.eq('adjusted'),['cohort','endpoint','n','events','HR','CI_low','CI_high','p','q_BH_across_two_cohorts_two_endpoints','status']])}

Signature gene coverage:

{md(cov[['signature','arm','n_requested','n_used','coverage','missing','constant']])}

KM median splits (full-cohort molecular median fixed for both cohorts):

{md(km.loc[km.split.eq('median'),['cohort','endpoint','n','events','n_low','n_high','p','q_BH_across_cohorts_endpoints']])}

`tables/clinical_exact_pair_KM_numbers_at_risk.csv` reports 0/1/2/3/4/5-year risk counts by cohort/endpoint/group, and `figures/02_clinical_KM_with_numbers_at_risk.png/.pdf` renders them. Outer-quartile KM, full curves, unadjusted Cox, model solver/gradient and approximate PH diagnostics remain in source tables. Exact model/clinical inputs and sample IDs: `clinical_exact_pair_scores.csv`, `clinical_prepared_PAAD_source.csv`, `clinical_exact_endpoint_model_source.csv`. No shared resource or other cancer output was changed.

## Independent numerical checks and real-data figures

Independent recomputation reads raw per-cell labels/expression rather than saved centroids/fractions, manually rebuilds donor-balanced cosine, reconstructs fractions, and reprojects W by fresh NNLS:

{md(check)}

- `figures/01_single_cluster_calibration_overview.png/.pdf`: expression-cluster UMAP, all-cluster similarity heatmap with both primary #1 maxima marked, actual CM1 composition, exact-pair single-cell, spatial and clinical effects.
- `figures/02_clinical_KM_with_numbers_at_risk.png/.pdf`: real OS/PFI KM estimates with explicit number-at-risk rows.
- `figures/03_NMF_rank_evidence.png/.pdf`: reconstruction and patient/study stability across every K10–14, with fixed selected K marked.
- Rendered PNGs were inspected directly with the image read tool; the CM legend, single-cell study legend and spatial pooled-effect annotation were repositioned to remove data overlap. Final overview and KM risk tables are unclipped.
- Scripts `scripts/00_*.py` through `05_verify_figures_report.py`; Python `{PY}`. All outputs stay in this calibration directory. No Figure7, manuscript, CRC or other cancer outputs are modified.

## Explicit limitations

1. This is a conservative historical-input calibration, not independent raw-droplet QC or the later expanded full-atlas epithelial repair. Original broad/ductal annotations and excluded potential malignant ANXA4/FXYD2 cells may bias which tumor states are available. No CNV-based malignancy call is claimed.
2. Six multi-enzyme groups are omitted for contamination; many retain own-lineage markers, so this cannot distinguish ambient contamination, doublets or biological uptake. Fifteen formerly excluded groups are restored for insufficient exclusion evidence; residual mixed-lineage expression remains possible.
3. CM1 purity is only {100*cm['CM1_TNK_loading_fraction']:.1f}%, not >=80%; other stable T/NK modules exist. It is a broad ecological nearest module, not a proven fine-subtype ccRCC CM1 homolog. K was not changed to improve this purity or associations.
4. Cross-tissue reference identity is uncertain; all original ccRCC labels are used literally. The archived core's kidney-only derivation is not independently documented, and the paired-ccRCC DEG sensitivity may disagree. A rank#1 is relative to eligible clusters, not proof of conserved JUN/VIM biology. Unstable bootstraps or study leaveouts are explicitly retained.
5. Exact old c9/c6 membership cannot be independently authenticated without an original sample-Harmony cluster-membership archive; count/program/pooled-endpoint replay agreement is not a substitute for that missing comparison. The archived and current HVG3000 ordered gene lists are identical; original/current NumPy, pandas, scikit-learn and Python versions differ. Literal numeric-ID endpoint agreement is not label-permutation invariant.
6. UCSC patient linkage is incomplete: many donor keys are sample-level proxies. The original ccRCC object likewise has series/sample but no true patient field, so its reference balance/paired-DEG units are sample/donor keys. Bootstrap/aggregation is conditional on these recorded identifiers; true paired-patient independence cannot be guaranteed. Spatial patient IDs are explicit; five overlap single-cell source patients, with seven-patient sensitivity saved.
7. Harmony sample correction can remove biological signal if sample and state are confounded. Partition/ref-uncertainty bootstraps are conditional on the fitted graph; cross-configurations are sensitivity analyses, not full reclustering bootstraps.
8. Single-cell frequency ecology is compositional. AP-1/immediate-early expression can reflect dissociation/stress rather than a stable malignant state. Spatial/bulk transfer scores mix cell compartments and do not equal single-cluster fractions; compartment adjustment and disjoint CM genes reduce, but do not eliminate, confounding. Clinical models lack tumor-purity/treatment adjustment and require proportional-hazards caution.
9. This is an exploratory calibration after earlier pooled analyses, not a preregistered prospective replication. All null, unstable and opposite-sign results are reported, and no downstream result selects genes, weights, K, clustering resolution or reference definition.
'''
    (B/'REPORT.md').write_text(text)
    js({'python':sys.version,'executable':sys.executable,'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scanpy','anndata','scikit-learn','harmonypy','statsmodels','matplotlib']}},'tables/software_versions.json')

if __name__=='__main__':
    # Earlier functions are retained for historical reproducibility only.
    # The completed fits/checks must not be restarted for a reporting update.
    import importlib
    importlib.import_module('10_finalize_balance_focus').main()
