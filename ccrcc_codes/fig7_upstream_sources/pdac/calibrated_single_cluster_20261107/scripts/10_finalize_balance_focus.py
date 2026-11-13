"""Presentation-only update for the user-clarified balance estimand.

Reads completed labels, fits and numerical tables. No clustering, NMF, clinical
refitting, signature selection, Visium reading/scoring or spatial computation.
"""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                     'axes.labelsize':10,'savefig.dpi':180,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})

def md(frame):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.6g}' if np.isfinite(x) else 'NA'
        return str(x)
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def save_figure(fig,name):
    for ext in ['png','pdf']:fig.savefig(B/f'figures/{name}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)

def load_saved():
    pair=json.loads((B/'tables/selected_single_cluster_pair.json').read_text())
    cm=json.loads((B/'tables/cm_calibration_contract.json').read_text())
    assoc=pd.read_csv(B/'tables/singlecell_exact_pair_primary_sensitivity.csv',float_precision='round_trip')
    clinical=pd.read_csv(B/'tables/clinical_exact_pair_Cox_OS_PFI.csv',float_precision='round_trip')
    km=pd.read_csv(B/'tables/clinical_exact_pair_KM_logrank.csv',float_precision='round_trip')
    samples=pd.read_csv(B/'tables/singlecell_exact_pair_sample_metrics.csv',index_col=0,float_precision='round_trip')
    return pair,cm,assoc,clinical,km,samples

def focus_tables(pair,cm,assoc,clinical,km,samples):
    # All effect estimates, p values and BH families are copied unchanged.
    tables={}
    for threshold,name in [(30,'primary'),(50,'sensitivity_epi50')]:
        d=assoc[assoc.min_epi.eq(threshold)].copy()
        d['estimand_role']=np.where(d.metric.eq('balance'),'PRIMARY_BALANCE','ARM_DIAGNOSTIC')
        d['metric_order']=d.metric.map({'balance':0,'JUN_fraction':1,'VIM_fraction':2})
        d=d.sort_values(['unit','adjustment','metric_order']).drop(columns='metric_order')
        d.to_csv(B/f'tables/balance_focused_singlecell_{name}.csv',index=False)
        tables[name]=d
    co=clinical[clinical.model.eq('adjusted')].copy()
    co.to_csv(B/'tables/balance_focused_clinical_adjusted.csv',index=False)
    km[km.split.eq('median')].to_csv(B/'tables/balance_focused_clinical_KM_median.csv',index=False)
    formula=np.log((samples.JUN_fraction+.005)/(samples.VIM_fraction+.005))
    delta=float(np.nanmax(np.abs(formula-samples.balance)))
    assert delta<1e-12
    per_sample=samples[['study','donor','status','n_epi','n_non_epi','JUN_fraction','VIM_fraction','balance','CM1_W','primary_eligible']].copy()
    per_sample=per_sample.rename(columns={'balance':'Balance'})
    per_sample.to_csv(B/'tables/balance_primary_per_sample.csv',index_label='sample_id')
    js({'primary_estimand':'ln((f_JUNlike+0.005)/(f_VIMlike+0.005)) vs CM1-like unweighted NNLS activity',
        'frozen_pair':{k:pair[k] for k in ['JUN','VIM']},'frozen_configuration':pair['configuration'],
        'frozen_K':cm['selected_K'],'frozen_CM1':cm['CM1'],
        'cluster_selection':'unchanged reference-similarity maxima; never balance-correlation-based',
        'VIM_pole_interpretation':'negative reference pole; canonical EMT/mesenchymal identity is not required',
        'two_sided_interpretation':'raw association is JUN-arm-dominated; opposing-arm signal is not supported; both adjusted arm associations are weak',
        'clinical_interpretation':'secondary clinical meaning may differ in direction or be null; not an optimization or success gate',
        'spatial_scope':'USER_DESCOPED; pre-existing outputs exploratory only, not validation; no further spatial computation',
        'original_numerical_tables_labels_fits_signatures_unchanged':True,
        'BH_families':'unchanged six arm/balance x adjustment tests per unit/threshold; original joint24-test q retained; adjusted clinical q across four cohort-endpoint tests',
        'saved_balance_formula_max_abs_difference':delta,
        'new_computation':'presentation, saved-fraction formula check and table extraction only; no statistical refitting'},
       'tables/balance_estimand_reporting_contract.json')
    js({'status':'USER_DESCOPED_EXPLORATORY_ONLY','computed_before_scope_change':True,
        'further_spatial_compute_performed':False,'required_for_primary_inference':False,
        'retained_tables':'tables/spatial_*.csv and tables/spatial_*.csv.gz',
        'retained_legacy_overview':'figures/01_single_cluster_calibration_overview.png/.pdf',
        'current_primary_overview':'figures/04_balance_primary_overview.png/.pdf'},
       'tables/spatial_scope_status.json')
    (B/'tables/SPATIAL_DESCOPED.md').write_text('# Spatial outputs: exploratory / user-descoped\n\nAll existing `spatial_*` source and result tables are preserved unchanged. They were computed before the scope clarification and are **not claimed as validation** of the primary balance hypothesis. No further Visium mapping, deconvolution, scoring or inference is performed. See `spatial_scope_status.json`.\n')
    return tables,co

def overview(pair,cm,assoc,clinical,samples):
    emb=ad.read_h5ad(B/'objects/primary_epithelial_embeddings.h5ad')
    sim=pd.read_csv(B/'tables/primary_all_cluster_Sim_JUN_Sim_VIM_ranks.csv')
    clusters=sorted(emb.obs.cluster_id.unique(),key=lambda z:int(z[1:]))
    colors={c:plt.get_cmap('tab20')(i%20) for i,c in enumerate(clusters)}
    fig=plt.figure(figsize=(18,11.5),layout='constrained')
    grid=fig.add_gridspec(2,6,height_ratios=[1.05,1])
    ax=fig.add_subplot(grid[0,0:2]);xy=emb.obsm['X_umap'];order=np.random.default_rng(42).permutation(len(xy))
    ax.scatter(xy[order,0],xy[order,1],c=[colors[c] for c in emb.obs.cluster_id.iloc[order]],s=1.0,alpha=.65,rasterized=True,linewidths=0)
    for c in clusters:
        mask=emb.obs.cluster_id.eq(c).to_numpy();position=np.median(xy[mask],axis=0)
        ax.text(*position,c,ha='center',va='center',fontsize=8,bbox={'boxstyle':'round,pad=.15','fc':'white','alpha':.85,'ec':'black' if c in [pair['JUN'],pair['VIM']] else 'none'})
    ax.set(title=f'A  Expression clusters · {len(xy):,} ductal cells',xlabel='UMAP1',ylabel='UMAP2');ax.set_xticks([]);ax.set_yticks([])
    ax.text(.01,.01,f'Positive pole {pair["JUN"]} / negative pole {pair["VIM"]}',transform=ax.transAxes,fontsize=9,bbox={'fc':'white','alpha':.85,'ec':'none'})

    ax=fig.add_subplot(grid[0,2:4]);s=sim.set_index('cluster_id').reindex(clusters);matrix=s[['Sim_JUN','Sim_VIM']].to_numpy()
    image=ax.imshow(matrix,cmap='coolwarm',vmin=-1,vmax=1,aspect='auto')
    ax.set_yticks(range(len(s)),[c+(' †' if not s.loc[c,'coverage_eligible'] else '') for c in clusters],fontsize=9)
    ax.set_xticks([0,1],['Sim_JUN','Sim_VIM']);ax.set_title('B  Frozen reference-similarity selection')
    for i,c in enumerate(clusters):
        for j,pole in enumerate(['JUN','VIM']):
            ax.text(j,i,f'{matrix[i,j]:.3f}'+(' ★ #1' if c==pair[pole] else ''),ha='center',va='center',fontsize=9)
    ax.set_xlabel('★ eligible maximum; † outside coverage gate',fontsize=9)
    fig.colorbar(image,ax=ax,shrink=.65,pad=.03,label='cosine')
    for spine in ax.spines.values():spine.set_visible(False)

    ax=fig.add_subplot(grid[0,4:6]);fraction=pd.read_csv(B/'tables/cm_selected_H_fraction.csv',index_col=0).loc[cm['CM1']].nlargest(12).sort_values()
    ax.barh(fraction.index,fraction.values,color=['#326a9d' if c.startswith(('T_','NK_')) else '#b5bac1' for c in fraction.index])
    ax.set(title=f'C  CM1-like {cm["CM1"]} · K={cm["selected_K"]}',xlabel='Fraction of module H loading')
    ax.tick_params(axis='y',labelsize=9)
    ax.legend(handles=[Patch(color='#326a9d',label='T/NK'),Patch(color='#b5bac1',label='Other')],loc='center right',bbox_to_anchor=(1,.5),frameon=False,fontsize=9)
    ax.text(.98,.02,f'T/NK loading {100*cm["CM1_TNK_loading_fraction"]:.1f}%',transform=ax.transAxes,ha='right',fontsize=10)

    ax=fig.add_subplot(grid[1,0:3]);eligible=samples[samples.primary_eligible];studies=sorted(eligible.study.unique())
    palette={study:plt.get_cmap('tab10')(i) for i,study in enumerate(studies)}
    for study,g in eligible.groupby('study'):ax.scatter(g.balance,g.CM1_W,color=palette[study],s=30,alpha=.8,edgecolors='white',linewidths=.3,label=study)
    a=assoc[(assoc.unit=='sample')&assoc.min_epi.eq(30)&assoc.metric.eq('balance')].set_index('adjustment');raw=a.loc['none'];partial=a.loc['study']
    ax.set(title=f'D  Primary balance–CM1 association · n={int(raw.n)}',xlabel='Balance = ln[(f JUNlike + .005)/(f VIMlike + .005)]',ylabel=f'{cm["CM1"]} unweighted NNLS activity')
    ax.text(.02,.98,f'Raw ρ={raw.rho:.3f}, p={raw.p:.3g}, q={raw.q_BH_within_unit_threshold_context:.3g}\nStudy-adjusted ρ={partial.rho:.3f}, p={partial.p:.3g}, q={partial.q_BH_within_unit_threshold_context:.3g}',transform=ax.transAxes,va='top',fontsize=10,bbox={'fc':'white','alpha':.88,'ec':'none'})
    ax.legend(fontsize=7,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False)

    ax=fig.add_subplot(grid[1,3:6]);co=clinical[clinical.model.eq('adjusted')].sort_values(['cohort','endpoint'],ascending=[False,True]);y=np.arange(len(co))
    ax.errorbar(co.HR,y,xerr=np.array([co.HR-co.CI_low,co.CI_high-co.HR]),fmt='o',color='#97515b',capsize=4)
    ax.set_yticks(y,[f'{"Full PAAD" if r.cohort=="full_primary" else "Ductal"} · {r.endpoint}\nn={int(r.n)}, events={int(r.events)}' for r in co.itertuples()],fontsize=9)
    ax.set_xscale('log');ax.axvline(1,color='grey',ls='--',lw=1);ax.set(title='E  Clinical balance association · age/sex/stage adjusted',xlabel='Hazard ratio per full-cohort expression-balance SD (95% CI)')
    hi=co.CI_high.max();ax.set_xlim(min(.45,co.CI_low.min()*.8),hi*1.7)
    for yy,r in zip(y,co.itertuples()):ax.text(hi*1.08,yy,f'p={r.p:.3g}\nq={r.q_BH_across_two_cohorts_two_endpoints:.3g}',va='center',fontsize=9)
    ax.set_ylim(len(co)-.3,-.7)
    fig.suptitle(f'PDAC: frozen {pair["JUN"]}/{pair["VIM"]} epithelial balance and CM1-like co-occurrence',fontsize=17)
    save_figure(fig,'04_balance_primary_overview')

def reporting(pair,cm,assoc,clinical,km,samples):
    old=(B/'REPORT.md').read_text()
    archive=B/'REPORT_pre_balance_focus_20261107.md'
    if not archive.exists():archive.write_text('# Superseded reporting interpretation\n\nArchived before the user clarified the balance estimand and descoped spatial validation. This is not the current conclusion; use `REPORT.md`. All underlying numerical results remain unchanged.\n\n'+old)
    # Keep the already documented fitting and historical-replay details without
    # rerunning either step. On a repeated presentation run, reuse this appendix.
    start=old.find('## 1. Balanced joint NMF')
    end=old.find('## 3. Uniform reference similarity')
    if start>=0 and end>start:methods=old[start:end]
    else:
        archived=archive.read_text();start=archived.index('## 1. Balanced joint NMF');end=archived.index('## 3. Uniform reference similarity');methods=archived[start:end]
    ec=json.loads((B/'tables/epithelial_clustering_contract.json').read_text());sc=json.loads((B/'tables/singlecell_contract.json').read_text())
    sim=pd.read_csv(B/'tables/primary_all_cluster_Sim_JUN_Sim_VIM_ranks.csv')
    stability=pd.read_csv(B/'tables/primary_reference_definition_winner_stability.csv')
    coverage=pd.read_csv(B/'tables/clinical_signature_gene_coverage.csv')
    checks=pd.read_csv(B/'tables/independent_numerical_crosschecks.csv')
    primary=assoc[assoc.min_epi.eq(30)].copy();sensitivity=assoc[assoc.min_epi.eq(50)].copy()
    def row(unit,adjustment,metric='balance'):
        return primary[primary.unit.eq(unit)&primary.adjustment.eq(adjustment)&primary.metric.eq(metric)].iloc[0]
    sr=row('sample','none');sa=row('sample','study');pr=row('patient','none');pa=row('patient','study')
    sj=row('sample','none','JUN_fraction');sv=row('sample','none','VIM_fraction');aj=row('sample','study','JUN_fraction');av=row('sample','study','VIM_fraction')
    btable=primary[primary.metric.eq('balance')][['unit','adjustment','n','rho','p','q_BH_within_unit_threshold_context']]
    arms=primary[~primary.metric.eq('balance')][['unit','adjustment','metric','n','rho','p','q_BH_within_unit_threshold_context']]
    adjusted=clinical[clinical.model.eq('adjusted')]
    poles=[]
    for name in ['JUN','VIM']:
        s=pair['similarities'][name]
        poles.append({'pole':'positive JUN-like' if name=='JUN' else 'negative VIM-reference','cluster':pair[name],'n_cells':int(s['n_cells']),'n_tumor_samples':int(s['n_tumor_samples']),'selected_similarity':s['Sim_'+name],'eligible_rank':int(s['rank_'+name+'_eligible']),'markers':', '.join(pair['marker_identity'][name][:6])})
    text=f'''# PDAC frozen single-cluster BALANCE analysis

## Primary estimand and final interpretation

The primary estimand is **epithelial JUN-versus-VIM balance versus CM1-like lymphocyte-module activity**, not either arm alone and not an EMT-conservation test:

**Balance = ln((f_JUNlike + 0.005) / (f_VIMlike + 0.005)).**

The exact single-cluster pair remains **positive pole c7 / negative reference pole c2**. Each is the independently selected #1 similarity maximum for its designated original ccRCC reference under the frozen primary definition. **c2 is a negative reference pole; canonical EMT/mesenchymal identity is not required.** Its marker identity and negative absolute cosine are descriptive reference properties, not reasons to discard this balance. Pair selection has not used the balance–CM1 correlation.

- **Primary sample analysis, n={int(sr.n)}:** raw balance rho={sr.rho:.6f}, p={sr.p:.7g}, BH q={sr.q_BH_within_unit_threshold_context:.7g}; study-adjusted/partial rho={sa.rho:.6f}, p={sa.p:.7g}, q={sa.q_BH_within_unit_threshold_context:.7g}.
- **Patient-key aggregation, n={int(pr.n)}:** raw balance rho={pr.rho:.6f}, p={pr.p:.7g}, q={pr.q_BH_within_unit_threshold_context:.7g}; study-adjusted rho={pa.rho:.6f}, p={pa.p:.7g}, q={pa.q_BH_within_unit_threshold_context:.7g}.
- **Arm contribution:** the observed raw association is **JUN-arm-dominated, not supported as a two-sided opposing-arm signal**. JUN fraction correlates positively with CM1 (rho={sj.rho:.6f}, p={sj.p:.7g}); the negative-reference fraction does not show the corresponding inverse association (rho={sv.rho:.6f}, p={sv.p:.7g}). After study adjustment the signs are opposing but weak (JUN rho={aj.rho:.6f}, p={aj.p:.7g}; negative pole rho={av.rho:.6f}, p={av.p:.7g}). Thus the raw balance signal attenuates after study adjustment; a study-independent or causal co-occurrence effect is not established here. Arm tests are diagnostic, not pair-selection gates.
- **Clinical meaning may differ across cancers.** Adjusted OS/PFI and fixed-cutpoint KM results below are reported as associations, not a significance/direction requirement. Null or opposite-direction estimates do not trigger optimization.
- **Spatial is user-descoped.** Previously computed spatial results are retained as exploratory only, are not claimed as validation, and are not included in the current primary figure. No further Visium mapping, deconvolution, scoring or spatial inference was run for this update.

This is a reporting/estimand clarification after the computations, not a new fitting or selection pass. All saved cell memberships, CM fits, original association tables, signatures and clinical models are preserved. No K, resolution, genes, subset, reference metric, signs or pseudocounts were retuned.

## Frozen population, pair and CM1-like identity

- Epithelial input: **{ec['n_cells']:,} tumor-associated ductal cells**, {ec['n_samples']} tumor samples, {ec['n_donors']} donor/patient keys, {ec['n_studies']} studies. The primary 74 samples contain {sc['primary_n_epi_cells']:,} epithelial and {sc['primary_n_non_epi_cells']:,} non-epithelial cells.
- Cleaned non-epithelial input: **{cm['cleaned_non_epi_cells']:,} cells / {cm['non_epi_subtypes']} subtypes**, with 6,208 multi-enzyme-contaminated cells excluded on explicit expression evidence. All125 samples remain inventoried; fitting uses **{cm['n_fit_samples']} samples ({cm['n_fit_tumor']} tumor/{cm['n_fit_normal']} normal-like)** with >=50 non-epithelial cells.
- **K={cm['selected_K']}, CM1-like `{cm['CM1']}`**, T/NK H loading **{100*cm['CM1_TNK_loading_fraction']:.2f}%**. Leading nodes: {', '.join(list(cm['top_CM1_nodes'])[:6])}. Module-specific patient-bootstrap mean cosine **{cm['CM1_stability']['patient_bootstrap']['mean']:.6f}**; leave-study-out mean **{cm['CM1_stability']['leave_study_out']['mean']:.6f}**. Purity and alternate lymphocyte-module similarities are descriptive limitations, not new exclusion gates.

{md(pd.DataFrame(poles))}

**Exact primary per-cell labels:** `labels/{PRIMARY}.csv.gz` (`cell_id`, `sample_id`, `cluster_id`, study, donor). All15 configurations and their exact paths remain in `tables/epithelial_configuration_manifest.csv`; the wide membership table is `labels/all_configurations_wide.csv.gz`. Non-epithelial labels: `labels/non_epithelial_54_subtypes.csv.gz`. Coordinates: `objects/primary_epithelial_embeddings.h5ad`; expression is linked to the unchanged `cm_epi_redesign/objects/cleaned_tumor_ductal.h5ad` by exact cell IDs. Acinar/normal-dominated FXYD2 reference labels remain in `labels/normal_epithelial_reference.csv.gz`.

Historical c9/c6 membership was not recovered: the current replay has12 clusters; the separate original-environment replay has13 but does not match the archived cluster-size multiset. Neither replay is renamed as an old pooled endpoint. Full completed comparisons are retained in the methods appendix below; no replay is being restarted.

## Primary balance–CM1 single-cell results

Fractions use **all cleaned tumor-associated ductal epithelial cells in the frozen partition** as denominator; each numerator contains one cluster only. Primary tumor-sample eligibility is >=30 epithelial and >=50 cleaned non-epithelial cells. CM activity is the saved **unweighted NNLS W** for `{cm['CM1']}` from the balanced joint NMF. Raw Spearman and partial Spearman (rank residuals on study indicators) are retained unchanged. Patient-key analysis aggregates tumor-only counts and uses the already completed fixed-H NNLS reprojection.

### Primary estimand: balance

{md(btable)}

### Arm diagnostics: is the balance genuinely two-sided?

{md(arms)}

**Conclusion:** positive-pole/JUN-driven raw co-occurrence; no supported inverse negative-pole association. The patient-key analysis gives the same qualitative pattern. Weak opposing adjusted signs are compatible with a two-pole contrast but do not establish a robust two-sided signal. The primary balance test itself remains valid as defined; lack of EMT markers in c2 is irrelevant to this diagnostic conclusion.

The original BH family is **unchanged**: three metrics (balance plus both arms) × two adjustments = six tests per unit/threshold. The existing joint24-test and all-saved-test q values remain in the original tables. No post hoc narrowing of the FDR family was used to improve significance.

### Completed >=50 epithelial sensitivity (unchanged)

{md(sensitivity[['unit','adjustment','metric','n','rho','p','q_BH_within_unit_threshold_context']])}

Presentation tables: `tables/balance_primary_per_sample.csv`, `balance_focused_singlecell_primary.csv`, `balance_focused_singlecell_sensitivity_epi50.csv`. Unchanged numerical sources: `singlecell_exact_pair_sample_metrics.csv`, `singlecell_exact_pair_patient_aggregates.csv`, `singlecell_exact_pair_primary_sensitivity.csv`, `singlecell_all_exact_pair_associations.csv`. The latter also retains within-study estimates.

## Adjusted TCGA-PAAD clinical associations of the exact-pair balance

The **same c7-versus-c2 pairwise-DEG signature** is used without changes: 41 paired donor keys (>=10 cells/cluster); top50 eligible genes per direction by the already completed paired-donor contrast. No target-versus-rest or pooled signature is substituted. Shared `clinical_eval.py` supplies age/recorded sex/categorical-stage-adjusted Cox with Efron ties. Scores are the mean full-PAAD gene-z JUN arm minus negative-pole arm, standardized to one full-cohort SD. This is an expression balance proxy, not the single-cell fraction ratio.

Full primary molecular n=178; exact TCGA-CDR ductal histology molecular n=147. Endpoint/covariate-complete counts and events are shown below. **Direction and non-significance are acceptable clinical findings and are not optimization criteria.** Existing BH q covers four cohort×endpoint adjusted tests.

{md(adjusted[['cohort','endpoint','n','events','HR','CI_low','CI_high','p','q_BH_across_two_cohorts_two_endpoints']])}

Signature coverage (unchanged):

{md(coverage[['arm','n_requested','n_used','coverage','missing','constant']])}

### Fixed-median KM/log-rank and number at risk

Cox above is covariate-adjusted. **Standard KM/log-rank below is unadjusted**, using the previously frozen full-molecular-cohort median for both full and ductal cohorts; no optimized cutpoint or new subgroup is introduced.

{md(km[km.split.eq('median')][['cohort','endpoint','n','events','n_low','n_high','p','q_BH_across_cohorts_endpoints']])}

Risk counts at 0/1/2/3/4/5 years remain in `tables/clinical_exact_pair_KM_numbers_at_risk.csv`, with saved curves in `clinical_exact_pair_KM_curves.csv`. The already rendered `figures/02_clinical_KM_with_numbers_at_risk.png/.pdf` is retained unchanged. Adjusted table: `tables/balance_focused_clinical_adjusted.csv`; source: `clinical_exact_pair_Cox_OS_PFI.csv`. Scores, genes, clinical cases and endpoint rows remain in `clinical_exact_pair_scores.csv`, `exact_selected_pair_signatures.json`, `clinical_prepared_PAAD_source.csv`, `clinical_exact_endpoint_model_source.csv`. PFI denotes TCGA-CDR progression-free interval, not clinical-trial PFS.

## Reference-similarity selection and uncertainty (not balance optimization)

The primary configuration and archived12+12 reference definition remain fixed. The same cosine on frozen ccRCC gene-standardized donor-balanced centroids is used for every cluster and both poles. Eligibility remains >=500 cells and >=10 tumor samples. The two selected maxima are different clusters; no arm was pooled or selected for a larger balance correlation.

{md(sim[['cluster_id','n_cells','n_tumor_samples','coverage_eligible','Sim_JUN','rank_JUN_eligible','Sim_VIM','rank_VIM_eligible']])}

{md(stability)}

The paired ccRCC DEG reference selects c5 rather than c7 for the positive pole; this is retained as reference-definition uncertainty, not used to swap the frozen pair. The negative pole c2 remains a valid operational reference in the defined balance even though its primary absolute cosine is negative. **Neither canonical EMT markers nor cross-cancer conservation of clinical direction is a requirement.** Similarity and nomenclature should not be mistaken for evidence of mechanistic equivalence.

All-cluster/all-configuration scores, frozen reference scaling, donor bootstraps and target/reference-study leaveouts remain in `all_configurations_all_cluster_similarities.csv`, `similarity_reference_and_target_donor_bootstrap.csv.gz`, `similarity_leave_study_out_all_clusters.csv`, `similarity_leave_ccRCC_reference_study_out_all_clusters.csv` and the ccRCC reference tables.

## Current figure and numerical checks

- **Primary figure:** `figures/04_balance_primary_overview.png/.pdf`: epithelial expression-cluster UMAP; all-cluster Sim_JUN/Sim_VIM heatmap with both frozen maxima; CM1-like composition; primary sample balance–CM1 plot; adjusted clinical forest plot. **No spatial panel.**
- Existing KM/risk figure: `figures/02_clinical_KM_with_numbers_at_risk.png/.pdf`.
- NMF evidence: `figures/03_NMF_rank_evidence.png/.pdf`.
- Reuse-only finalizer: `scripts/10_finalize_balance_focus.py`; it reads saved tables and embeddings and performs no fitting or spatial computation. The new PNG was directly inspected with the image read tool: all five panels, legends, labels and confidence intervals are visible without overlap/clipping; the corresponding PDF is saved.

Previously completed independent recomputations are preserved, not restarted:

{md(checks)}

## Spatial: exploratory / user-descoped archive only

All previously computed `tables/spatial_*` data and results are preserved unchanged. They were produced before the scope clarification and **are not a claimed validation, requirement or success criterion** for the primary balance hypothesis. The old `figures/01_single_cluster_calibration_overview.png/.pdf` contains a legacy spatial panel and is now an archived exploratory figure, not the primary overview. See `tables/SPATIAL_DESCOPED.md`, `tables/spatial_scope_status.json` and `figures/README.md`. This update does not open Visium matrices or run mapping, deconvolution, scoring, permutation or other spatial inference.

## Limitations and scope of inference

1. The raw balance association is JUN-arm-dominated and attenuates with study adjustment; opposing contributions from both arms are not demonstrated. Observational co-occurrence is not evidence that the balance causally determines module abundance.
2. The input is the frozen conservative 55,357-cell tumor-associated ductal population, not CNV-confirmed malignant epithelium or a rerun of the expanded atlas. Ambiguous ANXA4/FXYD2 exclusions and residual mixed/ambient profiles remain limitations; all exclusions and normal-reference counts are saved.
3. CM1-like calCM02 is T/NK-rich ({100*cm['CM1_TNK_loading_fraction']:.2f}% loading), not a uniquely proven fine-subtype ccRCC homolog. The chosen K/module and their uncertainty are fixed and are not optimized for this association.
4. Reference definition affects positive-pole identity. c2 is the negative reference pole, not an asserted canonical EMT state; lack of EMT markers does not invalidate the balance estimand.
5. Some recorded patient/donor keys are sample proxies, particularly UCSC and the original ccRCC reference. Patient-key aggregation/bootstrap is conditional on available identifiers; publication adjustment/leaveouts are also retained.
6. AP-1/immediate-early expression may reflect dissociation/stress, and compositional fractions can create dependencies. Bulk expression balance is not a calibrated cell-fraction ratio; clinical models lack purity/treatment adjustment. Null or opposite-direction clinical associations require no signature, K, resolution or subset retuning.
7. This is an exploratory analysis with a user-clarified estimand after computations, not a preregistered replication. Existing effect estimates, p/q values, labels and fits remain unchanged. Spatial validation is out of scope.

## Completed methods appendix (preserved)

{methods}
'''
    (B/'REPORT.md').write_text(text)
    (B/'figures/README.md').write_text('# Figure scope after balance-estimand clarification\n\n- **Primary:** `04_balance_primary_overview.png/.pdf` — expression clusters, similarity maxima, CM composition, balance–CM1 association and clinical forest; no spatial panel.\n- `02_clinical_KM_with_numbers_at_risk.png/.pdf` — retained exact-pair OS/PFI KM curves and risk counts.\n- `03_NMF_rank_evidence.png/.pdf` — retained rank/stability evidence.\n- **Archived exploratory / spatial descoped:** `01_single_cluster_calibration_overview.png/.pdf` — earlier overview containing spatial panel E; it is not current validation or a primary requirement. All files are retained.\n')

def main():
    pair,cm,assoc,clinical,km,samples=load_saved()
    focus_tables(pair,cm,assoc,clinical,km,samples)
    overview(pair,cm,assoc,clinical,samples)
    reporting(pair,cm,assoc,clinical,km,samples)
    print('Balance-focused report/tables/figure saved; frozen labels, fits, signatures and existing result tables unchanged; no spatial compute.',flush=True)

if __name__=='__main__':main()
