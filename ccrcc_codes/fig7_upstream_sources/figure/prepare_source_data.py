#!/usr/bin/env python3
"""Reuse and verify saved single-cell results; never refit a graph or an NMF.

The only new fit is a clearly labelled descriptive OLS line for the scatter
display, with its pointwise t-based 95% mean confidence band. Reported rank
statistics and multiplicity corrections are read, not replaced.
"""
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
DATA = OUT / 'source_data'
DATA.mkdir(parents=True, exist_ok=True)
BASE = Path('/home/lr_xcy/lr_xcy/riku/lr_riku/validation_4_kirc_agents')
CRC = BASE / 'crc_codex/calibrated_single_cluster_20261107'
PDAC = BASE / 'pdac_codex/calibrated_single_cluster_20261107'
KIRC = Path('/mnt/disk18t/lr_xcy/riku/kirc_205')
REF = KIRC / 'final/fig7_extension_20261004/ccRCC_reference'
OLD = KIRC / 'final/fig7_positive_redesign_20261106/source_data'
provenance, checks, mappings = [], [], []


def read_table(path, **kwargs):
    # Preserve saved IEEE double values rather than using pandas' decimal
    # fast-parser approximation for small p values and UMAP coordinates.
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def save(df, name, source, columns='', note=''):
    df.to_csv(DATA / name, index=False)
    provenance.append(dict(local_file='source_data/' + name, original_file=str(source),
                           original_columns=columns, transformation=note, rows=len(df)))
    return df


def check(name, condition, detail=''):
    checks.append(dict(check=name, passed=bool(condition), detail=str(detail)))
    if not condition:
        raise AssertionError(name + ': ' + str(detail))


def copy(source, name, note='Unmodified saved source table'):
    shutil.copyfile(source, DATA / name)
    provenance.append(dict(local_file='source_data/' + name, original_file=str(source),
                           original_columns='all', transformation=note, rows=''))


def mapping(element, file, column, selector, original_file, original_column, note=''):
    mappings.append(dict(figure_element=element, local_file='source_data/' + file,
                         column=column, row_selector=selector,
                         original_file=str(original_file), original_column=original_column,
                         note=note))


# A: the definitions are taken from saved methods/configuration contracts.
csum = json.loads((CRC / 'SUMMARY.json').read_text())
psingle = json.loads((PDAC / 'tables/singlecell_contract.json').read_text())
ppair = json.loads((PDAC / 'tables/selected_single_cluster_pair.json').read_text())
pref = json.loads((PDAC / 'figures_fig7style/source_data/PDAC_joint_atlas_plot_summary.json').read_text())
cref = read_table(REF / 'module_subtype_composition.csv')
cref = cref[cref.cm.eq('joint_01_sharedCM')].copy()
cc_tnk = float(cref.loc[cref.lineage.isin(['T', 'NK']), 'loading_fraction'].sum())
cr_tnk = csum['selected_CM']['CM1_T_NK_H_share']
pd_tnk = pref['tnk_H_fraction']
design = pd.DataFrame([
    dict(cancer='ccRCC', JUN_pole='Epi_JUN', negative_reference_pole='Epi_VIM',
         module='joint_01_sharedCM', K=12, T_NK_H_share=cc_tnk, pseudocount=0.005,
         denominator='all tumor epithelial cells per saved source sample',
         pole_definition='one frozen expression-derived state per pole'),
    dict(cancer='CRC', JUN_pole=csum['selected_pair']['pair']['Epi_JUN'],
         negative_reference_pole=csum['selected_pair']['pair']['Epi_VIM'],
         module=csum['selected_CM']['CM1_like'], K=csum['selected_CM']['K'],
         T_NK_H_share=cr_tnk, pseudocount=0.005,
         denominator='all cleaned tumor-source epithelial cells per eligible sample',
         pole_definition='one frozen expression-derived cluster per pole'),
    dict(cancer='PDAC', JUN_pole=ppair['JUN'], negative_reference_pole=ppair['VIM'],
         module=psingle['CM1'], K=11, T_NK_H_share=pd_tnk, pseudocount=0.005,
         denominator='all cleaned tumor-associated ductal cells per eligible sample',
         pole_definition='one frozen expression-derived cluster per pole')])
save(design, 'A_design_definition.csv',
     f'{REF}/cohort_summary.json; {CRC}/SUMMARY.json; {CRC}/singlecell/methods.json; '
     f'{PDAC}/tables/singlecell_contract.json; {PDAC}/tables/cm_calibration_contract.json',
     'frozen pair; K; CM1; balance definition', 'Saved definitions, no new selection')
for ca in ['ccRCC', 'CRC', 'PDAC']:
    mapping('A.' + ca + '.poles_and_definition', 'A_design_definition.csv',
            'JUN_pole;negative_reference_pole;pseudocount', f'cancer={ca}',
            provenance[-1]['original_file'], 'frozen pair;balance definition')

# B and the ccRCC row of F: exact source statistics and a second saved cross-check.
ks = read_table(OLD / 'C_single_cell_tumors.csv')
ks = ks[ks.cancer.eq('ccRCC')].copy()
ks = ks[['cancer', 'sample', 'positive_fraction', 'negative_fraction', 'balance', 'CM1_activity']]
ks.rename(columns={'sample': 'sample_id', 'positive_fraction': 'JUN_fraction',
                   'negative_fraction': 'reference_fraction'}, inplace=True)
save(ks, 'B_ccRCC_single_cell_tumors.csv', OLD / 'C_single_cell_tumors.csv',
     ';'.join(ks.columns), 'ccRCC rows only; descriptive column renaming')
ka = read_table(OLD / 'C_association_statistics.csv')
ka = ka[ka.cancer.eq('ccRCC')]
kr = read_table(REF / 'single_cell_associations.csv')
korig = Path('/mnt/disk18t/lr_xcy/riku/codex_research/CM_analysis_2/cm_epi_analysis/'
             'balanced_joint_nmf_outputs/joint_cm/epi_cm_coupling/'
             'balanced_joint_cm_epi_cm_association_tumor.csv')
koriginal = read_table(korig)
associations = []
for metric, feature, state in [('positive_fraction', 'JUN_fraction', 'Epi_JUN'),
                               ('negative_fraction', 'reference_fraction', 'Epi_VIM'),
                               ('balance', 'balance', None)]:
    a = ka[ka.metric.eq(metric)].iloc[0]
    q = np.nan
    if state:
        b = kr[kr.cm.eq('joint_01_sharedCM') & kr.epithelial_state.eq(state)].iloc[0]
        o = koriginal[koriginal.CM.eq('joint_01_sharedCM') & koriginal.epi_subtype.eq(state)].iloc[0]
        check('ccRCC ' + state + ' effect/old/reference/original agree',
              np.allclose([a.rho, a.p], [b.rho, b.p_value], rtol=1e-12, atol=1e-16)
              and np.allclose([a.rho, b.q_value], [o.rho, o.q], rtol=1e-12, atol=1e-16))
        q = b.q_value
    r, p = stats.spearmanr(ks[feature], ks.CM1_activity)
    check('ccRCC ' + feature + ' sample verification',
          np.allclose([r, p], [a.rho, a.p], rtol=1e-12, atol=1e-16))
    associations.append(dict(cancer='ccRCC', unit='sample', method='raw', feature=feature,
                             n=int(a.n), n_patients=np.nan, n_studies=np.nan,
                             rho=a.rho, p=a.p, q=q, q_scope='108 state-module tests' if state else 'not supplied',
                             original_file=str(REF / 'single_cell_associations.csv') if state else str(OLD / 'C_association_statistics.csv'),
                             original_rho_column='rho', original_p_column='p_value' if state else 'p',
                             original_q_column='q_value' if state else 'not supplied'))
be = pd.DataFrame(associations[:2])
save(be, 'B_ccRCC_reference_effects.csv', REF / 'single_cell_associations.csv',
     'epithelial_state;rho;p_value;q_value;n_samples', 'joint_01_sharedCM, frozen two poles only')
for f in ['JUN_fraction', 'reference_fraction']:
    mapping('B.' + f, 'B_ccRCC_reference_effects.csv', 'rho;q;n', f'feature={f}',
            REF / 'single_cell_associations.csv', 'rho;q_value;n_samples', 'raw Spearman; q over 108 tests')

# Review-revision reference uses the same fixed H; its different expression-
# continuum/CLR association is NOT substituted for the fraction/NNLS estimand.
review = KIRC / 'review_revision_20260819/results/cm1_module_preservation/fixed_cm1_reference_loadings.csv'
rv = read_table(review).set_index('cell_subtype')
cv = cref.set_index('subtype')
check('ccRCC CM1 H agrees with review_revision_20260819',
      set(rv.index) == set(cv.index) and
      np.allclose(rv.loading_fraction.sort_index(), cv.loading_fraction.sort_index(), atol=2e-14))

# C/D: retain the original cell identifiers, UMAP coordinates and labels verbatim.
for panel, ca, root, prefix, expected, jn, vn in [
    ('C', 'CRC', CRC, 'CRC_joint', 643655, 4855, 7748),
    ('D', 'PDAC', PDAC, 'PDAC_joint_atlas', 392388, 3245, 5227)]:
    src = root / f'figures_fig7style/source_data/{prefix}_atlas_per_cell_plot_data.csv.gz' if ca == 'CRC' else root / f'figures_fig7style/source_data/{prefix}_per_cell_plot_data.csv.gz'
    target = f'{panel}_{ca}_joint_atlas_per_cell_plot_data.csv.gz'
    copy(src, target)
    d = read_table(src, low_memory=False)
    check(ca + ' full atlas cell count', len(d) == expected, len(d))
    check(ca + ' unique cells/finite real coordinates', d.cell_id.is_unique and np.isfinite(d[['UMAP1', 'UMAP2']]).all().all())
    check(ca + ' frozen pole counts', (int(d.is_JUN_pole.sum()), int(d.is_VIM_pole.sum())) == (jn, vn))
    check(ca + ' poles do not overlap', not (d.is_JUN_pole & d.is_VIM_pole).any())
    save(pd.DataFrame([dict(cancer=ca, population='full_joint_atlas', n_cells=len(d)),
                       dict(cancer=ca, population='frozen_JUN_pole', n_cells=int(d.is_JUN_pole.sum())),
                       dict(cancer=ca, population='frozen_reference_pole', n_cells=int(d.is_VIM_pole.sum()))]),
         f'{panel}_{ca}_atlas_counts.csv', src, 'all cell rows;is_JUN_pole;is_VIM_pole',
         'Row count and frozen Boolean column sums, verified against saved primary membership labels')
    labels_path = root / ('epithelial/labels_hvg3000_r0.4.csv.gz' if ca == 'CRC' else 'labels/A_HVG3000_sample_r0.4.csv.gz')
    labels = read_table(labels_path, low_memory=False)
    pnames = (csum['selected_pair']['pair']['Epi_JUN'], csum['selected_pair']['pair']['Epi_VIM']) if ca == 'CRC' else ('c7', 'c2')
    for boolean, label in zip(['is_JUN_pole', 'is_VIM_pole'], pnames):
        check(ca + ' exact per-cell membership ' + label,
              set(d.loc[d[boolean], 'cell_id']) == set(labels.loc[labels.cluster_id.eq(label), 'cell_id']))
    lcsrc = root / f'figures_fig7style/source_data/{prefix}_lineage_counts.csv'
    lc = read_table(lcsrc)
    counts = d.groupby('broad_lineage').size()
    check(ca + ' lineage counts agree with coordinates',
          all(int(counts.get(row.broad_lineage, 0)) == row.n_cells for row in lc.itertuples())
          and int(lc.n_cells.sum()) == expected)
    save(lc, f'{panel}_{ca}_lineage_counts.csv', lcsrc, 'broad_lineage;n_cells', 'No lineage inferred or reassigned')
    if ca == 'CRC':
        cp = pd.DataFrame([dict(cancer=ca, cluster_id=r['cluster_id'], short_label=r['cluster_id'].split('_')[-1],
                                pole='JUN' if r['reference_state'] == 'Epi_JUN' else 'negative reference',
                                n_cells=r['n_cells'], n_samples=r['n_samples'], n_studies=r['n_studies'])
                           for r in csum['primary_maxima'] if r['definition'] == 'archived_core12x2'])
        check('CRC 15 frozen epithelial clusters', labels.cluster_id.nunique() == 15)
        top = d[d.is_CM1_top5_subtype]
        check('CRC actual top-five CM subtype cell count', len(top) == 68508, len(top))
    else:
        cp = pd.DataFrame([dict(cancer=ca, cluster_id=ppair[arm], short_label=ppair[arm],
                                pole='JUN' if arm == 'JUN' else 'negative reference',
                                n_cells=ppair['similarities'][arm]['n_cells'],
                                n_samples=ppair['similarities'][arm]['n_tumor_samples'],
                                n_studies=int(labels.loc[labels.cluster_id.eq(ppair[arm]), 'study'].nunique()))
                           for arm in ['JUN', 'VIM']])
    save(cp, f'{panel}_{ca}_frozen_poles.csv', CRC / 'SUMMARY.json' if ca == 'CRC' else PDAC / 'tables/selected_single_cluster_pair.json',
         'primary_maxima / similarities; labels study', 'No cluster or pole changed')
    # View windows change only display bounds, not coordinates or membership.
    a = d[['UMAP1', 'UMAP2']].to_numpy()
    lo, hi = a.min(axis=0), a.max(axis=0)
    pad = (hi - lo) * .025
    if ca == 'CRC':
        poles = d[d.is_JUN_pole | d.is_VIM_pole][['UMAP1', 'UMAP2']].to_numpy()
        zl, zh = poles.min(axis=0), poles.max(axis=0)
        zp = (zh - zl) * .08
        zl, zh = zl-zp, zh+zp
        rule = 'all frozen-pole coordinate extremes plus 8% display margin; no point selection for effect'
    else:
        z = read_table(PDAC / 'figures_fig7style/source_data/PDAC_joint_atlas_frozen_pair_zoom_windows.csv')
        zl = np.array([z.UMAP1_min.min(), z.UMAP2_min.min()])
        zh = np.array([z.UMAP1_max.max(), z.UMAP2_max.max()])
        rule = 'union of the two already saved frozen-pole zoom windows; full atlas always contains every cell'
    windows = pd.DataFrame([
        dict(view='full', x_min=lo[0]-pad[0], x_max=hi[0]+pad[0], y_min=lo[1]-pad[1], y_max=hi[1]+pad[1], rule='all saved coordinates plus 2.5% margin'),
        dict(view='epithelial_zoom', x_min=zl[0], x_max=zh[0], y_min=zl[1], y_max=zh[1], rule=rule)])
    save(windows, f'{panel}_{ca}_view_windows.csv', src, 'UMAP1;UMAP2', rule)
    mapping(panel + '.' + ca + '.UMAP_and_highlights', target,
            'cell_id;UMAP1;UMAP2;is_JUN_pole;is_VIM_pole', 'all cells, unchanged', src,
            'same columns', 'All cells rendered; no downsampling, graph fitting or new embedding')
    mapping(panel + '.' + ca + '.lineage_counts', f'{panel}_{ca}_lineage_counts.csv',
            'broad_lineage;n_cells', 'all saved rows', lcsrc, 'broad_lineage;n_cells')
    mapping(panel + '.' + ca + '.pole_counts', f'{panel}_{ca}_frozen_poles.csv',
            'n_cells;n_samples;n_studies', 'both frozen poles', provenance[-2]['original_file'],
            'n_cells;n_samples / n_tumor_samples; study')
    del d, labels

# E/F: saved sample and patient results, including all diagnostic arms.
cr = read_table(CRC / 'singlecell/ALL_exact_pair_associations.csv')
cr = cr[cr.min_epi.eq(30) & cr.CM_activity.eq('CM1_raw')].copy()
for a in cr.itertuples():
    associations.append(dict(cancer='CRC', unit=a.unit,
                             method='raw' if a.method == 'raw_Spearman' else 'study-adjusted',
                             feature={'f_Epi_JUN': 'JUN_fraction', 'f_Epi_VIM': 'reference_fraction', 'balance': 'balance'}[a.feature],
                             n=a.n, n_patients=a.n_patients, n_studies=a.n_studies,
                             rho=a.effect, p=a.p, q=a.q_BH_primary_6_tests,
                             q_scope='six primary sample tests' if a.unit == 'sample' else 'not supplied in primary six-test family',
                             original_file=str(CRC / 'singlecell/ALL_exact_pair_associations.csv'),
                             original_rho_column='effect', original_p_column='p', original_q_column='q_BH_primary_6_tests'))
pr = read_table(PDAC / 'tables/balance_focused_singlecell_primary.csv')
for a in pr.itertuples():
    associations.append(dict(cancer='PDAC', unit=a.unit,
                             method='raw' if a.adjustment == 'none' else 'study-adjusted',
                             feature='reference_fraction' if a.metric == 'VIM_fraction' else a.metric,
                             n=a.n, n_patients=np.nan, n_studies=np.nan,
                             rho=a.rho, p=a.p, q=a.q_BH_within_unit_threshold_context,
                             q_scope='six tests per unit/threshold/context',
                             original_file=str(PDAC / 'tables/balance_focused_singlecell_primary.csv'),
                             original_rho_column='rho', original_p_column='p', original_q_column='q_BH_within_unit_threshold_context'))
assoc = pd.DataFrame(associations)
save(assoc, 'E_F_single_cell_association_statistics.csv',
     f'{REF}/single_cell_associations.csv; {OLD}/C_association_statistics.csv; '
     f'{CRC}/singlecell/ALL_exact_pair_associations.csv; {PDAC}/tables/balance_focused_singlecell_primary.csv',
     'effect/rho;p;q;n;unit;method', 'Frozen primary threshold and raw unweighted NNLS activity; no q recalculation')
copy(CRC / 'singlecell/ALL_exact_pair_associations.csv', 'E_CRC_all_saved_associations.csv', 'All saved thresholds/units/activity scales preserved for transparency')
copy(PDAC / 'tables/balance_focused_singlecell_sensitivity_epi50.csv', 'E_PDAC_saved_epi50_sensitivity.csv', 'Existing sensitivity, not substituted for the primary sample result')
copy(CRC / 'singlecell/methods.json', 'E_CRC_methods.json')
copy(PDAC / 'tables/singlecell_contract.json', 'E_PDAC_methods.json')

cp = read_table(CRC / 'singlecell/exact_pair_sample_fractions_and_CM.csv')
cp = cp[cp.status.eq('tumor') & (cp.n_epi >= 30) & (cp.n_non_epi >= 50)]
cp = cp.rename(columns={'f_Epi_JUN': 'JUN_fraction', 'f_Epi_VIM': 'reference_fraction',
                        'CM1_raw': 'CM1_activity', 'donor_id_v1': 'patient_key'})
cp['cancer'] = 'CRC'
pp = read_table(PDAC / 'tables/balance_primary_per_sample.csv')
pp = pp[pp.primary_eligible].copy()
pp = pp.rename(columns={'study': 'study_id', 'donor': 'patient_key',
                        'VIM_fraction': 'reference_fraction', 'Balance': 'balance', 'CM1_W': 'CM1_activity'})
pp['cancer'] = 'PDAC'
cols = ['cancer', 'sample_id', 'study_id', 'patient_key', 'n_epi', 'n_non_epi',
        'JUN_fraction', 'reference_fraction', 'balance', 'CM1_activity']
points = pd.concat([cp[cols], pp[cols]], ignore_index=True)
save(points, 'E_balance_vs_CM1_per_sample.csv',
     f'{CRC}/singlecell/exact_pair_sample_fractions_and_CM.csv; {PDAC}/tables/balance_primary_per_sample.csv',
     'sample_id;study_id;donor_id_v1/donor;f_Epi_JUN/JUN_fraction;f_Epi_VIM/VIM_fraction;balance/Balance;CM1_raw/CM1_W',
     'Saved primary eligibility only; descriptive column renaming; all eligible rows retained')
save(points.groupby('cancer').agg(n_samples=('sample_id', 'size'), n_patient_keys=('patient_key', 'nunique'),
                                 n_studies=('study_id', 'nunique')).reset_index(),
     'E_primary_cohort_counts.csv', 'source_data/E_balance_vs_CM1_per_sample.csv',
     'sample_id;patient_key;study_id', 'Counts of all saved eligible samples and available keys')
fitrows = []
for ca, n in [('CRC', 127), ('PDAC', 74)]:
    d = points[points.cancer.eq(ca)]
    check(ca + ' primary sample count', len(d) == n)
    check(ca + ' fixed pseudocount balance agrees with source',
          np.allclose(d.balance, np.log((d.JUN_fraction+.005)/(d.reference_fraction+.005)), atol=2e-13))
    X = np.column_stack([np.ones(n), pd.get_dummies(d.study_id, drop_first=True).to_numpy(dtype=float)])
    for f in ['JUN_fraction', 'reference_fraction', 'balance']:
        for method in ['raw', 'study-adjusted']:
            a = assoc[assoc.cancer.eq(ca) & assoc.unit.eq('sample') & assoc.feature.eq(f) & assoc.method.eq(method)].iloc[0]
            if method == 'raw':
                rho, p = stats.spearmanr(d[f], d.CM1_activity)
            else:
                rx, ry = stats.rankdata(d[f]), stats.rankdata(d.CM1_activity)
                rx = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
                ry = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
                rho = np.corrcoef(rx, ry)[0, 1]
                df = n - np.linalg.matrix_rank(X) - 1
                p = 2*stats.t.sf(abs(rho)*np.sqrt(df/(1-rho*rho)), df)
            check(ca + ' ' + f + ' ' + method + ' statistic verification',
                  np.allclose([rho, p], [a.rho, a.p], rtol=2e-11, atol=2e-15))
            mapping(f'E/F.{ca}.sample.{method}.{f}', 'E_F_single_cell_association_statistics.csv',
                    'rho;p;q;n', f'cancer={ca};unit=sample;method={method};feature={f}',
                    a.original_file, f'{a.original_rho_column};{a.original_p_column};{a.original_q_column};n',
                    a.q_scope)
    x, y = d.balance.to_numpy(), d.CM1_activity.to_numpy()
    fit = stats.linregress(x, y)
    xx = np.linspace(x.min(), x.max(), 180)
    yy = fit.intercept + fit.slope*xx
    residual_sd = np.sqrt(np.sum((y-fit.intercept-fit.slope*x)**2)/(n-2))
    se = residual_sd*np.sqrt(1/n+(xx-x.mean())**2/np.sum((x-x.mean())**2))
    ci = stats.t.ppf(.975, n-2)*se
    fitrows.extend(dict(cancer=ca, balance=b, fitted_CM1=m, CI_low=lo, CI_high=hi,
                        method='descriptive unadjusted OLS; pointwise t 95% mean CI, not a rank-correlation CI')
                   for b, m, lo, hi in zip(xx, yy, yy-ci, yy+ci))
save(pd.DataFrame(fitrows), 'E_descriptive_OLS_line_and_95CI.csv',
     'source_data/E_balance_vs_CM1_per_sample.csv', 'balance;CM1_activity',
     'New descriptive display line only; no inference statistic changed or selected')
save(pd.DataFrame([dict(cancer=ca, fit='unadjusted OLS display only', confidence_level=.95,
                        CI_type='pointwise t-based confidence interval for conditional mean',
                        n=len(points[points.cancer.eq(ca)])) for ca in ['CRC', 'PDAC']]),
     'E_display_fit_definition.csv', 'source_data/E_balance_vs_CM1_per_sample.csv',
     'balance;CM1_activity', 'Descriptive visual fit; not a replacement for saved Spearman tests')
mapping('E.scatter_points', 'E_balance_vs_CM1_per_sample.csv', 'balance;CM1_activity;sample_id',
        'all 127 CRC and 74 PDAC primary samples',
        f'{CRC}/singlecell/exact_pair_sample_fractions_and_CM.csv; {PDAC}/tables/balance_primary_per_sample.csv',
        'balance/Balance;CM1_raw/CM1_W;sample_id')
mapping('E.descriptive_fit_and_CI', 'E_descriptive_OLS_line_and_95CI.csv',
        'balance;fitted_CM1;CI_low;CI_high', 'each cancer', 'source_data/E_balance_vs_CM1_per_sample.csv',
        'balance;CM1_activity', 'OLS mean confidence band is descriptive, not a Spearman confidence interval')

# F: full H compositions, plus top-five T/NK subtypes without renormalizing.
comp = []
for row in cref.itertuples():
    comp.append(dict(cancer='ccRCC', module='joint_01_sharedCM', subtype=row.subtype,
                     lineage='T_NK' if row.lineage in ['T', 'NK'] else row.lineage,
                     H_loading=row.loading, H_share=row.loading_fraction, n_cells=np.nan,
                     original_file=str(REF / 'module_subtype_composition.csv')))
crh = read_table(CRC / 'CM/CM1_composition_marker_evidence.csv')
for row in crh.itertuples():
    comp.append(dict(cancer='CRC', module='CRCcal_CM08', subtype=row.subtype_id,
                     lineage=row.lineage, H_loading=row.H_share, H_share=row.H_share, n_cells=row.n_cells,
                     original_file=str(CRC / 'CM/CM1_composition_marker_evidence.csv')))
ph = read_table(PDAC / 'figures_fig7style/source_data/PDAC_joint_atlas_subtype_to_CM1_loading.csv')
for row in ph.itertuples():
    comp.append(dict(cancer='PDAC', module='calCM02', subtype=row.non_epi_subtype,
                     lineage='T_NK' if row.lineage_original == 'TNK' else row.lineage_original,
                     H_loading=row.H_loading, H_share=row.H_fraction, n_cells=row.n_cells,
                     original_file=str(PDAC / 'figures_fig7style/source_data/PDAC_joint_atlas_subtype_to_CM1_loading.csv')))
comp = pd.DataFrame(comp)
comp['displayed_top5_TNK'] = False
for ca, expected in [('ccRCC', cc_tnk), ('CRC', cr_tnk), ('PDAC', pd_tnk)]:
    d = comp[comp.cancer.eq(ca)]
    check(ca + ' full H fractions sum to one', np.isclose(d.H_share.sum(), 1, atol=2e-12))
    check(ca + ' T/NK H share matches saved summary',
          np.isclose(d.loc[d.lineage.eq('T_NK'), 'H_share'].sum(), expected, atol=2e-12))
    indices = d[d.lineage.eq('T_NK')].nlargest(5, 'H_share').index
    comp.loc[indices, 'displayed_top5_TNK'] = True
save(comp, 'F_full_module_subtype_H_loadings.csv',
     f'{REF}/module_subtype_composition.csv; {CRC}/CM/CM1_composition_marker_evidence.csv; '
     f'{PDAC}/figures_fig7style/source_data/PDAC_joint_atlas_subtype_to_CM1_loading.csv',
     'subtype/subtype_id/non_epi_subtype;lineage;loading_fraction/H_share/H_fraction',
     'All H components retained; displayed top-five T/NK are not renormalized')
save(design[['cancer', 'module', 'K', 'T_NK_H_share']], 'F_module_TNK_shares.csv',
     'source_data/F_full_module_subtype_H_loadings.csv; source_data/A_design_definition.csv',
     'sum H_share of T/NK subtypes; saved K', 'H loading fraction, NOT a measured cell-count fraction')
for ca in ['ccRCC', 'CRC', 'PDAC']:
    mapping('F.' + ca + '.T_NK_share', 'F_module_TNK_shares.csv', 'T_NK_H_share;K',
            f'cancer={ca}', 'source_data/F_full_module_subtype_H_loadings.csv',
            'sum H_share where lineage=T_NK; saved K', 'Full module H denominator')
    mapping('F.' + ca + '.top5_TNK', 'F_full_module_subtype_H_loadings.csv', 'subtype;H_share',
            f'cancer={ca};displayed_top5_TNK=True',
            comp.loc[comp.cancer.eq(ca), 'original_file'].iloc[0],
            'loading_fraction' if ca == 'ccRCC' else 'H_share' if ca == 'CRC' else 'H_fraction')
mapping('F.ccRCC.balance', 'E_F_single_cell_association_statistics.csv', 'rho;p;n',
        'cancer=ccRCC;feature=balance;method=raw;unit=sample', OLD / 'C_association_statistics.csv',
        'rho;p;n', 'Existing saved fraction-ratio result, verified without replacing it')
for ca in ['CRC', 'PDAC']:
    a = assoc[assoc.cancer.eq(ca) & assoc.unit.eq('patient') & assoc.feature.eq('balance')]
    for row in a.itertuples():
        mapping('F.' + ca + '.patient_balance.' + row.method,
                'E_F_single_cell_association_statistics.csv', 'rho;p;n',
                f'cancer={ca};unit=patient;feature=balance;method={row.method}',
                row.original_file, f'{row.original_rho_column};p;n')

save(pd.DataFrame(provenance), 'SOURCE_PROVENANCE.csv', 'All original files listed',
     'all', 'No original source file modified')
pd.DataFrame(mappings).to_csv(DATA / 'FIGURE_ELEMENT_TO_SOURCE.csv', index=False)
(OUT / 'numeric_verification.json').write_text(json.dumps(dict(
    checks=checks, all_passed=all(c['passed'] for c in checks),
    reused_only='no clustering, dimension reduction, embedding, NMF, pole or module selection performed',
    statistical_estimates='All displayed rank statistics/q values read from saved files; independently checked but not replaced',
    scope='A-F single-cell only; no clinical rendering'), indent=2) + '\n')
(OUT / 'SOURCE_DISCREPANCIES_AND_DECISIONS.md').write_text('''# Source discrepancies resolved explicitly

- **CRC arm estimates:** +0.329927 and −0.323423 are **study-adjusted**, not raw. Raw sample effects are +0.306177 and −0.307155. Both methods are labelled and shown. CRC patient balance +0.412057 (p=5.454e-5) is study-adjusted; the raw patient result is +0.453293 (p=1.717e-6). Exact digits are in the saved association table.
- **CRC top-five subtype count:** the per-cell atlas and CM table both give **68,508** cells for CRCfa1TNK_r06_05/04/06/00/02, not the brief's 98,500. Their H shares sum to 67.4284%. The main panel displays exact H loadings, not an unverified cell count. Full source rows are retained.
- **PDAC T_GZMK:** the saved H fraction is 0.108452 (10.8452%); the figure rounds this to **10.8%**, not 10.9%.
- **ccRCC review cross-check:** the full CM1 H fractions agree with `review_revision_20260819/results/cm1_module_preservation/fixed_cm1_reference_loadings.csv`. The revision's expression-continuum/CLR statistics are different estimands and are not substituted for fraction/NNLS results.
- **UMAPs:** complete original joint-atlas coordinates and calibrated pole memberships are retained. CRC's table uses its supplied `broad_lineage` counts; no label is silently harmonized to `panel_B_group`. PDAC's zero explicit Plasma row is retained; B_MZB1 remains in the supplied B lineage. Unassigned/unresolved cells remain grey, not relabelled.
- **Scope update:** the main figure contains only single-cell panels A–F. Clinical results are neither plotted nor exported here; the future supplementary hook is documented in `SUPPLEMENT_PLAN.md`.
''')
print(f'Prepared {len(provenance)} source entries; {len(checks)} numerical checks passed. Output: {OUT}')
