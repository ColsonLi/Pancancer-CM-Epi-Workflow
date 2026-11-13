"""Fig7 JUN-only upstream entry points; importing this file does not run analyses.

Archived scientific source is kept byte-for-byte. Its original __file__ is
supplied at execution so ROOT/B and all data/output paths retain their meaning.
Only explicitly selected functions are called; old multimodality main functions
are never used for the current single-cell figure.
"""
from contextlib import contextmanager
from pathlib import Path
import ast
import hashlib
import json
import sys
import types

BUNDLE = Path(__file__).resolve().parent
PROJECT = Path('/mnt/disk18t/lr_xcy/riku/kirc_205')
VALIDATION = Path('/mnt/my_storage/data/lr_riku/validation_4_kirc_agents')
FIGURE = PROJECT / 'final/fig7_balance_cross_cancer_20261107'
CRC = VALIDATION / 'crc_codex/calibrated_single_cluster_20261107'
PDAC = VALIDATION / 'pdac_codex/calibrated_single_cluster_20261107'


def manifest():
    return json.loads((BUNDLE / 'SOURCE_MANIFEST.json').read_text())


def load_source(key, alias=None, functions_only=None):
    """Load the bundled copy, anchoring relative paths at its original location."""
    entry = next(item for item in manifest() if item['bundled_file'] == key)
    path = BUNDLE / key
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError(f'Archived source changed: {key}; review manifest first')
    name = alias or '_fig7_' + hashlib.sha256(key.encode()).hexdigest()[:12]
    module = types.ModuleType(name)
    module.__file__ = entry['original_file']
    if alias:
        sys.modules[alias] = module
    tree = ast.parse(raw.decode(), filename=module.__file__)
    if functions_only is not None:
        # Exact original function bodies, without unrelated imports/main calls.
        tree.body = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name in functions_only]
        if {node.name for node in tree.body} != set(functions_only):
            raise ValueError(f'Missing selected function in {key}')
    exec(compile(tree, module.__file__, 'exec'), module.__dict__)
    return module


@contextmanager
def environment(group):
    """Keep the distinct CRC/PDAC common.py modules from leaking across stages."""
    aliases = ['common', 'analysis_common', 'cal_common', 'full_common', 'expression_engine']
    previous = {name: sys.modules.get(name) for name in aliases}
    old_path = sys.path[:]
    old_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        for name in aliases:
            sys.modules.pop(name, None)
        if group in ('crc', 'crc_reference', 'crc_atlas'):
            load_source('crc/epi_expression_reintegration/analysis_common.py', 'analysis_common')
        if group == 'crc':
            load_source('crc/calibrated_single_cluster_20261107/cal_common.py', 'cal_common')
        elif group == 'crc_legacy':
            load_source('crc/cm_epi_redesign/common.py', 'common')
        elif group == 'crc_atlas':
            load_source('crc/full_atlas_reintegration_20261107/full_common.py', 'full_common')
            load_source('crc/full_atlas_reintegration_20261107/expression_engine.py', 'expression_engine')
        elif group == 'pdac':
            load_source('pdac/calibrated_single_cluster_20261107/scripts/common.py', 'common')
        elif group == 'pdac_legacy':
            load_source('pdac/cm_epi_redesign/scripts/common.py', 'common')
        yield
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


# Original stage functions, unchanged. Helper scripts are bundled, not merely linked.
STAGES = {
    'CRC.reviewed_clean_mask': ('crc_legacy', 'crc/cm_epi_redesign/01_prepare_clean_data.py', 'main', ()),
    'CRC.background_atlas': ('crc_atlas', 'crc/full_atlas_reintegration_20261107/01_full_atlas.py', 'main', ()),
    'CRC.non_epithelial_reclustering': ('crc_atlas', 'crc/full_atlas_reintegration_20261107/02_lineage_reclustering.py', 'main', ('non_epithelial',)),
    'CRC.freeze_reference': ('crc_reference', 'crc/epi_expression_reintegration/02_freeze_ccrcc_reference.py', 'main', ()),
    'CRC.prepare': ('crc', 'crc/calibrated_single_cluster_20261107/01_prepare.py', 'main', ()),
    'CRC.CM': ('crc', 'crc/calibrated_single_cluster_20261107/02_CM.py', 'main', ()),
    'CRC.epithelial_clustering': ('crc', 'crc/calibrated_single_cluster_20261107/03_expression_clustering.py', 'main', ()),
    'CRC.reference_matching': ('crc', 'crc/calibrated_single_cluster_20261107/04_reference_and_similarity.py', 'main', ()),
    'CRC.associations': ('crc', 'crc/calibrated_single_cluster_20261107/05_singlecell.py', 'main', ()),
    'PDAC.archived_ductal_preparation': ('pdac_legacy', 'pdac/cm_epi_redesign/scripts/02_epithelial_sweep.py', 'main', ()),
    'PDAC.freeze_reference': ('pdac', 'pdac/calibrated_single_cluster_20261107/scripts/00_ccRCC_reference.py', 'main', ()),
    'PDAC.cleanup_evidence': ('pdac', 'pdac/calibrated_single_cluster_20261107/scripts/00_cleanup_evidence.py', 'main', ()),
    'PDAC.CM': ('pdac', 'pdac/calibrated_single_cluster_20261107/scripts/01_cm_calibration.py', 'main', ()),
    'PDAC.epithelial_clustering': ('pdac', 'pdac/calibrated_single_cluster_20261107/scripts/02_expression_sweep.py', 'main', ()),
    'PDAC.reference_matching': ('pdac', 'pdac/calibrated_single_cluster_20261107/scripts/03_reference_matching.py', 'main', ()),
    'CRC.atlas_plot_data': ('none', 'crc/calibrated_single_cluster_20261107/figures_fig7style/render_crc_joint_atlas_fig7style.py', 'prepare_data', ()),
    'PDAC.atlas_plot_data': ('none', 'pdac/calibrated_single_cluster_20261107/figures_fig7style/render_pdac_joint_atlas_fig7style.py', 'prepare_data', ()),
}


def run_stage(stage, *, execute=False):
    """Dry run by default. Execution can overwrite original analysis outputs."""
    if stage == 'PDAC.associations':
        if not execute:
            return {'stage': stage, 'function': '04_three_modalities.py:singlecell only', 'executed': False}
        with environment('pdac'):
            common = sys.modules['common']
            module = load_source('pdac/calibrated_single_cluster_20261107/scripts/04_three_modalities.py',
                                 functions_only={'singlecell'})
            module.__dict__.update({k: v for k, v in vars(common).items() if not k.startswith('_')})
            from scipy.optimize import nnls
            module.nnls = nnls
            pair = json.loads((PDAC / 'tables/selected_single_cluster_pair.json').read_text())
            cm = json.loads((PDAC / 'tables/cm_calibration_contract.json').read_text())
            module.singlecell(pair, cm)
        return {'stage': stage, 'executed': True}
    group, key, function, args = STAGES[stage]
    if not execute:
        return {'stage': stage, 'source': key, 'function': function, 'args': args, 'executed': False}
    from threadpoolctl import threadpool_limits
    threads = 6 if stage in ('CRC.prepare', 'CRC.epithelial_clustering') else 4 if group == 'crc' else 8
    with environment(group), threadpool_limits(limits=threads):
        module = load_source(key)
        # CRC prepare_data's original caller creates DATA before invoking it.
        if stage.endswith('atlas_plot_data'):
            module.DATA.mkdir(parents=True, exist_ok=True)
        getattr(module, function)(*args)
    return {'stage': stage, 'executed': True}


def preflight():
    """Read-only integrity, source provenance and input-boundary checks."""
    rows = []
    for item in manifest():
        path = BUNDLE / item['bundled_file']
        ok = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        rows.append({'kind': 'bundled_source', 'path': str(path), 'exists_and_valid': ok})
    inputs = [
        VALIDATION / 'crc_codex/epi-cm-core-workflow/h5ad/02-cell_subtype_integration_clustering/06-project-subtypes-to-full-adata/adata_anno_cellsubtype.h5ad',
        VALIDATION / 'crc_codex/full_atlas_reintegration_20261107/all_cell_labels.csv.gz',
        VALIDATION / 'crc_codex/full_atlas_reintegration_20261107/lineages/T_NK/per_cell_labels.csv.gz',
        VALIDATION / 'crc_codex/full_atlas_reintegration_20261107/all_cell_final_labels.csv.gz',
        VALIDATION / 'crc_codex/full_atlas_reintegration_20261107/atlas/all_cells_expression_integrated.h5ad',
        VALIDATION / 'crc_codex/cm_epi_redesign/spatial/panel_genes.csv',
        VALIDATION / 'pdac_codex/cm_epi_redesign/objects/cleaned_tumor_ductal.h5ad',
        VALIDATION / 'pdac_codex/full_atlas_reintegration_20261107/tables/full_atlas_new_labels.csv.gz',
        VALIDATION / 'pdac_codex/full_atlas_reintegration_20261107/objects/full_atlas_verified_embeddings.h5ad',
        VALIDATION / 'external_atlas_for_cm-epi_validation/PDAC/UCSC/metadata_samples.csv',
        PROJECT / 'leiden_detailed/adata_epi.h5ad',
        VALIDATION / 'shared_clinical_and_signatures/reference/CM1_CM10_complete_subtype_loadings.csv',
    ]
    pdac_core = VALIDATION / 'pdac_codex/epi-cm-core-workflow/h5ad/02-cell_subtype_integration_clustering/05-subtype-deg-annotation'
    for lineage, name in [('epithelial', 'epi'), ('t_nk', 't'), ('myeloid', 'mye'), ('b', 'b'),
                          ('stromal', 's'), ('endothelial', 'endo'), ('mast', 'mast')]:
        # Actual directory/object names are verified against the archived common script.
        paths = list(pdac_core.glob(f'{lineage}/adata_*.h5ad'))
        inputs.extend(paths or [pdac_core / lineage / f'adata_{name}.h5ad'])
    rows.extend({'kind': 'existing_input', 'path': str(path), 'exists_and_valid': path.exists()} for path in inputs)
    return rows


def export_pdac_singlecell_display_inputs(*, execute=False):
    """The single-cell slices of old finalization/report code; no clinical work."""
    if not execute:
        return {'stage': 'PDAC.singlecell_display_inputs', 'executed': False}
    import numpy as np
    import pandas as pd
    table = PDAC / 'tables'
    assoc = pd.read_csv(table / 'singlecell_exact_pair_primary_sensitivity.csv')
    for threshold, name in [(30, 'primary'), (50, 'sensitivity_epi50')]:
        data = assoc[assoc.min_epi.eq(threshold)].copy()
        data['estimand_role'] = np.where(data.metric.eq('balance'), 'PRIMARY_BALANCE', 'ARM_DIAGNOSTIC')
        data['metric_order'] = data.metric.map({'balance': 0, 'JUN_fraction': 1, 'VIM_fraction': 2})
        data.sort_values(['unit', 'adjustment', 'metric_order']).drop(columns='metric_order').to_csv(
            table / f'balance_focused_singlecell_{name}.csv', index=False)
    samples = pd.read_csv(table / 'singlecell_exact_pair_sample_metrics.csv', index_col=0)
    samples[['study', 'donor', 'status', 'n_epi', 'n_non_epi', 'JUN_fraction', 'VIM_fraction',
             'balance', 'CM1_W', 'primary_eligible']].rename(columns={'balance': 'Balance'}).to_csv(
                 table / 'balance_primary_per_sample.csv', index_label='sample_id')
    # Exact normal-reference selection from 05_verify_figures_report.py:report.
    flow = pd.read_csv(table / 'epithelial_population_flow.csv')
    normal = flow[flow.cell_subtype.isin(['Epi_GP2', 'Epi_PRSS1', 'Epi_CELA3B', 'Epi_FXYD2'])].pivot_table(
        index='cell_subtype', columns='status', values='n', fill_value=0).reset_index()
    normal['normal_source_fraction'] = normal['normal-like'] / (normal['normal-like'] + normal['tumor'])
    normal.to_csv(table / 'normal_epithelial_reference_population_counts.csv', index=False)
    cleanup = pd.read_csv(table / 'epithelial_cleanup_and_normal_reference.csv.gz', index_col=0, low_memory=False)
    cells = cleanup[cleanup.cell_subtype.isin(normal.cell_subtype)].copy()
    cells['reference_population'] = np.where(cells.cell_subtype.eq('Epi_FXYD2'), 'normal_dominated_FXYD2', 'acinar_reference')
    cells.rename(columns={'sample': 'sample_id'}).to_csv(PDAC / 'labels/normal_epithelial_reference.csv.gz', index_label='cell_id')
    return {'stage': 'PDAC.singlecell_display_inputs', 'executed': True}


def export_figure_inputs(*, execute=False):
    """Prepare only current JUN-only inputs, without old clinical/spatial results.

This is an explicit delivery adapter, not a replacement scientific analysis.
Similarity extraction reuses the original source-preparation script. The other
exports reproduce its column mappings/eligibility rules, retaining unrelated
cancers in shared tables when they already exist.
    """
    if not execute:
        return {'stage': 'Fig7.source_tables', 'executed': False}
    import shutil
    import numpy as np
    import pandas as pd
    from scipy import stats
    target = FIGURE / 'source_data'
    target.mkdir(parents=True, exist_ok=True)
    def recorded_path(path):
        # Match the original export's path spelling; both aliases resolve alike.
        return str(path).replace(str(VALIDATION), '/home/lr_xcy/lr_xcy/riku/lr_riku/validation_4_kirc_agents')
    with environment('none'):
        module = load_source('figure/similarity_panels/prepare_similarity_source_data.py')
        module.main()
        destination = target / 'similarity'
        destination.mkdir(exist_ok=True)
        for path in module.DATA.iterdir():
            if path.is_file():
                shutil.copyfile(path, destination / path.name)
    exported, points, compositions = [], [], []
    for cancer, root, panel, prefix in [('CRC', CRC, 'C', 'CRC_joint_atlas'),
                                         ('PDAC', PDAC, 'D', 'PDAC_joint_atlas')]:
        origin = root / f'figures_fig7style/source_data/{prefix}_per_cell_plot_data.csv.gz'
        shutil.copyfile(origin, target / f'{panel}_{cancer}_joint_atlas_per_cell_plot_data.csv.gz')
        cells = pd.read_csv(origin, float_precision='round_trip', low_memory=False)
        lo = cells[['UMAP1', 'UMAP2']].min().to_numpy()
        hi = cells[['UMAP1', 'UMAP2']].max().to_numpy()
        pad = (hi - lo) * .025
        if cancer == 'CRC':
            poles = cells[cells.is_JUN_pole | cells.is_VIM_pole][['UMAP1', 'UMAP2']]
            zl, zh = poles.min().to_numpy(), poles.max().to_numpy()
            zp = (zh - zl) * .08
            zl, zh = zl - zp, zh + zp
        else:
            centers = [cells.loc[cells[key], ['UMAP1', 'UMAP2']].median().to_numpy()
                       for key in ['is_JUN_pole', 'is_VIM_pole']]
            zl = np.min([c - [4.5, 3.375] for c in centers], axis=0)
            zh = np.max([c + [4.5, 3.375] for c in centers], axis=0)
        pd.DataFrame([
            dict(view='full', x_min=lo[0]-pad[0], x_max=hi[0]+pad[0], y_min=lo[1]-pad[1], y_max=hi[1]+pad[1]),
            dict(view='epithelial_zoom', x_min=zl[0], x_max=zh[0], y_min=zl[1], y_max=zh[1]),
        ]).to_csv(target / f'{panel}_{cancer}_view_windows.csv', index=False, float_format='%.17g')
        if cancer == 'CRC':
            path = root / 'singlecell/ALL_exact_pair_associations.csv'
            assoc = pd.read_csv(path, float_precision='round_trip')
            assoc = assoc[assoc.min_epi.eq(30) & assoc.CM_activity.eq('CM1_raw')]
            for row in assoc.itertuples():
                exported.append(dict(cancer=cancer, unit=row.unit, method='raw' if row.method == 'raw_Spearman' else 'study-adjusted',
                                     feature={'f_Epi_JUN': 'JUN_fraction', 'f_Epi_VIM': 'reference_fraction', 'balance': 'balance'}[row.feature],
                                     n=row.n, n_patients=row.n_patients, n_studies=row.n_studies, rho=row.effect, p=row.p,
                                     q=row.q_BH_primary_6_tests,
                                     q_scope='six primary sample tests' if row.unit == 'sample' else 'not supplied in primary six-test family',
                                     original_file=recorded_path(path), original_rho_column='effect',
                                     original_p_column='p', original_q_column='q_BH_primary_6_tests'))
            frame = pd.read_csv(root / 'singlecell/exact_pair_sample_fractions_and_CM.csv', float_precision='round_trip')
            frame = frame[frame.status.eq('tumor') & frame.n_epi.ge(30) & frame.n_non_epi.ge(50)]
            frame = frame.rename(columns={'f_Epi_JUN': 'JUN_fraction', 'f_Epi_VIM': 'reference_fraction',
                                          'CM1_raw': 'CM1_activity', 'donor_id_v1': 'patient_key'})
            source = root / 'CM/CM1_composition_marker_evidence.csv'
            comp = pd.read_csv(source, float_precision='round_trip').rename(columns={'subtype_id': 'subtype', 'H_share': 'H_loading'})
            comp['H_share'] = comp.H_loading
            comp['module'] = 'CRCcal_CM08'
        else:
            path = root / 'tables/balance_focused_singlecell_primary.csv'
            assoc = pd.read_csv(path, float_precision='round_trip')
            for row in assoc.itertuples():
                exported.append(dict(cancer=cancer, unit=row.unit, method='raw' if row.adjustment == 'none' else 'study-adjusted',
                                     feature='reference_fraction' if row.metric == 'VIM_fraction' else row.metric,
                                     n=row.n, n_patients=np.nan, n_studies=np.nan, rho=row.rho, p=row.p,
                                     q=row.q_BH_within_unit_threshold_context, q_scope='six tests per unit/threshold/context',
                                     original_file=recorded_path(path), original_rho_column='rho',
                                     original_p_column='p', original_q_column='q_BH_within_unit_threshold_context'))
            frame = pd.read_csv(root / 'tables/balance_primary_per_sample.csv', float_precision='round_trip')
            frame = frame[frame.primary_eligible].rename(columns={'study': 'study_id', 'donor': 'patient_key',
                         'VIM_fraction': 'reference_fraction', 'Balance': 'balance', 'CM1_W': 'CM1_activity'})
            source = root / 'figures_fig7style/source_data/PDAC_joint_atlas_subtype_to_CM1_loading.csv'
            comp = pd.read_csv(source, float_precision='round_trip').rename(columns={
                'non_epi_subtype': 'subtype', 'H_fraction': 'H_share', 'lineage_original': 'lineage'})
            comp['lineage'] = comp.lineage.replace({'TNK': 'T_NK'})
        frame['cancer'] = cancer
        cols = ['cancer', 'sample_id', 'study_id', 'patient_key', 'n_epi', 'n_non_epi',
                'JUN_fraction', 'reference_fraction', 'balance', 'CM1_activity']
        points.append(frame[cols])
        comp['cancer'], comp['original_file'] = cancer, recorded_path(source)
        compositions.append(comp[['cancer', 'module', 'subtype', 'lineage', 'H_loading', 'H_share', 'n_cells', 'original_file']])
        selected = next(row for row in exported if row['cancer'] == cancer and row['unit'] == 'sample'
                        and row['method'] == 'raw' and row['feature'] == 'JUN_fraction')
        rho, p = stats.spearmanr(frame.JUN_fraction, frame.CM1_activity)
        assert len(frame) == selected['n']
        np.testing.assert_allclose([rho, p], [selected['rho'], selected['p']], rtol=1e-12, atol=1e-16)
    comp = pd.concat(compositions, ignore_index=True)
    comp['displayed_top5_TNK'] = False
    for _, data in comp.groupby('cancer'):
        assert np.isclose(data.H_share.sum(), 1)
        comp.loc[data[data.lineage.eq('T_NK')].nlargest(5, 'H_share').index, 'displayed_top5_TNK'] = True
    updates = {
        'E_balance_vs_CM1_per_sample.csv': pd.concat(points, ignore_index=True),
        'E_F_single_cell_association_statistics.csv': pd.DataFrame(exported),
        'F_full_module_subtype_H_loadings.csv': comp,
    }
    for name, data in updates.items():
        path = target / name
        if path.exists():
            old = pd.read_csv(path, float_precision='round_trip')
            data = pd.concat([old[~old.cancer.isin(['CRC', 'PDAC'])], data], ignore_index=True)
        data.to_csv(path, index=False, float_format='%.17g')
    return {'stage': 'Fig7.source_tables', 'executed': True,
            'note': 'Tables only; run fig7-1 separately to draw the panels.'}
