#!/usr/bin/env python3
"""Extract frozen similarity/support tables; never fit or recompute similarities.

The only summaries calculated here are counts/quantiles of EXISTING bootstrap
and leave-study-out results, plus study counts from saved PDAC labels.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
DATA = OUT / 'source_data'
CRC = Path('/home/lr_xcy/lr_xcy/riku/lr_riku/validation_4_kirc_agents/crc_codex/calibrated_single_cluster_20261107')
PDAC = Path('/home/lr_xcy/lr_xcy/riku/lr_riku/validation_4_kirc_agents/pdac_codex/calibrated_single_cluster_20261107')
CRC_PAIR = json.loads((CRC / 'selected_pair.json').read_text())
PDAC_PAIR = json.loads((PDAC / 'tables/selected_single_cluster_pair.json').read_text())
CONFIG = {'CRC': CRC_PAIR['configuration'], 'PDAC': PDAC_PAIR['configuration']}
DEFINITION = {'CRC': CRC_PAIR['primary_reference'], 'PDAC': PDAC_PAIR['reference_definition']}
POLES = {
    'CRC': {state: CRC_PAIR['pair']['Epi_' + state].split('_')[-1] for state in ['JUN', 'VIM']},
    'PDAC': {state: PDAC_PAIR[state] for state in ['JUN', 'VIM']},
}
manifest = []


def read(path, **kw):
    return pd.read_csv(path, float_precision='round_trip', **kw)


def write(df, name, original='', selector='', note=''):
    path = DATA / name
    df.to_csv(path, index=False, float_format='%.17g')
    manifest.append(dict(delivered_file='source_data/' + name,
                         original_file=str(original), original_selector=selector,
                         n_rows=len(df), note=note))


def sort_clusters(df):
    return df.assign(_sort=df.cluster_label.str[1:].astype(int)).sort_values('_sort').drop(columns='_sort').reset_index(drop=True)


def filter_saved(path, cancer, definition_column, definition=None):
    df = read(path)
    mask = df.configuration.eq(CONFIG[cancer])
    if definition is not None:
        mask &= df[definition_column].eq(definition)
    return df.loc[mask].copy()


def state_summary(draws, leaveouts, reference_leaveouts, cluster, state, cancer):
    b = draws[draws.cluster_id.eq(cluster) & draws.state.eq(state)]
    l = leaveouts[leaveouts.cluster_id.eq(cluster) & leaveouts.state.eq(state)]
    n_target = leaveouts.omitted_study.nunique()
    q = b.eligible_rank.quantile([.025, .975])
    out = dict(bootstrap_n=len(b), bootstrap_valid_n=int(b.similarity.notna().sum()),
               bootstrap_win_count=int(b.winner.sum()),
               bootstrap_rank_CI_low=q.loc[.025], bootstrap_rank_CI_high=q.loc[.975],
               target_LOSO_total_omissions=n_target, target_LOSO_present_rows=len(l),
               target_LOSO_valid_similarity_n=int(l.similarity.notna().sum()),
               target_LOSO_valid_rank_n=int(l.eligible_rank.notna().sum()),
               target_LOSO_similarity_min=l.similarity.min(), target_LOSO_similarity_max=l.similarity.max(),
               target_LOSO_rank_min=l.eligible_rank.min(), target_LOSO_rank_max=l.eligible_rank.max(),
               target_LOSO_win_count=int(l.winner.sum()), target_LOSO_win_fraction=float(l.winner.sum()/n_target),
               target_LOSO_gate_reapplied=(cancer == 'PDAC'))
    # Missing rows (for example a one-study cluster after omitting its only
    # study) count as unavailable, not as stable rank observations.
    if 'eligible_after_leaveout' in l:
        out['target_LOSO_ineligible_or_missing_n'] = int(n_target-l.eligible_after_leaveout.sum())
    else:
        out['target_LOSO_ineligible_or_missing_n'] = int(n_target-l.eligible_rank.notna().sum())
    if reference_leaveouts is not None:
        r = reference_leaveouts[reference_leaveouts.cluster_id.eq(cluster) & reference_leaveouts.state.eq(state)]
        nr = reference_leaveouts.omitted_ccRCC_study.nunique()
        out.update(reference_LOSO_n=nr, reference_LOSO_rank_min=r.eligible_rank.min(),
                   reference_LOSO_rank_max=r.eligible_rank.max(), reference_LOSO_win_count=int(r.winner.sum()),
                   reference_LOSO_win_fraction=float(r.winner.sum()/nr))
    else:
        out.update(reference_LOSO_n=np.nan, reference_LOSO_rank_min=np.nan,
                   reference_LOSO_rank_max=np.nan, reference_LOSO_win_count=np.nan,
                   reference_LOSO_win_fraction=np.nan)
    return out


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    all_wide, all_long, marker_sources = [], [], []
    for cancer, root in [('CRC', CRC), ('PDAC', PDAC)]:
        config, definition = CONFIG[cancer], DEFINITION[cancer]
        selector = f'configuration={config}; primary definition={definition}'
        if cancer == 'CRC':
            table_path = root / 'similarity/ALL_CLUSTERS_ALL_CONFIGURATIONS_similarity_ranks.csv'
            raw = filter_saved(table_path, cancer, 'definition', definition)
            full_definitions = filter_saved(table_path, cancer, 'definition')
            support_path = root / f'epithelial/support_{config}.csv'
            support = read(support_path).rename(columns={'n_samples': 'n_tumor_samples', 'n_patients': 'n_donors', 'candidate_gate': 'eligible'})
            boot_path = root / 'similarity/all_configuration_bootstrap_draws.csv.gz'
            boot_raw = filter_saved(boot_path, cancer, 'definition', definition)
            boot = boot_raw.rename(columns={'reference_state': 'state', 'replicate': 'bootstrap', 'rank_candidates': 'eligible_rank', 'winner_candidate': 'winner'})
            boot['state'] = boot.state.str.replace('Epi_', '', regex=False)
            loso_path = root / 'similarity/all_configuration_leave_study_out.csv'
            loso_raw = filter_saved(loso_path, cancer, 'definition', definition)
            loso = loso_raw.rename(columns={'reference_state': 'state', 'rank_candidates': 'eligible_rank'})
            loso['state'] = loso.state.str.replace('Epi_', '', regex=False)
            loso['winner'] = loso.eligible_rank.eq(1)
            ref_loso = None
            markers_path = root / 'similarity/all_configuration_expression_top20_markers.csv'
            markers = filter_saved(markers_path, cancer, 'configuration')
            raw_name = 'CRC_primary_frozen_table.csv'
        else:
            table_path = root / 'tables/all_configurations_all_cluster_similarities.csv'
            raw = filter_saved(table_path, cancer, 'reference_definition', definition)
            full_definitions = filter_saved(table_path, cancer, 'reference_definition')
            support_path = root / f'labels/{config}.csv.gz'
            labels = read(support_path, usecols=['sample_id', 'study', 'donor', 'cluster_id'])
            study_counts = labels.groupby('cluster_id').study.nunique().rename('n_studies')
            support = raw[['cluster_id', 'n_cells', 'n_tumor_samples', 'n_donors', 'n_profile_donors', 'coverage_eligible']].rename(columns={'coverage_eligible': 'eligible'})
            support = support.merge(study_counts, on='cluster_id', how='left')
            boot_path = root / 'tables/similarity_reference_and_target_donor_bootstrap.csv.gz'
            boot_raw = filter_saved(boot_path, cancer, 'reference_definition', definition)
            boot = boot_raw.copy()
            loso_path = root / 'tables/similarity_leave_study_out_all_clusters.csv'
            loso_raw = filter_saved(loso_path, cancer, 'reference_definition', definition)
            loso = loso_raw.rename(columns={'coverage_eligible_after_leaveout': 'eligible_after_leaveout'})
            ref_loso_path = root / 'tables/similarity_leave_ccRCC_reference_study_out_all_clusters.csv'
            ref_loso = filter_saved(ref_loso_path, cancer, 'reference_definition', definition)
            write(ref_loso, 'PDAC_reference_leave_study_out.csv', ref_loso_path, selector,
                  'Existing ccRCC reference-study leaveouts; no new similarities computed.')
            markers_path = root / 'tables/primary_cluster_expression_markers.csv'
            markers = read(markers_path)
            raw_name = 'PDAC_primary_frozen_table.csv'

        support['cancer'] = cancer
        support['configuration'] = config
        support['cluster_label'] = support.cluster_id.str.split('_').str[-1]
        support['min_cells_required'] = 500
        support['min_tumor_samples_required'] = 15 if cancer == 'CRC' else 10
        support['exclusion_reason'] = support.apply(lambda r: '; '.join([
            *(['cells <500'] if r.n_cells < 500 else []),
            *([f'tumor samples <{r.min_tumor_samples_required}'] if r.n_tumor_samples < r.min_tumor_samples_required else [])]), axis=1)
        support['role'] = support.cluster_label.map({POLES[cancer]['JUN']: 'positive_JUN_pole', POLES[cancer]['VIM']: 'negative_VIM_reference_pole'}).fillna('other')
        support = sort_clusters(support)
        assert len(support) == (15 if cancer == 'CRC' else 12)
        assert int(support.eligible.sum()) == 9
        assert np.array_equal(support.eligible.to_numpy(), ((support.n_cells >= 500) & (support.n_tumor_samples >= support.min_tumor_samples_required)).to_numpy())
        top = markers[markers['rank'].le(5)].sort_values(['cluster_id', 'rank']).groupby('cluster_id').gene.agg(';'.join)
        support['top5_expression_markers'] = support.cluster_id.map(top)

        rows = []
        for s in support.to_dict('records'):
            for state in ['JUN', 'VIM']:
                if cancer == 'CRC':
                    saved = raw[raw.cluster_id.eq(s['cluster_id']) & raw.reference_state.eq('Epi_' + state)].iloc[0]
                    vals = dict(similarity=saved.similarity, rank_all=saved.rank_all,
                                eligible_rank=saved.rank_candidates,
                                bootstrap_CI_low=saved.bootstrap_similarity_low,
                                bootstrap_CI_high=saved.bootstrap_similarity_high,
                                bootstrap_win_fraction=saved.bootstrap_win_fraction)
                else:
                    saved = raw[raw.cluster_id.eq(s['cluster_id'])].iloc[0]
                    vals = dict(similarity=saved['Sim_' + state], rank_all=saved[f'rank_{state}_all'],
                                eligible_rank=saved[f'rank_{state}_eligible'],
                                bootstrap_CI_low=saved[f'Sim_{state}_CI_low'],
                                bootstrap_CI_high=saved[f'Sim_{state}_CI_high'],
                                bootstrap_win_fraction=saved[f'bootstrap_winner_fraction_{state}'])
                vals.update(state_summary(boot, loso, ref_loso, s['cluster_id'], state, cancer))
                vals.update(state=state, reference_state='Epi_' + state, reference_definition=definition,
                            selected_for_this_reference=(s['cluster_label'] == POLES[cancer][state]))
                # Compare saved summaries to existing draw-level results only.
                b = boot[boot.cluster_id.eq(s['cluster_id']) & boot.state.eq(state)]
                assert np.isclose(vals['bootstrap_win_fraction'], b.winner.mean())
                assert np.isclose(vals['bootstrap_CI_low'], b.similarity.quantile(.025))
                assert np.isclose(vals['bootstrap_CI_high'], b.similarity.quantile(.975))
                rows.append({**s, **vals})
        long = pd.DataFrame(rows)
        for state in ['JUN', 'VIM']:
            winning = long[long.state.eq(state) & long.eligible & long.eligible_rank.eq(1)]
            assert winning.cluster_label.tolist() == [POLES[cancer][state]]
        wide = support.copy()
        wide['reference_definition'] = definition
        for state in ['JUN', 'VIM']:
            sub = long[long.state.eq(state)].set_index('cluster_id')
            wide['Sim_' + state] = wide.cluster_id.map(sub.similarity)
            for col in ['rank_all', 'eligible_rank', 'bootstrap_CI_low', 'bootstrap_CI_high', 'bootstrap_win_fraction',
                        'bootstrap_win_count', 'bootstrap_n', 'bootstrap_rank_CI_low', 'bootstrap_rank_CI_high',
                        'target_LOSO_rank_min', 'target_LOSO_rank_max', 'target_LOSO_win_count',
                        'target_LOSO_total_omissions', 'target_LOSO_win_fraction', 'target_LOSO_valid_rank_n',
                        'target_LOSO_ineligible_or_missing_n', 'reference_LOSO_n', 'reference_LOSO_win_count',
                        'reference_LOSO_win_fraction', 'reference_LOSO_rank_min', 'reference_LOSO_rank_max']:
                wide[state + '_' + col] = wide.cluster_id.map(sub[col])
        wide['JUN_minus_VIM_descriptive_only'] = wide.Sim_JUN - wide.Sim_VIM
        # Cross-check the separate frozen CRC wide export without substituting
        # its rounding for the primary long-table values.
        if cancer == 'CRC':
            w = filter_saved(root / 'similarity/ALL_CLUSTERS_WIDE_SimJUN_SimVIM.csv', cancer, 'definition', definition).set_index('cluster_id')
            for state in ['JUN', 'VIM']:
                assert np.allclose(wide['Sim_' + state], wide.cluster_id.map(w['Sim_' + state]), rtol=0, atol=1e-14)

        write(raw, raw_name, table_path, selector, 'Verbatim frozen primary-score rows; original column names retained.')
        write(wide, f'{cancer}_similarity.csv', table_path, selector, 'One row per cluster; actual frozen cosine values, support, ranks and existing uncertainty.')
        write(long, f'{cancer}_similarity_long.csv', table_path, selector, 'One row per cluster/reference; additional summaries aggregate existing draws only.')
        write(support, f'{cancer}_support.csv', f'{table_path}; {support_path}' if cancer == 'PDAC' else support_path, f'primary partition {config}',
              'PDAC n_studies counted from saved labels; all other counts/eligibility copied from frozen tables.' if cancer == 'PDAC' else 'Frozen CRC support counts; n_patients renamed n_donors.')
        write(boot_raw, f'{cancer}_bootstrap_draws.csv.gz', boot_path, selector, 'Unmodified primary-definition bootstrap rows from the saved all-configuration table.')
        write(loso_raw, f'{cancer}_target_leave_study_out.csv', loso_path, selector, 'Unmodified saved target-study leaveouts; CRC holds original gate fixed, PDAC reapplies its support gate.')
        write(markers, f'{cancer}_expression_markers.csv', markers_path, f'configuration={config}', 'Saved target-versus-other-expression-cluster markers; not reference-program genes or new DEGs.')
        write(full_definitions, f'{cancer}_reference_definition_sensitivity.csv', table_path, f'configuration={config}; all saved definitions', 'Alternate definitions retained for transparency; they do not replace the frozen pair.')
        all_wide.append(wide)
        all_long.append(long)
        marker_sources.append(markers_path)

    poles = pd.concat(all_long, ignore_index=True)
    poles = poles[poles.selected_for_this_reference].reset_index(drop=True)
    original_primary_tables = f'{CRC / "similarity/ALL_CLUSTERS_ALL_CONFIGURATIONS_similarity_ranks.csv"}; {PDAC / "tables/all_configurations_all_cluster_similarities.csv"}'
    write(poles, 'selected_poles.csv', f'{original_primary_tables}; {CRC / "selected_pair.json"}; {PDAC / "tables/selected_single_cluster_pair.json"}',
          'one independently selected support-eligible maximum per state/cancer',
          'The sign of similarity is not a selection gate; no selection on JUN-minus-VIM, markers, CM associations or outcomes.')
    comparisons = []
    for cancer, d in zip(['CRC', 'PDAC'], all_long):
        for state in ['JUN', 'VIM']:
            s = d[d.state.eq(state)]
            a = s.loc[s.rank_all.eq(1)].iloc[0]
            e = s.loc[s.eligible_rank.eq(1)].iloc[0]
            comparisons.append(dict(cancer=cancer, state=state, all_cluster_maximum=a.cluster_label,
                                    all_cluster_maximum_similarity=a.similarity, all_cluster_maximum_eligible=a.eligible,
                                    selected_eligible_maximum=e.cluster_label, selected_similarity=e.similarity,
                                    selected_all_cluster_rank=e.rank_all, selected_eligible_rank=e.eligible_rank))
    write(pd.DataFrame(comparisons), 'independent_selection_summary.csv', original_primary_tables,
          'separate reference-specific rankings', 'All-cluster and eligible maxima deliberately distinguished.')
    pd.DataFrame(manifest).to_csv(DATA / 'SOURCE_MANIFEST.csv', index=False)
    summary = dict(CRC_clusters=15, CRC_eligible=9, CRC_cells=int(all_wide[0].n_cells.sum()),
                   PDAC_clusters=12, PDAC_eligible=9, PDAC_cells=int(all_wide[1].n_cells.sum()),
                   no_clustering_or_similarity_recomputed=True,
                   selected_poles=poles[['cancer', 'cluster_label', 'state', 'similarity', 'eligible_rank',
                                         'bootstrap_win_fraction', 'target_LOSO_win_count', 'target_LOSO_total_omissions']].to_dict('records'))
    (OUT / 'source_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
