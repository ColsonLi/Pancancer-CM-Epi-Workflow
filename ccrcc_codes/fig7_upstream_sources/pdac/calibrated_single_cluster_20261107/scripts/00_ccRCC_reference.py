from common import *

def main():
    x,o,v=load_expr(KIRC,raw=True)
    o['donor']=o.series.astype(str)+'::'+o['sample'].astype(str)
    # Paired donor pseudobulk contrasts, using original ccRCC labels ONLY.
    mask=o.cell_subtype.astype(str).isin(['Epi_JUN','Epi_VIM'])
    dp,support=donor_profiles(x[mask.to_numpy()],o.loc[mask],'cell_subtype',v.index,min_cells=10)
    a=dp.loc['Epi_JUN'];b=dp.loc['Epi_VIM'];paired=a.index.intersection(b.index)
    delta=a.loc[paired].to_numpy()-b.loc[paired].to_numpy()
    effect=delta.mean(axis=0);tt,pp=stats.ttest_1samp(delta,0,axis=0);qq=fdr(pp)
    allowed=~v.index.str.match(r'^(MT-|RPL|RPS|HB[AB])')&~v.index.isin(KIDNEY|DIGESTIVE)
    tab=pd.DataFrame({'gene':v.index,'paired_mean_JUN_minus_VIM':effect,'t':tt,'p':pp,'q':qq,'n_paired_donors':len(paired),'eligible_conserved_feature':allowed})
    tab.to_csv(B/'tables/ccRCC_paired_original_JUN_vs_VIM_DEGs.csv',index=False)
    defs={'archived_core12':{'JUN':PANELS['JUN'],'VIM':PANELS['VIM']}}
    arms={}
    for state,sign in [('JUN',1),('VIM',-1)]:
        sub=tab[allowed&(qq<.05)&(effect*sign>=.1)].copy();sub['rank_statistic']=sub.t*sign
        arms[state]=sub.sort_values('rank_statistic',ascending=False).head(50).gene.tolist()
    defs['paired_ccRCC_DEG50']=arms
    genes=sorted(set(sum([sum(z.values(),[]) for z in defs.values()],[])))
    fi=v.index.get_indexer(genes);assert (fi>=0).all()
    dp,support=donor_profiles(x[:,fi],o,'cell_subtype',genes,min_cells=5)
    cp=dp.groupby(level='cluster').mean();mu=dp.mean();sd=dp.std(ddof=1).clip(lower=.1)
    dp.to_csv(B/'tables/ccRCC_frozen_reference_donor_profiles.csv.gz');support.to_csv(B/'tables/ccRCC_frozen_reference_donor_support.csv',index=False)
    cp.to_csv(B/'tables/ccRCC_frozen_reference_centroids.csv')
    pd.DataFrame({'mean':mu,'sd':sd}).to_csv(B/'tables/ccRCC_frozen_gene_standardization.csv',index_label='gene')
    js(defs,'tables/ccRCC_reference_definitions.json')
    js({'expression_source':KIRC,'expression_matrix':'raw.X, normalized-log native original data','original_state_labels':'cell_subtype Epi_JUN and Epi_VIM; no PDAC labels or outcomes','core_source':OLD/'scripts/common.py PANELS JUN/VIM (archived 12+12, unmodified)','DEG_source':'new paired donor pseudobulks of original ccRCC Epi_JUN minus original Epi_VIM; >=10 cells per donor-state; one-sample paired-difference t-test; q<.05, absolute mean normalized-log difference >=.1; top50 by signed t per direction','n_original_cells':len(o),'n_original_JUN':int(o.cell_subtype.eq('Epi_JUN').sum()),'n_original_VIM':int(o.cell_subtype.eq('Epi_VIM').sum()),'n_paired_donors_for_DEG':len(paired),'paired_donors':paired.tolist(),'DEG_arm_sizes':{k:len(vv) for k,vv in arms.items()},'filters':'ribosomal RPL/RPS, mitochondrial MT-, hemoglobin, fixed kidney-background and digestive lists; no PDAC expression/effects used','kidney_background_excluded':sorted(KIDNEY),'primary_definition':'archived_core12','scaling':'fixed ccRCC donor-state-profile gene means / SD floored at .1; >=5 cells/donor-state; equal donors within state centroids; identical cosine and coordinate system for both states and every target cluster; no gene weights','reference_bootstrap':'resample original ccRCC donor profiles within each original label; coordinate system held fixed'},'tables/ccRCC_reference_contract.json')
    print('REFERENCE FROZEN',len(paired),{k:len(z) for k,z in arms.items()},len(genes),flush=True)
if __name__=='__main__':main()
