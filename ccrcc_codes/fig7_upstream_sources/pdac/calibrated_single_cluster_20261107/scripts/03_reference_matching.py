from common import *
from sklearn.metrics.pairwise import cosine_similarity

def cosine(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return (a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-15))@(b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-15)).T
def weighted_centroid(dp,cl,weights):
    z=dp.loc[cl];w=weights.reindex(z.index).fillna(0).to_numpy()
    return np.average(z.to_numpy(),axis=0,weights=w) if w.sum()>0 else np.full(z.shape[1],np.nan)

def main():
    ep=ad.read_h5ad(OLD/'objects/cleaned_tumor_ductal.h5ad');ep.obs=enrich(ep.obs)
    defs=json.loads((B/'tables/ccRCC_reference_definitions.json').read_text())
    scaling=pd.read_csv(B/'tables/ccRCC_frozen_gene_standardization.csv',index_col=0)
    ref=pd.read_csv(B/'tables/ccRCC_frozen_reference_centroids.csv',index_col=0)
    rdp=pd.read_csv(B/'tables/ccRCC_frozen_reference_donor_profiles.csv.gz',index_col=[0,1]);rdp.index.names=['cluster','donor']
    allgenes=scaling.index.intersection(ep.var_names,sort=False);xx=ep.X[:,ep.var_names.get_indexer(allgenes)]
    manifest=pd.read_csv(B/'tables/epithelial_configuration_manifest.csv')
    scores=[];bootrows=[];lsorows=[];ref_lsorows=[];selection={};rng=np.random.default_rng(20261107)
    cov=[]
    for definition,arms in defs.items():
        for st,gs in arms.items():
            for g in gs:cov.append({'definition':definition,'state':st,'gene':g,'present_in_PDAC':g in allgenes})
    pd.DataFrame(cov).to_csv(B/'tables/reference_gene_coverage.csv',index=False)
    for config in manifest.configuration:
        lab=pd.read_csv(B/f'labels/{config}.csv.gz',index_col=0).reindex(ep.obs_names)
        o=ep.obs.copy();o['cluster_id']=lab.cluster_id
        dp,support=donor_profiles(xx,o,'cluster_id',allgenes)
        dp.to_csv(B/f'tables/{config}_reference_feature_donor_profiles.csv.gz')
        support.to_csv(B/f'tables/{config}_profile_support.csv',index=False)
        cent=dp.groupby(level='cluster').mean();cls=cent.index
        counts=o.cluster_id.value_counts();ns=o.groupby('cluster_id',observed=True)['sample'].nunique();nd=o.groupby('cluster_id',observed=True).donor.nunique()
        eligible=(counts.reindex(cls)>=500)&(ns.reindex(cls)>=10)
        donors=pd.Index(o.donor.unique());rdonors=pd.Index(rdp.index.get_level_values('donor').unique())
        for definition,arms in defs.items():
            gs=sorted(set(arms['JUN']+arms['VIM'])&set(allgenes))
            mu=scaling.loc[gs,'mean'].to_numpy();sd=scaling.loc[gs,'sd'].to_numpy()
            R=(ref.loc[['Epi_JUN','Epi_VIM'],gs].to_numpy()-mu)/sd
            sim=cosine((cent[gs].to_numpy()-mu)/sd,R)
            out=pd.DataFrame({'configuration':config,'reference_definition':definition,'cluster_id':cls,'n_cells':counts.reindex(cls).values,'n_tumor_samples':ns.reindex(cls).values,'n_donors':nd.reindex(cls).values,'n_profile_donors':support.groupby('cluster').donor.nunique().reindex(cls).values,'coverage_eligible':eligible.to_numpy(),'Sim_JUN':sim[:,0],'Sim_VIM':sim[:,1]})
            out.index=cls
            for st in ['JUN','VIM']:
                out[f'rank_{st}_all']=out[f'Sim_{st}'].rank(ascending=False,method='min')
                out[f'rank_{st}_eligible']=out.loc[eligible,f'Sim_{st}'].rank(ascending=False,method='min')
            winners={st:out.loc[eligible,f'Sim_{st}'].idxmax() for st in ['JUN','VIM']}
            bsim=[];bn=500 if config==PRIMARY else 200
            # Shared donor blocks preserve covariance among cluster centroids;
            # both ccRCC-reference and PDAC-donor uncertainty are propagated.
            for b in range(bn):
                tw=pd.Series(rng.choice(donors,len(donors),replace=True)).value_counts()
                rw=pd.Series(rng.choice(rdonors,len(rdonors),replace=True)).value_counts()
                tc=np.array([weighted_centroid(dp[gs],c,tw) for c in cls]);rc=np.array([weighted_centroid(rdp[gs],s,rw) for s in ['Epi_JUN','Epi_VIM']])
                ss=cosine((tc-mu)/sd,(rc-mu)/sd);bsim.append(ss)
                for j,st in enumerate(['JUN','VIM']):
                    eligidx=np.flatnonzero(eligible.to_numpy());valid=eligidx[np.isfinite(ss[eligidx,j])]
                    winner=cls[valid[np.argmax(ss[valid,j])]] if len(valid) else None
                    ranks=pd.Series(ss[:,j],index=cls).loc[eligible].rank(ascending=False,method='min')
                    for i,c in enumerate(cls):bootrows.append({'configuration':config,'reference_definition':definition,'bootstrap':b,'cluster_id':c,'state':st,'similarity':ss[i,j],'eligible_rank':ranks.get(c,np.nan),'winner':c==winner})
            bs=np.asarray(bsim)
            for j,st in enumerate(['JUN','VIM']):
                out[f'Sim_{st}_CI_low']=np.nanquantile(bs[:,:,j],.025,axis=0);out[f'Sim_{st}_CI_high']=np.nanquantile(bs[:,:,j],.975,axis=0)
                rows=[z for z in bootrows if z['configuration']==config and z['reference_definition']==definition and z['state']==st]
                out[f'bootstrap_winner_fraction_{st}']=pd.DataFrame(rows).groupby('cluster_id').winner.mean().reindex(cls).values
            for omitted in o.study.unique():
                keep=o.study.ne(omitted).to_numpy();oo=o.loc[keep];sp,_=donor_profiles(xx[keep],oo,'cluster_id',allgenes);cc=sp.groupby(level='cluster').mean()
                nn=oo.cluster_id.value_counts();ssamp=oo.groupby('cluster_id',observed=True)['sample'].nunique();ee=nn.reindex(cc.index).ge(500)&ssamp.reindex(cc.index).ge(10)
                cs=cosine((cc[gs].to_numpy()-mu)/sd,R)
                for j,st in enumerate(['JUN','VIM']):
                    winner=cc.index[np.flatnonzero(ee.to_numpy())[np.argmax(cs[ee.to_numpy(),j])]] if ee.any() else None
                    rank=pd.Series(cs[:,j],index=cc.index).loc[ee].rank(ascending=False,method='min')
                    for i,c in enumerate(cc.index):lsorows.append({'configuration':config,'reference_definition':definition,'omitted_study':omitted,'cluster_id':c,'state':st,'similarity':cs[i,j],'coverage_eligible_after_leaveout':bool(ee.loc[c]),'eligible_rank':rank.get(c,np.nan),'winner':c==winner,'n_cells_after_leaveout':int(nn.get(c,0)),'n_tumor_samples_after_leaveout':int(ssamp.get(c,0))})
            # Also omit each ccRCC reference study, not only target studies.
            # Feature sets and all-gene coordinate scaling remain frozen.
            refstudy=rdp.index.get_level_values('donor').str.split('::').str[0]
            for omitted in refstudy.unique():
                cc=rdp.loc[refstudy!=omitted,gs].groupby(level='cluster').mean()
                rr=(cc.loc[['Epi_JUN','Epi_VIM']].to_numpy()-mu)/sd
                cs=cosine((cent[gs].to_numpy()-mu)/sd,rr)
                for j,st in enumerate(['JUN','VIM']):
                    ix=np.flatnonzero(eligible.to_numpy());winner=cls[ix[np.argmax(cs[ix,j])]]
                    rank=pd.Series(cs[:,j],index=cls).loc[eligible].rank(ascending=False,method='min')
                    for i,c in enumerate(cls):ref_lsorows.append({'configuration':config,'reference_definition':definition,'omitted_ccRCC_study':omitted,'cluster_id':c,'state':st,'similarity':cs[i,j],'eligible_rank':rank.get(c,np.nan),'winner':c==winner})
            scores.append(out.reset_index(drop=True))
            if config==PRIMARY:
                selection[definition]={'JUN':winners['JUN'],'VIM':winners['VIM'],'different_clusters':winners['JUN']!=winners['VIM'],'similarities':{st:out.loc[winners[st],['Sim_JUN','Sim_VIM','rank_JUN_eligible','rank_VIM_eligible',f'Sim_{st}_CI_low',f'Sim_{st}_CI_high',f'bootstrap_winner_fraction_{st}','n_cells','n_tumor_samples','n_donors']].to_dict() for st in ['JUN','VIM']}}
            print('SIMILARITY',config,definition,winners,flush=True)
    allscores=pd.concat(scores,ignore_index=True);allscores.to_csv(B/'tables/all_configurations_all_cluster_similarities.csv',index=False)
    bs=pd.DataFrame(bootrows);bs.to_csv(B/'tables/similarity_reference_and_target_donor_bootstrap.csv.gz',index=False)
    ls=pd.DataFrame(lsorows);ls.to_csv(B/'tables/similarity_leave_study_out_all_clusters.csv',index=False)
    pd.DataFrame(ref_lsorows).to_csv(B/'tables/similarity_leave_ccRCC_reference_study_out_all_clusters.csv',index=False)
    primary=allscores[(allscores.configuration==PRIMARY)&allscores.reference_definition.eq('archived_core12')]
    primary.to_csv(B/'tables/primary_all_cluster_Sim_JUN_Sim_VIM_ranks.csv',index=False)
    chosen=selection['archived_core12'].copy();chosen.update(configuration=PRIMARY,reference_definition='archived_core12',all_reference_definition_selections=selection,selection_rule='independent highest cosine to actual ccRCC Epi_JUN/Epi_VIM centroids among >=500-cell, >=10-tumor-sample clusters; fixed primary HVG3000 sample r0.4; never pooled/retrospectively renamed',valid_pair=chosen['different_clusters'])
    for st in ['JUN','VIM']:
        sub=ls[(ls.configuration==PRIMARY)&ls.reference_definition.eq('archived_core12')&ls.state.eq(st)]
        chosen[f'leave_study_out_winner_fraction_{st}']=float(sub[sub.winner].cluster_id.eq(chosen[st]).mean())
    # Descriptive expression identity for every primary cluster, independently
    # of reference rank, and exact-pair donor-paired DEG transfer signatures.
    lab=pd.read_csv(B/f'labels/{PRIMARY}.csv.gz',index_col=0).reindex(ep.obs_names);o=ep.obs.copy();o['cluster_id']=lab.cluster_id
    dp,sup=donor_profiles(ep.X,o,'cluster_id',ep.var_names);cp=dp.groupby(level='cluster').mean()
    cp.to_csv(B/'tables/primary_all_cluster_donor_balanced_gene_centroids.csv.gz')
    marks=[];keepgenes=~ep.var_names.str.match(r'^(MT-|RPL|RPS)')&~ep.var_names.isin(DIGESTIVE)
    for c in cp.index:
        diff=cp.loc[c]-cp.drop(c).mean()
        for rank,g in enumerate(diff[keepgenes].sort_values(ascending=False).head(30).index,1):marks.append({'cluster_id':c,'gene':g,'rank':rank,'donor_balanced_effect_vs_other_clusters':diff[g]})
    pd.DataFrame(marks).to_csv(B/'tables/primary_cluster_expression_markers.csv',index=False)
    if chosen['valid_pair']:
        # Recompute pair profiles using >=10 cells per donor-state.
        pp=o.cluster_id.isin([chosen['JUN'],chosen['VIM']]);dd,sup=donor_profiles(ep.X[pp.to_numpy()],o.loc[pp],'cluster_id',ep.var_names,min_cells=10)
        aa=dd.loc[chosen['JUN']];bb=dd.loc[chosen['VIM']];paired=aa.index.intersection(bb.index)
        delta=aa.loc[paired].to_numpy()-bb.loc[paired].to_numpy();ef=delta.mean(axis=0);tt,p=stats.ttest_1samp(delta,0,axis=0);q=fdr(p)
        deg=pd.DataFrame({'gene':ep.var_names,'mean_JUN_minus_VIM':ef,'t':tt,'p':p,'q':q,'n_paired_donors':len(paired),'eligible_transfer_gene':keepgenes})
        deg.to_csv(B/'tables/exact_selected_pair_donor_paired_DEGs.csv',index=False);dd.to_csv(B/'tables/exact_selected_pair_donor_expression_profiles.csv.gz');sup.to_csv(B/'tables/exact_selected_pair_donor_support.csv',index=False)
        signatures={}
        for st,sign in [('JUN',1),('VIM',-1)]:
            d=deg[keepgenes&(q<.05)&(ef*sign>=.1)].copy();d['signed_t']=d.t*sign
            signatures['single_'+st]=d.sort_values('signed_t',ascending=False).head(50).gene.tolist()
        js(signatures,'tables/exact_selected_pair_signatures.json')
        chosen['transfer_DEG_paired_donors']=len(paired);chosen['transfer_signature_sizes']={k:len(v) for k,v in signatures.items()}
        chosen['transfer_signature_rule']='top50 by signed paired-donor t, q<.05, signed normalized-log effect >=.1; exclude MT/RPL/RPS and digestive enzymes; no clinical/spatial outcomes used'
    chosen['marker_identity']={st:[z['gene'] for z in marks if z['cluster_id']==chosen[st]][:15] for st in ['JUN','VIM']}
    js(chosen,'tables/selected_single_cluster_pair.json')
    print('SELECTED SINGLE PAIR',chosen,flush=True)
if __name__=='__main__':main()
