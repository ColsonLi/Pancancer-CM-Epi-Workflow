from common import *

def main():
    records=[];centroids=[]
    names={'t_nk':'TNK','myeloid':'Myeloid','b':'B','stromal':'Stromal','endothelial':'Endothelial','mast':'Mast'}
    for lineage,own in names.items():
        p=next((C/f'h5ad/02-cell_subtype_integration_clustering/05-subtype-deg-annotation/{lineage}').glob('*.h5ad'))
        x,o,v=load_expr(p,raw=True);ms=means(x,v.index);ms.index=o.index
        detected={k:np.asarray((x[:,v.index.get_indexer([g for g in PANELS[k] if g in v.index])]>0).sum(axis=1)).ravel() for k in ['TNK','Myeloid','B','Stromal','Endothelial','Mast','Ductal','Acinar']}
        for cl,ix in o.groupby('cell_subtype',observed=True).indices.items():
            av=ms.iloc[ix].mean();other=av[['TNK','Myeloid','B','Stromal','Endothelial','Mast','Ductal','Acinar']].drop(own).idxmax()
            records.append({'cell_subtype':str(cl),'lineage':own,'n_cells':len(ix),'archived_removed':cl in FLAGGED,**av.to_dict(),'largest_other_lineage':other,'other_over_own':av[other]/max(av[own],.01),'own_3markers_fraction':np.mean(detected[own][ix]>=3),'acinar_3markers_fraction':np.mean(detected['Acinar'][ix]>=3),'ductal_3markers_fraction':np.mean(detected['Ductal'][ix]>=3),'acinar3_own_lt2_fraction':np.mean((detected['Acinar'][ix]>=3)&(detected[own][ix]<2))})
        print('AUDITED',lineage,len(o),flush=True)
        del x;gc.collect()
    out=pd.DataFrame(records);out.to_csv(B/'tables/original_non_epi_cleanup_expression_evidence.csv',index=False)
    print(out[out.archived_removed][['cell_subtype','n_cells','lineage','largest_other_lineage','other_over_own','own_3markers_fraction','acinar_3markers_fraction']].to_string(index=False),flush=True)
if __name__=='__main__':main()
