from full_common import *
from expression_engine import integrate
from threadpoolctl import threadpool_limits
import argparse,importlib.util
PREFIX={'Epithelial':'Epi','T_NK':'TNK','Myeloid':'Mye','B':'B','Plasma':'Plas','Stromal':'Str','Pericyte':'Peri','Endothelial':'Endo','Mast':'Mast','Schwann':'Schw'}
def match_epithelial(epidir):
 spec=importlib.util.spec_from_file_location('epi_matching',OLD_EPI/'03_match_and_markers.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.ROOT=epidir;m.REF=OLD_EPI/'reference'
 m.load_full=lambda:(sparse.load_npz(epidir/'crc_retained_log1p.npz'),pd.read_csv(epidir/'crc_retained_metadata.csv.gz',index_col=0,low_memory=False),pd.read_csv(epidir/'crc_genes.csv').gene.to_numpy())
 m.main('primary_all')
 savejson(epidir/'reference_reuse_provenance.json',{'frozen_reference':str(OLD_EPI/'reference/reference_manifest.json'),'reuse':'fixed ccRCC-only reference genes/scaling/profiles; no old CRC cluster labels or optimized signatures','new_crc_scope':'epithelial lineage extracted from corrected full atlas','cell_scope_not_old_tumor_subset':True})
def main(scope):
 x,original,genes=load_original();allobs=load_metadata();assert original.index.equals(allobs.index)
 lineages=['Epithelial'] if scope=='epithelial' else sorted(set(allobs.broad_lineage_v1)-{'Epithelial'})
 for lin in lineages:
  keep=allobs.broad_lineage_v1.eq(lin).to_numpy();obs=allobs.loc[keep].copy();obs['atlas_cluster_v1']=obs['cluster_v1'];xx=x[keep]
  if lin=='Epithelial':out=ROOT/'epithelial/primary_all';cap=100000
  else:out=ROOT/'lineages'/lin;cap=50000
  if not (out/'completion.json').exists():integrate(xx,obs,genes,out,'CRCfa1'+PREFIX.get(lin,lin),cap=cap,sensitivities=lin=='Epithelial')
  if lin=='Epithelial':
   epidir=ROOT/'epithelial';sparse.save_npz(epidir/'crc_retained_log1p.npz',xx,compressed=False);obs.to_csv(epidir/'crc_retained_metadata.csv.gz',index_label='cell_id');pd.Series(genes,name='gene').to_csv(epidir/'crc_genes.csv',index=False)
   match_epithelial(epidir)
  else:
   support=pd.read_csv(out/'cluster_support_markers.csv',index_col=0)
   support['lineage']=lin;support['descriptive_subtype']=support.top5_markers.str.split(';').str[:3].str.join('/')
   support.to_csv(out/'new_subtype_annotation.csv',index_label='subtype_id')
  del xx;gc.collect()
 savejson(ROOT/f'{scope}_reclustering_completion.json',{'complete':True,'lineages':lineages})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scope',choices=['epithelial','non_epithelial'],required=True);a=p.parse_args()
 with threadpool_limits(limits=8):main(a.scope)
