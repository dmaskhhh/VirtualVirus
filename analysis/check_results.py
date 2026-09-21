#!/usr/bin/env python3
"""Check selected research records; optionally compare a generated main example."""
from pathlib import Path
import argparse,csv,json,math,statistics

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'inputs/reference'
def rows(name,root=REF):
    with (root/name).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys(), 'JSON keys differ'
        for key in a:close(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,(int,float)):
        assert math.isclose(a,float(b),rel_tol=1e-10,abs_tol=1e-9),(a,b)
    elif isinstance(a,str):
        try:
            x,y=float(a),float(b)
        except (ValueError,TypeError):assert a==b,(a,b)
        else:assert math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-9),(a,b)
    else:assert a==b,(a,b)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated',action='store_true')
    args=parser.parse_args()
    p1=rows('p1_marker_timeseries.csv');l1=rows('l1_recorded_states.csv')
    assert len(p1)==371 and float(p1[-1]['time_s'])==1480
    assert len(l1)==1237 and float(l1[-1]['time_s'])==4944
    l1tot=rows('l1_gene_event_totals.csv')
    tx=sum(int(r['TX_events']) for r in l1tot);tl=sum(int(r['TL_events']) for r in l1tot)
    assert (tx,tl)==(363,858)
    assert (float(l1[0]['V_L1_Genome']),float(l1[-1]['V_L1_Genome']))==(3,19)
    for k in ['V_L1_packaged_genome','V_L1_virion','V_L1_breach','V_L1_released']:
        assert all(float(r[k])==0 for r in l1),k
    p1tot=rows('p1_gene_event_totals.csv');events=rows('p1_gene_events.csv')
    for r in p1tot:
        for event in ['TX','TL']:
            assert int(r[event+'_logged_events'])==sum(int(float(e['value'])) for e in events if e['source_key']==event+'_'+r['locus'])
    balance=rows('p1_protein_balance.csv')
    assert all(float(r['residual'])==0 for r in balance)
    objects=rows('particle_exit_readouts.csv');conds=rows('exit_condition_summary.csv')
    assert len(objects)==99 and len(conds)==9
    for c in conds:
        selected=[r for r in objects if r['spatial_condition']==c['condition']]
        assert len(selected)==11 and len({r['object_id'] for r in selected})==11
        assert all(r['censor_status']=='observed' for r in selected)
        assert sum(int(r['anchor_return_count']) for r in selected)==int(c['returns'])
        close(statistics.median(float(r['delta_s']) for r in selected),c['delta_median_s'])
    summary={'P1_checkpoints':len(p1),'P1_TX':sum(int(r['TX_logged_events']) for r in p1tot),
      'P1_TL':sum(int(r['TL_logged_events']) for r in p1tot),'L1_checkpoints':len(l1),
      'L1_TX':tx,'L1_TL':tl,'L1_genomes':[3,19],'spatial_conditions':len(conds),
      'object_condition_readouts':len(objects),'delta_median_s':statistics.median(float(r['delta_s']) for r in objects),
      'delta_max_s':max(float(r['delta_s']) for r in objects),
      'anchor_returns':sum(int(r['anchor_return_count']) for r in objects)}
    if args.generated:
        dest=ROOT/'outputs/r3_s0'
        close(json.loads((REF/'p1_main_spatial_summary.json').read_text()),json.loads((dest/'summary.json').read_text()))
        close(rows('p1_main_spatial_timeseries.csv'),rows('timeseries.csv',dest))
        close(json.loads((REF/'p1_main_birth_events.json').read_text()),json.loads((dest/'birth_events.json').read_text()))
        generated=rows('object_readouts.csv',ROOT/'outputs/readouts')
        generated=[r for r in generated if r['spatial_condition']=='r3_s0']
        expected=[r for r in objects if r['spatial_condition']=='r3_s0']
        close(sorted(expected,key=lambda r:int(r['object_id'])),sorted(generated,key=lambda r:int(r['object_id'])))
        summary['generated_main_matches_reference']=True
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
