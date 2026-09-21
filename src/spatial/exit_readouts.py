"""Exact same-process replay; anchor/whole-body readouts, without new dynamics."""
from pathlib import Path
import argparse, csv, hashlib, json, os, sys, time
import numpy as np

R=Path(__file__).resolve().parents[2]
SOLVER=R/'src/spatial'
OUT=R/'outputs/readouts'
CONDITIONS=[(3,0)]

def summarize_events(times,k_out,anchor,birth,end,opening):
    t=np.asarray(times);k=np.asarray(k_out);a=np.asarray(anchor,dtype=bool)
    if np.any((k==47)&~a):raise ValueError('body exit must imply anchor outside')
    if len(t)>1 and np.any(np.diff(t)<0):raise ValueError('unordered event times')
    tp=float(t[np.flatnonzero(a)[0]]) if a.any() else None
    tb=float(t[np.flatnonzero(k==47)[0]]) if np.any(k==47) else None
    delta=tb-tp if tb is not None and tp is not None else None
    post=tb-max(birth,opening) if tb is not None else None
    returns=sum(bool(a[j-1] and not a[j]) for j in range(1,len(a)) if tb is None or t[j]<=tb)
    return dict(first_anchor_outside_time=tp,whole_body_exit_time=tb,delta_s=delta,
                post_open_s=post,delta_over_post_open=delta/post if post and delta is not None else None,
                anchor_return_count=int(returns),observation_end=end,
                censor_status='observed' if tb is not None else 'body_right_censored')

def traced_solver():
    sys.path.insert(0,str(SOLVER))
    import walk
    src=(SOLVER/'walk.py').read_text()
    assert src.count('    now = start')==1
    src=src.replace('def advance(', 'def advance_readout(')
    src=src.replace('    now = start','    trace = np.empty((250000,4), np.float64)\n    nt = 0\n    now = start')
    src=src.replace('        return stats','        return stats, np.empty((0,4), np.float64)',1)
    anchor='        if entirely_external and not released[i]:'
    instrumentation='''        if not released[i]:
            ko = 0
            for kk in range(offsets.shape[1]):
                xx,yy,zz = centers[i]+offsets[i,kk]
                if not volume[xx,yy,zz]: ko += 1
            xx,yy,zz = centers[i]
            if nt >= len(trace): raise ValueError("trace capacity")
            trace[nt,0] = now
            trace[nt,1] = i
            trace[nt,2] = ko
            trace[nt,3] = not volume[xx,yy,zz]
            nt += 1
'''
    assert src.count(anchor)==1
    src=src.replace(anchor,instrumentation+anchor)
    src=src.rsplit('    return stats',1)[0]+'    return stats, trace[:nt]\n'
    ns={};exec(compile(src,'<readout-only-instrumentation>','exec'),ns)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'instrumented_walk.py.txt').write_text(src)
    return walk.seed_rng,ns['advance_readout']

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    start=time.time();seed_rng,advance=traced_solver();objects=[];audits=[];long=[];inputs={}
    for radius in sorted(set(r for r,s in CONDITIONS)):
      for seed in [s for r,s in CONDITIONS if r==radius]:
        condition=f'r{radius}_s{seed}';folder=R/'outputs'/condition
        path=folder/'states.npz';inputs[str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
        d=np.load(path);opening=json.loads((folder/'summary.json').read_text())['breach_time_s']
        birth=json.loads((folder/'birth_events.json').read_text());n=len(d['birth_times'])
        assert all(np.any(np.all(o==0,axis=1)) for o in d['offsets'])
        seed_rng(seed);c=d['centers'][0].copy();released=d['released'][0].copy();rt=np.full(n,np.nan)
        stats=np.zeros(6,dtype=np.int64);events=[[] for _ in range(n)];checks=0
        def add(t,i,k,a):
            # Piecewise-constant state, exact changes plus birth. No interpolation.
            row=(float(t),int(k),bool(a))
            if not events[i] or row[1:]!=events[i][-1][1:]:events[i].append(row)
        for ix,t in enumerate(d['times']):
            assert np.array_equal(c,d['centers'][ix]),(condition,int(t),'centers')
            assert np.array_equal(released,d['released'][ix]),(condition,int(t),'released')
            checks+=1
            active=d['active'][ix].copy();ver=d['versions'][ix];v=d['volumes'][ver];m=d['membranes'][ver]
            occ=np.full(v.shape,-1,np.int32)
            for i in np.flatnonzero(active):
                b=c[i]+d['offsets'][i];pos=tuple(b.T)
                assert not m[pos].any() and (occ[pos]<0).all()
                occ[pos]=i;k=int((~v[pos]).sum());a=not bool(v[tuple(c[i])])
                if released[i]:assert k==47 and a
                else:add(t,i,k,a)
            if t>=opening and ix<len(d['times'])-1:
                stat,tr=advance(c,d['offsets'],active,released,rt,m,v,occ,float(t),1.,1000.)
                stats+=stat
                for tt,i,k,a in tr:add(tt,int(i),int(k),bool(a))
        assert np.array_equal(rt,d['release_times'],equal_nan=True),(condition,'exit times')
        old=json.loads((folder/'summary.json').read_text())['move_stats']
        assert stats.tolist()==list(old.values()),(condition,'proposal accounting')
        for i,ev in enumerate(events):
            info=next(b for b in birth if b['molecule_id']==i+1)
            t,k,a=zip(*ev)
            row=dict(upstream_run_id='r12ce_g1_2500s',spatial_condition=condition,object_id=i+1,
                     anchor_definition='head/assembly anchor c; occupied zero-offset site',
                     scheduled_time=info['scheduled_s'],accepted_birth_time=info['actual_s'],opening_time=opening,
                     event_resolution='exact CTMC event; checkpoint-identical replay')
            row.update(summarize_events(t,k,a,info['actual_s'],int(d['times'][-1]),opening))
            assert row['whole_body_exit_time']==d['release_times'][i]
            objects.append(row)
            for tt,kk,aa in ev:long.append(dict(spatial_condition=condition,object_id=i+1,time_s=tt,k_out=kk,anchor_outside=int(aa)))
        audit=dict(condition=condition,checkpoints_exact=checks,release_times_exact=True,proposal_counts_exact=True,
                   state_change_rows=sum(map(len,events)),seconds=round(time.time()-start,2))
        audits.append(audit);print(json.dumps(audit),flush=True)
    write_csv(OUT/'object_readouts.csv',objects);write_csv(OUT/'event_state_changes.csv',long)
    deltas=np.array([r['delta_s'] for r in objects]);mainrows=[r for r in objects if r['spatial_condition']=='r3_s0']
    summary=dict(conditions=len(audits),objects=len(objects),censored=sum(r['censor_status']!='observed' for r in objects),
      delta_s_min=float(deltas.min()),delta_s_median=float(np.median(deltas)),delta_s_max=float(deltas.max()),
      anchor_returns_total=sum(r['anchor_return_count'] for r in objects),
      objects_with_anchor_returns=sum(r['anchor_return_count']>0 for r in objects),
      main_delta_s=[r['delta_s'] for r in mainrows],
      main_delta_median_s=float(np.median([r['delta_s'] for r in mainrows])),
      main_delta_max_s=max(r['delta_s'] for r in mainrows),
      final_anchor_count_per_condition=11,final_body_count_per_condition=11,
      elapsed_s=time.time()-start,audits=audits,inputs=inputs,
      solver_sha256=hashlib.sha256((SOLVER/'walk.py').read_bytes()).hexdigest(),
      analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      estimand='readout difference on identical full-body dynamics; not a point-particle dynamics comparison')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--all',action='store_true',help='Replay all nine previously generated conditions.')
    args=p.parse_args()
    if args.all:CONDITIONS=[(r,s) for r in (2,3,4) for s in (0,1,2)]
    main()

