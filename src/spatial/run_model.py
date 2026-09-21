"""Run the paper's conditional spatial model from included inputs."""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes
from membrane_model import shell, contain, pores, body_sites, damage_update, sphere

from walk import advance, seed_rng

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs'
INPUT = ROOT/'inputs/spatial'

def save_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def run(radius, seed):
    dest = OUT/f'r{radius}_s{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    raw = np.load(INPUT/'initial_geometry.npz')['sites']
    v0, m0 = raw != 0, raw == 6
    assert np.array_equal(shell(v0), m0)
    assert np.array_equal(binary_fill_holes(m0), v0)
    events = json.loads((INPUT/'physical_events.json').read_text())
    cfg = json.loads((INPUT/'archived_config.json').read_text())['phase7_config']
    centers = np.array([e['physical_center'] for e in events], np.int32)
    orientations = np.array([e['physical_orientation'] for e in events], np.int32)
    offsets = np.array([body_sites((0,0,0), o) for o in orientations], np.int32)
    n = len(events)
    assert offsets.shape == (n,47,3)
    active, released = np.zeros(n,bool), np.zeros(n,bool)
    releases, births = np.full(n, np.nan), np.full(n, np.nan)
    occ = np.full(raw.shape,-1,np.int32)
    volume, membrane = v0.copy(), m0.copy()
    contact = np.zeros_like(m0)
    damage = None
    rows, snapshots, revisions, rev_volumes, rev_membranes = [], [], [], [], []
    birth_events, delay_events = [], []
    previous_m, previous_v = np.zeros_like(m0), np.zeros_like(v0)
    version = -1
    seed_rng(seed)
    total_stats = np.zeros(6,np.int64)
    for t in range(1236, 1481):
        old_contact_count = int(contact.sum())
        for i,e in enumerate(events):
            if active[i] or t < e['time_s']:
                continue
            b = centers[i]+offsets[i]
            if (occ[tuple(b.T)] >= 0).any():
                delay_events.append(dict(time_s=t, molecule_id=i+1, reason='body_collision'))
                continue
            proposed_v = contain(volume, b)
            seedmask = np.zeros_like(m0); seedmask[tuple(b.T)] = True
            new_contact = contact | (shell(volume) & binary_dilation(seedmask, structure=sphere(1)))
            proposed_m = shell(proposed_v)
            if damage is not None and damage['breach_time'] is not None:
                proposed_m = pores(proposed_m, new_contact, radius)
            blocked = False
            for j in np.where(active)[0]:
                ix = tuple((centers[j]+offsets[j]).T)
                if proposed_m[ix].any() or (released[j] and proposed_v[ix].any()):
                    blocked = True
                    break
            if blocked:
                delay_events.append(dict(time_s=t, molecule_id=i+1, reason='membrane_update_conflict'))
                continue
            assert not proposed_m[tuple(b.T)].any()
            assert proposed_v[tuple(b.T)].all()
            birth_events.append(dict(molecule_id=i+1, scheduled_s=e['time_s'], actual_s=t,
                                     added_volume_voxels=int((proposed_v & ~volume).sum()),
                                     new_contact_voxels=int((new_contact & ~contact).sum())))
            volume, membrane, contact = proposed_v, proposed_m, new_contact
            occ[tuple(b.T)] = i
            active[i] = True; births[i] = t
        failures = sum(bool(e['forced_placement']) and e['time_s'] <= t for e in events)
        damage = damage_update(damage,t,int(contact.sum()),int(contact.sum())-old_contact_count,
                               int((active & ~released).sum()),cfg,failures)
        closed = shell(volume)
        membrane = pores(closed,contact,radius) if damage['breach_time'] is not None else closed
        if not np.array_equal(membrane,previous_m) or not np.array_equal(volume,previous_v):
            version += 1
            revisions.append(dict(version=version,time_s=t,
                membrane_added=np.argwhere(membrane & ~previous_m).tolist() if version else [],
                membrane_removed=np.argwhere(previous_m & ~membrane).tolist(),
                volume_added=np.argwhere(volume & ~previous_v).tolist() if version else []))
            rev_volumes.append(volume.copy());rev_membranes.append(membrane.copy())
            previous_m,previous_v=membrane.copy(),volume.copy()
        # Independent per-second assertions, before the next interval.
        audit_occ = np.full(raw.shape,-1,np.int32)
        fully_inside = 0
        for i in np.where(active)[0]:
            b = centers[i]+offsets[i]; ix=tuple(b.T)
            assert not membrane[ix].any(), (t,i,'membrane collision')
            assert (audit_occ[ix] == -1).all(), (t,i,'object collision')
            audit_occ[ix]=i
            if released[i]: assert not volume[ix].any()
            if volume[ix].all(): fully_inside += 1
        assert np.array_equal(occ,audit_occ)
        assert int(active.sum()) == int((active & ~released).sum())+int(released.sum())
        snapshots.append(dict(centers=centers.copy(),active=active.copy(),released=released.copy(),version=version))
        rows.append(dict(time_s=t,assembled=int(active.sum()),retained=int((active & ~released).sum()),
            fully_inside=fully_inside,released=int(released.sum()),pending=sum(e['time_s']<=t for e in events)-int(active.sum()),
            membrane_voxels=int(membrane.sum()),volume_voxels=int(volume.sum()),
            membrane_removed_from_initial=int((m0 & ~membrane).sum()),membrane_added_to_initial=int((membrane & ~m0).sum()),
            pore_voxels=int((closed & ~membrane).sum()),contact_voxels=int(contact.sum()),
            damage=damage['cumulative_membrane_damage'],integrity=damage['membrane_integrity'],
            breach=int(damage['breach_time'] is not None),membrane_version=version))
        if t < 1480 and damage['breach_time'] is not None:
            total_stats += advance(centers,offsets,active,released,releases,membrane,volume,occ,t,1.,1000.)
    np.savez_compressed(dest/'states.npz', times=np.arange(1236,1481),
        centers=np.array([s['centers'] for s in snapshots]), active=np.array([s['active'] for s in snapshots]),
        released=np.array([s['released'] for s in snapshots]), versions=np.array([s['version'] for s in snapshots]),
        volumes=np.array(rev_volumes),membranes=np.array(rev_membranes),offsets=offsets,orientations=orientations,
        release_times=releases,birth_times=births,contact=contact)
    save_csv(dest/'timeseries.csv',rows)
    (dest/'membrane_events.json').write_text(json.dumps(revisions,separators=(',',':')))
    (dest/'birth_events.json').write_text(json.dumps(birth_events,indent=2))
    (dest/'delayed_events.json').write_text(json.dumps(delay_events,indent=2))
    summary=dict(radius_voxels=radius,seed=seed,input_events=n,assembled=int(active.sum()),released=int(released.sum()),
        retained=int((active & ~released).sum()),pending=n-int(active.sum()),
        breach_time_s=damage['breach_time'],breach_reason=damage['breach_reason'],
        first_three_fully_inside_s=next((r['time_s'] for r in rows if r['fully_inside']>=3),None),
        max_fully_inside=max(r['fully_inside'] for r in rows),first_release_s=float(np.nanmin(releases)) if released.any() else None,
        delayed_attempts=len(delay_events),membrane_versions=version+1,
        added_volume_voxels=int((volume & ~v0).sum()),final_pore_voxels=rows[-1]['pore_voxels'],
        move_stats=dict(zip(['proposed','accepted','membrane_rejected','collision_rejected','bounds_rejected','reentry_rejected'],map(int,total_stats))),
        snapshot_audit_frames=len(rows),snapshot_violations=0)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--radius',type=int,choices=[2,3,4],default=3,
                   help='Contact-neighborhood radius in 10-nm lattice sites.')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--inputs',type=Path,default=INPUT)
    p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args()
    INPUT=args.inputs.resolve();OUT=args.output.resolve()
    run(args.radius,args.seed)
