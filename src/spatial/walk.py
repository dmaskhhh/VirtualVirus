"""Exact continuous-time proposal process with hard excluded-volume rejection."""
import numpy as np
from numba import njit

@njit
def seed_rng(seed):
    np.random.seed(seed)

@njit
def advance(centers, offsets, active, released, release_times, membrane, volume,
            occupied, start, duration, rate):
    ids = np.where(active)[0]
    stats = np.zeros(6, dtype=np.int64) # proposed, accepted, membrane, collision, bounds, reentry
    if len(ids) == 0:
        return stats
    now = start
    end = start+duration
    while True:
        now += np.random.exponential(1./(6*len(ids)*rate))
        if now >= end:
            break
        i = ids[np.random.randint(len(ids))]
        direction = np.random.randint(6)
        axis, sign = direction//2, 1 if direction%2 else -1
        stats[0] += 1
        ok, entirely_external, reject = True, True, 0
        for k in range(offsets.shape[1]):
            x = centers[i,0]+offsets[i,k,0]+(sign if axis == 0 else 0)
            y = centers[i,1]+offsets[i,k,1]+(sign if axis == 1 else 0)
            z = centers[i,2]+offsets[i,k,2]+(sign if axis == 2 else 0)
            if x < 0 or y < 0 or z < 0 or x >= volume.shape[0] or y >= volume.shape[1] or z >= volume.shape[2]:
                ok, reject = False, 4
                break
            if membrane[x,y,z]:
                ok, reject = False, 2
                break
            other = occupied[x,y,z]
            if other >= 0 and other != i:
                ok, reject = False, 3
                break
            if volume[x,y,z]:
                entirely_external = False
                if released[i]:
                    ok, reject = False, 5
                    break
        if not ok:
            stats[reject] += 1
            continue
        for k in range(offsets.shape[1]):
            x,y,z = centers[i]+offsets[i,k]
            occupied[x,y,z] = -1
        centers[i,axis] += sign
        for k in range(offsets.shape[1]):
            x,y,z = centers[i]+offsets[i,k]
            occupied[x,y,z] = i
        stats[1] += 1
        if entirely_external and not released[i]:
            released[i] = True
            release_times[i] = now
    return stats
