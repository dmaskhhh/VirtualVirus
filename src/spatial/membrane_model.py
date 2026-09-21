"""Conditional lattice geometry model. No force, elasticity or WCM feedback."""
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

NEIGHBOR26 = np.ones((3, 3, 3), dtype=bool)

def shell(volume):
    return volume & ~binary_erosion(volume, structure=NEIGHBOR26, border_value=0)

def sphere(radius):
    q = np.indices((2*radius+1,)*3) - radius
    return np.sum(q*q, axis=0) <= radius*radius

def body_parts(center, orientation):
    c = np.asarray(center, dtype=np.int32)
    o = np.asarray(orientation, dtype=np.int32)
    if np.abs(o).sum() != 1:
        raise ValueError('An axial unit orientation is required')
    head = np.argwhere(sphere(2)) - 2 + c
    tail = np.arange(3, 11)[:, None]*o + c
    base = np.argwhere(sphere(1)) - 1 + c + 11*o
    return head, tail, base

def body_sites(center, orientation):
    return np.unique(np.concatenate(body_parts(center, orientation)), axis=0).astype(np.int32)

def contain(volume, body):
    b = np.zeros_like(volume)
    if np.any(body < 1) or np.any(body >= np.asarray(volume.shape)-1):
        raise ValueError('Containment needs a one-voxel margin')
    b[tuple(body.T)] = True
    return volume | binary_dilation(b, structure=NEIGHBOR26)

def pores(membrane, contact, radius):
    if not contact.any():
        return membrane.copy()
    return membrane & ~binary_dilation(contact, structure=sphere(radius))

def can_move(center, offsets, delta, membrane, occupied, ident, volume, released):
    b = center + offsets + delta
    if np.any(b < 0) or np.any(b >= np.asarray(membrane.shape)):
        return False
    ix = tuple(b.T)
    if membrane[ix].any() or ((occupied[ix] >= 0) & (occupied[ix] != ident)).any():
        return False
    return not (released and volume[ix].any())

def damage_update(previous, time_s, displacement_total, displacement_new, retained, config, failures):
    previous = previous or {}
    dt = max(1, time_s-previous.get('time_s', time_s-1))
    damage = previous.get('cumulative_membrane_damage', 0.)
    damage += displacement_new*config['membrane_damage_per_new_displacement_site']
    damage += retained*dt*config['membrane_damage_per_internal_virion_s']
    integrity = max(0., 1.-damage)
    breach = previous.get('breach_time')
    reason = previous.get('breach_reason')
    pending = previous.get('pending_breach_reason')
    if breach is None:
        candidate = None
        if displacement_total >= config['membrane_breach_displacement_threshold']:
            candidate = 'displacement_threshold'
        elif 0 < config['membrane_breach_capacity_failure_threshold'] <= failures:
            candidate = 'capacity_failure_threshold'
        elif config.get('membrane_damage_breach_threshold') is not None:
            if damage >= config['membrane_damage_breach_threshold']:
                candidate = 'cumulative_damage_threshold'
        elif integrity <= config['membrane_integrity_breach_threshold']:
            candidate = 'integrity_threshold'
        if pending is None:
            pending = candidate
        if pending is not None and retained >= config['membrane_breach_min_internal_virions']:
            breach, reason, pending = time_s, pending, None
    return dict(time_s=time_s, cumulative_membrane_damage=damage, membrane_integrity=integrity,
                breach_time=breach, breach_reason=reason, pending_breach_reason=pending)
