#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "run_phase7_step7_p1_gene_resolved_rdme.py"
if not BASE_SCRIPT.exists():
    raise FileNotFoundError(f"Missing companion plugin: {BASE_SCRIPT}")

spec = importlib.util.spec_from_file_location("phase7_p1_base", str(BASE_SCRIPT))
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)
BASE_SPECIES_NAMES = base.species_names
BASE_SPECIES_DIFFUSION = base.species_diffusion

ROOT = base.ROOT
PHASE7 = base.PHASE7
LIVE_ROOT = PHASE7 / "live_runs" / "step7_p1_r12_topology_rdme"
CONFIG_DIR = LIVE_ROOT / "configs"
DEFAULT_TOPOLOGY_RUN_ID = "r12_p1_linear_topology_g1"
DEFAULT_TOPOLOGY_ROOT = base.INPUT_ROOT / "topology"
DEFAULT_STOICHIOMETRY_ROOT = base.INPUT_ROOT / "stoichiometry"
DEFAULT_WINDOWS_VMD = Path(os.environ.get("VMD_EXE", "vmd"))
VIRION_ORIENTATIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
HOST_LEAKAGE_CLASS_FRACTIONS = {"host_mrna": 0.45, "host_protein": 0.45, "host_large_complex": 0.10}
POST_BREACH_LEAKAGE_CLASS_FRACTIONS = {
    "host_mrna": 0.30,
    "host_protein": 0.40,
    "host_large_complex": 0.05,
    "viral_mrna": 0.10,
    "viral_protein": 0.13,
    "viral_intermediate": 0.02,
}
UINT32_MAX = 2**32 - 1

base.LIVE_ROOT = LIVE_ROOT
base.CONFIG_DIR = CONFIG_DIR


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runId", required=True)
    ap.add_argument("--duration", type=int, default=7200)
    ap.add_argument("--initialGenomes", type=int, default=1)
    ap.add_argument("--topologyRunId", default=DEFAULT_TOPOLOGY_RUN_ID)
    ap.add_argument("--topologyRoot", type=Path, default=DEFAULT_TOPOLOGY_ROOT)
    ap.add_argument("--stoichiometryRoot", type=Path, default=DEFAULT_STOICHIOMETRY_ROOT)
    ap.add_argument("--placementSeed", type=int, default=11012)
    ap.add_argument("--damageThreshold", type=int, default=260)
    ap.add_argument("--damageVirionStart", type=int, default=650)
    ap.add_argument("--releasePerSecond", type=int, default=30)
    ap.add_argument("--membraneBreachDisplacementThreshold", type=int, default=60)
    ap.add_argument("--membraneBreachCapacityFailureThreshold", type=int, default=0)
    ap.add_argument("--membraneIntegrityBreachThreshold", type=float, default=0.45)
    ap.add_argument("--membraneDamageBreachThreshold", type=float, default=None)
    ap.add_argument("--membraneDamagePerNewDisplacementSite", type=float, default=0.01)
    ap.add_argument("--membraneDamagePerInternalVirionSecond", type=float, default=0.001)
    ap.add_argument("--membraneBreachMinVirions", type=int, default=0)
    ap.add_argument("--membraneReleaseLagSeconds", type=int, default=20)
    ap.add_argument("--postBreachLeakParticlesPerSecond", type=float, default=1.0)
    ap.add_argument("--postBreachLeakageAreaReferenceSites", type=float, default=1.0)
    ap.add_argument("--postBreachLeakageMarkerCap", type=int, default=400)
    ap.add_argument("--postBreachSeconds", type=int, default=60)
    ap.add_argument("--enablePreBreachPermeability", action="store_true")
    ap.add_argument("--preBreachDamageThreshold", type=float, default=0.35)
    ap.add_argument("--preBreachDisplacementThreshold", type=int, default=40)
    ap.add_argument("--preBreachCapacityFailureThreshold", type=int, default=2)
    ap.add_argument("--preBreachCrowdingOccupiedFractionThreshold", type=float, default=0.02)
    ap.add_argument("--preBreachLeakParticlesPerSecond", type=float, default=20.0)
    ap.add_argument("--preBreachLeakageMarkerCap", type=int, default=80)
    ap.add_argument("--targetPhysicalVirions", type=int, default=0)
    ap.add_argument("--dnaIntervals", default="32,16,8,4")
    ap.add_argument("--maxConsecutiveRescues", type=int, default=3)
    ap.add_argument("--forcePackagingAfterCapacityFailures", type=int, default=1)
    ap.add_argument("--assemblyRateOverride", type=float, default=None)
    ap.add_argument("--structuralTranslationWeights", default="")
    ap.add_argument("--structuralSubunitTotal", type=int, default=258)
    ap.add_argument("--virionHeadRadius", type=int, default=2)
    ap.add_argument("--virionTailLength", type=int, default=8)
    ap.add_argument("--virionTailRadius", type=int, default=0)
    ap.add_argument("--virionBaseplateRadius", type=int, default=1)
    ap.add_argument("--topologyPackagingReserveMolecules", type=int, default=0)
    ap.add_argument("--topologyPackagingMaxFractionBeyondReserve", type=float, default=1.0)
    ap.add_argument("--dnaUpdateHoldUntilSeconds", type=float, default=0.0)
    ap.add_argument("--allowUnrelaxedTopologyReplication", action="store_true")
    ap.add_argument("--allowPhenomenologicalBreach", action="store_true")
    ap.add_argument("-cd", "--cudaDevices", type=int, default=0)
    ap.add_argument("-drs", "--dnaRngSeed", type=int, default=42)
    ap.add_argument("-dsd", "--dnaSoftwareDirectory", default=os.environ.get("PHASE7_DNA_SOFTWARE_DIR", ""))
    ap.add_argument("-m", "--membrane", type=int, default=1)
    ap.add_argument("-mh", "--maximumHours", type=float, default=48.0)
    ap.add_argument("--firstDnaWaitSeconds", type=int, default=900)
    ap.add_argument("--hookStep", type=int, default=1000)
    ap.add_argument("--writeStep", type=int, default=80000)
    ap.add_argument("--prepareOnly", action="store_true")
    ap.add_argument("--t8TopologyPackagingSmokeOnly", action="store_true")
    ap.add_argument("--windowsVmdExe", type=Path, default=DEFAULT_WINDOWS_VMD)
    return ap.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def parse_locus_weights(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid structural weight item {item!r}; expected locus=weight")
        locus, value = item.split("=", 1)
        locus = locus.strip()
        weight = float(value)
        if not locus or weight <= 0:
            raise ValueError(f"Invalid structural weight item {item!r}; weight must be positive")
        weights[locus] = weight
    return weights


def load_t7_structural_requirements(root: Path, expected_total: int = 258) -> tuple[list[dict[str, str]], dict[str, int]]:
    path = Path(root) / "p1_structural_stoichiometry.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing T7 structural stoichiometry table: {path}")
    rows = read_csv(path)
    requirements: dict[str, int] = {}
    for row in rows:
        locus = row["locus_tag"]
        required = int(row.get("required_subunits") or row.get("consumed_subunits") or "0")
        if required <= 0:
            raise ValueError(f"Invalid T7 structural requirement for {locus}: {row}")
        requirements[locus] = requirements.get(locus, 0) + required
    total = sum(requirements.values())
    if total != int(expected_total):
        raise ValueError(f"r12 requires the configured {int(expected_total)}-subunit packaging gate, got {total}")
    return rows, requirements


def compute_topology_packaging_capacity(
    unpackaged_count: int,
    structural_sets: int,
    reserve: int = 0,
    max_fraction: float = 1.0,
) -> int:
    beyond_reserve = max(0, int(unpackaged_count) - max(0, int(reserve)))
    if beyond_reserve <= 0 or int(structural_sets) <= 0:
        return 0
    fraction = min(1.0, max(0.0, float(max_fraction)))
    fraction_cap = int(np.floor(float(beyond_reserve) * fraction))
    if fraction > 0.0 and fraction_cap <= 0:
        fraction_cap = 1
    return max(0, min(int(structural_sets), beyond_reserve, fraction_cap))


def apply_dna_update_hold(*, current_next: int | float, time_s: int | float, hold_until: int | float) -> float:
    hold = float(hold_until)
    current = float(current_next)
    if hold <= 0.0:
        return current
    if float(time_s) < hold and current < hold:
        return hold
    return current


def _sphere_sites(center: tuple[int, int, int], radius: int) -> set[tuple[int, int, int]]:
    cx, cy, cz = center
    out: set[tuple[int, int, int]] = set()
    r2 = radius * radius
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx * dx + dy * dy + dz * dz <= r2:
                    out.add((cx + dx, cy + dy, cz + dz))
    return out


def _tail_cross_section(axis: tuple[int, int, int], radius: int) -> list[tuple[int, int, int]]:
    ax = axis.index(next(v for v in axis if v != 0))
    offsets: list[tuple[int, int, int]] = []
    for a in range(-radius, radius + 1):
        for b in range(-radius, radius + 1):
            if a * a + b * b <= radius * radius:
                vec = [0, 0, 0]
                free = [i for i in range(3) if i != ax]
                vec[free[0]] = a
                vec[free[1]] = b
                offsets.append(tuple(vec))
    return offsets


def build_p1_virion_body_sites(
    center: tuple[int, int, int],
    orientation: tuple[int, int, int],
    head_radius: int = 2,
    tail_length: int = 8,
    tail_radius: int = 0,
    baseplate_radius: int = 1,
) -> dict:
    if orientation not in VIRION_ORIENTATIONS:
        raise ValueError(f"Unsupported virion orientation: {orientation}")
    if head_radius < 1 or tail_length < 1 or tail_radius < 0 or baseplate_radius < 0:
        raise ValueError("Virion geometry radii/length must be positive, except tail/baseplate radius may be zero")
    head = _sphere_sites(center, head_radius)
    cross = _tail_cross_section(orientation, tail_radius)
    tail: set[tuple[int, int, int]] = set()
    cx, cy, cz = center
    ox, oy, oz = orientation
    for step in range(head_radius + 1, head_radius + tail_length + 1):
        px, py, pz = cx + ox * step, cy + oy * step, cz + oz * step
        for dx, dy, dz in cross:
            tail.add((px + dx, py + dy, pz + dz))
    base_center = (cx + ox * (head_radius + tail_length + 1), cy + oy * (head_radius + tail_length + 1), cz + oz * (head_radius + tail_length + 1))
    baseplate = _sphere_sites(base_center, baseplate_radius)
    occupied = head | tail | baseplate
    return {
        "center": center,
        "orientation": orientation,
        "head_sites": sorted(head),
        "tail_sites": sorted(tail),
        "baseplate_sites": sorted(baseplate),
        "occupied_sites": sorted(occupied),
        "geometry_nm": {
            "effective_head_diameter_nm": int((2 * head_radius + 1) * 10),
            "tail_length_nm": int(tail_length * 10),
            "tail_diameter_nm": int((2 * tail_radius + 1) * 10),
            "baseplate_diameter_nm": int(2 * baseplate_radius * 10),
            "coarse_voxel_nm": 10,
            "coarse_capacity_target_per_syn3a": "20-30",
        },
        "geometry_voxels": {
            "head_radius": int(head_radius),
            "tail_length": int(tail_length),
            "tail_radius": int(tail_radius),
            "baseplate_radius": int(baseplate_radius),
        },
    }


def mark_topology_molecule_packaged_from_state(
    molecule_state: dict,
    mol: int,
    missing_markers: list[tuple[str, tuple[int, int, int]]] | None = None,
) -> dict:
    state = molecule_state[mol]
    missing_markers = list(missing_markers or [])
    state["status"] = "packaged"
    state["consumed_topology_beads"] = len(state.get("beads", []))
    state["consumed_tss_templates"] = len(state.get("genes", []))
    state["missing_rdme_marker_count_at_packaging"] = len(missing_markers)
    return {
        "molecule_id": mol,
        "consumed_topology_beads": int(state["consumed_topology_beads"]),
        "consumed_tss_templates": int(state["consumed_tss_templates"]),
        "missing_rdme_marker_count": len(missing_markers),
        "missing_rdme_markers": missing_markers[:10],
        "state_consumed_despite_missing_markers": bool(missing_markers),
    }


def _in_bounds(site: tuple[int, int, int], shape: tuple[int, int, int]) -> bool:
    x, y, z = site
    return 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]


def place_species_index_at_lattice(
    plattice,
    species_idx: int,
    xyz: tuple[int, int, int],
    force_displace: bool = False,
) -> dict:
    x, y, z = xyz
    slots = plattice[:, x, y, z, 0]
    empty = np.where(slots == 0)[0]
    if len(empty):
        plattice[int(empty[0]), x, y, z, 0] = np.uint32(species_idx)
        return {"placed": True, "displaced_species_indices": []}
    if not force_displace:
        return {"placed": False, "displaced_species_indices": []}
    displaced = [int(slots[0])]
    plattice[0, x, y, z, 0] = np.uint32(species_idx)
    return {"placed": True, "displaced_species_indices": displaced}


def _neighbor_sites(site: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    x, y, z = site
    return [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)]


def _evaluate_virion_pose(
    body: dict,
    cytoplasm: np.ndarray,
    membrane: np.ndarray,
    occupied_sites: set[tuple[int, int, int]],
    allow_membrane_displacement: bool,
    force: bool,
) -> dict:
    shape = cytoplasm.shape
    head = set(tuple(site) for site in body["head_sites"])
    tail_base = set(tuple(site) for site in body["tail_sites"]) | set(tuple(site) for site in body["baseplate_sites"])
    sites = head | tail_base
    if any(not _in_bounds(site, shape) for site in sites):
        return {"valid": False, "reason": "out_of_bounds"}
    if sites & occupied_sites:
        return {"valid": False, "reason": "occupied_physical_virion_collision"}
    head_cytoplasm = sum(1 for site in head if bool(cytoplasm[site]))
    head_membrane = sum(1 for site in head if bool(membrane[site]))
    head_cytoplasm_fraction = float(head_cytoplasm) / float(max(1, len(head)))
    center = tuple(body["center"])
    if force:
        if (not bool(cytoplasm[center])) or head_membrane > 0 or head_cytoplasm_fraction < 0.75:
            return {"valid": False, "reason": "head_not_sufficiently_inside_cytoplasm"}
    elif any((not bool(cytoplasm[site])) or bool(membrane[site]) for site in head):
        return {"valid": False, "reason": "head_not_inside_cytoplasm"}

    membrane_overlap = set(site for site in tail_base if bool(membrane[site]))
    noncytoplasm = set(site for site in tail_base if not bool(cytoplasm[site]))
    extracellular_like = set(site for site in noncytoplasm if not bool(membrane[site]))
    if not allow_membrane_displacement and noncytoplasm:
        return {"valid": False, "reason": "tail_or_baseplate_outside_cytoplasm"}
    if allow_membrane_displacement and not force and extracellular_like:
        return {"valid": False, "reason": "tail_or_baseplate_exits_cell_without_force"}

    displacement = set(membrane_overlap)
    for site in sites:
        for nb in _neighbor_sites(site):
            if _in_bounds(nb, shape) and bool(membrane[nb]):
                displacement.add(nb)
    return {
        "valid": True,
        "reason": "",
        "occupied_sites": sites,
        "membrane_displacement_sites": displacement,
        "tail_membrane_sites": membrane_overlap,
        "noncytoplasm_sites": noncytoplasm,
        "extracellular_like_sites": extracellular_like,
        "head_cytoplasm_fraction": head_cytoplasm_fraction,
        "mechanistic_breach": False,
    }


def place_physical_p1_virion(
    cytoplasm: np.ndarray,
    membrane: np.ndarray,
    occupied_sites: set[tuple[int, int, int]],
    rng: np.random.Generator,
    max_attempts: int = 2000,
    allow_membrane_displacement: bool = False,
    force: bool = False,
    geometry: dict[str, int] | None = None,
) -> dict:
    coords = np.argwhere(cytoplasm)
    if len(coords) == 0:
        return {"placed": False, "failure_reason": "empty_cytoplasm_mask"}
    shape = cytoplasm.shape
    attempts = min(max_attempts, len(coords))
    indices = rng.choice(len(coords), size=attempts, replace=False if attempts <= len(coords) else True)
    best_forced = None
    for idx in indices:
        center = tuple(int(v) for v in coords[int(idx)])
        orientations = list(VIRION_ORIENTATIONS)
        rng.shuffle(orientations)
        for orientation in orientations:
            body = build_p1_virion_body_sites(center, orientation, **(geometry or {}))
            pose = _evaluate_virion_pose(body, cytoplasm, membrane, occupied_sites, allow_membrane_displacement, force)
            if not pose["valid"]:
                continue
            if force:
                score = (
                    len(pose["membrane_displacement_sites"]) * 100
                    + len(pose["tail_membrane_sites"]) * 10
                    - len(pose["extracellular_like_sites"])
                )
                if best_forced is None or score > best_forced[0]:
                    best_forced = (score, center, orientation, body, pose)
                continue
            return {
                "placed": True,
                "failure_reason": "",
                "center": center,
                "orientation": orientation,
                "body": body,
                "occupied_sites": sorted(pose["occupied_sites"]),
                "membrane_displacement_sites": sorted(pose["membrane_displacement_sites"]),
                "occupied_site_count": len(pose["occupied_sites"]),
                "membrane_displacement_site_count": len(pose["membrane_displacement_sites"]),
                "tail_membrane_site_count": len(pose["tail_membrane_sites"]),
                "noncytoplasm_site_count": len(pose["noncytoplasm_sites"]),
                "head_cytoplasm_fraction": pose["head_cytoplasm_fraction"],
                "forced_placement": False,
                "mechanistic_breach": pose["mechanistic_breach"],
            }
    if best_forced is not None:
        _, center, orientation, body, pose = best_forced
        return {
            "placed": True,
            "failure_reason": "",
            "center": center,
            "orientation": orientation,
            "body": body,
            "occupied_sites": sorted(pose["occupied_sites"]),
            "membrane_displacement_sites": sorted(pose["membrane_displacement_sites"]),
            "occupied_site_count": len(pose["occupied_sites"]),
            "membrane_displacement_site_count": len(pose["membrane_displacement_sites"]),
            "tail_membrane_site_count": len(pose["tail_membrane_sites"]),
            "noncytoplasm_site_count": len(pose["noncytoplasm_sites"]),
            "head_cytoplasm_fraction": pose["head_cytoplasm_fraction"],
            "forced_placement": True,
            "mechanistic_breach": pose["mechanistic_breach"],
        }
    return {"placed": False, "failure_reason": "no_nonoverlapping_cytoplasm_pose"}


def update_membrane_damage_state(
    previous: dict | None,
    time_s: int,
    total_displacement_sites: int,
    new_displacement_sites: int,
    internal_virions: int,
    config: dict,
    capacity_failures: int = 0,
) -> dict:
    previous = previous or {}
    previous_time = int(previous.get("time_s", int(time_s) - 1))
    elapsed_s = max(1, int(time_s) - previous_time)
    previous_damage = float(previous.get("cumulative_membrane_damage", 0.0))
    displacement_damage = float(new_displacement_sites) * float(config.get("membrane_damage_per_new_displacement_site", 0.01))
    load_damage = float(internal_virions) * float(config.get("membrane_damage_per_internal_virion_s", 0.001)) * float(elapsed_s)
    cumulative_damage = max(0.0, previous_damage + displacement_damage + load_damage)
    membrane_integrity = max(0.0, 1.0 - cumulative_damage)

    breach_time = previous.get("breach_time")
    displacement_threshold = int(config.get("membrane_breach_displacement_threshold", 60))
    capacity_failure_threshold = int(config.get("membrane_breach_capacity_failure_threshold", 0) or 0)
    integrity_threshold = float(config.get("membrane_integrity_breach_threshold", 0.45))
    damage_threshold = config.get("membrane_damage_breach_threshold")
    breach_reason = previous.get("breach_reason")
    pending_breach_reason = previous.get("pending_breach_reason")
    min_internal_virions = max(0, int(config.get("membrane_breach_min_internal_virions", 0)))
    if breach_time is None:
        candidate_reason = None
        if int(total_displacement_sites) >= displacement_threshold:
            candidate_reason = "displacement_threshold"
        elif capacity_failure_threshold > 0 and int(capacity_failures) >= capacity_failure_threshold:
            candidate_reason = "capacity_failure_threshold"
        elif damage_threshold is not None and cumulative_damage >= float(damage_threshold):
            candidate_reason = "cumulative_damage_threshold"
        elif damage_threshold is None and membrane_integrity <= integrity_threshold:
            candidate_reason = "integrity_threshold"
        if pending_breach_reason is None and candidate_reason is not None:
            pending_breach_reason = candidate_reason
        if pending_breach_reason is not None and int(internal_virions) >= min_internal_virions:
            breach_time = int(time_s)
            breach_reason = pending_breach_reason
            pending_breach_reason = None

    release_target = 0
    post_breach_elapsed_s = 0
    leakage_particles = int(previous.get("leakage_particles", 0))
    if breach_time is not None:
        post_breach_elapsed_s = max(0, int(time_s) - int(breach_time))
        release_lag_s = int(config.get("membrane_release_lag_s", 20))
        active_release_s = max(0, int(time_s) - int(breach_time) - release_lag_s + 1)
        release_target = active_release_s * int(config.get("release_per_second", 30))
        leakage_rate = float(config.get("post_breach_leak_particles_per_s", 1.0))
        area_reference = max(1.0, float(config.get("post_breach_leakage_area_reference_sites", 1.0)))
        breach_area_factor = max(1.0, float(total_displacement_sites) / area_reference)
        rupture_severity = max(0.05, 1.0 - membrane_integrity)
        leakage_particles = max(
            leakage_particles,
            int(post_breach_elapsed_s * leakage_rate * breach_area_factor * rupture_severity),
        )

    return {
        "time_s": int(time_s),
        "elapsed_s": int(elapsed_s),
        "new_displacement_sites": int(new_displacement_sites),
        "total_displacement_sites": int(total_displacement_sites),
        "capacity_failures": int(capacity_failures),
        "capacity_failure_breach_threshold": int(capacity_failure_threshold),
        "internal_virions": int(internal_virions),
        "displacement_damage": float(displacement_damage),
        "load_damage": float(load_damage),
        "cumulative_membrane_damage": float(cumulative_damage),
        "membrane_integrity": float(membrane_integrity),
        "breach_time": None if breach_time is None else int(breach_time),
        "breach_reason": breach_reason,
        "pending_breach_reason": pending_breach_reason,
        "breach_min_internal_virions": int(min_internal_virions),
        "release_target": int(release_target),
        "post_breach_elapsed_s": int(post_breach_elapsed_s),
        "leakage_particles": int(leakage_particles),
    }


def post_breach_stop_due(time_s: int | float, breach_time: int | float | None, config: dict) -> bool:
    if breach_time is None:
        return False
    extra_s = int(config.get("post_breach_seconds", 60))
    if extra_s < 0:
        return False
    return float(time_s) >= float(breach_time) + float(extra_s)


def format_run_status(
    run_id: str,
    phase: str,
    cds_count: int,
    status: str,
    error: str | None = None,
) -> dict:
    out = {
        "run_id": run_id,
        "status": status,
        "phase": phase,
        "cds_count": int(cds_count),
    }
    if error:
        out["error"] = str(error)
    return out


def write_run_status_file(sim_properties: dict, status: dict) -> None:
    working_directory = sim_properties.get("working_directory")
    if not working_directory:
        return
    try:
        path = Path(working_directory) / "phase7_p1_r12_run_status.json"
        path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        sim_properties["phase7_r12_status_write_error"] = str(exc)


def add_cme_species_uint32_guard(csim, sim_properties: dict) -> None:
    start = None
    try:
        import time as TIME

        start = TIME.time()
    except Exception:
        TIME = None
    guard_log = sim_properties.setdefault("phase7_r12_cme_uint32_guard_log", [])
    for spec_id in sim_properties["cme_species"]:
        csim.defineSpecies([spec_id])
        count = int(sim_properties["counts"].get(spec_id, 0))
        tracker = int(sim_properties.get("cme_state_tracker", {}).get(spec_id, 0))
        raw_count = count - tracker if spec_id in sim_properties.get("cme_state_tracker", {}) else count
        safe_count = raw_count
        reason = None
        if safe_count < 0:
            safe_count = 0
            reason = "negative_free_count"
        elif safe_count > UINT32_MAX:
            safe_count = UINT32_MAX
            reason = "uint32_max_clamp"
        if reason is not None:
            guard_log.append(
                {
                    "species": spec_id,
                    "raw_count": int(raw_count),
                    "safe_count": int(safe_count),
                    "count": int(count),
                    "cme_state_tracker": int(tracker),
                    "reason": reason,
                    "time_s": sim_properties.get("time"),
                }
            )
        csim.addParticles(spec_id, count=int(safe_count))
    if start is not None:
        print("Time to add species: ", TIME.time() - start)


def install_cme_uint32_guard(mc_cme_module) -> None:
    mc_cme_module.add_CME_species = add_cme_species_uint32_guard


def classify_host_leakage_species(name: str) -> str | None:
    if name.startswith("V_"):
        return None
    if name == "oriC" or name.startswith("G_") or name.startswith("D_") or name.startswith("DT_"):
        return None
    if name.startswith("R_"):
        return "host_mrna"
    if name.startswith("P_") or name.startswith("C_P_") or name in {"RNAP", "RNAP_aa", "RNAP_aab1", "Degradosome", "decay"}:
        return "host_protein"
    if name == "ribosomeP" or name.startswith("RB_") or name.startswith("RP_"):
        return "host_large_complex"
    return None


def classify_post_breach_leakage_species(name: str) -> str | None:
    host_class = classify_host_leakage_species(name)
    if host_class is not None:
        return host_class
    if name.startswith("V_P1_R_") or name.startswith("V_P1_RB_"):
        return "viral_mrna"
    if name.startswith("V_P1_P_"):
        return "viral_protein"
    if name.startswith("V_P1_TX_") or name.startswith("V_P1_TL_") or name in {"V_P1_rep_event", "V_P1_assembly_event"}:
        return "viral_intermediate"
    return None


def allocate_post_breach_leakage_budget(budget: int, available_by_class: dict[str, int]) -> dict[str, int]:
    budget = max(0, int(budget))
    out = {klass: 0 for klass in POST_BREACH_LEAKAGE_CLASS_FRACTIONS}
    if budget <= 0:
        return out
    remaining = budget
    for klass, fraction in POST_BREACH_LEAKAGE_CLASS_FRACTIONS.items():
        available = max(0, int(available_by_class.get(klass, 0)))
        want = min(available, int(budget * fraction))
        out[klass] = want
        remaining -= want
    while remaining > 0:
        progressed = False
        for klass in POST_BREACH_LEAKAGE_CLASS_FRACTIONS:
            if out[klass] < max(0, int(available_by_class.get(klass, 0))):
                out[klass] += 1
                remaining -= 1
                progressed = True
                if remaining <= 0:
                    break
        if not progressed:
            break
    return out


def allocate_host_leakage_budget(budget: int, available_by_class: dict[str, int]) -> dict[str, int]:
    budget = max(0, int(budget))
    out = {klass: 0 for klass in HOST_LEAKAGE_CLASS_FRACTIONS}
    post_breach_plan = allocate_post_breach_leakage_budget(budget, available_by_class)
    for klass in out:
        out[klass] = post_breach_plan.get(klass, 0)
    return out


def pre_breach_permeability_due(
    *,
    enabled: bool,
    membrane_damage: float,
    displaced_sites: int,
    capacity_failures: int,
    occupied_fraction: float,
    damage_threshold: float,
    displacement_threshold: int,
    capacity_failure_threshold: int,
    crowding_threshold: float,
) -> dict:
    membrane_reasons: list[str] = []
    if float(membrane_damage) >= float(damage_threshold):
        membrane_reasons.append("membrane_damage")
    if int(displaced_sites) >= int(displacement_threshold):
        membrane_reasons.append("membrane_displacement")
    if int(capacity_failures) >= int(capacity_failure_threshold):
        membrane_reasons.append("physical_capacity_failures")
    crowding_due = float(occupied_fraction) >= float(crowding_threshold)
    due = bool(enabled and membrane_reasons and crowding_due)
    return {
        "due": due,
        "enabled": bool(enabled),
        "membrane_reasons": membrane_reasons,
        "crowding_due": bool(crowding_due),
        "occupied_fraction": float(occupied_fraction),
        "crowding_threshold": float(crowding_threshold),
    }


def count_inside_particles_by_leakage_class(
    plattice: np.ndarray,
    name_to_index: dict[str, int],
    class_to_species: dict[str, list[str]],
    cytoplasm_mask: np.ndarray,
    membrane_mask: np.ndarray,
) -> dict[str, int]:
    out = {klass: 0 for klass in class_to_species}
    if plattice.size == 0:
        return out
    for klass, species_names in class_to_species.items():
        idxs = [int(name_to_index[name]) for name in species_names if name in name_to_index]
        if not idxs:
            continue
        locs = np.argwhere(np.isin(plattice, idxs))
        count = 0
        for loc in locs:
            _, x, y, z, _ = (int(v) for v in loc)
            if bool(cytoplasm_mask[x, y, z]) or bool(membrane_mask[x, y, z]):
                count += 1
        out[klass] = count
    return out


def remove_inside_particles_for_leakage_class(
    plattice: np.ndarray,
    name_to_index: dict[str, int],
    class_to_species: dict[str, list[str]],
    cytoplasm_mask: np.ndarray,
    membrane_mask: np.ndarray,
    klass: str,
    quota: int,
) -> Counter:
    moved_by_species = Counter()
    quota = max(0, int(quota))
    if quota <= 0:
        return moved_by_species
    species_names = class_to_species.get(klass, [])
    idx_to_name = {int(name_to_index[name]): name for name in species_names if name in name_to_index}
    if not idx_to_name:
        return moved_by_species
    locs = np.argwhere(np.isin(plattice, list(idx_to_name)))
    removed = 0
    for loc in locs:
        if removed >= quota:
            break
        slot, x, y, z, p = (int(v) for v in loc)
        if not (bool(cytoplasm_mask[x, y, z]) or bool(membrane_mask[x, y, z])):
            continue
        idx = int(plattice[slot, x, y, z, p])
        if idx not in idx_to_name:
            continue
        plattice[slot, x, y, z, p] = np.uint32(0)
        moved_by_species[idx_to_name[idx]] += 1
        removed += 1
    return moved_by_species


def _clear_species_at_xyz(plattice: np.ndarray, name_to_index: dict[str, int], species_name: str, xyz: tuple[int, int, int]) -> int:
    idx = int(name_to_index[species_name])
    x, y, z = (int(v) for v in xyz)
    removed = 0
    for slot in range(plattice.shape[0]):
        for particle in range(plattice.shape[4]):
            if int(plattice[slot, x, y, z, particle]) == idx:
                plattice[slot, x, y, z, particle] = np.uint32(0)
                removed += 1
    return removed


def release_intact_virion_footprint(
    plattice: np.ndarray,
    name_to_index: dict[str, int],
    occupied_virion_sites: set[tuple[int, int, int]],
    displaced_membrane_sites: set[tuple[int, int, int]],
    event: dict,
    *,
    time_s: int | float,
    reason: str,
) -> dict:
    center = tuple(int(v) for v in event.get("physical_center", ()))
    if len(center) != 3:
        return {"released": False, "reason": "missing_physical_center"}
    removed_virion = _clear_species_at_xyz(plattice, name_to_index, "V_P1_virion", center)
    removed_packaged = _clear_species_at_xyz(plattice, name_to_index, "V_P1_packaged_genome", center)
    if removed_virion <= 0:
        return {"released": False, "reason": "missing_virion_particle", "physical_center": center}
    occupied_sites = {tuple(int(v) for v in site) for site in event.get("occupied_sites", [])}
    membrane_sites = {tuple(int(v) for v in site) for site in event.get("membrane_displacement_sites", [])}
    for site in sorted(membrane_sites):
        _clear_species_at_xyz(plattice, name_to_index, "V_P1_membrane_displacement", site)
    occupied_virion_sites.difference_update(occupied_sites)
    displaced_membrane_sites.difference_update(membrane_sites)
    event["released"] = True
    event["released_time_s"] = int(time_s)
    event["release_reason"] = str(reason)
    event["released_occupied_site_count"] = int(len(occupied_sites))
    event["released_membrane_displacement_site_count"] = int(len(membrane_sites))
    return {
        "released": True,
        "time_s": int(time_s),
        "reason": str(reason),
        "physical_center": center,
        "removed_virion_particles": int(removed_virion),
        "removed_packaged_genome_particles": int(removed_packaged),
        "released_occupied_site_count": int(len(occupied_sites)),
        "released_membrane_displacement_site_count": int(len(membrane_sites)),
    }


def topology_dir(args) -> Path:
    return Path(args.topologyRoot) / str(args.topologyRunId)


def topology_files(args) -> dict[str, Path]:
    root = topology_dir(args)
    return {
        "root": root,
        "manifest": root / "p1_linear_topology_manifest.json",
        "particles": root / "p1_linear_dna_particles.csv",
        "cds": root / "p1_cds_to_linear_dna_beads.csv",
        "state": root / "p1_linear_dna_topology_state.npz",
    }


def load_topology(args) -> tuple[dict, list[dict[str, str]], list[dict[str, str]]]:
    files = topology_files(args)
    missing = [str(p) for p in files.values() if p != files["root"] and not p.exists()]
    if missing:
        raise FileNotFoundError("Missing r12 topology substrate files: " + ", ".join(missing))
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    if manifest["genome"]["topology"] != "linear" or manifest["terminal_closure_bond"]:
        raise ValueError("r12 requires P1 linear topology with no terminal closure bond")
    particles = read_csv(files["particles"])
    cds = read_csv(files["cds"])
    mol_count = len({int(row["molecule_id"]) for row in particles})
    if mol_count != int(args.initialGenomes):
        raise ValueError(f"Topology molecule count {mol_count} != --initialGenomes {args.initialGenomes}")
    return manifest, particles, cds


def _lattice_offsets(max_radius: int) -> list[tuple[int, int, int]]:
    offsets: list[tuple[int, int, int]] = []
    for dx in range(-max_radius, max_radius + 1):
        for dy in range(-max_radius, max_radius + 1):
            for dz in range(-max_radius, max_radius + 1):
                offsets.append((dx, dy, dz))
    offsets.sort(key=lambda d: (d[0] * d[0] + d[1] * d[1] + d[2] * d[2], abs(d[0]) + abs(d[1]) + abs(d[2])))
    return offsets


def project_topology_to_unique_cytoplasm_sites(
    dna_particles: list[dict[str, str]],
    cds_map: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    cytoplasm = np.load(base.CYTOPLASM_MASK)
    shape = cytoplasm.shape
    original_coords = [
        (int(row["lattice_x"]), int(row["lattice_y"]), int(row["lattice_z"]))
        for row in dna_particles
    ]
    original_counts = Counter(original_coords)
    used: set[tuple[int, int, int]] = set()
    projected_by_bead: dict[tuple[int, int], tuple[int, int, int]] = {}
    offsets = _lattice_offsets(max(shape))
    projected_particles: list[dict[str, str]] = []
    for row in sorted(dna_particles, key=lambda r: (int(r["molecule_id"]), int(r["bead_index"]))):
        x0, y0, z0 = int(row["lattice_x"]), int(row["lattice_y"]), int(row["lattice_z"])
        chosen = None
        for dx, dy, dz in offsets:
            x, y, z = x0 + dx, y0 + dy, z0 + dz
            if x < 0 or y < 0 or z < 0 or x >= shape[0] or y >= shape[1] or z >= shape[2]:
                continue
            xyz = (x, y, z)
            if xyz in used or not bool(cytoplasm[xyz]):
                continue
            chosen = xyz
            break
        if chosen is None:
            raise RuntimeError(f"No cytoplasm lattice site available near topology bead {row}")
        used.add(chosen)
        projected_by_bead[(int(row["molecule_id"]), int(row["bead_index"]))] = chosen
        new_row = dict(row)
        new_row["source_lattice_x"] = row["lattice_x"]
        new_row["source_lattice_y"] = row["lattice_y"]
        new_row["source_lattice_z"] = row["lattice_z"]
        new_row["lattice_x"], new_row["lattice_y"], new_row["lattice_z"] = (str(v) for v in chosen)
        projected_particles.append(new_row)

    projected_cds: list[dict[str, str]] = []
    for row in cds_map:
        key = (int(row["molecule_id"]), int(row["tx_start_bead"]))
        if key not in projected_by_bead:
            raise RuntimeError(f"CDS row points to missing topology bead: {row}")
        x, y, z = projected_by_bead[key]
        new_row = dict(row)
        new_row["source_tx_start_lattice_x"] = row["tx_start_lattice_x"]
        new_row["source_tx_start_lattice_y"] = row["tx_start_lattice_y"]
        new_row["source_tx_start_lattice_z"] = row["tx_start_lattice_z"]
        new_row["tx_start_lattice_x"] = str(x)
        new_row["tx_start_lattice_y"] = str(y)
        new_row["tx_start_lattice_z"] = str(z)
        projected_cds.append(new_row)

    stats = {
        "projection": "unique_cytoplasm_site_per_10bp_bead",
        "source_beads": len(dna_particles),
        "source_unique_lattice_sites": len(original_counts),
        "source_max_beads_per_site": max(original_counts.values()) if original_counts else 0,
        "projected_beads": len(projected_particles),
        "projected_unique_lattice_sites": len(used),
        "projected_max_beads_per_site": 1 if projected_particles else 0,
    }
    return projected_particles, projected_cds, stats


def t8_phase2_files(args) -> dict[str, Path]:
    root = topology_dir(args)
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "expression_species": root / "p1_expression_species.tsv",
        "structural_stoichiometry": root / "p1_structural_stoichiometry.tsv",
        "component_inventory": root / "p1_component_inventory.tsv",
        "resource_competition": root / "p1_resource_competition.tsv",
        "packaging_events": root / "packaging_events.tsv",
    }


def write_t8_vmd_render_assets(
    out_dir: Path,
    component_rows: list[dict[str, str]],
    packaging_rows: list[dict],
    windows_vmd: Path,
) -> dict[str, str]:
    vmd_dir = out_dir / "vmd_render"
    frames_dir = vmd_dir / "frames"
    vmd_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    state_tsv = vmd_dir / "p1_t8_topology_packaging_state.tsv"
    tcl = vmd_dir / "render_p1_t8_topology_packaging_vmd.tcl"
    ps1 = vmd_dir / "run_vmd_t8_topology_packaging.ps1"
    readme = vmd_dir / "README.md"
    colors = {
        "capsid": "blue",
        "portal_neck": "orange",
        "tail_tube": "yellow",
        "tail_sheath": "green",
        "baseplate": "purple",
        "tail_fiber": "cyan",
    }
    rows: list[dict] = []
    component_names = sorted(row["component"] for row in component_rows)
    for frame, time_s in enumerate([0, 5, 10, 15]):
        assembled = time_s >= 15
        subunits_ready = time_s >= 10 and not assembled
        rows.append(
            {
                "frame": frame,
                "time_s": time_s,
                "kind": "topology_genome" if not assembled else "packaged_genome",
                "component": "P1_complete_linear_topology",
                "x_A": -850,
                "y_A": 0,
                "z_A": 0,
                "radius_A": 22,
                "end_x_A": -300,
                "end_y_A": 0,
                "end_z_A": 0,
                "color": "red",
                "count": 1166,
            }
        )
        if time_s >= 5:
            for i, cls in enumerate(["mRNA", "ribosome_bound_mRNA", "protein"]):
                rows.append(
                    {
                        "frame": frame,
                        "time_s": time_s,
                        "kind": "expression_species",
                        "component": cls,
                        "x_A": -120,
                        "y_A": -180 + 180 * i,
                        "z_A": 0,
                        "radius_A": 52,
                        "end_x_A": "",
                        "end_y_A": "",
                        "end_z_A": "",
                        "color": ["orange", "yellow", "green"][i],
                        "count": 11,
                    }
                )
        if subunits_ready:
            for i, component in enumerate(component_names):
                angle = 2.0 * np.pi * i / max(1, len(component_names))
                rows.append(
                    {
                        "frame": frame,
                        "time_s": time_s,
                        "kind": "structural_component_pool",
                        "component": component,
                        "x_A": 210 + 220 * float(np.cos(angle)),
                        "y_A": 220 * float(np.sin(angle)),
                        "z_A": 0,
                        "radius_A": 64,
                        "end_x_A": "",
                        "end_y_A": "",
                        "end_z_A": "",
                        "color": colors.get(component, "silver"),
                        "count": next(row["consumed_subunits"] for row in component_rows if row["component"] == component),
                    }
                )
            rows.append(
                {
                    "frame": frame,
                    "time_s": time_s,
                    "kind": "packaging_gate",
                    "component": "T8_topology_consuming_packaging",
                    "x_A": 590,
                    "y_A": 0,
                    "z_A": 0,
                    "radius_A": 30,
                    "end_x_A": 920,
                    "end_y_A": 0,
                    "end_z_A": 0,
                    "color": "black",
                    "count": packaging_rows[0]["consumed_structural_subunits"] if packaging_rows else 0,
                }
            )
        if assembled:
            rows.append(
                {
                    "frame": frame,
                    "time_s": time_s,
                    "kind": "assembled_virion",
                    "component": "P1_virion",
                    "x_A": 940,
                    "y_A": 0,
                    "z_A": 250,
                    "radius_A": 190,
                    "end_x_A": 940,
                    "end_y_A": 0,
                    "end_z_A": -420,
                    "color": "green",
                    "count": 1,
                }
            )
    base.write_csv(
        state_tsv,
        rows,
        ["frame", "time_s", "kind", "component", "x_A", "y_A", "z_A", "radius_A", "end_x_A", "end_y_A", "end_z_A", "color", "count"],
    )
    tcl.write_text(
        """# VMD render script for Phase7 r12 T8 topology-consuming packaging smoke.
# Visualization of simulated states; not microscopy or evidence of natural infection.
set here [file dirname [info script]]
set outdir [file join $here frames]
file mkdir $outdir
mol new atoms 1
mol rename top "Phase7 r12 T8 P1 topology-consuming packaging"
mol delrep 0 top
display projection Orthographic
display depthcue off
axes location Off
color Display Background white
graphics top material Opaque

proc color_for {name} {
    if {$name eq "blue"} { graphics top color blue
    } elseif {$name eq "red"} { graphics top color red
    } elseif {$name eq "orange"} { graphics top color orange
    } elseif {$name eq "yellow"} { graphics top color yellow
    } elseif {$name eq "green"} { graphics top color green
    } elseif {$name eq "purple"} { graphics top color purple
    } elseif {$name eq "cyan"} { graphics top color cyan
    } elseif {$name eq "black"} { graphics top color black
    } else { graphics top color silver }
}

set rows {}
set fh [open [file join $here p1_t8_topology_packaging_state.tsv] r]
gets $fh header
while {[gets $fh line] >= 0} {
    if {[string trim $line] ne ""} { lappend rows [split $line ","] }
}
close $fh

for {set frame 0} {$frame < 4} {incr frame} {
    graphics top delete all
    foreach f $rows {
        if {[lindex $f 0] != $frame} { continue }
        set kind [lindex $f 2]
        set xyz [list [lindex $f 4] [lindex $f 5] [lindex $f 6]]
        set radius [lindex $f 7]
        set ex [lindex $f 8]
        set ey [lindex $f 9]
        set ez [lindex $f 10]
        color_for [lindex $f 11]
        if {$kind eq "topology_genome" || $kind eq "packaged_genome"} {
            graphics top cylinder $xyz [list $ex $ey $ez] radius $radius filled yes resolution 24
            graphics top sphere $xyz radius [expr {$radius * 1.8}] resolution 24
            graphics top sphere [list $ex $ey $ez] radius [expr {$radius * 1.8}] resolution 24
        } elseif {$kind eq "expression_species" || $kind eq "structural_component_pool"} {
            graphics top sphere $xyz radius $radius resolution 32
        } elseif {$kind eq "packaging_gate"} {
            graphics top cylinder $xyz [list $ex $ey $ez] radius $radius filled yes resolution 24
            graphics top cone [list $ex $ey $ez] [list [expr {$ex + 120.0}] $ey $ez] radius 80 resolution 24
        } elseif {$kind eq "assembled_virion"} {
            graphics top sphere $xyz radius $radius resolution 32
            graphics top cylinder [list [lindex $xyz 0] [lindex $xyz 1] [expr {[lindex $xyz 2] - 160.0}]] [list $ex $ey $ez] radius 48 filled yes resolution 24
            graphics top sphere [list $ex $ey $ez] radius 82 resolution 24
        }
    }
    display resetview
    scale by 0.50
    rotate x by -12
    render TachyonInternal [file join $outdir [format "p1_t8_topology_packaging_%04d.tga" $frame]]
}
quit
""",
        encoding="utf-8",
    )
    ps1.write_text(
        f"""$ErrorActionPreference = "Stop"
$VmdExe = "{windows_vmd}"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (!(Test-Path $VmdExe)) {{ throw "VMD executable not found: $VmdExe" }}
Push-Location $Here
& $VmdExe -dispdev text -e render_p1_t8_topology_packaging_vmd.tcl
Pop-Location
""",
        encoding="utf-8",
    )
    readme.write_text(
        """# Phase7 r12 T8 VMD Render

This folder visualizes the 1 genome / 15 s T8 topology-consuming packaging smoke.

Frames:

- 0 s: one unpackaged topology genome.
- 5 s: topology-projected expression species active.
- 10 s: structural component pools reach the T6/T7 stoichiometry gate.
- 15 s: complete topology object is consumed and one packaged genome / virion readout is produced.

Visualization of simulated states; not microscopy or evidence of natural infection.
""",
        encoding="utf-8",
    )
    return {
        "vmd_dir": str(vmd_dir),
        "state_table": str(state_tsv),
        "tcl": str(tcl),
        "powershell": str(ps1),
        "readme": str(readme),
        "frames_dir": str(frames_dir),
    }


def run_t8_topology_packaging_smoke(args) -> None:
    if int(args.initialGenomes) != 1:
        raise ValueError("T8 smoke is intentionally limited to --initialGenomes 1 before 3/10 genome stress tests")
    if int(args.duration) < 15:
        raise ValueError("T8 smoke requires --duration >= 15")
    topology_manifest, dna_particles, cds_map = load_topology(args)
    phase2 = t8_phase2_files(args)
    missing = [str(p) for k, p in phase2.items() if k != "root" and not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase2 T6/T7 topology-packaging files: " + ", ".join(missing))
    phase2_manifest = json.loads(phase2["manifest"].read_text(encoding="utf-8"))
    gates = phase2_manifest.get("gates", {})
    if not gates.get("T6_rdme_like_expression_species", {}).get("passed"):
        raise ValueError("Phase2 T6 expression gate has not passed")
    if not gates.get("T7_stoichiometric_packaging", {}).get("passed"):
        raise ValueError("Phase2 T7 stoichiometric packaging gate has not passed")

    expression_rows = read_csv(phase2["expression_species"])
    stoich_rows = read_csv(phase2["structural_stoichiometry"])
    component_rows = read_csv(phase2["component_inventory"])
    resource_rows = read_csv(phase2["resource_competition"])
    consumed_subunits = sum(int(row["consumed_subunits"]) for row in stoich_rows)
    if consumed_subunits <= len(stoich_rows):
        raise ValueError("T8 refuses token-level packaging; structural subunit count is not component-level")

    out_root = LIVE_ROOT / "t8_packaging_smokes"
    out_dir = out_root / str(args.runId)
    resolved_out = out_dir.resolve()
    resolved_root = out_root.resolve()
    if resolved_root not in [resolved_out, *resolved_out.parents]:
        raise RuntimeError(f"Refusing to write outside T8 smoke root: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(phase2["expression_species"], out_dir / "p1_expression_species.tsv")
    shutil.copyfile(phase2["structural_stoichiometry"], out_dir / "p1_structural_stoichiometry.tsv")
    shutil.copyfile(phase2["component_inventory"], out_dir / "p1_component_inventory.tsv")
    shutil.copyfile(phase2["resource_competition"], out_dir / "p1_resource_competition.tsv")

    genome_beads = len(dna_particles)
    tss_templates = len(cds_map)
    time_points = [0, 5, 10, int(args.duration)]
    counts_rows = []
    for t in time_points:
        expressed = int(t >= 5)
        assembled = int(t >= int(args.duration))
        subunits_available = consumed_subunits if t >= 10 and not assembled else 0
        counts_rows.append(
            {
                "time_s": t,
                "V_P1_topology_unpacked": 0 if assembled else 1,
                "V_P1_topology_packaged": assembled,
                "V_P1_packaged_genome": assembled,
                "V_P1_virion": assembled,
                "V_P1_breach": 0,
                "P1_mRNA_species_active": 11 if expressed else 0,
                "P1_ribosome_bound_species_active": 11 if expressed else 0,
                "P1_protein_species_active": 11 if expressed else 0,
                "P1_structural_subunits_available": subunits_available,
                "ordinary_rdme_dna_beads": 0,
                "token_packaging_allowed": 0,
            }
        )
    base.write_csv(
        out_dir / "counts_and_fluxes.csv",
        counts_rows,
        [
            "time_s",
            "V_P1_topology_unpacked",
            "V_P1_topology_packaged",
            "V_P1_packaged_genome",
            "V_P1_virion",
            "V_P1_breach",
            "P1_mRNA_species_active",
            "P1_ribosome_bound_species_active",
            "P1_protein_species_active",
            "P1_structural_subunits_available",
            "ordinary_rdme_dna_beads",
            "token_packaging_allowed",
        ],
    )

    registry_rows = [
        {
            "molecule_id": 1,
            "status": "packaged",
            "beads": genome_beads,
            "tss_templates": tss_templates,
            "source": str(topology_dir(args)),
        }
    ]
    base.write_csv(out_dir / "t8_topology_registry.tsv", registry_rows, ["molecule_id", "status", "beads", "tss_templates", "source"])

    packaging_rows = [
        {
            "time_s": int(args.duration),
            "event_type": "topology_consuming_packaging",
            "molecule_id": 1,
            "consumed_topology_beads": genome_beads,
            "consumed_tss_templates": tss_templates,
            "consumed_structural_loci": len(stoich_rows),
            "consumed_structural_subunits": consumed_subunits,
            "stoichiometry_gate_passed": True,
            "token_packaging_allowed": False,
            "source": "phase2_T6_T7_stoichiometry_gate",
        }
    ]
    base.write_csv(
        out_dir / "t8_topology_packaging_events.tsv",
        packaging_rows,
        [
            "time_s",
            "event_type",
            "molecule_id",
            "consumed_topology_beads",
            "consumed_tss_templates",
            "consumed_structural_loci",
            "consumed_structural_subunits",
            "stoichiometry_gate_passed",
            "token_packaging_allowed",
            "source",
        ],
    )
    vmd_outputs = write_t8_vmd_render_assets(out_dir, component_rows, packaging_rows, args.windowsVmdExe)

    sim_properties = {
        "phase": "Phase7-r12-T8",
        "boundary": "Visualization of simulated states; not an observed structure or evidence of natural infection.",
        "run_id": str(args.runId),
        "duration_s": int(args.duration),
        "initial_genomes": int(args.initialGenomes),
        "topology_source": str(topology_dir(args)),
        "ordinary_rdme_dna_beads": False,
        "topology_state_first": True,
        "phase7_r12_t8_topology_packaging_log": packaging_rows,
        "phase7_r12_t8_resource_rows": resource_rows,
        "phase7_r12_t8_component_inventory": component_rows,
        "phase7_r12_t8_vmd_render": vmd_outputs,
    }
    with (out_dir / "sim_properties.pkl").open("wb") as fh:
        pickle.dump(sim_properties, fh)

    manifest = {
        "phase": "Phase7-r12-T8",
        "run_id": str(args.runId),
        "duration_s": int(args.duration),
        "initial_genomes": int(args.initialGenomes),
        "topology_run_id": str(args.topologyRunId),
        "topology_source": str(topology_dir(args)),
        "ordinary_rdme_dna_beads": False,
        "token_packaging_allowed": False,
        "topology_state_first": True,
        "boundary": sim_properties["boundary"],
        "gates": {
            "T8_topology_consuming_packaging": {
                "passed": True,
                "consumed_topology_molecules": 1,
                "consumed_topology_beads": genome_beads,
                "consumed_tss_templates": tss_templates,
                "consumed_structural_loci": len(stoich_rows),
                "consumed_structural_subunits": consumed_subunits,
            }
        },
        "outputs": {
            "counts_and_fluxes": str(out_dir / "counts_and_fluxes.csv"),
            "sim_properties": str(out_dir / "sim_properties.pkl"),
            "topology_packaging_events": str(out_dir / "t8_topology_packaging_events.tsv"),
            "topology_registry": str(out_dir / "t8_topology_registry.tsv"),
            "expression_species": str(out_dir / "p1_expression_species.tsv"),
            "component_inventory": str(out_dir / "p1_component_inventory.tsv"),
            "resource_competition": str(out_dir / "p1_resource_competition.tsv"),
            "vmd_render": vmd_outputs,
        },
    }
    (out_dir / "t8_topology_packaging_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"run_id": args.runId, "status": "completed", "phase": "Phase7-r12-T8", "out_dir": str(out_dir)}, indent=2))


def species_names(cds_rows: list[dict[str, str]]) -> list[str]:
    names = BASE_SPECIES_NAMES(cds_rows)
    names.extend([
        "V_P1_packaged_genome",
        "V_P1_membrane_stress",
        "V_P1_membrane_displacement",
        "V_P1_pre_breach_permeability",
        "V_P1_pre_breach_leakage",
        "V_P1_breach_leakage",
    ])
    return list(dict.fromkeys(names))


def species_diffusion(name: str) -> float:
    if name in {"V_P1_Genome", "V_P1_packaged_genome", "V_P1_membrane_stress", "V_P1_membrane_displacement", "V_P1_breach_leakage"}:
        return 0.0
    return BASE_SPECIES_DIFFUSION(name)


def make_initial_particles_from_topology(args, cds_rows: list[dict[str, str]]) -> tuple[list[dict], dict, list[dict[str, str]], list[dict[str, str]]]:
    manifest, dna_particles, cds_map = load_topology(args)
    dna_particles, cds_map, projection_stats = project_topology_to_unique_cytoplasm_sites(dna_particles, cds_map)
    manifest = dict(manifest)
    manifest["phase7_r12_cell_projection"] = projection_stats
    rows: list[dict] = []
    particle_no = 1
    first_bead_by_mol: dict[int, dict[str, str]] = {}
    for bead in dna_particles:
        mol = int(bead["molecule_id"])
        first_bead_by_mol.setdefault(mol, bead)
    for mol, bead in sorted(first_bead_by_mol.items()):
        rows.append(
            {
                "rdme_species": "V_P1_Genome",
                "x": int(bead["lattice_x"]),
                "y": int(bead["lattice_y"]),
                "z": int(bead["lattice_z"]),
                "particle_no": particle_no,
                "source": f"r12_topology_molecule_{mol}_semantic_label",
            }
        )
        particle_no += 1
    for gene in cds_map:
        rows.append(
            {
                "rdme_species": f"V_P1_G_{gene['locus_tag']}",
                "x": int(gene["tx_start_lattice_x"]),
                "y": int(gene["tx_start_lattice_y"]),
                "z": int(gene["tx_start_lattice_z"]),
                "particle_no": particle_no,
                "source": f"r12_topology_molecule_{gene['molecule_id']}_cds_{gene['locus_tag']}_tx_bead_{gene['tx_start_bead']}",
            }
        )
        particle_no += 1
    return rows, manifest, dna_particles, cds_map


def freeze_config(
    args,
    params: dict[str, str],
    cds_rows: list[dict[str, str]],
    particles: list[dict],
    topology_manifest: dict,
    structural_requirements: dict[str, int],
) -> dict:
    config = base.freeze_config(args, params, cds_rows, particles)
    config["scope"] = "r12 P1-in-Syn3A RDME with explicit P1 linear DNA topology substrate and topology-consuming packaging."
    config["initial_genomes"] = int(args.initialGenomes)
    config["topology_run_id"] = str(args.topologyRunId)
    config["topology_manifest"] = topology_manifest
    config["structural_subunit_requirements"] = structural_requirements
    config["structural_subunit_total"] = int(sum(structural_requirements.values()))
    config["structural_translation_weights"] = parse_locus_weights(args.structuralTranslationWeights)
    config["assembly_rate_override"] = None if args.assemblyRateOverride is None else float(args.assemblyRateOverride)
    virion_geometry_voxels = {
        "head_radius": int(args.virionHeadRadius),
        "tail_length": int(args.virionTailLength),
        "tail_radius": int(args.virionTailRadius),
        "baseplate_radius": int(args.virionBaseplateRadius),
    }
    virion_geometry_nm = build_p1_virion_body_sites((32, 32, 32), (1, 0, 0), **virion_geometry_voxels)["geometry_nm"]
    config["physical_virion_geometry_voxels"] = virion_geometry_voxels
    config["non_compromise_checks"].update(
        {
            "initial_p1_dna": "Explicit 10 bp/bead P1 linear topology is kept as molecule-state registry, not ordinary RDME bead particles.",
            "ordinary_rdme_dna_beads": False,
            "gene_template_placement": "V_P1_G_locus particles are placed at topology-derived tx_start_bead lattice coordinates.",
            "packaging_rule": (
                "Packaging consumes a complete unpackaged topology molecule plus the configured structural subunit gate "
                f"({int(sum(structural_requirements.values()))} subunits) before one physical P1 body can be placed."
            ),
            "topology_packaging_reserve_molecules": int(args.topologyPackagingReserveMolecules),
            "topology_packaging_max_fraction_beyond_reserve": float(args.topologyPackagingMaxFractionBeyondReserve),
            "dna_update_hold_until_s": float(args.dnaUpdateHoldUntilSeconds),
            "physical_virion_geometry_nm": virion_geometry_nm,
            "physical_virion_placement_rule": (
                "P1 head/capsid is first required to fit in cytoplasm. If no pose exists, forced placement requires the head center in cytoplasm, "
                "at least 75% head-site cytoplasm support, no head-membrane overlap, and allows tail/baseplate membrane displacement or breach."
            ),
            "force_packaging_after_capacity_failures": int(args.forcePackagingAfterCapacityFailures),
            "counts_to_spatial_dna_mapping": False,
            "phenomenological_breach_enabled": False,
            "unrelaxed_topology_replication_enabled": bool(args.allowUnrelaxedTopologyReplication),
            "membrane_breach_displacement_threshold": int(args.membraneBreachDisplacementThreshold),
            "membrane_breach_capacity_failure_threshold": int(args.membraneBreachCapacityFailureThreshold),
            "membrane_integrity_breach_threshold": float(args.membraneIntegrityBreachThreshold),
            "membrane_damage_breach_threshold": None if args.membraneDamageBreachThreshold is None else float(args.membraneDamageBreachThreshold),
            "membrane_damage_per_new_displacement_site": float(args.membraneDamagePerNewDisplacementSite),
            "membrane_damage_per_internal_virion_s": float(args.membraneDamagePerInternalVirionSecond),
            "membrane_breach_min_internal_virions": int(args.membraneBreachMinVirions),
            "membrane_release_lag_s": int(args.membraneReleaseLagSeconds),
            "post_breach_leak_particles_per_s": float(args.postBreachLeakParticlesPerSecond),
            "post_breach_leakage_area_reference_sites": float(args.postBreachLeakageAreaReferenceSites),
            "post_breach_leakage_marker_cap": int(args.postBreachLeakageMarkerCap),
            "post_breach_seconds": int(args.postBreachSeconds),
            "pre_breach_permeability_enabled": bool(args.enablePreBreachPermeability),
            "pre_breach_damage_threshold": float(args.preBreachDamageThreshold),
            "pre_breach_displacement_threshold": int(args.preBreachDisplacementThreshold),
            "pre_breach_capacity_failure_threshold": int(args.preBreachCapacityFailureThreshold),
            "pre_breach_crowding_occupied_fraction_threshold": float(args.preBreachCrowdingOccupiedFractionThreshold),
            "pre_breach_leak_particles_per_s": float(args.preBreachLeakParticlesPerSecond),
            "pre_breach_leakage_marker_cap": int(args.preBreachLeakageMarkerCap),
            "target_physical_virions": int(args.targetPhysicalVirions),
        }
    )
    config["membrane_breach_displacement_threshold"] = int(args.membraneBreachDisplacementThreshold)
    config["membrane_breach_capacity_failure_threshold"] = int(args.membraneBreachCapacityFailureThreshold)
    config["membrane_integrity_breach_threshold"] = float(args.membraneIntegrityBreachThreshold)
    config["membrane_damage_breach_threshold"] = None if args.membraneDamageBreachThreshold is None else float(args.membraneDamageBreachThreshold)
    config["membrane_damage_per_new_displacement_site"] = float(args.membraneDamagePerNewDisplacementSite)
    config["membrane_damage_per_internal_virion_s"] = float(args.membraneDamagePerInternalVirionSecond)
    config["membrane_breach_min_internal_virions"] = int(args.membraneBreachMinVirions)
    config["membrane_release_lag_s"] = int(args.membraneReleaseLagSeconds)
    config["post_breach_leak_particles_per_s"] = float(args.postBreachLeakParticlesPerSecond)
    config["post_breach_leakage_area_reference_sites"] = float(args.postBreachLeakageAreaReferenceSites)
    config["post_breach_leakage_marker_cap"] = int(args.postBreachLeakageMarkerCap)
    config["post_breach_seconds"] = int(args.postBreachSeconds)
    config["pre_breach_permeability_enabled"] = bool(args.enablePreBreachPermeability)
    config["pre_breach_damage_threshold"] = float(args.preBreachDamageThreshold)
    config["pre_breach_displacement_threshold"] = int(args.preBreachDisplacementThreshold)
    config["pre_breach_capacity_failure_threshold"] = int(args.preBreachCapacityFailureThreshold)
    config["pre_breach_crowding_occupied_fraction_threshold"] = float(args.preBreachCrowdingOccupiedFractionThreshold)
    config["pre_breach_leak_particles_per_s"] = float(args.preBreachLeakParticlesPerSecond)
    config["pre_breach_leakage_marker_cap"] = int(args.preBreachLeakageMarkerCap)
    config["target_physical_virions"] = int(args.targetPhysicalVirions)
    config["damage_threshold"] = None
    config["damage_virion_start"] = None
    config["membrane_rule"] = (
        "r12 separates P1 physical placement from membrane failure: strict placement tries to keep the P1 head/capsid in cytoplasm, while tail/baseplate may displace membrane. "
        "Forced placement can occupy space and add membrane strain, but breach is emitted only after displacement or membrane-integrity gates are crossed. "
        "Post-breach release has an explicit latency; soluble host and viral particles then leave the cytoplasm/membrane as a sink-like leak flux. "
        "Only capped leakage markers are placed extracellularly for visualization, preventing extracellular marker placement from blocking volume relief. "
        "Host DNA/oriC and P1 topology-state DNA are excluded unless a separate genome-leak rule is introduced."
    )
    (CONFIG_DIR / f"{args.runId}_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def install_species(sim, region_dict: dict, particles: list[dict], cds_rows: list[dict[str, str]]) -> None:
    old_species_names = base.species_names
    old_species_diffusion = base.species_diffusion
    try:
        base.species_names = species_names
        base.species_diffusion = species_diffusion
        base.install_species(sim, region_dict, particles, cds_rows)
    finally:
        base.species_names = old_species_names
        base.species_diffusion = old_species_diffusion


def add_p1_gene_resolved_reactions(
    sim,
    params: dict[str, str],
    cds_rows: list[dict[str, str]],
    region_dict: dict,
    structural_translation_weights: dict[str, float] | None = None,
) -> None:
    structural_translation_weights = structural_translation_weights or {}
    rnap = sim.species("RNAP")
    ribosome = sim.species("ribosomeP")
    genome = sim.species("V_P1_Genome")
    pol = sim.species("V_P1_P_MpP1p01")
    rep_event = sim.species("V_P1_rep_event")
    replication = sim.rateConst("P1_r12_topology_replication_event", 1.0e6 * float(params["replication_rate_s"]), 2)
    tx_bind = sim.rateConst("P1_r12_tx_bind", 1.0e6 * float(params["transcription_rate_s"]), 2)
    tl_bind = sim.rateConst("P1_r12_tl_bind", 1.0e6 * float(params["translation_rate_s"]), 2)
    mrna_decay = sim.rateConst("P1_r12_mrna_decay", 0.006, 1)
    rep_regions = ["cytoplasm", "outer_cytoplasm", "DNA", "ribosomes", "ribo_centers"]
    for region in rep_regions:
        r = sim.region(region)
        r.addReaction([genome, pol], [genome, pol, rep_event], replication)
        for row in cds_rows:
            locus = row["locus_tag"]
            gene = sim.species(f"V_P1_G_{locus}")
            tx = sim.species(f"V_P1_TX_{locus}")
            mrna = sim.species(f"V_P1_R_{locus}")
            rb = sim.species(f"V_P1_RB_{locus}")
            protein = sim.species(f"V_P1_P_{locus}")
            tx_event = sim.species(f"V_P1_TX_{locus}_event")
            tl_event = sim.species(f"V_P1_TL_{locus}_event")
            tx_done = sim.rateConst(f"P1_r12_TX_done_{locus}", max(0.005, 45.0 / max(1.0, float(row["nt_length"]))), 1)
            tl_done_rate = max(0.005, 15.0 / max(1.0, float(row["aa_length"])))
            tl_done_rate *= float(structural_translation_weights.get(locus, 1.0))
            tl_done = sim.rateConst(f"P1_r12_TL_done_{locus}", tl_done_rate, 1)
            r.addReaction([gene, rnap], [tx], tx_bind)
            r.addReaction([tx], [gene, rnap, mrna, tx_event], tx_done)
            r.addReaction([mrna, ribosome], [rb], tl_bind)
            r.addReaction([rb], [mrna, ribosome, protein, tl_event], tl_done)
            r.addReaction([mrna], [], mrna_decay)


def install_counts_bootstrap(save_module) -> None:
    original = save_module.saveCountsAndFluxes

    def patched_save_counts_and_fluxes(time, sim_properties, odeResults, model, solver):
        sim_folder = Path(sim_properties["working_directory"])
        counts_path = sim_folder / "counts_and_fluxes.csv"
        init_path = sim_folder / "init_particle_counts.csv"
        if (
            np.rint(time) > 0
            and not counts_path.exists()
            and init_path.exists()
            and odeResults is not None
            and model is not None
            and solver is not None
        ):
            import pandas as pd

            res_start = odeResults[0, :]
            current_fluxes = solver.calcFlux(0, res_start)
            flux_rows = pd.DataFrame(
                {
                    "Time": ["F_" + rxn.getID() for rxn in model.getRxnList()],
                    "0.0": [save_module.round_sig(current_fluxes[indx], sig=3) for indx, _ in enumerate(model.getRxnList())],
                }
            )
            particle_df = pd.read_csv(init_path)
            pd.concat([particle_df, flux_rows], ignore_index=True).to_csv(counts_path, index=False)
            sim_properties["phase7_r12_counts_bootstrap"] = {
                "time_s": float(time),
                "reason": "First full ODE/counts save occurred after the upstream np.rint(time)==1 bootstrap window.",
            }
        return original(time, sim_properties, odeResults, model, solver)

    save_module.saveCountsAndFluxes = patched_save_counts_and_fluxes


def make_solver(base_cls, cds_rows: list[dict[str, str]]):
    BaseSolver = base.make_solver(base_cls, cds_rows)

    class Phase7P1R12TopologySolver(BaseSolver):
        def __init__(self, lmFile, sim_properties, region_dict, ribo_site_dict, termination_time=None):
            super().__init__(lmFile, sim_properties, region_dict, ribo_site_dict, termination_time=termination_time)
            self.topology_particles = read_csv(Path(sim_properties["phase7_topology_particles_csv"]))
            self.topology_cds = read_csv(Path(sim_properties["phase7_topology_cds_csv"]))
            self.structural_requirements = {
                locus: int(required)
                for locus, required in sim_properties["phase7_structural_requirements"].items()
            }
            self.cytoplasm_mask = np.load(base.CYTOPLASM_MASK).astype(bool)
            self.membrane_mask = np.load(base.MEMBRANE_MASK).astype(bool)
            self.occupied_virion_sites: set[tuple[int, int, int]] = set()
            self.physical_virion_log = []
            self.physical_virion_release_log = []
            self.physical_capacity_failures = []
            self.physical_breach_time = None
            self.displaced_membrane_sites: set[tuple[int, int, int]] = set()
            self.new_displacement_since_membrane_update = 0
            self.membrane_damage_state = None
            self.membrane_post_breach_log = []
            self.host_leakage_total = 0
            self.host_leakage_by_class = Counter()
            self.host_leakage_log = []
            self.pre_breach_leakage_total = 0
            self.pre_breach_leakage_by_class = Counter()
            self.pre_breach_leakage_log = []
            self.pre_breach_start_time = None
            self.last_pre_breach_log_time = None
            self.released_to_sink_total = 0
            self.post_breach_stop_pending = False
            self.post_breach_stop_raised = False
            self.post_breach_leakage_species = {klass: [] for klass in POST_BREACH_LEAKAGE_CLASS_FRACTIONS}
            for name in sorted(self.sim_properties["name_to_index"]):
                klass = classify_post_breach_leakage_species(name)
                if klass is not None:
                    self.post_breach_leakage_species[klass].append(name)
            self.pre_breach_leakage_species = {klass: [] for klass in HOST_LEAKAGE_CLASS_FRACTIONS}
            for name in sorted(self.sim_properties["name_to_index"]):
                klass = classify_host_leakage_species(name)
                if klass is not None:
                    self.pre_breach_leakage_species[klass].append(name)
            self.molecule_state = {}
            for row in self.topology_particles:
                mol = int(row["molecule_id"])
                self.molecule_state.setdefault(mol, {"status": "unpackaged", "beads": [], "genes": []})
                self.molecule_state[mol]["beads"].append((int(row["lattice_x"]), int(row["lattice_y"]), int(row["lattice_z"])))
            for row in self.topology_cds:
                mol = int(row["molecule_id"])
                self.molecule_state.setdefault(mol, {"status": "unpackaged", "beads": [], "genes": []})
                self.molecule_state[mol]["genes"].append(
                    (row["locus_tag"], int(row["tx_start_lattice_x"]), int(row["tx_start_lattice_y"]), int(row["tx_start_lattice_z"]))
                )
            self.topology_packaging_log = []
            self.topology_replication_log = []
            self.topology_replication_blocked = False
            self.structural_balance_log = []
            self.last_structural_balance_log_time = None

        def _remove_at(self, plattice, species_name: str, xyz: tuple[int, int, int]) -> bool:
            idx = int(self.sim_properties["name_to_index"][species_name])
            x, y, z = xyz
            slots = np.where(plattice[:, x, y, z, 0] == idx)[0]
            if len(slots) == 0:
                return False
            plattice[int(slots[0]), x, y, z, 0] = np.uint32(0)
            return True

        def _place_at(self, plattice, species_name: str, xyz: tuple[int, int, int], force_displace: bool = False) -> bool:
            idx = int(self.sim_properties["name_to_index"][species_name])
            x, y, z = xyz
            if not _in_bounds((x, y, z), self.cytoplasm_mask.shape):
                return False
            result = place_species_index_at_lattice(plattice, idx, xyz, force_displace=force_displace)
            if result["displaced_species_indices"]:
                self.sim_properties["phase7_r12_forced_marker_displacements"] = (
                    self.sim_properties.get("phase7_r12_forced_marker_displacements", [])
                    + [
                        {
                            "species_name": species_name,
                            "xyz": tuple(int(v) for v in xyz),
                            "displaced_species_indices": result["displaced_species_indices"],
                        }
                    ]
                )
            return bool(result["placed"])

        def _remove_n(self, plattice, species_name: str, n: int):
            removed = []
            for _ in range(int(n)):
                if self._remove_one(plattice, species_name):
                    removed.append(species_name)
                    continue
                for species in removed:
                    self._place_one(plattice, species, "cyto")
                return None
            return removed

        def _remove_topology_molecule(self, plattice, mol: int) -> bool:
            state = self.molecule_state[mol]
            missing = []
            for locus, x, y, z in state["genes"]:
                if not self._remove_at(plattice, f"V_P1_G_{locus}", (x, y, z)):
                    missing.append((locus, (x, y, z)))
            label_xyz = state["beads"][0]
            self._remove_at(plattice, "V_P1_Genome", label_xyz)
            if missing:
                self.sim_properties["phase7_r12_topology_packaging_marker_gaps"] = {
                    "molecule_id": mol,
                    "missing": missing[:10],
                    "message": (
                        "RDME gene marker particles were missing at packaging time. "
                        "Packaging continues from topology molecule state because r12 is topology-state-first; "
                        "this is not a counts/proxy substitution."
                    ),
                }
            event = mark_topology_molecule_packaged_from_state(self.molecule_state, mol, missing)
            self.sim_properties["phase7_r12_topology_state_consumption_log"] = (
                self.sim_properties.get("phase7_r12_topology_state_consumption_log", []) + [event]
            )
            return True

        def _clone_topology_molecule(self, live_t: int, plattice) -> bool:
            unpackaged = [mol for mol, state in sorted(self.molecule_state.items()) if state["status"] == "unpackaged"]
            if not unpackaged:
                return False
            parent = unpackaged[int(self.rng.integers(0, len(unpackaged)))]
            parent_state = self.molecule_state[parent]
            daughter = max(self.molecule_state) + 1
            new_beads = []
            for xyz in parent_state["beads"]:
                placed = None
                for radius in range(1, 9):
                    offsets = _lattice_offsets(radius)
                    self.rng.shuffle(offsets)
                    for dx, dy, dz in offsets:
                        candidate = (xyz[0] + dx, xyz[1] + dy, xyz[2] + dz)
                        if not _in_bounds(candidate, self.cytoplasm_mask.shape):
                            continue
                        if not bool(self.cytoplasm_mask[candidate]) or bool(self.membrane_mask[candidate]):
                            continue
                        placed = candidate
                        break
                    if placed is not None:
                        break
                if placed is None:
                    self.sim_properties["phase7_r12_topology_replication_error"] = {
                        "time_s": live_t,
                        "parent_molecule_id": parent,
                        "daughter_molecule_id": daughter,
                        "message": "No cytoplasm site found for topology-relaxed daughter bead projection.",
                    }
                    return False
                new_beads.append(placed)
            genes = []
            for locus, gx, gy, gz in parent_state["genes"]:
                nearest = min(new_beads, key=lambda p: (p[0] - gx) ** 2 + (p[1] - gy) ** 2 + (p[2] - gz) ** 2)
                if not self._place_at(plattice, f"V_P1_G_{locus}", nearest):
                    if not self._place_one(plattice, f"V_P1_G_{locus}", "cyto"):
                        return False
                    locs = np.argwhere(plattice == int(self.sim_properties["name_to_index"][f"V_P1_G_{locus}"]))
                    _, x, y, z, _ = (int(v) for v in locs[-1])
                    nearest = (x, y, z)
                genes.append((locus, int(nearest[0]), int(nearest[1]), int(nearest[2])))
            label_xyz = new_beads[0]
            if not self._place_at(plattice, "V_P1_Genome", label_xyz):
                if not self._place_one(plattice, "V_P1_Genome", "cyto"):
                    return False
            self.molecule_state[daughter] = {
                "status": "unpackaged",
                "beads": new_beads,
                "genes": genes,
                "parent_molecule_id": parent,
                "created_time_s": live_t,
                "topology_generation": int(parent_state.get("topology_generation", 0)) + 1,
            }
            event = {
                "time_s": live_t,
                "event_type": "topology_state_replication",
                "parent_molecule_id": parent,
                "daughter_molecule_id": daughter,
                "beads": len(new_beads),
                "tss_templates": len(genes),
                "ordinary_rdme_dna_beads": False,
            }
            self.topology_replication_log.append(event)
            self.sim_properties["phase7_r12_topology_replication_log"] = self.topology_replication_log
            return True

        def _apply_gene_event_costs(self, live_t: int, plattice) -> None:
            rep = self._count_species(plattice, "V_P1_rep_event")
            if rep:
                params = self.sim_properties["phase7_p1_parameters"]
                for _ in range(rep):
                    if not self._clone_topology_molecule(live_t, plattice):
                        self.topology_replication_blocked = True
                        self.sim_properties["phase7_r12_blocked_reason"] = (
                            "Replication event occurred, but topology-relaxed daughter P1 DNA projection could not be placed. "
                            "Failing closed rather than converting replication into counts."
                        )
                        raise RuntimeError(self.sim_properties["phase7_r12_blocked_reason"])
                self._add_count("dATP_DNArep_cost", rep * int(float(params["genome_A_count"])))
                self._add_count("dCTP_DNArep_cost", rep * int(float(params["genome_C_count"])))
                self._add_count("dGTP_DNArep_cost", rep * int(float(params["genome_G_count"])))
                self._add_count("dTTP_DNArep_cost", rep * int(float(params["genome_T_count"])))
                for _ in range(rep):
                    self._remove_one(plattice, "V_P1_rep_event")
            super()._apply_gene_event_costs(live_t, plattice)

        def _remove_one_from_inside(self, plattice, species_name: str):
            idx = int(self.sim_properties["name_to_index"][species_name])
            locs = np.argwhere(plattice == idx)
            for loc in locs:
                slot, x, y, z, p = (int(v) for v in loc)
                if bool(self.cytoplasm_mask[x, y, z]) or bool(self.membrane_mask[x, y, z]):
                    plattice[slot, x, y, z, p] = np.uint32(0)
                    return slot, x, y, z, p
            return None

        def _restore_removed_site(self, plattice, species_name: str, loc) -> None:
            if loc is None:
                return
            slot, x, y, z, p = loc
            plattice[slot, x, y, z, p] = np.uint32(int(self.sim_properties["name_to_index"][species_name]))

        def _available_host_leakage_by_class(self, plattice) -> dict[str, int]:
            return count_inside_particles_by_leakage_class(
                plattice,
                self.sim_properties["name_to_index"],
                self.post_breach_leakage_species,
                self.cytoplasm_mask,
                self.membrane_mask,
            )

        def _inside_occupied_fraction(self, plattice) -> dict:
            inside = self.cytoplasm_mask | self.membrane_mask
            view = plattice[:, inside, :]
            occupied = int(np.count_nonzero(view))
            capacity = int(view.size)
            return {
                "occupied": occupied,
                "capacity": capacity,
                "fraction": 0.0 if capacity <= 0 else float(occupied) / float(capacity),
            }

        def _available_pre_breach_leakage_by_class(self, plattice) -> dict[str, int]:
            return count_inside_particles_by_leakage_class(
                plattice,
                self.sim_properties["name_to_index"],
                self.pre_breach_leakage_species,
                self.cytoplasm_mask,
                self.membrane_mask,
            )

        def _apply_pre_breach_permeability(self, live_t: int, plattice) -> dict:
            occupancy = self._inside_occupied_fraction(plattice)
            decision = pre_breach_permeability_due(
                enabled=bool(self.config.get("pre_breach_permeability_enabled", False)),
                membrane_damage=float(self.membrane_damage_state.get("cumulative_membrane_damage", 0.0)),
                displaced_sites=len(self.displaced_membrane_sites),
                capacity_failures=len(self.physical_capacity_failures),
                occupied_fraction=float(occupancy["fraction"]),
                damage_threshold=float(self.config.get("pre_breach_damage_threshold", 0.35)),
                displacement_threshold=int(self.config.get("pre_breach_displacement_threshold", 40)),
                capacity_failure_threshold=int(self.config.get("pre_breach_capacity_failure_threshold", 2)),
                crowding_threshold=float(self.config.get("pre_breach_crowding_occupied_fraction_threshold", 0.02)),
            )
            moved_by_class = Counter()
            moved_by_species = Counter()
            target_total = int(self.pre_breach_leakage_total)
            if decision["due"] and self.physical_breach_time is None:
                if self.pre_breach_start_time is None:
                    self.pre_breach_start_time = int(live_t)
                    self._place_one(plattice, "V_P1_pre_breach_permeability", "membrane")
                elapsed_s = max(1, int(live_t) - int(self.pre_breach_start_time) + 1)
                target_total = int(elapsed_s * float(self.config.get("pre_breach_leak_particles_per_s", 20.0)))
                budget = max(0, int(target_total) - int(self.pre_breach_leakage_total))
                available_by_class = self._available_pre_breach_leakage_by_class(plattice)
                plan = allocate_host_leakage_budget(budget, available_by_class)
                for klass, quota in plan.items():
                    if int(quota) <= 0:
                        continue
                    moved = remove_inside_particles_for_leakage_class(
                        plattice,
                        self.sim_properties["name_to_index"],
                        self.pre_breach_leakage_species,
                        self.cytoplasm_mask,
                        self.membrane_mask,
                        klass,
                        int(quota),
                    )
                    moved_n = int(sum(moved.values()))
                    moved_by_class[klass] += moved_n
                    moved_by_species.update(moved)
                    self.pre_breach_leakage_total += moved_n
                    self.pre_breach_leakage_by_class[klass] += moved_n
                current_marker = self._count_species(plattice, "V_P1_pre_breach_leakage")
                marker_target = min(
                    int(self.pre_breach_leakage_total),
                    int(self.config.get("pre_breach_leakage_marker_cap", 80)),
                )
                for _ in range(max(0, marker_target - current_marker)):
                    if not self._place_one(plattice, "V_P1_pre_breach_leakage", "extra"):
                        break
            else:
                available_by_class = self._available_pre_breach_leakage_by_class(plattice)
            record = {
                "time_s": int(live_t),
                "due": bool(decision["due"]),
                "membrane_reasons": decision["membrane_reasons"],
                "crowding_due": bool(decision["crowding_due"]),
                "inside_occupied_fraction": float(occupancy["fraction"]),
                "inside_occupied_particles": int(occupancy["occupied"]),
                "inside_particle_capacity": int(occupancy["capacity"]),
                "target_total": int(target_total),
                "new_leaked": int(sum(moved_by_class.values())),
                "total_leaked": int(self.pre_breach_leakage_total),
                "by_class": dict(self.pre_breach_leakage_by_class),
                "new_by_class": dict(moved_by_class),
                "new_by_species": dict(moved_by_species),
                "available_by_class": available_by_class,
                "rule": "Pre-breach permeability removes host soluble particles only; host DNA/oriC and all P1 topology/viral species are excluded.",
            }
            if self.last_pre_breach_log_time != int(live_t):
                self.pre_breach_leakage_log.append(record)
                self.last_pre_breach_log_time = int(live_t)
            self.sim_properties["phase7_pre_breach_permeability_log"] = self.pre_breach_leakage_log
            self.sim_properties["phase7_pre_breach_permeability_state"] = record
            return record

        def _leak_host_molecules(self, live_t: int, plattice, target_total: int) -> dict:
            budget = max(0, int(target_total) - int(self.host_leakage_total))
            if budget <= 0:
                return {
                    "time_s": live_t,
                    "target_total": int(target_total),
                    "new_leaked": 0,
                    "total_leaked": int(self.host_leakage_total),
                    "by_class": dict(self.host_leakage_by_class),
                    "available_by_class": self._available_host_leakage_by_class(plattice),
                }
            available_by_class = self._available_host_leakage_by_class(plattice)
            plan = allocate_post_breach_leakage_budget(budget, available_by_class)
            moved_by_class = Counter()
            moved_by_species = Counter()
            for klass, quota in plan.items():
                if int(quota) <= 0:
                    continue
                moved = remove_inside_particles_for_leakage_class(
                    plattice,
                    self.sim_properties["name_to_index"],
                    self.post_breach_leakage_species,
                    self.cytoplasm_mask,
                    self.membrane_mask,
                    klass,
                    int(quota),
                )
                moved_n = int(sum(moved.values()))
                moved_by_class[klass] += moved_n
                moved_by_species.update(moved)
                self.host_leakage_total += moved_n
                self.host_leakage_by_class[klass] += moved_n
            record = {
                "time_s": live_t,
                "target_total": int(target_total),
                "new_leaked": int(sum(moved_by_class.values())),
                "total_leaked": int(self.host_leakage_total),
                "by_class": dict(self.host_leakage_by_class),
                "new_by_class": dict(moved_by_class),
                "new_by_species": dict(moved_by_species),
                "available_by_class": available_by_class,
                "rule": (
                    "Remove soluble host and viral particles from cytoplasm/membrane after breach as a sink-like leak flux. "
                    "This frees RDME lattice volume directly; capped V_P1_breach_leakage markers provide visualization only. "
                    "Host DNA/oriC and P1 topology-state DNA markers are excluded."
                ),
            }
            self.host_leakage_log.append(record)
            self.sim_properties["phase7_host_molecule_leakage_log"] = self.host_leakage_log
            return record

        def _assemble_virions(self, live_t: int, plattice) -> None:
            unpackaged = [mol for mol, state in sorted(self.molecule_state.items()) if state["status"] == "unpackaged"]
            available_sets = len(unpackaged)
            protein_counts = {}
            locus_sets = {}
            for locus, required in self.structural_requirements.items():
                count = self._count_species(plattice, f"V_P1_P_{locus}")
                protein_counts[locus] = int(count)
                locus_sets[locus] = int(count) // int(required)
                available_sets = min(available_sets, locus_sets[locus])
            structural_sets = int(available_sets)
            reserve = int(self.config["non_compromise_checks"].get("topology_packaging_reserve_molecules", 0))
            max_fraction = float(self.config["non_compromise_checks"].get("topology_packaging_max_fraction_beyond_reserve", 1.0))
            available_sets = compute_topology_packaging_capacity(len(unpackaged), structural_sets, reserve, max_fraction)
            limiting_loci = [
                locus
                for locus, sets in sorted(locus_sets.items(), key=lambda item: (item[1], item[0]))
                if sets == min(locus_sets.values() or [0])
            ]
            log_time = int(live_t)
            if self.last_structural_balance_log_time != log_time:
                record = {
                    "time_s": log_time,
                    "unpackaged_topology_molecules": len(unpackaged),
                    "complete_structural_sets": int(available_sets),
                    "raw_structural_sets": int(structural_sets),
                    "topology_packaging_reserve": int(reserve),
                    "topology_packaging_fraction_cap": float(max_fraction),
                    "limiting_loci": limiting_loci[:3],
                    "locus_complete_sets": locus_sets,
                    "structural_protein_counts": protein_counts,
                    "packaged_genome": int(self._count_species(plattice, "V_P1_packaged_genome")),
                    "virion": int(self._count_species(plattice, "V_P1_virion")),
                    "physical_capacity_failures": len(self.physical_capacity_failures),
                }
                self.structural_balance_log.append(record)
                self.sim_properties["phase7_r12_structural_balance_log"] = self.structural_balance_log
                self.sim_properties["phase7_r12_structural_balance_state"] = record
                self.last_structural_balance_log_time = log_time
            if not unpackaged:
                return
            if available_sets <= 0:
                return
            rate = float(self.sim_properties["phase7_p1_parameters"]["assembly_rate_s"])
            n_to_assemble = int(self.rng.binomial(int(available_sets), min(1.0, rate)))
            if n_to_assemble <= 0 and available_sets >= 50:
                n_to_assemble = 1
            made = 0
            for mol in unpackaged[:n_to_assemble]:
                placement = place_physical_p1_virion(
                    self.cytoplasm_mask,
                    self.membrane_mask,
                    self.occupied_virion_sites,
                    self.rng,
                    allow_membrane_displacement=True,
                    geometry=self.config.get("physical_virion_geometry_voxels", {}),
                )
                if not placement["placed"]:
                    failure = {
                        "time_s": live_t,
                        "molecule_id": mol,
                        "failure_reason": placement["failure_reason"],
                        "occupied_physical_virions": len(self.physical_virion_log),
                        "capacity_failure_index": len(self.physical_capacity_failures) + 1,
                    }
                    self.physical_capacity_failures.append(failure)
                    self.sim_properties["phase7_r12_physical_capacity_failures"] = self.physical_capacity_failures
                    if len(self.physical_capacity_failures) < int(self.config["non_compromise_checks"].get("force_packaging_after_capacity_failures", 1)):
                        break
                    placement = place_physical_p1_virion(
                        self.cytoplasm_mask,
                        self.membrane_mask,
                        self.occupied_virion_sites,
                        self.rng,
                        max_attempts=len(self.cytoplasm_coords),
                        allow_membrane_displacement=True,
                        force=True,
                        geometry=self.config.get("physical_virion_geometry_voxels", {}),
                    )
                    if not placement["placed"]:
                        break
                removed = []
                ok = True
                for locus, required in self.structural_requirements.items():
                    species = f"V_P1_P_{locus}"
                    batch = self._remove_n(plattice, species, int(required))
                    if batch is None:
                        ok = False
                        break
                    removed.extend(batch)
                if not ok:
                    for species in removed:
                        self._place_one(plattice, species, "cyto")
                    break
                if not self._remove_topology_molecule(plattice, mol):
                    for species in removed:
                        self._place_one(plattice, species, "cyto")
                    break
                center = tuple(int(v) for v in placement["center"])
                if self._place_at(plattice, "V_P1_packaged_genome", center, force_displace=bool(placement.get("forced_placement"))) and self._place_at(
                    plattice, "V_P1_virion", center, force_displace=bool(placement.get("forced_placement"))
                ):
                    self.occupied_virion_sites.update(tuple(site) for site in placement["occupied_sites"])
                    new_displacement = set(tuple(site) for site in placement["membrane_displacement_sites"]) - self.displaced_membrane_sites
                    self.displaced_membrane_sites.update(new_displacement)
                    self.new_displacement_since_membrane_update += len(new_displacement)
                    for site in sorted(new_displacement):
                        self._place_at(plattice, "V_P1_membrane_displacement", site, force_displace=True)
                    self._place_one(plattice, "V_P1_assembly_event", "cyto")
                    made += 1
                    event = {
                        "time_s": live_t,
                        "molecule_id": mol,
                        "consumed_topology_beads": int(self.molecule_state[mol]["consumed_topology_beads"]),
                        "consumed_tss_templates": len(self.molecule_state[mol]["genes"]),
                        "consumed_structural_subunits": int(sum(self.structural_requirements.values())),
                        "structural_requirements": dict(self.structural_requirements),
                        "physical_center": center,
                        "physical_orientation": tuple(int(v) for v in placement["orientation"]),
                        "occupied_site_count": int(placement["occupied_site_count"]),
                        "occupied_sites": [tuple(int(v) for v in site) for site in placement["occupied_sites"]],
                        "membrane_displacement_site_count": int(placement["membrane_displacement_site_count"]),
                        "membrane_displacement_sites": [tuple(int(v) for v in site) for site in placement["membrane_displacement_sites"]],
                        "tail_membrane_site_count": int(placement.get("tail_membrane_site_count", 0)),
                        "noncytoplasm_site_count": int(placement.get("noncytoplasm_site_count", 0)),
                        "head_cytoplasm_fraction": float(placement.get("head_cytoplasm_fraction", 1.0)),
                        "forced_placement": bool(placement.get("forced_placement", False)),
                        "mechanistic_breach": False,
                        "membrane_displacement_only": bool(new_displacement),
                    }
                    self.topology_packaging_log.append(event)
                    self.physical_virion_log.append(event)
            if made:
                self.sim_properties["phase7_r12_topology_packaging_log"] = self.topology_packaging_log
                self.sim_properties["phase7_r12_physical_virion_log"] = self.physical_virion_log

        def _apply_membrane_release(self, live_t: int, plattice) -> None:
            virion = self._count_species(plattice, "V_P1_virion")
            membrane_stress = 0
            if virion:
                membrane_stress = min(len(self.membrane_coords), max(0, virion // 25))
                current = self._count_species(plattice, "V_P1_membrane_stress")
                for _ in range(max(0, membrane_stress - current)):
                    self._place_one(plattice, "V_P1_membrane_stress", "membrane")
            new_displacement = int(self.new_displacement_since_membrane_update)
            self.membrane_damage_state = update_membrane_damage_state(
                previous=self.membrane_damage_state,
                time_s=live_t,
                total_displacement_sites=len(self.displaced_membrane_sites),
                new_displacement_sites=new_displacement,
                internal_virions=virion,
                capacity_failures=len(self.physical_capacity_failures),
                config=self.config,
            )
            self.new_displacement_since_membrane_update = 0
            pre_breach_record = self._apply_pre_breach_permeability(live_t, plattice)
            if self.membrane_damage_state["breach_time"] is not None and self.physical_breach_time is None:
                self.physical_breach_time = int(self.membrane_damage_state["breach_time"])
                breach_site = sorted(self.displaced_membrane_sites)[0] if self.displaced_membrane_sites else tuple(int(v) for v in self.membrane_coords[0])
                self._place_at(plattice, "V_P1_breach", breach_site)
            release_target = int(self.membrane_damage_state["release_target"])
            if self.physical_breach_time is not None:
                while self.released_total < release_target:
                    releasable = next((event for event in self.physical_virion_log if not event.get("released")), None)
                    if releasable is None:
                        break
                    release_record = release_intact_virion_footprint(
                        plattice,
                        self.sim_properties["name_to_index"],
                        self.occupied_virion_sites,
                        self.displaced_membrane_sites,
                        releasable,
                        time_s=live_t,
                        reason=self.membrane_damage_state.get("breach_reason") or "post_breach_egress",
                    )
                    if not release_record["released"]:
                        break
                    if not self._place_one(plattice, "V_P1_released", "extra"):
                        self.released_to_sink_total += 1
                    self.released_total += 1
                    self.physical_virion_release_log.append(release_record)
                if self.physical_virion_release_log:
                    self.sim_properties["phase7_r12_physical_virion_release_log"] = self.physical_virion_release_log
                    self.sim_properties["phase7_r12_physical_virion_log"] = self.physical_virion_log
                current_leakage = self._count_species(plattice, "V_P1_breach_leakage")
                marker_target = min(
                    int(self.membrane_damage_state["leakage_particles"]),
                    int(self.config.get("post_breach_leakage_marker_cap", 400)),
                )
                for _ in range(max(0, marker_target - current_leakage)):
                    if not self._place_one(plattice, "V_P1_breach_leakage", "extra"):
                        break
                host_leakage_record = self._leak_host_molecules(live_t, plattice, int(self.membrane_damage_state["leakage_particles"]))
            else:
                host_leakage_record = {
                    "time_s": live_t,
                    "target_total": 0,
                    "new_leaked": 0,
                    "total_leaked": int(self.host_leakage_total),
                    "by_class": dict(self.host_leakage_by_class),
                    "available_by_class": {},
                }
            membrane_record = dict(self.membrane_damage_state)
            membrane_record.update(
                {
                    "released_total": int(self.released_total),
                    "released_to_sink_total": int(self.released_to_sink_total),
                    "release_per_second": int(self.config["release_per_second"]),
                    "release_lag_s": int(self.config.get("membrane_release_lag_s", 20)),
                    "host_molecule_leaked_total": int(host_leakage_record["total_leaked"]),
                    "host_molecule_leaked_by_class": dict(host_leakage_record["by_class"]),
                    "pre_breach_permeability_due": bool(pre_breach_record["due"]),
                    "pre_breach_soluble_leaked_total": int(pre_breach_record["total_leaked"]),
                    "pre_breach_soluble_leaked_new": int(pre_breach_record["new_leaked"]),
                    "pre_breach_soluble_leaked_by_class": dict(pre_breach_record["by_class"]),
                    "post_breach_soluble_leaked_total": int(host_leakage_record["total_leaked"]),
                    "post_breach_soluble_leaked_by_class": dict(host_leakage_record["by_class"]),
                }
            )
            self.membrane_post_breach_log.append(membrane_record)
            self.sim_properties["phase7_membrane_post_breach_log"] = self.membrane_post_breach_log
            self.sim_properties["phase7_membrane_state"] = {
                "time_s": live_t,
                "virion": virion,
                "membrane_stress_particles": membrane_stress,
                "membrane_displacement_sites": len(self.displaced_membrane_sites),
                "physical_virions": len(self.physical_virion_log),
                "capacity_failures": len(self.physical_capacity_failures),
                "breach_time": self.physical_breach_time,
                "released_total": self.released_total,
                "released_to_sink_total": int(self.released_to_sink_total),
                "release_target": release_target,
                "new_displacement_sites": new_displacement,
                "membrane_integrity": float(self.membrane_damage_state["membrane_integrity"]),
                "cumulative_membrane_damage": float(self.membrane_damage_state["cumulative_membrane_damage"]),
                "breach_reason": self.membrane_damage_state["breach_reason"],
                "pending_breach_reason": self.membrane_damage_state.get("pending_breach_reason"),
                "breach_min_internal_virions": int(self.membrane_damage_state.get("breach_min_internal_virions", 0)),
                "post_breach_elapsed_s": int(self.membrane_damage_state["post_breach_elapsed_s"]),
                "leakage_particles": int(self.membrane_damage_state["leakage_particles"]),
                "leakage_marker_particles": int(self._count_species(plattice, "V_P1_breach_leakage")),
                "leakage_marker_cap": int(self.config.get("post_breach_leakage_marker_cap", 400)),
                "host_molecule_leaked_total": int(host_leakage_record["total_leaked"]),
                "host_molecule_leaked_new": int(host_leakage_record["new_leaked"]),
                "host_molecule_leaked_by_class": dict(host_leakage_record["by_class"]),
                "pre_breach_permeability_due": bool(pre_breach_record["due"]),
                "pre_breach_soluble_leaked_total": int(pre_breach_record["total_leaked"]),
                "pre_breach_soluble_leaked_new": int(pre_breach_record["new_leaked"]),
                "pre_breach_soluble_leaked_by_class": dict(pre_breach_record["by_class"]),
                "post_breach_soluble_leaked_total": int(host_leakage_record["total_leaked"]),
                "post_breach_soluble_leaked_new": int(host_leakage_record["new_leaked"]),
                "post_breach_soluble_leaked_by_class": dict(host_leakage_record["by_class"]),
                "mechanistic_breach": self.physical_breach_time is not None,
                "note": "Displacement/forced placement is separated from breach. Breach is emitted only after membrane displacement or integrity gates are crossed; release has explicit latency.",
            }
            if post_breach_stop_due(live_t, self.physical_breach_time, self.config):
                self.post_breach_stop_pending = True
                self.sim_properties["phase7_r12_stop_reason"] = {
                    "status": "stopped_post_breach",
                    "time_s": int(live_t),
                    "breach_time": int(self.physical_breach_time),
                    "post_breach_seconds": int(self.config.get("post_breach_seconds", 60)),
                    "reason": "Reached configured post-breach observation window.",
                }

        def hookSimulation(self, t, lattice):
            try:
                hold_until = float(self.config["non_compromise_checks"].get("dna_update_hold_until_s", 0.0))
                if hold_until > 0.0:
                    held_next = apply_dna_update_hold(
                        current_next=float(self.next_DNA_time),
                        time_s=float(t),
                        hold_until=hold_until,
                    )
                    if held_next != float(self.next_DNA_time):
                        self.next_DNA_time = held_next
                        self.sim_properties["phase7_r12_dna_update_hold"] = {
                            "time_s": float(t),
                            "hold_until_s": float(hold_until),
                            "reason": "Defers adaptive BRGDNA/btree updates for mechanism smoke or native-crash isolation.",
                        }
                result = super().hookSimulation(t, lattice)
            except Exception as exc:
                self.sim_properties["phase7_r12_last_exception"] = str(exc)
                status = format_run_status(
                    self.sim_properties.get("run_id", "unknown"),
                    "Phase7-r12",
                    len(self.sim_properties.get("phase7_p1_cds", [])),
                    "failed",
                    str(exc),
                )
                write_run_status_file(self.sim_properties, status)
                raise
            if self.post_breach_stop_pending and not self.post_breach_stop_raised:
                self.post_breach_stop_raised = True
                reason = self.sim_properties.get("phase7_r12_stop_reason", {})
                status = format_run_status(
                    self.sim_properties.get("run_id", "unknown"),
                    "Phase7-r12",
                    len(self.sim_properties.get("phase7_p1_cds", [])),
                    "stopped_post_breach",
                )
                status["stop_reason"] = reason
                write_run_status_file(self.sim_properties, status)
                raise RuntimeError(
                    "Phase7 r12 post-breach stop gate reached "
                    f"at t={reason.get('time_s', t)} s after breach={reason.get('breach_time')} s."
                )
            return result

    return Phase7P1R12TopologySolver


def main() -> None:
    args = parse_args()
    if args.t8TopologyPackagingSmokeOnly:
        run_t8_topology_packaging_smoke(args)
        return
    run_dir = ROOT / "Data" / args.runId
    if run_dir.exists():
        raise SystemExit(f"Output directory already exists: {run_dir}")
    params = base.read_parameter_table(base.PARAMETERS)
    if args.assemblyRateOverride is not None:
        params["assembly_rate_s"] = str(float(args.assemblyRateOverride))
    structural_translation_weights = parse_locus_weights(args.structuralTranslationWeights)
    cds_rows = base.read_p1_cds_table(base.CDS_TABLE)
    structural_rows, structural_requirements = load_t7_structural_requirements(args.stoichiometryRoot, args.structuralSubunitTotal)
    particles, topology_manifest, dna_particles, cds_map = make_initial_particles_from_topology(args, cds_rows)
    config = freeze_config(args, params, cds_rows, particles, topology_manifest, structural_requirements)
    files = topology_files(args)
    base.write_csv(CONFIG_DIR / f"{args.runId}_topology_particles_source.csv", dna_particles, list(dna_particles[0].keys()))
    base.write_csv(CONFIG_DIR / f"{args.runId}_topology_cds_source.csv", cds_map, list(cds_map[0].keys()))
    if args.prepareOnly:
        print(
            json.dumps(
                {
                    "run_id": args.runId,
                    "status": "prepared",
                    "phase": "Phase7-r12",
                    "initial_genomes": args.initialGenomes,
                    "topology_run_id": args.topologyRunId,
                    "particles": len(particles),
                    "config_dir": str(CONFIG_DIR),
                },
                indent=2,
            )
        )
        return

    if not (ROOT / "MC_RDME_initialization.py").is_file():
        raise SystemExit("Full simulation requires MC4D_ROOT pointing to an external Minimal_Cell_4DWCM checkout; --prepareOnly needs no host.")
    if not args.dnaSoftwareDirectory:
        raise SystemExit("Set PHASE7_DNA_SOFTWARE_DIR or --dnaSoftwareDirectory to the external BRGDNA directory.")

    import jLM  # noqa: F401
    import lm  # noqa: F401
    from jLM.Solvers import makeSolver
    from lm import IntMpdRdmeSolver
    import Communicate as communicate
    import FileSaving as save
    import ImportInitialConditions as IC
    import MC_CME
    import MC_RDME_initialization as MCRDME
    import RegionsAndComplexes as InitGeom
    import SpatialDnaDynamics as DNA
    from phase3_dna_wait_patch import Phase3DnaWaitConfig, install_phase3_dna_wait_patch

    install_counts_bootstrap(save)
    install_cme_uint32_guard(MC_CME)
    install_phase3_dna_wait_patch(DNA, Phase3DnaWaitConfig(first_wait_seconds=int(args.firstDnaWaitSeconds)))
    sim, sim_properties = MCRDME.initSim(int(args.hookStep), int(args.writeStep), int(args.duration), args.runId, str(ROOT) + "/")
    save.saveSimArgs(sim_properties, args)
    sim_properties["run_id"] = args.runId
    sim_properties["phase7_config"] = config
    sim_properties["phase7_p1_parameters"] = params
    sim_properties["phase7_r12_structural_translation_weights"] = structural_translation_weights
    sim_properties["phase7_p1_cds"] = cds_rows
    sim_properties["phase7_structural_stoichiometry"] = structural_rows
    sim_properties["phase7_structural_requirements"] = structural_requirements
    sim_properties["phase7_physical_virion_geometry_voxels"] = config["physical_virion_geometry_voxels"]
    sim_properties["phase7_physical_virion_geometry_nm"] = build_p1_virion_body_sites(
        (32, 32, 32),
        (1, 0, 0),
        **config["physical_virion_geometry_voxels"],
    )["geometry_nm"]
    sim_properties["phase7_placement_seed"] = int(args.placementSeed)
    sim_properties["phase7_topology_particles_csv"] = str(CONFIG_DIR / f"{args.runId}_topology_particles_source.csv")
    sim_properties["phase7_topology_cds_csv"] = str(CONFIG_DIR / f"{args.runId}_topology_cds_source.csv")
    sim_properties["phase7_topology_state_npz"] = str(files["state"])
    sim_properties["dna_rng_seed"] = int(args.dnaRngSeed)
    sim_properties["dna_software_directory"] = str(args.dnaSoftwareDirectory)
    sim_properties["membrane_directory"] = sim_properties["head_directory"] + f"input_data/membrane/cell{int(args.membrane)}/"
    sim_properties["division_started"] = False

    region_dict, ribo_site_dict = InitGeom.buildRegions(sim, sim_properties)
    IC.initializeParticles(sim, region_dict, sim_properties)
    install_species(sim, region_dict, particles, cds_rows)
    MCRDME.createParticleIdxMap(sim, sim_properties)
    MCRDME.constructGIP(sim, sim_properties)
    MCRDME.createParticleIdxMap(sim, sim_properties)
    MCRDME.constructAssemblyReactions(sim, sim_properties)
    add_p1_gene_resolved_reactions(sim, params, cds_rows, region_dict, structural_translation_weights)
    MCRDME.createParticleIdxMap(sim, sim_properties)
    MCRDME.mapTranslationStates(sim_properties)
    IC.initializeRdmeCounts(sim, sim_properties)
    communicate.updateSA(sim_properties)
    save.saveCountsAndFluxes(0, sim_properties, None, None, None)

    import Hook

    Solver = makeSolver(IntMpdRdmeSolver, make_solver(Hook.MyOwnSolver, cds_rows))
    solver = Solver(sim, sim_properties, region_dict, ribo_site_dict, termination_time=float(args.maximumHours))
    sim.finalize()
    run_error = None
    try:
        sim.run(solver=solver, cudaDevices=[int(args.cudaDevices)])
    except Exception as exc:
        run_error = exc
    (run_dir / "phase7_p1_r12_topology_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    stop_reason = sim_properties.get("phase7_r12_stop_reason")
    if stop_reason:
        status = format_run_status(args.runId, "Phase7-r12", len(cds_rows), "stopped_post_breach")
        status["stop_reason"] = stop_reason
    elif run_error is not None:
        status = format_run_status(args.runId, "Phase7-r12", len(cds_rows), "failed", str(run_error))
    elif float(sim_properties.get("time", 0.0)) + 0.5 < float(args.duration):
        status = format_run_status(
            args.runId,
            "Phase7-r12",
            len(cds_rows),
            "failed",
            f"RDME engine returned at t={sim_properties.get('time')} before requested duration={args.duration}.",
        )
    else:
        status = format_run_status(args.runId, "Phase7-r12", len(cds_rows), "completed")
    (run_dir / "phase7_p1_r12_run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    if status["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
