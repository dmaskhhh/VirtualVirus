#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio import SeqIO


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
INPUT_ROOT = REPO_ROOT / "inputs" / "molecular"
PHASE2 = INPUT_ROOT / "phase2_records"
PHASE7 = Path(os.environ.get("VV_MOLECULAR_OUTPUT_ROOT", str(REPO_ROOT / "outputs" / "molecular"))).expanduser().resolve() / "phase7_records"
MASK_DIR = INPUT_ROOT / "geometry"
P1_GENBANK = PHASE2 / "viral_genomes" / "NC_002515.1.gb"
P1_CDS_TABLE = PHASE2 / "sequence_accounting" / "cds_table.csv"
DEFAULT_OUT_ROOT = PHASE7 / "live_runs" / "step7_p1_linear_dna_topology"
DEFAULT_LAMMPS_EXE = Path(os.environ.get("VV_LAMMPS_EXE", "lmp_twistable_kokkos"))
DEFAULT_DNA_MODEL = Path(os.environ.get("VV_DNA_MODEL_DIR", "external/LAMMPS_DNA_model_kk"))

BP_PER_BEAD = 10
DNA_BEAD_RADIUS_A = 17.0
DNA_BEAD_SPACING_A = 2.0 * DNA_BEAD_RADIUS_A
RDME_LATTICE_A = 100.0


@dataclass(frozen=True)
class GenomeMeta:
    accession: str
    length_bp: int
    topology: str
    molecule_type: str
    description: str
    sha256: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build a Cell-method-aligned P1 linear dsDNA topology substrate for RDME. "
            "This is not the old counts/proxy runner."
        )
    )
    ap.add_argument("--runId", required=True)
    ap.add_argument("--initialGenomes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11011)
    ap.add_argument("--outRoot", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--genbank", type=Path, default=P1_GENBANK)
    ap.add_argument("--cdsTable", type=Path, default=P1_CDS_TABLE)
    ap.add_argument("--maskDir", type=Path, default=MASK_DIR)
    ap.add_argument("--cellRadiusAngstrom", type=float, default=1900.0)
    ap.add_argument("--minNonbondedDistanceAngstrom", type=float, default=24.0)
    ap.add_argument("--placementAttempts", type=int, default=20)
    ap.add_argument("--chainTrialLimit", type=int, default=2500)
    ap.add_argument("--lammpsExe", type=Path, default=DEFAULT_LAMMPS_EXE)
    ap.add_argument("--dnaModelDir", type=Path, default=DEFAULT_DNA_MODEL)
    ap.add_argument("--runLammps", action="store_true")
    ap.add_argument("--lammpsSteps", type=int, default=200)
    ap.add_argument("--lammpsTimestepFs", type=float, default=1.0e5)
    ap.add_argument("--prepareOnly", action="store_true")
    return ap.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_genome_meta(genbank: Path) -> GenomeMeta:
    if not genbank.exists():
        raise FileNotFoundError(genbank)
    record = next(SeqIO.parse(str(genbank), "genbank"))
    topology = str(record.annotations.get("topology", "")).lower()
    molecule_type = str(record.annotations.get("molecule_type", "")).upper()
    if topology != "linear":
        raise ValueError(f"P1 adapter requires GenBank topology=linear; got {topology!r}")
    if "DNA" not in molecule_type:
        raise ValueError(f"P1 adapter requires DNA molecule_type; got {molecule_type!r}")
    return GenomeMeta(
        accession=str(record.id),
        length_bp=len(record.seq),
        topology=topology,
        molecule_type=molecule_type,
        description=str(record.description),
        sha256=sha256(genbank),
    )


def parse_location(location: str) -> tuple[int, int, str]:
    strand = "-" if location.startswith("complement(") else "+"
    nums = [int(v) for v in re.findall(r"\d+", location)]
    if not nums:
        raise ValueError(f"Cannot parse CDS location: {location}")
    return min(nums), max(nums), strand


def bead_index_for_bp(bp: int) -> int:
    return max(1, int(math.ceil(bp / BP_PER_BEAD)))


def load_p1_cds(cds_table: Path, accession: str) -> list[dict[str, str]]:
    rows = [
        row.copy()
        for row in read_csv(cds_table)
        if row.get("accession") == accession and row.get("virus") == "P1-like"
    ]
    rows.sort(key=lambda row: int(row["cds_index"]))
    if len(rows) != 11:
        raise ValueError(f"Expected 11 P1 CDS rows for {accession}; found {len(rows)}")
    for row in rows:
        start, end, strand = parse_location(row["location"])
        tx_start = end if strand == "-" else start
        tx_end = start if strand == "-" else end
        row.update(
            {
                "start_bp": str(start),
                "end_bp": str(end),
                "strand": strand,
                "tx_start_bp": str(tx_start),
                "tx_end_bp": str(tx_end),
                "start_bead": str(bead_index_for_bp(start)),
                "end_bead": str(bead_index_for_bp(end)),
                "tx_start_bead": str(bead_index_for_bp(tx_start)),
                "tx_end_bead": str(bead_index_for_bp(tx_end)),
            }
        )
    return rows


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return random_unit_vector(rng)
    return vec / norm


def coord_to_lattice_site(coord: np.ndarray, lattice_shape: tuple[int, int, int]) -> np.ndarray:
    center = np.array(lattice_shape, dtype=np.float64) / 2.0
    return np.floor(coord / RDME_LATTICE_A + center).astype(np.int64)


def coord_is_cytoplasm(coord: np.ndarray, cytoplasm: np.ndarray, membrane: np.ndarray) -> bool:
    site = coord_to_lattice_site(coord, tuple(int(v) for v in cytoplasm.shape))
    if np.any(site < 0) or np.any(site >= np.array(cytoplasm.shape)):
        return False
    x, y, z = (int(v) for v in site)
    return bool(cytoplasm[x, y, z]) and not bool(membrane[x, y, z])


def random_cytoplasm_coord(
    rng: np.random.Generator,
    cytoplasm: np.ndarray,
    membrane: np.ndarray,
    max_radius_a: float,
) -> np.ndarray:
    sites = np.argwhere(cytoplasm & ~membrane)
    center = np.array(cytoplasm.shape, dtype=np.float64) / 2.0
    for _ in range(2000):
        site = sites[int(rng.integers(0, len(sites)))]
        coord = (site + rng.uniform(0.15, 0.85, size=3) - center) * RDME_LATTICE_A
        if np.linalg.norm(coord) < max_radius_a:
            return coord
    raise RuntimeError("Failed to sample an initial P1 coordinate from the cytoplasm mask")


def self_avoiding_linear_chain(
    bead_count: int,
    rng: np.random.Generator,
    cytoplasm: np.ndarray,
    membrane: np.ndarray,
    start_radius_a: float,
    min_nonbonded_a: float,
    chain_trial_limit: int,
    forbidden: np.ndarray | None = None,
) -> np.ndarray:
    coords = np.zeros((bead_count, 3), dtype=np.float64)
    direction = random_unit_vector(rng)
    coords[0] = random_cytoplasm_coord(rng, cytoplasm, membrane, start_radius_a)
    for i in range(1, bead_count):
        accepted = False
        for _ in range(chain_trial_limit):
            direction = 0.65 * direction + 0.35 * random_unit_vector(rng)
            direction /= np.linalg.norm(direction)
            if np.linalg.norm(coords[i - 1]) > start_radius_a * 0.85:
                direction = 0.45 * direction - 0.55 * coords[i - 1] / np.linalg.norm(coords[i - 1])
                direction /= np.linalg.norm(direction)
            candidate = coords[i - 1] + DNA_BEAD_SPACING_A * direction
            if not coord_is_cytoplasm(candidate, cytoplasm, membrane):
                continue
            if i > 2:
                d = np.linalg.norm(coords[: i - 2] - candidate, axis=1)
                if np.any(d < min_nonbonded_a):
                    continue
            if forbidden is not None and len(forbidden):
                d_forbidden = np.linalg.norm(forbidden - candidate, axis=1)
                if np.any(d_forbidden < min_nonbonded_a):
                    continue
            coords[i] = candidate
            accepted = True
            break
        if not accepted:
            raise RuntimeError(f"Failed to place linear P1 bead {i + 1}/{bead_count}")
    return coords


def place_initial_genomes(
    initial_genomes: int,
    bead_count: int,
    seed: int,
    cytoplasm: np.ndarray,
    membrane: np.ndarray,
    start_radius_a: float,
    min_nonbonded_a: float,
    placement_attempts: int,
    chain_trial_limit: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    molecules = []
    forbidden = np.empty((0, 3), dtype=np.float64)
    for genome_i in range(initial_genomes):
        for attempt in range(placement_attempts):
            try:
                chain = self_avoiding_linear_chain(
                    bead_count=bead_count,
                    rng=rng,
                    cytoplasm=cytoplasm,
                    membrane=membrane,
                    start_radius_a=start_radius_a,
                    min_nonbonded_a=min_nonbonded_a,
                    chain_trial_limit=chain_trial_limit,
                    forbidden=forbidden,
                )
                molecules.append(chain)
                forbidden = np.vstack([forbidden, chain])
                break
            except RuntimeError:
                if attempt == placement_attempts - 1:
                    raise RuntimeError(f"Failed to place genome molecule {genome_i + 1}")
    return np.stack(molecules, axis=0)


def build_lammps_topology(coords: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    molecules, beads, _ = coords.shape
    atoms: list[dict] = []
    bonds: list[dict] = []
    angles: list[dict] = []
    atom_id = 1
    bond_id = 1
    angle_id = 1
    for mol_i in range(molecules):
        mol_id = mol_i + 1
        first_atom = atom_id
        for bead_i in range(beads):
            x, y, z = coords[mol_i, bead_i]
            atoms.append(
                {
                    "atom_id": atom_id,
                    "mol_id": mol_id,
                    "bead_index": bead_i + 1,
                    "atom_type": 3,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
            atom_id += 1
        for bead_i in range(beads - 1):
            bonds.append(
                {
                    "bond_id": bond_id,
                    "bond_type": 1,
                    "a1": first_atom + bead_i,
                    "a2": first_atom + bead_i + 1,
                }
            )
            bond_id += 1
        for bead_i in range(beads - 2):
            angles.append(
                {
                    "angle_id": angle_id,
                    "angle_type": 1,
                    "a1": first_atom + bead_i,
                    "a2": first_atom + bead_i + 1,
                    "a3": first_atom + bead_i + 2,
                }
            )
            angle_id += 1
    return atoms, bonds, angles


def assert_linear_topology(atoms: list[dict], bonds: list[dict], bead_count: int, molecules: int) -> None:
    expected_bonds = molecules * (bead_count - 1)
    if len(bonds) != expected_bonds:
        raise AssertionError(f"Linear topology requires {expected_bonds} bonds; found {len(bonds)}")
    bond_pairs = {tuple(sorted((int(b["a1"]), int(b["a2"])))) for b in bonds}
    for mol_i in range(molecules):
        first = mol_i * bead_count + 1
        last = first + bead_count - 1
        if tuple(sorted((first, last))) in bond_pairs:
            raise AssertionError("Terminal closure bond detected; P1 must remain linear")
    if any(int(atom["atom_type"]) != 3 for atom in atoms):
        raise AssertionError("P1 adapter should not reuse circular Ori/Ter/Fork labels as topology")


def write_lammps_data(path: Path, atoms: list[dict], bonds: list[dict], angles: list[dict], box_a: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("LAMMPS data: P1 linear dsDNA topology, 10 bp/bead, no terminal closure\n")
    lines.append(f"{len(atoms)} atoms")
    lines.append(f"{len(bonds)} bonds")
    lines.append(f"{len(angles)} angles\n")
    lines.append("8 atom types")
    lines.append("2 bond types")
    lines.append("4 angle types\n")
    lines.append(f"{-box_a:.6f} {box_a:.6f} xlo xhi")
    lines.append(f"{-box_a:.6f} {box_a:.6f} ylo yhi")
    lines.append(f"{-box_a:.6f} {box_a:.6f} zlo zhi\n")
    lines.append("Atoms # angle\n")
    for atom in atoms:
        lines.append(
            f"{atom['atom_id']} {atom['mol_id']} {atom['atom_type']} "
            f"{atom['x']:.8f} {atom['y']:.8f} {atom['z']:.8f}"
        )
    lines.append("\nBonds\n")
    for bond in bonds:
        lines.append(f"{bond['bond_id']} {bond['bond_type']} {bond['a1']} {bond['a2']}")
    lines.append("\nAngles\n")
    for angle in angles:
        lines.append(f"{angle['angle_id']} {angle['angle_type']} {angle['a1']} {angle['a2']} {angle['a3']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_lammps_input(path: Path, data_file: Path, args: argparse.Namespace, run_dir: Path) -> None:
    dump_file = run_dir / "p1_linear_dna_relaxed.lammpstrj"
    relaxed_data = run_dir / "p1_linear_dna_relaxed.data"
    text = f"""# P1 linear dsDNA topology smoke run.
# Generated by run_phase7_step7_p1_real_topology_amplification.py

variable DNA_model_dir string {args.dnaModelDir}
variable rng_seed internal {int(args.seed)}
variable delta_t internal {float(args.lammpsTimestepFs):.8g}
variable steps internal {int(args.lammpsSteps)}

include ${{DNA_model_dir}}/protocol_subroutines/subroutine.global_setup
read_data {data_file}
include ${{DNA_model_dir}}/lmp.DNA_physical_params
include ${{DNA_model_dir}}/potentials/lmp.twistable_DNA_model_FENE_kk
include ${{DNA_model_dir}}/protocol_subroutines/subroutine.neighbor_setup
include ${{DNA_model_dir}}/protocol_subroutines/subroutine.group_setup
variable Ori_bdry_attraction internal 0
variable Ori_pair_repulsion internal 0
include ${{DNA_model_dir}}/potentials/lmp.DNA_pair_hard_kk

group p1dna type 3
fix dnaBD p1dna brownian/kk ${{T}} ${{rng_seed}} gamma_t ${{gamma_t_mono}}

thermo 50
thermo_style custom step atoms temp epair ebond eangle
timestep ${{delta_t}}
dump p1dump all custom 50 {dump_file} id mol type x y z
dump_modify p1dump sort id first yes
run ${{steps}}
write_data {relaxed_data}
"""
    path.write_text(text, encoding="utf-8")


def parse_last_lammps_dump(path: Path, expected_atoms: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    atom_header_indices = [i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS")]
    if not atom_header_indices:
        raise ValueError(f"No atom frames found in {path}")
    header_i = atom_header_indices[-1]
    fields = lines[header_i].split()[2:]
    idx = {name: fields.index(name) for name in fields}
    needed = {"id", "x", "y", "z"}
    if not needed.issubset(idx):
        raise ValueError(f"LAMMPS dump lacks fields {needed}; got {fields}")
    coords = np.zeros((expected_atoms, 3), dtype=np.float64)
    for line in lines[header_i + 1 : header_i + 1 + expected_atoms]:
        parts = line.split()
        atom_id = int(parts[idx["id"]])
        coords[atom_id - 1] = [float(parts[idx["x"]]), float(parts[idx["y"]]), float(parts[idx["z"]])]
    return coords


def run_lammps(args: argparse.Namespace, input_file: Path, run_dir: Path, expected_atoms: int) -> np.ndarray:
    if not args.lammpsExe.exists():
        raise FileNotFoundError(args.lammpsExe)
    if not args.dnaModelDir.exists():
        raise FileNotFoundError(args.dnaModelDir)
    log_file = run_dir / "p1_linear_dna_lammps.log"
    cmd = [str(args.lammpsExe), "-k", "on", "g", "1", "-sf", "kk", "-in", str(input_file), "-log", str(log_file)]
    subprocess.run(cmd, cwd=str(run_dir), check=True)
    return parse_last_lammps_dump(run_dir / "p1_linear_dna_relaxed.lammpstrj", expected_atoms)


def coords_to_lattice(coords: np.ndarray, lattice_shape: tuple[int, int, int]) -> np.ndarray:
    center = np.array(lattice_shape, dtype=np.float64) / 2.0
    lattice = np.floor(coords / RDME_LATTICE_A + center).astype(np.int64)
    if np.any(lattice < 0) or np.any(lattice >= np.array(lattice_shape)):
        bad = lattice[(lattice < 0).any(axis=1) | (lattice >= np.array(lattice_shape)).any(axis=1)][0]
        raise ValueError(f"DNA bead maps outside RDME lattice: {bad.tolist()}")
    return lattice


def assert_inside_rdme_cytoplasm(lattice: np.ndarray, cytoplasm: np.ndarray, membrane: np.ndarray) -> None:
    bad = []
    for i, site in enumerate(lattice):
        x, y, z = (int(v) for v in site)
        if not bool(cytoplasm[x, y, z]) or bool(membrane[x, y, z]):
            bad.append((i + 1, x, y, z))
            if len(bad) >= 5:
                break
    if bad:
        raise ValueError(f"P1 DNA bead not in cytoplasm RDME lattice: first bad sites {bad}")


def write_topology_particles(path: Path, coords: np.ndarray, lattice: np.ndarray, bead_count: int) -> None:
    rows = []
    if len(coords) != len(lattice):
        raise ValueError(f"Coordinate/lattice length mismatch: {len(coords)} vs {len(lattice)}")
    for flat_i, (coord, site) in enumerate(zip(coords, lattice)):
        molecule_id = flat_i // bead_count + 1
        bead_index = flat_i % bead_count + 1
        rows.append(
            {
                "particle": f"P1_DNA_bead_{molecule_id}_{bead_index}",
                "molecule_id": molecule_id,
                "bead_index": bead_index,
                "bp_start": (bead_index - 1) * BP_PER_BEAD + 1,
                "bp_end": bead_index * BP_PER_BEAD,
                "x_A": f"{coord[0]:.8f}",
                "y_A": f"{coord[1]:.8f}",
                "z_A": f"{coord[2]:.8f}",
                "lattice_x": int(site[0]),
                "lattice_y": int(site[1]),
                "lattice_z": int(site[2]),
            }
        )
    write_csv(
        path,
        rows,
        [
            "particle",
            "molecule_id",
            "bead_index",
            "bp_start",
            "bp_end",
            "x_A",
            "y_A",
            "z_A",
            "lattice_x",
            "lattice_y",
            "lattice_z",
        ],
    )


def write_cds_mapping(path: Path, cds_rows: list[dict], coords: np.ndarray, lattice: np.ndarray, bead_count: int) -> None:
    rows = []
    for mol_i in range(coords.shape[0] // bead_count):
        offset = mol_i * bead_count
        for row in cds_rows:
            tx_bead = int(row["tx_start_bead"])
            bead_flat = offset + tx_bead - 1
            site = lattice[bead_flat]
            coord = coords[bead_flat]
            mapped = row.copy()
            mapped.update(
                {
                    "molecule_id": mol_i + 1,
                    "tx_start_x_A": f"{coord[0]:.8f}",
                    "tx_start_y_A": f"{coord[1]:.8f}",
                    "tx_start_z_A": f"{coord[2]:.8f}",
                    "tx_start_lattice_x": int(site[0]),
                    "tx_start_lattice_y": int(site[1]),
                    "tx_start_lattice_z": int(site[2]),
                }
            )
            rows.append(mapped)
    fields = [
        "virus",
        "accession",
        "molecule_id",
        "cds_index",
        "locus_tag",
        "gene",
        "product",
        "location",
        "strand",
        "start_bp",
        "end_bp",
        "tx_start_bp",
        "tx_end_bp",
        "start_bead",
        "end_bead",
        "tx_start_bead",
        "tx_end_bead",
        "tx_start_x_A",
        "tx_start_y_A",
        "tx_start_z_A",
        "tx_start_lattice_x",
        "tx_start_lattice_y",
        "tx_start_lattice_z",
    ]
    write_csv(path, rows, fields)


def write_rdme_init_json(path: Path, meta: GenomeMeta, bead_count: int, initial_genomes: int) -> None:
    payload = {
        "species": {
            "V_P1_linear_DNA_bead": {
                "count": bead_count * initial_genomes,
                "bp_per_bead": BP_PER_BEAD,
                "physical_particle": True,
            },
            "V_P1_Genome": {
                "count": initial_genomes,
                "semantic_role": "whole linear genome molecule label; not a spatial proxy particle",
            },
        },
        "topology": {
            "genome_accession": meta.accession,
            "genome_topology": meta.topology,
            "molecule_type": meta.molecule_type,
            "beads_per_genome": bead_count,
            "bonds_per_genome": bead_count - 1,
            "angles_per_genome": bead_count - 2,
            "terminal_closure_bond": False,
            "copy_number_source": "explicit initialized molecules only",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build(args: argparse.Namespace) -> dict:
    if args.initialGenomes < 1:
        raise ValueError("--initialGenomes must be >= 1")

    meta = load_genome_meta(args.genbank)
    cds_rows = load_p1_cds(args.cdsTable, meta.accession)
    bead_count = int(math.ceil(meta.length_bp / BP_PER_BEAD))

    run_dir = args.outRoot / args.runId
    run_dir.mkdir(parents=True, exist_ok=True)

    cytoplasm = np.load(args.maskDir / "cytoplasm.npy")
    membrane = np.load(args.maskDir / "membrane.npy")
    coords_by_molecule = place_initial_genomes(
        initial_genomes=args.initialGenomes,
        bead_count=bead_count,
        seed=args.seed,
        cytoplasm=cytoplasm,
        membrane=membrane,
        start_radius_a=args.cellRadiusAngstrom,
        min_nonbonded_a=args.minNonbondedDistanceAngstrom,
        placement_attempts=args.placementAttempts,
        chain_trial_limit=args.chainTrialLimit,
    )
    atoms, bonds, angles = build_lammps_topology(coords_by_molecule)
    assert_linear_topology(atoms, bonds, bead_count, args.initialGenomes)

    data_file = run_dir / "p1_linear_dna.data"
    input_file = run_dir / "in.p1_linear_dna_lammps"
    write_lammps_data(data_file, atoms, bonds, angles, box_a=max(args.cellRadiusAngstrom * 1.35, 2500.0))
    write_lammps_input(input_file, data_file, args, run_dir)

    coords_flat = coords_by_molecule.reshape(args.initialGenomes * bead_count, 3)
    lammps_status = "not_run"
    if args.runLammps:
        coords_flat = run_lammps(args, input_file, run_dir, coords_flat.shape[0])
        lammps_status = "completed"

    lattice = coords_to_lattice(coords_flat, tuple(int(v) for v in cytoplasm.shape))
    assert_inside_rdme_cytoplasm(lattice, cytoplasm, membrane)

    occupancy = np.zeros(cytoplasm.shape, dtype=np.uint8)
    occupancy[lattice[:, 0], lattice[:, 1], lattice[:, 2]] = 1
    np.save(run_dir / "p1_linear_dna_rdme_lattice_occupancy.npy", occupancy)
    np.savez_compressed(
        run_dir / "p1_linear_dna_topology_state.npz",
        coords_A=coords_flat,
        lattice_xyz=lattice,
        bead_count=np.array([bead_count], dtype=np.int64),
        initial_genomes=np.array([args.initialGenomes], dtype=np.int64),
        bp_per_bead=np.array([BP_PER_BEAD], dtype=np.int64),
    )
    write_topology_particles(run_dir / "p1_linear_dna_particles.csv", coords_flat, lattice, bead_count)
    write_cds_mapping(run_dir / "p1_cds_to_linear_dna_beads.csv", cds_rows, coords_flat, lattice, bead_count)
    write_rdme_init_json(run_dir / "p1_rdme_initial_particles.json", meta, bead_count, args.initialGenomes)

    manifest = {
        "run_id": args.runId,
        "standard": "Cell-style RDME lattice plus explicit DNA topology adapter",
        "proxy_forbidden": True,
        "counts_mapping_forbidden": True,
        "guide_filament_forbidden": True,
        "sc_chain_generation_used": False,
        "btree_circular_replication_used": False,
        "reason": "P1 GenBank topology is linear; circular Syn3A tools are not accepted for P1 substrate generation.",
        "genome": meta.__dict__,
        "initial_genomes": args.initialGenomes,
        "bp_per_bead": BP_PER_BEAD,
        "beads_per_genome": bead_count,
        "placement_attempts": args.placementAttempts,
        "chain_trial_limit": args.chainTrialLimit,
        "atoms": len(atoms),
        "bonds": len(bonds),
        "angles": len(angles),
        "terminal_closure_bond": False,
        "rdme_lattice_spacing_nm": 10,
        "rdme_lattice_shape": list(cytoplasm.shape),
        "lammps_status": lammps_status,
        "outputs": {
            "lammps_data": str(data_file),
            "lammps_input": str(input_file),
            "topology_state_npz": str(run_dir / "p1_linear_dna_topology_state.npz"),
            "rdme_lattice_occupancy": str(run_dir / "p1_linear_dna_rdme_lattice_occupancy.npy"),
            "particles_csv": str(run_dir / "p1_linear_dna_particles.csv"),
            "cds_mapping_csv": str(run_dir / "p1_cds_to_linear_dna_beads.csv"),
            "rdme_initial_particles_json": str(run_dir / "p1_rdme_initial_particles.json"),
        },
        "next_required_gate": (
            "Inject these topology particles into the RDME lattice/hook before enabling "
            "P1 replication, packaging, and membrane breach readouts. Old r11 cannot be resumed."
        ),
    }
    manifest_path = run_dir / "p1_linear_topology_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not args.prepareOnly:
        raise SystemExit(
            "Topology substrate prepared only. Full RDME reaction/packaging is intentionally blocked "
            "until this manifest is wired into the RDME hook and smoke-tested from t=0."
        )


if __name__ == "__main__":
    main()
