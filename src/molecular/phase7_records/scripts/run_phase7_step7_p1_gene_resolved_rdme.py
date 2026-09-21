#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
INPUT_ROOT = REPO_ROOT / "inputs" / "molecular"
ROOT = Path(os.environ.get("MC4D_ROOT", str(REPO_ROOT / "external" / "Minimal_Cell_4DWCM"))).expanduser().resolve()
PHASE3_SCRIPTS = SCRIPT_DIR.parents[1] / "phase3_records" / "scripts"
PHASE2 = INPUT_ROOT / "phase2_records"
PHASE7 = Path(os.environ.get("VV_MOLECULAR_OUTPUT_ROOT", str(REPO_ROOT / "outputs" / "molecular"))).expanduser().resolve() / "phase7_records"
CONDA_BIN_TEXT = os.environ.get("PHASE7_CONDA_BIN", "")
if CONDA_BIN_TEXT:
    os.environ["PATH"] = CONDA_BIN_TEXT + os.pathsep + os.environ.get("PATH", "")
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + str(PHASE3_SCRIPTS) + os.pathsep + os.environ.get("PYTHONPATH", "")
for path in [ROOT, SCRIPT_DIR, PHASE3_SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MASK_DIR = INPUT_ROOT / "geometry"
CYTOPLASM_MASK = MASK_DIR / "cytoplasm.npy"
MEMBRANE_MASK = MASK_DIR / "membrane.npy"
EXTRACELLULAR_MASK = MASK_DIR / "extracellular.npy"
CDS_TABLE = PHASE2 / "sequence_accounting" / "cds_table.csv"
SUMMARY_TABLE = PHASE2 / "sequence_accounting" / "sequence_accounting_summary.csv"
PARAMETERS = PHASE2 / "parameters" / "P1_parameters.csv"

LIVE_ROOT = PHASE7 / "live_runs" / "step7_p1_gene_resolved"
CONFIG_DIR = LIVE_ROOT / "configs"
REPORTS = PHASE7 / "reports"
INPUT_FREEZE = PHASE7 / "input_freeze"
GENERATED = PHASE7 / "generated"

AA_NAMES = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_parameter_table(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["parameter"]] = row["value"]
    return out


def read_p1_cds_table(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["virus"] == "P1-like":
                rows.append(row)
    rows.sort(key=lambda r: int(r["cds_index"]))
    if len(rows) != 11:
        raise ValueError(f"Phase7 P1 requires 11 CDS from cds_table.csv, got {len(rows)}")
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runId", required=True)
    ap.add_argument("--duration", type=int, default=1800)
    ap.add_argument("--initialGenomes", type=int, default=5)
    ap.add_argument("--placementSeed", type=int, default=70701)
    ap.add_argument("--damageThreshold", type=int, default=260)
    ap.add_argument("--damageVirionStart", type=int, default=650)
    ap.add_argument("--releasePerSecond", type=int, default=30)
    ap.add_argument("--postBreachSeconds", type=int, default=60)
    ap.add_argument("--dnaIntervals", default="32,16,8,4")
    ap.add_argument("--maxConsecutiveRescues", type=int, default=3)
    ap.add_argument("-cd", "--cudaDevices", type=int, default=0)
    ap.add_argument("-drs", "--dnaRngSeed", type=int, default=42)
    ap.add_argument("-dsd", "--dnaSoftwareDirectory", default=os.environ.get("PHASE7_DNA_SOFTWARE_DIR", ""))
    ap.add_argument("-m", "--membrane", type=int, default=1)
    ap.add_argument("-mh", "--maximumHours", type=float, default=48.0)
    ap.add_argument("--firstDnaWaitSeconds", type=int, default=900)
    ap.add_argument("--hookStep", type=int, default=1000)
    ap.add_argument("--writeStep", type=int, default=80000)
    ap.add_argument("--prepareOnly", action="store_true")
    return ap.parse_args()


def species_names(cds_rows: list[dict[str, str]]) -> list[str]:
    names = ["V_P1_Genome", "V_P1_virion", "V_P1_membrane_damage", "V_P1_breach", "V_P1_released", "V_P1_rep_event", "V_P1_assembly_event"]
    for row in cds_rows:
        locus = row["locus_tag"]
        names.extend([
            f"V_P1_G_{locus}",
            f"V_P1_TX_{locus}",
            f"V_P1_R_{locus}",
            f"V_P1_RB_{locus}",
            f"V_P1_P_{locus}",
            f"V_P1_TX_{locus}_event",
            f"V_P1_TL_{locus}_event",
        ])
    return names


def species_diffusion(name: str) -> float:
    if name in {"V_P1_membrane_damage", "V_P1_breach", "V_P1_rep_event", "V_P1_assembly_event"} or name.endswith("_event"):
        return 0.0
    if name == "V_P1_Genome" or name.startswith("V_P1_G_") or name.startswith("V_P1_TX_"):
        return 5e-14
    if name.startswith("V_P1_RB_"):
        return 5e-13
    if name.startswith("V_P1_R_") or name.startswith("V_P1_P_"):
        return 1e-12
    return 1e-13


def make_initial_particles(cds_rows: list[dict[str, str]], initial_genomes: int, placement_seed: int) -> list[dict]:
    if not 1 <= initial_genomes <= 10:
        raise ValueError("--initialGenomes must be between 1 and 10 for Phase7 low-MOI initialization")
    cytoplasm = np.load(CYTOPLASM_MASK)
    coords = np.argwhere(cytoplasm)
    rng = np.random.default_rng(placement_seed)
    rng.shuffle(coords)
    rows: list[dict] = []
    cursor = 0
    particle_no = 1
    for genome_i in range(initial_genomes):
        x, y, z = (int(v) for v in coords[cursor])
        cursor += 1
        rows.append({"rdme_species": "V_P1_Genome", "x": x, "y": y, "z": z, "particle_no": particle_no, "source": f"phase7_initial_genome_{genome_i+1}"})
        particle_no += 1
        for cds in cds_rows:
            x, y, z = (int(v) for v in coords[cursor])
            cursor += 1
            rows.append({"rdme_species": f"V_P1_G_{cds['locus_tag']}", "x": x, "y": y, "z": z, "particle_no": particle_no, "source": f"phase7_initial_gene_template_{genome_i+1}"})
            particle_no += 1
    return rows


def freeze_config(args, params: dict[str, str], cds_rows: list[dict[str, str]], particles: list[dict]) -> dict:
    for directory in [INPUT_FREEZE, CONFIG_DIR, REPORTS, GENERATED]:
        directory.mkdir(parents=True, exist_ok=True)
    inputs = [CYTOPLASM_MASK, MEMBRANE_MASK, EXTRACELLULAR_MASK, CDS_TABLE, SUMMARY_TABLE, PARAMETERS, Path(__file__).resolve()]
    manifest = [{"run_id": args.runId, "source_path": str(src), "size_bytes": src.stat().st_size, "sha256": sha256(src)} for src in inputs]
    write_csv(INPUT_FREEZE / f"{args.runId}_phase7_input_manifest.csv", manifest, ["run_id", "source_path", "size_bytes", "sha256"])
    write_csv(CONFIG_DIR / f"{args.runId}_initial_particles.csv", particles, ["rdme_species", "x", "y", "z", "particle_no", "source"])
    write_csv(GENERATED / f"{args.runId}_p1_cds_table.csv", cds_rows, list(cds_rows[0].keys()))
    intervals = [int(v) for v in str(args.dnaIntervals).split(",")]
    if intervals != [32, 16, 8, 4]:
        raise ValueError("Phase7 P1 is locked to adaptive DNA intervals 32,16,8,4")
    config = {
        "run_id": args.runId,
        "phase": "Phase7",
        "scope": "Gene-resolved P1-in-Syn3A RDME/WCM: 11 P1 CDS, independent mRNA/protein/RNAP/ribosome-bound states, no target trajectory forcing.",
        "non_compromise_checks": {
            "step5c_trajectory_forcing": False,
            "single_lumped_p1_mrna": False,
            "single_lumped_p1_protein": False,
            "rnapi_occupancy": "RNAP is consumed into V_P1_TX_locus and released on transcript completion.",
            "ribosome_occupancy": "ribosomeP is consumed into V_P1_RB_locus and released on protein completion.",
            "assembly_rule": "Virions are assembled only by consuming one genome and one copy of every non-polymerase P1 protein species.",
        },
        "duration_s": int(args.duration),
        "initial_genomes": int(args.initialGenomes),
        "cds_count": len(cds_rows),
        "dna_update_intervals_s_biological": intervals,
        "damage_threshold": int(args.damageThreshold),
        "damage_virion_start": int(args.damageVirionStart),
        "release_per_second": int(args.releasePerSecond),
        "post_breach_seconds": int(args.postBreachSeconds),
        "max_consecutive_rescues": int(args.maxConsecutiveRescues),
        "p1_parameters": params,
        "p1_cds": cds_rows,
        "inputs": manifest,
    }
    (CONFIG_DIR / f"{args.runId}_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def install_species(sim, region_dict: dict, particles: list[dict], cds_rows: list[dict[str, str]]) -> None:
    mobile_regions = ["cytoplasm", "outer_cytoplasm", "DNA", "ribosomes", "ribo_centers"]
    all_regions = mobile_regions + ["membrane", "extracellular"]
    handles = {}
    for name in species_names(cds_rows):
        sp = sim.species(name)
        handles[name] = sp
        diff = species_diffusion(name)
        dc = sim.diffusionConst(f"{name}_diff", diff)
        regions = all_regions if diff == 0.0 or name == "V_P1_released" else mobile_regions
        if name == "V_P1_membrane_damage" or name == "V_P1_breach":
            regions = ["membrane"]
        for r1 in regions:
            for r2 in regions:
                if r1 in region_dict and r2 in region_dict:
                    sim.transitionRate(sp, sim.region(r1), sim.region(r2), dc)
    for row in particles:
        handles[row["rdme_species"]].placeParticle(int(row["x"]), int(row["y"]), int(row["z"]), 1)


def add_p1_gene_resolved_reactions(sim, params: dict[str, str], cds_rows: list[dict[str, str]], region_dict: dict) -> None:
    rnap = sim.species("RNAP")
    ribosome = sim.species("ribosomeP")
    genome = sim.species("V_P1_Genome")
    pol = sim.species("V_P1_P_MpP1p01")
    rep_event = sim.species("V_P1_rep_event")
    replication = sim.rateConst("P1_gene_resolved_replication", 1.0e6 * float(params["replication_rate_s"]), 2)
    tx_bind = sim.rateConst("P1_gene_resolved_tx_bind", 1.0e6 * float(params["transcription_rate_s"]), 2)
    tl_bind = sim.rateConst("P1_gene_resolved_tl_bind", 1.0e6 * float(params["translation_rate_s"]), 2)
    mrna_decay = sim.rateConst("P1_gene_resolved_mrna_decay", 0.006, 1)
    rep_regions = ["cytoplasm", "outer_cytoplasm", "DNA", "ribosomes", "ribo_centers"]
    for region in rep_regions:
        r = sim.region(region)
        r.addReaction([genome, pol], [genome, genome, pol, rep_event], replication)
        for row in cds_rows:
            locus = row["locus_tag"]
            gene = sim.species(f"V_P1_G_{locus}")
            tx = sim.species(f"V_P1_TX_{locus}")
            mrna = sim.species(f"V_P1_R_{locus}")
            rb = sim.species(f"V_P1_RB_{locus}")
            protein = sim.species(f"V_P1_P_{locus}")
            tx_event = sim.species(f"V_P1_TX_{locus}_event")
            tl_event = sim.species(f"V_P1_TL_{locus}_event")
            tx_done = sim.rateConst(f"P1_TX_done_{locus}", max(0.005, 45.0 / max(1.0, float(row["nt_length"]))), 1)
            tl_done = sim.rateConst(f"P1_TL_done_{locus}", max(0.005, 15.0 / max(1.0, float(row["aa_length"]))), 1)
            r.addReaction([gene, rnap], [tx], tx_bind)
            r.addReaction([tx], [gene, rnap, mrna, tx_event], tx_done)
            r.addReaction([mrna, ribosome], [rb], tl_bind)
            r.addReaction([rb], [mrna, ribosome, protein, tl_event], tl_done)
            r.addReaction([mrna], [], mrna_decay)


def make_solver(base_cls, cds_rows: list[dict[str, str]]):
    import Communicate as communicate  # noqa: F401

    structural_loci = [row["locus_tag"] for row in cds_rows if row["locus_tag"] != "MpP1p01"]
    cds_by_locus = {row["locus_tag"]: row for row in cds_rows}

    class Phase7P1GeneResolvedSolver(base_cls):
        def __init__(self, lmFile, sim_properties, region_dict, ribo_site_dict, termination_time=None):
            super().__init__(lmFile, sim_properties, region_dict, ribo_site_dict, termination_time=termination_time)
            self.config = sim_properties["phase7_config"]
            self.cytoplasm_coords = np.argwhere(np.load(CYTOPLASM_MASK))
            self.membrane_coords = np.argwhere(np.load(MEMBRANE_MASK))
            self.extra_coords = np.argwhere(np.load(EXTRACELLULAR_MASK))
            rng = np.random.default_rng(int(sim_properties["phase7_placement_seed"]) + 47)
            rng.shuffle(self.cytoplasm_coords)
            rng.shuffle(self.membrane_coords)
            rng.shuffle(self.extra_coords)
            self.rng = rng
            self.coord_cursor = {"cyto": 0, "membrane": 0, "extra": 0}
            self.damage_total = 0
            self.breach_time = None
            self.released_total = 0
            self.last_virion_count = 0
            self.cadence_events = []
            self.gene_event_log = []
            self.assembly_log = []

        def _count_species(self, plattice, name: str) -> int:
            idx = int(self.sim_properties["name_to_index"][name])
            return int(np.count_nonzero(plattice == idx))

        def _place_one(self, plattice, species_name: str, region: str) -> bool:
            coords = {"cyto": self.cytoplasm_coords, "membrane": self.membrane_coords, "extra": self.extra_coords}[region]
            idx = int(self.sim_properties["name_to_index"][species_name])
            n = len(coords)
            for _ in range(n):
                cursor = self.coord_cursor[region]
                x, y, z = (int(v) for v in coords[cursor])
                self.coord_cursor[region] = (cursor + 1) % n
                slots = plattice[:, x, y, z, 0]
                empty = np.where(slots == 0)[0]
                if len(empty):
                    plattice[int(empty[0]), x, y, z, 0] = np.uint32(idx)
                    return True
            return False

        def _remove_one(self, plattice, species_name: str) -> bool:
            idx = int(self.sim_properties["name_to_index"][species_name])
            locs = np.argwhere(plattice == idx)
            if len(locs) == 0:
                return False
            slot, x, y, z, p = (int(v) for v in locs[0])
            plattice[slot, x, y, z, p] = np.uint32(0)
            return True

        def _add_count(self, key: str, amount: int, cap: int = 5000) -> None:
            counts = self.sim_properties["counts"]
            if key in counts:
                counts[key] += int(min(max(amount, 0), cap))

        def _dna_interval(self, virion: int) -> int:
            if self.damage_total > 0:
                return 4
            if virion >= 200:
                return 8
            if virion >= 50:
                return 16
            return 32

        def _apply_replication_templates(self, plattice, rep_events: int) -> None:
            for _ in range(rep_events):
                for row in cds_rows:
                    self._place_one(plattice, f"V_P1_G_{row['locus_tag']}", "cyto")

        def _apply_gene_event_costs(self, live_t: int, plattice) -> None:
            params = self.sim_properties["phase7_p1_parameters"]
            rep = self._count_species(plattice, "V_P1_rep_event")
            asm = self._count_species(plattice, "V_P1_assembly_event")
            if rep:
                self._add_count("dATP_DNArep_cost", rep * int(float(params["genome_A_count"])))
                self._add_count("dCTP_DNArep_cost", rep * int(float(params["genome_C_count"])))
                self._add_count("dGTP_DNArep_cost", rep * int(float(params["genome_G_count"])))
                self._add_count("dTTP_DNArep_cost", rep * int(float(params["genome_T_count"])))
                self._apply_replication_templates(plattice, rep)
            if asm:
                self._add_count("ATP_trsc_cost_second", asm)

            tx_total = 0
            tl_total = 0
            event_row = {"time_s": live_t, "rep_event": rep, "assembly_event": asm}
            for row in cds_rows:
                locus = row["locus_tag"]
                tx_name = f"V_P1_TX_{locus}_event"
                tl_name = f"V_P1_TL_{locus}_event"
                tx = self._count_species(plattice, tx_name)
                tl = self._count_species(plattice, tl_name)
                event_row[f"TX_{locus}"] = tx
                event_row[f"TL_{locus}"] = tl
                tx_total += tx
                tl_total += tl
                if tx:
                    self._add_count("ATP_trsc_cost", tx)
                    self._add_count("ATP_mRNA_cost", tx * int(row["nt_A"]))
                    self._add_count("CTP_mRNA_cost", tx * int(row["nt_C"]))
                    self._add_count("GTP_mRNA_cost", tx * int(row["nt_G"]))
                    self._add_count("UTP_mRNA_cost", tx * int(row["nt_T"]))
                if tl:
                    self._add_count("GTP_translat_cost", tl * 2 * int(row["aa_length"]))
                    for aa in AA_NAMES:
                        self._add_count(f"{aa}_cost", tl * int(row.get(f"aa_{aa}", 0)))
            event_row["tx_total"] = tx_total
            event_row["tl_total"] = tl_total
            if rep or asm or tx_total or tl_total:
                self.gene_event_log.append(event_row)
                self.sim_properties["phase7_gene_event_log"] = self.gene_event_log

            for name in ["V_P1_rep_event", "V_P1_assembly_event"]:
                for _ in range(self._count_species(plattice, name)):
                    self._remove_one(plattice, name)
            for row in cds_rows:
                for suffix in ["TX", "TL"]:
                    name = f"V_P1_{suffix}_{row['locus_tag']}_event"
                    for _ in range(self._count_species(plattice, name)):
                        self._remove_one(plattice, name)

        def _assemble_virions(self, live_t: int, plattice) -> None:
            genome = self._count_species(plattice, "V_P1_Genome")
            available_sets = genome
            for locus in structural_loci:
                available_sets = min(available_sets, self._count_species(plattice, f"V_P1_P_{locus}"))
            if available_sets <= 0:
                return
            rate = float(self.sim_properties["phase7_p1_parameters"]["assembly_rate_s"])
            n_to_assemble = int(self.rng.binomial(int(available_sets), min(1.0, rate)))
            if n_to_assemble <= 0 and available_sets >= 50:
                n_to_assemble = 1
            made = 0
            for _ in range(n_to_assemble):
                if not self._remove_one(plattice, "V_P1_Genome"):
                    break
                ok = True
                removed = []
                for locus in structural_loci:
                    species = f"V_P1_P_{locus}"
                    if self._remove_one(plattice, species):
                        removed.append(species)
                    else:
                        ok = False
                        break
                if not ok:
                    for species in removed:
                        self._place_one(plattice, species, "cyto")
                    self._place_one(plattice, "V_P1_Genome", "cyto")
                    break
                if self._place_one(plattice, "V_P1_virion", "cyto"):
                    self._place_one(plattice, "V_P1_assembly_event", "cyto")
                    made += 1
            if made:
                event = {"time_s": live_t, "assembled_virions": made, "available_sets": available_sets}
                self.assembly_log.append(event)
                self.sim_properties["phase7_assembly_log"] = self.assembly_log

        def _apply_membrane_release(self, live_t: int, plattice) -> None:
            virion = self._count_species(plattice, "V_P1_virion")
            start = int(self.config["damage_virion_start"])
            threshold = int(self.config["damage_threshold"])
            if virion > start:
                burden_increment = max(0, virion - max(self.last_virion_count, start))
                self.damage_total += max(1, burden_increment // 8)
            self.last_virion_count = virion
            current_damage = self._count_species(plattice, "V_P1_membrane_damage")
            for _ in range(max(0, self.damage_total - current_damage)):
                self._place_one(plattice, "V_P1_membrane_damage", "membrane")
            if self.damage_total >= threshold and self.breach_time is None:
                self.breach_time = live_t
                self._place_one(plattice, "V_P1_breach", "membrane")
            if self.breach_time is not None:
                release_target = (live_t - int(self.breach_time) + 1) * int(self.config["release_per_second"])
                while self.released_total < release_target and self._remove_one(plattice, "V_P1_virion"):
                    if self._place_one(plattice, "V_P1_released", "extra"):
                        self.released_total += 1
                    else:
                        break
            self.sim_properties["phase7_membrane_state"] = {
                "time_s": live_t,
                "virion": virion,
                "damage": self.damage_total,
                "breach_time": self.breach_time,
                "released_total": self.released_total,
            }

        def hookSimulation(self, t, lattice):
            live_t = int(round(float(t)))
            plattice = lattice.getParticleLatticeView()
            if abs(float(t) - live_t) < 1e-6:
                self._apply_gene_event_costs(live_t, plattice)
                self._assemble_virions(live_t, plattice)
                self._apply_membrane_release(live_t, plattice)

            prev_next_dna = float(self.next_DNA_time)
            result = super().hookSimulation(t, lattice)
            if float(t) >= prev_next_dna and float(t) > 0:
                virion = self._count_species(plattice, "V_P1_virion")
                interval = self._dna_interval(virion)
                self.next_DNA_time = float(t) + float(interval)
                event = {"time_s": float(t), "virion": virion, "damage": self.damage_total, "next_DNA_time": self.next_DNA_time, "interval_s": interval}
                self.cadence_events.append(event)
                self.sim_properties["phase7_dna_cadence_log"] = self.cadence_events
                print("Phase7 adaptive DNA interval:", event)
            return result

    return Phase7P1GeneResolvedSolver


def main() -> None:
    args = parse_args()
    run_dir = ROOT / "Data" / args.runId
    if run_dir.exists():
        raise SystemExit(f"Output directory already exists: {run_dir}")
    params = read_parameter_table(PARAMETERS)
    cds_rows = read_p1_cds_table(CDS_TABLE)
    particles = make_initial_particles(cds_rows, int(args.initialGenomes), int(args.placementSeed))
    config = freeze_config(args, params, cds_rows, particles)
    if args.prepareOnly:
        print(json.dumps({"run_id": args.runId, "status": "prepared", "phase": "Phase7", "cds_count": len(cds_rows), "config_dir": str(CONFIG_DIR)}, indent=2))
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
    import MC_RDME_initialization as MCRDME
    import RegionsAndComplexes as InitGeom
    import SpatialDnaDynamics as DNA
    from phase3_dna_wait_patch import Phase3DnaWaitConfig, install_phase3_dna_wait_patch

    install_phase3_dna_wait_patch(DNA, Phase3DnaWaitConfig(first_wait_seconds=int(args.firstDnaWaitSeconds)))
    sim, sim_properties = MCRDME.initSim(int(args.hookStep), int(args.writeStep), int(args.duration), args.runId, str(ROOT) + "/")
    save.saveSimArgs(sim_properties, args)
    sim_properties["phase7_config"] = config
    sim_properties["phase7_p1_parameters"] = params
    sim_properties["phase7_p1_cds"] = cds_rows
    sim_properties["phase7_placement_seed"] = int(args.placementSeed)
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
    add_p1_gene_resolved_reactions(sim, params, cds_rows, region_dict)
    MCRDME.createParticleIdxMap(sim, sim_properties)
    MCRDME.mapTranslationStates(sim_properties)
    IC.initializeRdmeCounts(sim, sim_properties)
    communicate.updateSA(sim_properties)
    save.saveCountsAndFluxes(0, sim_properties, None, None, None)

    import Hook
    Solver = makeSolver(IntMpdRdmeSolver, make_solver(Hook.MyOwnSolver, cds_rows))
    solver = Solver(sim, sim_properties, region_dict, ribo_site_dict, termination_time=float(args.maximumHours))
    sim.finalize()
    sim.run(solver=solver, cudaDevices=[int(args.cudaDevices)])
    (run_dir / "phase7_p1_gene_resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": args.runId, "status": "completed", "phase": "Phase7", "cds_count": len(cds_rows)}, indent=2))


if __name__ == "__main__":
    main()
