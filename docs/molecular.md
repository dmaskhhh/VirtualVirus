# Molecular extension source

The five scripts in `src/molecular/` preserve the current P1/L1 molecular extension logic, separated from the external whole-cell host. This is a source extension package with checked input preparation. It is not a validated reconstruction of the historical production trajectory.

## Contents and inputs

- `phase7_records/scripts/run_phase7_step7_p1_gene_resolved_rdme.py`: P1 gene-resolved base, also imported by r12.
- `phase7_records/scripts/run_phase7_step7_p1_r12_topology_rdme.py`: P1 topology, packaging and membrane-state extension.
- `phase7_records/scripts/run_phase7_step7_viral_gene_resolved_rdme.py`: generic P1/L1 entry point. Shared sequence tables retain only the P1/L1 records; retained row values are unchanged.
- `phase7_records/scripts/run_phase7_step7_p1_real_topology_amplification.py`: P1 topology substrate generation; Biopython required. Optional LAMMPS relaxation requires a separately installed compatible executable and DNA model.
- `phase3_records/scripts/phase3_dna_wait_patch.py`: DNA wait adapter used by the three molecular entry points.

`inputs/molecular/phase2_records/` contains CDS accounting, sequence summary, process parameters and the P1 reference GenBank record. Structural stoichiometry is stored at `inputs/molecular/stoichiometry/p1_structural_stoichiometry.tsv`; r12 uses this short portable path by default, or accepts `--stoichiometryRoot`. Preserve evidence-level/source/notes columns: the model rates are assumptions, not experimentally validated rates. P1 is RefSeq NC_002515.1; L1 is NC_001341.1. These archetypes do not establish natural infection of JCVI-syn3A.

`inputs/molecular/geometry/` contains three 64 x 64 x 128 region masks, derived from the historical host initialization `phase0_smoke_300s_seed42`. These inputs have no clinical or personal data. They are upstream-model-derived geometry and are distinct from original software covered by the repository code license.

`inputs/molecular/topology/r12_p1_linear_topology_g1/` is the historical **one-genome** fixture named by the archived P1 configuration: 1,166 beads. The r12 defaults select this fixture and `--initialGenomes 1`. A different genome count requires a matching topology fixture supplied with `--topologyRoot` and `--topologyRunId`. The original g1 manifest explicitly records `lammps_status: not_run`: this fixture is unrelaxed, and this release did not perform or validate relaxation. Its availability is not evidence that the full archived trajectory can be reproduced. Original absolute output paths were reduced to basenames. Files mentioned in its output inventory beyond the four distributed fixture files are not included.

## Preparation without the whole-cell host

Use Python 3.10+ and NumPy. Install Biopython as well to inspect/use the topology generator. Run from the repository root, for example:

```bash
python src/molecular/phase7_records/scripts/run_phase7_step7_p1_gene_resolved_rdme.py --runId inspect_p1 --prepareOnly
python src/molecular/phase7_records/scripts/run_phase7_step7_viral_gene_resolved_rdme.py --virus L1 --runId inspect_l1 --prepareOnly
python src/molecular/phase7_records/scripts/run_phase7_step7_p1_r12_topology_rdme.py --help
```

Input preparation writes configs and input manifests to `outputs/molecular/phase7_records/`. Override the parent output directory with `VV_MOLECULAR_OUTPUT_ROOT`. It does not execute the whole-cell host. The r12 `--prepareOnly` route performs a large topology-to-lattice projection and was not executed in this release; it is not a lightweight self-test.

## External host for full simulation

Obtain [Minimal_Cell_4DWCM](https://github.com/Luthey-Schulten-Lab/Minimal_Cell_4DWCM) separately and follow its dependency and input-data instructions. The inspected host commit was `08f5f2080dc5d766448b02e20c74a6d04c737f94`. No complete host module is redistributed here. Set `MC4D_ROOT` to that external checkout (or place it at `external/Minimal_Cell_4DWCM`). Its complete `input_data/`, LM/jLM CUDA environment, compatible BRGDNA installation and other upstream dependencies are still required; installing NumPy alone is insufficient.

Set `PHASE7_DNA_SOFTWARE_DIR` to the external BRGDNA directory (including the trailing directory separator required by upstream code). `--dnaSoftwareDirectory` overrides it. Optionally set `PHASE7_CONDA_BIN` to an existing environment bin directory. Host imports are resolved from `MC4D_ROOT`, while this repository supplies the phase3 adapter and molecular input tables.

**Full simulation writes host results under `$MC4D_ROOT/Data/<runId>`**, because the original upstream initialization contract is retained. Use a writable external checkout and unique run IDs; an existing run directory is rejected. Plugin configs/manifests remain under `VV_MOLECULAR_OUTPUT_ROOT` or repository `outputs/molecular`. The plugin does not modify upstream source code, but uses in-process solver inheritance and runtime function adapters.

For topology generation, `VV_LAMMPS_EXE` / `--lammpsExe` and `VV_DNA_MODEL_DIR` / `--dnaModelDir` identify separately installed tools. `--runLammps` explicitly enables relaxation. `--genbank`, `--maskDir`, `--cdsTable`, and `--outRoot` support alternate inputs/outputs. `VMD_EXE` / `--windowsVmdExe` controls optional VMD rendering. No personal machine path is embedded.

The generic L1 runner retains the existing descriptive model implementation. Inclusion of source does not validate L1 release, provide a P1-matched biological control, or add experimental mechanism evidence.

## Attribution and checks

The P1 GenBank record is the source record for [NC_002515.1](https://www.ncbi.nlm.nih.gov/nuccore/NC_002515.1). Sequence accounting preserves RefSeq accessions and annotation provenance. Original reference records, upstream-derived geometry and external software are not relicensed by the repository's license for original code. The local fixed upstream commit had no root LICENSE/COPYING file; obtain upstream terms independently, and do not infer a grant over its code from this package. No upstream source files are bundled.

Release checks: all five scripts compiled in memory; all four CLI entry points returned `--help` successfully; P1 and L1 `--prepareOnly` passed with 11 and 4 CDS respectively, without the host. AST comparison showed all numerical helper functions/classes unchanged; changes were path constants, CLI defaults and missing-dependency checks in entry-point setup. Full RDME/CME/ODE execution, GPU simulation, LAMMPS, r12 projection preparation and historical trajectory reproduction were **not** validated. The bundled geometry must also be checked for compatibility with any new host setup.
