# VirtualVirus

Research code accompanying *Constructing Virtual Viruses in A Spatial Minimal Cell*.

VirtualVirus connects annotation-defined viral molecular programs in a JCVI-syn3A-based spatial model to finite-size particle assembly and exit. This compact release includes the P1/L1 molecular extensions, selected reference tables, and a runnable P1 spatial continuation from archived placement events and geometry.

This project concerns computational biology and visualization only. It includes no wet-laboratory procedures or human-related research.

## Quick start

Use Python 3.11 on Linux or WSL. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
python analysis/check_results.py
python -m unittest discover -s tests -v
python src/spatial/run_model.py --radius 3 --seed 0
python src/spatial/exit_readouts.py
python analysis/check_results.py --generated
```

The main example computes the 1236–1480 s spatial continuation, writing `outputs/r3_s0/`. It uses 11 recorded placement inputs, a dynamic lattice boundary and finite-size rigid particles. The reference main condition has 11 complete-body exits and an opening time of 1301 s. The readout replay checks saved states before distinguishing anchor crossing from complete-body exit. Installation time is additional to execution time.

## Contents

| Path | Purpose |
|---|---|
| `src/spatial/` | Particle motion, membrane updates and event-resolved exit readouts |
| `src/molecular/` | P1/L1 extensions for the separately installed upstream host model |
| `inputs/spatial/` | Self-contained geometry, placement inputs and archived configuration |
| `inputs/reference/` | Selected P1/L1 molecular records and all nine reported spatial-condition summaries |
| `inputs/molecular/` | Small inputs used by the molecular extensions |
| `analysis/check_results.py` | Numerical checks and compact summaries from the included tables |
| `docs/reproduction.md` | Parameters, evidence map and reproduction scope |
| `docs/molecular.md` | External host dependencies and molecular entry points |

## Scope

The spatial example is a one-way continuation of fixed upstream records; it does not feed particle behavior back into molecular production. Nine spatial conditions share the same upstream P1 record, and L1 is a separate historical trajectory, not a matched control. These are descriptive computational cases, not biological validation of infection, host compatibility, infectivity or lysis. Quantitative upstream production remains provisional because count synchronization and template handling require correction and a new production run. The spatial opening and mobility rules are prescribed model assumptions; particle motion is translation-only and excludes host molecular crowding.

The release supports rerunning the spatial example and checking included reference results. It does not claim exact regeneration of the full historical upstream trajectory or provide a complete figure/video production workflow. Molecular extensions require the external host software and its own dependencies; see [molecular setup](docs/molecular.md).

## Citation and reuse

Please cite the software using `CITATION.cff` and cite the associated manuscript when its publication record becomes available. No manuscript DOI is assigned here. Cite the upstream model when using it as well.

Project-authored software is available under the [MIT license](LICENSE). The license permits reuse, modification and redistribution, including commercial reuse, while retaining the license notice. External code, database records and upstream-derived inputs are not relicensed by this repository; see [THIRD_PARTY.md](THIRD_PARTY.md).
