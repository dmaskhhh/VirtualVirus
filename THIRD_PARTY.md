# Attribution and license scope

The root MIT license applies to project-authored software in `src/`, `analysis/` and `tests/` and to project-authored documentation. It does not assert ownership of external host software, third-party databases or their derived content. Contributor attribution does not imply an institutional copyright transfer.

## Host model

The molecular extensions use [Luthey-Schulten-Lab/Minimal_Cell_4DWCM](https://github.com/Luthey-Schulten-Lab/Minimal_Cell_4DWCM). The inspected host checkout is commit `08f5f2080dc5d766448b02e20c74a6d04c737f94`. It is an external dependency; obtain the host and follow its installation and citation instructions separately. Complete host modules (including Hook, MCRDME, CME and reaction modules) are not redistributed here. No root license file was found at the inspected host commit; this repository does not grant rights to that external code.

Initial cell masks and the spatial geometry are numerical inputs derived from the upstream computational framework, retained for reproducibility with that attribution. This project's software license does not override any rights in the underlying host material. Project-generated trajectories and parameter tables are supplied as computational research records; please retain their provenance when reusing them. No separate exclusive rights are claimed over numerical facts.

## Sequence annotations

P1 and L1 annotation/sequence-accounting inputs refer to NCBI RefSeq accessions [NC_002515.1](https://www.ncbi.nlm.nih.gov/nuccore/NC_002515.1) and [NC_001341.1](https://www.ncbi.nlm.nih.gov/nuccore/NC_001341.1), respectively. P1 here denotes the Mycoplasma phage accession, not an interchangeable model of every phage called P1. Preserve accession and source attribution; original database records and their source annotations are not covered by the software MIT grant.

## Runtime dependencies

NumPy, SciPy, Numba, llvmlite and optional molecular-runtime packages retain their respective licenses. They are installed separately rather than bundled. See the dependency projects for their notices. The molecular runtime additionally requires the upstream Lattice Microbes/CUDA and host-model environment; the spatial requirements file is not an installer for that environment.
