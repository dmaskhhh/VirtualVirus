# Reproduction and evidence map

## Spatial example

`python src/spatial/run_model.py --radius 3 --seed 0` uses the included event schedule and geometry. `--radius` is the contact-neighborhood radius in lattice sites (10 nm/site); it is not the particle head radius. The reported radii are 2, 3 and 4, each with seeds 0, 1 and 2. Change these two arguments to generate another condition. `--inputs` and `--output` accept alternative directories. The readout tool expects generated condition directories under the default `outputs/` directory; `--all` processes all nine after they have been generated.

Each object retains a stable identity. Release occurs only when every occupied body site is outside the current cellular volume. The readout analysis compares first anchor crossing with complete-body exit on identical full-body dynamics; it is not a comparison with a point-particle motion model. The input schedule, initialization geometry, numerical rules and main random seed are retained from the archived calculation. Packaging/placement in the upstream record and acceptance into this continuation are distinct events.

The main example should reproduce `inputs/reference/p1_main_spatial_summary.json`, its 245-row timeseries, birth events and 11 object readouts. `analysis/check_results.py --generated` compares those outputs numerically. Tests cover boundary invariants, finite-body motion and readout definitions. Small platform-dependent floating-point differences may occur; numerical comparisons allow absolute tolerance 1e-9 and relative tolerance 1e-10. This is not a cross-platform validation claim.

## Reference results

| Included records | Interpretation |
|---|---|
| `p1_marker_timeseries.csv`, `p1_gene_*`, `p1_protein_balance.csv` | Archived P1 molecular record, 0–1480 s; expression and protein accounting |
| `p1_replication_events.json`, `p1_packaging_events.json` | Recorded upstream events, retained with their original field semantics |
| `l1_recorded_states.csv`, `l1_gene_*`, `l1_annotation.csv`, `l1_morphology.csv` | Separate L1 record, 1237 samples over 0–4944 s; expression, genome accumulation and morphology |
| `particle_exit_readouts.csv`, `exit_condition_summary.csv`, `spatial_condition_summary.json` | All nine reported spatial conditions; 99 object-condition readouts from the same 11 upstream inputs |
| `p1_main_*` | Main spatial condition r3_s0 |

Reference P1 run identifier: `r12ce_g1_2500s`. The identifier's requested duration is not its observed endpoint. L1 identifier: `p7_l1_continuous7200_genome3_dmg50_05161111`; its requested 7200 s is likewise not the saved observation window. L1 has 363 transcription and 858 translation events, genomes increasing from 3 to 19, and no recorded particle, breach or release output. Its packaging threshold is 50 genomes, above the observed maximum; absence of output is not evidence for a reconstructed productive non-lytic cycle.

The archived configuration JSON files document historical settings. They include descriptive labels inherited from those records; they are not a certification that each stated check was biologically validated or that the original full software environment is recoverable. The current inspected upstream commit is recorded in `THIRD_PARTY.md`, but the complete historical dependency set is not hash-bound. Private installation prefixes in archived metadata have been replaced with portable placeholders; numeric records are retained.

Detailed figure layouts, movies, large regenerable trajectories, internal reviews, exploratory branches and unintegrated repair candidates are outside this compact release. Necessary interpretation limits are summarized in the README. The two accession-defined phage programs are modeling archetypes, not demonstrated natural infections of JCVI-syn3A.
