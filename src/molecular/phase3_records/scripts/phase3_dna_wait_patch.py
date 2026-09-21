from __future__ import annotations

import os
import time as timepy
from dataclasses import dataclass


@dataclass(frozen=True)
class Phase3DnaWaitConfig:
    first_wait_seconds: int = 900
    fallback_wait_seconds: int = 300
    poll_seconds: int = 10


def install_phase3_dna_wait_patch(dna_module, config: Phase3DnaWaitConfig) -> None:
    original_rescue = dna_module.rescueDNA

    def phase3_check_last_chromosome(sim_properties):
        print("Checking for configuration from previous DNA step")
        work_dir = sim_properties["working_directory"] + "DNA/"
        dna_file = work_dir + "dna_monomers_{:d}.bin".format(sim_properties["last_DNA_step"])
        print(dna_file)

        last_dna_complete = os.path.isfile(dna_file)
        if not last_dna_complete:
            print("Waiting on BRGDNA to complete configuration")

        last_last = sim_properties.get("last_last_DNA_step")
        wait_limit = config.first_wait_seconds if last_last is None else config.fallback_wait_seconds
        dna_wait = 0
        while not last_dna_complete:
            if dna_wait >= wait_limit:
                if last_last is None:
                    raise RuntimeError(
                        "Phase3 DNA wait exceeded first-step limit without a rescue source: "
                        f"{dna_file}; waited {dna_wait}s"
                    )
                original_rescue(sim_properties)
                print("Created Rescue Files")
                return None

            last_dna_complete = os.path.isfile(dna_file)
            if last_dna_complete:
                break
            timepy.sleep(config.poll_seconds)
            dna_wait += config.poll_seconds

        print("Waited seconds: " + str(dna_wait))
        print("Chromosome configuration ready to load")
        return None

    dna_module.checkLastChromosome = phase3_check_last_chromosome
