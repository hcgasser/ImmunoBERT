import numpy as np
from tqdm import tqdm

from pytorch_lightning.utilities.seed import seed_everything

import pMHC
from .mhc_allele import MhcAllele
from .protein import Protein
from .example import Sample, Peptide, Observation, Decoy
from .datasources import Datasource
from .utils import pseudo_pos

from .split import load_split_mhc_alleles, load_split_proteins

from .view import create_views


def setup_system(proportion, decoys_per_obs):
    from_input()

    load_split_mhc_alleles()
    load_split_proteins()

    create_views(proportion)
    Decoy.from_input(proportion, decoys_per_obs)


def from_data():
    MhcAllele.from_data()
    Protein.from_data()

    for datasource in Datasource.datasources.values():
        datasource.from_data()

    for peptide in tqdm(list(Peptide.peptides.values()), "find flanks", disable=pMHC.TQDM_DISABLE):
        peptide.post_from_data()

    seed_everything(42)

    perm = "; ".join([str(x) for x in np.random.permutation(list(Observation.key_to_observation.keys()))])
    with open(pMHC.OBSERVATION_PERMUTATION_FILENAME, 'w') as file:
        file.write(perm)


def to_input():
    MhcAllele.to_input()
    Protein.to_input()
    Sample.to_input()
    Peptide.to_input()
    Observation.to_input()


def from_input():
    MhcAllele.from_input()
    Protein.from_input()
    Sample.from_input()
    Peptide.from_input()
    Observation.from_input()

