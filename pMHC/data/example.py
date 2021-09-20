import pdb
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from collections import defaultdict

import torch

import pMHC
from pMHC import logger, POSITIVE_THRESHOLD, \
    FLANK_LEN, \
    VIEW_SA, VIEW_SAMA, VIEW_DECONV, \
    SPLIT_NAMES, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_VAL_PROTEINS, \
    SPLIT_TEST_PROTEINS, SPLIT_VAL_MHC_ALLELES, SPLIT_TEST_MHC_ALLELES, SPLITS
from pMHC.data.mhc_allele import MhcAllele
from pMHC.data.protein import Protein
from pMHC.data.utils import get_input_rep_PSEUDO, get_input_rep_FULL


class Sample:
    samples = {}

    def __init__(self, name):
        self.name = name
        self.mhc_alleles = []
        self.split_mhc = SPLIT_TRAIN

        self.observations = []
        Sample.samples.update({name: self})

        self._proteins = None
        self._hit_seqs = None

    def add_mhc_alleles(self, mhc_alleles):
        self.mhc_alleles += mhc_alleles

    def add_observation(self, observation):
        if observation not in self.observations:
            self.observations.append(observation)

    def get_proteins(self):
        if self._proteins is None:
            self._proteins = []
            for observation in self.observations:
                self._proteins += observation.peptide.proteins

            self._proteins = list(set(self._proteins))

        return self._proteins

    def get_hit_seqs(self):
        if self._hit_seqs is None:
            self._hit_seqs = []
            for observation in self.observations:
                if observation.target > POSITIVE_THRESHOLD:
                    self._hit_seqs += observation.peptide.seq

        return self._hit_seqs

    def set_split(self, split):
        if split != "":
            self.split = split

    def link_observation(self, observation):
        self.observations.append(observation)

    @staticmethod
    def get(name):
        if name in Sample.samples:
            return Sample.samples[name]
        else:
            return Sample(name)

    @staticmethod
    def find(name):
        if name in Sample.samples:
            return Sample.samples[name]
        return None

    @staticmethod
    def to_input():
        print(f"Sample.to_input")
        logger.info(f"Sample.to_input")

        names, mhc_allele_names = [], []
        for sample in Sample.samples.values():
            names.append(sample.name)
            mhc_allele_names.append(MhcAllele.mhc_alleles_to_names(sample.mhc_alleles))

        samples_df = pd.DataFrame({'name': names, 'mhc_allele_names': mhc_allele_names})
        samples_df.to_csv(pMHC.SAMPLE_INPUT_FILENAME, sep=",", index=False)

    @staticmethod
    def from_input():
        print(f"Sample.from_input")
        logger.info(f"Sample.from_input")

        samples = pd.read_csv(pMHC.SAMPLE_INPUT_FILENAME).fillna("")

        # insert "splits"
        for idx, (name, mhc_allele_names) in tqdm(samples.iterrows(), "Samples from input", disable=pMHC.TQDM_DISABLE):
            mhc_alleles = MhcAllele.names_to_mhc_alleles(mhc_allele_names.split("; "))
            sample = Sample(name)
            sample.add_mhc_alleles(mhc_alleles)


class Peptide:
    peptides = {}

    def __init__(self, seq):
        self.seq = seq
        self.n_flank = None
        self.c_flank = None
        self.proteins = []

        Peptide.peptides.update({seq: self})

    def add_proteins(self, proteins):
        self.proteins += proteins

    def post_from_data(self):
        self.proteins = [x for x in list(set(self.proteins)) if x is not None]

        if len(self.proteins) == 0:
            del Peptide.peptides[self.seq]
        else:
            exclude_context = False
            n_flank, c_flank = "", ""
            for protein in self.proteins:
                position = protein.seq.find(self.seq)

                current_n_flank, current_c_flank = None, None
                if position >= 0:
                    current_n_flank = protein.seq[max(position - FLANK_LEN, 0):position]
                    current_c_flank = protein.seq[(position + len(self.seq)
                                                   ):min(position + len(self.seq) + FLANK_LEN, len(protein.seq))]

                # do not consider the context of observations that map to multiple contexts
                if position < 0 or (
                        (n_flank != "" or c_flank != "") and
                        (n_flank != current_n_flank or c_flank != current_c_flank)):
                    exclude_context = True
                else:
                    n_flank = current_n_flank
                    c_flank = current_c_flank

            if exclude_context:
                n_flank, c_flank = "", ""

            self.n_flank = n_flank
            self.c_flank = c_flank

    @staticmethod
    def get(seq):
        if seq in Peptide.peptides:
            return Peptide.peptides[seq]
        else:
            return Peptide(seq)

    @staticmethod
    def to_input():
        print(f"Peptide.to_input")
        logger.info(f"Peptide.to_input")

        seqs, splits, n_flanks, c_flanks, protein_names = [], [], [], [], []
        for peptide in list(Peptide.peptides.values()):
            seqs.append(peptide.seq)
            n_flanks.append(peptide.n_flank)
            c_flanks.append(peptide.c_flank)
            protein_names.append("; ".join(list(set([protein.name for protein in peptide.proteins]))))

        peptides_df = pd.DataFrame({'seq': seqs, 'n_flank': n_flanks, 'c_flank': c_flanks,
                                    'protein_names': protein_names})
        peptides_df.to_csv(pMHC.PEPTIDE_INPUT_FILENAME, sep=",", index=False)

    @staticmethod
    def from_input():
        print(f"Peptide.from_input")
        logger.info(f"Peptide.from_input")

        peptides = pd.read_csv(pMHC.PEPTIDE_INPUT_FILENAME).fillna("")

        # insert "splits"
        for idx, (seq, n_flank, c_flank, protein_names) in tqdm(peptides.iterrows(), "Peptides from input", disable=pMHC.TQDM_DISABLE):
            peptide = Peptide.get(seq)
            peptide.n_flank = n_flank
            peptide.c_flank = c_flank
            peptide.proteins = Protein.names_to_proteins(protein_names.split("; "))


class Example:
    examples = []

    def __init__(self, key):
        self.key = key
        Example.examples.append(self)

    def get_tokenization(self, model):
        raise NotImplemented

    @property
    def n_flank(self):
        raise NotImplemented

    @property
    def peptide_seq(self):
        raise NotImplemented

    @property
    def c_flank(self):
        raise NotImplemented

    @property
    def mhc_allele(self):
        raise NotImplemented


class Observation(Example):
    observations = []
    key_to_observation = {}

    # obs_views[SPLIT_TRAIN | SPLIT_VAL | SPLIT_TEST][VIEW_SA | VIEW_SAMA | VIEW_DECONV]
    obs_views = [
        [[], [], []],   # train split
        [[], [], []],   # val split
        [[], [], []],   # test split
        [[], [], []],   # val-proteins
        [[], [], []],   # test-proteins
        [[], [], []],   # val-mhc_alleles
        [[], [], []]    # val-mhc_alleles
    ]

    def __init__(self, key, sample, peptide, target, datasource):
        super().__init__(key)

        self.sample = sample
        self.peptide = peptide
        self.target = target
        self.datasource = datasource

        self.split_protein = SPLIT_TRAIN
        self.deconvoluted_mhc_allele = None
        if len(sample.mhc_alleles) == 1:
            self.deconvoluted_mhc_allele = self.sample.mhc_alleles[0]

        Observation.observations.append(self)
        Observation.key_to_observation.update({self.key: self})
        sample.link_observation(self)

    def get_splits(self):
        sample = self.sample

        if sample.split_mhc != SPLIT_TRAIN:
            splits = [sample.split_mhc]
            if sample.split_mhc == SPLIT_TEST_MHC_ALLELES:
                splits.append(SPLIT_TEST)
            else:
                splits.append(SPLIT_VAL)
        else:
            splits = [self.split_protein]
            if self.split_protein == SPLIT_TEST_PROTEINS:
                splits.append(SPLIT_TEST)
            elif self.split_protein == SPLIT_VAL_PROTEINS:
                splits.append(SPLIT_VAL)

        return splits

    def get_example(self, decoy_idx):
        if decoy_idx < 0:
            return self
        else:
            return Decoy.get_decoy(self.key*1000 + decoy_idx)

    def get_tokenization(self, model):
        y, n_flank, peptide, c_flank = torch.tensor([self.target]), self.n_flank, self.peptide_seq, self.c_flank

        if model.input_mode == pMHC.INPUT_PEPTIDE:  # if context should not be input to model
            n_flank, c_flank = "", ""

        if model.mhc_rep == pMHC.MHC_FULL:
            sample = get_input_rep_FULL(n_flank, peptide, c_flank, self.deconvoluted_mhc_allele.seq, model)
        else:
            sample = get_input_rep_PSEUDO(n_flank, peptide, c_flank, self.deconvoluted_mhc_allele.pseudo_seq, model)

        sample.update({'targets': y})
        sample.update({'objects': (self, self.deconvoluted_mhc_allele, -1)})
        return sample

    @property
    def n_flank(self):
        return self.peptide.n_flank

    @property
    def peptide_seq(self):
        return self.peptide.seq

    @property
    def c_flank(self):
        return self.peptide.c_flank

    @property
    def mhc_allele(self):
        return self.deconvoluted_mhc_allele

    @staticmethod
    def to_input():
        print(f"Observation.to_input")
        logger.info(f"Observation.to_input")

        keys, sample_names, peptide_seqs, targets, datasources, deconv_mhc_allele_names = [], [], [], [], [], []
        for observation in Observation.observations:
            keys.append(observation.key)
            sample_names.append(observation.sample.name)
            peptide_seqs.append(observation.peptide.seq)
            targets.append(observation.target)
            datasources.append(observation.datasource)
            deconv_mhc_allele_names.append("")

        observations_df = pd.DataFrame({'key': keys, 'sample_name': sample_names,
                                        'peptide_seq': peptide_seqs, 'target': targets, 'datasource': datasources,
                                        'deconv_mhc_allele_name': deconv_mhc_allele_names})
        observations_df.to_csv(pMHC.OBSERVATION_INPUT_FILENAME, sep=",", index=False)

    @staticmethod
    def from_input():
        print(f"Observation.from_input")
        logger.info(f"Observation.from_input")

        observations_df = pd.read_csv(pMHC.OBSERVATION_INPUT_FILENAME).fillna("")

        # insert "splits"
        for idx, (key, sample_name, peptide_seq, target, datasource, deconv_mhc_allele_name) in \
                tqdm(observations_df.iterrows(), "Observations from input", disable=pMHC.TQDM_DISABLE):
            sample = Sample.find(sample_name)
            if sample is not None:
                peptide = Peptide.get(peptide_seq)

                obs = Observation(key, sample, peptide, target, datasource)
                if deconv_mhc_allele_name != "":
                    obs.deconvoluted_mhc_allele = MhcAllele.name_to_mhc_allele(deconv_mhc_allele_name)


class Decoy(Example):
    decoys = []
    cnt = defaultdict(lambda: 0)

    df = None

    def __init__(self, key, protein, position, peptide=None):
        super().__init__(key)

        self.observation = Observation.key_to_observation[key // 1000]
        self.protein = protein
        self.position = position
        self._peptide = peptide

    def get_tokenization(self, model):
        y, n_flank, peptide, c_flank = torch.zeros(1), self.n_flank, self.peptide_seq, self.c_flank

        if model.input_mode == pMHC.INPUT_PEPTIDE:  # if context should not be input to model
            n_flank, c_flank = "", ""

        if model.mhc_rep == pMHC.MHC_FULL:
            sample = get_input_rep_FULL(n_flank, peptide, c_flank, self.mhc_allele.seq, model)
        else:
            sample = get_input_rep_PSEUDO(n_flank, peptide, c_flank, self.mhc_allele.pseudo_seq, model)

        sample.update({'targets': y})
        sample.update({'objects': (self, self.mhc_allele, self.key % 1000)})
        return sample

    @property
    def n_flank(self):
        if self.observation.n_flank == "":
            return ""
        return self.protein.seq[max(self.position - FLANK_LEN, 0):self.position]

    @property
    def peptide_seq(self):
        if self._peptide is not None:
            return self._peptide
        else:
            return self.protein.seq[self.position:(self.position + len(self.observation.peptide.seq))]

    @property
    def c_flank(self):
        if self.observation.c_flank == "":
            return ""
        return self.protein.seq[(self.position + len(self.observation.peptide_seq)):
                                min(self.position + len(self.observation.peptide_seq) + FLANK_LEN, len(self.protein.seq))]

    @property
    def mhc_allele(self):
        return self.observation.deconvoluted_mhc_allele

    @staticmethod
    def create_decoys(observations=None):
        if observations is None:
            observations = Observation.observations
        for obs in tqdm(observations, disable=pMHC.TQDM_DISABLE):
            for idx in range(99):
                Decoy.create_decoy(obs)

    @staticmethod
    def create_decoy(observation):
        peptide = ""
        l_peptide = len(observation.peptide.seq)

        found = False
        cnt = 0
        protein = None
        position = None

        while not found:
            protein = random.choice(observation.sample.get_proteins())
            position = random.choice(range(len(protein.seq)))

            peptide = protein.seq[position:position+l_peptide]
            if peptide not in observation.sample.get_hit_seqs() and len(peptide) == l_peptide:
                found = True

            cnt += 1
            if not found and cnt >= 10:
                logger.info(f"Decoy.create_decoy: had to create a random example")
                peptide = Decoy.generate_random_peptide(observation.split)
                found = True
            else:
                peptide = None

        decoy = Decoy(observation.key * 1000 + Decoy.cnt[observation.key], protein, position, peptide)
        Decoy.cnt[observation.key] = Decoy.cnt[observation.key] + 1
        Decoy.decoys.append(decoy)

    @staticmethod
    def generate_random_peptide(split):
        sel = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        length = 9
        return "".join(np.random.choice(list(sel), length))

    @staticmethod
    def to_input(postfix=""):
        print(f"Decoy.to_input")
        logger.info(f"Decoy.to_input")

        keys, observation_keys, protein_names, positions, peptides = [], [], [], [], []
        for decoy in Decoy.decoys:
            keys.append(decoy.key)
            protein_names.append(decoy.protein.name)
            positions.append(decoy.position)
            peptides.append(decoy._peptide)

        observations_df = pd.DataFrame({'key': keys, 'protein_name': protein_names, 'position': positions, 'peptide': peptides})

        postfix = f"_{postfix}" if postfix != "" else ""
        filename = f"{pMHC.DECOY_INPUT_FILENAME_PREFIX}{postfix}.csv"
        observations_df.to_csv(filename, sep=",", index=False)

    @staticmethod
    def from_input(proportion=1.0, decoys_per_obs=99):
        print(f"Decoy.to_input")
        logger.info(f"Decoy.to_input")

        # get decoy files
        length = int(len(pMHC.OBSERVATION_PERMUTATION) * proportion) + 1
        print(f"Load decoys for {length} observations")

        for dirname, _, filenames in os.walk(pMHC.INPUT_FOLDER):
            for filename in filenames:
                if filename.startswith("decoys"):
                    prefix, from_, to_ = filename.split("_")
                    if int(from_) < length:
                        print(filename)
                        decoy_df = pd.read_csv(os.path.join(dirname, filename)).fillna("")
                        if Decoy.df is None:
                            Decoy.df = decoy_df
                        else:
                            Decoy.df = pd.concat([Decoy.df, decoy_df])

        Decoy.df = Decoy.df.set_index("key")

    @staticmethod
    def get_decoy(key):
        protein_name, position, peptide = Decoy.df.loc[key]
        protein = Protein.proteins[protein_name]
        if peptide == "":
            peptide = None
        return Decoy(key, protein, position, peptide)
