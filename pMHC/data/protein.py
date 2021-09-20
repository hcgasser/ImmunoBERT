import pandas as pd
import re
from tqdm import tqdm
from collections import defaultdict
import random

from Bio import SeqIO

import pMHC
from pMHC import logger, FLANK_LEN


def generate_random_peptides(len_peptide=9, count=100000):
    n_flanks = []
    peptides = []
    c_flanks = []
    for i in tqdm(range(count)):
        found = False
        while not found:
            protein = random.choice(list(Protein.proteins.values()))
            if protein is not None:
                position = random.choice(range(len(protein.seq)))
                peptide = protein.seq[position:position + len_peptide]
                if len(peptide) == len_peptide:
                    found = True

        n_flank = protein.seq[max(position - FLANK_LEN, 0):position]
        c_flank = protein.seq[(position + len_peptide):min(position + len_peptide + FLANK_LEN, len(protein.seq))]

        n_flanks.append(n_flank)
        peptides.append(peptide)
        c_flanks.append(c_flank)

    return n_flanks, peptides, c_flanks


class Protein:
    proteins = defaultdict(lambda: None)   # holds both - ensembl and uniprot names

    def __init__(self, name, seq, kind):
        self.name = name
        self.seq = seq
        self.kind = kind

        self.paralogue = None
        self.observations = []
        Protein.proteins[name] = self

    def link_observation(self, observation):
        self.observations.append(observation)

    @staticmethod
    def register_protein(name, seq, kind):
        if name not in Protein.proteins:
            Protein(name, seq, kind)
        else:
            logger.warning(f"Protein.register_protein: tried to register {name} multiple times")

    @staticmethod
    def from_data():
        print(f"Protein: from_data")
        logger.info(f"Protein.from_data")

        for seq in tqdm(SeqIO.parse(pMHC.PROTEIN_DATA_UNIPROT_FILENAME, "fasta")):
            m = re.search(r"(\w+)\|(\w+)\|(\w+)", seq.id)
            name = m.group(2)
            Protein.register_protein(name, seq.seq, "uniprot")

        for seq in tqdm(SeqIO.parse(pMHC.PROTEIN_DATA_ENSEMBL_FILENAME, "fasta")):
            m = re.search(r"(\w+)\|(\w+)\|(\w+)\|", seq.id)
            name = m.group(1)
            Protein.register_protein(name, seq.seq, "ensembl")

    @staticmethod
    def to_input():
        print(f"Protein.to_input")
        logger.info(f"Protein.to_input")

        names, seqs, kinds = [], [], []
        for protein in list(Protein.proteins.values()):
            if protein is not None:
                names.append(protein.name)
                seqs.append(protein.seq)
                kinds.append(protein.kind)

        proteins = pd.DataFrame({'name': names, 'seq': seqs, 'kind': kinds})
        proteins.to_csv(pMHC.PROTEIN_INPUT_FILENAME, sep=",", index=False)

    @staticmethod
    def from_input():
        print(f"Protein.from_input")
        logger.info(f"Protein.from_input")

        proteins = pd.read_csv(pMHC.PROTEIN_INPUT_FILENAME).fillna("")
        for idx, (name, seq, kind) in tqdm(proteins.iterrows(), "Proteins from input"):
            Protein.register_protein(name, seq, kind)

    @staticmethod
    def find_protein(name):
        if name in Protein.proteins:
            return Protein.proteins[name]
        else:
            logger.warning(f"Protein.get_protein: protein name {name} not found")
            return None

    @staticmethod
    def names_to_proteins(names):
        return [Protein.proteins[name] for name in names]

    @staticmethod
    def proteins_to_names(proteins):
        return "; ".join([protein.name for protein in proteins])