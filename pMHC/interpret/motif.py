import random
from tqdm import tqdm

from pMHC import FLANK_LEN
from pMHC.data.protein import Protein


def generate_random_peptides_from_proteome(len_peptide=9, length=100000):
    n_flanks = []
    peptides = []
    c_flanks = []
    for i in tqdm(range(length)):
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
