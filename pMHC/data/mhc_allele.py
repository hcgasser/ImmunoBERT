import pandas as pd
import re
from tqdm import tqdm
from collections import defaultdict

from Bio import SeqIO

import pMHC
from pMHC import logger


class MhcAllele:
    mhc_alleles = defaultdict(lambda: None)
    mhc_allele_groups = defaultdict(lambda: [])

    def __init__(self, name, seq, pseudo_seq, split=None):
        self.name = name
        self.seq = seq
        self.pseudo_seq = pseudo_seq
        self.split = split
        self.observations = []

        MhcAllele.mhc_alleles[name] = self
        MhcAllele.mhc_allele_groups[name[:7]].append(self)

    @staticmethod
    def register_mhc_allele(name, seq, pseudo_seq, split=None):
        if name not in MhcAllele.mhc_alleles:
            MhcAllele(name, seq, pseudo_seq, split)
        else:
            logger.warning(f"MhcAllele.register_mhc_allele: tried to register {name} multiple times")

    @staticmethod
    def from_data():
        print(f"MhcAllele.from_data")
        logger.info(f"MhcAllele.from_data")

        # load pseudo sequences
        mhc_pseudo_sequences = {}
        lines = open(pMHC.MHC_DATA_PSEUDO_FILENAME).read().splitlines()
        for line in lines:
            name, pseudo_seq = re.split("[ |\t]{1,}", line.strip())
            name = MhcAllele.text_to_names(name)
            if len(name) == 1:
                mhc_pseudo_sequences.update({name[0]: pseudo_seq})

        # load full sequences
        mhc_full_sequences = {}
        for mhc_filename in tqdm([pMHC.MHC_DATA_A_FILENAME, pMHC.MHC_DATA_B_FILENAME, pMHC.MHC_DATA_C_FILENAME]):
            for seq in tqdm(SeqIO.parse(mhc_filename, "fasta")):
                groove = None
                if len(seq.seq) == 181:
                    groove = 'G' + seq.seq
                elif len(seq.seq) > 360:
                    groove = seq.seq[24:24 + 182]

                if groove is not None:
                    m = re.search(r"\w+:\w+ (\w)\*(\w+):(\w+)", seq.description)
                    mhc_full_sequences.update({f"HLA-{m.group(1)}{m.group(2)}:{m.group(3)}": groove})

        for name, seq in mhc_full_sequences.items():
            pseudo_seq = None
            if name in mhc_pseudo_sequences:
                pseudo_seq = mhc_pseudo_sequences[name]
            MhcAllele.register_mhc_allele(name, seq, pseudo_seq)

    @staticmethod
    def to_input():
        print(f"MhcAllele.to_input")
        logger.info(f"MhcAllele.to_input")

        names, seqs, pseudo_seqs, splits = [], [], [], []
        for mhc_allele in list(MhcAllele.mhc_alleles.values()):
            names.append(mhc_allele.name)
            seqs.append(mhc_allele.seq)
            pseudo_seqs.append(mhc_allele.pseudo_seq)
            splits.append(mhc_allele.split)

        mhc_alleles = pd.DataFrame({'name': names, 'seq': seqs, 'pseudo_seq': pseudo_seqs, 'split': splits})
        mhc_alleles.to_csv(pMHC.MHC_INPUT_FILENAME, sep=",", index=False)

    @staticmethod
    def from_input():
        print(f"MhcAllele.from_input")
        logger.info(f"MhcAllele.from_input")

        mhc_alleles = pd.read_csv(pMHC.MHC_INPUT_FILENAME).fillna("")

        # insert "splits"
        for idx, (name, seq, pseudo_seq, split) in tqdm(mhc_alleles.iterrows(), "MhcAlleles from input"):
            MhcAllele.register_mhc_allele(name, seq, pseudo_seq, split)

    @staticmethod
    def name_to_mhc_allele(name):
        if name in MhcAllele.mhc_alleles:
            return MhcAllele.mhc_alleles[name]
        logger.warning(f"MhcAllele.name_to_mhc_allele: MHC name {name} not found")
        return None

    @staticmethod
    def names_to_mhc_alleles(names):
        return [mhc_allele for mhc_allele in
                [MhcAllele.name_to_mhc_allele(name) for name in names]
                if mhc_allele is not None]

    @staticmethod
    def text_to_names(text):
        # HLA-[ABC]\d{2}:\d{2}
        candidate_list = [f"HLA-{x[0]}{int(x[1]):02d}:{int(x[2]):02d}"
                         for x in re.findall(r"HLA-([ABC])\*?(\d{1,}):(\d{1,})", text)]
        if len(candidate_list) != text.count("HLA-"):
            candidate_list = []

        if text.count("HLA-A") > 2 or text.count("HLA-B") > 2 or text.count("HLA-C") > 2:
            candidate_list = []

        candidate_list = list(set(candidate_list))
        candidate_list.sort()

        return candidate_list

    @staticmethod
    def mhc_alleles_to_names(mhc_alleles):
        return "; ".join(sorted([mhc_allele.name for mhc_allele in mhc_alleles]))
