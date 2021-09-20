import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import pdb
import os

from .example import Sample, Peptide, Example, Observation, Decoy
from .mhc_allele import MhcAllele
from .protein import Protein

import pMHC
from pMHC import logger, MIN_PEPTIDE_LEN, MAX_PEPTIDE_LEN


class Datasource:
    datasources = defaultdict(lambda: None)

    def __init__(self, name, input_filename):
        self.name = name
        self.input_filename = input_filename

        self.observations = []
        Datasource.datasources[name] = self

    def from_data(self):
        raise NotImplemented


class EdiDatasource(Datasource):

    def __init__(self):
        super().__init__("Edi", pMHC.EDI_INPUT_FILENAME)

    def from_data(self):
        print(f"EdiDatasource: from_data")
        logger.info(f"EdiDatasource: from_data")

        # load Samples

        edi_data_samples = pd.read_csv(pMHC.EDI_DATA_SAMPLES_FILENAME, delimiter="\t").fillna("")

        edi_data_samples = edi_data_samples[edi_data_samples["HLA Class"] == 1]
        edi_data_samples = edi_data_samples[edi_data_samples["HLA Type"] != ""]

        edi_data_samples = edi_data_samples[['Sample Name', 'HLA Type']]

        def add_sample(row):
            _sample_name, _mhc_allele_names = row
            _sample_name = f"Edi_{_sample_name}"
            _sample = Sample.get(_sample_name)
            _mhc_allele_names = [_mhc_allele_name.replace("*", "") for _mhc_allele_name in _mhc_allele_names.split("; ")]
            _sample.add_mhc_alleles(MhcAllele.names_to_mhc_alleles(_mhc_allele_names))

        edi_data_samples.progress_apply(add_sample, axis=1)

        # load Observations

        edi_data_observations = pd.read_csv(pMHC.EDI_DATA_OBSERVATIONS_FILENAME).fillna("")
        edi_data_observations = edi_data_observations[(edi_data_observations["tag"] == 'closed search')
                                            | (edi_data_observations["tag"] == 'open search')]
        edi_data_observations = edi_data_observations[["peptide", "protein", "sample_name"]]
        print("  load_peptide_to_protein_names")
        peptide_to_protein_names = defaultdict(lambda: [])
        def load_peptide_to_protein_names(row):
            _peptide_seq, _ensembl, _ = row
            peptide_to_protein_names[_peptide_seq].append(_ensembl)
        edi_data_observations.progress_apply(load_peptide_to_protein_names, axis=1)
        for peptide_seq, _ in peptide_to_protein_names.items():
            peptide_to_protein_names[peptide_seq] = list(set(peptide_to_protein_names[peptide_seq]))
        edi_data_observations = edi_data_observations[["peptide", "sample_name"]]
        edi_data_observations = edi_data_observations.drop_duplicates()

        def add_observation(row):
            _peptide_seq, _sample_name = row
            _proteins = [x for x in Protein.names_to_proteins(peptide_to_protein_names[_peptide_seq]) if x is not None]
            if len(_proteins) > 0 and MIN_PEPTIDE_LEN <= len(_peptide_seq) <= MAX_PEPTIDE_LEN:
                _sample_name = f"Edi_{_sample_name}"
                _target = 1.

                _sample = Sample.find(_sample_name)
                if _sample is not None:
                    _peptide = Peptide.get(_peptide_seq)

                    _peptide.add_proteins(_proteins)
                    _key = len(Example.examples)
                    Observation(_key, _sample, _peptide, _target, "Edi")

        edi_data_observations.progress_apply(add_observation, axis=1)


class AtlasDatasource(Datasource):

    def __init__(self):
        super().__init__("Atlas", pMHC.ATLAS_INPUT_FILENAME)

    def from_data(self):
        print(f"AtlasDatasource: from_data")
        logger.info(f"AtlasDatasource: from_data")

        # load Samples

        atlas_donors = pd.read_csv(pMHC.ATLAS_DATA_DONORS_FILENAME, delimiter="\t").fillna("")
        donor_to_mhc_alleles = defaultdict(lambda: [])
        for idx, (donor, hla_allele) in tqdm(atlas_donors.iterrows(), "load Atlas donors"):
            mhc_allele = MhcAllele.name_to_mhc_allele(f"HLA-{hla_allele.replace('*', '')}")
            if mhc_allele is not None:
                donor_to_mhc_alleles[donor].append(mhc_allele)

        atlas_sample_hits = pd.read_csv(pMHC.ATLAS_DATA_SAMPLES_FILENAME, delimiter="\t").fillna("")
        atlas_sample_hits = atlas_sample_hits[atlas_sample_hits["hla_class"] == "HLA-I"]
        atlas_samples = atlas_sample_hits[["donor", "tissue"]]
        atlas_samples = atlas_samples.drop_duplicates()

        def add_sample(row):
            _donor, _tissue = row
            _sample_name = f"Atlas_{_donor}_{_tissue}"
            _sample = Sample.get(_sample_name)
            _sample.add_mhc_alleles(donor_to_mhc_alleles[_donor])

        atlas_samples.progress_apply(add_sample, axis=1)

        # load Observations

        atlas_peptides = pd.read_csv(pMHC.ATLAS_DATA_PEPTIDES_FILENAME, delimiter="\t").fillna("")
        atlas_observations = pd.merge(atlas_sample_hits, atlas_peptides, how="left", on="peptide_sequence_id")

        atlas_protein_map = pd.read_csv(pMHC.ATLAS_DATA_PROTEINS_FILENAME, delimiter="\t").fillna("")
        seqid_to_protein_names = defaultdict(lambda: [])
        for idx, (seqid, uniprot) in tqdm(atlas_protein_map.iterrows(), "load Atlas protein map"):
            seqid_to_protein_names[seqid].append(uniprot)
        for seqid, _ in seqid_to_protein_names.items():
            seqid_to_protein_names[seqid] = list(set(seqid_to_protein_names[seqid]))

        atlas_observations = atlas_observations[["peptide_sequence_id", "peptide_sequence", "donor", "tissue"]].drop_duplicates()

        def add_observation(row):
            _seqid, _peptide_seq, _donor, _tissue = row
            _proteins = [x for x in Protein.names_to_proteins(seqid_to_protein_names[_seqid]) if x is not None]
            if len(_proteins) > 0 and MIN_PEPTIDE_LEN <= len(_peptide_seq) <= MAX_PEPTIDE_LEN:
                _peptide = Peptide.get(_peptide_seq)
                _sample_name = f"Atlas_{_donor}_{_tissue}"
                _target = 1.

                _sample = Sample.find(_sample_name)
                if _sample is not None:
                    _peptide.add_proteins(_proteins)
                    _key = len(Example.examples)
                    Observation(_key, _sample, _peptide, _target, "Atlas")

        atlas_observations.progress_apply(add_observation, axis=1)


def overview_datasources():
    samples = {"Edi": [], "Atlas": [], "Total": []}

    obs = {"Edi": [], "Atlas": [], "Total": []}
    obs_sa = {"Edi": [], "Atlas": [], "Total": []}
    obs_ma = {"Edi": [], "Atlas": [], "Total": []}

    peptides = {"Edi": [], "Atlas": [], "Total": []}
    mhc_alleles = {"Edi": [], "Atlas": [], "Total": []}

    for observation in Observation.observations:
        datasource = observation.datasource

        samples[datasource].append(observation.sample)
        samples["Total"].append(observation.sample)

        obs[datasource].append(observation)
        obs["Total"].append(observation)

        if len(observation.sample.mhc_alleles) == 1:
            obs_sa[datasource].append(observation)
            obs_sa["Total"].append(observation)
        else:
            obs_ma[datasource].append(observation)
            obs_ma["Total"].append(observation)

        peptides[datasource].append(observation.peptide.seq)
        peptides["Total"].append(observation.peptide.seq)

        mhc_alleles[datasource] += observation.sample.mhc_alleles
        mhc_alleles["Total"] += observation.sample.mhc_alleles

    print(f"{'Datasource':<20s} {'Samples':>15s} {'Observations':>15s}  {'Obs - SA':>15s} {'Obs - MA':>15s} {'peptides':>15s} {'MHC alleles':>15s}")
    for datasource in ["Edi", "Atlas", "Total"]:
        print(f"{datasource:<20s} {len(set(samples[datasource])):>15,d} {len(set(obs[datasource])):>15,d}  {len(set(obs_sa[datasource])):>15,d} {len(set(obs_ma[datasource])):>15,d} {len(set(peptides[datasource])):>15,d} {len(set(mhc_alleles[datasource])):>15,d}")
