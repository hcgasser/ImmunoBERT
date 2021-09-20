import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from collections import defaultdict

import networkx as nx
from Bio import SeqIO

from pytorch_lightning.utilities.seed import seed_everything

from .protein import Protein
from .example import Sample, Observation, MhcAllele

import pMHC
from pMHC import \
    PARALOGUE_DATA_HGNC, PARALOGUE_DATA_UNIPROT_MAPPING, PARALOGUE_DATA_UNIPROT_PROTEIN, \
    PARALOGUE_DATA_ENSEMBL_MAPPING, PARALOGUE_DATA_ENSEMBL_PROTEIN, PARALOGUE_DATA_MAIN, \
    SPLITS, SPLIT_NAMES, \
    SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_VAL_MHC_ALLELES, SPLIT_TEST_MHC_ALLELES, \
    SPLIT_OBS_TARGETS, SPLIT_TOLERANCE_MHC, SPLIT_TOLERANCE_PROTEIN, \
    SPLIT_TEST_PROTEINS, SPLIT_VAL_PROTEINS, \
    VIEW_SA, VIEW_SAMA, VIEW_DECONV, VIEW_NAMES, VIEWS


def overview_split():
    cnt = [0, 0, 0, 0, 0, 0, 0]
    cnt_sa = [0, 0, 0, 0, 0, 0, 0]
    cnt_ma = [0, 0, 0, 0, 0, 0, 0]

    obs_tot = len(Observation.observations)

    for observation in Observation.observations:
        splits = observation.get_splits()

        for split in splits:
            cnt[split] += 1
            if len(observation.sample.mhc_alleles) == 1:
                cnt_sa[split] += 1
            else:
                cnt_ma[split] += 1

    text_abs = "\n\nABSOLUTE\n                        "
    text_rel = "\n\nRELATIVE\n                        "
    for split in SPLITS:
        text = f"{SPLIT_NAMES[split]:>10s} "
        text_abs += text
        text_rel += text
    text = f"\nObservations in splits: "
    text_abs += text
    text_rel += text
    for split in SPLITS:
        text_abs += f"{cnt[split]:>10,d} "
        text_rel += f"{cnt[split]/obs_tot:>10.1%} "
    text = f"\n                    SA: "
    text_abs += text
    text_rel += text
    for split in SPLITS:
        text_abs += f"{cnt_sa[split]:>10,d} "
        text_rel += f"{cnt_sa[split]/obs_tot:>10.1%} "
    text = f"\n                    MA: "
    text_abs += text
    text_rel += text
    for split in SPLITS:
        text_abs += f"{cnt_ma[split]:>10,d} "
        text_rel += f"{cnt_ma[split]/obs_tot:>10.1%} "

    print(text_abs)
    print(text_rel)


def find_connected_mhc_alleles():
    graph = nx.Graph()

    for sample in Sample.samples.values():
        for mhc_alleles in sample.mhc_alleles:
            graph.add_edge(mhc_alleles.name[:7], sample)

    groups = list(nx.connected_components(graph))

    total_cnt = 0

    for group in groups:
        cnt = 0
        for element in group:
            if type(element) == Sample:
                cnt += len(element.observations)
            else:
                print(f"{element}: ", end="")
        print(f" {cnt}\n")
        total_cnt += cnt


def suggest_split_mhc_alleles():
    graph = nx.DiGraph()

    for sample in Sample.samples.values():
        for mhc_alleles in sample.mhc_alleles:
            graph.add_edge(mhc_alleles.name[:7], sample)

    print(f"{'group name':<10s}: {'samples':>10s} {'observations':>15s}")

    mhc_allele_groups_names = []
    mhc_allele_groups_obscnts = []

    obs_cnt_total = 0
    for mhc_group_name in MhcAllele.mhc_allele_groups.keys():
        if mhc_group_name in graph:
            samples = nx.descendants(graph, mhc_group_name)
            obs_cnt = 0
            for sample in samples:
                obs_cnt += len(sample.observations)

            print(f"{mhc_group_name:<10s}: {len(samples):>10,d} {obs_cnt:>15,d}")
            mhc_allele_groups_names.append(mhc_group_name)
            mhc_allele_groups_obscnts.append(obs_cnt)
            obs_cnt_total += obs_cnt


    total_obscnt = len(Observation.observations)

    found = False
    seed_everything(42)
    while not found:
        perm = np.random.permutation(range(len(mhc_allele_groups_names)))
        j = 0
        mhc_allele_groups_test = []
        cnt_test = 0
        while cnt_test < total_obscnt * SPLIT_OBS_TARGETS[SPLIT_TEST_MHC_ALLELES] * (1 - SPLIT_TOLERANCE_MHC):
            idx = perm[j]
            mhc_allele_groups_test.append(mhc_allele_groups_names[idx])
            cnt_test += mhc_allele_groups_obscnts[idx]
            j += 1

        if cnt_test > total_obscnt * SPLIT_OBS_TARGETS[SPLIT_TEST_MHC_ALLELES] * (1 + SPLIT_TOLERANCE_MHC) \
                or len(mhc_allele_groups_test) < 5:
            continue

        mhc_allele_groups_val = []
        cnt_val = 0
        while cnt_val < total_obscnt * SPLIT_OBS_TARGETS[SPLIT_VAL_MHC_ALLELES] * (1 - SPLIT_TOLERANCE_MHC):
            idx = perm[j]
            mhc_allele_groups_val.append(mhc_allele_groups_names[idx])
            cnt_val += mhc_allele_groups_obscnts[idx]
            j += 1

        if cnt_val > total_obscnt * SPLIT_OBS_TARGETS[SPLIT_VAL_MHC_ALLELES] * (1 + SPLIT_TOLERANCE_MHC) \
                or len(mhc_allele_groups_val) < 5:
            continue

        found = True

    print("\n\n")
    print(f"Suggested Test MHCs: {mhc_allele_groups_test}")
    print(f"Suggested Val MHCs:  {mhc_allele_groups_val}")


def save_split_mhc_alleles(test_mhc_allele_groups, val_mhc_allele_groups):
    val_groups = "; ".join(val_mhc_allele_groups)
    with open(pMHC.SPLIT_MHC_ALLELES_VAL_FILENAME, 'w') as file:
        file.write(val_groups)

    test_groups = "; ".join(test_mhc_allele_groups)
    with open(pMHC.SPLIT_MHC_ALLELES_TEST_FILENAME, 'w') as file:
        file.write(test_groups)


def load_split_mhc_alleles():
    with open(pMHC.SPLIT_MHC_ALLELES_VAL_FILENAME, 'r') as file:
        val_mhc_allele_groups = file.read()
    val_mhc_allele_groups = val_mhc_allele_groups.split("; ")

    with open(pMHC.SPLIT_MHC_ALLELES_TEST_FILENAME, 'r') as file:
        test_mhc_allele_groups = file.read()
    test_mhc_allele_groups = test_mhc_allele_groups.split("; ")

    for sample in Sample.samples.values():
        sample.split_mhc = SPLIT_TRAIN

    for sample in Sample.samples.values():
        for mhc_allele in sample.mhc_alleles:
            if mhc_allele.name[:7] in val_mhc_allele_groups:
                sample.split_mhc = SPLIT_VAL_MHC_ALLELES

    for sample in Sample.samples.values():
        for mhc_allele in sample.mhc_alleles:
            if mhc_allele.name[:7] in test_mhc_allele_groups:
                sample.split_mhc = SPLIT_TEST_MHC_ALLELES


def find_split_proteins():
    graph = nx.Graph()
    add_links_proteins(graph)

    # add links established in files
    add_links_hgnc(graph)
    add_links_uniprot(graph)
    add_links_ensembl(graph)

    # add paralogue info
    add_paralogue_links(graph)

    for observation in Observation.observations:
        for protein in observation.peptide.proteins:
            graph.add_edge(observation, protein.name)
            graph.add_edge(observation.peptide.seq, protein.name)

    obs_groups = create_observation_groups(graph)

    obs_groups_with_cnts = []
    for obs_group in obs_groups:
        obs_groups_with_cnts.append((len(obs_group), obs_group))

    total_obscnt = len(Observation.observations)

    obs_groups_test, obs_groups_val = [], []

    found = False
    seed_everything(42)
    while not found:
        perm = np.random.permutation(range(1, len(obs_groups_with_cnts)))
        j = 0
        obs_groups_test = []
        cnt_test = 0
        while cnt_test < total_obscnt * SPLIT_OBS_TARGETS[SPLIT_TEST_PROTEINS] * (1 - SPLIT_TOLERANCE_PROTEIN):
            idx = perm[j]
            obs_groups_test.append(obs_groups_with_cnts[idx][1])
            cnt_test += obs_groups_with_cnts[idx][0]
            j += 1

        if cnt_test > total_obscnt * SPLIT_OBS_TARGETS[SPLIT_TEST_PROTEINS] * (1 + SPLIT_TOLERANCE_PROTEIN):
            continue

        obs_groups_val = []
        cnt_val = 0
        while cnt_val < total_obscnt * SPLIT_OBS_TARGETS[SPLIT_VAL_PROTEINS] * (1 - SPLIT_TOLERANCE_PROTEIN):
            idx = perm[j]
            obs_groups_val.append(obs_groups_with_cnts[idx][1])
            cnt_val += obs_groups_with_cnts[idx][0]
            j += 1

        if cnt_test > total_obscnt * SPLIT_OBS_TARGETS[SPLIT_VAL_PROTEINS] * (1 + SPLIT_TOLERANCE_PROTEIN):
            continue

        found = True

    observation_val = []
    for observations in obs_groups_val:
        observation_val += observations
    observation_val = [str(x.key) for x in observation_val]

    observation_test = []
    for observations in obs_groups_test:
        observation_test += observations
    observation_test = [str(x.key) for x in observation_test]

    val_keys = "; ".join(observation_val)
    with open(pMHC.SPLIT_PROTEINS_VAL_FILENAME, 'w') as file:
        file.write(val_keys)

    test_keys = "; ".join(observation_test)
    with open(pMHC.SPLIT_PROTEINS_TEST_FILENAME, 'w') as file:
        file.write(test_keys)


def create_observation_groups(graph):
    groups = list(nx.connected_components(graph))

    observation_groups = []
    for group in groups:
        observations = []
        for element in group:
            if type(element).__name__ == "Observation":
                observations.append(element)
        if len(observations) > 0:
            observation_groups.append(observations)

    return observation_groups


def load_split_proteins():
    with open(pMHC.SPLIT_PROTEINS_VAL_FILENAME, 'r') as file:
        val_keys = file.read()
    val_keys = [int(x) for x in val_keys.split("; ")]

    with open(pMHC.SPLIT_PROTEINS_TEST_FILENAME, 'r') as file:
        test_keys = file.read()
    test_keys = [int(x) for x in test_keys.split("; ")]

    for val_key in val_keys:
        Observation.key_to_observation[val_key].split_protein = SPLIT_VAL_PROTEINS

    for test_key in test_keys:
        Observation.key_to_observation[test_key].split_protein = SPLIT_TEST_PROTEINS


def add_links_proteins(graph):
    for key, protein in Protein.proteins.items():
        graph.add_edge(protein, key)


def add_links_hgnc(graph):
    df = pd.read_csv(PARALOGUE_DATA_HGNC, delimiter="\t").fillna("")

    def process_hgnc(row):
        _hgnc_id, _approved, _previous, _alias, _ensg, _uniprot = row

        graph.add_edge(_approved, _hgnc_id)

        _list = _previous.split(", ")
        for _term in [x for x in _list if x != ""]:
            graph.add_edge(_approved, _term)

        _list = _alias.split(", ")
        for _term in [x for x in _list if x != ""]:
            graph.add_edge(_approved, _term)

        if _ensg != "":
            graph.add_edge(_approved, _ensg)

        if _uniprot != "":
            graph.add_edge(_approved, _uniprot)

    df.progress_apply(process_hgnc, axis=1)


def add_links_uniprot(graph):
    # process the mappings file
    df = pd.read_csv(PARALOGUE_DATA_UNIPROT_MAPPING, delimiter="\t").fillna("")
    df.columns = [
        "UniProtKB-AC", "UniProtKB-ID", "GeneID_(EntrezGene)", "RefSeq", "GI", "PDB", "GO", "UniRef100", "UniRef90",
        "UniRef50", "UniParc", "PIR", "NCBI-taxon", "MIM", "UniGene", "PubMed", "EMBL", "EMBL-CDS", "Ensembl",
        "Ensembl_TRS", "Ensembl_PRO", "Additional_PubMed"
    ]
    df = df[["UniProtKB-AC", "Ensembl_PRO", "Ensembl"]]

    def process_uniprot_mapping(row):
        _uniprot, _ENSGs, _ENSPs = row
        _uniprot.replace("_HUMAN", "")

        _list = _ENSGs.split("; ")
        for _term in [x for x in _list if x != ""]:
            graph.add_edge(_uniprot, _term)

        _list = _ENSPs.split("; ")
        for _term in [x for x in _list if x != ""]:
            graph.add_edge(_uniprot, _term)

    df.progress_apply(process_uniprot_mapping, axis=1)

    # process the protein file
    for seq in tqdm(SeqIO.parse(PARALOGUE_DATA_UNIPROT_PROTEIN, "fasta"), disable=pMHC.TQDM_DISABLE):
        m = re.search(r"(\w+)\|(\w+)\|(\w+)", seq.id)
        uniprot = m.group(2)
        gene = m.group(3).replace("_HUMAN", "")
        graph.add_edge(uniprot, gene)


def add_links_ensembl(graph):
    # process the mappings file
    df = pd.read_csv(PARALOGUE_DATA_ENSEMBL_MAPPING, delimiter=",").fillna("")

    def process_ensembl_mapping(row):
        _gene, _hgnc_id, _hgnc = row

        if _hgnc != "":
            graph.add_edge(_hgnc, _gene)
            if _hgnc_id != "":
                graph.add_edge(_hgnc, _hgnc_id)

    df.progress_apply(process_ensembl_mapping, axis=1)

    # process the protein file
    for seq in tqdm(SeqIO.parse(PARALOGUE_DATA_ENSEMBL_PROTEIN, "fasta")):
        m = re.search(r"(\w+)\|(\w+)\|(\w+)\|(\w+)", seq.id)
        ENSP = m.group(1)
        ENST = m.group(2)
        ENSG = m.group(3)
        HGNC = m.group(4)
        graph.add_edge(HGNC, ENSP)
        graph.add_edge(HGNC, ENST)
        graph.add_edge(HGNC, ENSG)

        uniprots = re.findall(r"\|(\w{3,10})$", seq.description)
        for uniprot in uniprots:
            graph.add_edge(HGNC, uniprot)


def add_paralogue_links(graph):
    df = pd.read_csv(PARALOGUE_DATA_MAIN, delimiter="\t").fillna("")

    def process_paralogue_links(row):
        _src_ensg, _src_enst, _src_ensp, _tgt_ensg, _tgt_gene_name, _tgt_ensp = row

        if _src_enst != "":
            graph.add_edge(_src_ensg, _src_enst)
        if _src_ensp != "":
            graph.add_edge(_src_ensg, _src_ensp)
        if _tgt_ensg != "":
            graph.add_edge(_src_ensg, _tgt_ensg)
        if _tgt_gene_name != "":
            graph.add_edge(_src_ensg, _tgt_gene_name)
        if _tgt_ensp != "":
            graph.add_edge(_src_ensg, _tgt_ensp)

    df.progress_apply(process_paralogue_links, axis=1)



