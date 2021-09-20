from .view import overview_views
from .protein import Protein
from .mhc_allele import MhcAllele
from .example import Sample, Peptide, Observation

from pMHC import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_NAMES, SPLITS, SPLIT_OBS_TARGETS, SPLIT_TOLERANCE_MHC, \
    VIEW_SA, VIEW_SAMA, VIEW_DECONV, VIEW_NAMES, VIEWS


def overview(model):
    overview_datasources()
    overview_views()
    overview_dataloaders(model)

def overview_peptides(model):
    peptides_splits_sa = []
    peptides_splits_sama = []

    prev_view = model.view
    model.view = VIEW_SAMA
    for split in SPLITS:
        peptides_split_sa = []
        peptides_split_sama = []
        for obs in model.ds[split]:
            if len(obs.sample.mhc_alleles) == 1:
                peptides_split_sa.append(obs.peptide.seq)
            else:
                peptides_split_sama.append(obs.peptide.seq)

        peptides_splits_sa.append(set(peptides_split_sa))
        peptides_splits_sama.append(set(peptides_split_sama))

    print(f"{'Intersection SA peptides':<30s} ")
    for split_col in SPLITS:
        print(f"{SPLIT_NAMES[split_col]:>20s} ", end="")
    print("")
    for split_row in SPLITS:
        print(f"{SPLIT_NAMES[split_row]:<30s} ", end="")
        for split_col in SPLITS:
            print(f"{len(peptides_splits_sa[split_row].intersection(peptides_splits_sa[split_col])):>20d}", end="")
        print("")

    model.view = prev_view


def overview_datasources():
    print("\n\nDatasources\n")
    cnt_src = {"Edi": 0, "Atlas": 0, "TOTAL": 0}
    cnt_mhc = {"Edi": 0, "Atlas": 0, "TOTAL": 0}

    for obs in list(Observation.observations):
        cnt_mhc[obs.datasource] += len(obs.sample.mhc_alleles)
        cnt_src[obs.datasource] += 1

    print(f"{'OBSERVATIONS':<20s}     {'':>10s} /{'MHC/Obs comb':>15s}")
    for key in ["Edi", "Atlas", "TOTAL"]:
        print(f"   {key:<20s}: {cnt_src[key]:>10,d} /{cnt_mhc[key]:>15,d}")
        cnt_src["TOTAL"] += cnt_src[key]
        cnt_mhc["TOTAL"] += cnt_mhc[key]

    print("\n")


def overview_dataloaders(model):
    print("\n\nDataloaders\n")

    print("EXAMPLES\n")
    old_view = model.view
    for split in SPLITS:
        split_name = SPLIT_NAMES[split]
        print(f"   {split_name:<20s}")
        for view in VIEWS:
            model.view = view
            view_name = VIEW_NAMES[view]
            length = model.ds[split].length()
            print(f"      {view_name:<17s}: {length:>10,d} /{int(length / model.batch_size):>15,d}")

    model.view = old_view


def print_observation_statistics():
    peptides = []
    proteins = []
    mhc_alleles = []

    observations_split = [[], [], []]
    peptides_split = [[], [], []]
    proteins_split = [[], [], []]
    mhc_alleles_split = [[], [], []]

    for observation in list(Observation.observations.values()):
        peptides.append(observation.peptide)
        peptides_split[observation.split].append(observation.peptide)

        proteins.append += observation.proteins
        proteins_split[observation.split] += observation.proteins

        observations_split[observation.split].append(observation)

        mhc_alleles += observation.mhc_alleles
        mhc_alleles_split[observation.split] += observation.mhc_alleles


    print(f"Total:")
    print(f"{len(Observation.observations)} observations "
          + f"of {len(set(peptides))} different peptides "
          + f"from {len(set(proteins))} proteins "
          + f"concerning {len(set(mhc_alleles))} mhc alleles")
    for split in SPLITS:
        print(f"{SPLIT_NAMES[split]}:")
        print(f"{len(observations_split[split])} observations "
              + f"of {len(set(peptides_split[split]))} different peptides "
              + f"from {len(set(proteins_split[split]))} proteins "
              + f"concerning {len(set(mhc_alleles_split[split]))} mhc alleles")

