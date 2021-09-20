from tqdm import tqdm

import pMHC
from pMHC import \
    SPLITS, SPLIT_NAMES, \
    VIEWS, VIEW_SA, VIEW_SAMA, VIEW_DECONV, VIEW_NAMES
from .example import Observation


def overview_views():
    print("\n\nSplits\n")
    print(f"{'OBSERVATIONS':<20s}    ", end="")
    for view in VIEWS:
        print(f" {VIEW_NAMES[view]:>15s}", end="")
    for split in SPLITS:
        print(f"\n   {SPLIT_NAMES[split]:<20s}: ", end="")
        for view in VIEWS:
            print(f" {len(Observation.obs_views[split][view]):>15,d}", end="")

    print("\n")


def create_views(proportion=1.0):
    obs_views = [
        [[], [], []],   # train split
        [[], [], []],   # val split
        [[], [], []],   # test split
        [[], [], []],   # val-proteins
        [[], [], []],   # test-proteins
        [[], [], []],   # val-mhc_alleles
        [[], [], []]    # val-mhc_alleles
    ]

    observations = Observation.observations
    if proportion < 1.0:
        print("Reduce observations")
        perm = list(pMHC.OBSERVATION_PERMUTATION[:int(len(pMHC.OBSERVATION_PERMUTATION) * proportion)])
        observations = [Observation.key_to_observation[key] for key in perm]

    for observation in tqdm(observations, disable=pMHC.TQDM_DISABLE):
        splits = observation.get_splits()

        if len(observation.sample.mhc_alleles) == 1:
            for split in splits:
                obs_views[split][VIEW_SA].append(observation)
                obs_views[split][VIEW_SAMA].append(observation)
        else:
            for split in splits:
                obs_views[split][VIEW_SAMA].append(observation)
                for mhc_allele in observation.sample.mhc_alleles:
                    obs_views[split][VIEW_DECONV].append((observation, mhc_allele))

    Observation.obs_views = obs_views
