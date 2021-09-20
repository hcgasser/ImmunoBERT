import pdb
from argparse import ArgumentParser
from tqdm import tqdm

import pMHC
from pMHC.data.example import Observation, Decoy
from pMHC.data import from_input

tqdm.pandas()

if __name__ == "__main__":
    print("START")

    parser = ArgumentParser()
    parser.add_argument('--from', type=int, default=0)
    parser.add_argument('--to', type=int, default=100)
    parser.add_argument('--project_directory', type=str, default=r"C:\Users\tux\Documents\project_v2")
    args = parser.parse_args()

    args_dict = vars(args)
    print(args_dict)

    pMHC.set_paths(args_dict["project_directory"])

    from_input()

    with open(pMHC.OBSERVATION_PERMUTATION_FILENAME, 'r') as file:
        perm = file.read()

    perm = perm.split("; ")

    if args_dict["to"] < 0:
        selection_keys = perm[args_dict["from"]:]
    else:
        selection_keys = perm[args_dict["from"]:args_dict["to"]]
    selection_obs = [Observation.key_to_observation[int(key)] for key in selection_keys]

    Decoy.create_decoys(selection_obs)

    Decoy.to_input(f"{args_dict['from']}_{args_dict['to']}")
