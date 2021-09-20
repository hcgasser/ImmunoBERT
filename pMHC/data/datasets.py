import pdb
import numpy as np

import torch.utils.data

from pytorch_lightning.utilities.seed import seed_everything

from .example import Observation

from pMHC import logger, \
    VIEW_SA, VIEW_SAMA, VIEW_DECONV, \
    INPUT_PEPTIDE, INPUT_CONTEXT, MHC_FULL, MHC_PSEUDO, \
    SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from pMHC.data.example import Decoy
from .utils import get_input_rep_PSEUDO, get_input_rep_FULL


class StandardDataset(torch.utils.data.IterableDataset):

    def __init__(self, split, model):
        self.split = split
        self.model = model
        self.collator = StandardCollator(model)

    def __getitem__(self, idx):
        if self.model.view != VIEW_DECONV:
            obs = Observation.obs_views[self.split][self.model.view][idx // (self.model.decoys_per_obs + 1)]
            relevant_mhc_allele = obs.deconvoluted_mhc_allele
        else:
            obs, relevant_mhc_allele = Observation.obs_views[self.split][self.model.view][idx]

        # prepare dependent variable
        decoy_idx = (idx % (self.model.decoys_per_obs + 1)) - 1
        if self.model.view == VIEW_DECONV or decoy_idx == -1:  # the example is an actual observation
            y, n_flank, peptide, c_flank = torch.tensor([obs.target]), obs.n_flank, obs.peptide_seq, obs.c_flank
        else:  # the example is a decoy
            decoy = Decoy.get_decoy(obs.key*1000 + decoy_idx)
            y, n_flank, peptide, c_flank = torch.zeros(1), decoy.n_flank, decoy.peptide_seq, decoy.c_flank

        if self.model.input_mode == INPUT_PEPTIDE:  # if context should not be input to model
            n_flank, c_flank = "", ""

        if self.model.mhc_rep == MHC_FULL:
            sample = get_input_rep_FULL(n_flank, peptide, c_flank, relevant_mhc_allele.seq, self.model)
        else:
            sample = get_input_rep_PSEUDO(n_flank, peptide, c_flank, relevant_mhc_allele.pseudo_seq, self.model)

        sample.update({'targets': y})
        sample.update({'objects': (obs, relevant_mhc_allele, decoy_idx)})
        return sample

    def length(self):
        length = len(Observation.obs_views[self.split][self.model.view])
        if self.model.view != VIEW_DECONV:
            return (self.model.decoys_per_obs + 1) * length
        return length

    def __iter__(self):
        self.idx = 0

        self.deconv_idx = 0
        self.deconv_set = 0

        # randomly shuffle the data
        self.ds_permutation = None
        if self.split == SPLIT_TRAIN and self.model.shuffle_data:
            seed_everything(self.model.seed + self.model.current_epoch)
            self.ds_permutation = np.random.permutation(range(self.length()))
        return self

    def __next__(self):
        # currently deconvoluting
        if self.split == SPLIT_VAL and self.model.view == VIEW_DECONV:
            # if no observations left to deconvolute in the current split
            if self.deconv_set <= SPLIT_TEST and self.idx >= self.model.ds[self.deconv_set].length():
                self.idx = 0
                self.deconv_set += 1  # deconvolute the next not empty data split
                while self.deconv_set <= SPLIT_TEST and self.model.ds[self.deconv_set].length() == 0:
                    self.deconv_set += 1

            # if no observations left for deconvolution
            if self.deconv_set > SPLIT_TEST:
                # fill with dummy until batch full, then start validation
                if self.deconv_idx % self.model.batch_size == 0:
                    self.idx = 0
                    self.model.view = VIEW_SAMA
                    return self.__next__()
                else:
                    self.deconv_idx += 1
                    return self.model.ds[SPLIT_TRAIN][0]

            # return one observation for deconvolution
            else:
                self.idx += 1
                self.deconv_idx += 1
                return self.model.ds[self.deconv_set][self.idx - 1]
        elif self.idx < self.length() and (self.split != SPLIT_VAL or self.model.only_deconvolute_ is False):
            self.idx += 1
            ds_idx = self.idx - 1
            if self.ds_permutation is not None:  # implements the shuffling of the data during training
                ds_idx = self.ds_permutation[ds_idx]
            return self[ds_idx]
        else:
            raise StopIteration


class StandardCollator:

    def __init__(self, model):
        self.model = model

    def __call__(self, examples):
        max_x_length = 0
        for example in examples:
            x = example["input_ids"]
            if max_x_length < x.shape[0]:
                max_x_length = x.shape[0]

        pad_id = self.model.tokenizer_.pad_token_id

        use_tgts = "targets" in examples[0]

        X = torch.ones((len(examples), max_x_length), dtype=x.dtype) * pad_id  # device=self.model.device)*pad_id
        T = torch.zeros_like(X)
        P = torch.zeros_like(X)
        M = torch.zeros_like(X)
        if use_tgts:
            Y = torch.zeros((len(examples), 1), dtype=examples[0]["targets"].dtype)  # , device=self.model.device)
        O = [None] * len(examples)

        for idx, example in enumerate(examples):
            length = len(example["input_ids"])

            X[idx, :length] = example["input_ids"]
            T[idx, :length] = example["token_type_ids"]
            P[idx, :length] = example["position_ids"]
            M[idx, :length] = example["input_mask"]

            if use_tgts:
                Y[idx] = example["targets"]
            O[idx] = example["objects"] if "objects" in example else None

        ret = {"input_ids": X, "token_type_ids": T, "position_ids": P, "input_mask": M, "objects": O}
        if use_tgts:
            ret.update({"targets": Y})
        return ret
