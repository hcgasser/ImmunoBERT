import sys
from datetime import datetime
import random
import pdb

from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything

from pMHC import logger, PACKAGE_NAME, POSITIVE_THRESHOLD, \
    VIEWS, VIEW_NAMES, VIEW_SA, VIEW_SAMA, VIEW_DECONV, \
    SPLITS, SPLIT_NAMES, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_VAL_PROTEINS, SPLIT_TEST_PROTEINS, \
    SPLIT_VAL_MHC_ALLELES, SPLIT_TEST_MHC_ALLELES
from pMHC.data import setup_system

import pMHC.logic.backbones
import pMHC.logic.heads
import pMHC.logic.tokenizers
from pMHC.data.datasources import Datasource
from pMHC.data.datasets import StandardDataset
from pMHC.data.example import Observation
from pMHC.data.mhc_allele import MhcAllele
from pMHC.data.protein import Protein
from pMHC.data import from_input
from pMHC.data.exploratory import overview


class PresentationPredictor(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        logger.info("PresentationPredictor: __init__")

        # identification
        self.name = kwargs["name"]
        self.version = kwargs["version"]

        # Backbone
        self.backbone = kwargs["backbone"]
        self.backbone_ = getattr(sys.modules[f'{PACKAGE_NAME}.logic.backbones'],
                                 f"{self.backbone}Backbone").get_backbone(kwargs["output_attentions"])
        h = {f"backbone_{key}": item for key, item in self.backbone_.configuration().items()}
        h.update(kwargs)

        # Head
        self.head = kwargs["head"] if kwargs["head"] != "Classification" else "Cls"
        self.head_ = getattr(sys.modules[f'{PACKAGE_NAME}.logic.heads'], f"{self.backbone}{self.head}Head")(**h)

        # Tokenizer
        self.tokenizer = kwargs["tokenizer"]
        self.tokenizer_ = getattr(sys.modules[f'{PACKAGE_NAME}.logic.tokenizers'], f"{self.tokenizer}Tokenizer")()
        self.tokenizer_.seq_length = kwargs["max_seq_length"]

        # input modes
        self.input_mode = getattr(sys.modules["pMHC"], "INPUT_" + kwargs["input_mode"])
        self.mhc_rep = getattr(sys.modules["pMHC"], "MHC_" + kwargs["mhc_rep"])
        self.max_seq_length = kwargs["max_seq_length"]

        # data parameters
        self.datasources = kwargs["datasources"]
        for source in self.datasources.split(";"):
            getattr(sys.modules["pMHC.data.datasources"], f"{source}Datasource")()
        self.decoys_per_obs = kwargs["decoys_per_obs"]
        self.proportion = kwargs["proportion"]
        self.shuffle_data = kwargs["shuffle_data"]
        self.seed = kwargs["seed"]
        seed_everything(self.seed)

        # training parameters
        self.batch_size = kwargs["batch_size"]
        self.learning_rate = kwargs["learning_rate"]

        # technical parameters
        self.num_workers = kwargs["num_workers"]
        self.precision = kwargs["precision"]

        # control variables
        self.view = VIEW_SA

        # datasets and dataloaders
        self.ds = [None, None, None, None, None, None, None]
        self.dl = [None, None, None, None, None, None, None]

        # metrics
        metrics_tmp = MetricCollection(self.head_.get_metrics())
        self.metrics = nn.ModuleList([
            metrics_tmp.clone(prefix=f'{SPLIT_NAMES[SPLIT_TRAIN]}_'),
            metrics_tmp.clone(prefix=f'{SPLIT_NAMES[SPLIT_VAL]}_'),
            metrics_tmp.clone(prefix=f'{SPLIT_NAMES[SPLIT_TEST]}_'),
            metrics_tmp.clone(prefix=f'{SPLIT_NAMES[SPLIT_VAL_PROTEINS]}_'),
            metrics_tmp.clone(prefix=f'{SPLIT_NAMES[SPLIT_TEST_PROTEINS]}_'),
            metrics_tmp.clone(prefix=f'{SPLIT_NAMES[SPLIT_VAL_MHC_ALLELES]}_'),
            metrics_tmp.clone(prefix=f'{SPLIT_NAMES[SPLIT_TEST_MHC_ALLELES]}_')
        ])

        self.data_loaded = False
        self.only_SA = kwargs["only_SA"] if "only_SA" in kwargs else False

        # whether a validation epoch should be run during validation
        self.only_deconvolute = kwargs["only_deconvolute"] if "only_deconvolute" in kwargs else False
        self.only_deconvolute_ = self.only_deconvolute

        self.test_step_split = SPLIT_TEST   # the split that gets used during evaluation (testing)
        self.save_predictions = None   # None if test predictions are not saved - otherwise a dictionary {{obs_nr}_{decoy_idx} : (target, prediction, )}
        self.idx = 0

        self.save_hyperparameters('name', 'version',
                                  'backbone', 'head', 'head_hidden_features', 'tokenizer',
                                  'input_mode', 'mhc_rep', 'max_seq_length',
                                  'datasources', 'proportion', 'decoys_per_obs', 'seed',
                                  'batch_size', 'learning_rate',
                                  'precision'
                                  )

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("PresentationPredictor")

        # identification parameters
        parser.add_argument('--name', type=str, default="model")
        parser.add_argument('--version', type=str, default="version")

        # architecture parameters
        parser.add_argument('--backbone', type=str, default="TAPE",
                            help="specify the backbone. Only 'TAPE' implemented")
        parser.add_argument('--head', type=str, default="Cls", help="specify the used head. Values: Cls, Avg")
        parser.add_argument('--head_hidden_features', type=int, default=512,
                            help="the hidden dimension used in the head. Int Values > 0")
        parser.add_argument('--tokenizer', type=str, default="Standard",
                            help="which tokenizer to use. Only 'Standard' implemented")

        # input modes
        parser.add_argument('--input_mode', type=str, help="Values: PEPTIDE or CONTEXT", default="PEPTIDE")
        parser.add_argument('--mhc_rep', type=str, help="Values: PSEUDO or FULL", default="PSEUDO")
        parser.add_argument('--max_seq_length', type=int, default=260)

        # data parameters
        parser.add_argument('--datasources', type=str, default="IEDB")
        parser.add_argument('--decoys_per_obs', type=int, default=1)
        parser.add_argument('--proportion', type=float, default=1.0,
                            help="which proportion of the dataset to use")
        parser.add_argument('--shuffle_data', default=False, action="store_true",
                            help="if set, the data gets shuffled for each training run")
        parser.add_argument('--seed', type=int, default=42)

        # training parameters
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--learning_rate', type=float, default=0.00001)

        # technical parameters
        parser.add_argument('--num_workers', type=int, default=0)

        # transient parameter
        parser.add_argument('--output_attentions', default=False, action="store_true")
        parser.add_argument('--only_deconvolute', default=False, action="store_true")
        parser.add_argument('--only_SA', default=False, action="store_true")

        return parent_parser

    def setup(self, stage: Optional[str] = None) -> None:
        print(f"PresentationPredictor.setup: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("PresentationPredictor: setup")
        seed_everything(self.seed)

        if not self.data_loaded:
            setup_system(self.proportion, self.decoys_per_obs)
            self.data_loaded = True

        seed_everything(self.seed)

        for split in SPLITS:
            self.ds[split] = StandardDataset(split, self)
            self.dl[split] = DataLoader(self.ds[split],
                                        shuffle=False, num_workers=self.num_workers,
                                        batch_size=self.batch_size,
                                        collate_fn=lambda x: self.ds[split].collator(x))

        info = f"PresentationPredictor.setup finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(info)
        logger.info(info)
        overview(self)

    def train_dataloader(self):
        print("PresentationPredictor: train_dataloader")
        logger.info("PresentationPredictor: train_dataloader")
        return self.dl[SPLIT_TRAIN]

    def val_dataloader(self):
        print("PresentationPredictor: val_dataloader")
        logger.info("PresentationPredictor: val_dataloader")
        return self.dl[SPLIT_VAL]

    def test_dataloader(self):
        print("PresentationPredictor: test_dataloader")
        logger.info("PresentationPredictor: test_dataloader")
        return self.dl[SPLIT_TEST]

    def forward(self, x):
        x = self.backbone_(input_ids=x["input_ids"],
                           token_type_ids=x["token_type_ids"],
                           position_ids=x["position_ids"],
                           input_mask=x["input_mask"])
        x = self.head_(x)
        return x

    def step(self, batch, batch_idx, split):
        logits = self(batch)

        loss = self.head_.loss(logits, batch["targets"])
        self.log(f'{SPLIT_NAMES[split]}_loss', loss, prog_bar=True)

        y_hat = torch.sigmoid(logits)
        self.metrics[split](y_hat, batch["targets"].to(dtype=torch.short))
        # self.log_dict(metrics, prog_bar=False)

        if self.save_predictions is not None:
            for idx, (obs, rel_mhc, decoy_idx) in enumerate(batch["objects"]):
                # pdb.set_trace()
                self.save_predictions[f"{obs.key}_{decoy_idx}"] = (float(batch["targets"][idx]), float(y_hat[idx]))

        if self.idx % 1000 == 0:
            accuracy = self.metrics[split]['Accuracy']
            accumulated = {"tp": accuracy.tp, "tn": accuracy.tn, "fp": accuracy.fp, "fn": accuracy.fn}
            print(f"   {SPLIT_NAMES[split]}: tp... {accumulated['tp']}, tn... {accumulated['tn']}, fp... {accumulated['fp']}, fn... {accumulated['fn']}")
        self.idx += 1

        return loss

    def epoch_end(self, split):
        if split != SPLIT_VAL or not self.only_deconvolute_:
            accuracy = self.metrics[split]['Accuracy']
            accumulated = {"tp": accuracy.tp, "tn": accuracy.tn, "fp": accuracy.fp, "fn": accuracy.fn}
            for key, value in accumulated.items():
                self.log(f'{SPLIT_NAMES[split]}_{key}', value, prog_bar=False)
            self.log_dict(self.metrics[split].compute(), prog_bar=True)
            self.metrics[split].reset()
            print(f"   {SPLIT_NAMES[split]}: tp... {accumulated['tp']}, tn... {accumulated['tn']}, fp... {accumulated['fp']}, fn... {accumulated['fn']}")

    def on_train_epoch_start(self) -> None:
        if self.global_step == 0 and self.current_epoch == 0:
            self.view = VIEW_SA
        else:
            if self.only_SA:
                self.view = VIEW_SA
            else:
                self.view = VIEW_SAMA
            self.only_deconvolute_ = self.only_deconvolute

        print(f"PresentationPredictor: on_train_epoch_start - {VIEW_NAMES[self.view]} Epoch: {self.current_epoch} Global step: {self.global_step}")

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, SPLIT_TRAIN)

    def training_epoch_end(self, outputs) -> None:
        print(f"PresentationPredictor: train_epoch_end - {VIEW_NAMES[self.view]} ")
        self.epoch_end(SPLIT_TRAIN)

    def on_validation_epoch_start(self) -> None:
        print("MyTransformer: on_validation_epoch_start")
        logger.info("MyTransformer: on_validation_epoch_start")
        for obs, _ in (Observation.obs_views[SPLIT_TRAIN][VIEW_DECONV]
                       + Observation.obs_views[SPLIT_VAL][VIEW_DECONV]
                       + Observation.obs_views[SPLIT_TEST][VIEW_DECONV]):
            obs.max_prediction = 0
        self.view = VIEW_DECONV

    def validation_step(self, batch, batch_idx) -> Optional:
        if self.view != VIEW_DECONV:
            return self.step(batch, batch_idx, SPLIT_VAL)
        else:
            logits = self(batch)
            y_hat = torch.sigmoid(logits)

            for i, (obs, rel_mhc, example) in enumerate(batch["objects"]):
                obs_prediction = float(y_hat[i])

                if obs.target > POSITIVE_THRESHOLD:
                    if obs.max_prediction < obs_prediction:
                        obs.max_prediction = obs_prediction
                        obs.deconvoluted_mhc_allele = rel_mhc
                else:
                    obs.deconvoluted_mhc_allele = random.choice(obs.mhc_alleles)

    def validation_epoch_end(self, outputs) -> None:
        print(f"PresentationPredictor: validation_epoch_end - {VIEW_NAMES[self.view]} ")
        self.epoch_end(SPLIT_VAL)

    def on_epoch_end(self):
        if self.logger:
            self.logger.save()  # ensure logger gets written down to disk

    def test_step(self, batch, batch_idx) -> Optional:
        return self.step(batch, batch_idx, self.test_step_split)

    def test_epoch_end(self, outputs) -> None:
        print(f"PresentationPredictor: test_epoch_end - {VIEW_NAMES[self.view]} ")
        self.epoch_end(self.test_step_split)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
