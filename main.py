import pdb
from argparse import ArgumentParser
from tqdm import tqdm

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import pMHC
from pMHC import SEP
from pMHC.logic import PresentationPredictor
from pMHC.logic.backbones import TAPEBackbone
from pMHC.data import from_data, to_input

tqdm.pandas()

if __name__ == "__main__":
    print("START")

    parser = ArgumentParser()
    parser.add_argument('--do', type=str, default="train")
    parser.add_argument('--tqdm_disable', default=False, action="store_true")
    parser.add_argument('--project_directory', type=str, default=r"C:\Users\tux\Documents\MScProject")
    parser.add_argument('--save_every_n_steps', type=int, default=100000)
    parser = PresentationPredictor.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args_dict = vars(args)
    print(args_dict)

    pMHC.set_paths(args_dict["project_directory"])
    pMHC.set_tqdm_disabled(args_dict["tqdm_disable"])

    if args_dict["resume_from_checkpoint"] is not None:  # load model
        args_dict_ = {key: item for key, item in args_dict.items()
                     if key in ["output_attentions", "shuffle_data", "num_workers", "only_deconvolute"]}
        model = PresentationPredictor.load_from_checkpoint(checkpoint_path=args_dict["resume_from_checkpoint"], **args_dict_)
        model.only_deconvolute_ = True
    else:
        model = PresentationPredictor(**args_dict)

    if args_dict["do"] == "train":

        checkpoint_callback_epoch = ModelCheckpoint(
            every_n_val_epochs=1,
            save_top_k=-1,
            save_last=True)

        checkpoint_callback_step = ModelCheckpoint(
            every_n_train_steps=args_dict["save_every_n_steps"],
            save_top_k=-1,
            save_last=False)

        tb_logger = TensorBoardLogger(f'{pMHC.OUTPUT_FOLDER}{SEP}', name=model.name, version=model.version)

        trainer = Trainer.from_argparse_args(args, logger=tb_logger, callbacks=[checkpoint_callback_epoch, checkpoint_callback_step])

        trainer.fit(model)

    elif args_dict["do"] == "to_input":
        from_data()
        to_input()
        backbone = TAPEBackbone.from_pretrained('bert-base')
        backbone.save_pretrained(f"{pMHC.PROJECT_FOLDER}{SEP}tape_pretrained{SEP}")
