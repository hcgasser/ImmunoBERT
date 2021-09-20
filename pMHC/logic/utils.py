import pandas as pd
import yaml
import os
import json
import copy

from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score, \
    precision_recall_curve, PrecisionRecallDisplay, average_precision_score, \
    accuracy_score

from pytorch_lightning import Trainer

import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator

import pMHC
from pMHC import SEP, SPLIT_NAMES, SPLIT_VAL, SPLIT_VAL_MHC_ALLELES, SPLIT_VAL_PROTEINS, VIEW_NAMES, VIEW_SA, VIEW_SAMA
from pMHC.logic import PresentationPredictor
from pMHC.data.example import Observation


def load_latest(version_, **kwargs):
    checkpoint = "last"

    MODEL_PATH = f"{pMHC.OUTPUT_FOLDER}{SEP}lightning_logs{SEP}{version_}{SEP}checkpoints{SEP}{checkpoint}.ckpt"
    return PresentationPredictor.load_from_checkpoint(MODEL_PATH, num_workers=0, **kwargs)


def list_versions(folder):
    models = []
    for dirname, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.split(".")[0] == "events" and filename.split(".")[1] == "out":
                version = dirname.split(SEP)[-1]

                event_file_url = os.path.join(dirname, filename)
                ea = event_accumulator.EventAccumulator(event_file_url,
                                                        size_guidance={
                                                            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                            event_accumulator.IMAGES: 4,
                                                            event_accumulator.AUDIO: 4,
                                                            event_accumulator.SCALARS: 0,
                                                            event_accumulator.HISTOGRAMS: 1,
                                                        })
                ea.Reload()

                hyp = {}
                with open(os.path.join(dirname, "hparams.yaml")) as file:
                    hyp = yaml.load(file, Loader=yaml.FullLoader)

                last_val_Recall = ea.Scalars("val_Recall")[-1].value if "val_Recall" in ea.Tags()["scalars"] else 0
                last_val_Precision = ea.Scalars("val_Precision")[-1].value if "val_Precision" in ea.Tags()[
                    "scalars"] else 0
                last_val_Accuracy = ea.Scalars("val_Accuracy")[-1].value if "val_Accuracy" in ea.Tags()[
                    "scalars"] else 0
                models.append((version, hyp, last_val_Recall, last_val_Precision, last_val_Accuracy))

    print(f"{'Version':<40s}: {'datasources':>12s} {'decoys':>6s} {'head':>4s} {'prop':>4s} {'LR':>8s} {'v_recall':>8s} {'v_prec':>8s} {'v_acc':>8s} {'F1':>8s}")
    for version, hyp, last_val_Recall, last_val_Precision, last_val_Accuracy in models:
        datasources = hyp['datasources'] if 'datasources' in hyp else ""
        decoys_per_obs = str(hyp['decoys_per_obs']) if 'decoys_per_obs' in hyp else ""
        head = hyp['head'] if 'head' in hyp else ""
        proportion = str(hyp['proportion']) if 'proportion' in hyp else ""
        learning_rate = hyp['learning_rate'] if 'learning_rate' in hyp else ""

        last_val_F1 = 2 * (last_val_Precision*last_val_Recall) / (last_val_Precision + last_val_Recall + 1e-5)
        print(f"{version:<40s}: {datasources:>12s} {decoys_per_obs:>6s} {head:>4s} {proportion:>4s} {learning_rate:>8.2e} "
              + f"{last_val_Recall:>8.4f} {last_val_Precision:>8.4f} {last_val_Accuracy:>8.4f} {last_val_F1:>8.4f}")


def validate_checkpoints(to_assess):
    template_level_3 = {"Accuracy": -1, "Precision": -1, "Recall": -1, "tp": -1, "tn": -1, "fp": -1, "fn": -1}
    template_val = {"Proportion": 0,
                    SPLIT_VAL: {
                        VIEW_SA: copy.deepcopy(template_level_3), VIEW_SAMA: copy.deepcopy(template_level_3)},
                    SPLIT_VAL_MHC_ALLELES: {
                        VIEW_SA: copy.deepcopy(template_level_3), VIEW_SAMA: copy.deepcopy(template_level_3)},
                    SPLIT_VAL_PROTEINS: {
                        VIEW_SA: copy.deepcopy(template_level_3), VIEW_SAMA: copy.deepcopy(template_level_3)}}

    prev_proportion = 0
    val = copy.deepcopy(template_val)
    for model_name, model_version, model_checkpoint, proportion in to_assess:
        filename = f"{pMHC.OUTPUT_FOLDER}{SEP}{model_name}{SEP}{model_version}{SEP}checkpoints{SEP}{model_checkpoint}"
        print(f"Validate: {filename}")
        model = PresentationPredictor.load_from_checkpoint(
            f"{filename}.ckpt", num_workers=0, output_attentions=False, only_deconvolute=False, shuffle_data=False)
        model.decoys_per_obs = 99
        if proportion is not None:
            model.proportion = proportion
        else:
            proportion = model.proportion
        if prev_proportion == proportion:
            model.data_loaded = True
        prev_proportion = proportion
        val["Proportion"] = proportion

        trainer = Trainer(default_root_dir=f'{pMHC.OUTPUT_FOLDER}{SEP}',
                          gpus=1,
                          resume_from_checkpoint=f"{filename}.ckpt",
                          checkpoint_callback=False, logger=False)

        model.only_deconvolute = True
        model.only_deconvolute_ = True

        trainer.validate(model, model.dl[SPLIT_VAL])

        for split, assessments in val.items():
            if split != "Proportion":
                for view, metrics in assessments.items():
                    model.view = view
                    model.only_deconvolute_ = False
                    result = trainer.test(model, model.dl[split])

                    result = result[0]

                    for idx, k in enumerate(template_level_3.keys()):
                        val[split][view][k] = result[f"test_{k}"]

        json_object = json.dumps(val, indent=4)
        with open(f"{filename}_val.json", "w") as outfile:
            outfile.write(json_object)

        val = copy.deepcopy(template_val)


def predict_checkpoints(to_assess, split, view):
    prev_proportion = 0
    for model_name, model_version, model_checkpoint, proportion in to_assess:
        filename = f"{pMHC.OUTPUT_FOLDER}{SEP}{model_name}{SEP}{model_version}{SEP}checkpoints{SEP}{model_checkpoint}"
        print(f"Validate: {filename}")
        model = PresentationPredictor.load_from_checkpoint(
            f"{filename}.ckpt", num_workers=0, output_attentions=False, only_deconvolute=False, shuffle_data=False)
        model.decoys_per_obs = 99
        if proportion is not None:
            model.proportion = proportion
        else:
            proportion = model.proportion
        if prev_proportion == proportion:
            model.data_loaded = True
        prev_proportion = proportion

        trainer = Trainer(default_root_dir=f'{pMHC.OUTPUT_FOLDER}{SEP}',
                          gpus=1,
                          resume_from_checkpoint=f"{filename}.ckpt",
                          checkpoint_callback=False, logger=False)

        model.setup()  # needs to be called here, otherwise if .test is called, the dataloader would not exist yet

        if view != VIEW_SA:
            model.only_deconvolute = True
            model.only_deconvolute_ = True

            trainer.validate(model, model.dl[SPLIT_VAL])

        model.view = view
        model.only_deconvolute_ = False
        model.save_predictions = {}
        model.test_step_split = split
        result = trainer.test(model, model.dl[split])

        keys = []
        targets = []
        predictions = []
        for key, item in model.save_predictions.items():
            keys.append(key)
            targets.append(item[0])
            predictions.append(item[1])

        df = pd.DataFrame({"keys": keys, "targets": targets, "predictions": predictions})
        df.to_csv(f"{filename}_{SPLIT_NAMES[split]}_{VIEW_NAMES[view]}_{proportion}_preds.csv", sep=",", index=False)


def load_checkpoint_predictions(split, view, model_name, model_version, model_checkpoint, proportion):
    filename = f"{pMHC.OUTPUT_FOLDER}{SEP}{model_name}{SEP}{model_version}{SEP}checkpoints{SEP}{model_checkpoint}_"
    filename += f"{SPLIT_NAMES[split].split('-')[0]}_{VIEW_NAMES[view]}_{proportion}_preds.csv"
    df = pd.read_csv(filename)
    df["key_obs"] = df["keys"].apply(lambda x: x.split("_")[0]).astype("int64")
    df["decoy_idx"] = df["keys"].apply(lambda x: x.split("_")[1]).astype("int64")

    df_selection = pd.DataFrame({"key_obs": [obs.key for obs in Observation.obs_views[split][view]]})
    df = pd.merge(df, df_selection, how="inner", on="key_obs")

    return list(df["keys"]), list(df["targets"]), list(df["predictions"])


def evaluate_checkpoints(checkpoints, splits, views):
    info = {}
    for model_name, model_version, model_checkpoint, proportion in checkpoints:
        checkpoint_id = f"{model_name}_{model_version}_{model_checkpoint}_{proportion}"
        info.update({checkpoint_id: {}})
        for split in splits:  # [SPLIT_VAL_PROTEINS, SPLIT_VAL_MHC_ALLELES]
            info[checkpoint_id].update({f"{SPLIT_NAMES[split]}":  {}})
            for view in views: # [VIEW_SA, VIEW_SAMA]:
                print(f"Load: {checkpoint_id} - {SPLIT_NAMES[split]} - {VIEW_NAMES[view]}")
                keys, targets, predictions = load_checkpoint_predictions(
                    split, view, model_name, model_version, model_checkpoint, proportion)

                curve_fpr, curve_tpr, _ = roc_curve(targets, predictions)
                precision, recall, _ = precision_recall_curve(targets, predictions)
                accuracy = accuracy_score(targets, [1 if x > 0.5 else 0 for x in predictions])

                info[checkpoint_id][SPLIT_NAMES[split]][VIEW_NAMES[view]] = {
                    "accuracy": accuracy,
                    "curve_fpr": curve_fpr,
                    "curve_tpr": curve_tpr,
                    "roc_auc": roc_auc_score(targets, predictions),
                    "precision": precision,
                    "recall": recall,
                    "AP": average_precision_score(targets, predictions)
                }

    return info


