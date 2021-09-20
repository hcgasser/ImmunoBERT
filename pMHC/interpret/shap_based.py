import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import shap

import seaborn as sns
import matplotlib.pyplot as plt

from pytorch_lightning.utilities.seed import seed_everything

import pMHC
from pMHC import FLANK_LEN, MHC_PSEUDO, TT_N_FLANK, TT_PEPTIDE, TT_C_FLANK, TT_MHC, MAX_PEPTIDE_LEN
from pMHC.data.utils import pseudo_pos, convert_example_to_batch, convert_examples_to_batch, move_dict_to_device, \
    get_input_rep_PSEUDO
from pMHC.data.example import Observation, Decoy
from pMHC.data.mhc_allele import MhcAllele


def array_position(token_type_id, position_id):
    position = -1
    if pMHC.SHAP_ARRAY_FLANK_LEN > 0 and token_type_id == TT_N_FLANK and position_id > 0:
        position = position_id - 1
    elif pMHC.SHAP_ARRAY_PEPTIDE_LEN > 0 and token_type_id == TT_PEPTIDE and position_id > 0:
        position = pMHC.SHAP_ARRAY_FLANK_LEN + position_id - 1
    elif pMHC.SHAP_ARRAY_FLANK_LEN > 0 and token_type_id == TT_C_FLANK and position_id > 0:
        position = pMHC.SHAP_ARRAY_FLANK_LEN + pMHC.SHAP_ARRAY_PEPTIDE_LEN + position_id - 1
    elif pMHC.SHAP_ARRAY_MHC_LEN > 0 and token_type_id == TT_MHC and position_id > 0:
        p = pseudo_pos.index(position_id)
        position = 2 * pMHC.SHAP_ARRAY_FLANK_LEN + pMHC.SHAP_ARRAY_PEPTIDE_LEN + p

    return position


def example_to_array(example):
    length = 2*pMHC.SHAP_ARRAY_FLANK_LEN + pMHC.SHAP_ARRAY_PEPTIDE_LEN + pMHC.SHAP_ARRAY_MHC_LEN

    x = np.zeros((length,))
    x.fill(np.nan)
    for input_id, token_type_id, position_id, input_mask in \
            zip(example["input_ids"], example["token_type_ids"],
                example["position_ids"], example["input_mask"]):

        position = array_position(token_type_id, position_id)
        if position >= 0:
            x[position] = input_id

    return x


def array_to_example(array, model):
    tokenizer = model.tokenizer_

    array_tt_ids = [TT_N_FLANK] * pMHC.SHAP_ARRAY_FLANK_LEN \
                   + [TT_PEPTIDE] * pMHC.SHAP_ARRAY_PEPTIDE_LEN \
                   + [TT_C_FLANK] * pMHC.SHAP_ARRAY_FLANK_LEN \
                   + [TT_MHC] * pMHC.SHAP_ARRAY_MHC_LEN

    assert(len(array_tt_ids) == len(array))

    n_flank_input_ids = []
    n_flank_position_ids = []
    n_flank_input_mask = []
    peptide_input_ids = []
    peptide_position_ids = []
    peptide_input_mask = []
    c_flank_input_ids = []
    c_flank_position_ids = []
    c_flank_input_mask = []
    mhc_input_ids = []
    mhc_position_ids = []
    mhc_input_mask = []

    for idx, input_id in enumerate(array):
        if array_tt_ids[idx] == TT_N_FLANK:
            position_id = idx + 1
            n_flank_input_ids = ([input_id] if not np.isnan(input_id) else [tokenizer.unk_token_id]) + n_flank_input_ids
            n_flank_position_ids = [position_id] + n_flank_position_ids
            n_flank_input_mask = [1] + n_flank_input_mask if not np.isnan(input_id) else [0] + n_flank_input_mask
        elif array_tt_ids[idx] == TT_PEPTIDE:
            position_id = idx - pMHC.SHAP_ARRAY_FLANK_LEN + 1
            peptide_input_ids = peptide_input_ids + ([input_id] if not np.isnan(input_id) else [tokenizer.unk_token_id])
            peptide_position_ids = peptide_position_ids + [position_id]
            peptide_input_mask = peptide_input_mask + [1] if not np.isnan(input_id) else peptide_input_mask + [0]
        elif array_tt_ids[idx] == TT_C_FLANK:
            position_id = idx - pMHC.SHAP_ARRAY_FLANK_LEN - pMHC.SHAP_ARRAY_PEPTIDE_LEN + 1
            c_flank_input_ids = c_flank_input_ids + ([input_id] if not np.isnan(input_id) else [tokenizer.unk_token_id])
            c_flank_position_ids = c_flank_position_ids + [position_id]
            c_flank_input_mask = c_flank_input_mask + [1] if not np.isnan(input_id) else c_flank_input_mask + [0]
        elif array_tt_ids[idx] == TT_MHC:
            position_id = idx - 2 * pMHC.SHAP_ARRAY_FLANK_LEN - pMHC.SHAP_ARRAY_PEPTIDE_LEN + 1 \
                if model.mhc_rep != MHC_PSEUDO \
                else pseudo_pos[idx - 2 * pMHC.SHAP_ARRAY_FLANK_LEN - pMHC.SHAP_ARRAY_PEPTIDE_LEN]
            mhc_input_ids = mhc_input_ids + ([input_id] if not np.isnan(input_id) else [tokenizer.unk_token_id])
            mhc_position_ids = mhc_position_ids + [position_id]
            mhc_input_mask = mhc_input_mask + [1] if not np.isnan(input_id) else mhc_input_mask + [0]

    input_ids = [tokenizer.start_token_id]
    token_type_ids = [TT_PEPTIDE]
    position_ids = [0]
    input_mask = [1]

    # N-Flank
    if pMHC.SHAP_ARRAY_FLANK_LEN > 0:
        input_ids += (n_flank_input_ids + [tokenizer.stop_token_id])
        token_type_ids += ([TT_N_FLANK] * len(n_flank_input_ids) + [TT_N_FLANK])
        position_ids += (n_flank_position_ids + [0])
        input_mask += (n_flank_input_mask + [1])
    elif pMHC.SHAP_ARRAY_N_FLANK_STD is not None:
        input_ids += (pMHC.SHAP_ARRAY_N_FLANK_STD + [tokenizer.stop_token_id])
        token_type_ids += ([TT_N_FLANK] * len(pMHC.SHAP_ARRAY_N_FLANK_STD) + [TT_N_FLANK])
        position_ids += (list(range(len(pMHC.SHAP_ARRAY_N_FLANK_STD), 0, -1)) + [0])
        input_mask += ([1]*len(pMHC.SHAP_ARRAY_N_FLANK_STD) + [1])

    # peptide
    input_ids += (peptide_input_ids + [tokenizer.stop_token_id])
    token_type_ids += ([TT_PEPTIDE] * len(peptide_input_ids) + [TT_PEPTIDE])
    position_ids += (peptide_position_ids + [0])
    input_mask += (peptide_input_mask + [1])

    # C-Flank
    if pMHC.SHAP_ARRAY_FLANK_LEN > 0:
        input_ids += (c_flank_input_ids + [tokenizer.stop_token_id])
        token_type_ids += ([TT_C_FLANK] * len(c_flank_input_ids) + [TT_C_FLANK])
        position_ids += (c_flank_position_ids + [0])
        input_mask += (c_flank_input_mask + [1])
    elif pMHC.SHAP_ARRAY_C_FLANK_STD is not None:
        input_ids += (pMHC.SHAP_ARRAY_C_FLANK_STD + [tokenizer.stop_token_id])
        token_type_ids += ([TT_C_FLANK] * len(pMHC.SHAP_ARRAY_C_FLANK_STD) + [TT_C_FLANK])
        position_ids += (list(range(1, len(pMHC.SHAP_ARRAY_C_FLANK_STD)+1)) + [0])
        input_mask += ([1]*len(pMHC.SHAP_ARRAY_C_FLANK_STD) + [1])

    # MHC
    if pMHC.SHAP_ARRAY_MHC_LEN > 0:
        input_ids += (mhc_input_ids + [tokenizer.stop_token_id])
        token_type_ids += ([TT_MHC] * len(mhc_input_ids) + [TT_MHC])
        position_ids += (mhc_position_ids + [0])
        input_mask += (mhc_input_mask + [1])
    elif pMHC.SHAP_ARRAY_MHC_STD is not None:
        input_ids += (pMHC.SHAP_ARRAY_MHC_STD + [tokenizer.stop_token_id])
        token_type_ids += ([TT_MHC] * len(pMHC.SHAP_ARRAY_MHC_STD) + [TT_MHC])
        if model.mhc_rep == MHC_PSEUDO:
            mhc_position_ids = pseudo_pos
        else:
            mhc_position_ids = list(range(1, len(pMHC.SHAP_ARRAY_MHC_STD)+1))
        position_ids += (mhc_position_ids + [0])
        input_mask += ([1] * len(pMHC.SHAP_ARRAY_MHC_STD) + [1])

    # pdb.set_trace()
    return {
        'input_ids': torch.tensor(np.array(input_ids).astype(int), device=model.device),
        'token_type_ids': torch.tensor(np.array(token_type_ids).astype(int), device=model.device),
        'position_ids': torch.tensor(np.array(position_ids).astype(int), device=model.device),
        'input_mask': torch.tensor(np.array(input_mask).astype(int), device=model.device)
    }


def shap_predict(x, model):
    """ organizes the batched processing of the variants
     converts the array into a format to be input into the transformer """

    examples = []
    preds = []
    for i in range(x.shape[0]):
        examples.append(array_to_example(x[i, :], model))
        # print(examples)
        if len(examples) >= model.batch_size:
            batch = convert_examples_to_batch(examples)
            move_dict_to_device(batch, model)
            y_hat = torch.sigmoid(model(batch).detach().cpu())
            # print(y_hat)
            preds += [x[0] for x in y_hat]
            examples = []

    if len(examples) >= 1:
        y_hat = torch.sigmoid(model(convert_examples_to_batch(examples)).detach().cpu())
        # print(y_hat)
        preds += [x[0] for x in y_hat]
        examples = []

    return np.array(preds)


def shap_get_list(example, shap_values, model):
    tokenizer = model.tokenizer_
    importances = []
    array = example_to_array(example)
    for idx, shap_value in enumerate(shap_values):
        if idx < FLANK_LEN:
            element = "N-flank"
            position = idx + 1
        elif idx < FLANK_LEN + MAX_PEPTIDE_LEN:
            element = "peptide"
            position = idx - FLANK_LEN + 1
        elif idx < 2 * FLANK_LEN + MAX_PEPTIDE_LEN:
            element = "C-flank"
            position = idx - FLANK_LEN - MAX_PEPTIDE_LEN + 1
        else:
            element = "MHC"
            position = pseudo_pos[idx - 2 * FLANK_LEN - MAX_PEPTIDE_LEN]

        amino_acid = tokenizer._convert_id_to_token(array[idx]) if not np.isnan(array[idx]) else ""

        importances.append((f"{amino_acid}_{element}_{position}", shap_value))
    return importances


def shap_analyse_mhc_allele(mhc_allele_name, model, len_peptide, num_obs=25, decoys_per_obs=9, nsamples=128, nbackground=128, seed=42):
    # filter the observations that are single-allele, that
    # have the right length and come from the mhc allele
    model.eval()

    observations = []
    for obs in Observation.observations:
        if len(obs.sample.mhc_alleles) == 1 \
                and obs.mhc_allele.name == mhc_allele_name \
                and len(obs.peptide_seq) == len_peptide:
            observations.append(obs)

    # select a random subset of observations from the mhc allele
    seed_everything(seed)
    obs_selection = np.random.choice(observations, num_obs)

    # add the decoys to the example list
    example_selection = []
    peptide_selection = []
    for obs in obs_selection:
        example_selection.append(obs)
        peptide_selection.append(obs.peptide_seq)
        for decoy_idx in range(decoys_per_obs):
            decoy = Decoy.get_decoy(obs.key * 1000 + decoy_idx)
            example_selection.append(decoy)
            peptide_selection.append(decoy.peptide_seq)

    # generate the background distribution
    background_peptide_arrays = []
    for background_peptide in peptide_selection:
        background_peptide_arrays.append(model.tokenizer_(background_peptide)['input_ids'])
    background_peptide_array = np.stack(background_peptide_arrays, axis=0)
    background_peptide_array = shap.sample(background_peptide_array, nbackground)

    pseudo_seq = MhcAllele.mhc_alleles[mhc_allele_name].pseudo_seq

    # set the global variables for the SHAP analysis (get used in 'shap_predict')
    pMHC.SHAP_ARRAY_FLANK_LEN = 0  # disables the flanks in the SHAP features
    pMHC.SHAP_ARRAY_N_FLANK_STD = None  # disables standard n-flank. alt: model.tokenizer_(obs.n_flank)['input_ids']
    pMHC.SHAP_ARRAY_C_FLANK_STD = None  # disables standard c-flank. alt: model.tokenizer_(obs.c_flank)['input_ids']
    pMHC.SHAP_ARRAY_PEPTIDE_LEN = len_peptide  # sets the length of the peptide SHAP features
    pMHC.SHAP_ARRAY_MHC_LEN = 0  # disables mhc in the SHAP features
    pMHC.SHAP_ARRAY_MHC_STD = model.tokenizer_(pseudo_seq)['input_ids']  # sets the current mhc as the standard

    orig_pred_list = []
    orig_pred_wo_flanks_list = []
    peptides_list = []
    shap_values_list = []
    with torch.no_grad():
        for example in tqdm(example_selection):
            example_tokens = get_input_rep_PSEUDO(example.n_flank, example.peptide_seq, example.c_flank, pseudo_seq, model)
            orig_pred = float(torch.sigmoid(
                model(move_dict_to_device(convert_example_to_batch(example_tokens), model))).detach().cpu())

            example_tokens = get_input_rep_PSEUDO("", example.peptide_seq, "", pseudo_seq, model)
            orig_pred_wo_flanks = float(torch.sigmoid(
                model(move_dict_to_device(convert_example_to_batch(example_tokens), model))).detach().cpu())

            array_to_explain = example_to_array(example_tokens).astype(int)

            # calculate the shapley values for a single example
            shap_explainer = shap.KernelExplainer(lambda x: shap_predict(x, model), background_peptide_array)
            shap_values = shap_explainer.shap_values(array_to_explain, nsamples=nsamples)

            orig_pred_list.append(orig_pred)
            orig_pred_wo_flanks_list.append(orig_pred_wo_flanks)
            peptides_list.append(example.peptide_seq)
            shap_values_list.append(shap_values)

    return orig_pred_list, orig_pred_wo_flanks_list, peptides_list, shap_values_list


def plot_AA_pos(list_peptides, list_shap_values, filename):
    list_aa = []
    list_pos = []
    list_value = []

    for peptide, shap_values in zip(list_peptides, list_shap_values):
        shap_values = [
            float(x) for x \
            in shap_values.replace("[", "").replace("]",  "").replace("\n",  "").split(" ") \
            if x != ""]
        for pos, (aa, shap_value) in enumerate(zip(peptide, shap_values)):
            list_aa.append(aa)
            list_pos.append(pos + 1)
            list_value.append(shap_value)

    df_SHAP = pd.DataFrame({"amino acid": list_aa, "peptide position": list_pos, "value": list_value})
    df_SHAP_mean = df_SHAP.groupby(["amino acid", "peptide position"]).mean()
    df_SHAP_count = df_SHAP.groupby(["amino acid", "peptide position"]).count()
    df_SHAP_total = pd.merge(df_SHAP_count, df_SHAP_mean, how="outer", on=["amino acid", "peptide position"])

    df_SHAP_mean = df_SHAP_mean.reset_index().pivot("amino acid", "peptide position", "value")

    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    ax = fig.add_subplot(111)

    sns.set(font_scale=2)
    hm = sns.heatmap(df_SHAP_mean*100, cmap="YlGnBu", linewidths=.5,
                annot=True, fmt=".0f", annot_kws={"size": 18},
                cbar=False)
    hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize=18)

    fig.tight_layout()
    plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches='tight')
    plt.show()

    df_SHAP_mean.to_csv(f"{filename}_mean.csv", sep=",", index=True)
    df_SHAP_count.to_csv(f"{filename}_count.csv", sep=",", index=True)
    df_SHAP_total.to_csv(f"{filename}_count_mean.csv", sep=",", index=True)


def shap_plot_change(mhc_allele_name, checkpoint, position, df_hits_decoys):
    df = df_hits_decoys[df_hits_decoys["peptide position"] == position]

    AAs = []
    hits_mean = []
    decoys_mean = []
    hits_prop = []
    decoys_prop = []
    for aa in sorted(hydrophobic_aa + hydrophilic_aa):
        hit_SHAP_mean = df[(df["amino acid"] == aa) & (df["example type"] == "hit")]["mean SHAP value of AA at position"]
        decoy_SHAP_mean = df[(df["amino acid"] == aa) & (df["example type"] == "decoy")]["mean SHAP value of AA at position"]
        hit_SHAP_prop = df[(df["amino acid"] == aa) & (df["example type"] == "hit")]["proportion of AA at position"]
        decoy_SHAP_prop = df[(df["amino acid"] == aa) & (df["example type"] == "decoy")]["proportion of AA at position"]

        if hit_SHAP_mean.shape[0] > 0 and decoy_SHAP_mean.shape[0] > 0 \
                and hit_SHAP_prop.shape[0] > 0 and decoy_SHAP_prop.shape[0] > 0:
            AAs.append(aa)
            hits_mean.append(float(hit_SHAP_mean))
            decoys_mean.append(float(decoy_SHAP_mean))
            hits_prop.append(float(hit_SHAP_prop))
            decoys_prop.append(float(decoy_SHAP_prop))

    df_mean = pd.DataFrame({"AA": AAs, "hits": hits_mean, "decoys": decoys_mean})
    df_prop = pd.DataFrame({"AA": AAs, "hits": hits_prop, "decoys": decoys_prop})
    for data, (kind, title, xlabel, base) in zip([df_mean, df_prop],
        [("mean", "mean SHAP value (hits to decoys)", "difference in mean SHAP value", 10),
         ("prop", "proportions at position (hits to decoys)", "difference in proportion", 0.005)]):

        ax = plt.figure(figsize=(5, 10))

        ax = sns.stripplot(data=data,
                           x='hits',
                           y='AA',
                           orient='h',
                           order=data['AA'],
                           size=15,
                           color='black')

        lim_hits = np.max(base * np.ceil(np.abs(data['hits'].values) / base))
        lim_decoys = np.max(base * np.ceil(np.abs(data['decoys'].values) / base))
        lim = max(lim_hits, lim_decoys)

        arrow_starts = data['hits'].values
        arrow_lengths = data['decoys'].values - arrow_starts

        # add arrows
        for i, subject in enumerate(data['AA']):
            arrow_color = "black"
            if arrow_lengths[i] != 0:
                ax.arrow(arrow_starts[i],  # start x
                         i,  # start y
                         arrow_lengths[i],  # change in x
                         0,  # change in y
                         head_width=0.3,
                         head_length=2*lim/50,
                         width=0.1,
                         fc=arrow_color,
                         ec=arrow_color,
                         length_includes_head=True)

        ax.set_title(title)
        ax.axvline(x=0, color='black', ls='--', lw=2, zorder=0)
        ax.grid(axis='y', color='0.95')
        ax.grid(axis='x', color='1.0')
        ax.set_xlim(-lim, lim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('amino acid')
        sns.despine(left=True, bottom=True)

        filename = f"{pMHC.OUTPUT_FOLDER}{SEP}shap{SEP}pics{SEP}change{SEP}" \
                   + f"{SEP}shap_{mhc_allele_name.replace(':', '')}_{checkpoint}_change_{position}_{kind}"
        plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches='tight')
        plt.show()