import numpy as np

from functools import partial
import sklearn
from sklearn.utils import check_random_state
import scipy as sp
import pickle
from tqdm import tqdm
import copy

from matplotlib import pyplot as plt

import torch

import lime
from lime import lime_base
from lime.explanation import DomainMapper

import pMHC
from pMHC import MAX_PEPTIDE_LEN, FLANK_LEN, TT_MHC, TT_N_FLANK, TT_PEPTIDE, TT_C_FLANK, TT_NAMES, SEP, MHC_PSEUDO
from pMHC.data.utils import convert_examples_to_batch, move_dict_to_device, pseudo_pos


class MyExplainer(object):
    """Explains epitope classifiers."""

    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 feature_selection='auto',
                 split_expression=r'\W+',
                 mask_string=None,
                 random_state=None,
                 seed=42):

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(seed)
        self.base = lime_base.LimeBase(kernel_fn, verbose,
                                       random_state=self.random_state)
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.mask_string = mask_string
        self.split_expression = split_expression

    def explain_instance(self,
                         example,
                         model,
                         labels=(1,),
                         top_labels=None,
                         num_samples=5000,
                         batch_size=5,
                         distance_metric='cosine',
                         model_regressor=None,
                         explain_peptide=True,
                         explain_context=True,
                         explain_mhc=True
                         ):
        """
        Adapted from lime.lime_text
        """

        model.eval()

        domain_mapper = MyDomainMapper(example, model)

        self.features_range = []

        i = 0
        for input_id, token_type_id, position_id, input_mask in \
                zip(example["input_ids"], example["token_type_ids"],
                    example["position_ids"], example["input_mask"]):
            if i < (len(example["input_ids"]) - 1):
                if explain_peptide and token_type_id == TT_PEPTIDE and position_id > 0:
                    self.features_range.append(i)
                if explain_context and (
                        token_type_id == TT_N_FLANK or token_type_id == TT_C_FLANK) and position_id > 0:
                    self.features_range.append(i)
                if explain_mhc and token_type_id == TT_MHC and position_id > 0:
                    self.features_range.append(i)

            i += 1

        num_features = len(self.features_range)

        data, yss, distances = self.__data_labels_distances(
            example, model, num_samples,
            self.features_range,
            batch_size=batch_size,
            distance_metric=distance_metric)
        ret_exp = lime.explanation.Explanation(domain_mapper=domain_mapper,
                                               class_names=["not presented", "presented"],
                                               random_state=self.random_state)
        ret_exp.score = {}
        ret_exp.local_pred = {}
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def __data_labels_distances(self,
                                example,
                                model,
                                num_samples,
                                features_range,
                                batch_size=5,
                                distance_metric='cosine'):
        """
        Adapted from lime.lime_text
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        def multiply_example(example, num_samples):
            examples = [example]
            for i in range(num_samples):
                new_example = {}
                new_example.update({"input_ids": example["input_ids"].data.clone()})
                new_example.update({"token_type_ids": example["token_type_ids"].data.clone()})
                new_example.update({"position_ids": example["position_ids"].data.clone()})
                new_example.update({"input_mask": example["input_mask"].data.clone()})
                examples.append(new_example)

            return examples

        sample = self.random_state.randint(1, len(features_range), num_samples)
        examples = multiply_example(example, num_samples)

        data = np.ones((num_samples + 1, example["input_ids"].shape[-1]))

        for i, size in enumerate(sample, start=1):
            inactive = self.random_state.choice(features_range, size, replace=False)
            data[i, inactive] = 0

            examples[i]["input_mask"][inactive] = 0

        i = 0
        batch_idx = 0
        model.eval()
        while i <= num_samples:
            batch = convert_examples_to_batch(examples[i:min(i + batch_size, num_samples + 1)])

            # add prediction
            batch_labels_1 = torch.sigmoid(model(move_dict_to_device(batch, model)).detach().cpu())
            batch_labels_0 = 1 - batch_labels_1
            batch_labels = torch.cat([batch_labels_0, batch_labels_1], dim=1)

            labels = torch.cat([labels, batch_labels], dim=0) if i > 0 else batch_labels

            i += batch_size
            batch_idx += 1
            if batch_idx % 100 == 0:
                print(f"batch: {batch_idx} ({i} of {num_samples})")
                # print(f"\b\b\b\b\b\b\b\b\b\b\b\b {i:7d}", end="")

        labels.numpy()
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances


class MyDomainMapper(lime.explanation.DomainMapper):
    """Maps feature ids to words or word-positions"""

    def __init__(self, example, model):
        """Initializer.

        Args:
            indexed_string: lime_text.IndexedString, original string
        """
        self.example = example
        self.model = model

    def map_exp_ids(self, exp):
        """Maps ids to words or word-position strings.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]
            positions: if True, also return word positions

        Returns:
            list of tuples (word, weight), or (word_positions, weight) if
            examples: ('bad', 1) or ('bad_3-6-12', 1)
        """

        ret = []
        for x in exp:
            ret.append(
                ("{}_{}_{}".format(
                    self.model.tokenizer_._convert_id_to_token(int(self.example['input_ids'][x[0]])),
                    TT_NAMES[int(self.example['token_type_ids'][x[0]])],
                    int(self.example['position_ids'][x[0]])
                ), x[1])
            )

        return ret


def lime_analysis(mhc_allele_name, examples, model, peptide_length=9, num_samples=1000):
    imp_cnt_all, imp_cnt_hits, imp_cnt_decoys = calc_imp_cnts(title, model, examples, peptide_length, num_samples)

    stacked_bar_charts(mhc_allele_name, "", model,
                       imp_cnt_all["N-flank"], imp_cnt_all["peptide"], imp_cnt_all["C-flank"], imp_cnt_all["MHC"])
    stacked_bar_charts(mhc_allele_name, "-hits", model,
                       imp_cnt_hits["N-flank"], imp_cnt_hits["peptide"], imp_cnt_hits["C-flank"], imp_cnt_hits["MHC"])
    stacked_bar_charts(mhc_allele_name, "-decoys", model,
                       imp_cnt_decoys["N-flank"], imp_cnt_decoys["peptide"], imp_cnt_decoys["C-flank"], imp_cnt_decoys["MHC"])


def stacked_bar_charts(mhc_allele_name, kind, model, imp_cnt_N=None, imp_cnt_peptide=None, imp_cnt_C=None, imp_cnt_MHC=None):
    axis_label_size = 18

    fig = plt.figure(figsize=(15, 6.5), constrained_layout=True)
    # fig.suptitle(title, fontsize=(axis_label_size+4))

    # define the categories to aggregate the rankings into
    # general
    cat_names = ["1-4", "5-9", "10-24", "25-49", "Lower"]
    cats = [range(0, 5), range(5, 10), range(10, 25), range(25, 50), range(50, 250)]
    cat_colors = ["#555555", "#0000ff", "#0055aa", "#00aa55", "#00ff00"]
        #["#0000ff", "#0033cc", "#008899", "#009988", "#00cc33", "#00ff00"]
        # ["#ffff00", "#ccff00", "#99ff00", "#88ff00", "#33ff00", "#00ff00"]
    # for peptides
    cat_names_peptide = ["1st", "2nd", "3rd", "4th", "Lower"]
    cats_peptide = [0, 1, 2, 3, range(4, 250)]
    cat_colors_peptide = ["#ff0000", "#cc0033", "#770077", "#3300cc", "#555555"]
        # ["#ff0000", "#cc0033", "#990088", "#880099", "#3300cc", "#0000ff"]

    legend_plotted = False

    # plot N-flank
    if imp_cnt_N is not None:
        imp_prop_N = imp_cnt_N/np.sum(imp_cnt_N, axis=0)

        bottom = np.array([0.] * FLANK_LEN)
        ax = fig.add_subplot(231)
        ax.set_title(f'N-flank', fontsize=axis_label_size)
        positions = [str(p) for p in list(range(FLANK_LEN, 0, -1))]
        for idx_cat, cat in enumerate(cats):
            imp_props = []
            for idx_position in range(FLANK_LEN-1, -1, -1):
                imp_props.append(np.sum(imp_prop_N[cat, idx_position]))
            ax.bar(positions, imp_props, bottom=bottom, color=cat_colors[idx_cat], label=f'{cat_names[idx_cat]}')
            bottom += imp_props

        ax.margins(x=0, y=0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        # ax.set_xlabel('position', fontsize=axis_label_size)
        # ax.set_xticks(positions)
        ax.set_xticklabels(positions, rotation=90)
        ax.tick_params(axis='x', labelsize=axis_label_size)
        ax.tick_params(axis='y', labelsize=axis_label_size)
        if not legend_plotted:
            ax.legend(cat_names, fontsize=(axis_label_size-2), ncol=2)
            legend_plotted = True

    # plot peptide
    if imp_cnt_peptide is not None:
        peptide_length = imp_cnt_peptide.shape[1]
        imp_prop_peptide = imp_cnt_peptide/np.sum(imp_cnt_peptide, axis=0)

        bottom = np.array([0.] * peptide_length)
        ax = fig.add_subplot(232)
        ax.set_title(f'{peptide_length}-mer peptide', fontsize=axis_label_size)
        positions = [p+1 for p in range(peptide_length)]
        for idx_cat, cat in enumerate(cats_peptide):
            imp_props = []
            for idx_position in range(peptide_length):
                imp_props.append(np.sum(imp_prop_peptide[cat, idx_position]))
            ax.bar(positions, imp_props, bottom=bottom, color=cat_colors_peptide[idx_cat], label=f'{cat_names_peptide[idx_cat]}')
            bottom += imp_props

        ax.margins(x=0, y=0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        #ax.set_xlabel('position', fontsize=axis_label_size)
        ax.set_xticks(positions)
        ax.set_xticklabels(positions, rotation=90)
        ax.tick_params(axis='x', labelsize=axis_label_size)
        ax.tick_params(axis='y', labelsize=axis_label_size)
        ax.legend(cat_names_peptide, fontsize=(axis_label_size-2), ncol=2)

    # plot C-flank
    if imp_cnt_C is not None:
        imp_prop_C = imp_cnt_C/np.sum(imp_cnt_C, axis=0)

        bottom = np.array([0.] * FLANK_LEN)
        ax = fig.add_subplot(233)
        ax.set_title(f'C-flank', fontsize=axis_label_size)
        positions = list(range(1, FLANK_LEN+1))
        for idx_cat, cat in enumerate(cats):
            imp_props = []
            for idx_position in range(FLANK_LEN):
                imp_props.append(np.sum(imp_prop_C[cat, idx_position]))
            ax.bar(positions, imp_props, bottom=bottom, color=cat_colors[idx_cat], label=f'{cat_names[idx_cat]}')
            bottom += imp_props

        ax.margins(x=0, y=0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        # ax.set_xlabel('position', fontsize=axis_label_size)
        ax.set_xticks(positions)
        ax.set_xticklabels(positions, rotation=90)
        ax.tick_params(axis='x', labelsize=axis_label_size)
        ax.tick_params(axis='y', labelsize=axis_label_size)
        if not legend_plotted:
            ax.legend(cat_names, fontsize=(axis_label_size-2), ncol=2)
            legend_plotted = True

    # plot MHC
    if imp_cnt_MHC is not None:
        imp_prop_MHC = imp_cnt_MHC/np.sum(imp_cnt_MHC, axis=0)
        mhc_positions = [p-1 for p in pseudo_pos] if model.mhc_rep == MHC_PSEUDO else list(range(0, 200))

        bottom = np.array([0.] * len(mhc_positions))
        ax = fig.add_subplot(212)
        ax.set_title(f'{mhc_allele_name} protein', fontsize=axis_label_size)
        positions = [str(p+1) for p in mhc_positions]
        for idx_cat, cat in enumerate(cats):
            imp_props = []
            for idx_position in mhc_positions:
                imp_props.append(np.sum(imp_prop_MHC[cat, idx_position]))
            ax.bar(positions, imp_props, bottom=bottom, color=cat_colors[idx_cat], label=f'{cat_names[idx_cat]}')
            bottom += imp_props

        ax.margins(x=0, y=0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel('position', fontsize=axis_label_size)
        ax.set_xticklabels(positions, rotation=90)
        ax.tick_params(axis='x', labelsize=axis_label_size)
        ax.tick_params(axis='y', labelsize=axis_label_size)

        if not legend_plotted:
            ax.legend(cat_names, fontsize=(axis_label_size-2), ncol=len(cat_names))
            legend_plotted = True

    fig.tight_layout()

    filename = f"{pMHC.OUTPUT_FOLDER}{SEP}lime{SEP}lime_{mhc_allele_name.replace(':', '')}{kind}"
    plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches='tight')

    plt.show()


def calc_imp_cnts(title, model, examples, peptide_length, num_samples):
    explainer = pMHC.interpret.lime_based.MyExplainer()

    # initilize arrays storing the occurence of importances
    # rows... importance ranks
    # columns... positions
    imp_cnt_all = {"peptide": np.zeros((250, peptide_length)),
                    "MHC": np.zeros((250, 200)),
                    "N-flank": np.zeros((250, FLANK_LEN)),
                    "C-flank": np.zeros((250, FLANK_LEN))}
    imp_cnt_hits = copy.deepcopy(imp_cnt_all)
    imp_cnt_decoys = copy.deepcopy(imp_cnt_all)

    for example_ in tqdm(examples, "explain examples", disable=pMHC.TQDM_DISABLE):
        example = example_.get_tokenization(model)
        move_dict_to_device(example, model)

        exp = explainer.explain_instance(
            example,
            model,
            explain_peptide=True, explain_context=True, explain_mhc=True,
            batch_size=model.batch_size,
            num_samples=num_samples)

        feature_importances = exp.as_list()

        rank = 1
        imp_cnt = imp_cnt_hits if example['targets'] == 1.0 else imp_cnt_decoys
        for feature, importance in feature_importances:
            amino_acid, element, position = feature.split('_')

            # hits and decoys separately
            imp_cnt[element][rank-1, int(position)-1] += 1

            # both
            imp_cnt_all[element][rank-1, int(position)-1] += 1

            rank += 1

    with open(f"{pMHC.OUTPUT_FOLDER}{SEP}lime{SEP}{title.replace(':', '')}_importances.pickle", "wb") as f:
        pickle.dump((imp_cnt_all, imp_cnt_hits, imp_cnt_decoys), f)

    return imp_cnt_all, imp_cnt_hits, imp_cnt_decoys