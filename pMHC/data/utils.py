import torch

from pMHC import TT_MHC, TT_N_FLANK, TT_PEPTIDE, TT_C_FLANK

pseudo_pos = \
    [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95,
     97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171]


def move_dict_to_device(dictionary, model):
    for key, tensor in dictionary.items():
        if type(tensor) == torch.Tensor:
            dictionary[key] = tensor.to(model.device)

    return dictionary


def convert_example_to_batch(example):
    batch = {"input_ids": example["input_ids"].unsqueeze(dim=0),
             "token_type_ids": example["token_type_ids"].unsqueeze(dim=0),
             "position_ids": example["position_ids"].unsqueeze(dim=0),
             "input_mask": example["input_mask"].unsqueeze(dim=0)}
    if "targets" in example:
        batch.update({"targets": example["targets"].unsqueeze(dim=0)})
    return batch


def convert_examples_to_batch(examples):
    batch = {"input_ids": torch.cat([example["input_ids"].unsqueeze(dim=0) for example in examples], dim=0),
             "token_type_ids": torch.cat([example["token_type_ids"].unsqueeze(dim=0) for example in examples], dim=0),
             "position_ids": torch.cat([example["position_ids"].unsqueeze(dim=0) for example in examples], dim=0),
             "input_mask": torch.cat([example["input_mask"].unsqueeze(dim=0) for example in examples], dim=0)}
    if "targets" in examples:
        batch.update({"targets": torch.cat([example["targets"].unsqueeze(dim=0) for example in examples], dim=0)})
    return batch


#
# create input representations
#


def get_input_rep_PSEUDO(n_flank, peptide_seq, c_flank, mhc_allele_pseudo_seq, model):
    mhc_seq = mhc_allele_pseudo_seq
    mhc_pos = pseudo_pos
    return _get_input(n_flank, peptide_seq, c_flank, mhc_seq, mhc_pos, model)


def get_input_rep_FULL(n_flank, peptide_seq, c_flank, mhc_allele_full_seq, model):
    mhc_seq = mhc_allele_full_seq
    mhc_pos = list(range(1, len(mhc_seq) + 1))
    return _get_input(n_flank, peptide_seq, c_flank, mhc_seq, mhc_pos, model)


def _get_input(n_flank, peptide_seq, c_flank, mhc_seq, mhc_pos, model):
    l_n_flank = len(n_flank)
    input_n_flank = n_flank + "<sep>" if l_n_flank > 0 else ""
    pos_n_flank = list(range(l_n_flank, 0, -1)) + [0] if l_n_flank > 0 else []
    tt_n_flank = [TT_N_FLANK] * (l_n_flank + 1) if l_n_flank > 0 else []

    l_peptide = len(peptide_seq)
    input_peptide = peptide_seq + "<sep>"
    pos_peptide = list(range(1, l_peptide + 1)) + [0]
    tt_peptide = [TT_PEPTIDE] * (l_peptide + 1)

    l_c_flank = len(c_flank)
    input_c_flank = c_flank + "<sep>" if l_c_flank > 0 else ""
    pos_c_flank = list(range(1, l_c_flank + 1)) + [0] if l_c_flank > 0 else []
    tt_c_flank = [TT_C_FLANK] * (l_c_flank + 1) if l_c_flank > 0 else []

    l_mhc = len(mhc_seq)
    input_mhc = mhc_seq + "<sep>"
    pos_mhc = mhc_pos + [0]
    tt_mhc = [TT_MHC] * (l_mhc + 1) if l_mhc > 0 else []

    input_seq = "<cls>" + input_n_flank + input_peptide + input_c_flank + input_mhc

    input_ids = model.tokenizer_.encode(input_seq)
    tt_ids = [TT_PEPTIDE] + tt_n_flank + tt_peptide + tt_c_flank + tt_mhc
    pos_ids = [0] + pos_n_flank + pos_peptide + pos_c_flank + pos_mhc
    input_mask = [1] * len(tt_ids)

    return {"input_ids": torch.tensor(input_ids),
            "token_type_ids": torch.tensor(tt_ids),
            "position_ids": torch.tensor(pos_ids),
            "input_mask": torch.tensor(input_mask)}
