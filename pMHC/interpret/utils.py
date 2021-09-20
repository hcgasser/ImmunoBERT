import py3Dmol
import colorsys

import torch

import pMHC
from pMHC import SEP
from pMHC.data.utils import move_dict_to_device, get_input_rep_PSEUDO, convert_example_to_batch

hydrophobic_aa = ["G", "A", "V", "L", "I", "F", "M", "P", "W"]
hydrophilic_aa = ["S", "T", "Y", "C", "N", "Q", "D", "E", "H", "K", "R"]


def get_alternative_predictions(obs, alt_position, alt_AAs, model, orig_pred=None):
    preds = []
    for alt_AA in alt_AAs:
        alt_seq = obs.peptide_seq[:(alt_position-1)] + alt_AA + obs.peptide_seq[alt_position:]
        alt_example = get_input_rep_PSEUDO(obs.n_flank, alt_seq, obs.c_flank,
                                           obs.deconvoluted_mhc_allele.pseudo_seq, model)

        alt_pred = float(torch.sigmoid(
            model(move_dict_to_device(convert_example_to_batch(alt_example), model))).detach().cpu())
        if orig_pred is None:
            preds.append((alt_AA, alt_pred))
        else:
            preds.append((alt_AA, alt_pred, orig_pred - alt_pred))
    return preds


def to_pdb_filename(mhc_allele_ID):
    return f"{pMHC.DATA_FOLDER}{SEP}pHLA3D{SEP}{mhc_allele_ID[:5]}{SEP}{mhc_allele_ID[4]}_{mhc_allele_ID[5:7]}_{mhc_allele_ID[8:10]}_V1.pdb"


def show_MHC(mhc_allele_ID, importances, to_filename=None):
    pdb_filename = to_pdb_filename(mhc_allele_ID)

    with open(pdb_filename) as file:
        mhc_pdb = "".join([x for x in file])

    mhc_pdb_split = [x.split() for x in mhc_pdb.split("\n")]

    view = py3Dmol.view(width=800, height=800)
    view.addModelsAsFrames(mhc_pdb)

    for line in mhc_pdb_split:
        if len(line) > 5 and line[0] == "ATOM":
            atom_idx = int(line[1])
            aa_idx = int(line[5]) - 1
            domain = line[4]

            if domain == "A":
                if aa_idx >= len(importances):  # atom outside our considered range
                    color_hex = "0x00FF00"
                else:
                    # color = [x*255 for x in colorsys.hsv_to_rgb(0, 0.5, rel_importances[aa_idx])]
                    print(f"aa_idx: {aa_idx} {line[3]} {importances[aa_idx]}")
                    color_dec = (255 * importances[aa_idx], 0, 255 * (1 - importances[aa_idx]))
                    color_hex = f"0x{int(color_dec[0]):02X}{int(color_dec[1]):02X}{int(color_dec[2]):02X}"

                view.setStyle({'model': -1, 'serial': int(line[1])}, {"cartoon": {'color': color_hex}})
    view.zoomTo()
    view.show()


