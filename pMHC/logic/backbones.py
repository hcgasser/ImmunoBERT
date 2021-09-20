import torch
from tape import ProteinBertModel

import pMHC
from pMHC import SEP


class TAPEBackbone(ProteinBertModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configuration(self):
        return self.config.to_dict()

    # only added token_type_ids and position_ids to the original tape function
    def forward(self, input_ids, token_type_ids, position_ids, input_mask, targets=None):
        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, chunks=None)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs

    @staticmethod
    def get_backbone(output_attentions=False):
        backbone = TAPEBackbone.from_pretrained(f"{pMHC.PROJECT_FOLDER}{SEP}pretrained{SEP}TAPE{SEP}",
                                                output_attentions=output_attentions) #'bert-base')

        # expand the embedding weights
        token_type_weights = backbone.embeddings.token_type_embeddings.weight
        token_type_weights = token_type_weights.expand(torch.Size((5, backbone.config.hidden_size)))
        backbone.embeddings.token_type_embeddings = torch.nn.Embedding(
            5, backbone.config.hidden_size, _weight=token_type_weights.clone())

        return backbone


class RobertaBackbone(ProteinBertModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configuration(self):
        return self.config.to_dict()

    # only added token_type_ids and position_ids to the original tape function
    def forward(self, input_ids, token_type_ids, position_ids, input_mask, targets=None):
        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, chunks=None)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs

    @staticmethod
    def get_backbone(output_attentions=False):
        backbone = TAPEBackbone.from_pretrained(f"{pMHC.PROJECT_FOLDER}{SEP}pretrained{SEP}Roberta{SEP}",
                                                output_attentions=output_attentions) #'bert-base')

        # expand the embedding weights
        token_type_weights = backbone.embeddings.token_type_embeddings.weight
        token_type_weights = token_type_weights.expand(torch.Size((5, backbone.config.hidden_size)))
        backbone.embeddings.token_type_embeddings = torch.nn.Embedding(
            5, backbone.config.hidden_size, _weight=token_type_weights.clone())

        return backbone
