import json

from typing import List

from transformers import PreTrainedTokenizer

import pMHC


class StandardTokenizer(PreTrainedTokenizer):

    def __init__(self, pretrain=False, **kwargs):
        super().__init__(**kwargs)
        self.pretrain = pretrain
        self.vocab = json.loads(open(pMHC.VOCAB_FILENAME, 'r').read())
        self.vocab_inv = {value: key for key, value in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def start_token(self) -> str:
        return "<cls>"

    @property
    def start_token_id(self):
        return self.vocab["<cls>"]

    @property
    def stop_token(self) -> str:
        return "<sep>"

    @property
    def stop_token_id(self):
        return self.vocab["<sep>"]

    @property
    def mask_token(self) -> str:
        return "<mask>"

    @property
    def pad_token(self) -> str:
        return "<pad>"

    @property
    def pad_token_id(self):
        return self.vocab["<pad>"]

    @property
    def unk_token_id(self):
        return self.vocab["<unk>"]

    def _tokenize(self, text, **kwargs):
        return self.tokenize(text)

    def _convert_token_to_id(self, token):
        return self.vocab[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self.vocab_inv[index]

    def tokenize(self, text: str, **kwargs) -> List[str]:
        ret = []
        i = 0
        while i < len(text):
            if text[i] == "<":
                j = i+text[i:].find(">")
                ret += [text[i:j+1]]
                i = j
            else:
                ret += [text[i]]
            i += 1

        return ret

    def add_special_tokens(self, token_ids: List[str]) -> List[str]:
        cls_token = [self.start_token]
        sep_token = [self.stop_token]
        return cls_token + token_ids + sep_token