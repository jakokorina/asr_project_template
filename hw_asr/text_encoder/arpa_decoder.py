from typing import List, Union
from pyctcdecode import build_ctcdecoder
import torch

import numpy as np
from torch import Tensor

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class ArpaTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"
    EMPTY_IND = 0

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        self.decoder = build_ctcdecoder(self.alphabet, "../language_model/3-gram.arpa")

    def ctc_decode(self, inds: List[int]) -> str:
        last_char = self.EMPTY_IND
        decoded = []
        for ind in inds:
            if ind != last_char and ind != self.EMPTY_IND:
                decoded.append(self.ind2char[ind])
            last_char = ind
        return ''.join(decoded)

    def beam_search(self, logits: Tensor, beam_size: int = 50) -> str:
        return self.decoder.decode(logits.detach().cpu().numpy(), beam_size)
