from typing import List
from pyctcdecode import build_ctcdecoder
from torch import Tensor
import gzip
import os, shutil, wget

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class ArpaTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"
    EMPTY_IND = 0

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        lm_gzip_path = "3-gram.arpa.gz"
        if not os.path.exists(lm_gzip_path):
            print('Downloading pruned 3-gram model.')
            lm_url = 'http://www.openslr.org/resources/11/3-gram.arpa.gz'
            lm_gzip_path = wget.download(lm_url)
            print('Downloaded the 3-gram language model.')
        else:
            print('.arpa.gz already exists.')

        uppercase_lm_path = '3-gram.arpa'
        if not os.path.exists(uppercase_lm_path):
            with gzip.open(lm_gzip_path, 'rb') as f_zipped:
                with open(uppercase_lm_path, 'wb') as f_unzipped:
                    shutil.copyfileobj(f_zipped, f_unzipped)
            print('Unzipped the 3-gram language model.')
        else:
            print('Unzipped .arpa already exists.')

        lm_path = 'lowercase_3-gram.arpa'
        if not os.path.exists(lm_path):
            with open(uppercase_lm_path, 'r') as f_upper:
                with open(lm_path, 'w') as f_lower:
                    for line in f_upper:
                        f_lower.write(line.lower())
        print('Converted language model file to lowercase.')

        self.decoder = build_ctcdecoder(self.alphabet, kenlm_model_path=lm_path)

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
