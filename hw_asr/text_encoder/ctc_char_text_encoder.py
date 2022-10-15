from typing import List, NamedTuple, Dict
from collections import defaultdict
import torch

from .char_text_encoder import CharTextEncoder


class Probabilities():
    def __init__(self, prob_emp=0, prob_nonemp=0):
        self.prob_emp = prob_emp
        self.prob_nonemp = prob_nonemp

    def prob(self) -> float:
        return self.prob_emp + self.prob_nonemp

    def __add__(self, other):
        return Probabilities(prob_emp=self.prob_emp + other.prob_emp,
                             prob_nonemp=self.prob_nonemp + other.prob_nonemp)


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"
    EMPTY_IND = 0

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        last_char = self.EMPTY_IND
        decoded = []
        for ind in inds:
            if ind != last_char and ind != self.EMPTY_IND:
                decoded.append(self.ind2char[ind])
            last_char = ind
        return ''.join(decoded)

    def extend_and_merge_(self, dp: Dict[str, Probabilities], prob_all: torch.tensor) -> Dict[str, Probabilities]:
        """
        https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7
        Make 1 step of beam search algorithm
        :param dp: dict from previous step
        :param prob_all: probabilities for current step
        :return: updated dict
        """
        new_dp: Dict[str, Probabilities] = defaultdict(Probabilities)
        for line, prob in dp.items():
            # copy
            pr_nonemp = 0
            if line != "":
                pr_nonemp = prob.prob_nonemp * prob_all[self.char2ind[line[-1]]]
            new_dp[line] += Probabilities(prob_emp=prob.prob() * prob_all[self.EMPTY_IND],
                                          prob_nonemp=pr_nonemp)

            # extend
            for i in range(1, len(prob_all)):
                if line != "" and self.ind2char[i] == line[-1]:
                    # str type 'aa'
                    pr_nonemp = prob.prob_emp * prob_all[i]
                else:
                    # str type 'ab'
                    pr_nonemp = prob.prob() * prob_all[i]

                new_dp[line + self.ind2char[i]] += Probabilities(prob_emp=0, prob_nonemp=pr_nonemp)

        return dict(new_dp)

    @staticmethod
    def cut_beams_(dp: Dict[str, Probabilities], beam_size: int) -> Dict[str, Probabilities]:
        """
        Keep strings with only with the highest probability
        :param dp: dict of strings with probas to cut
        :param beam_size: number of items to keep
        :return: cut dict
        """
        sort: list[str, Probabilities] = sorted(dp.items(), key=lambda x: x[1].prob())[-beam_size:]
        return {key: value for key, value in sort}

    def ctc_beam_search(self, probs: torch.tensor,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        dp: Dict[str, Probabilities] = {"": Probabilities(prob_emp=1, prob_nonemp=0)}

        for proba in probs:
            dp = self.extend_and_merge_(dp, proba)
            dp = self.cut_beams_(dp, beam_size)

        hypos: List[Hypothesis] = [Hypothesis(line, pr.prob()) for line, pr in dp.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
