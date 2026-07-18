"""Lexical (string-overlap) metrics via sacrebleu: BLEU, chrF++, TER.

Corpus scores use the standard corpus-level statistics. Segment scores use
sentence-level variants (sentence-BLEU with exponential smoothing), which are
noisier — they feed the metric-agreement analysis, not headline numbers.

BLEU is tokenization-sensitive: CJK targets need language-aware tokenizers
(zh / ja-mecab / ko-mecab), otherwise scores collapse to ~0. chrF++ is
character-based and safe everywhere; TER runs with asian_support.
"""

from sacrebleu.metrics import BLEU, CHRF, TER
from tqdm import tqdm

from ..config import target_code
from .base import Metric


def _sentence_scores(metric, hypotheses, references, desc: str) -> list[float]:
    return [
        metric.sentence_score(h, [r]).score
        for h, r in tqdm(
            zip(hypotheses, references), total=len(hypotheses), desc=desc, leave=False
        )
    ]


def _bleu_tokenizer(pair: str | None) -> str:
    lang = target_code(pair).split("_")[0] if pair else ""
    return {"zh": "zh", "ja": "ja-mecab", "ko": "ko-mecab"}.get(lang, "13a")


class BleuMetric(Metric):
    key = "bleu"

    def __init__(self):
        self._by_tok: dict[str, tuple[BLEU, BLEU]] = {}

    def _metrics(self) -> tuple[BLEU, BLEU]:
        tok = _bleu_tokenizer(self.pair)
        if tok not in self._by_tok:
            self._by_tok[tok] = (
                BLEU(tokenize=tok),
                BLEU(tokenize=tok, effective_order=True),
            )
        return self._by_tok[tok]

    def score_segments(self, sources, hypotheses, references):
        _, segment_metric = self._metrics()
        return _sentence_scores(segment_metric, hypotheses, references, "bleu segments")

    def score_corpus(self, sources, hypotheses, references):
        corpus_metric, _ = self._metrics()
        return corpus_metric.corpus_score(hypotheses, [references]).score


class ChrfMetric(Metric):
    key = "chrf++"

    def __init__(self):
        self.metric = CHRF(word_order=2)

    def score_segments(self, sources, hypotheses, references):
        return _sentence_scores(self.metric, hypotheses, references, "chrf++ segments")

    def score_corpus(self, sources, hypotheses, references):
        return self.metric.corpus_score(hypotheses, [references]).score


class TerMetric(Metric):
    key = "ter"
    higher_is_better = False

    def __init__(self):
        self.metric = TER(normalized=True, asian_support=True)

    def score_segments(self, sources, hypotheses, references):
        return _sentence_scores(self.metric, hypotheses, references, "ter segments")

    def score_corpus(self, sources, hypotheses, references):
        return self.metric.corpus_score(hypotheses, [references]).score
