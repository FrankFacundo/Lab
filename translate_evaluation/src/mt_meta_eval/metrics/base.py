from abc import ABC, abstractmethod


class Metric(ABC):
    """One MT evaluation method.

    Segment scores drive the metric-agreement analysis; the corpus score is
    what papers usually report (for lexical metrics it is NOT the mean of
    segment scores, so both are computed explicitly).
    """

    key: str
    requires_reference: bool = True
    higher_is_better: bool = True
    # set by the scoring loop before each pair (used by BLEU tokenizer choice
    # and the LLM judge's prompt language)
    pair: str | None = None

    @abstractmethod
    def score_segments(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ) -> list[float]: ...

    def score_corpus(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ) -> float:
        segs = self.score_segments(sources, hypotheses, references)
        return sum(segs) / len(segs)
