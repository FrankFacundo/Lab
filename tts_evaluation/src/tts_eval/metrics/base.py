from abc import ABC, abstractmethod


class Metric(ABC):
    """One TTS evaluation method.

    score_items receives the test items (with text/ref_audio/...) and the path
    of each synthesized wav, aligned; returns one score per item.
    """

    key: str
    higher_is_better: bool = True

    @abstractmethod
    def score_items(self, items: list[dict], wav_paths: list[str]) -> list[float]: ...

    def score_corpus(self, items: list[dict], wav_paths: list[str]) -> float:
        segs = self.score_items(items, wav_paths)
        return sum(segs) / len(segs)
