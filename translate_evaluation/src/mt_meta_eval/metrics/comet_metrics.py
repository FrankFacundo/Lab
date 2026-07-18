"""Neural metrics from the COMET family.

- comet22 (Unbabel/wmt22-comet-da): reference-based, 0-1 scale.
- cometkiwi22 (Unbabel/wmt22-cometkiwi-da): reference-free QE, 0-1 scale.
  NOTE: the CometKiwi checkpoint is gated on Hugging Face — accept the terms at
  https://huggingface.co/Unbabel/wmt22-cometkiwi-da while logged in.
"""

import torch

from .base import Metric


class _CometBase(Metric):
    model_name: str

    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from comet import download_model, load_from_checkpoint

            self._model = load_from_checkpoint(download_model(self.model_name))
        return self._model

    def _predict(self, data: list[dict]) -> list[float]:
        model = self._load()
        use_accel = torch.backends.mps.is_available() or torch.cuda.is_available()
        out = model.predict(
            data,
            batch_size=self.batch_size,
            gpus=1 if use_accel else 0,
            accelerator="auto",
        )
        return list(out.scores)


class Comet22Metric(_CometBase):
    key = "comet22"
    model_name = "Unbabel/wmt22-comet-da"

    def score_segments(self, sources, hypotheses, references):
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        return self._predict(data)


class CometKiwi22Metric(_CometBase):
    key = "cometkiwi22"
    model_name = "Unbabel/wmt22-cometkiwi-da"
    requires_reference = False

    def score_segments(self, sources, hypotheses, references):
        data = [{"src": s, "mt": h} for s, h in zip(sources, hypotheses)]
        return self._predict(data)
