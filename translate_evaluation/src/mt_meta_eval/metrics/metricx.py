"""MetricX-24 (optional) — the metric family Google's TranslateGemma report uses.

Lower is better (0 = perfect, 25 = worst). Requires the google-research
implementation, which is not on PyPI:

    .venv/bin/pip install "git+https://github.com/google-research/metricx.git"

Uses the hybrid checkpoint, which scores with a reference when available.
"""

import torch

from .base import Metric


class MetricX24Metric(Metric):
    key = "metricx24"
    model_name = "google/metricx-24-hybrid-large-v2p6"
    tokenizer_name = "google/mt5-large"
    higher_is_better = False

    def __init__(self, batch_size: int = 8, max_input_length: int = 1536):
        try:
            from metricx24 import models  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "MetricX requires the google-research/metricx package:\n"
                '  .venv/bin/pip install "git+https://github.com/google-research/metricx.git"'
            ) from e
        self.batch_size = batch_size
        self.max_input_length = max_input_length

    def score_segments(self, sources, hypotheses, references):
        from metricx24 import models
        from transformers import AutoTokenizer

        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        model = models.MT5ForRegression.from_pretrained(
            self.model_name, dtype=torch.bfloat16 if device != "cpu" else torch.float32
        ).to(device)
        model.eval()

        texts = [
            f"source: {s} candidate: {h} reference: {r}"
            for s, h, r in zip(sources, hypotheses, references)
        ]
        from tqdm import tqdm

        scores: list[float] = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), self.batch_size), desc="metricx24", leave=False
            ):
                batch = tokenizer(
                    texts[i : i + self.batch_size],
                    max_length=self.max_input_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                # MetricX drops the trailing EOS token, per the reference implementation
                input_ids = batch["input_ids"][:, :-1].to(device)
                attention_mask = batch["attention_mask"][:, :-1].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                scores.extend(out.predictions.float().cpu().tolist())
        return scores
