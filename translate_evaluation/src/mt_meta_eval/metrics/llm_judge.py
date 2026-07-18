"""LLM-as-judge (optional) — GEMBA-DA-style 0-100 scoring via the Claude API.

Scores each segment with a reference-aware direct-assessment prompt, submitted
through the Message Batches API (50% of standard token prices; a batch of a
few thousand segments typically completes well within an hour).

Requires the `anthropic` package and credentials (ANTHROPIC_API_KEY or an
`ant auth login` profile).
"""

import time

from ..config import target_language_name
from .base import Metric

PROMPT = """Score the following translation from English to {lang} with respect to the human reference on a continuous scale from 0 to 100, where 0 means "no meaning preserved" and 100 means "perfect meaning and grammar".

English source: {source}
{lang} human reference: {reference}
{lang} machine translation: {hypothesis}"""

SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
    "additionalProperties": False,
}


class LLMJudgeMetric(Metric):
    key = "llm_judge"

    def __init__(self, judge_model: str = "claude-opus-4-8"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.judge_model = judge_model

    def score_segments(self, sources, hypotheses, references):
        import json

        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        lang = target_language_name(self.pair)
        requests = [
            Request(
                custom_id=f"seg-{i}",
                params=MessageCreateParamsNonStreaming(
                    model=self.judge_model,
                    max_tokens=64,
                    output_config={"format": {"type": "json_schema", "schema": SCORE_SCHEMA}},
                    messages=[
                        {
                            "role": "user",
                            "content": PROMPT.format(
                                lang=lang, source=s, reference=r, hypothesis=h
                            ),
                        }
                    ],
                ),
            )
            for i, (s, h, r) in enumerate(zip(sources, hypotheses, references))
        ]

        batch = self.client.messages.batches.create(requests=requests)
        print(f"[llm_judge] submitted batch {batch.id} ({len(requests)} segments)")
        while True:
            batch = self.client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            time.sleep(30)

        scores: dict[int, float] = {}
        for result in self.client.messages.batches.results(batch.id):
            idx = int(result.custom_id.split("-")[1])
            if result.result.type == "succeeded":
                msg = result.result.message
                text = next((b.text for b in msg.content if b.type == "text"), "")
                try:
                    scores[idx] = float(json.loads(text)["score"])
                except (ValueError, KeyError):
                    scores[idx] = float("nan")
            else:
                scores[idx] = float("nan")
        return [scores.get(i, float("nan")) for i in range(len(sources))]
