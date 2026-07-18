"""Intelligibility: WER of an ASR transcript of the synthesized audio vs the
target text. Two engines, deliberately — the metric itself is under evaluation:

- wer_whisper: openai/whisper-large-v3 (the common community choice for
  seed-tts-eval style benchmarks on en and most European languages).
- wer_qwen3asr: local Qwen/Qwen3-ASR-1.7B (what the Qwen3-TTS paper used —
  keeping it lets us check whether the ASR engine changes model rankings).

Scores are WER in % per item (capped at 100); corpus = pooled WER over all
words, matching how benchmarks report it. Lower is better.
"""

import jiwer
import torch
from transformers.models.whisper.english_normalizer import (
    BasicTextNormalizer,
    EnglishTextNormalizer,
)

from ..config import LANG_NAMES, MODELS_ROOT
from .base import Metric

_EN_NORM = EnglishTextNormalizer({})
_BASIC_NORM = BasicTextNormalizer()


def _normalize(text: str, lang: str) -> str:
    return _EN_NORM(text) if lang == "en" else _BASIC_NORM(text)


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )


class _AsrWERBase(Metric):
    higher_is_better = False

    def _transcribe(self, wav_paths: list[str], lang: str) -> list[str]:
        raise NotImplementedError

    def score_items(self, items, wav_paths):
        scores = []
        for item, hyp_text in zip(items, self._transcribe(wav_paths, items[0]["lang"])):
            ref = _normalize(item["text"], item["lang"])
            hyp = _normalize(hyp_text, item["lang"])
            wer = jiwer.wer(ref, hyp) * 100 if ref.strip() else 0.0
            scores.append(min(100.0, wer))
        return scores

    def score_corpus(self, items, wav_paths):
        lang = items[0]["lang"]
        refs = [_normalize(i["text"], lang) for i in items]
        hyps = [_normalize(h, lang) for h in self._transcribe(wav_paths, lang)]
        keep = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
        return jiwer.wer([r for r, _ in keep], [h for _, h in keep]) * 100


class WhisperWER(_AsrWERBase):
    key = "wer_whisper"
    model_id = "openai/whisper-large-v3"

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self._pipe = None

    def _transcribe(self, wav_paths, lang):
        from tqdm import tqdm
        from transformers import pipeline

        if self._pipe is None:
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                dtype=torch.float16,
                device=_device(),
            )
        out = []
        for i in tqdm(range(0, len(wav_paths), self.batch_size), desc="whisper", leave=False):
            chunk = wav_paths[i : i + self.batch_size]
            res = self._pipe(
                chunk,
                generate_kwargs={"language": LANG_NAMES[lang], "task": "transcribe"},
            )
            out.extend(r["text"] for r in res)
        return out


class Qwen3AsrWER(_AsrWERBase):
    key = "wer_qwen3asr"
    model_id = str(MODELS_ROOT / "Qwen" / "Qwen3-ASR-1.7B")

    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _load(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, dtype=torch.bfloat16
        ).to(_device())
        self._model.eval()

    def _transcribe(self, wav_paths, lang):
        import librosa
        from tqdm import tqdm

        if self._model is None:
            self._load()
        out = []
        for i in tqdm(range(0, len(wav_paths), self.batch_size), desc="qwen3-asr", leave=False):
            chunk = wav_paths[i : i + self.batch_size]
            audios = [librosa.load(p, sr=16_000)[0] for p in chunk]
            inputs = self._processor(
                audios, sampling_rate=16_000, return_tensors="pt", padding=True
            ).to(self._model.device)
            with torch.no_grad():
                ids = self._model.generate(**inputs, max_new_tokens=256)
            out.extend(self._processor.batch_decode(ids, skip_special_tokens=True))
        return out
