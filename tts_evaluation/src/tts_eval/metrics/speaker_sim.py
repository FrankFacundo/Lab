"""Speaker similarity: cosine between x-vectors of the synthesized audio and
the cloning reference audio (microsoft/wavlm-base-plus-sv), the WavLM-based
protocol the Qwen3-TTS paper follows. Range ~0-1, higher is better."""

import librosa
import torch

from .base import Metric


class WavLMSpeakerSim(Metric):
    key = "sim_wavlm"
    model_id = "microsoft/wavlm-base-plus-sv"

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self._model = None
        self._extractor = None
        # x-vector conv stack is numerically unhappy in fp16; keep fp32
        self._device = "mps" if torch.backends.mps.is_available() else (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def _load(self):
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self._extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        self._model = WavLMForXVector.from_pretrained(self.model_id).to(self._device)
        self._model.eval()

    def _embed(self, paths: list[str]) -> torch.Tensor:
        from tqdm import tqdm

        embs = []
        for i in tqdm(range(0, len(paths), self.batch_size), desc="wavlm-sv", leave=False):
            chunk = [librosa.load(p, sr=16_000)[0] for p in paths[i : i + self.batch_size]]
            inputs = self._extractor(
                chunk, sampling_rate=16_000, return_tensors="pt", padding=True
            ).to(self._device)
            with torch.no_grad():
                embs.append(self._model(**inputs).embeddings.cpu())
        return torch.cat(embs)

    def score_items(self, items, wav_paths):
        if self._model is None:
            self._load()
        hyp = self._embed(wav_paths)
        ref = self._embed([i["ref_audio"] for i in items])
        return torch.nn.functional.cosine_similarity(hyp, ref).clamp(0, 1).tolist()
