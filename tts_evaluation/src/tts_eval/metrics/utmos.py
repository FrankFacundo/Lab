"""UTMOS (predicted naturalness MOS, 1-5, higher better) via SpeechMOS.

Trained on English MOS data (VoiceMOS'22); commonly applied cross-lingually —
treat es/fr numbers as comparative rather than absolute.
"""

import librosa
import torch

from .base import Metric


class UTMOSMetric(Metric):
    key = "utmos"

    def __init__(self):
        self._model = None

    def score_items(self, items, wav_paths):
        from tqdm import tqdm

        if self._model is None:
            self._model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            )
        scores = []
        for p in tqdm(wav_paths, desc="utmos", leave=False):
            wav, sr = librosa.load(p, sr=16_000)
            with torch.no_grad():
                s = self._model(torch.from_numpy(wav).unsqueeze(0), sr)
            scores.append(float(s.item()))
        return scores
