from .base import Metric

DEFAULT_METRICS = ["wer_whisper", "sim_wavlm", "utmos"]
ALL_METRICS = DEFAULT_METRICS + ["wer_qwen3asr"]

HIGHER_IS_BETTER = {
    "wer_whisper": False,
    "wer_qwen3asr": False,
    "sim_wavlm": True,
    "utmos": True,
}


def get_metric(key: str, **kwargs) -> Metric:
    if key == "wer_whisper":
        from .asr_wer import WhisperWER

        return WhisperWER(**kwargs)
    if key == "wer_qwen3asr":
        from .asr_wer import Qwen3AsrWER

        return Qwen3AsrWER(**kwargs)
    if key == "sim_wavlm":
        from .speaker_sim import WavLMSpeakerSim

        return WavLMSpeakerSim(**kwargs)
    if key == "utmos":
        from .utmos import UTMOSMetric

        return UTMOSMetric(**kwargs)
    raise ValueError(f"Unknown metric {key!r}; choose from: {ALL_METRICS}")
