"""Shared configuration: paths, datasets, languages, model/metric registries."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
SYNTH_DIR = PROJECT_ROOT / "outputs" / "audio"
SCORES_DIR = PROJECT_ROOT / "outputs" / "scores"
REPORT_DIR = PROJECT_ROOT / "outputs"

MODELS_ROOT = Path("/Users/frankfacundo/Models")

SEED = 42
SAMPLE_RATE = 16_000  # all metrics consume 16 kHz mono

# Datasets. Zero-shot voice-cloning protocol everywhere: each item has a
# reference audio (+ its transcript) from a speaker, and a target text the
# model must speak in that voice.
#   seedtts — seed-tts-eval English test set (Common Voice domain); what the
#             Qwen3-TTS paper reports on.
#   mls     — cloning pairs built from Multilingual LibriSpeech test splits
#             (audiobook domain, has speaker ids): for each target utterance,
#             a different 3-10 s utterance of the same speaker is the reference.
DATASETS = ["seedtts", "mls"]
DATASET_LANGS = {"seedtts": ["en"], "mls": ["es", "fr"]}
DEFAULT_EVAL = [("seedtts", "en"), ("mls", "es"), ("mls", "fr")]

SEEDTTS_GDRIVE_ID = "1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP"
MLS_DATASET = "facebook/multilingual_librispeech"
MLS_CONFIGS = {"es": "spanish", "fr": "french"}

# TTS models under test. `worker_env` is the venv whose python runs the
# synthesis worker (heterogeneous inference stacks conflict, so heavyweight
# models get their own env — see setup_envs.sh).
MODEL_REGISTRY = {
    "qwen3-tts": {
        "path": MODELS_ROOT / "Qwen" / "Qwen3-TTS-12Hz-1.7B-Base",
        "tokenizer_path": MODELS_ROOT / "Qwen" / "Qwen3-TTS-Tokenizer-12Hz",
        "langs": ["en", "es", "fr"],
        "worker": "qwen3_tts_worker.py",
        "worker_env": PROJECT_ROOT / ".venv",
    },
    "s2-pro": {
        "path": MODELS_ROOT / "fishaudio" / "s2-pro",
        "langs": ["en", "es", "fr"],
        "worker": "s2_pro_worker.py",
        "worker_env": PROJECT_ROOT / ".venv-s2",
    },
    "step-audio-editx": {
        # zero-shot TTS supports zh/en (+ja/ko via tags) — es/fr NOT supported
        # by the model per its README; it only joins the English track.
        "path": MODELS_ROOT / "stepfun-ai" / "Step-Audio-EditX",
        "langs": ["en"],
        "worker": "step_audio_worker.py",
        "worker_env": PROJECT_ROOT / ".venv-step",
    },
}
MODEL_KEYS = list(MODEL_REGISTRY)

# Whisper language names for ASR normalization
LANG_NAMES = {"en": "english", "es": "spanish", "fr": "french"}


def data_dir(dataset: str) -> Path:
    return DATA_ROOT / dataset
