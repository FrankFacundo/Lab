# TTS metric meta-evaluation

Companion to `../translate_evaluation`, for text-to-speech: synthesize the
same zero-shot voice-cloning test items with several TTS models and score
them with **multiple evaluation methods**, so you can see which model wins
and where the metrics disagree.

**Protocol** (as in the [Qwen3-TTS report](https://arxiv.org/abs/2601.15621)):
each item = a 3–10 s reference audio + its transcript (the voice to clone)
and a target text the model must speak in that voice.

**Models under test** (weights from `/Users/frankfacundo/Models`):

| Key | Model | Languages here | Runtime |
|---|---|---|---|
| `qwen3-tts` | Qwen/Qwen3-TTS-12Hz-1.7B-Base | en es fr | `qwen-tts` package, main `.venv` |
| `s2-pro` | fishaudio/s2-pro | en es fr | fish-speech repo, `.venv-s2` |
| `step-audio-editx` | stepfun-ai/Step-Audio-EditX | **en only** (es/fr not supported by the model) | Step repo, `.venv-step` — **experimental on macOS** (upstream is Linux/CUDA) |

**Test sets:**

| Dataset | Languages | Source |
|---|---|---|
| `seedtts` | en | official [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) English set (Common Voice domain) — what the Qwen3-TTS paper reports; first `prepare` downloads a few GB from Google Drive |
| `mls` | es, fr | cloning pairs built from [Multilingual LibriSpeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) test splits (audiobooks, speaker ids): reference = different 3–10 s utterance of the same speaker; streamed, only sampled items downloaded |

**Evaluation methods under evaluation:**

| Metric | What | Range | Notes |
|---|---|---|---|
| `wer_whisper` | intelligibility: WER of Whisper-large-v3 transcript vs target text | %, ↓ | community standard |
| `wer_qwen3asr` | same, with local Qwen/Qwen3-ASR-1.7B | %, ↓ | the ASR the Qwen3-TTS paper used — lets you check whether the ASR engine changes rankings |
| `sim_wavlm` | speaker similarity to the reference (WavLM-SV x-vectors, cosine) | 0–1 ↑ | the paper's SIM protocol |
| `utmos` | predicted naturalness MOS (UTMOS22) | 1–5 ↑ | English-trained; treat es/fr as comparative |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e .
uv pip install -p .venv/bin/python qwen-tts
./setup_envs.sh        # clones fish-speech + Step repos, creates .venv-s2/.venv-step
```

## Run

```bash
./run_eval.sh                      # 200 items x {seedtts:en, mls:es, mls:fr}, all models
LIMIT=50 MODELS="qwen3-tts s2-pro" ./run_eval.sh    # smaller/custom

# or step by step
.venv/bin/ttseval prepare  --limit 200
.venv/bin/ttseval synthesize --models qwen3-tts
.venv/bin/ttseval score
.venv/bin/ttseval analyze          # -> outputs/report.md
```

Outputs: `outputs/audio/<model>/<dataset>/<lang>/<item>.wav`,
`outputs/scores/*.csv`, and `outputs/report.md` with (1) system scores,
(2) per-language winners + metric flips, (3) item-level metric correlations,
(4) pairwise preference agreement.

Everything is cached and resumable, as in translate_evaluation: `prepare`
keeps data, `synthesize` skips existing wavs, `score` fingerprints
(audio set + data file) per (dataset, lang, model, metric) and skips
unchanged combos (`--rescore` / `--overwrite` to force). A model failing on
an item prints `FAIL <id>` and the scorer simply uses the wavs that exist.

## Notes / limitations

- Synthesis runs on MPS; large models are slow — budget hours for 200 items
  x 3 langs x 3 models. Start with `LIMIT=50`.
- `step-audio-editx` is **effectively unavailable on macOS**: its declared
  dependencies (`vllm`, `onnxruntime-gpu`, `deepspeed`, CUDA torch) have no
  mac wheels. The worker + env plumbing is in place, so the model can join
  the English track when run on a Linux/CUDA machine (it needs the
  Step-Audio-Tokenizer weights, which setup_envs.sh downloads). It is scoped
  to English regardless — its README lists es/fr as unsupported.
- Whisper-large-v3 (~3 GB) and WavLM-SV download on first `score`.
- WER on es/fr uses Whisper's basic text normalizer (no number spelling
  normalization) — consistent across models, so comparisons remain fair.
