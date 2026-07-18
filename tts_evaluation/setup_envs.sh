#!/usr/bin/env bash
# One-time setup of the per-model synthesis environments.
# The main .venv (created by README setup) runs the CLI, metrics, and qwen3-tts.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p third_party

echo "== s2-pro env (.venv-s2: fish-speech) =="
[ -d third_party/fish-speech ] || git clone --depth 1 https://github.com/fishaudio/fish-speech third_party/fish-speech
# pyaudio is only used by fish-speech's realtime mic tools and needs the
# portaudio system lib — drop it for batch inference
sed -i '' '/pyaudio/d' third_party/fish-speech/pyproject.toml
# keep uv from resolving a python<3.10-only numba
printf "numba>=0.60\nllvmlite>=0.43\nnumpy>=1.26\n" > third_party/fish_constraints.txt
uv venv --python 3.12 .venv-s2
uv pip install -p .venv-s2/bin/python -e third_party/fish-speech -c third_party/fish_constraints.txt
# torch<=2.8 lacks torch.mps.current_device, which torch internals call on MPS
cat > .venv-s2/lib/python3.12/site-packages/sitecustomize.py <<'PY'
try:
    import torch
    if not hasattr(torch.mps, "current_device"):
        torch.mps.current_device = lambda: 0
    if not hasattr(torch.mps, "set_device"):
        torch.mps.set_device = lambda device: None
except Exception:
    pass
PY

echo "== step-audio-editx env (.venv-step) — EXPERIMENTAL on macOS =="
[ -d third_party/Step-Audio-EditX ] || git clone --depth 1 https://github.com/stepfun-ai/Step-Audio-EditX third_party/Step-Audio-EditX
uv venv --python 3.12 .venv-step
uv pip install -p .venv-step/bin/python -e third_party/Step-Audio-EditX || \
  echo "WARNING: Step-Audio-EditX deps failed to resolve (Linux/CUDA-oriented); step model unavailable"
[ -d third_party/Step-Audio-Tokenizer ] || \
  hf download stepfun-ai/Step-Audio-Tokenizer --local-dir third_party/Step-Audio-Tokenizer

echo "Done."
