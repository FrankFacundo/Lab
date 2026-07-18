"""Synthesis worker for Fish Audio S2-Pro via the fish-speech repo CLI.

Requires third_party/fish-speech (installed in .venv-s2 by setup_envs.sh).
Three-stage documented flow per item:
  1) encode reference wav -> prompt tokens (dac/inference.py, codec.pth)
  2) text + prompt -> semantic codes    (text2semantic/inference.py)
  3) decode codes -> wav                 (dac/inference.py)
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "third_party" / "fish-speech"


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import torch

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = Path(args.model_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    with open(args.tasks) as f:
        tasks = [json.loads(line) for line in f]

    for t in tasks:
        try:
            with tempfile.TemporaryDirectory() as td:
                # 1) reference audio -> prompt tokens
                r = run(
                    [py, "fish_speech/models/dac/inference.py",
                     "-i", t["ref_audio"], "--checkpoint-path", str(model / "codec.pth"),
                     "--device", device,
                     "--output-path", f"{td}/prompt.wav"],
                    cwd=REPO,
                )
                if r.returncode != 0:
                    raise RuntimeError(f"codec encode: {r.stderr[-400:]}")
                prompt_npy = f"{td}/prompt.npy"

                # 2) text -> semantic codes
                r = run(
                    [py, "fish_speech/models/text2semantic/inference.py",
                     "--text", t["text"], "--prompt-text", t["ref_text"],
                     "--prompt-tokens", prompt_npy,
                     "--checkpoint-path", str(model),
                     "--device", device,
                     "--output-dir", td],
                    cwd=REPO,
                )
                if r.returncode != 0:
                    raise RuntimeError(f"text2semantic: {r.stderr[-400:]}")

                # 3) codes -> audio
                r = run(
                    [py, "fish_speech/models/dac/inference.py",
                     "-i", f"{td}/codes_0.npy",
                     "--checkpoint-path", str(model / "codec.pth"),
                     "--device", device,
                     "--output-path", f"{td}/out.wav"],
                    cwd=REPO,
                )
                if r.returncode != 0:
                    raise RuntimeError(f"codec decode: {r.stderr[-400:]}")
                shutil.copy(f"{td}/out.wav", out_dir / f"{t['item_id']}.wav")
            print(f"DONE {t['item_id']}", flush=True)
        except Exception as e:
            print(f"FAIL {t['item_id']} {type(e).__name__}: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
