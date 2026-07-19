"""Synthesis worker for Step-Audio-EditX zero-shot TTS (clone mode).

EXPERIMENTAL on macOS: upstream tests Linux+CUDA only. Requires
third_party/Step-Audio-EditX (repo) and third_party/Step-Audio-Tokenizer
(weights), both installed by setup_envs.sh. English only for this eval.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "third_party" / "Step-Audio-EditX"
TOKENIZER = Path(__file__).resolve().parents[1] / "third_party" / "Step-Audio-Tokenizer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError:  # keep worker usable in a minimal env
        def tqdm(x, **kw):
            return x

    with open(args.tasks) as f:
        tasks = [json.loads(line) for line in f]

    for t in tqdm(tasks, desc="step-audio-editx", unit="item"):
        try:
            with tempfile.TemporaryDirectory() as td:
                r = subprocess.run(
                    [sys.executable, "tts_infer.py",
                     "--model-path", args.model_path,
                     "--tokenizer-path", str(TOKENIZER),
                     "--prompt-text", t["ref_text"],
                     "--prompt-audio", t["ref_audio"],
                     "--generated-text", t["text"],
                     "--edit-type", "clone",
                     "--output-dir", td],
                    cwd=REPO, capture_output=True, text=True,
                )
                if r.returncode != 0:
                    raise RuntimeError(r.stderr[-400:])
                wavs = sorted(Path(td).glob("*.wav"))
                if not wavs:
                    raise RuntimeError("no output wav produced")
                shutil.copy(wavs[0], out_dir / f"{t['item_id']}.wav")
            print(f"DONE {t['item_id']}", flush=True)
        except Exception as e:
            print(f"FAIL {t['item_id']} {type(e).__name__}: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
