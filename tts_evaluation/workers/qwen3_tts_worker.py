"""Synthesis worker for Qwen3-TTS-12Hz Base (voice clone) via the qwen-tts package.

Standalone: no tts_eval imports, runs in whatever venv is configured.
Protocol: reads --tasks JSONL ({item_id, text, lang, ref_audio, ref_text}),
writes <out-dir>/<item_id>.wav, prints DONE/FAIL lines to stdout.
"""

import argparse
import json
import sys

import soundfile as sf
import torch

LANG_NAMES = {"en": "English", "es": "Spanish", "fr": "French"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from pathlib import Path

    from qwen_tts import Qwen3TTSModel

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model = Qwen3TTSModel.from_pretrained(
        args.model_path, device_map=device, dtype=torch.bfloat16
    )

    import transformers
    from tqdm import tqdm

    transformers.logging.set_verbosity_error()  # silence per-item generate logs

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.tasks) as f:
        tasks = [json.loads(line) for line in f]

    for t in tqdm(tasks, desc="qwen3-tts", unit="item"):
        try:
            wavs, sr = model.generate_voice_clone(
                text=t["text"],
                language=LANG_NAMES[t["lang"]],
                ref_audio=t["ref_audio"],
                ref_text=t["ref_text"],
            )
            sf.write(out_dir / f"{t['item_id']}.wav", wavs[0], sr)
            print(f"DONE {t['item_id']}", flush=True)
        except Exception as e:  # keep going; the scorer only sees finished wavs
            print(f"FAIL {t['item_id']} {type(e).__name__}: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
