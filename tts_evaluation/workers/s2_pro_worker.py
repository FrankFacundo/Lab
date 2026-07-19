"""Synthesis worker for Fish Audio S2-Pro — in-process fish-speech engine.

Loads the 4B text2semantic model and the DAC codec ONCE, then loops items
through TTSInferenceEngine (the same engine fish-speech's server/webui use).
This replaces the previous 3-subprocess-per-item flow, which reloaded the
model every item and left the GPU idle most of the time.

Requires third_party/fish-speech installed in .venv-s2 (setup_envs.sh).
Protocol: reads --tasks JSONL ({item_id, text, lang, ref_audio, ref_text}),
writes <out-dir>/<item_id>.wav, prints DONE/FAIL lines.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "third_party" / "fish-speech"
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from tqdm import tqdm

    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    precision = torch.bfloat16 if device != "cpu" else torch.float32
    model = Path(args.model_path)

    llama_queue = launch_thread_safe_queue(
        checkpoint_path=str(model), device=device, precision=precision, compile=False
    )
    decoder_model = load_decoder_model(
        config_name="modded_dac_vq",
        checkpoint_path=str(model / "codec.pth"),
        device=device,
    )
    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=False,
        precision=precision,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.tasks) as f:
        tasks = [json.loads(line) for line in f]

    # reuse the reference across items sharing the same prompt audio
    for t in tqdm(tasks, desc="s2-pro", unit="item"):
        try:
            ref_bytes = Path(t["ref_audio"]).read_bytes()
            req = ServeTTSRequest(
                text=t["text"],
                references=[ServeReferenceAudio(audio=ref_bytes, text=t["ref_text"])],
                format="wav",
                seed=42,
            )
            final = None
            for result in engine.inference(req):
                if result.code == "error":
                    raise result.error or RuntimeError("engine error")
                if result.code == "final":
                    final = result
            if final is None or final.audio is None:
                raise RuntimeError("no final audio")
            sr, audio = final.audio
            sf.write(out_dir / f"{t['item_id']}.wav", audio, sr)
            print(f"DONE {t['item_id']}", flush=True)
        except Exception as e:
            print(f"FAIL {t['item_id']} {type(e).__name__}: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
