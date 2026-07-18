"""Prepare zero-shot voice-cloning test items.

Every dataset produces one JSONL per language at data/<dataset>/<lang>.jsonl:
  {"item_id", "lang", "text",           # target text the model must speak
   "ref_audio",                          # path to reference wav (the voice)
   "ref_text",                           # transcript of the reference audio
   "gt_audio"}                           # ground-truth human wav (optional)

seedtts: official seed-tts-eval English set (downloaded once from the
  project's Google Drive tarball via gdown; ~few GB, cached).
mls: built from facebook/multilingual_librispeech test splits (streaming,
  so only the sampled items are downloaded): for each target utterance we take
  a different 3-10 s utterance of the SAME speaker as the cloning reference.
"""

import json
import random
import tarfile

import soundfile as sf

from .config import (
    MLS_CONFIGS,
    MLS_DATASET,
    SEED,
    SEEDTTS_GDRIVE_ID,
    data_dir,
)


# ---------------------------------------------------------------- seedtts ---

def _seedtts_root():
    root = data_dir("seedtts") / "raw"
    marker = root / "seedtts_testset"
    if marker.exists():
        return marker
    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / "seedtts_testset.tgz"
    if not tar_path.exists():
        import gdown

        print("[data] downloading seed-tts-eval testset (Google Drive, a few GB)…")
        gdown.download(id=SEEDTTS_GDRIVE_ID, output=str(tar_path), quiet=False)
    print("[data] extracting", tar_path)
    with tarfile.open(tar_path) as tf:
        tf.extractall(root, filter="data")
    return marker


def _seedtts_rows(lang: str) -> list[dict]:
    assert lang == "en", "seed-tts-eval covers en (and zh) only"
    root = _seedtts_root() / "en"
    rows = []
    # meta.lst: filename|prompt_text|prompt_wav|target_text[|gt_wav]
    for line in (root / "meta.lst").read_text().splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        name, ref_text, ref_wav, text = parts[0], parts[1], parts[2], parts[3]
        gt = parts[4] if len(parts) > 4 else None
        ref_path = root / ref_wav
        if not ref_path.exists():
            ref_path = root / "prompt-wavs" / ref_wav
        rows.append(
            {
                "item_id": name,
                "lang": "en",
                "text": text,
                "ref_audio": str(ref_path),
                "ref_text": ref_text,
                "gt_audio": str(root / gt) if gt else None,
            }
        )
    return rows


# -------------------------------------------------------------------- mls ---

def _mls_rows(lang: str, limit: int | None) -> list[dict]:
    from datasets import load_dataset

    n_items = limit or 200
    wav_dir = data_dir("mls") / "wavs" / lang
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Stream the test split, bucket utterances by speaker until we have enough
    # speakers with >=2 usable utterances (target + 3-10s reference).
    ds = load_dataset(MLS_DATASET, MLS_CONFIGS[lang], split="test", streaming=True)
    by_speaker: dict = {}
    rows = []
    for ex in ds:
        spk = ex["speaker_id"]
        dur = len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
        by_speaker.setdefault(spk, []).append((ex, dur))
        bucket = by_speaker[spk]
        if len(bucket) == 2:  # first = reference (if 3-10 s), second = target
            (ref_ex, ref_dur), (tgt_ex, tgt_dur) = bucket
            if not (3.0 <= ref_dur <= 10.0):
                bucket.pop(0)  # drop unusable ref, wait for another utterance
                continue
            if tgt_dur > 30.0:
                bucket.pop(1)
                continue
            item_id = f"{lang}_{tgt_ex['id']}".replace("/", "-")
            ref_path = wav_dir / f"{item_id}_ref.wav"
            gt_path = wav_dir / f"{item_id}_gt.wav"
            sf.write(ref_path, ref_ex["audio"]["array"], ref_ex["audio"]["sampling_rate"])
            sf.write(gt_path, tgt_ex["audio"]["array"], tgt_ex["audio"]["sampling_rate"])
            rows.append(
                {
                    "item_id": item_id,
                    "lang": lang,
                    "text": tgt_ex["transcript"],
                    "ref_audio": str(ref_path),
                    "ref_text": ref_ex["transcript"],
                    "gt_audio": str(gt_path),
                }
            )
            by_speaker[spk] = []  # one pair per speaker pass; allow more later
            if len(rows) >= n_items:
                break
    return rows


# ------------------------------------------------------------------ shared --

def prepare(dataset: str, lang: str, limit: int | None = None, overwrite: bool = False) -> int:
    out_path = data_dir(dataset) / f"{lang}.jsonl"
    if out_path.exists() and not overwrite:
        n = sum(1 for _ in out_path.open())
        print(f"[data] {dataset}/{lang}: exists with {n} items, skipping (--overwrite to redo)")
        return n

    if dataset == "seedtts":
        rows = _seedtts_rows(lang)
        if limit is not None and limit < len(rows):
            rng = random.Random(SEED)
            rows = rng.sample(rows, limit)
            rows.sort(key=lambda r: r["item_id"])
    elif dataset == "mls":
        rows = _mls_rows(lang, limit)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[data] {dataset}/{lang}: wrote {len(rows)} items to {out_path}")
    return len(rows)


def load_items(dataset: str, lang: str) -> list[dict]:
    path = data_dir(dataset) / f"{lang}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `ttseval prepare` first")
    with path.open() as f:
        return [json.loads(line) for line in f]
