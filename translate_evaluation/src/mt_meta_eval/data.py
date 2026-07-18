"""Prepare evaluation data.

Datasets:
- wmt24pp: google/wmt24pp from Hugging Face (sentence-level, en->X, human
  post-edited references). Filters is_bad_source rows and the 'canary' domain,
  per the WMT24++ paper.
- wmt25: WMT25 General MT test set from the official wmt-conference GitHub
  repo (document-level). Only records with a human reference (refA) are kept;
  15 of the 31 pairs (including en-de_DE) have no reference and cannot be
  scored with reference-based metrics.
"""

import json
import random
import urllib.request

from .config import SEED, WMT24PP_DATASET, WMT25_URL, data_dir


def _wmt24pp_rows(pair: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(WMT24PP_DATASET, pair, split="train")
    return [
        {
            "segment_id": r["segment_id"],
            "domain": r["domain"],
            "source": r["source"],
            "reference": r["target"],  # human post-edited reference
        }
        for r in ds
        if not r["is_bad_source"] and r["domain"] != "canary"
    ]


def _wmt25_raw_path():
    raw = data_dir("wmt25") / "raw" / "wmt25-genmt.jsonl"
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        print(f"[data] downloading WMT25 test set from {WMT25_URL}")
        urllib.request.urlretrieve(WMT25_URL, raw)
    return raw


def _wmt25_rows(pair: str) -> list[dict]:
    rows, seen_pairs = [], set()
    with _wmt25_raw_path().open() as f:
        for line in f:
            d = json.loads(line)
            lp = d["doc_id"].split("_#_")[0]
            seen_pairs.add(lp)
            if lp != pair:
                continue
            ref = (d.get("refs") or {}).get("refA", {}).get("ref")
            if not ref:
                continue
            rows.append(
                {
                    "segment_id": d["doc_id"],
                    "domain": d["domain"],
                    "source": d["src_text"],
                    "reference": ref,
                }
            )
    if not rows:
        raise ValueError(
            f"No WMT25 records with a human reference for {pair!r}. "
            f"Available pairs: {sorted(seen_pairs)} (not all have references)"
        )
    return rows


def prepare_pair(
    dataset: str, pair: str, limit: int | None = None, overwrite: bool = False
) -> int:
    out_path = data_dir(dataset) / f"{pair}.jsonl"
    if out_path.exists() and not overwrite:
        n = sum(1 for _ in out_path.open())
        print(f"[data] {dataset}/{pair}: exists with {n} segments, skipping (--overwrite to redo)")
        return n

    rows = _wmt24pp_rows(pair) if dataset == "wmt24pp" else _wmt25_rows(pair)

    if limit is not None and limit < len(rows):
        rng = random.Random(SEED)
        rows = rng.sample(rows, limit)
        rows.sort(key=lambda r: str(r["segment_id"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[data] {dataset}/{pair}: wrote {len(rows)} segments to {out_path}")
    return len(rows)


def load_pair(dataset: str, pair: str) -> list[dict]:
    path = data_dir(dataset) / f"{pair}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `mteval prepare` first")
    with path.open() as f:
        return [json.loads(line) for line in f]
