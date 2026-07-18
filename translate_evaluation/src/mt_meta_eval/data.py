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

from .config import (
    FLORES_CODES,
    FLORES_DATASET,
    SEED,
    WMT24PP_DATASET,
    WMT25_URL,
    data_dir,
    source_code,
    target_code,
)


def _wmt24pp_rows(pair: str) -> list[dict]:
    """Load a wmt24pp pair; 'xx_XX-en' loads 'en-xx_XX' with the direction reversed.

    Reversed pairs translate the human (post-edited) target back into English and
    score against the original English source. Caveat: the source side is then
    translationese, so reversed results are indicative, not a WMT-blessed setup.
    """
    from datasets import load_dataset

    reverse = pair.endswith("-en")
    hf_config = f"en-{pair.removesuffix('-en')}" if reverse else pair
    ds = load_dataset(WMT24PP_DATASET, hf_config, split="train")
    return [
        {
            "segment_id": r["segment_id"],
            "domain": r["domain"],
            "source": r["target"] if reverse else r["source"],
            "reference": r["source"] if reverse else r["target"],
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


def _flores_lang(code: str):
    from datasets import load_dataset

    try:
        cfg = FLORES_CODES[code]
    except KeyError:
        raise ValueError(
            f"No FLORES config registered for {code!r}; add it to FLORES_CODES "
            "in config.py (flores200 pairs use bare codes, e.g. 'es-fr')"
        ) from None
    try:
        return load_dataset(FLORES_DATASET, cfg, split="devtest")
    except Exception as e:
        if "gated" in str(e).lower():
            raise RuntimeError(
                f"{FLORES_DATASET} is gated (auto-accept): while logged in to "
                f"Hugging Face, click Agree at https://huggingface.co/datasets/{FLORES_DATASET}"
            ) from e
        raise


def _flores_rows(pair: str) -> list[dict]:
    """FLORES+ devtest is multi-way parallel: join source/target languages by id."""
    src_ds = _flores_lang(source_code(pair))
    tgt_ds = _flores_lang(target_code(pair))
    tgt_by_id = {r["id"]: r["text"] for r in tgt_ds}
    return [
        {
            "segment_id": r["id"],
            "domain": r.get("domain", ""),
            "source": r["text"],
            "reference": tgt_by_id[r["id"]],
        }
        for r in src_ds
        if r["id"] in tgt_by_id
    ]


_LOADERS = {"wmt24pp": _wmt24pp_rows, "wmt25": _wmt25_rows, "flores200": _flores_rows}


def prepare_pair(
    dataset: str, pair: str, limit: int | None = None, overwrite: bool = False
) -> int:
    out_path = data_dir(dataset) / f"{pair}.jsonl"
    if out_path.exists() and not overwrite:
        n = sum(1 for _ in out_path.open())
        print(f"[data] {dataset}/{pair}: exists with {n} segments, skipping (--overwrite to redo)")
        return n

    rows = _LOADERS[dataset](pair)

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
