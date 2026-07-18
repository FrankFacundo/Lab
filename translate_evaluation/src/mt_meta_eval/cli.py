"""mteval — prepare / translate / score / analyze pipeline."""

import argparse
import json

from .config import (
    DATASETS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PAIRS,
    SCORES_DIR,
    TRANSLATIONS_DIR,
)


def _resolve_pairs(args) -> list[str]:
    return args.pairs if args.pairs else DEFAULT_PAIRS[args.dataset]


def cmd_prepare(args):
    from .data import prepare_pair

    for pair in _resolve_pairs(args):
        prepare_pair(args.dataset, pair, limit=args.limit, overwrite=args.overwrite)


def cmd_translate(args):
    from tqdm import tqdm

    from .data import load_pair
    from .translators import get_translator

    max_new_tokens = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS[args.dataset]
    translator = get_translator(args.model, max_new_tokens=max_new_tokens)
    out_dir = TRANSLATIONS_DIR / translator.key / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair in _resolve_pairs(args):
        rows = load_pair(args.dataset, pair)
        out_path = out_dir / f"{pair}.jsonl"

        done = 0
        if out_path.exists() and not args.overwrite:
            done = sum(1 for _ in out_path.open())
        if done >= len(rows):
            print(f"[translate] {translator.key}/{args.dataset}/{pair}: complete ({done}), skipping")
            continue

        mode = "a" if done else "w"
        with out_path.open(mode) as f:
            batches = range(done, len(rows), args.batch_size)
            for start in tqdm(batches, desc=f"{translator.key}/{args.dataset}/{pair}"):
                chunk = rows[start : start + args.batch_size]
                hyps = translator.translate_batch([r["source"] for r in chunk], pair)
                for row, hyp in zip(chunk, hyps):
                    f.write(
                        json.dumps(
                            {"segment_id": row["segment_id"], "hypothesis": hyp},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                f.flush()
        print(f"[translate] {translator.key}/{args.dataset}/{pair}: wrote {out_path}")


def _load_hypotheses(model_key: str, dataset: str, pair: str) -> dict:
    path = TRANSLATIONS_DIR / model_key / dataset / f"{pair}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `mteval translate` first")
    with path.open() as f:
        return {r["segment_id"]: r["hypothesis"] for r in map(json.loads, f)}


def _inputs_md5(model_key: str, dataset: str, pair: str) -> str:
    """Fingerprint of everything a score depends on: hypotheses + prepared data."""
    import hashlib

    from .config import data_dir

    h = hashlib.md5()
    for path in (
        TRANSLATIONS_DIR / model_key / dataset / f"{pair}.jsonl",
        data_dir(dataset) / f"{pair}.jsonl",
    ):
        h.update(path.read_bytes() if path.exists() else b"missing")
    return h.hexdigest()


def cmd_score(args):
    import pandas as pd

    from .data import load_pair
    from .metrics import get_metric

    pairs = _resolve_pairs(args)
    sys_path = SCORES_DIR / "system_scores.csv"

    # Results cache: a (dataset, pair, model, metric) combo is skipped when it
    # was already scored against byte-identical hypotheses (--rescore forces).
    cache: dict[tuple, str] = {}
    if sys_path.exists() and not args.overwrite and not args.rescore:
        prev = pd.read_csv(sys_path)
        if "hyp_md5" in prev.columns:
            cache = {
                (r.dataset, r.pair, r.model, r.metric): r.hyp_md5
                for r in prev.itertuples()
            }

    seg_rows, sys_rows = [], []
    for metric_key in args.metrics:
        kwargs = {}
        if metric_key == "llm_judge":
            kwargs["judge_model"] = args.judge_model
        try:
            metric = get_metric(metric_key, **kwargs)
            for pair in pairs:
                data = load_pair(args.dataset, pair)
                metric.pair = pair
                for model_key in args.models:
                    hyp_md5 = _inputs_md5(model_key, args.dataset, pair)
                    key = (args.dataset, pair, model_key, metric_key)
                    if cache.get(key) == hyp_md5:
                        print(
                            f"[score] cached  {metric_key} | {model_key} | "
                            f"{args.dataset}/{pair} (unchanged, skipping)"
                        )
                        continue
                    hyps_by_id = _load_hypotheses(model_key, args.dataset, pair)
                    rows = [r for r in data if r["segment_id"] in hyps_by_id]
                    sources = [r["source"] for r in rows]
                    references = [r["reference"] for r in rows]
                    hypotheses = [hyps_by_id[r["segment_id"]] for r in rows]

                    print(
                        f"[score] {metric_key} | {model_key} | {args.dataset}/{pair} "
                        f"({len(rows)} segs)"
                    )
                    seg_scores = metric.score_segments(sources, hypotheses, references)
                    corpus = metric.score_corpus(sources, hypotheses, references)

                    seg_rows.extend(
                        {
                            "dataset": args.dataset,
                            "pair": pair,
                            "model": model_key,
                            "metric": metric_key,
                            "segment_id": r["segment_id"],
                            "score": s,
                        }
                        for r, s in zip(rows, seg_scores)
                    )
                    sys_rows.append(
                        {
                            "dataset": args.dataset,
                            "pair": pair,
                            "model": model_key,
                            "metric": metric_key,
                            "score": corpus,
                            "n_segments": len(rows),
                            "hyp_md5": hyp_md5,
                        }
                    )
        except Exception as e:
            # drop the failed metric entirely, keep results from the others
            print(f"[score] FAILED {metric_key}: {e}")
            seg_rows = [r for r in seg_rows if r["metric"] != metric_key]
            sys_rows = [r for r in sys_rows if r["metric"] != metric_key]

    if not sys_rows:
        print("[score] nothing to do — all requested results are cached")
        return

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    seg_path = SCORES_DIR / "segment_scores.csv"

    seg_new = pd.DataFrame(seg_rows)
    sys_new = pd.DataFrame(sys_rows)
    if seg_path.exists() and not args.overwrite:
        # merge with previous runs, new results win on (dataset, pair, model, metric)
        key = ["dataset", "pair", "model", "metric"]
        done = sys_new[key].drop_duplicates()
        old = pd.read_csv(seg_path)
        old = old.merge(done, on=key, how="left", indicator=True)
        seg_new = pd.concat(
            [old[old["_merge"] == "left_only"].drop(columns="_merge"), seg_new]
        )
        old_sys = pd.read_csv(sys_path)
        old_sys = old_sys.merge(done, on=key, how="left", indicator=True)
        sys_new = pd.concat(
            [old_sys[old_sys["_merge"] == "left_only"].drop(columns="_merge"), sys_new]
        )
    seg_new.to_csv(seg_path, index=False)
    sys_new.to_csv(sys_path, index=False)
    print(f"[score] wrote {seg_path} and {sys_path}")


def cmd_analyze(args):
    from .analyze import write_report

    write_report(models=args.models)


def main():
    from .metrics import ALL_METRICS, DEFAULT_METRICS
    from .translators import MODEL_KEYS

    parser = argparse.ArgumentParser(prog="mteval")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--dataset", default="wmt24pp", choices=DATASETS)
        p.add_argument(
            "--pairs", nargs="+", default=None, help="default: per-dataset pair list"
        )

    p = sub.add_parser("prepare", help="download + filter test set pairs")
    add_common(p)
    p.add_argument("--limit", type=int, default=None, help="subsample N segments per pair")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("translate", help="run one model over the prepared data")
    add_common(p)
    p.add_argument("--model", required=True, choices=MODEL_KEYS)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="default: 1024 (wmt24pp) / 4096 (wmt25, document-level)",
    )
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_translate)

    p = sub.add_parser("score", help="score translations with the evaluation methods")
    add_common(p)
    p.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, choices=ALL_METRICS)
    p.add_argument("--judge-model", default="claude-opus-4-8")
    p.add_argument(
        "--rescore",
        action="store_true",
        help="recompute the requested combos even if cached (other results kept)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="discard ALL previous scores and start the CSVs fresh",
    )
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("analyze", help="meta-evaluation report of metric (dis)agreement")
    p.add_argument("--models", nargs="+", default=MODEL_KEYS)
    p.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
