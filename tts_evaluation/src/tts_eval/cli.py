"""ttseval — prepare / synthesize / score / analyze pipeline."""

import argparse
import json
import subprocess
import sys

from .config import (
    DATASET_LANGS,
    DATASETS,
    DEFAULT_EVAL,
    MODEL_KEYS,
    MODEL_REGISTRY,
    PROJECT_ROOT,
    SCORES_DIR,
    SYNTH_DIR,
    data_dir,
)


def _eval_targets(args) -> list[tuple[str, str]]:
    """(dataset, lang) combos to run."""
    if args.dataset and args.langs:
        return [(args.dataset, lang) for lang in args.langs]
    if args.dataset:
        return [(args.dataset, lang) for lang in DATASET_LANGS[args.dataset]]
    return DEFAULT_EVAL


def cmd_prepare(args):
    from .data import prepare

    for dataset, lang in _eval_targets(args):
        prepare(dataset, lang, limit=args.limit, overwrite=args.overwrite)


def _wav_dir(model_key: str, dataset: str, lang: str):
    return SYNTH_DIR / model_key / dataset / lang


def cmd_synthesize(args):
    from .data import load_items

    for model_key in args.models:
        spec = MODEL_REGISTRY[model_key]
        for dataset, lang in _eval_targets(args):
            if lang not in spec["langs"]:
                print(f"[synth] {model_key} does not support {lang!r}, skipping")
                continue
            items = load_items(dataset, lang)
            out_dir = _wav_dir(model_key, dataset, lang)
            out_dir.mkdir(parents=True, exist_ok=True)
            todo = [
                i for i in items
                if args.overwrite or not (out_dir / f"{i['item_id']}.wav").exists()
            ]
            if not todo:
                print(f"[synth] {model_key} | {dataset}/{lang}: complete ({len(items)}), skipping")
                continue

            tasks_path = out_dir / "_tasks.jsonl"
            with tasks_path.open("w") as f:
                for i in todo:
                    f.write(json.dumps(i, ensure_ascii=False) + "\n")

            worker_py = spec["worker_env"] / "bin" / "python"
            if not worker_py.exists():
                print(
                    f"[synth] SKIPPING {model_key}: missing venv {spec['worker_env']} "
                    f"— run ./setup_envs.sh"
                )
                break
            print(f"[synth] {model_key} | {dataset}/{lang}: {len(todo)}/{len(items)} items")
            cmd = [
                str(worker_py),
                str(PROJECT_ROOT / "workers" / spec["worker"]),
                "--model-path", str(spec["path"]),
                "--tasks", str(tasks_path),
                "--out-dir", str(out_dir),
            ]
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"[synth] worker for {model_key} exited {res.returncode}", file=sys.stderr)
            tasks_path.unlink(missing_ok=True)


def _inputs_md5(model_key: str, dataset: str, lang: str) -> str:
    """Fingerprint of scoring inputs: the wav set (names+sizes) + data file."""
    import hashlib

    h = hashlib.md5()
    d = _wav_dir(model_key, dataset, lang)
    for p in sorted(d.glob("*.wav")) if d.exists() else []:
        h.update(f"{p.name}:{p.stat().st_size}".encode())
    data_file = data_dir(dataset) / f"{lang}.jsonl"
    h.update(data_file.read_bytes() if data_file.exists() else b"missing")
    return h.hexdigest()


def cmd_score(args):
    import pandas as pd

    from .data import load_items
    from .metrics import get_metric

    targets = _eval_targets(args)
    sys_path = SCORES_DIR / "system_scores.csv"

    cache: dict[tuple, str] = {}
    if sys_path.exists() and not args.overwrite and not args.rescore:
        prev = pd.read_csv(sys_path)
        if "inputs_md5" in prev.columns:
            cache = {
                (r.dataset, r.lang, r.model, r.metric): r.inputs_md5
                for r in prev.itertuples()
            }

    seg_rows, sys_rows = [], []
    for metric_key in args.metrics:
        try:
            metric = get_metric(metric_key)
            for dataset, lang in targets:
                items_all = load_items(dataset, lang)
                for model_key in args.models:
                    if lang not in MODEL_REGISTRY[model_key]["langs"]:
                        continue
                    inputs_md5 = _inputs_md5(model_key, dataset, lang)
                    key = (dataset, lang, model_key, metric_key)
                    if cache.get(key) == inputs_md5:
                        print(f"[score] cached  {metric_key} | {model_key} | {dataset}/{lang}")
                        continue
                    d = _wav_dir(model_key, dataset, lang)
                    pairs = [
                        (i, str(d / f"{i['item_id']}.wav"))
                        for i in items_all
                        if (d / f"{i['item_id']}.wav").exists()
                    ]
                    if not pairs:
                        print(f"[score] no audio for {model_key} | {dataset}/{lang}, skipping")
                        continue
                    items = [p[0] for p in pairs]
                    wavs = [p[1] for p in pairs]
                    print(f"[score] {metric_key} | {model_key} | {dataset}/{lang} ({len(items)} items)")
                    seg = metric.score_items(items, wavs)
                    corpus = metric.score_corpus(items, wavs)
                    seg_rows.extend(
                        {
                            "dataset": dataset,
                            "lang": lang,
                            "model": model_key,
                            "metric": metric_key,
                            "item_id": i["item_id"],
                            "score": s,
                        }
                        for i, s in zip(items, seg)
                    )
                    sys_rows.append(
                        {
                            "dataset": dataset,
                            "lang": lang,
                            "model": model_key,
                            "metric": metric_key,
                            "score": corpus,
                            "n_items": len(items),
                            "inputs_md5": inputs_md5,
                        }
                    )
        except Exception as e:
            print(f"[score] FAILED {metric_key}: {e}")
            seg_rows = [r for r in seg_rows if r["metric"] != metric_key]
            sys_rows = [r for r in sys_rows if r["metric"] != metric_key]

    if not sys_rows:
        print("[score] nothing to do — all requested results are cached")
        return

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    seg_path = SCORES_DIR / "segment_scores.csv"
    seg_new, sys_new = pd.DataFrame(seg_rows), pd.DataFrame(sys_rows)
    if seg_path.exists() and not args.overwrite:
        key = ["dataset", "lang", "model", "metric"]
        done = sys_new[key].drop_duplicates()
        old = pd.read_csv(seg_path).merge(done, on=key, how="left", indicator=True)
        seg_new = pd.concat([old[old["_merge"] == "left_only"].drop(columns="_merge"), seg_new])
        old_sys = pd.read_csv(sys_path).merge(done, on=key, how="left", indicator=True)
        sys_new = pd.concat([old_sys[old_sys["_merge"] == "left_only"].drop(columns="_merge"), sys_new])
    seg_new.to_csv(seg_path, index=False)
    sys_new.to_csv(sys_path, index=False)
    print(f"[score] wrote {seg_path} and {sys_path}")


def cmd_analyze(args):
    from .analyze import write_report

    write_report()


def main():
    from .metrics import ALL_METRICS, DEFAULT_METRICS

    parser = argparse.ArgumentParser(prog="ttseval")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--dataset", default=None, choices=DATASETS)
        p.add_argument("--langs", nargs="+", default=None, choices=["en", "es", "fr"])

    p = sub.add_parser("prepare", help="build voice-cloning test items")
    add_common(p)
    p.add_argument("--limit", type=int, default=200, help="items per (dataset, lang)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("synthesize", help="run TTS models over the test items")
    add_common(p)
    p.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_synthesize)

    p = sub.add_parser("score", help="score synthesized audio with the evaluation methods")
    add_common(p)
    p.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, choices=ALL_METRICS)
    p.add_argument("--rescore", action="store_true", help="recompute requested combos")
    p.add_argument("--overwrite", action="store_true", help="discard ALL previous scores")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("analyze", help="meta-evaluation report of metric (dis)agreement")
    p.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
