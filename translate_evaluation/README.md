# MT metric meta-evaluation

Model papers pick their own metrics and baselines, so their numbers rarely agree —
[Hy-MT2](https://arxiv.org/abs/2605.22064) and
[TranslateGemma](https://arxiv.org/abs/2601.09012) each claim wins on different
setups. This project runs an independent head-to-head on shared test sets
([WMT24++](https://arxiv.org/abs/2502.12404) and the
[WMT25 General MT test set](https://github.com/wmt-conference/wmt25-general-mt))
and scores the **same outputs with many evaluation methods**, so you can see
both which model wins and where the metrics themselves disagree.

**Test sets** (select with `--dataset`, default `wmt24pp`):

| Dataset | Granularity | Default pairs | Notes |
|---|---|---|---|
| `wmt24pp` | sentence/segment | en→de es fr ja ru zh | ~960 segs/pair, human post-edited refs |
| `wmt25` | **document** | en→cs ja ko ru uk zh | 87 docs/pair; only 16/31 pairs have human refs (`refA`) — en→de has none. cs→de, cs→uk, ja→zh also selectable |

**Models under test** (full precision, bfloat16):

- [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B) — sampling per model card (T=0.7, top_p=0.6, top_k=20, rep. penalty 1.05)
- [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) — greedy per model card

Each model runs with its own recommended decoding settings (as the papers do);
both are configurable in `src/mt_meta_eval/translators/`.

**Evaluation methods under evaluation:**

| Metric | Type | Scale | Notes |
|---|---|---|---|
| `bleu` | lexical n-gram | 0–100 ↑ | sacrebleu; segment scores use smoothed sentence-BLEU |
| `chrf++` | lexical char n-gram | 0–100 ↑ | sacrebleu, word_order=2 |
| `ter` | lexical edit rate | 0–100+ ↓ | sacrebleu |
| `comet22` | neural, reference-based | 0–1 ↑ | Unbabel/wmt22-comet-da |
| `cometkiwi22` | neural, reference-free QE | 0–1 ↑ | **gated**: accept terms at [Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) |
| `metricx24` | neural, hybrid (optional) | 0–25 ↓ | what the TranslateGemma paper uses; needs `pip install "git+https://github.com/google-research/metricx.git"` |
| `llm_judge` | LLM-as-judge (optional) | 0–100 ↑ | GEMBA-DA-style, Claude via Message Batches API; needs `ANTHROPIC_API_KEY` |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e ".[judge]"
hf auth login          # needed for the gated TranslateGemma + CometKiwi repos
```

Accept the license/terms pages once on Hugging Face:
[google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) and
[Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da).

## Run

One-shot launcher (progress bars for every stage; interrupt + relaunch freely —
data prep is kept, translation resumes mid-pair, scores are cached):

```bash
./run_eval.sh                                     # default: en-es_MX + es_MX-en
DATASET=wmt25 PAIRS="en-ja_JP" ./run_eval.sh      # any dataset/pairs/metrics
```

Or step by step:

```bash
# 1. Data (default dataset: wmt24pp; add --dataset wmt25 for WMT25)
.venv/bin/mteval prepare                       # full ~960 segments/pair
.venv/bin/mteval prepare --limit 200           # or a subsample for a faster pass
.venv/bin/mteval prepare --dataset wmt25       # WMT25 (87 documents/pair)

# 2. Translate (resumable; safe to interrupt and rerun)
.venv/bin/mteval translate --model hy-mt2
.venv/bin/mteval translate --model translategemma
.venv/bin/mteval translate --model hy-mt2 --dataset wmt25 --batch-size 2
.venv/bin/mteval translate --model translategemma --dataset wmt25 --batch-size 2

# 3. Score with the evaluation methods (results merge across runs/datasets)
.venv/bin/mteval score                                        # default 5 metrics
.venv/bin/mteval score --dataset wmt25
.venv/bin/mteval score --metrics metricx24 llm_judge          # optional extras

# 4. Meta-evaluation report (covers whatever has been scored so far)
.venv/bin/mteval analyze
```

Everything lands in `outputs/`:

- `outputs/translations/<model>/<pair>.jsonl` — hypotheses
- `outputs/scores/segment_scores.csv`, `system_scores.csv` — raw scores
- `outputs/report.md` — the deliverable:
  1. system-level score table per (pair, model, metric)
  2. the winner each metric picks per pair, flagging pairs where metrics flip
  3. segment-level Pearson/Spearman/Kendall correlation between metrics
  4. pairwise preference agreement (per segment, do two metrics pick the same model?)

All comparisons orient every metric so higher = better (TER and MetricX are flipped).

## Notes

- Runs on Apple Silicon (MPS) in bfloat16; also works on CUDA/CPU.
- **Everything is cached** — relaunching the pipeline never recomputes:
  `prepare` skips existing data files, `translate` resumes/skips per pair, and
  `score` skips any (dataset, pair, model, metric) combo whose inputs
  (hypotheses + data files) are byte-identical to what it scored before.
  Force with `--rescore` (recompute requested combos, keep the rest) or
  `--overwrite` (translate: redo pair; score: discard ALL previous scores).
- `mteval score` merges into existing CSVs, so you can add metrics
  incrementally (e.g. run `llm_judge` later) and re-run `analyze`.
- `llm_judge` defaults to `claude-opus-4-8` (override with `--judge-model`).
  It uses the Message Batches API (50% of standard token prices). Rough cost at
  full size: 6 pairs × 2 models × ~960 segments ≈ 11.5k judgments of ~250
  input tokens each — estimate before running if cost matters.
- WMT24++ is English-source only. Hy-MT2 claims 33 languages, TranslateGemma 55;
  the default pairs are in the intersection. To add pairs, pass `--pairs`
  (any `en-xx_XX` config of [google/wmt24pp](https://huggingface.co/datasets/google/wmt24pp))
  and add the language name to `TARGET_LANGUAGE_NAMES` in `config.py` if missing.
- **Reverse directions (X→en)**: pass a reversed wmt24pp pair like
  `es_MX-en` — it loads `en-es_MX` and swaps the sides, translating the human
  Spanish reference back into English and scoring against the original English.
  Caveat: the source is then translationese (a human translation, not original
  text), so reversed results are indicative rather than a WMT-blessed setup.
  WMT25 has no Spanish pairs at all.
- WMT25 is **document-level**: segments are whole paragraphs/articles (median
  ~800 chars, max ~28k). `translate` defaults to `--max-new-tokens 4096` there
  (vs 1024 for WMT24++); use `--max-new-tokens 8192` to be safe on the literary
  domain, and small `--batch-size` (1–2) to keep memory in check. Sentence-BLEU
  segment scores are least meaningful at document length — the neural metrics
  and the report's preference-agreement view matter more there.
