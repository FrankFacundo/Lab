#!/usr/bin/env bash
# Full evaluation pipeline: prepare -> translate (both models) -> score -> analyze.
#
# Safe to interrupt (Ctrl-C) and re-run at any time: prepared data is kept,
# translation resumes mid-pair, and already-scored results are cached.
#
# Progress bars: tqdm per translation batch, per lexical metric, and
# Lightning's bar for COMET. Override defaults via env vars, e.g.:
#   PAIRS="en-de_DE" DATASET=wmt24pp ./run_eval.sh
set -euo pipefail
cd "$(dirname "$0")"

DATASET=${DATASET:-wmt24pp}
PAIRS=${PAIRS:-"en-es_MX es_MX-en"}
METRICS=${METRICS:-"bleu chrf++ ter comet22 cometkiwi22"}
HY_BATCH=${HY_BATCH:-16}
GEMMA_BATCH=${GEMMA_BATCH:-8}
MTEVAL=.venv/bin/mteval

# shellcheck disable=SC2086  # word-splitting of $PAIRS/$METRICS is intended
{
  echo "== 1/5 prepare ($DATASET: $PAIRS) =="
  $MTEVAL prepare --dataset "$DATASET" --pairs $PAIRS

  echo "== 2/5 translate hy-mt2 =="
  $MTEVAL translate --dataset "$DATASET" --pairs $PAIRS --model hy-mt2 --batch-size "$HY_BATCH"

  echo "== 3/5 translate translategemma =="
  $MTEVAL translate --dataset "$DATASET" --pairs $PAIRS --model translategemma --batch-size "$GEMMA_BATCH"

  echo "== 4/5 score ($METRICS) =="
  $MTEVAL score --dataset "$DATASET" --pairs $PAIRS --metrics $METRICS

  echo "== 5/5 analyze =="
  $MTEVAL analyze
}

echo
echo "Done. Report: outputs/report.md"
