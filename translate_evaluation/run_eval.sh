#!/usr/bin/env bash
# Full evaluation pipeline: prepare -> translate (all models) -> score, per
# dataset, then one combined analyze.
#
# Default: wmt24pp en<->es  AND  flores200 (en<->es, en<->fr, es<->fr),
# with all four models (Hy-MT2 1.8B/7B, TranslateGemma 4B/12B) loaded from
# /Users/frankfacundo/Models when present.
#
# Overrides:
#   DATASET=wmt25 PAIRS="en-ja_JP" ./run_eval.sh      # single custom evaluation
#   MODELS="hy-mt2-7b:8 translategemma-12b:4" ./run_eval.sh   # subset, key:batch
#   MTEVAL=echo ./run_eval.sh                          # dry run
#
# Safe to interrupt (Ctrl-C) and re-run at any time: prepared data is kept,
# translation resumes mid-pair, and already-scored results are cached.
set -euo pipefail
cd "$(dirname "$0")"

METRICS=${METRICS:-"bleu chrf++ ter comet22 cometkiwi22 script_purity"}
MODELS=${MODELS:-"hy-mt2-1.8b:16 hy-mt2-7b:8 translategemma-4b:8 translategemma-12b:4"}
MODEL_KEYS=$(sed -E 's/:[0-9]+//g' <<<"$MODELS")
MTEVAL=${MTEVAL:-.venv/bin/mteval}

# Each evaluation is "dataset|pairs"; empty pairs = that dataset's defaults.
if [[ -n "${DATASET:-}" || -n "${PAIRS:-}" ]]; then
  EVALS=("${DATASET:-wmt24pp}|${PAIRS:-}")
else
  EVALS=(
    "wmt24pp|en-es_MX es_MX-en"
    "flores200|"   # defaults: en-es es-en en-fr fr-en es-fr fr-es
  )
fi

for spec in "${EVALS[@]}"; do
  ds=${spec%%|*}
  pairs=${spec#*|}
  args=(--dataset "$ds")
  # shellcheck disable=SC2206  # word-splitting of $pairs is intended
  [[ -n "$pairs" ]] && args+=(--pairs $pairs)

  echo "==== [$ds] prepare ===="
  $MTEVAL prepare "${args[@]}"

  for mspec in $MODELS; do
    model=${mspec%%:*}
    batch=${mspec##*:}
    echo "==== [$ds] translate $model (batch $batch) ===="
    $MTEVAL translate "${args[@]}" --model "$model" --batch-size "$batch"
  done

  echo "==== [$ds] score ($METRICS) ===="
  # shellcheck disable=SC2086
  $MTEVAL score "${args[@]}" --models $MODEL_KEYS --metrics $METRICS
done

echo "==== analyze ===="
$MTEVAL analyze

echo
echo "Done. Report: outputs/report.md"
