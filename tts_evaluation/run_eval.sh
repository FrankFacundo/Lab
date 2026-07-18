#!/usr/bin/env bash
# Full TTS evaluation: prepare -> synthesize (all models) -> score -> analyze.
# Default: seed-tts-eval (en) + MLS cloning pairs (es, fr), 200 items each,
# models qwen3-tts + s2-pro (+ step-audio-editx on en where its env exists).
#
# Safe to interrupt and re-run: data is kept, synthesis skips existing wavs,
# scores are cached against the audio+data fingerprint.
#   MTEVAL-style dry run:  TTSEVAL=echo ./run_eval.sh
set -euo pipefail
cd "$(dirname "$0")"

LIMIT=${LIMIT:-200}
MODELS=${MODELS:-"qwen3-tts s2-pro step-audio-editx"}
METRICS=${METRICS:-"wer_whisper sim_wavlm utmos"}
TTSEVAL=${TTSEVAL:-.venv/bin/ttseval}

echo "==== 1/4 prepare ===="
$TTSEVAL prepare --limit "$LIMIT"

echo "==== 2/4 synthesize ===="
# shellcheck disable=SC2086
$TTSEVAL synthesize --models $MODELS

echo "==== 3/4 score ($METRICS) ===="
# shellcheck disable=SC2086
$TTSEVAL score --models $MODELS --metrics $METRICS

echo "==== 4/4 analyze ===="
$TTSEVAL analyze

echo
echo "Done. Report: outputs/report.md"
