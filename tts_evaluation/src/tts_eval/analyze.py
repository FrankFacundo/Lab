"""Meta-evaluation: compare what the TTS evaluation methods say about the same audio.

Same structure as translate_evaluation: system table, per-(dataset, lang) winners
with flip detection, item-level metric correlations, pairwise preference agreement
pooled over model duels. WER metrics are error rates (lower better) and are
orientation-flipped for all agreement analyses.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .config import REPORT_DIR, SCORES_DIR
from .metrics import HIGHER_IS_BETTER


def _oriented(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flip = df["metric"].map(lambda m: -1.0 if not HIGHER_IS_BETTER.get(m, True) else 1.0)
    df["oriented_score"] = df["score"] * flip
    return df


def system_table(system_df):
    return system_df.pivot_table(
        index=["dataset", "lang", "model"], columns="metric", values="score"
    ).round(3)


def winner_table(system_df):
    rows = []
    for (dataset, lang, metric), grp in _oriented(system_df).groupby(
        ["dataset", "lang", "metric"]
    ):
        grp = grp.sort_values("oriented_score", ascending=False)
        rows.append(
            {
                "dataset": dataset,
                "lang": lang,
                "metric": metric,
                "winner": grp.iloc[0]["model"],
                "margin": round(abs(grp.iloc[0]["score"] - grp.iloc[-1]["score"]), 3),
            }
        )
    wt = pd.DataFrame(rows)
    flips = (
        wt.groupby(["dataset", "lang"])["winner"]
        .nunique()
        .rename("n_distinct_winners")
        .reset_index()
    )
    return wt.merge(flips, on=["dataset", "lang"])


def item_correlations(seg_df):
    rows = []
    for (dataset, lang), ds_df in _oriented(seg_df).groupby(["dataset", "lang"]):
        wide = ds_df.pivot_table(
            index=["model", "item_id"], columns="metric", values="oriented_score"
        )
        for m1, m2 in itertools.combinations(list(wide.columns), 2):
            sub = wide[[m1, m2]].dropna()
            if len(sub) < 3 or sub[m1].nunique() < 2 or sub[m2].nunique() < 2:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "lang": lang,
                    "metric_a": m1,
                    "metric_b": m2,
                    "n": len(sub),
                    "pearson": round(stats.pearsonr(sub[m1], sub[m2])[0], 3),
                    "spearman": round(stats.spearmanr(sub[m1], sub[m2])[0], 3),
                }
            )
    return pd.DataFrame(rows)


def preference_agreement(seg_df):
    rows = []
    for (dataset, lang), ds_df in _oriented(seg_df).groupby(["dataset", "lang"]):
        wide = ds_df.pivot_table(
            index="item_id", columns=["metric", "model"], values="oriented_score"
        )
        metrics = sorted({m for m, _ in wide.columns})
        models = sorted({mod for _, mod in wide.columns})
        if len(models) < 2:
            continue
        prefs: dict[str, list] = {m: [] for m in metrics}
        for ma, mb in itertools.combinations(models, 2):
            for m in metrics:
                try:
                    prefs[m].append(np.sign(wide[(m, ma)] - wide[(m, mb)]))
                except KeyError:
                    continue
        prefs = {m: pd.concat(v, ignore_index=True) for m, v in prefs.items() if v}
        for m1, m2 in itertools.combinations(prefs.keys(), 2):
            both = pd.concat([prefs[m1], prefs[m2]], axis=1, keys=["a", "b"]).dropna()
            both = both[(both["a"] != 0) & (both["b"] != 0)]
            if len(both) == 0:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "lang": lang,
                    "metric_a": m1,
                    "metric_b": m2,
                    "n_duels": len(both),
                    "agreement": round(float((both["a"] == both["b"]).mean()), 3),
                }
            )
    return pd.DataFrame(rows)


def _score_label(metric: str, score: float) -> str:
    if metric.startswith("wer_"):
        return f"{score:.3f}%"
    return f"{score:.3f}"


def system_commentary(system_df):
    """Render a compact, data-driven interpretation of the system table."""
    labels = {
        "sim_wavlm": "WavLM similarity",
        "utmos": "UTMOS",
        "wer_whisper": "Whisper WER",
        "wer_qwen3asr": "Qwen3-ASR WER",
    }
    rows = []
    oriented = _oriented(system_df)
    for (dataset, lang), ds_df in oriented.groupby(["dataset", "lang"]):
        comparisons = []
        for metric, grp in ds_df.groupby("metric"):
            ranked = grp.sort_values("oriented_score", ascending=False)
            if len(ranked) < 2:
                continue
            winner, runner_up = ranked.iloc[0], ranked.iloc[1]
            margin = abs(float(winner["score"]) - float(runner_up["score"]))
            direction = "lower" if not HIGHER_IS_BETTER.get(metric, True) else "higher"
            if round(margin, 3) == 0:
                comparison = (
                    f"{labels.get(metric, metric)} is effectively tied at the displayed "
                    f"precision ({winner['model']} leads by {margin:.6f} unrounded)"
                )
            else:
                unit = " percentage points" if metric.startswith("wer_") else ""
                comparison = (
                    f"{labels.get(metric, metric)} favors {winner['model']} "
                    f"({_score_label(metric, winner['score'])} vs "
                    f"{_score_label(metric, runner_up['score'])}; "
                    f"{margin:.3f}{unit} {direction})"
                )
            comparisons.append(comparison)
        rows.append(f"- **{dataset}/{lang}:** " + "; ".join(comparisons) + ".")
    return rows


def correlation_commentary(corr_tbl):
    """Summarize the strongest observed relationships without overclaiming."""
    if corr_tbl.empty:
        return ["- Not enough paired observations were available to interpret correlations."]
    pearson_row = corr_tbl.loc[corr_tbl["pearson"].abs().idxmax()]
    spearman_row = corr_tbl.loc[corr_tbl["spearman"].abs().idxmax()]
    return [
        (
            f"- All reported relationships are weak: the largest absolute Pearson "
            f"correlation is {abs(pearson_row['pearson']):.3f} "
            f"({pearson_row['dataset']}/{pearson_row['lang']}, "
            f"{pearson_row['metric_a']} vs {pearson_row['metric_b']}), and the "
            f"largest absolute Spearman correlation is "
            f"{abs(spearman_row['spearman']):.3f} "
            f"({spearman_row['dataset']}/{spearman_row['lang']}, "
            f"{spearman_row['metric_a']} vs {spearman_row['metric_b']})."
        ),
        (
            "- In practical terms, intelligibility, speaker identity, and predicted "
            "naturalness are measuring different failure modes; a strong score on one "
            "does not reliably imply a strong score on another."
        ),
    ]


def preference_commentary(seg_df, pref_tbl):
    """Explain agreement rates and expose how many WER comparisons were ties."""
    if pref_tbl.empty:
        return ["- Not enough model duels were available to measure preference agreement."]
    lowest = pref_tbl.loc[pref_tbl["agreement"].idxmin()]
    highest = pref_tbl.loc[pref_tbl["agreement"].idxmax()]
    lines = [
        (
            f"- Agreement ranges from {lowest['agreement']:.3f} "
            f"({lowest['dataset']}/{lowest['lang']}, {lowest['metric_a']} vs "
            f"{lowest['metric_b']}) to {highest['agreement']:.3f} "
            f"({highest['dataset']}/{highest['lang']}, {highest['metric_a']} vs "
            f"{highest['metric_b']}). Values this close to 0.5 indicate that metric "
            "pairs often choose different model outputs on the same item."
        ),
        (
            "- `n_duels` excludes comparisons where either metric ties, so small values "
            "should not be read as equally strong evidence."
        ),
    ]

    wer = seg_df[seg_df["metric"] == "wer_whisper"]
    tie_notes = []
    for (dataset, lang), grp in wer.groupby(["dataset", "lang"]):
        wide = grp.pivot_table(index="item_id", columns="model", values="score").dropna()
        if wide.shape[1] != 2:
            continue
        ties = int(np.isclose(wide.iloc[:, 0], wide.iloc[:, 1]).sum())
        tie_notes.append(f"{dataset}/{lang}: {ties}/{len(wide)}")
    if tie_notes:
        lines.append(
            "- Whisper WER produces many exact model ties ("
            + ", ".join(tie_notes)
            + "); this is especially important when interpreting agreement on the "
            "English set."
        )
    return lines


def conclusion_commentary(win_tbl):
    """Build conclusions from the current set of system-level winners."""
    winner_counts = win_tbl["winner"].value_counts()
    leader = winner_counts.index[0]
    total = int(winner_counts.sum())
    unanimous = (
        win_tbl.groupby(["dataset", "lang"])["winner"]
        .nunique()
        .loc[lambda s: s == 1]
        .index
    )
    unanimous_labels = ", ".join(f"{d}/{l}" for d, l in unanimous) or "none"

    consistent_metrics = []
    for metric, grp in win_tbl.groupby("metric"):
        if grp["winner"].nunique() == 1:
            consistent_metrics.append(f"{metric} ({grp.iloc[0]['winner']})")
    consistent_text = ", ".join(consistent_metrics) or "none"

    return [
        (
            f"- **Most consistent overall system:** {leader} wins "
            f"{int(winner_counts.iloc[0])} of {total} system-level "
            "dataset/language/metric comparisons. This is a breadth result, not proof "
            "of universal superiority."
        ),
        (
            f"- **Full metric consensus:** {unanimous_labels}. On the remaining tracks, "
            "the preferred system changes with the evaluation objective."
        ),
        (
            f"- **Metric with a consistent winner across tracks:** {consistent_text}. "
            "Margins still matter: several differences are small enough that listening "
            "tests or repeated samples could change the practical conclusion."
        ),
        (
            "- **Selection guidance:** optimize Whisper WER when exact wording is the "
            "priority, WavLM similarity when voice identity is the priority, and UTMOS "
            "when naturalness is the priority. Do not average the raw values directly "
            "because their scales and meanings differ."
        ),
        (
            "- **Recommended next step:** validate the system-level result with a "
            "blinded human listening test covering naturalness, speaker similarity, "
            "and intelligibility as separate questions."
        ),
    ]


def write_report():
    seg_df = pd.read_csv(SCORES_DIR / "segment_scores.csv")
    system_df = pd.read_csv(SCORES_DIR / "system_scores.csv")

    sys_tbl = system_table(system_df)
    win_tbl = winner_table(system_df)
    corr_tbl = item_correlations(seg_df)
    pref_tbl = preference_agreement(seg_df)

    sys_tbl.to_csv(SCORES_DIR / "system_table.csv")
    win_tbl.to_csv(SCORES_DIR / "winner_table.csv", index=False)
    corr_tbl.to_csv(SCORES_DIR / "metric_correlations.csv", index=False)
    pref_tbl.to_csv(SCORES_DIR / "preference_agreement.csv", index=False)

    flips = win_tbl[win_tbl["n_distinct_winners"] > 1][["dataset", "lang"]]
    flip_labels = sorted({f"{d}/{l}" for d, l in flips.itertuples(index=False)})

    lines = [
        "# TTS metric meta-evaluation report",
        "",
        "Zero-shot voice cloning: reference audio + target text per item.",
        "Metrics: WER via Whisper-large-v3 (and optionally Qwen3-ASR), speaker",
        "similarity via WavLM-SV x-vectors, UTMOS predicted naturalness.",
        "WER is an error rate (lower is better); agreement analyses orient all",
        "metrics so higher = better. UTMOS is English-trained — treat es/fr values",
        "as comparative only.",
        "",
        "## 1. System-level scores",
        "",
        sys_tbl.to_markdown(),
        "",
        "### Commentary",
        "",
        *system_commentary(system_df),
        "",
        "## 2. Which model does each metric prefer?",
        "",
        win_tbl.to_markdown(index=False),
        "",
        (
            f"**Metrics disagree on the winner for: {', '.join(flip_labels)}.**"
            if flip_labels
            else "**All metrics agree on the winning model for every (dataset, lang).**"
        ),
        "",
        (
            "A zero margin after rounding should be treated as an operational tie, "
            "even though the unrounded value forces a winner in the table."
        ),
        "",
        "## 3. Item-level correlation between metrics",
        "",
        corr_tbl.to_markdown(index=False),
        "",
        "### Commentary",
        "",
        *correlation_commentary(corr_tbl),
        "",
        "## 4. Pairwise preference agreement (pooled over model duels)",
        "",
        pref_tbl.to_markdown(index=False),
        "",
        "### Commentary",
        "",
        *preference_commentary(seg_df, pref_tbl),
        "",
        "## 5. Conclusions",
        "",
        *conclusion_commentary(win_tbl),
        "",
        "## 6. Limitations",
        "",
        (
            "- UTMOS22 was trained on English MOS data. Its Spanish and French scores "
            "are best interpreted as within-language model comparisons, not absolute "
            "human-quality ratings."
        ),
        (
            "- WavLM similarity is a learned speaker-verification proxy. Very high "
            "MLS scores suggest a possible ceiling effect, so differences of only a "
            "few thousandths may have limited perceptual significance."
        ),
        (
            "- Whisper WER depends on the ASR model and text normalization. It can miss "
            "prosody, pronunciation quality, punctuation, and other audible defects "
            "when the recognized word sequence is unchanged."
        ),
        (
            "- The results cover 200 items per dataset/language/model in this run and "
            "do not include confidence intervals or significance tests. Small margins "
            "should therefore be treated as directional rather than definitive."
        ),
        (
            "- Correlations pool both systems within each dataset/language. They describe "
            "association among metric scores, not causal relationships or agreement "
            "with human preference."
        ),
        "",
    ]
    report_path = REPORT_DIR / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"[analyze] wrote {report_path}")
