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
        "## 3. Item-level correlation between metrics",
        "",
        corr_tbl.to_markdown(index=False),
        "",
        "## 4. Pairwise preference agreement (pooled over model duels)",
        "",
        pref_tbl.to_markdown(index=False),
        "",
    ]
    report_path = REPORT_DIR / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"[analyze] wrote {report_path}")
