"""Meta-evaluation: compare what the evaluation methods say about the same outputs.

Inputs: outputs/scores/segment_scores.csv and system_scores.csv (from `mteval score`).
Outputs: outputs/report.md plus supporting CSVs with
  1. system-level scores per (dataset, pair, model, metric),
  2. the winner each metric picks per pair — and where metrics disagree,
  3. segment-level rank correlations between metrics (per dataset),
  4. pairwise-preference agreement between metrics (which model wins per segment).

WMT24++ is sentence-level and WMT25 document-level, so segment-level analyses
are computed per dataset, never pooled across them.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .config import REPORT_DIR, SCORES_DIR
from .metrics import HIGHER_IS_BETTER


def _oriented(df: pd.DataFrame) -> pd.DataFrame:
    """Flip lower-is-better metric scores so that greater always means better."""
    df = df.copy()
    flip = df["metric"].map(lambda m: -1.0 if not HIGHER_IS_BETTER.get(m, True) else 1.0)
    df["oriented_score"] = df["score"] * flip
    return df


def system_table(system_df: pd.DataFrame) -> pd.DataFrame:
    return system_df.pivot_table(
        index=["dataset", "pair", "model"], columns="metric", values="score"
    ).round(3)


def winner_table(system_df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, pair, metric): which model wins and by how much."""
    rows = []
    oriented = _oriented(system_df)
    for (dataset, pair, metric), grp in oriented.groupby(["dataset", "pair", "metric"]):
        grp = grp.sort_values("oriented_score", ascending=False)
        best, second = grp.iloc[0], grp.iloc[-1]
        rows.append(
            {
                "dataset": dataset,
                "pair": pair,
                "metric": metric,
                "winner": best["model"],
                "margin": round(abs(best["score"] - second["score"]), 3),
            }
        )
    wt = pd.DataFrame(rows)
    flips = (
        wt.groupby(["dataset", "pair"])["winner"]
        .nunique()
        .rename("n_distinct_winners")
        .reset_index()
    )
    return wt.merge(flips, on=["dataset", "pair"])


def segment_correlations(seg_df: pd.DataFrame) -> pd.DataFrame:
    """Pearson/Spearman/Kendall between metric pairs over (model, segment) points."""
    rows = []
    for dataset, ds_df in _oriented(seg_df).groupby("dataset"):
        wide = ds_df.pivot_table(
            index=["pair", "model", "segment_id"],
            columns="metric",
            values="oriented_score",
        )
        for m1, m2 in itertools.combinations(list(wide.columns), 2):
            sub = wide[[m1, m2]].dropna()
            if len(sub) < 3 or sub[m1].nunique() < 2 or sub[m2].nunique() < 2:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "metric_a": m1,
                    "metric_b": m2,
                    "n": len(sub),
                    "pearson": round(stats.pearsonr(sub[m1], sub[m2])[0], 3),
                    "spearman": round(stats.spearmanr(sub[m1], sub[m2])[0], 3),
                    "kendall": round(stats.kendalltau(sub[m1], sub[m2])[0], 3),
                }
            )
    return pd.DataFrame(rows)


def preference_agreement(seg_df: pd.DataFrame) -> pd.DataFrame:
    """Per segment, each metric prefers one model; how often do metrics agree?"""
    rows = []
    for dataset, ds_df in _oriented(seg_df).groupby("dataset"):
        wide = ds_df.pivot_table(
            index=["pair", "segment_id"],
            columns=["metric", "model"],
            values="oriented_score",
        )
        metrics = sorted({m for m, _ in wide.columns})
        models = sorted({mod for _, mod in wide.columns})
        if len(models) != 2:
            continue

        prefs = {}
        for m in metrics:
            try:
                delta = wide[(m, models[0])] - wide[(m, models[1])]
            except KeyError:
                continue
            prefs[m] = np.sign(delta)

        for m1, m2 in itertools.combinations(prefs.keys(), 2):
            both = pd.concat([prefs[m1], prefs[m2]], axis=1, keys=["a", "b"]).dropna()
            both = both[(both["a"] != 0) & (both["b"] != 0)]  # drop ties
            if len(both) == 0:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "metric_a": m1,
                    "metric_b": m2,
                    "n_segments": len(both),
                    "agreement": round(float((both["a"] == both["b"]).mean()), 3),
                }
            )
    return pd.DataFrame(rows)


def write_report(models: list[str]) -> None:
    seg_df = pd.read_csv(SCORES_DIR / "segment_scores.csv")
    system_df = pd.read_csv(SCORES_DIR / "system_scores.csv")

    sys_tbl = system_table(system_df)
    win_tbl = winner_table(system_df)
    corr_tbl = segment_correlations(seg_df)
    pref_tbl = preference_agreement(seg_df)

    sys_tbl.to_csv(SCORES_DIR / "system_table.csv")
    win_tbl.to_csv(SCORES_DIR / "winner_table.csv", index=False)
    corr_tbl.to_csv(SCORES_DIR / "metric_correlations.csv", index=False)
    pref_tbl.to_csv(SCORES_DIR / "preference_agreement.csv", index=False)

    flips = win_tbl[win_tbl["n_distinct_winners"] > 1][["dataset", "pair"]]
    flip_labels = sorted({f"{d}/{p}" for d, p in flips.itertuples(index=False)})

    lines = [
        "# MT metric meta-evaluation report",
        "",
        f"Models: {', '.join(models)}.",
        "Test sets: WMT24++ (sentence-level, human post-edited references) and/or",
        "WMT25 General MT (document-level, human references where available).",
        "TER and MetricX-24 are error metrics (lower is better); all analyses below",
        "orient every metric so that agreement/correlation is comparable.",
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
            else "**All metrics agree on the winning model for every language pair.**"
        ),
        "",
        "## 3. Segment-level correlation between metrics",
        "",
        "Computed per dataset over all (pair, model, segment) points. Low correlation",
        "between a lexical metric (BLEU/chrF/TER) and a neural metric (COMET/CometKiwi/",
        "MetricX) is the usual source of 'papers disagree' effects.",
        "",
        corr_tbl.to_markdown(index=False),
        "",
        "## 4. Pairwise preference agreement",
        "",
        "For each segment, each metric picks the model it scores higher (ties dropped).",
        "Agreement = fraction of segments where two metrics pick the same model.",
        "",
        pref_tbl.to_markdown(index=False),
        "",
    ]
    report_path = REPORT_DIR / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"[analyze] wrote {report_path}")
