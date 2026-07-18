# MT metric meta-evaluation report

Models: hy-mt2-1.8b, hy-mt2-7b, translategemma-4b, translategemma-12b.
Test sets: WMT24++ (sentence-level, human post-edited references) and/or
WMT25 General MT (document-level, human references where available).
TER and MetricX-24 are error metrics (lower is better); all analyses below
orient every metric so that agreement/correlation is comparable.

## 1. System-level scores

|                                              |   bleu |   chrf++ |   comet22 |    ter |
|:---------------------------------------------|-------:|---------:|----------:|-------:|
| ('wmt24pp', 'en-de_DE', 'hy-mt2-1.8b')       | 29.289 |   58.48  |     0.807 | 57.077 |
| ('wmt24pp', 'en-de_DE', 'translategemma-4b') | 27.077 |   56.555 |     0.838 | 59.397 |
| ('wmt24pp', 'en-es_MX', 'hy-mt2-1.8b')       | 40.66  |   63.359 |     0.825 | 41.095 |
| ('wmt24pp', 'en-es_MX', 'translategemma-4b') | 34.725 |   59.627 |     0.828 | 48.07  |
| ('wmt24pp', 'es_MX-en', 'hy-mt2-1.8b')       | 38.439 |   61.814 |     0.837 | 42.414 |
| ('wmt24pp', 'es_MX-en', 'translategemma-4b') | 35.467 |   60.21  |     0.835 | 45.828 |
| ('wmt25', 'en-ja_JP', 'hy-mt2-1.8b')         | 17.211 |   23.005 |     0.85  | 86.087 |
| ('wmt25', 'en-ja_JP', 'translategemma-4b')   | 16.622 |   21.773 |     0.833 | 89.249 |

## 2. Which model does each metric prefer?

| dataset   | pair     | metric   | winner            |   margin |   n_distinct_winners |
|:----------|:---------|:---------|:------------------|---------:|---------------------:|
| wmt24pp   | en-de_DE | bleu     | hy-mt2-1.8b       |    2.212 |                    2 |
| wmt24pp   | en-de_DE | chrf++   | hy-mt2-1.8b       |    1.926 |                    2 |
| wmt24pp   | en-de_DE | comet22  | translategemma-4b |    0.031 |                    2 |
| wmt24pp   | en-de_DE | ter      | hy-mt2-1.8b       |    2.32  |                    2 |
| wmt24pp   | en-es_MX | bleu     | hy-mt2-1.8b       |    5.936 |                    2 |
| wmt24pp   | en-es_MX | chrf++   | hy-mt2-1.8b       |    3.732 |                    2 |
| wmt24pp   | en-es_MX | comet22  | translategemma-4b |    0.003 |                    2 |
| wmt24pp   | en-es_MX | ter      | hy-mt2-1.8b       |    6.975 |                    2 |
| wmt24pp   | es_MX-en | bleu     | hy-mt2-1.8b       |    2.971 |                    1 |
| wmt24pp   | es_MX-en | chrf++   | hy-mt2-1.8b       |    1.604 |                    1 |
| wmt24pp   | es_MX-en | comet22  | hy-mt2-1.8b       |    0.002 |                    1 |
| wmt24pp   | es_MX-en | ter      | hy-mt2-1.8b       |    3.414 |                    1 |
| wmt25     | en-ja_JP | bleu     | hy-mt2-1.8b       |    0.589 |                    1 |
| wmt25     | en-ja_JP | chrf++   | hy-mt2-1.8b       |    1.232 |                    1 |
| wmt25     | en-ja_JP | comet22  | hy-mt2-1.8b       |    0.017 |                    1 |
| wmt25     | en-ja_JP | ter      | hy-mt2-1.8b       |    3.162 |                    1 |

**Metrics disagree on the winner for: wmt24pp/en-de_DE, wmt24pp/en-es_MX.**

## 3. Segment-level correlation between metrics

Computed per dataset over all (pair, model, segment) points. Low correlation
between a lexical metric (BLEU/chrF/TER) and a neural metric (COMET/CometKiwi/
MetricX) is the usual source of 'papers disagree' effects.

| dataset   | metric_a   | metric_b   |    n |   pearson |   spearman |   kendall |
|:----------|:-----------|:-----------|-----:|----------:|-----------:|----------:|
| wmt24pp   | bleu       | chrf++     | 3856 |     0.854 |      0.887 |     0.723 |
| wmt24pp   | bleu       | comet22    | 3856 |     0.501 |      0.457 |     0.325 |
| wmt24pp   | bleu       | ter        | 3856 |     0.742 |      0.875 |     0.708 |
| wmt24pp   | chrf++     | comet22    | 3856 |     0.604 |      0.549 |     0.401 |
| wmt24pp   | chrf++     | ter        | 3856 |     0.721 |      0.822 |     0.653 |
| wmt24pp   | comet22    | ter        | 3856 |     0.479 |      0.491 |     0.355 |
| wmt25     | bleu       | chrf++     |    4 |     0.994 |      0.8   |     0.667 |
| wmt25     | bleu       | comet22    |    4 |     0.944 |      1     |     1     |
| wmt25     | bleu       | ter        |    4 |    -0.319 |     -0.8   |    -0.667 |
| wmt25     | chrf++     | comet22    |    4 |     0.947 |      0.8   |     0.667 |
| wmt25     | chrf++     | ter        |    4 |    -0.353 |     -0.6   |    -0.333 |
| wmt25     | comet22    | ter        |    4 |    -0.611 |     -0.8   |    -0.667 |

## 4. Pairwise preference agreement

For each segment, each metric picks the model it scores higher (ties dropped).
Agreement = fraction of segments where two metrics pick the same model.

| dataset   | metric_a   | metric_b   |   n_duels |   agreement |
|:----------|:-----------|:-----------|----------:|------------:|
| wmt24pp   | bleu       | chrf++     |      1801 |       0.849 |
| wmt24pp   | bleu       | comet22    |      1799 |       0.644 |
| wmt24pp   | bleu       | ter        |      1646 |       0.882 |
| wmt24pp   | chrf++     | comet22    |      1859 |       0.684 |
| wmt24pp   | chrf++     | ter        |      1662 |       0.856 |
| wmt24pp   | comet22    | ter        |      1659 |       0.673 |
| wmt25     | bleu       | chrf++     |         2 |       0.5   |
| wmt25     | bleu       | comet22    |         2 |       1     |
| wmt25     | bleu       | ter        |         2 |       0.5   |
| wmt25     | chrf++     | comet22    |         2 |       0.5   |
| wmt25     | chrf++     | ter        |         2 |       1     |
| wmt25     | comet22    | ter        |         2 |       0.5   |
