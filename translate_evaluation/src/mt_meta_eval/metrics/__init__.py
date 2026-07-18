from .base import Metric

DEFAULT_METRICS = ["bleu", "chrf++", "ter", "comet22", "cometkiwi22", "script_purity"]
ALL_METRICS = DEFAULT_METRICS + ["metricx24", "llm_judge"]

# Metadata needed by the analysis step without instantiating (heavy) metrics.
HIGHER_IS_BETTER = {
    "bleu": True,
    "chrf++": True,
    "ter": False,
    "comet22": True,
    "cometkiwi22": True,
    "script_purity": True,
    "metricx24": False,
    "llm_judge": True,
}


def get_metric(key: str, **kwargs) -> Metric:
    if key == "bleu":
        from .lexical import BleuMetric

        return BleuMetric()
    if key == "chrf++":
        from .lexical import ChrfMetric

        return ChrfMetric()
    if key == "ter":
        from .lexical import TerMetric

        return TerMetric()
    if key == "comet22":
        from .comet_metrics import Comet22Metric

        return Comet22Metric(**kwargs)
    if key == "cometkiwi22":
        from .comet_metrics import CometKiwi22Metric

        return CometKiwi22Metric(**kwargs)
    if key == "script_purity":
        from .script_purity import ScriptPurityMetric

        return ScriptPurityMetric(**kwargs)
    if key == "metricx24":
        from .metricx import MetricX24Metric

        return MetricX24Metric(**kwargs)
    if key == "llm_judge":
        from .llm_judge import LLMJudgeMetric

        return LLMJudgeMetric(**kwargs)
    raise ValueError(f"Unknown metric {key!r}; choose from: {ALL_METRICS}")
