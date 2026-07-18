from ..config import MODEL_REGISTRY, model_source
from .base import Translator

MODEL_KEYS = list(MODEL_REGISTRY)


def get_translator(key: str, **kwargs) -> Translator:
    try:
        family, repo_id = MODEL_REGISTRY[key]
    except KeyError:
        raise ValueError(f"Unknown model key {key!r}; choose from: {MODEL_KEYS}") from None

    if family == "hy_mt2":
        from .hy_mt2 import HyMT2Translator as cls
    else:
        from .translategemma import TranslateGemmaTranslator as cls
    return cls(key=key, source=model_source(repo_id), **kwargs)
