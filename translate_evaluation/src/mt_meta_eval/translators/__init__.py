from .base import Translator


def get_translator(key: str, **kwargs) -> Translator:
    if key == "hy-mt2":
        from .hy_mt2 import HyMT2Translator

        return HyMT2Translator(**kwargs)
    if key == "translategemma":
        from .translategemma import TranslateGemmaTranslator

        return TranslateGemmaTranslator(**kwargs)
    raise ValueError(f"Unknown model key {key!r}; choose from: hy-mt2, translategemma")


MODEL_KEYS = ["hy-mt2", "translategemma"]
