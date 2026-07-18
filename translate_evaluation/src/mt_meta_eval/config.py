"""Shared configuration: paths, datasets, language pairs, model/metric registries."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
TRANSLATIONS_DIR = PROJECT_ROOT / "outputs" / "translations"
SCORES_DIR = PROJECT_ROOT / "outputs" / "scores"
REPORT_DIR = PROJECT_ROOT / "outputs"

WMT24PP_DATASET = "google/wmt24pp"
WMT25_URL = (
    "https://github.com/wmt-conference/wmt25-general-mt"
    "/raw/refs/heads/main/data/wmt25-genmt.jsonl"
)
# FLORES+ (successor of FLORES-200), maintained by OLDI. Gated (auto-accept):
# click "Agree" once at https://huggingface.co/datasets/openlanguagedata/flores_plus
FLORES_DATASET = "openlanguagedata/flores_plus"

DATASETS = ["wmt24pp", "wmt25", "flores200"]

# flores200 pairs use bare language codes ("es-fr"); this maps them to
# FLORES+ configs. Multi-way parallel, so any combination of these works.
FLORES_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "cs": "ces_Latn",
    "pl": "pol_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "ar": "arb_Arab",
    "he": "heb_Hebr",
    "tr": "tur_Latn",
    "hi": "hin_Deva",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
}

# wmt24pp: sentence/segment-level, en->X, human post-edited references.
# wmt25: document-level; only pairs with a human reference (refA) are usable —
# note en-de_DE has NO reference in WMT25. Defaults are en-source pairs with
# refs that both models support; cs-de_DE, cs-uk_UA, ja-zh_CN also have refs.
DEFAULT_PAIRS = {
    "wmt24pp": [
        "en-de_DE",
        "en-es_MX",
        "en-fr_FR",
        "en-ja_JP",
        "en-ru_RU",
        "en-zh_CN",
    ],
    "wmt25": [
        "en-cs_CZ",
        "en-ja_JP",
        "en-ko_KR",
        "en-ru_RU",
        "en-uk_UA",
        "en-zh_CN",
    ],
    # multi-way parallel: includes the non-English es<->fr directions
    "flores200": ["en-es", "es-en", "en-fr", "fr-en", "es-fr", "fr-es"],
}

# Document-level WMT25 needs a much larger generation budget than WMT24++;
# FLORES sentences are short.
DEFAULT_MAX_NEW_TOKENS = {"wmt24pp": 1024, "wmt25": 4096, "flores200": 512}

# English language names used in Hy-MT2's translation prompt.
TARGET_LANGUAGE_NAMES = {
    # bare codes (flores200 pairs)
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "cs": "Czech",
    "pl": "Polish",
    "nl": "Dutch",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ar": "Arabic",
    "he": "Hebrew",
    "tr": "Turkish",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Simplified Chinese",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    # locale codes (wmt24pp / wmt25 pairs)
    "ar_EG": "Arabic",
    "bho_IN": "Bhojpuri",
    "bn_BD": "Bengali",
    "bn_IN": "Bengali",
    "cs_CZ": "Czech",
    "de_DE": "German",
    "el_GR": "Greek",
    "es_MX": "Spanish",
    "et_EE": "Estonian",
    "fa_IR": "Persian",
    "fil_PH": "Filipino",
    "fr_FR": "French",
    "gu_IN": "Gujarati",
    "he_IL": "Hebrew",
    "hi_IN": "Hindi",
    "id_ID": "Indonesian",
    "is_IS": "Icelandic",
    "it_IT": "Italian",
    "ja_JP": "Japanese",
    "km_KH": "Khmer",
    "kn_IN": "Kannada",
    "ko_KR": "Korean",
    "lt_LT": "Lithuanian",
    "mas_KE": "Maasai",
    "mr_IN": "Marathi",
    "ms_MY": "Malay",
    "my_MM": "Burmese",
    "nl_NL": "Dutch",
    "pl_PL": "Polish",
    "pt_BR": "Portuguese",
    "ro_RO": "Romanian",
    "ru_RU": "Russian",
    "sr_Cyrl_RS": "Serbian (Cyrillic)",
    "sr_Latn_RS": "Serbian (Latin)",
    "sv_SE": "Swedish",
    "ta_IN": "Tamil",
    "te_IN": "Telugu",
    "th_TH": "Thai",
    "tr_TR": "Turkish",
    "uk_UA": "Ukrainian",
    "ur_PK": "Urdu",
    "vi_VN": "Vietnamese",
    "zh_CN": "Simplified Chinese",
    "zh_TW": "Traditional Chinese",
}

SEED = 42

# Models under test. Weights are loaded from MODELS_ROOT/<repo_id> when that
# local copy exists (it does for all four), falling back to the HF hub id.
MODELS_ROOT = Path("/Users/frankfacundo/Models")

MODEL_REGISTRY = {
    "hy-mt2-1.8b": ("hy_mt2", "tencent/Hy-MT2-1.8B"),
    "hy-mt2-7b": ("hy_mt2", "tencent/Hy-MT2-7B"),
    "translategemma-4b": ("translategemma", "google/translategemma-4b-it"),
    "translategemma-12b": ("translategemma", "google/translategemma-12b-it"),
}


def model_source(repo_id: str) -> str:
    local = MODELS_ROOT / repo_id
    return str(local) if (local / "config.json").exists() else repo_id


def data_dir(dataset: str) -> Path:
    return DATA_ROOT / dataset


def source_code(pair: str) -> str:
    """'en-de_DE' -> 'en', 'cs-de_DE' -> 'cs'"""
    return pair.split("-", 1)[0]


def target_code(pair: str) -> str:
    """'en-de_DE' -> 'de_DE'"""
    return pair.split("-", 1)[1]


def target_language_name(pair: str) -> str:
    code = target_code(pair)
    try:
        return TARGET_LANGUAGE_NAMES[code]
    except KeyError:
        raise ValueError(
            f"No language name registered for {code!r}; add it to "
            "TARGET_LANGUAGE_NAMES in config.py"
        ) from None
