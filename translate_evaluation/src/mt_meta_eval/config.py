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

DATASETS = ["wmt24pp", "wmt25"]

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
}

# Document-level WMT25 needs a much larger generation budget than WMT24++.
DEFAULT_MAX_NEW_TOKENS = {"wmt24pp": 1024, "wmt25": 4096}

# English language names used in Hy-MT2's translation prompt.
TARGET_LANGUAGE_NAMES = {
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
