"""script_purity — penalizes characters from scripts foreign to the target language.

Motivation: Hy-MT2 sometimes injects CJK tokens into non-CJK output ("La看望ería").
Lexical metrics barely notice one bad token and neural metrics dilute it into an
average; this metric makes such corruption explicit and expensive.

Definition (segment level, 0-100, higher is better):
  allowed scripts = scripts of the target language
                    ∪ {Latin, Common, Inherited}   (URLs, digits, punctuation)
                    ∪ scripts present in the reference (legit quoted foreign text)
  score = max(0, 100 - char_cost * n_foreign_chars)      with char_cost = 25

A single intruding character costs 25 points; four or more zero the segment.
The corpus score is the mean, so it reads as "average segment purity" and its
gap from 100 tracks both how often and how badly a system corrupts output.
Reference-aware but tolerant: with no reference it degrades gracefully to the
whitelist alone.
"""

import regex

from ..config import target_code
from .base import Metric

# scripts we can identify; anything else falls through as "Unknown" (foreign
# unless the reference also contains it)
_KNOWN_SCRIPTS = [
    "Latin", "Cyrillic", "Greek", "Han", "Hiragana", "Katakana", "Hangul",
    "Arabic", "Hebrew", "Devanagari", "Bengali", "Tamil", "Telugu", "Kannada",
    "Gujarati", "Gurmukhi", "Thai", "Khmer", "Myanmar", "Lao", "Sinhala",
    "Georgian", "Armenian", "Ethiopic",
]
_SCRIPT_RE = {s: regex.compile(rf"\p{{Script={s}}}") for s in _KNOWN_SCRIPTS}
_ALWAYS_OK = regex.compile(r"[\p{Common}\p{Inherited}]")

# scripts of each target language (base code, before any locale suffix)
_TARGET_SCRIPTS = {
    "ru": {"Cyrillic"}, "uk": {"Cyrillic"}, "sr_Cyrl": {"Cyrillic"},
    "el": {"Greek"},
    "zh": {"Han"},
    "ja": {"Han", "Hiragana", "Katakana"},
    "ko": {"Hangul", "Han"},
    "ar": {"Arabic"}, "fa": {"Arabic"}, "ur": {"Arabic"},
    "he": {"Hebrew"},
    "hi": {"Devanagari"}, "mr": {"Devanagari"}, "bho": {"Devanagari"},
    "bn": {"Bengali"}, "ta": {"Tamil"}, "te": {"Telugu"},
    "kn": {"Kannada"}, "gu": {"Gujarati"},
    "th": {"Thai"}, "km": {"Khmer"}, "my": {"Myanmar"},
    # everything else (es, fr, de, en, it, cs, vi, tr, ...) defaults to Latin
}


def _char_script(c: str) -> str:
    for name, pat in _SCRIPT_RE.items():
        if pat.match(c):
            return name
    return "Unknown"


def _scripts_in(text: str) -> set[str]:
    return {_char_script(c) for c in text if not _ALWAYS_OK.match(c)}


def _target_scripts(pair: str) -> set[str]:
    code = target_code(pair)  # es_MX, zh_CN, sr_Cyrl_RS, es, ...
    for prefix in ("sr_Cyrl", "sr_Latn"):
        if code.startswith(prefix):
            return _TARGET_SCRIPTS.get(prefix, {"Latin"})
    return _TARGET_SCRIPTS.get(code.split("_")[0], {"Latin"})


class ScriptPurityMetric(Metric):
    key = "script_purity"
    requires_reference = False  # reference only widens the allowed set

    def __init__(self, char_cost: float = 25.0):
        self.char_cost = char_cost

    def _segment_score(self, hypothesis: str, reference: str, allowed: set[str]) -> float:
        allowed = allowed | _scripts_in(reference or "")
        n_foreign = sum(
            1
            for c in hypothesis
            if not _ALWAYS_OK.match(c) and _char_script(c) not in allowed
        )
        return max(0.0, 100.0 - self.char_cost * n_foreign)

    def score_segments(self, sources, hypotheses, references):
        allowed = _target_scripts(self.pair) | {"Latin"}
        return [
            self._segment_score(h, r, allowed)
            for h, r in zip(hypotheses, references)
        ]
