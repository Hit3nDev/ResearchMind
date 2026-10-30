"""Unit 4: Machine Translation of English output into Indian languages.

Backends are tried in order until one works:
  1. Claude (LLM-based MT)              - only when an API key is configured
  2. Google Translate (neural MT)       - via deep-translator
  3. MyMemory (translation memory + MT) - via deep-translator, 500 chars per request
"""
import logging

from deep_translator import GoogleTranslator, MyMemoryTranslator

from app.nlp import llm
from app.nlp.preprocessing import sentence_tokenize

log = logging.getLogger("litlens.translate")

INDIAN_LANGUAGES = {
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu",
    "or": "Odia", "as": "Assamese", "sa": "Sanskrit",
}
MYMEMORY_CODES = {"ur": "ur-PK"}   # everything else is "<code>-IN"

# Case study notes shown in the UI (Unit 4: characteristics of Indian languages).
INDIAN_LANGUAGE_NOTES = [
    "Word order is mostly Subject-Object-Verb (SOV), unlike English SVO, so MT must reorder constituents.",
    "Rich morphology: Dravidian languages (Tamil, Telugu, Kannada, Malayalam) are agglutinative; one word can carry tense, person, number and case.",
    "Postpositions instead of prepositions (e.g. Hindi 'ghar mein' = 'in the house').",
    "Relatively free word order with case markers, and grammatical gender on verbs and adjectives.",
    "Multiple scripts (Devanagari, Bengali, Tamil, Gurmukhi, Perso-Arabic...) and frequent code-mixing with English.",
    "Low-resource problem: far less parallel text than European languages, which is why initiatives like Bhashini exist.",
]


def _chunks(text: str, limit: int) -> list[str]:
    """Split on sentence boundaries so no request exceeds the service's character limit."""
    out, buf = [], ""
    for sent in sentence_tokenize(text):
        sent = sent[:limit]
        if buf and len(buf) + len(sent) + 1 > limit:
            out.append(buf)
            buf = ""
        buf = f"{buf} {sent}".strip()
    if buf:
        out.append(buf)
    return out


def _translate_paragraphs(text: str, fn, limit: int) -> str:
    return "\n\n".join(" ".join(fn(c) or "" for c in _chunks(p, limit)) if p.strip() else ""
                       for p in text.split("\n\n")).strip()


def translate(text: str, target: str = "hi") -> dict:
    if target not in INDIAN_LANGUAGES:
        raise ValueError(f"Unsupported language code '{target}'")
    lang = INDIAN_LANGUAGES[target]
    out = llm.call_text(f"Translate the user's text into {lang}. Output only the translation, keep paragraph breaks "
                        f"and keep technical names (models, datasets) in English.", text, effort="low")
    if out:
        return {"language": lang, "code": target, "translation": out, "engine": "Claude (LLM-based MT)"}
    errors = []
    backends = [
        ("Google Translate (neural MT)", GoogleTranslator(source="en", target=target).translate, 4500),
        ("MyMemory (translation memory + MT)",
         MyMemoryTranslator(source="en-GB", target=MYMEMORY_CODES.get(target, f"{target}-IN")).translate, 480),
    ]
    for name, fn, limit in backends:
        try:
            return {"language": lang, "code": target, "translation": _translate_paragraphs(text, fn, limit), "engine": name}
        except Exception as e:  # network blocked, rate limited, quota exceeded...
            log.info("%s failed: %s", name, e)
            errors.append(f"{name.split(' (')[0]}: {type(e).__name__}")
    raise RuntimeError("; ".join(errors))
