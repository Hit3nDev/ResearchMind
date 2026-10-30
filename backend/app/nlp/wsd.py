"""Unit 3: lexical semantics with WordNet - ambiguity detection, Word Sense
Disambiguation (Lesk) and synonym-based query expansion."""
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

from app.nlp.preprocessing import pos_tag, stop_words, wordnet_pos


def ambiguous_words(sentence: str, min_senses: int = 2) -> list[dict]:
    """Every content word with more than one WordNet sense is lexically ambiguous;
    the simplified Lesk algorithm picks the sense whose gloss overlaps the context most."""
    tagged = pos_tag(sentence)
    context = [w.lower() for w, _ in tagged]
    out = []
    for word, tag in tagged:
        w = word.lower()
        if not w.isalpha() or w in stop_words() or tag[:1] not in "NVJR":
            continue
        pos = wordnet_pos(tag)
        senses = wn.synsets(w, pos=pos)
        if len(senses) < min_senses:
            continue
        chosen = lesk(context, w, pos)
        out.append({
            "word": word, "pos": tag, "num_senses": len(senses),
            "senses": [{"name": s.name(), "definition": s.definition()} for s in senses[:5]],
            "lesk_sense": chosen.name() if chosen else None,
            "lesk_definition": chosen.definition() if chosen else None,
        })
    return out


def synonyms(word: str, limit: int = 4, context: list[str] | None = None) -> list[str]:
    """Synonyms of one sense only, keeping query expansion precise. With a context the sense
    is chosen by Lesk (WSD); without one, the most frequent sense is used. Expanding the
    wrong sense is a real failure mode: 'Indian' -> 'red indian' for 'Indian languages'."""
    senses = wn.synsets(word, pos=wn.NOUN)
    if not senses:
        return []
    sense = senses[0]
    if context and lesk(context, word, wn.NOUN) != sense:
        # Lesk and the most-frequent-sense baseline disagree -> too ambiguous to expand safely
        return []
    from app.nlp.spell import frequencies
    freq = frequencies()
    names = [l.name().replace("_", " ").lower() for l in sense.lemmas()]
    # drop rare/archaic synonyms (e.g. learn -> 'larn') that would hurt precision
    return [n for n in dict.fromkeys(names)
            if n != word.lower() and all(freq[w] >= 5 for w in n.split())][:limit]


def hypernyms(word: str) -> list[str]:
    senses = wn.synsets(word, pos=wn.NOUN)
    return [l.name().replace("_", " ") for h in (senses[0].hypernyms() if senses else []) for l in h.lemmas()][:3]
