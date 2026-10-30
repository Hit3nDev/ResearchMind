"""Unit 2: minimum edit distance and spelling error detection/correction.

Noisy-channel style: candidates are generated within edit distance 1 (then 2),
and ranked by (Levenshtein distance, -word frequency). Word frequencies come
from the Brown corpus plus a small research-domain lexicon."""
from collections import Counter
from functools import lru_cache

from nltk.corpus import brown
from nltk.corpus import wordnet as wn

LETTERS = "abcdefghijklmnopqrstuvwxyz"
# Research vocabulary that is rare in 1960s Brown text but common in queries.
DOMAIN_WORDS = """learning machine deep neural network networks model models detection classification
segmentation dataset datasets transformer transformers embedding embeddings language translation
retrieval generation summarization recognition prediction disease diseases crop crops plant
medical image images speech text semantic syntactic algorithm algorithms accuracy evaluation
supervised unsupervised reinforcement convolutional recurrent attention graph knowledge""".split()


def edit_distance(a: str, b: str, sub_cost: int = 1) -> tuple[int, list[list[int]]]:
    """Levenshtein distance via dynamic programming; returns the full DP table for display."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else sub_cost
            d[i][j] = min(d[i - 1][j] + 1,        # deletion
                          d[i][j - 1] + 1,        # insertion
                          d[i - 1][j - 1] + cost)  # substitution
    return d[n][m], d


@lru_cache(maxsize=1)
def frequencies() -> Counter:
    freq = Counter(w.lower() for w in brown.words() if w.isalpha())
    for w in DOMAIN_WORDS:
        freq[w] += 200
    return freq


@lru_cache(maxsize=1)
def wordnet_lemmas() -> frozenset[str]:
    return frozenset(wn.all_lemma_names())


def is_known(word: str) -> bool:
    w = word.lower()
    return w in frequencies() or w in wordnet_lemmas() or bool(wn.morphy(w))


def _edits1(word: str) -> set[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [l + r[1:] for l, r in splits if r]
    transposes = [l + r[1] + r[0] + r[2:] for l, r in splits if len(r) > 1]
    replaces = [l + c + r[1:] for l, r in splits if r for c in LETTERS]
    inserts = [l + c + r for l, r in splits for c in LETTERS]
    return set(deletes + transposes + replaces + inserts)


def correct_word(word: str, extra_vocab: set[str] | None = None) -> str:
    w = word.lower()
    if not w.isalpha() or len(w) <= 3 or is_known(w) or (extra_vocab and w in extra_vocab):
        return word
    freq = frequencies()
    known = lambda ws: {c for c in ws if c in freq or (extra_vocab and c in extra_vocab)}  # noqa: E731
    cands = known(_edits1(w)) or known(e2 for e1 in _edits1(w) for e2 in _edits1(e1))
    if not cands:
        return word
    return min(cands, key=lambda c: (edit_distance(w, c)[0], -freq[c], c))


def correct_text(tokens: list[str], extra_vocab: set[str] | None = None) -> list[dict]:
    """Returns only the tokens that were changed: [{original, corrected, distance}]."""
    fixes = []
    for t in tokens:
        c = correct_word(t, extra_vocab)
        if c.lower() != t.lower():
            fixes.append({"original": t, "corrected": c, "distance": edit_distance(t.lower(), c)[0]})
    return fixes
