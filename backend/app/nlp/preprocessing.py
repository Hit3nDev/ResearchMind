"""Unit 1 + Unit 2: tokenization, text normalization, regex, stop words,
stemming vs lemmatization, word frequency, POS tagging and parsing (NLTK)."""
import re
import string
from collections import Counter
from functools import lru_cache

import nltk

from app.config import BACKEND_DIR

NLTK_DIR = BACKEND_DIR / "nltk_data"
nltk.data.path.insert(0, str(NLTK_DIR))
NLTK_PACKAGES = [
    "punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "brown",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
]


def ensure_nltk_data() -> None:
    """Download the NLTK corpora once into backend/nltk_data (no admin rights needed)."""
    NLTK_DIR.mkdir(exist_ok=True)
    for pkg in NLTK_PACKAGES:
        if not (any(NLTK_DIR.glob(f"*/{pkg}*"))):
            nltk.download(pkg, download_dir=str(NLTK_DIR), quiet=True)


from nltk.corpus import stopwords, wordnet as wn  # noqa: E402
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer  # noqa: E402

_porter, _lancaster, _snowball = PorterStemmer(), LancasterStemmer(), SnowballStemmer("english")
_lemmatizer = WordNetLemmatizer()

# Research-domain words that are too generic to be useful search keywords.
DOMAIN_STOPWORDS = {
    "latest", "recent", "new", "current", "technique", "techniques", "method", "methods",
    "approach", "approaches", "paper", "papers", "study", "studies", "research", "use",
    "using", "used", "based", "different", "various", "best", "state", "art", "way", "ways",
    "work", "works", "survey", "review", "find", "help", "also", "et", "al", "fig", "table",
}


@lru_cache(maxsize=1)
def stop_words() -> frozenset[str]:
    return frozenset(stopwords.words("english")) | DOMAIN_STOPWORDS


# ---------------- Unit 1: tokenization ----------------
def sentence_tokenize(text: str) -> list[str]:
    return nltk.sent_tokenize(text)


def word_tokenize(text: str) -> list[str]:
    return nltk.word_tokenize(text)


def regex_tokenize(text: str) -> list[str]:
    """Rule-based tokenizer: keeps hyphenated words, numbers with decimals and model names like YOLOv8."""
    return re.findall(r"[A-Za-z]+(?:[-'][A-Za-z0-9]+)*[0-9]*|\d+(?:\.\d+)?%?", text)


# ---------------- Unit 2: normalization ----------------
_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize(text: str) -> str:
    """Case folding + punctuation removal + whitespace squeezing."""
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def wordnet_pos(treebank_tag: str) -> str:
    return {"J": wn.ADJ, "V": wn.VERB, "R": wn.ADV}.get(treebank_tag[:1], wn.NOUN)


def lemmatize(tokens: list[str]) -> list[str]:
    tagged = nltk.pos_tag(tokens)
    return [_lemmatizer.lemmatize(w.lower(), wordnet_pos(t)) for w, t in tagged]


def preprocess(text: str, lemma: bool = False) -> list[str]:
    """Full IR pipeline: normalize -> tokenize -> drop stop words -> stem (or lemmatize)."""
    tokens = [t for t in regex_tokenize(text.lower()) if t not in stop_words() and len(t) > 1]
    if lemma:
        return lemmatize(tokens)
    return [_porter.stem(t) for t in tokens]


def compare_stem_lemma(text: str) -> list[dict]:
    tokens = [t for t in word_tokenize(text) if t.isalpha()]
    lemmas = lemmatize(tokens)
    return [
        {"word": w, "porter": _porter.stem(w), "lancaster": _lancaster.stem(w),
         "snowball": _snowball.stem(w), "lemma": l}
        for w, l in zip(tokens, lemmas)
    ]


def word_frequency(text: str, top: int = 20) -> dict:
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    content = [t for t in tokens if t not in stop_words()]
    return {
        "total_tokens": len(tokens),
        "stopwords_removed": len(tokens) - len(content),
        "with_stopwords": Counter(tokens).most_common(top),
        "without_stopwords": Counter(content).most_common(top),
    }


def pos_tag(text: str) -> list[tuple[str, str]]:
    return nltk.pos_tag(word_tokenize(text))


# ---------------- Unit 2: parsing ----------------
# Cascaded regular-expression chunk grammar (shallow / partial parsing).
CHUNK_GRAMMAR = r"""
  NP: {<DT|PRP\$>?<JJ.*|VBN|VBG>*<NN.*>+}
      {<NNP>+}
  PP: {<IN><NP>}
  VP: {<MD>?<VB.*>+<RB.*>?<NP|PP>*}
"""
_chunker = nltk.RegexpParser(CHUNK_GRAMMAR)

# Small context-free grammar over POS tags for full (deep) parsing of short sentences.
_CFG_RULES = """
  S -> NP VP | VP | NP | S CC S
  NP -> DT NOM | NOM | PRP | NP PP | NP CC NP
  NOM -> ADJ NOM | N NOM | N
  ADJ -> 'JJ' | 'JJR' | 'JJS' | 'VBN' | 'VBG' | 'CD'
  N -> 'NN' | 'NNS' | 'NNP' | 'NNPS'
  VP -> V | V NP | V PP | V NP PP | MD VP | V ADVP | V VP | 'TO' VP
  V -> 'VB' | 'VBD' | 'VBG' | 'VBN' | 'VBP' | 'VBZ'
  ADVP -> 'RB'
  PP -> IN NP
  IN -> 'IN'
  DT -> 'DT' | 'WDT'
  PRP -> 'PRP'
  MD -> 'MD'
  CC -> 'CC'
"""


def chunk_parse(text: str) -> dict:
    tree = _chunker.parse(pos_tag(text))
    phrases = {"NP": [], "VP": [], "PP": []}
    for st in tree.subtrees(lambda t: t.label() in phrases):
        phrases[st.label()].append(" ".join(w for w, _ in st.leaves()))
    return {"tree": str(tree), "phrases": phrases}


def noun_phrases(text: str) -> list[str]:
    tree = _chunker.parse(pos_tag(text))
    out = []
    for st in tree.subtrees(lambda t: t.label() == "NP"):
        words = [w for w, tag in st.leaves() if tag not in ("DT", "PRP$") and w.lower() not in stop_words()]
        if words:
            out.append(" ".join(words))
    return out


def cfg_parse(text: str, max_words: int = 14) -> dict:
    """Chart (CYK-style, top-down) and recursive-descent parsing with a POS-level CFG."""
    tagged = [(w, t) for w, t in pos_tag(text) if t not in (".", ",", ":", "``", "''", "WP", "WRB", "EX")]
    tags = [t for _, t in tagged][:max_words]
    grammar = nltk.CFG.fromstring(_CFG_RULES)
    known = {str(p.rhs()[0]) for p in grammar.productions() if p.is_lexical()}
    if not tags or any(t not in known for t in tags):
        unknown = sorted({t for t in tags if t not in known})
        return {"ok": False, "message": f"Tags outside the demo grammar: {unknown or 'empty input'}", "tags": tags}
    parser = nltk.ChartParser(grammar)
    try:
        tree = next(parser.parse(tags))
    except StopIteration:
        return {"ok": False, "message": "No parse found with the demo grammar.", "tags": tags}
    # put the words back on the leaves
    words = iter(w for w, _ in tagged)
    for pos in tree.treepositions("leaves"):
        tree[pos] = f"{tree[pos]}:{next(words)}"
    return {"ok": True, "parser": "ChartParser (bottom-up chart over a POS-level CFG)",
            "tree": str(tree), "tags": tags}
