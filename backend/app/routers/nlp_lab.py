"""NLP Lab: the ten syllabus lab exercises, exposed as endpoints so they can be
run interactively from the UI on any text or on the corpus of a research query."""
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db import store
from app.models.schemas import TextIn, TranslateIn
from app.nlp import ir, preprocessing as pp, spell, wsd
from app.nlp.ngram_lm import NGramLM
from app.nlp.translate import INDIAN_LANGUAGE_NOTES, INDIAN_LANGUAGES, translate
from app.nlp.word2vec import Word2Vec, bag_of_words, one_hot

router = APIRouter(prefix="/nlp", tags=["nlp-lab"])


def corpus_for(query_id: str) -> str:
    q = store.get_query(query_id)
    if not q or not q.paper_ids:
        raise HTTPException(404, "No papers for this query yet")
    return "\n".join(f"{p.title}. {p.abstract}\n{p.full_text or ''}" for p in store.get_papers(q.paper_ids, with_text=True))


# Lab 1 - tokenization
@router.post("/tokenize")
def tokenize(body: TextIn):
    return {"sentences": pp.sentence_tokenize(body.text), "words": pp.word_tokenize(body.text),
            "regex_tokens": pp.regex_tokenize(body.text), "normalized": pp.normalize(body.text)}


# Lab 2 - stemming vs lemmatization
@router.post("/stem-lemma")
def stem_lemma(body: TextIn):
    return {"rows": pp.compare_stem_lemma(body.text)}


# Lab 3 - word frequency, stop words, POS tagging
@router.post("/frequency-pos")
def frequency_pos(body: TextIn):
    return {**pp.word_frequency(body.text), "pos_tags": pp.pos_tag(body.text)}


# Lab 4 - parsing
@router.post("/parse")
def parse(body: TextIn):
    first = pp.sentence_tokenize(body.text)[0] if body.text.strip() else body.text
    return {"chunk": pp.chunk_parse(first), "cfg": pp.cfg_parse(first), "sentence": first}


# Unit 2 - edit distance + spelling correction
class EditIn(BaseModel):
    a: str
    b: str


@router.post("/edit-distance")
def edit_distance(body: EditIn):
    dist, table = spell.edit_distance(body.a, body.b)
    return {"distance": dist, "table": table}


@router.post("/spell")
def spell_check(body: TextIn):
    tokens = pp.regex_tokenize(body.text)
    return {"fixes": spell.correct_text(tokens)}


# Lab 5 - N-gram next-word prediction
class NGramIn(BaseModel):
    prefix: str = ""
    query_id: str | None = None
    corpus: str | None = None
    n: int = 3
    sentence: str | None = None


@lru_cache(maxsize=8)
def _lm(key: str, text: str, n: int) -> NGramLM:
    return NGramLM(n=n, k=0.1).fit(text)


@router.post("/ngram")
def ngram(body: NGramIn):
    text = body.corpus or (corpus_for(body.query_id) if body.query_id else None)
    if not text:
        raise HTTPException(400, "Provide a corpus or a query_id")
    n = max(2, min(body.n, 4))
    lm = _lm(body.query_id or str(hash(text)), text, n)
    out = {"stats": lm.stats(), "predictions": lm.predict_next(body.prefix), "generated": lm.generate(body.prefix or "the", 12)}
    if body.sentence:
        out["sentence"] = lm.sentence_probability(body.sentence)
    return out


# Lab 6 - ambiguous words with WordNet
@router.post("/ambiguity")
def ambiguity(body: TextIn):
    return {"words": wsd.ambiguous_words(body.text)}


# Lab 8 - translation to Indian languages
@router.get("/languages")
def languages():
    return {"languages": INDIAN_LANGUAGES, "notes": INDIAN_LANGUAGE_NOTES}


@router.post("/translate")
def translate_text(body: TranslateIn):
    try:
        return translate(body.text, body.target)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Translation service unavailable: {e}")


# Lab 9 - Precision / Recall / F-measure on a toy IR system
class IREvalIn(BaseModel):
    documents: list[str]
    query: str
    relevant: list[int]          # indices of relevant documents
    k: int = 3


@router.post("/ir-eval")
def ir_eval(body: IREvalIn):
    docs = {f"d{i}": d for i, d in enumerate(body.documents)}
    index = ir.InvertedIndex(docs)
    relevant = {f"d{i}" for i in body.relevant}
    out = {}
    for name, scores in {"boolean": index.coordination_scores(body.query), "tfidf": index.tfidf_scores(body.query),
                         "bm25": index.bm25_scores(body.query)}.items():
        ranking = [d for d in ir.rank(scores) if scores[d] > 0]
        out[name] = {"ranking": ranking, "scores": {d: round(s, 3) for d, s in scores.items()},
                     **ir.precision_recall_f(ranking[:body.k], relevant),
                     "average_precision": ir.average_precision(ranking, relevant)}
    return {"models": out, "index": index.stats()}


# Lab 10 - Word2Vec (CBOW / Skip-gram)
class W2VIn(BaseModel):
    query_id: str | None = None
    corpus: str | None = None
    mode: str = "skipgram"
    word: str | None = None


_w2v_cache: dict[tuple, Word2Vec] = {}


@router.post("/word2vec")
def word2vec(body: W2VIn):
    text = body.corpus or (corpus_for(body.query_id) if body.query_id else None)
    if not text:
        raise HTTPException(400, "Provide a corpus or a query_id")
    mode = "cbow" if body.mode == "cbow" else "skipgram"
    key = (body.query_id or hash(text), mode)
    if key not in _w2v_cache:
        try:
            _w2v_cache[key] = Word2Vec(mode=mode, min_count=1 if len(text.split()) < 2000 else 2).fit(text[:400_000])
        except ValueError as e:
            raise HTTPException(400, str(e))
    m = _w2v_cache[key]
    word = (body.word or m.i2w[0]).lower()
    return {"mode": mode, "vocab_size": len(m.i2w), "dim": m.dim, "loss_per_epoch": m.losses,
            "word": word, "vector": m.vector(word), "similar": m.most_similar(word),
            "projection": m.projection_2d(50), "vocab_sample": m.i2w[:40]}


@router.post("/bow")
def bow(body: TextIn):
    sents = pp.sentence_tokenize(body.text)[:6]
    b = bag_of_words(sents)
    vocab = b["vocab"][:30]
    return {"sentences": sents, "vocab": vocab, "bow": [v[:30] for v in b["vectors"]],
            "one_hot_example": {"word": vocab[0], "vector": one_hot(vocab, vocab[0])} if vocab else None}
