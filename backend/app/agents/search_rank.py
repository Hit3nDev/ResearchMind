"""Search + rank agent.

1. Query understanding (Units 1-3): tokenize, POS tag, spell-correct, chunk-parse
   noun phrases, detect ambiguous words (WordNet + Lesk), expand with synonyms.
2. Retrieval: arXiv API + Semantic Scholar API, de-duplicated by title similarity.
3. Ranking (Unit 5): Boolean, TF-IDF vector space and BM25 over our own inverted
   index, fused with Reciprocal Rank Fusion. If sentence-transformers is
   installed, dense embeddings are added as a fourth ranker.
"""
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET

import httpx

from app.config import HTTP_RETRIES, HTTP_TIMEOUT, SEMANTIC_SCHOLAR_API_KEY
from app.models.schemas import Paper
from app.nlp import ir, wsd
from app.nlp.preprocessing import lemmatize, noun_phrases, pos_tag, regex_tokenize, stop_words
from app.nlp.spell import correct_text

log = logging.getLogger("litlens.search")
ATOM = {"a": "http://www.w3.org/2005/Atom"}


# ---------------- query understanding ----------------
def analyze_query(question: str) -> dict:
    tokens = regex_tokenize(question)
    fixes = correct_text(tokens)
    corrected = question
    for f in fixes:
        corrected = re.sub(rf"\b{re.escape(f['original'])}\b", f["corrected"], corrected)
    tagged = pos_tag(corrected)
    content = [w for w, t in tagged if t[:2] in ("NN", "JJ", "VB") and w.lower() not in stop_words() and w.isalnum()]
    keywords = list(dict.fromkeys(lemmatize(content)))
    search_terms = list(dict.fromkeys(w.lower() for w in content))
    phrases = [p for p in noun_phrases(corrected) if len(p.split()) > 1]
    # Expand only common nouns (NN/NNS), choosing the WordNet sense with Lesk over the query context
    context = [w.lower() for w, _ in tagged]
    nouns = list(dict.fromkeys(lemmatize([w for w, t in tagged if t in ("NN", "NNS") and w.lower() not in stop_words()])))
    expansions = {k: s for k in nouns[:4] if (s := wsd.synonyms(k, 3, context))}
    return {
        "original": question,
        "corrected": corrected,
        "spelling_fixes": fixes,
        "tokens": [w for w, _ in tagged],
        "pos_tags": tagged,
        "noun_phrases": phrases,
        "keywords": keywords,
        "search_terms": search_terms,
        "ambiguous_words": wsd.ambiguous_words(corrected),
        "expansions": expansions,
    }


def _get(url: str, params: dict, headers: dict | None = None) -> httpx.Response | None:
    for attempt in range(HTTP_RETRIES):
        try:
            r = httpx.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT, follow_redirects=True)
            if r.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            r.raise_for_status()
            return r
        except httpx.HTTPError as e:
            log.warning("GET %s failed (%s), attempt %d", url, e, attempt + 1)
            time.sleep(2 ** attempt)
    return None


def _pid(query_id: str, title: str) -> str:
    return "p_" + hashlib.sha1(f"{query_id}|{title.lower()}".encode()).hexdigest()[:10]


def search_arxiv(keywords: list[str], phrases: list[str], n: int, query_id: str) -> list[Paper]:
    terms = [f'all:"{p}"' for p in phrases[:1]] + [f"all:{k}" for k in keywords if " " not in k][:4]
    papers: list[Paper] = []
    for joiner in (" AND ", " OR "):           # strict first, relax if nothing comes back
        r = _get("http://export.arxiv.org/api/query",
                 {"search_query": joiner.join(terms), "max_results": n, "sortBy": "relevance"})
        if r is None:
            break
        root = ET.fromstring(r.text)
        for e in root.findall("a:entry", ATOM):
            title = re.sub(r"\s+", " ", e.findtext("a:title", "", ATOM)).strip()
            abs_url = e.findtext("a:id", "", ATOM)
            pdf = next((l.get("href") for l in e.findall("a:link", ATOM) if l.get("title") == "pdf"), None)
            papers.append(Paper(
                paper_id=_pid(query_id, title), title=title,
                authors=[a.findtext("a:name", "", ATOM) for a in e.findall("a:author", ATOM)],
                year=int(e.findtext("a:published", "0", ATOM)[:4] or 0) or None,
                source_url=abs_url, pdf_url=pdf, venue="arXiv",
                abstract=re.sub(r"\s+", " ", e.findtext("a:summary", "", ATOM)).strip()))
        if len(papers) >= max(3, n // 3):
            break
    return papers


def search_semantic_scholar(query: str, n: int, query_id: str) -> list[Paper]:
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else None
    r = _get("https://api.semanticscholar.org/graph/v1/paper/search",
             {"query": query, "limit": n,
              "fields": "title,authors,year,abstract,url,venue,openAccessPdf,externalIds"}, headers)
    if r is None:
        return []
    papers = []
    for d in r.json().get("data", []):
        if not d.get("abstract") or not d.get("title"):
            continue
        arxiv_id = (d.get("externalIds") or {}).get("ArXiv")
        pdf = (d.get("openAccessPdf") or {}).get("url") or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None)
        papers.append(Paper(
            paper_id=_pid(query_id, d["title"]), title=d["title"].strip(),
            authors=[a["name"] for a in d.get("authors", [])], year=d.get("year"),
            source_url=d.get("url") or "", pdf_url=pdf, venue=d.get("venue") or "Semantic Scholar",
            abstract=d["abstract"]))
    return papers


def _title_key(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Two titles are duplicates when their word-set Jaccard similarity is >= 0.8."""
    kept: list[Paper] = []
    for p in papers:
        k = _title_key(p.title)
        dup = next((q for q in kept if len(k & _title_key(q.title)) / (len(k | _title_key(q.title)) or 1) >= 0.8), None)
        if dup is None:
            kept.append(p)
        elif not dup.pdf_url and p.pdf_url:
            dup.pdf_url = p.pdf_url
    return kept


def _dense_scores(query: str, papers: list[Paper]) -> dict[str, float] | None:
    """Optional: Sentence-Transformers bi-encoder cosine similarity (pip install sentence-transformers)."""
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        return None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q = model.encode(query, convert_to_tensor=True)
    d = model.encode([f"{p.title}. {p.abstract}" for p in papers], convert_to_tensor=True)
    sims = util.cos_sim(q, d)[0]
    return {p.paper_id: float(s) for p, s in zip(papers, sims)}


def rank_papers(analysis: dict, papers: list[Paper], top_k: int) -> tuple[list[Paper], dict]:
    expanded = analysis["corrected"] + " " + " ".join(s for syns in analysis["expansions"].values() for s in syns[:1])
    index = ir.InvertedIndex({p.paper_id: f"{p.title} {p.title} {p.abstract}" for p in papers})  # title counted twice
    scores = {
        "boolean": index.coordination_scores(analysis["corrected"]),
        "tfidf": index.tfidf_scores(expanded),
        "bm25": index.bm25_scores(expanded),
    }
    dense = _dense_scores(analysis["corrected"], papers)
    if dense:
        scores["embedding"] = dense
    rankings = {name: ir.rank(s) for name, s in scores.items()}
    fused = ir.reciprocal_rank_fusion(list(rankings.values()))
    rankings["hybrid"] = ir.rank(fused)
    scores["hybrid"] = fused
    by_id = {p.paper_id: p for p in papers}
    for p in papers:
        p.scores = {name: round(s.get(p.paper_id, 0.0), 4) for name, s in scores.items()}
    boolean_expr = " AND ".join(k for k in analysis["search_terms"][:3])
    selected = [by_id[i] for i in rankings["hybrid"][:top_k]]
    retrieval = {
        "candidates": len(papers),
        "selected": len(selected),
        "models": list(scores),
        "rankings": rankings,
        "boolean_expression": boolean_expr,
        "boolean_matches": sorted(index.boolean(boolean_expr)) if boolean_expr else [],
        "index": index.stats(),
        "expanded_query": expanded,
        "titles": {p.paper_id: p.title for p in papers},
    }
    return selected, retrieval


def search_and_rank(question: str, max_papers: int, query_id: str, on_stage=None) -> tuple[list[Paper], dict, dict]:
    analysis = analyze_query(question)
    pool = max(max_papers * 2, 10)
    papers = search_arxiv(analysis["search_terms"], analysis["noun_phrases"], pool, query_id)
    papers += search_semantic_scholar(analysis["corrected"], pool, query_id)
    papers = deduplicate([p for p in papers if p.abstract])
    if on_stage:
        on_stage("ranking", len(papers))
    if not papers:
        return [], analysis, {"candidates": 0}
    selected, retrieval = rank_papers(analysis, papers, max_papers)
    return selected, analysis, retrieval
