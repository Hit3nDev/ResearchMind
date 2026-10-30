"""Multi-agent orchestration: search -> rank -> PDF extraction -> summarise/extract
-> compare + gaps -> literature review -> citations -> report."""
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

from app.agents import citation, compare_gap, lit_review, pdf_extract, search_rank, summarize_extract
from app.db import store
from app.models.schemas import Paper, Report
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from app.nlp.ngram_lm import NGramLM
from app.nlp.ir import InvertedIndex
from app.nlp.preprocessing import regex_tokenize

_stemmer = PorterStemmer()

log = logging.getLogger("litlens.pipeline")


def _process_paper(p: Paper) -> Paper:
    """PDF extraction + summarise/extract for one paper. Never raises."""
    try:
        if not p.full_text:
            p.full_text, meta = pdf_extract.extract_paper(p.pdf_url)
            p.nlp["pdf"] = {"pages": meta.get("pages"), "sections": meta.get("sections", {})} if meta else {"pages": None}
        summarize_extract.summarize_and_extract(p)
    except Exception as e:
        log.warning("paper %s failed: %s", p.paper_id, e)
        p.nlp["error"] = str(e)
    return p


def corpus_nlp(papers: list[Paper]) -> None:
    """Per-paper key terms (TF-IDF) and perplexity under a trigram LM trained on the other papers."""
    if not papers:
        return
    texts = {p.paper_id: f"{p.title}. {p.abstract} {(p.full_text or '')[:20000]}" for p in papers}
    index = InvertedIndex(texts)
    # The index stores Porter stems ("classi"); show each stem as its most frequent real word ("classification").
    surface: dict[str, Counter] = defaultdict(Counter)
    for text in texts.values():
        for w in regex_tokenize(text.lower()):
            if w.isalpha() and len(w) > 2:
                surface[_stemmer.stem(w)][w] += 1
    for p in papers:
        vec = {t: index.tf_weight(post[p.paper_id]) * index.idf(t) for t, post in index.postings.items()
               if p.paper_id in post and t.isalpha() and len(t) > 2 and t in surface}
        p.nlp["key_terms"] = [surface[t].most_common(1)[0][0] for t, _ in sorted(vec.items(), key=lambda kv: -kv[1])[:8]]
        others = " ".join(t for pid, t in texts.items() if pid != p.paper_id)
        if len(papers) > 1:
            p.nlp["abstract_perplexity"] = NGramLM(3, 0.1).fit(others).perplexity(p.abstract)


def build_report(query_id: str, question: str, papers: list[Paper], analysis: dict, retrieval: dict, on_stage=None) -> Report:
    stage = on_stage or (lambda *_: None)
    stage("comparing")
    table, highlights = compare_gap.compare(papers)
    gaps, gap_method = compare_gap.find_gaps(papers, table)
    stage("writing_review")
    review, method, trace = lit_review.write_review(analysis.get("corrected", question), papers, table, highlights, gaps)
    stage("formatting_citations")
    report = Report(
        query_id=query_id, question=question, papers=[p.paper_id for p in papers],
        comparison_table=table, highlights=highlights, gaps=gaps, literature_review=review,
        review_method=method, citations=citation.format_citations(papers),
        query_analysis=analysis, retrieval={**retrieval, "gap_method": gap_method, "nlg_trace": trace},
    )
    store.save_report(report)
    return report


def run_pipeline(query_id: str, question: str, max_papers: int) -> None:
    def stage(name: str, found: int | None = None):
        fields = {"stage": name}
        if found is not None:
            fields["papers_found"] = found
        store.update_query(query_id, **fields)

    try:
        stage("searching")
        papers, analysis, retrieval = search_rank.search_and_rank(question, max_papers, query_id, on_stage=stage)
        if not papers:
            store.update_query(query_id, status="complete", stage="done", papers_found=0,
                               error="No papers found. Try a broader or differently worded question.")
            build_report(query_id, question, [], analysis, retrieval)
            return
        store.update_query(query_id, stage="extracting_papers", papers_found=len(papers),
                           paper_ids=[p.paper_id for p in papers])
        for p in papers:
            store.save_paper(p, query_id)

        done = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            for p in pool.map(_process_paper, papers):
                done += 1
                store.save_paper(p, query_id)
                store.update_query(query_id, papers_processed=done,
                                   stage="summarizing" if done > len(papers) // 2 else "extracting_papers")
        corpus_nlp(papers)
        for p in papers:
            store.save_paper(p, query_id)
        build_report(query_id, question, papers, analysis, retrieval, on_stage=stage)
        store.update_query(query_id, status="complete", stage="done")
    except Exception as e:
        log.error("pipeline failed: %s", traceback.format_exc())
        store.update_query(query_id, status="failed", error=str(e))


def rebuild_report(query_id: str) -> Report | None:
    """Re-run the synthesis stages (used after a user uploads an extra PDF)."""
    q = store.get_query(query_id)
    old = store.get_report(query_id)
    if not q:
        return None
    papers = store.get_papers(q.paper_ids, with_text=True)
    return build_report(query_id, q.question, papers, old.query_analysis if old else {}, old.retrieval if old else {})
