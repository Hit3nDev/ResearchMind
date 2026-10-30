"""Shared Pydantic models - the single source of truth for data shapes."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedInfo(BaseModel):
    model: str | None = None
    dataset: str | None = None
    accuracy: str | None = None
    precision: str | None = None
    recall: str | None = None
    limitations: list[str] = []


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = []
    year: int | None = None
    source_url: str = ""
    pdf_url: str | None = None
    venue: str | None = None
    abstract: str = ""
    full_text: str | None = None
    one_line_summary: str | None = None
    extracted: ExtractedInfo | None = None
    scores: dict[str, float] = {}      # per retrieval model (boolean, tfidf, bm25, hybrid)
    nlp: dict = {}                     # keywords, perplexity, readability ...


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    max_papers: int = Field(default=8, ge=1, le=25)


class QueryAccepted(BaseModel):
    query_id: str
    status: str


class QueryStatus(BaseModel):
    query_id: str
    question: str
    status: str          # processing | complete | failed
    stage: str           # searching | ranking | extracting_papers | summarizing | comparing | writing_review | formatting_citations | done
    papers_found: int = 0
    papers_processed: int = 0
    error: str | None = None


class Report(BaseModel):
    query_id: str
    question: str
    papers: list[str]
    comparison_table: list[dict]
    highlights: dict = {}
    gaps: list[str]
    literature_review: str
    review_method: str = "template_nlg"   # template_nlg | llm
    citations: dict[str, list[str]]       # keys: apa, ieee, mla, bibtex
    query_analysis: dict = {}
    retrieval: dict = {}


class ProjectCreate(BaseModel):
    query_id: str
    name: str = Field(min_length=1)


class Project(BaseModel):
    project_id: str
    name: str
    query_id: str
    question: str
    created_at: str


class TextIn(BaseModel):
    text: str = Field(min_length=1)


class TranslateIn(BaseModel):
    text: str = Field(min_length=1)
    target: str = "hi"


class RelevanceIn(BaseModel):
    relevant_ids: list[str]
