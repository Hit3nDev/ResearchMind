"""Summarize + extract agent: one LLM call per paper (JSON validated against
ExtractedInfo) with a classical TF-IDF / regex fallback."""
from app.models.schemas import ExtractedInfo, Paper
from app.nlp import extractive, llm


class LLMExtraction(ExtractedInfo):
    one_line_summary: str


SYSTEM = """You extract structured facts from research papers. Schema:
{
  "one_line_summary": string,
  "model": string or null,
  "dataset": string or null,
  "accuracy": string or null,
  "precision": string or null,
  "recall": string or null,
  "limitations": array of strings
}
Use null when the paper does not state a value. Metrics as percentages like "97.1%". At most 3 short limitations."""


def summarize_and_extract(paper: Paper) -> Paper:
    body = (paper.full_text or "")[:15000]
    result = llm.call_json(SYSTEM, f"Title: {paper.title}\n\nAbstract: {paper.abstract}\n\nFull text (truncated):\n{body}",
                           LLMExtraction)
    if result:
        paper.one_line_summary = result.one_line_summary
        paper.extracted = ExtractedInfo(**result.model_dump(exclude={"one_line_summary"}))
        paper.nlp["extraction_method"] = "llm"
    else:
        summary = extractive.summarize(paper.abstract or body, 1)
        words = summary.split()
        paper.one_line_summary = " ".join(words[:40]) + ("..." if len(words) > 40 else "")
        paper.extracted = extractive.extract_info(paper.abstract, paper.full_text)
        paper.nlp["extraction_method"] = "classical (TF-IDF summary + regex IE)"
    return paper
