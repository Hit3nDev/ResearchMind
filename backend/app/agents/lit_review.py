"""Literature review agent: LLM synthesis when a key is configured, otherwise
the template-based NLG pipeline from app.nlp.nlg (always used for the trace)."""
from app.models.schemas import Paper
from app.nlp import llm, nlg

SYSTEM = """You write academic literature reviews. Write 400-800 words in four parts:
Introduction, Comparative analysis, Research gaps, Conclusion (use these as plain-text headings
on their own lines). Cite papers inline as given (e.g. "Sharma et al. (2024)"). Use only the facts provided."""


def write_review(question: str, papers: list[Paper], table: list[dict], highlights: dict, gaps: list[str]) -> tuple[str, str, list[dict]]:
    template_text, trace = nlg.generate_review(question, papers, highlights, gaps)
    facts = "\n".join(
        f"- {r['ref']}: {p.title}. Summary: {p.one_line_summary}. Model: {r['model']}; dataset: {r['dataset']}; "
        f"accuracy: {r['accuracy']}." for r, p in zip(table, papers))
    prompt = f"Question: {question}\n\nPapers:\n{facts}\n\nHighlights: {highlights}\n\nGaps:\n" + "\n".join(f"- {g}" for g in gaps)
    text = llm.call_text(SYSTEM, prompt, effort="medium")
    if text:
        return text, "llm", trace
    return template_text, "template_nlg", trace
