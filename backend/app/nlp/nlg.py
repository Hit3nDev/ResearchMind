"""Unit 4: template-based Natural Language Generation.

Follows the classic Reiter & Dale NLG pipeline and records what each stage did:
  1. Content determination  - choose the facts worth saying
  2. Document planning      - order them into sections (intro, themes, comparison, gaps, conclusion)
  3. Microplanning          - aggregation, referring expressions, lexical choice
  4. Surface realisation    - fill sentence templates, punctuation, list joining
"""
import random
import re
from collections import defaultdict

FAMILIES = {
    "Transformer / attention models": r"transformer|vit\b|bert|gpt|attention|swin|t5|llama|deberta|detr|clip",
    "Object-detection networks": r"yolo|r-cnn|rcnn|ssd|detr|retinanet",
    "Convolutional neural networks": r"cnn|resnet|vgg|efficientnet|mobilenet|densenet|inception|alexnet|convnext|u-net",
    "Recurrent / sequence models": r"lstm|gru|rnn|seq2seq",
    "Classical machine learning": r"svm|random forest|xgboost|k-nn|knn|naive bayes|decision tree|logistic regression",
    "Graph and generative models": r"gnn|gcn|gan|diffusion|vae",
}
REPORT_VERBS = ["propose", "present", "introduce", "investigate", "develop", "evaluate"]


def join_list(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + (", and " if len(items) > 2 else " and ") + items[-1]


def short_ref(paper) -> str:
    """Referring expression: 'Sharma et al. (2024)' / 'Sharma and Lee (2024)'."""
    surnames = [a.split()[-1] for a in paper.authors if a.strip()] or ["Anonymous"]
    who = surnames[0] if len(surnames) == 1 else (
        f"{surnames[0]} and {surnames[1]}" if len(surnames) == 2 else f"{surnames[0]} et al.")
    return f"{who} ({paper.year or 'n.d.'})"


def rewrite_summary(summary: str | None, max_words: int = 35) -> str:
    """Referring-expression rewrite so a first-person summary reads well inside a review."""
    s = re.sub(r"\s+", " ", (summary or "")).strip().rstrip(".")
    if not s:
        return ""
    s = re.sub(r"^(?:In this (?:paper|work|study),?\s*)?[Ww]e\b", "The authors", s)
    s = re.sub(r"^(?:This|The present|Our) (?:paper|research|study|work|article)\b", "The study", s)
    words = s.split()
    s = " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")
    return s[0].upper() + s[1:] + ("" if s.endswith("...") else ".")


def family_of(model: str | None) -> str:
    m = (model or "").lower()
    for fam, pat in FAMILIES.items():
        if re.search(pat, m):
            return fam
    return "Other approaches"


def _pct(v: str | None) -> float | None:
    m = re.search(r"\d+(?:\.\d+)?", v or "")
    return float(m.group()) if m else None


def generate_review(question: str, papers: list, highlights: dict, gaps: list[str], seed: int = 0) -> tuple[str, list[dict]]:
    rnd = random.Random(seed)
    trace = []

    # 1. Content determination
    facts = [p for p in papers if p.extracted or p.one_line_summary]
    years = sorted(p.year for p in facts if p.year)
    trace.append({"stage": "Content determination",
                  "detail": f"Selected {len(facts)} papers with extracted facts (model, dataset, metrics, summary)."})
    if not facts:
        return "No relevant papers were found for this question, so no review could be generated.", trace

    # 2. Document planning: group by method family -> one thematic paragraph each
    groups: dict[str, list] = defaultdict(list)
    for p in facts:
        groups[family_of(p.extracted.model if p.extracted else None)].append(p)
    plan = ["Introduction"] + [f"Theme: {g}" for g in groups] + ["Comparative analysis", "Research gaps", "Conclusion"]
    trace.append({"stage": "Document planning", "detail": " -> ".join(plan)})

    # 3 + 4. Microplanning and surface realisation
    paras = []
    span = f"between {years[0]} and {years[-1]}" if len(years) > 1 and years[0] != years[-1] else (
        f"in {years[0]}" if years else "recently")
    paras.append(
        f"This review synthesises {len(facts)} studies published {span} that address the question: "
        f"“{question.strip().rstrip('?')}?” The literature clusters into "
        f"{len(groups)} methodological theme{'s' if len(groups) != 1 else ''}: {join_list([g.lower() for g in groups])}.")

    aggregated = 0
    for fam, ps in groups.items():
        sents = []
        models = sorted({p.extracted.model for p in ps if p.extracted and p.extracted.model})
        if len(ps) > 1 and models:
            # Aggregation: several papers sharing a theme are mentioned in a single sentence.
            sents.append(f"{fam} are represented by {len(ps)} studies, which employ {join_list(models)}.")
            aggregated += 1
        for p in ps:
            ex = p.extracted
            verb = rnd.choice(REPORT_VERBS)
            if len(p.authors) == 1:                       # subject-verb agreement
                verb += "es" if verb.endswith("sh") else "s"
            method = f"a {ex.model}-based approach" if ex and ex.model else "an approach"
            s = f"{short_ref(p)} {verb} {method}"
            details = []
            if ex and ex.dataset:
                details.append(f"evaluated on {ex.dataset}")
            if ex and ex.accuracy:
                details.append(f"reporting {ex.accuracy} accuracy")
            if details:
                s += ", " + join_list(details)
            sents.append(s + ".")
            summary = rewrite_summary(p.one_line_summary)
            if summary:
                sents.append(summary)
        paras.append(" ".join(sents))
    trace.append({"stage": "Microplanning",
                  "detail": f"Aggregated {aggregated} groups into summary sentences; built author-year referring "
                            f"expressions; varied reporting verbs ({', '.join(REPORT_VERBS[:4])}...)."})

    comp = []
    if highlights.get("highest_accuracy"):
        h = highlights["highest_accuracy"]
        comp.append(f"In terms of reported performance, {h['ref']} achieves the highest accuracy ({h['value']}) "
                    f"using {h.get('model') or 'its proposed method'}")
    if highlights.get("most_recent"):
        h = highlights["most_recent"]
        comp.append(f"the most recent contribution is {h['ref']}")
    datasets = sorted({p.extracted.dataset for p in facts if p.extracted and p.extracted.dataset})
    if datasets:
        comp.append(f"the evaluation datasets include {join_list(datasets[:6])}")
    accs = [a for p in facts if p.extracted and (a := _pct(p.extracted.accuracy)) is not None]
    if len(accs) > 1:
        comp.append(f"reported accuracies range from {min(accs):g}% to {max(accs):g}%, although they are not "
                    f"directly comparable because datasets and protocols differ")
    if comp:
        s = "; ".join(comp)
        paras.append(s[0].upper() + s[1:] + ".")

    if gaps:
        paras.append("Several gaps remain open. " + " ".join(
            f"{['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth'][i] if i < 6 else 'Additionally'}, "
            f"{g[0].lower() + g[1:].rstrip('.')}." for i, g in enumerate(gaps)))

    top_fam = max(groups, key=lambda g: len(groups[g])) if groups else "deep learning"
    paras.append(f"In conclusion, {top_fam.lower()} currently dominate this area. Future work should address the "
                 f"gaps above, favour shared benchmarks so results become comparable, and validate methods in "
                 f"realistic deployment conditions.")
    text = "\n\n".join(paras)
    trace.append({"stage": "Surface realisation",
                  "detail": f"Filled sentence templates, joined lists with commas/'and', produced "
                            f"{len(paras)} paragraphs and {len(text.split())} words."})
    return text, trace
