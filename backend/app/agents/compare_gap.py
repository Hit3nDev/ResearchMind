"""Comparison + research-gap agent."""
import re
from collections import Counter

from pydantic import BaseModel

from app.models.schemas import Paper
from app.nlp import llm
from app.nlp.nlg import _pct, short_ref

SIZE_RE = re.compile(r"(\d{1,3}(?:,\d{3})+|\d{3,7})\s+(?:images|samples|sentences|documents|patients|videos|records|examples|instances|utterances)", re.I)


def _dataset_size(p: Paper) -> int | None:
    sizes = [int(m.group(1).replace(",", "")) for m in SIZE_RE.finditer(f"{p.abstract} {p.full_text or ''}")]
    return max(sizes) if sizes else None


def compare(papers: list[Paper]) -> tuple[list[dict], dict]:
    table = []
    for p in papers:
        ex = p.extracted
        table.append({
            "paper_id": p.paper_id, "ref": short_ref(p), "title": p.title, "year": p.year,
            "model": ex.model if ex else None, "dataset": ex.dataset if ex else None,
            "accuracy": ex.accuracy if ex else None, "precision": ex.precision if ex else None,
            "recall": ex.recall if ex else None, "dataset_size": _dataset_size(p),
        })
    highlights = {}
    with_acc = [r for r in table if _pct(r["accuracy"]) is not None]
    if with_acc:
        best = max(with_acc, key=lambda r: _pct(r["accuracy"]))
        highlights["highest_accuracy"] = {"paper_id": best["paper_id"], "ref": best["ref"], "value": best["accuracy"], "model": best["model"]}
    dated = [r for r in table if r["year"]]
    if dated:
        rec = max(dated, key=lambda r: r["year"])
        highlights["most_recent"] = {"paper_id": rec["paper_id"], "ref": rec["ref"], "value": str(rec["year"])}
    sized = [r for r in table if r["dataset_size"]]
    if sized:
        big = max(sized, key=lambda r: r["dataset_size"])
        highlights["largest_dataset"] = {"paper_id": big["paper_id"], "ref": big["ref"],
                                         "value": f"{big['dataset_size']:,} items" + (f" ({big['dataset']})" if big["dataset"] else "")}
    return table, highlights


class Gaps(BaseModel):
    gaps: list[str]


GAP_SYSTEM = """You are a research analyst. Given a comparison table and the limitations of each paper,
identify 3-6 concrete research gaps (small datasets, no real-world validation, compute cost,
limited geographic/demographic/language diversity, missing metrics, reproducibility...).
Schema: {"gaps": array of strings}. Each gap is one sentence."""


def rule_based_gaps(papers: list[Paper], table: list[dict]) -> list[str]:
    n = len(papers) or 1
    corpus = [f"{p.abstract} {p.full_text or ''}".lower() for p in papers]

    def share(pattern: str) -> float:
        return sum(1 for t in corpus if re.search(pattern, t)) / n

    gaps = []
    if share(r"real[- ]world|in the wild|field (?:trial|condition|data)|deploy|clinical (?:trial|setting)") < 0.4:
        gaps.append("Few studies validate their methods under real-world or deployment conditions; most results come from curated benchmark data.")
    sizes = [r["dataset_size"] for r in table if r["dataset_size"]]
    if (sizes and sorted(sizes)[len(sizes) // 2] < 10000) or share(r"small (?:dataset|sample)|limited data") > 0.25:
        gaps.append("Datasets are small across most papers, which limits how well the reported results generalise.")
    ds = Counter(r["dataset"] for r in table if r["dataset"])
    if ds and ds.most_common(1)[0][1] / n >= 0.4:
        gaps.append(f"There is heavy reliance on a single benchmark ({ds.most_common(1)[0][0]}), so methods may be overfitted to it.")
    if share(r"india|africa|low[- ]resource|multilingual|cross[- ]lingual|diverse population|demograph") < 0.25:
        gaps.append("Geographic, demographic and linguistic diversity is limited (e.g. few studies on Indian or other low-resource settings).")
    if share(r"lightweight|real[- ]time|edge|mobile|latency|efficien|low[- ]cost") < 0.4:
        gaps.append("Computational cost is rarely addressed; lightweight models suitable for edge or mobile deployment are under-explored.")
    if sum(1 for r in table if r["precision"] or r["recall"]) < n / 2:
        gaps.append("Many papers report only accuracy, without precision, recall or F1, which hides class-imbalance problems.")
    if share(r"github|code (?:is )?available|open[- ]source") < 0.3:
        gaps.append("Few papers release code or trained models, which makes results hard to reproduce.")
    lims = [l for p in papers if p.extracted for l in p.extracted.limitations]
    if not gaps and lims:
        gaps.append(lims[0])
    return gaps[:6]


def find_gaps(papers: list[Paper], table: list[dict]) -> tuple[list[str], str]:
    if not papers:
        return [], "none"
    lines = [f"- {r['ref']}: model={r['model']}, dataset={r['dataset']}, accuracy={r['accuracy']}, "
             f"limitations={'; '.join(p.extracted.limitations if p.extracted else [])}"
             for r, p in zip(table, papers)]
    res = llm.call_json(GAP_SYSTEM, "\n".join(lines), Gaps)
    if res and res.gaps:
        return res.gaps[:6], "llm"
    return rule_based_gaps(papers, table), "rule_based"
