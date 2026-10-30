from fastapi import APIRouter, HTTPException

from app.db import store
from app.models.schemas import RelevanceIn, Report, TranslateIn
from app.nlp import ir
from app.nlp.translate import translate

router = APIRouter(prefix="/report", tags=["report"])


def _report(query_id: str) -> Report:
    r = store.get_report(query_id)
    if not r:
        raise HTTPException(404, "Report not ready (or unknown query_id)")
    return r


@router.get("/{query_id}", response_model=Report)
def get_report(query_id: str):
    return _report(query_id)


@router.post("/{query_id}/translate")
def translate_report(query_id: str, body: TranslateIn):
    """Translate the review (body.text is ignored unless the review is empty)."""
    r = _report(query_id)
    try:
        return translate(r.literature_review or body.text, body.target)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Translation service unavailable: {e}")


@router.post("/{query_id}/evaluate")
def evaluate_retrieval(query_id: str, body: RelevanceIn):
    """Precision / Recall / F-measure / AP for every ranking model, using the user's
    relevance judgments over the candidate pool. The judgments are stored so MAP can be
    computed across all judged queries."""
    r = _report(query_id)
    rankings: dict[str, list[str]] = r.retrieval.get("rankings", {})
    if not rankings:
        raise HTTPException(400, "This report has no retrieval rankings to evaluate")
    relevant = set(body.relevant_ids)
    k = len(r.papers) or 5
    results = {}
    for model, ranking in rankings.items():
        results[model] = {**ir.precision_recall_f(ranking[:k], relevant),
                          "average_precision": ir.average_precision(ranking, relevant)}
    r.retrieval["judgments"] = sorted(relevant)
    store.save_report(r)

    # MAP across every query that has judgments
    from app.db.database import SessionLocal
    from app.db.models import ReportRow
    with SessionLocal() as s:
        judged = [Report(**row.data) for row in s.query(ReportRow).all()]
    judged = [j for j in judged if j.retrieval.get("judgments")]
    mean_ap = {m: ir.mean_average_precision([(j.retrieval["rankings"].get(m, []), set(j.retrieval["judgments"]))
                                             for j in judged if m in j.retrieval.get("rankings", {})])
               for m in rankings}
    return {"k": k, "relevant": len(relevant), "metrics": results, "map": mean_ap, "judged_queries": len(judged)}
