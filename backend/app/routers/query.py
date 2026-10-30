import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.db.database import SessionLocal
from app.db.models import QueryRow
from app.models.schemas import QueryAccepted, QueryRequest, QueryStatus
from app.pipeline import run_pipeline

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", status_code=202, response_model=QueryAccepted)
def create_query(req: QueryRequest, background: BackgroundTasks):
    qid = "q_" + uuid.uuid4().hex[:6]
    with SessionLocal() as s:
        s.add(QueryRow(query_id=qid, question=req.question.strip(), status="processing", stage="searching"))
        s.commit()
    background.add_task(run_pipeline, qid, req.question.strip(), req.max_papers)
    return QueryAccepted(query_id=qid, status="processing")


@router.get("/{query_id}/status", response_model=QueryStatus)
def query_status(query_id: str):
    with SessionLocal() as s:
        q = s.get(QueryRow, query_id)
        if not q:
            raise HTTPException(404, "Unknown query_id")
        return QueryStatus(query_id=q.query_id, question=q.question, status=q.status, stage=q.stage,
                           papers_found=q.papers_found or 0, papers_processed=q.papers_processed or 0, error=q.error)


@router.get("", response_model=list[QueryStatus])
def recent_queries(limit: int = 10):
    with SessionLocal() as s:
        rows = s.query(QueryRow).order_by(QueryRow.created_at.desc()).limit(limit).all()
        return [QueryStatus(query_id=q.query_id, question=q.question, status=q.status, stage=q.stage,
                            papers_found=q.papers_found or 0, papers_processed=q.papers_processed or 0, error=q.error)
                for q in rows]
