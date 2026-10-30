"""Small repository layer so agents/routers never touch SQLAlchemy directly."""
from app.db.database import SessionLocal
from app.db.models import PaperRow, QueryRow, ReportRow
from app.models.schemas import Paper, Report


def update_query(query_id: str, **fields) -> None:
    with SessionLocal() as s:
        row = s.get(QueryRow, query_id)
        for k, v in fields.items():
            setattr(row, k, v)
        s.commit()


def get_query(query_id: str) -> QueryRow | None:
    with SessionLocal() as s:
        return s.get(QueryRow, query_id)


def save_paper(paper: Paper, query_id: str | None) -> None:
    with SessionLocal() as s:
        row = s.get(PaperRow, paper.paper_id) or PaperRow(paper_id=paper.paper_id)
        row.query_id = query_id
        row.data = paper.model_dump(exclude={"full_text"})
        row.full_text = paper.full_text
        s.merge(row)
        s.commit()


def get_paper(paper_id: str, with_text: bool = False) -> Paper | None:
    with SessionLocal() as s:
        row = s.get(PaperRow, paper_id)
        if not row:
            return None
        p = Paper(**row.data)
        if with_text:
            p.full_text = row.full_text
        return p


def get_papers(paper_ids: list[str], with_text: bool = False) -> list[Paper]:
    return [p for pid in paper_ids if (p := get_paper(pid, with_text))]


def save_report(report: Report) -> None:
    with SessionLocal() as s:
        s.merge(ReportRow(query_id=report.query_id, data=report.model_dump()))
        s.commit()


def get_report(query_id: str) -> Report | None:
    with SessionLocal() as s:
        row = s.get(ReportRow, query_id)
        return Report(**row.data) if row else None
