import uuid

from fastapi import APIRouter, HTTPException

from app.db import store
from app.db.database import SessionLocal
from app.db.models import ProjectRow, QueryRow
from app.models.schemas import Project, ProjectCreate

router = APIRouter(prefix="/projects", tags=["projects"])


def _to_schema(row: ProjectRow, question: str) -> Project:
    return Project(project_id=row.project_id, name=row.name, query_id=row.query_id,
                   question=question, created_at=row.created_at.isoformat(timespec="seconds"))


@router.post("", response_model=Project, status_code=201)
def create_project(body: ProjectCreate):
    q = store.get_query(body.query_id)
    if not q:
        raise HTTPException(404, "Unknown query_id")
    with SessionLocal() as s:
        row = ProjectRow(project_id="pr_" + uuid.uuid4().hex[:6], name=body.name.strip(), query_id=body.query_id)
        s.add(row)
        s.commit()
        return _to_schema(row, q.question)


@router.get("", response_model=list[Project])
def list_projects():
    with SessionLocal() as s:
        rows = s.query(ProjectRow, QueryRow.question).join(QueryRow, QueryRow.query_id == ProjectRow.query_id) \
            .order_by(ProjectRow.created_at.desc()).all()
        return [_to_schema(r, question) for r, question in rows]


@router.get("/{project_id}")
def get_project(project_id: str):
    with SessionLocal() as s:
        row = s.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(404, "Unknown project_id")
    q = store.get_query(row.query_id)
    return {"project": _to_schema(row, q.question if q else ""), "report": store.get_report(row.query_id)}
