import hashlib
import re

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.agents import pdf_extract, summarize_extract
from app.db import store
from app.models.schemas import Paper
from app.pipeline import rebuild_report

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("/{paper_id}", response_model=Paper, response_model_exclude={"full_text"})
def get_paper(paper_id: str):
    p = store.get_paper(paper_id)
    if not p:
        raise HTTPException(404, "Unknown paper_id")
    return p


@router.post("/upload", response_model=Paper, response_model_exclude={"full_text"})
async def upload_paper(file: UploadFile = File(...), query_id: str | None = Form(None)):
    data = await file.read()
    if data[:4] != b"%PDF":
        raise HTTPException(400, "Please upload a PDF file")
    try:
        text, meta = pdf_extract.extract_text(data)
    except Exception:
        raise HTTPException(422, "Could not read this PDF (it may be scanned or corrupted)")
    if len(text.split()) < 50:
        raise HTTPException(422, "The PDF has almost no extractable text (scanned images are not supported)")
    sections = pdf_extract.split_sections(text)
    visual = meta.get("visual_title", "")
    title = meta.get("title") or (visual if len(visual) > 10 else None) or (meta.get("first_lines") or [file.filename])[0]
    authors = [a.strip() for a in re.split(r"[,;]| and ", meta.get("author") or "") if a.strip()]
    abstract = sections.get("abstract", text[:1500])[:2500]
    years = re.findall(r"\b(19[89]\d|20[0-4]\d)\b", text[:3000])
    paper = Paper(
        paper_id="p_up_" + hashlib.sha1(data).hexdigest()[:8], title=title.strip(), authors=authors,
        year=int(max(years)) if years else None, source_url=f"uploaded:{file.filename}", venue="Uploaded PDF",
        abstract=re.sub(r"^\s*abstract\s*", "", abstract, flags=re.I), full_text=text,
        nlp={"pdf": {"pages": meta.get("pages"), "sections": {k: len(v.split()) for k, v in sections.items()}}},
    )
    summarize_extract.summarize_and_extract(paper)
    if query_id:
        q = store.get_query(query_id)
        if not q:
            raise HTTPException(404, "Unknown query_id")
        store.save_paper(paper, query_id)
        if paper.paper_id not in q.paper_ids:
            store.update_query(query_id, paper_ids=q.paper_ids + [paper.paper_id])
        rebuild_report(query_id)
    else:
        store.save_paper(paper, None)
    return paper
