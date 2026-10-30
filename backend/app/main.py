"""FastAPI entrypoint. Serves the API and the single-page frontend on one port."""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import APP_NAME, FRONTEND_DIR, LLM_MODEL, llm_enabled
from app.db.database import init_db
from app.nlp.preprocessing import ensure_nltk_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
ensure_nltk_data()
init_db()

from app.routers import nlp_lab, papers, projects, query, report  # noqa: E402  (needs NLTK data)

app = FastAPI(title=f"{APP_NAME} API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
for r in (query.router, papers.router, report.router, projects.router, nlp_lab.router):
    app.include_router(r)


@app.get("/health")
def health():
    return {"app": APP_NAME, "llm": LLM_MODEL if llm_enabled() else "disabled (classical NLP mode)"}


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
