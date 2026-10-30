"""PDF extraction agent: download -> PyMuPDF text -> clean -> split into sections."""
import logging
import re
import unicodedata

import pymupdf as fitz  # PyMuPDF
import httpx

from app.config import HTTP_RETRIES, HTTP_TIMEOUT, MAX_PDF_PAGES

log = logging.getLogger("litlens.pdf")
SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\s*|[IVX]+\.\s*)?(abstract|introduction|related work|background|methods?|methodology|"
    r"proposed (?:method|approach)|materials and methods|experiments?|results?|discussion|conclusions?|limitations)\s*$",
    re.I | re.M)


def download_pdf(url: str) -> bytes | None:
    for attempt in range(HTTP_RETRIES):
        try:
            r = httpx.get(url, timeout=HTTP_TIMEOUT, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0 (LitLens research assistant)"})
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                return r.content
            return None          # HTML landing page / paywall - skip rather than block
        except httpx.HTTPError as e:
            log.info("PDF download failed (%s), attempt %d: %s", url, attempt + 1, e)
    return None


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)         # ligatures: "ﬁ" -> "fi", else "identiﬁcation" splits in two
    text = re.sub(r"-\n(\w)", r"\1", text)              # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # drop the bibliography - it confuses summarisation and extraction
    m = re.search(r"\n\s*(references|bibliography)\s*\n", text, re.I)
    if m and m.start() > len(text) * 0.4:
        text = text[:m.start()]
    return text.strip()


def extract_text(pdf_bytes: bytes) -> tuple[str, dict]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        meta = {"title": (doc.metadata or {}).get("title") or "", "author": (doc.metadata or {}).get("author") or "",
                "pages": doc.page_count}
        raw = "\n".join(page.get_text() for page in doc.pages(0, min(MAX_PDF_PAGES, doc.page_count)))
        meta["first_lines"] = [l.strip() for l in doc[0].get_text().split("\n") if len(l.strip()) > 15][:3] if doc.page_count else []
        meta["visual_title"] = _largest_font_text(doc[0]) if doc.page_count else ""
    return clean_text(raw), meta


def _largest_font_text(page) -> str:
    """The title is usually the text set in the largest font on the first page."""
    # horizontal lines only: skips rotated margin stamps such as arXiv's "arXiv:1706.03762v7 [cs.CL]"
    spans = [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) if abs(l["dir"][1]) < 0.01
             for s in l["spans"] if s["text"].strip() and not s["text"].strip().lower().startswith("arxiv:")]
    if not spans:
        return ""
    biggest = max(s["size"] for s in spans)
    title = " ".join(s["text"].strip() for s in spans if s["size"] >= biggest - 0.5)
    return re.sub(r"\s+", " ", title).strip()[:300]


def split_sections(text: str) -> dict[str, str]:
    marks = [(m.start(), m.group(1).lower()) for m in SECTION_RE.finditer(text)]
    sections = {}
    for i, (start, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        sections.setdefault(name, text[start:end].strip())
    return sections


def extract_paper(pdf_url: str | None) -> tuple[str | None, dict]:
    if not pdf_url:
        return None, {}
    data = download_pdf(pdf_url)
    if not data:
        return None, {}
    try:
        text, meta = extract_text(data)
    except Exception as e:  # malformed PDF
        log.info("PyMuPDF failed on %s: %s", pdf_url, e)
        return None, {}
    meta["sections"] = {k: len(v.split()) for k, v in split_sections(text).items()}
    return text, meta
