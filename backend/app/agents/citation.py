"""Citation agent: deterministic templates for APA 7, IEEE, MLA 9 and BibTeX."""
import re

from app.models.schemas import Paper


def _split(name: str) -> tuple[list[str], str]:
    parts = name.replace(".", ". ").split()
    return (parts[:-1], parts[-1]) if len(parts) > 1 else ([], name.strip())


def _initials(given: list[str]) -> str:
    return " ".join(f"{g[0]}." for g in given if g and g[0].isalpha())


def apa_author(name: str) -> str:
    given, last = _split(name)
    return f"{last}, {_initials(given)}".strip().rstrip(",")


def ieee_author(name: str) -> str:
    given, last = _split(name)
    return f"{_initials(given)} {last}".strip()


def apa(p: Paper) -> str:
    names = [apa_author(a) for a in p.authors] or ["Anonymous"]
    if len(names) > 20:
        authors = ", ".join(names[:19]) + ", ... " + names[-1]
    elif len(names) > 1:
        authors = ", ".join(names[:-1]) + ", & " + names[-1]
    else:
        authors = names[0]
    return f"{authors} ({p.year or 'n.d.'}). {p.title.rstrip('.')}. {p.venue or 'Preprint'}. {p.source_url}".strip()


def ieee(p: Paper, n: int) -> str:
    names = [ieee_author(a) for a in p.authors] or ["Anonymous"]
    if len(names) > 6:
        authors = names[0] + " et al."
    elif len(names) > 2:
        authors = ", ".join(names[:-1]) + ", and " + names[-1]
    else:
        authors = " and ".join(names)
    return (f"[{n}] {authors}, “{p.title.rstrip('.')},” {p.venue or 'Preprint'}, {p.year or 'n.d.'}. "
            f"[Online]. Available: {p.source_url}")


def mla(p: Paper) -> str:
    a = p.authors
    if not a:
        authors = ""
    else:
        given, last = _split(a[0])
        first = f"{last}, {' '.join(given)}".strip().rstrip(",")
        authors = first if len(a) == 1 else (f"{first}, and {a[1]}" if len(a) == 2 else f"{first}, et al")
        authors += ". "
    return f"{authors}“{p.title.rstrip('.')}.” {p.venue or 'Preprint'}, {p.year or 'n.d.'}, {p.source_url}."


def _bib_escape(s: str) -> str:
    return re.sub(r"([&%$#_])", r"\\\1", s)


def bibtex(p: Paper) -> str:
    last = _split(p.authors[0])[1].lower() if p.authors else "anon"
    first_word = next((w for w in re.findall(r"[a-z0-9]+", p.title.lower()) if len(w) > 3), "paper")
    key = re.sub(r"[^a-z0-9]", "", f"{last}{p.year or ''}{first_word}")
    authors = " and ".join(f"{_split(a)[1]}, {' '.join(_split(a)[0])}".strip().rstrip(",") for a in p.authors)
    entry = "misc" if (p.venue or "").lower() in ("arxiv", "", "semantic scholar") else "article"
    fields = [("title", f"{{{_bib_escape(p.title)}}}"), ("author", _bib_escape(authors) or "Anonymous"),
              ("year", str(p.year or "")), ("journal" if entry == "article" else "howpublished", _bib_escape(p.venue or "Preprint")),
              ("url", p.source_url)]
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields if v)
    return f"@{entry}{{{key},\n{body}\n}}"


def format_citations(papers: list[Paper]) -> dict[str, list[str]]:
    return {
        "apa": sorted(apa(p) for p in papers),          # APA and MLA are alphabetical
        "ieee": [ieee(p, i) for i, p in enumerate(papers, 1)],  # IEEE is numbered in citation order
        "mla": sorted(mla(p) for p in papers),
        "bibtex": [bibtex(p) for p in papers],
    }
