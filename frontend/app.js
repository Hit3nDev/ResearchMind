/* LitLens frontend - plain JS, no build step. Talks to the FastAPI backend on the same origin. */
const API = "";
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const STAGES = [
  ["searching", "Searching"], ["ranking", "Ranking"], ["extracting_papers", "Reading PDFs"],
  ["summarizing", "Summarising"], ["comparing", "Comparing"], ["writing_review", "Writing"],
  ["formatting_citations", "Citing"], ["done", "Done"],
];
const SAMPLES = [
  "What are the latest machine learning techniques for crop disease detection?",
  "How are transformers used for machine translation of Indian languages?",
  "Which deep learning models work best for fake news detection?",
];

/* ---------- api ---------- */
async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
    ...opts,
    body: opts.body && !(opts.body instanceof FormData) ? JSON.stringify(opts.body) : opts.body,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${res.status})`);
  return data;
}

/* ---------- shared UI states ---------- */
const Loading = (msg = "Loading…") => `<div class="state"><div class="spinner"></div>${esc(msg)}</div>`;
const ErrorState = (msg) => `<div class="state error"><b>Something went wrong.</b><br>${esc(msg)}</div>`;
const EmptyState = (msg) => `<div class="state">${esc(msg)}</div>`;
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), 2600);
}
const paragraphs = (text) => String(text || "").split(/\n{2,}/).map((p) => p.trim()).filter(Boolean).map((p) => `<p>${esc(p)}</p>`).join("");

/* ---------- navigation ---------- */
function showView(name) {
  document.querySelectorAll("#mainTabs button").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  if (name === "projects") loadProjects();
  window.scrollTo({ top: 0 });
}
document.querySelectorAll("#mainTabs button").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
$(".brand").addEventListener("click", (e) => { e.preventDefault(); showView("research"); $("#hero").classList.remove("compact"); $("#report").innerHTML = ""; $("#progress").innerHTML = ""; });

/* ---------- research flow ---------- */
let currentReport = null;
$("#samples").innerHTML = SAMPLES.map((s) => `<button type="button" class="chip">${esc(s)}</button>`).join("");
$("#samples").querySelectorAll(".chip").forEach((c, i) => c.addEventListener("click", () => { $("#question").value = SAMPLES[i]; $("#question").focus(); }));
$("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#askForm").requestSubmit(); } });

$("#askForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("#question").value.trim();
  if (question.length < 3) return toast("Type a question first");
  $("#askBtn").disabled = true;
  $("#report").innerHTML = "";
  $("#hero").classList.add("compact");
  try {
    const { query_id } = await api("/query", { method: "POST", body: { question, max_papers: +$("#maxPapers").value } });
    await poll(query_id);
  } catch (err) {
    $("#progress").innerHTML = ErrorState(err.message);
  } finally {
    $("#askBtn").disabled = false;
  }
});

function renderProgress(s) {
  const idx = s.status === "complete" ? STAGES.length : STAGES.findIndex(([k]) => k === s.stage);
  $("#progress").innerHTML = `<div class="run">
    <div class="run-head"><h3>“${esc(s.question)}”</h3>
      <span class="muted">${s.papers_found ? `${s.papers_processed} of ${s.papers_found} papers read` : "Looking for papers…"}</span></div>
    <div class="timeline">${STAGES.map(([, label], i) => `<div class="tl ${i < idx ? "done" : i === idx ? "current" : ""}">${label}</div>`).join("")}</div></div>`;
}

async function poll(qid) {
  for (;;) {
    const s = await api(`/query/${qid}/status`);
    renderProgress(s);
    if (s.status === "failed") throw new Error(s.error || "Pipeline failed");
    if (s.status === "complete") {
      $("#progress").innerHTML = "";
      if (s.error) toast(s.error);
      return loadReport(qid);
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
}

async function loadReport(qid, target = $("#report")) {
  target.innerHTML = Loading("Opening the report…");
  try {
    const report = await api(`/report/${qid}`);
    const papers = await Promise.all(report.papers.map((id) => api(`/papers/${id}`).catch(() => null)));
    currentReport = { report, papers: papers.filter(Boolean) };
    renderReport(target);
  } catch (err) {
    target.innerHTML = ErrorState(err.message);
  }
}

/* ---------- report ---------- */
const REPORT_TABS = [["review", "Review"], ["papers", "Papers"], ["comparison", "Comparison"], ["gaps", "Gaps"],
  ["citations", "Citations"], ["insights", "NLP insights"], ["translate", "Translate"]];

function renderReport(target, tab = "review") {
  const { report, papers } = currentReport;
  const years = papers.map((p) => p.year).filter(Boolean).sort();
  const span = years.length ? (years[0] === years.at(-1) ? years[0] : `${years[0]}–${years.at(-1)}`) : "";
  target.innerHTML = `<article class="report">
    <div class="report-meta"><p class="eyebrow">Research report</p><span class="dot"></span>
      <span class="muted">${papers.length} papers${span ? " · " + span : ""}</span><span class="dot"></span>
      <span class="muted">${report.review_method === "llm" ? "Written with Claude" : "Classical NLP"}</span></div>
    <h2 class="title">${esc(report.query_analysis?.corrected || report.question)}</h2>
    <div class="report-actions">
      <label class="ghost">↑ Add a PDF<input type="file" accept="application/pdf" id="uploadPdf" hidden></label>
      <button class="ghost" id="saveProject">Save to library</button>
      <button class="ghost" id="exportMd">Export .md</button>
    </div>
    <nav class="subtabs">${REPORT_TABS.map(([k, l]) => `<button data-tab="${k}" class="${k === tab ? "active" : ""}">${l}</button>`).join("")}</nav>
    <div id="tabBody"></div></article>`;
  target.querySelectorAll(".subtabs button").forEach((b) => b.addEventListener("click", () => renderReport(target, b.dataset.tab)));
  $("#saveProject").addEventListener("click", saveProject);
  $("#exportMd").addEventListener("click", exportMarkdown);
  $("#uploadPdf").addEventListener("change", (e) => uploadPdf(e.target.files[0], target));
  const body = $("#tabBody");
  if (!report.papers.length && tab !== "insights") {
    body.innerHTML = EmptyState("No papers matched this question. Try broader wording, or open NLP insights to see how it was understood.");
    return;
  }
  ({ review: tabReview, papers: tabPapers, comparison: tabComparison, gaps: tabGaps, citations: tabCitations,
    insights: tabInsights, translate: tabTranslate })[tab](body);
}

function Highlights(h) {
  const items = [["highest_accuracy", "Highest accuracy"], ["most_recent", "Most recent"], ["largest_dataset", "Largest dataset"]]
    .filter(([k]) => h[k]).map(([k, l]) => `<div class="stat"><div class="k">${l}</div><div class="v">${esc(String(h[k].value).replace(/ \(.*\)$/, ""))}</div><div class="s">${esc(h[k].ref)}${h[k].model ? " · " + esc(h[k].model) : ""}</div></div>`);
  return items.length ? `<div class="highlights">${items.join("")}</div>` : "";
}

function tabReview(el) {
  const r = currentReport.report;
  const models = [...new Set(r.comparison_table.map((x) => x.model).filter(Boolean))];
  const datasets = [...new Set(r.comparison_table.map((x) => x.dataset).filter(Boolean))];
  el.innerHTML = `${Highlights(r.highlights)}
    <div class="article">
      <div><div class="review">${paragraphs(r.literature_review)}</div>
        <p class="method">${r.review_method === "llm" ? "Synthesised by Claude." : "Written by the template-based NLG pipeline — see NLP insights for each stage."} ${r.literature_review.split(/\s+/).length} words.</p></div>
      <aside class="aside"><p class="eyebrow">At a glance</p><dl>
        <dt>Methods</dt><dd>${esc(models.join(", ") || "—")}</dd>
        <dt>Datasets</dt><dd>${esc(datasets.join(", ") || "—")}</dd>
        <dt>Open gaps</dt><dd>${r.gaps.length}</dd></dl></aside>
    </div>`;
}

function PaperCard(p, i) {
  const ex = p.extracted || {};
  const facts = [["Model", ex.model], ["Data", ex.dataset], ["Acc", ex.accuracy], ["Prec", ex.precision], ["Rec", ex.recall]].filter(([, v]) => v);
  const link = p.source_url.startsWith("http") ? p.source_url : null;
  return `<article class="paper">
    <span class="idx">${String(i + 1).padStart(2, "0")} · ${esc(p.venue || "")}</span>
    <h4>${link ? `<a href="${esc(link)}" target="_blank" rel="noopener">${esc(p.title)}</a>` : esc(p.title)}</h4>
    <div class="meta">${esc(p.authors.slice(0, 3).join(", "))}${p.authors.length > 3 ? " et al." : ""}${p.year ? " · " + p.year : ""}</div>
    <div class="summary">${esc(p.one_line_summary || p.abstract.slice(0, 220) + "…")}</div>
    ${facts.length ? `<div class="facts">${facts.map(([k, v]) => `<span><b>${k}</b>${esc(v)}</span>`).join("")}</div>` : ""}
    ${p.nlp?.key_terms ? `<div>${p.nlp.key_terms.slice(0, 6).map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>` : ""}
    ${ex.limitations?.length ? `<details><summary>Limitations (${ex.limitations.length})</summary><ul>${ex.limitations.map((l) => `<li class="muted">${esc(l)}</li>`).join("")}</ul></details>` : ""}
    <details><summary>Abstract &amp; NLP details</summary><p class="muted" style="margin-top:8px">${esc(p.abstract)}</p>
      <p class="muted">Extraction: ${esc(p.nlp?.extraction_method || "—")} · PDF: ${p.nlp?.pdf?.pages ? p.nlp.pdf.pages + " pages" : "not available (abstract only)"}
      ${p.nlp?.abstract_perplexity ? ` · trigram perplexity vs. other papers: ${p.nlp.abstract_perplexity}` : ""}</p>
      ${Object.keys(p.scores || {}).length ? `<div class="mono muted">${Object.entries(p.scores).map(([k, v]) => `${k} ${v}`).join("  ·  ")}</div>` : ""}
    </details></article>`;
}
function tabPapers(el) { el.innerHTML = `<div class="papers">${currentReport.papers.map(PaperCard).join("")}</div>`; }

function tabComparison(el) {
  const r = currentReport.report;
  const best = r.highlights.highest_accuracy?.paper_id;
  const nums = ["accuracy", "precision", "recall", "dataset_size"];
  const cols = [["ref", "Paper"], ["year", "Year"], ["model", "Model"], ["dataset", "Dataset"], ["accuracy", "Acc."], ["precision", "Prec."], ["recall", "Rec."], ["dataset_size", "Size"]];
  const cell = (row, k) => k === "ref" ? `${esc(row.ref)}<span class="ttl">${esc(row.title)}</span>`
    : esc(k === "dataset_size" && row[k] ? row[k].toLocaleString("en-US") : row[k] ?? "—");
  el.innerHTML = `${Highlights(r.highlights)}<div class="table-wrap"><table>
    <thead><tr>${cols.map(([k, l]) => `<th class="${nums.includes(k) ? "num" : ""}">${l}</th>`).join("")}</tr></thead>
    <tbody>${r.comparison_table.map((row) => `<tr class="${row.paper_id === best ? "best" : ""}">${cols.map(([k]) =>
      `<td class="${nums.includes(k) ? "num" : ""}">${cell(row, k)}</td>`).join("")}</tr>`).join("")}</tbody>
  </table></div><p class="method">Highlighted row: highest reported accuracy. A dash means the paper (or its readable text) did not state it. Figures are as reported and not directly comparable.</p>`;
}

function tabGaps(el) {
  const r = currentReport.report;
  el.innerHTML = r.gaps.length ? `<ol class="gaps">${r.gaps.map((g) => `<li>${esc(g)}</li>`).join("")}</ol>
    <p class="method">Detected by ${r.retrieval.gap_method === "llm" ? "Claude" : "rule-based corpus analysis (how many papers mention real-world tests, dataset size, efficiency, diversity, code release…)"}.</p>` : EmptyState("No research gaps detected.");
}

function tabCitations(el, style = "apa") {
  const c = currentReport.report.citations;
  el.innerHTML = `<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap"><nav class="subtabs pills">${[["apa", "APA 7"], ["ieee", "IEEE"], ["mla", "MLA 9"], ["bibtex", "BibTeX"]].map(([k, l]) => `<button data-s="${k}" class="${k === style ? "active" : ""}">${l}</button>`).join("")}</nav>
    <button class="ghost" id="copyAll" style="margin:0 0 18px auto">Copy all</button></div>
    ${c[style].map((t, i) => `<div class="cite-item"><pre>${esc(t)}</pre><button class="ghost" data-i="${i}">Copy</button></div>`).join("")}`;
  el.querySelectorAll("[data-s]").forEach((b) => b.addEventListener("click", () => tabCitations(el, b.dataset.s)));
  el.querySelectorAll("[data-i]").forEach((b) => b.addEventListener("click", () => copy(c[style][+b.dataset.i])));
  $("#copyAll").addEventListener("click", () => copy(c[style].join(style === "bibtex" ? "\n\n" : "\n")));
}
const copy = (t) => navigator.clipboard.writeText(t).then(() => toast("Copied to clipboard"));

function tabInsights(el) {
  const r = currentReport.report, qa = r.query_analysis || {}, rt = r.retrieval || {};
  const titles = rt.titles || {};
  const models = Object.keys(rt.rankings || {});
  el.innerHTML = `<div class="grid2">
    <div><p class="eyebrow">Units 1–3</p><h3>How your question was read</h3>
      ${qa.spelling_fixes?.length ? `<p>Spelling fixed by edit distance: ${qa.spelling_fixes.map((f) => `<span class="tag">${esc(f.original)} → ${esc(f.corrected)}<b>d=${f.distance}</b></span>`).join("")}</p>` : `<p class="muted">No spelling errors found.</p>`}
      <p class="muted">Part-of-speech tags</p><p>${(qa.pos_tags || []).map(([w, t]) => `<span class="tag">${esc(w)}<b>${esc(t)}</b></span>`).join("")}</p>
      <p class="muted">Noun phrases (regexp chunk parser)</p><p>${(qa.noun_phrases || []).map((p) => `<span class="tag">${esc(p)}</span>`).join("") || "—"}</p>
      <p class="muted">Lemmatised keywords</p><p>${(qa.keywords || []).map((k) => `<span class="tag">${esc(k)}</span>`).join("")}</p>
      <p class="muted">WordNet expansion (only when Lesk agrees with the most frequent sense)</p><p>${Object.entries(qa.expansions || {}).map(([k, v]) => `<span class="tag">${esc(k)} → ${esc(v.join(", "))}</span>`).join("") || "—"}</p>
      <p class="muted">Ambiguous words</p>
      ${(qa.ambiguous_words || []).map((w) => `<details><summary><b>${esc(w.word)}</b> · ${w.num_senses} senses · Lesk picks <i>${esc(w.lesk_sense)}</i></summary>
        <p class="muted" style="margin-top:6px">${esc(w.lesk_definition)}</p><ul>${w.senses.map((s) => `<li class="muted"><span class="mono">${esc(s.name)}</span> ${esc(s.definition)}</li>`).join("")}</ul></details>`).join("") || `<p class="muted">None.</p>`}
    </div>
    <div><p class="eyebrow">Unit 5</p><h3>How papers were ranked</h3>
      <p class="muted">Inverted index of ${rt.index?.documents ?? 0} candidates and ${rt.index?.vocabulary ?? 0} terms. Ranked by ${models.join(", ")}; <i>hybrid</i> fuses them with reciprocal rank fusion. Boolean query <span class="mono">${esc(rt.boolean_expression || "")}</span> matched ${(rt.boolean_matches || []).length}.</p>
      <p class="muted">Tick the relevant candidates, then measure each ranker.</p>
      <div class="table-wrap" style="max-height:320px;overflow:auto"><table><thead><tr><th></th><th>Candidate</th>${models.map((m) => `<th class="num">${m}</th>`).join("")}</tr></thead>
      <tbody>${(rt.rankings?.hybrid || []).map((id) => `<tr><td><input type="checkbox" class="rel" value="${id}" ${(rt.judgments || r.papers).includes(id) ? "checked" : ""}></td><td>${esc((titles[id] || id).slice(0, 64))}</td>${models.map((m) => `<td class="num">${rt.rankings[m].indexOf(id) + 1}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
      <button class="ghost" id="evalBtn" style="margin-top:14px">Compute precision, recall, F1, MAP</button><div id="evalOut" style="margin-top:12px"></div>
    </div></div>
    <hr class="section-rule"><p class="eyebrow">Unit 4</p><h3 style="font-size:1.35rem;margin:6px 0 12px">How the review was written</h3>
    <ol class="trace">${(rt.nlg_trace || []).map((t) => `<li><b>${esc(t.stage)}</b> — ${esc(t.detail)}</li>`).join("")}</ol>`;
  $("#evalBtn")?.addEventListener("click", async () => {
    const ids = [...el.querySelectorAll(".rel:checked")].map((c) => c.value);
    $("#evalOut").innerHTML = Loading();
    try {
      const res = await api(`/report/${r.query_id}/evaluate`, { method: "POST", body: { relevant_ids: ids } });
      $("#evalOut").innerHTML = `<table><thead><tr><th>Model</th><th class="num">P@${res.k}</th><th class="num">R@${res.k}</th><th class="num">F1</th><th class="num">AP</th><th class="num">MAP·${res.judged_queries}</th></tr></thead><tbody>
        ${Object.entries(res.metrics).map(([m, v]) => `<tr><td>${m}</td><td class="num">${v.precision}</td><td class="num">${v.recall}</td><td class="num">${v.f_measure}</td><td class="num">${v.average_precision}</td><td class="num">${res.map[m]}</td></tr>`).join("")}</tbody></table>`;
      r.retrieval.judgments = ids;
    } catch (err) { $("#evalOut").innerHTML = ErrorState(err.message); }
  });
}

async function tabTranslate(el) {
  el.innerHTML = Loading();
  const { languages, notes } = await api("/nlp/languages");
  el.innerHTML = `<div class="article"><div>
      <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
        <label class="select-inline">Translate the review into
          <select id="lang">${Object.entries(languages).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select></label>
        <button class="btn-primary" id="trBtn">Translate</button></div>
      <div id="trOut" style="margin-top:28px"></div></div>
    <aside class="aside"><p class="eyebrow">Case study</p><p style="margin-top:8px"><b>Why Indian languages are hard to translate</b></p>
      <ul style="padding-left:18px">${notes.map((n) => `<li class="muted" style="margin-bottom:8px">${esc(n)}</li>`).join("")}</ul></aside></div>`;
  $("#trBtn").addEventListener("click", async () => {
    $("#trOut").innerHTML = Loading("Translating…");
    try {
      const t = await api(`/report/${currentReport.report.query_id}/translate`, { method: "POST", body: { text: ".", target: $("#lang").value } });
      $("#trOut").innerHTML = `<div class="review">${paragraphs(t.translation)}</div><p class="method">${esc(t.engine)} → ${esc(t.language)}</p>`;
    } catch (err) { $("#trOut").innerHTML = ErrorState(err.message); }
  });
}

async function saveProject() {
  const name = prompt("Name this project", currentReport.report.question.slice(0, 60));
  if (!name) return;
  try { await api("/projects", { method: "POST", body: { query_id: currentReport.report.query_id, name } }); toast("Saved to your library"); }
  catch (err) { toast(err.message); }
}

async function uploadPdf(file, target) {
  if (!file) return;
  toast("Reading your PDF…");
  const fd = new FormData();
  fd.append("file", file);
  fd.append("query_id", currentReport.report.query_id);
  try { await api("/papers/upload", { method: "POST", body: fd }); toast("PDF added to the report"); loadReport(currentReport.report.query_id, target); }
  catch (err) { toast(err.message); }
}

function exportMarkdown() {
  const r = currentReport.report;
  const md = [`# ${r.question}`, "", "## Literature review", "", r.literature_review, "", "## Comparison", "",
    "| Paper | Year | Model | Dataset | Accuracy |", "|---|---|---|---|---|",
    ...r.comparison_table.map((x) => `| ${x.ref} | ${x.year ?? ""} | ${x.model ?? "-"} | ${x.dataset ?? "-"} | ${x.accuracy ?? "-"} |`),
    "", "## Research gaps", "", ...r.gaps.map((g) => `- ${g}`), "", "## References (APA)", "", ...r.citations.apa.map((c) => `- ${c}`)].join("\n");
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([md], { type: "text/markdown" })), download: `litlens-${r.query_id}.md` });
  a.click();
}

/* ---------- library ---------- */
async function loadProjects() {
  const el = $("#projectList");
  el.innerHTML = Loading();
  try {
    const [projects, recent] = await Promise.all([api("/projects"), api("/query?limit=10")]);
    const row = (title, sub, meta, qid) => `<div class="lib-row"><div><div class="t">${esc(title)}</div>${sub ? `<div class="q">${esc(sub)}</div>` : ""}</div><span class="muted">${meta}</span>${qid ? `<button class="ghost" data-q="${qid}">Open →</button>` : "<span></span>"}</div>`;
    el.innerHTML = `<section class="lib"><p class="eyebrow">Projects</p><h2>Saved reports</h2>
        ${projects.length ? projects.map((p) => row(p.name, p.question, esc(p.created_at.slice(0, 10)), p.query_id)).join("") : EmptyState("Nothing saved yet. Open a report and choose “Save to library”.")}</section>
      <section class="lib"><p class="eyebrow">History</p><h2>Recent questions</h2>
        ${recent.length ? recent.map((q) => row(q.question, "", `${q.status} · ${q.papers_found} papers`, q.status === "complete" ? q.query_id : null)).join("") : EmptyState("No questions yet.")}</section>`;
    el.querySelectorAll("[data-q]").forEach((b) => b.addEventListener("click", () => { showView("research"); $("#progress").innerHTML = ""; $("#hero").classList.add("compact"); loadReport(b.dataset.q); }));
  } catch (err) { el.innerHTML = ErrorState(err.message); }
}

/* ---------- boot ---------- */
// Deep links: /?report=<query_id> opens a report, /?view=projects opens the library.
const params = new URLSearchParams(location.search);
if (params.get("view")) showView(params.get("view"));
if (params.get("report")) { $("#hero").classList.add("compact"); loadReport(params.get("report")).then(() => params.get("tab") && renderReport($("#report"), params.get("tab"))); }
api("/health").then((h) => {
  const on = !h.llm.startsWith("disabled");
  $("#llmBadge span").textContent = on ? `Claude · ${h.llm}` : "Classical NLP mode";
  $("#llmBadge").classList.toggle("on", true);
}).catch(() => ($("#llmBadge span").textContent = "Backend offline"));
