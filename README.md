# LitLens — an AI literature-review assistant built on NLP

*(formerly "ResearchMind AI")*

Ask a research question in plain English. LitLens searches arXiv and Semantic Scholar, ranks the
papers with its own information-retrieval engine, reads the PDFs, extracts model / dataset / metrics,
compares the papers, finds research gaps, writes a literature review, formats citations
(APA / IEEE / MLA / BibTeX), and can translate the review into 13 Indian languages.

It works **with no API key at all**: every stage has a classical-NLP implementation. If you add a
Claude API key, the summarisation, gap-finding, review-writing and translation steps use the LLM
instead, and the classical version stays as the fallback.

---

## Run it (one step)

1. Install **Python 3.10+** from python.org. Tick "Add Python to PATH" during setup.
2. Double-click **`start.bat`**.

The first run installs the dependencies and downloads the NLTK data, which takes a few minutes.
After that, the app opens at **http://localhost:8000**. Close the black window to stop it.

*Optional:* to use Claude, copy `backend/.env.example` to `backend/.env` and set `LLM_API_KEY=`.
The badge in the top-right corner shows which mode the app is running in.

Manual start (any OS):
```bash
python -m venv .venv
.venv/Scripts/pip install -r backend/requirements.txt     # Linux/Mac: .venv/bin/pip
cd backend && ../.venv/Scripts/python -m uvicorn app.main:app --port 8000
```
API docs (Swagger): http://localhost:8000/docs

---

## Features

| Tab | What you can do |
|---|---|
| **Research** | Ask a question → live pipeline progress → report with tabs: Review, Papers, Comparison, Gaps, Citations, **NLP insights**, Translate. Upload your own PDF into a report, save it as a project, or export it as Markdown. |
| **Library** | Saved projects and recent queries, which you can reopen at any time (stored in SQLite). |

---

## Architecture

```
Question
  │  Query understanding  - tokenize, spell-correct (edit distance), POS tag,
  │                         chunk-parse noun phrases, WordNet WSD (Lesk), expansion
  ▼
Search agent            - arXiv API + Semantic Scholar API, de-duplicated (Jaccard on titles)
  ▼
Rank agent              - own inverted index → Boolean + TF-IDF/VSM + BM25 → Reciprocal Rank Fusion
  ▼
PDF agent               - PyMuPDF text, de-hyphenation, section splitting, references removed
  ▼
Summarise/extract agent - Claude JSON (validated by Pydantic)  OR  TF-IDF extractive summary + regex IE
  ▼
Compare + gap agent     - comparison table, highlights; gaps via Claude OR rule-based corpus analysis
  ▼
Review agent            - Claude OR template-based NLG (content determination → document planning
  ▼                       → microplanning → surface realisation)
Citation agent          - APA 7 / IEEE / MLA 9 / BibTeX templates
  ▼
Report  (+ optional machine translation into Indian languages)
```

Stack: FastAPI · SQLite/SQLAlchemy · NLTK · NumPy · PyMuPDF · plain-JS frontend (no build step) ·
optional Claude (`claude-opus-5-5`).

```
backend/app/
  agents/      search_rank, pdf_extract, summarize_extract, compare_gap, lit_review, citation
  nlp/         preprocessing, spell, ngram_lm, wsd, ir, word2vec, nlg, translate, extractive, llm
  routers/     query, papers, report, projects, nlp_lab
  pipeline.py  orchestration
frontend/      index.html, app.js, styles.css
```

---

## Syllabus coverage (BCA305B-5)

Everything below is implemented **from scratch or with NLTK**. "Lab N" entries are API endpoints
under `/nlp/...`; you can run them interactively at http://localhost:8000/docs.

| Unit | Concept | Where in the code | Where in the UI |
|---|---|---|---|
| U1 | Word/sentence tokenization (NLTK + regex) | `nlp/preprocessing.py` | Lab 1 |
| U1 | Rule-based vs statistical vs neural paradigms | Regex IE + TF-IDF + n-gram (classical) vs Claude (neural), switchable | Mode badge, "method" notes |
| U1 | Ambiguity in NLP | WordNet sense counts for query words | Report → NLP insights; Lab 6 |
| U2 | Regular expressions, text normalization | `preprocessing.normalize`, `extractive.py` patterns | Lab 1 |
| U2 | Edit distance, spelling correction | `nlp/spell.py` (DP Levenshtein + candidate generation + Brown frequencies) | Lab 2b; typos in questions are auto-corrected |
| U2 | Stemming vs lemmatization | Porter / Lancaster / Snowball vs WordNet lemmatizer | Lab 2 |
| U2 | Word frequency, stop words, POS tagging | `preprocessing.word_frequency`, `pos_tag` | Lab 3; query POS tags |
| U2 | Parsing | Cascaded regexp chunker (shallow) + CFG ChartParser (deep) | Lab 4; noun phrases drive the search |
| U3 | N-gram LM, sentence probability, perplexity | `nlp/ngram_lm.py` (trigram, add-k smoothing, stupid backoff) | Lab 5; each paper's perplexity |
| U3 | Lexical semantics, WSD | `nlp/wsd.py` (Lesk); synonym expansion only when Lesk agrees with the most-frequent sense | Lab 6; NLP insights |
| U4 | NLG architecture | `nlp/nlg.py` writes the whole literature review and records each stage | Lab 7; NLP insights → NLG trace |
| U4 | Machine translation, Indian languages case study | `nlp/translate.py` (LLM → Google NMT → MyMemory fallback) | Lab 8; Report → Translate |
| U5 | Inverted index, Boolean, VSM, TF-IDF, probabilistic (BM25) | `nlp/ir.py` | Lab 9; ranks every search |
| U5 | Precision, Recall, F-measure, MAP | `ir.precision_recall_f`, `average_precision`, `mean_average_precision` | Lab 9; NLP insights → "Evaluate the rankings" |
| U5 | One-hot, Bag-of-Words | `nlp/word2vec.py` | Lab "+" |
| U5 | Word2Vec (CBOW & Skip-gram) | NumPy implementation with negative sampling | Lab 10 (trains on the retrieved papers, with a PCA plot) |

### Lab exercise checklist
1 Tokenization ✔ · 2 Stemming vs lemmatization ✔ · 3 Frequency/stop words/POS ✔ · 4 Parsing ✔ ·
5 N-gram next word ✔ · 6 Ambiguous words (WordNet) ✔ · 7 Template NLG ✔ · 8 English → Indian language ✔ ·
9 Precision/Recall/F ✔ · 10 Word2Vec ✔

---

## Viva cheat-sheet

- **Why a hybrid ranker?** Boolean matching is precise but can't rank. TF-IDF rewards rare matching
  terms. BM25 adds term-frequency saturation and document-length normalisation. Reciprocal Rank
  Fusion combines the three rankings without having to calibrate their scores.
  You can show this with real numbers in *NLP insights → Evaluate the rankings*.
- **Why gate query expansion?** For "Indian languages", WordNet's first sense of *Indian* gives
  "red indian", which is the wrong sense. Simplified Lesk alone is also unreliable on short queries.
  So a word is expanded only when Lesk and the most-frequent sense agree.
- **Spelling correction:** generate every candidate within edit distance 1 (then 2). Rank by
  (Levenshtein distance, −corpus frequency). Frequency is what picks *learning* over *leaning*
  for "lerning".
- **N-gram LM:** add-k smoothing gives unseen n-grams non-zero probability. Stupid backoff handles
  prediction when the trigram context has never been seen. Perplexity = exp(−avg log-prob).
- **Word2Vec:** skip-gram predicts each context word from the center word. CBOW predicts the center
  from the averaged context. Negative sampling turns the softmax into k+1 logistic regressions.
  Updates are averaged per word inside each mini-batch for stability.
- **NLG:** content determination → document planning (themes from model families) → microplanning
  (aggregation, author-year referring expressions, verb variety, subject-verb agreement) →
  surface realisation.
- **Why the LLM is optional:** every LLM output is requested as JSON and validated with Pydantic.
  If the call fails (no key, refusal, or invalid JSON), the classical path runs, so the pipeline
  never crashes.

## Known limitations
- Semantic Scholar rate-limits anonymous use (HTTP 429). arXiv results still work, and you can add
  `SEMANTIC_SCHOLAR_API_KEY` for more results.
- Regex extraction misses metrics the paper doesn't state as a percentage (e.g. BLEU scores).
- Some networks block `translate.google.com`. The app then falls back to MyMemory
  (about 5,000 free characters per day).
- Scanned (image-only) PDFs aren't supported, because there is no OCR.

## Future work
Knowledge graph of methods (NetworkX) · PubMed / OpenAlex sources · cross-encoder re-ranking ·
slide generation from the review · alerts for new papers.
