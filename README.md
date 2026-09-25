# Equations Knowledge Graph — Source Code

**Project Work: Natural Language Processing, Summer 2026**
**OTH Amberg-Weiden — Prof. Dr. Patrick Levi**
**Student: Satya Naga Vamsi Ganesh Manepalli**

---

## What this code does

Extracts a structured equation knowledge graph from arXiv quantum physics papers.
For each enumerated equation the system extracts:

- **equation** — LaTeX source string
- **meaning** — natural-language description from surrounding prose
- **symbols** — dictionary of symbol → definition
- **relations** — pairwise graded relations to all other equations in the paper
- **audit-trail** — method-by-method evidence log (proves arXiv-only extraction)

Output is a single JSON file (`dataset_001.json`) ready for submission.

---

## File overview

| File | Role |
|---|---|
| `main_pipeline.py` | **Main entry point.** Run this to generate the dataset. |
| `equations_extractor.py` | Core extraction logic — equations, symbols, meanings, relations, audit trail. |
| `html_fetcher.py` | Fetches arXiv HTML from ar5iv.org with disk cache and polite crawl delay. |


---

## Requirements

- Python 3.9 or higher
- Internet access (for fetching papers not yet in cache)
- Approximately 500 MB disk space for the HTML cache

### Install dependencies

```bash
pip install -r requirements.txt
```

The full dependency list:

```
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
spacy>=3.0.0              # optional — improves symbol/meaning extraction
sentence-transformers     # optional — used for embedding similarity
```

Install spaCy language model (optional but recommended):

```bash
python -m spacy download en_core_web_sm
```

---

## How to run

### Generate the full dataset (350–356 equations)

```bash
python main_pipeline.py --paper-list paper_list_49.txt
```

Output: `dataset_001.json` (auto-numbered; subsequent runs produce `dataset_002.json`, etc.)

### Test on a few papers first

```bash
python main_pipeline.py --paper-list paper_list_49.txt --limit 5
```

### Resume after a crash

```bash
python main_pipeline.py --paper-list paper_list_49.txt --resume
```

### Write to a custom output path

```bash
python main_pipeline.py --paper-list paper_list_49.txt --out my_dataset.json --no-autonumber
```


---

## Expected output

```
dataset_001.json       ← submission dataset
cache/
  html/
    2506.23039.html    ← cached HTML per paper (auto-created)
    2401.12877.html
    ...
pipeline.log           ← full run log with per-paper stats
```

Final run summary (logged to console and `pipeline.log`):

```
Papers total   : 66  (with equations: 54)
Equations      : 352  (target: 350–356)
Symbols def'd  : 1118 / 1947  (57%)
Meanings found : 298 / 352  (84%)
```

---

## How extraction works

```
ar5iv HTML
    │
    ├─ Find equations   <span class="ltx_tag_equation">
    ├─ Extract LaTeX    <annotation encoding="application/x-tex">
    ├─ Extract symbols  MathML <mi> tokens (Unicode-folded, function-names dropped)
    │
    ├─ Define symbols (tiered sniper on ±10 sentence context window)
    │     Tier 1: trigger phrase  ("where X is", "let X be", "X denotes")
    │     Tier 2: appositive noun ("the Hamiltonian H")
    │     Tier 3: bare adjacency  (nearest content word)
    │     Fallback: weighted noun frequency + glue-word penalty
    │     Cross-eq: memory inheritance (eq n → n+1, n+2)
    │
    ├─ Extract meaning (4-strategy cascade)
    │     1. Named equation pattern  ("Schrödinger equation")
    │     2. Colon-introducer        (last sentence ending in ":")
    │     3. Governing verb          ("is given by", "reads", "describes")
    │     4. spaCy lemma scan        (defining verbs in preceding sentences)
    │
    └─ Grade relations (all pairs within a paper)
          Strong    : ≥2 shared symbols OR LaTeX Jaccard ≥ 0.85
          Potential : ≥1 shared symbol  OR LaTeX Jaccard ≥ 0.40
          None      : otherwise
```

---

## Constraints respected

- **arXiv HTML only** — all information extracted from ar5iv.org HTML. No PDF parsing, no external knowledge bases.
- **No prompting** — no language model API calls. spaCy is used only for dependency parsing (not text generation). `sentence-transformers` is used only for cosine similarity between phrase embeddings (vector operation, not generation).
- **Polite crawling** — 3-second delay between HTTP requests; disk cache prevents re-downloading.

---

## Audit trail format

Every equation in the JSON contains an `"audit-trail"` dictionary. Keys are the method names from the code; values show the evidence found:

```json
"audit-trail": {
  "fetch_html":        "fetched ar5iv.org/abs/2506.23039 HTML (145 KB); source: arXiv HTML only; no prompting",
  "extract_equations": "found eq (1) via <span class='ltx_tag_equation'>; LaTeX from <annotation>: 'H\\psi = E\\psi'",
  "extract_symbols":   "MathML <mi> tokens: ['H', 'psi', 'E']; summation indices: (none)",
  "define_symbols: H": "extracted from text: 'where H is the Hamiltonian' → H: Hamiltonian operator",
  "define_symbols: psi":"definition inherited from eq (1) [distance=1] → psi: wave function",
  "extract_meaning":   "extracted meaning from context of eq (1): 'Time-independent Schrödinger equation'"
}
```

---

## Known limitations

- **ZX-calculus papers** — equations are circuit diagrams in figures; symbol definitions not accessible from HTML text.
- **Figure captions** — definitions appearing only in figure captions are not scanned.
- **Ambiguous single-letter symbols** — `e`, `t`, `n` etc. may carry different meanings in different papers; memory inheritance helps but cannot resolve true ambiguity.
- **Standard symbol dictionary** — `phi`, `hbar` show low definition rates in this dataset; a pre-defined lookup table would improve coverage.

---

## References

- ar5iv.org — HTML rendering of arXiv papers: https://ar5iv.org
- Beautiful Soup 4: https://www.crummy.com/software/BeautifulSoup/
- spaCy: https://spacy.io
- sentence-transformers (Reimers & Gurevych, 2019): https://www.sbert.net
