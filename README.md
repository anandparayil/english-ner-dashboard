# English Named Entity Recogniser

Named Entity Recognition (NER) on any English text using pre-trained BERT-based
models, with an interactive Streamlit dashboard supporting single input, batch
processing, and PDF/image extraction with OCR fallback.

---

## Models

Two models are available and can be switched from the dashboard sidebar at any time.
The CLI always uses `dslim/bert-base-NER`.

| Model | Size | Speed | Best for |
|-------|------|-------|----------|
| [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) | ~430 MB | Fast | Formal / news-style text |
| [Jean-Baptiste/roberta-large-ner-english](https://huggingface.co/Jean-Baptiste/roberta-large-ner-english) | ~1.4 GB | Slower | General text, emails, mixed-case input |

Both models are downloaded automatically on first use and cached locally by HuggingFace.

### Entity types recognised

| Tag  | Meaning              | Example                   |
|------|----------------------|---------------------------|
| PER  | Person name          | *Elon Musk*           |
| ORG  | Organisation         | *Tesla*, *the UN*         |
| LOC  | Location             | *New Delhi*, *California* |
| MISC | Miscellaneous entity | *May*, *Nobel Prize*      |

---

## Project structure

```
english_ner/
├── predict_english.py    # Core NER logic, extraction helpers, CLI entry-point
├── dashboard_english.py  # Streamlit interactive dashboard
├── requirements.txt      # Python dependencies
└── README.md
```

---

## System dependencies

Two OS-level tools are required for PDF rendering and OCR. Install them once
before running the project.

### Tesseract OCR

Required for: image uploads (PNG/JPG) and scanned PDF fallback.

| OS              | Command                                                                                                               |
|-----------------|-----------------------------------------------------------------------------------------------------------------------|
| Ubuntu / Debian | `sudo apt install tesseract-ocr`                                                                                      |
| macOS           | `brew install tesseract`                                                                                              |
| Windows         | Download installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH                 |

Verify: `tesseract --version`

### Poppler

Required for: rendering PDF pages as images (`pdf2image` depends on it).

| OS              | Command                                                                                                               |
|-----------------|-----------------------------------------------------------------------------------------------------------------------|
| Ubuntu / Debian | `sudo apt install poppler-utils`                                                                                      |
| macOS           | `brew install poppler`                                                                                                |
| Windows         | Download from [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) and add `bin/` to PATH |

Verify: `pdftoppm -v`

---

## Python setup

**1. Create and activate a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

> **Note:** `torch` (~2 GB) and the chosen model weights are downloaded on first
> run. Subsequent runs use the local HuggingFace cache.

---

## CLI usage (`predict_english.py`)

The CLI always runs `dslim/bert-base-NER`. To use a different model, use the dashboard.

```bash
# Analyse a single piece of text
python predict_english.py --text "Elon Musk founded SpaceX in Hawthorne, California."

# Analyse all lines in a text file (one input per line)
python predict_english.py --file data/eng_news.txt

# Save results as JSON files in results/english/
python predict_english.py --file data/eng_news.txt --save

# --save also works with --text
python predict_english.py --text "The Nobel Prize was awarded to the UN in 2001." --save
```

Output is colour-coded in the terminal and grouped by entity type.
JSON output includes entity text, type, character span, and confidence score.

---

## Dashboard (`dashboard_english.py`)

```bash
streamlit run dashboard_english.py
```

### Sidebar

- **Model selector** — switch between `dslim/bert-base-NER` (fast, ~430 MB) and
  `Jean-Baptiste/roberta-large-ner-english` (more accurate on general text, ~1.4 GB).
  Each model is loaded once and cached for the session; switching triggers a fresh load.
- **Confidence threshold** — filter out predictions below a chosen confidence score.
  In the Single Text tab this is applied reactively. In Batch Mode it is applied at
  run-time when **Run NER** is clicked.
- **Entity type colours** — colour legend for the four entity types.

### Single Text tab

Paste any English text and click **Analyse**. Results include:

- Colour-highlighted text with inline entity labels
- Entity details table (text, type, confidence score)
- Entity type distribution bar chart (colour-matched to entity type)
- Summary metrics (total entities, unique types, average confidence, text length)

### Batch Mode tab

Upload one of three file types:

#### Plain-text file (`.txt`)
One input per line. All non-empty lines are loaded into the editable preview.

#### PDF file (`.pdf`)

The dashboard automatically probes the PDF and routes accordingly:

**Digital-native PDF** (has an embedded text layer)
- Choose **All lines** to extract all text from the PDF.
- Choose **Headings only (font-size heuristic)** to keep only text whose font
  size is significantly larger than the median page font size — useful for
  extracting titles or section headings from structured documents.
- The **Heading size ratio** slider (1.2 – 2.0) controls how selective the
  heuristic is. A ratio of 1.4 means text must be at least 40 % larger than
  the median body font to be retained. Increase it to be more restrictive.
- Page images are rendered alongside the editable text for visual cross-referencing.

**Scanned / image-only PDF** (no embedded text layer detected)
- Pages are rendered as images and passed through Tesseract OCR automatically.
- The same side-by-side layout shows each page image next to the OCR output
  for review and editing before NER runs.

> **Note on font-size extraction:** This heuristic works best on structured
> digital-native PDFs. If your PDF encodes all text at the same nominal size
> (common in some print-to-PDF workflows), the heuristic may not distinguish
> headings from body text — use "All lines" in that case and remove unwanted
> lines in the preview manually.

#### Image file (`.png`, `.jpg`, `.jpeg`)
Tesseract OCR is run on the image. The image is shown on the left alongside
the editable extracted text on the right for cross-referencing and cleanup
before NER runs.

---

## How the editable preview works

Every batch path — TXT, PDF (both variants), and image — shows an editable
text area before NER runs. This is intentional:

- PDF extraction and OCR are imperfect. Page numbers, running headers,
  captions, and ligature errors are common artefacts.
- Editing the preview lets you remove or correct lines before they reach the
  model, giving cleaner results without any code changes.
- NER only runs when you click the **Run NER** button. Adjusting the confidence
  threshold slider or switching tabs does not re-run the model.

---

## Known limitations

- Both models were fine-tuned on the **CoNLL-2003 Reuters newswire** corpus, so
  they perform best on formal, written English. Accuracy may be lower on casual
  text, social media, or heavily technical writing.
- BERT's hard token limit is **512 word-piece tokens**. Inputs approaching this
  length will be silently truncated. A soft warning is emitted in the terminal
  for inputs exceeding 60 tokens, which is unusual for typical short texts.
- The font-size heading heuristic in Batch Mode is a best-effort approach — it
  is not a substitute for a document parser and may require manual cleanup in
  the editable preview.

---

## Notes

- Batch inference uses the HuggingFace pipeline's native batching
  (`batch_size=16`) for efficient multi-input processing.
- Dashboard results are cached in `st.session_state` keyed by the MD5 hash of
  the current preview text, so slider adjustments and tab switches never
  re-invoke the model unnecessarily.
- CLI results saved with `--save` are written to `results/english/` as
  individual JSON files, one per input line, with a sanitised filename slug.
