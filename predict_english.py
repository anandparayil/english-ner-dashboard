"""
predict_english.py
Runs Named Entity Recognition on any English text using
dslim/bert-base-NER — a BERT model fine-tuned on CoNLL-2003 Reuters data.
No training needed. Model is downloaded automatically on first run (~430 MB).

Usage:
    # Single input
    python predict_english.py --text "Elon Musk founded SpaceX in Hawthorne, California."

    # From a text file (one input per line)
    python predict_english.py --file data/eng_news.txt

    # Save output to results folder
    python predict_english.py --text "The Nobel Prize was awarded to the UN in 2001." --save
"""

import argparse
import html
import io
import json
import os
import re
import statistics

import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# MODEL_NAME = "dslim/bert-base-NER"
MODEL_NAME = "Jean-Baptiste/roberta-large-ner-english"
# Registry of models available for selection in the dashboard.
# predict_english.py always uses MODEL_NAME above; this dict is imported
# by dashboard_english.py to populate the model selector.
AVAILABLE_MODELS = {
    "dslim/bert-base-NER": {
        "description": "BERT-base · ~430 MB · Fast · Fine-tuned on CoNLL-2003 news data",
        "size": "~430 MB",
        "speed": "Fast",
    },
    "Jean-Baptiste/roberta-large-ner-english": {
        "description": "RoBERTa-large · ~1.4 GB · Slower · Better on general / non-news text",
        "size": "~1.4 GB",
        "speed": "Slower",
    },
}
DEFAULT_MODEL = "dslim/bert-base-NER"

# Soft token warning threshold. BERT's hard limit is 512, but news headlines
# should never approach that. We warn above 60 word-piece tokens so that
# unusually long inputs are flagged before silent truncation can occur.
TOKEN_WARN_THRESHOLD = 60

ENTITY_COLORS = {
    "PER":  "\033[94m",   # Blue
    "ORG":  "\033[92m",   # Green
    "LOC":  "\033[93m",   # Yellow
    "MISC": "\033[95m",   # Magenta
}
RESET = "\033[0m"
BOLD  = "\033[1m"

ENTITY_HTML_COLORS = {
    "PER":  "#cce5ff",   # Light blue
    "ORG":  "#d4edda",   # Light green
    "LOC":  "#fff3cd",   # Light yellow
    "MISC": "#e2d9f3",   # Light purple
}

# ---------------------------------------------------------------------------
# Module-level singleton so load_model() never re-downloads mid-session
# ---------------------------------------------------------------------------
_pipeline = None
_tokenizer = None


def load_model(model_name: str = MODEL_NAME):
    """
    Load a NER model and return (pipeline, tokenizer).

    When called from the CLI (no argument), MODEL_NAME is used and the result
    is cached in the module-level singleton so repeated CLI calls never
    re-download.  When called from the dashboard with an explicit model_name,
    st.cache_resource handles caching keyed by model_name, so each model is
    loaded at most once per session.
    """
    global _pipeline, _tokenizer

    # Module-level singleton — only used for the default CLI model.
    if model_name == MODEL_NAME:
        if _pipeline is None:
            print(f"Loading model: {model_name}")
            print("(Downloads on first run, cached after that)\n")
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _m = AutoModelForTokenClassification.from_pretrained(model_name)
            _pipeline = pipeline(
                "ner",
                model=_m,
                tokenizer=_tokenizer,
                aggregation_strategy="simple",
            )
            print("Model loaded.\n")
        return _pipeline, _tokenizer

    # Non-default model (dashboard only) — caller (st.cache_resource) caches.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _m = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline(
        "ner",
        model=_m,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    return ner_pipeline, tokenizer


def check_token_length(text, tokenizer):
    """
    Soft token guard.
    Tokenise without truncation and warn if the input exceeds TOKEN_WARN_THRESHOLD.
    Returns the token count so callers can decide what to do with it.
    """
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    n = len(tokens)
    if n > TOKEN_WARN_THRESHOLD:
        print(
            f"[WARNING] Input has {n} tokens, which exceeds the soft threshold "
            f"of {TOKEN_WARN_THRESHOLD}. "
            "BERT will silently truncate at 512 tokens if exceeded."
        )
    return n


def run_ner(pipe, text, tokenizer=None):
    """
    Run NER on *text* and return a list of entity dicts.
    Guards against empty / whitespace-only input.
    Emits a soft warning for unexpectedly long inputs when a tokenizer is supplied.
    """
    if not text or not text.strip():
        return []

    if tokenizer is not None:
        check_token_length(text, tokenizer)

    results = pipe(text)
    entities = []
    for r in results:
        entities.append({
            "entity": r["entity_group"],
            "word":   r["word"],
            "score":  round(float(r["score"]), 4),
            "start":  r["start"],
            "end":    r["end"],
        })
    return entities


def build_highlight_html(text, entities, color_map=None):
    """
    Returns an HTML string with entity spans highlighted and labelled.
    All user-supplied text is passed through html.escape() before being
    inserted into the HTML string, preventing injection.
    Overlapping spans are silently skipped (last-writer-wins on start offset).
    """
    if color_map is None:
        color_map = ENTITY_HTML_COLORS

    # Sort and deduplicate overlapping spans
    entities_sorted = sorted(entities, key=lambda e: e["start"])
    clean = []
    last_end = 0
    for e in entities_sorted:
        if e["start"] >= last_end:
            clean.append(e)
            last_end = e["end"]

    html_parts = []
    cursor = 0
    for e in clean:
        # Safe plain text before this entity
        html_parts.append(html.escape(text[cursor:e["start"]]))
        color = color_map.get(e["entity"], "#f0f0f0")
        word  = html.escape(e["word"])
        label = html.escape(e["entity"])
        html_parts.append(
            f"<span style='background-color:{color}; padding:2px 6px; "
            f"border-radius:3px; margin:0 2px;'>"
            f"{word}"
            f"<sup style='font-size:0.7em; margin-left:3px; "
            f"font-weight:bold; color:#333;'>{label}</sup>"
            f"</span>"
        )
        cursor = e["end"]
    html_parts.append(html.escape(text[cursor:]))
    return "".join(html_parts)


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

# Minimum character count returned by pdfplumber for us to consider the PDF
# as having a real embedded text layer (vs. being effectively image-only).
_PDF_TEXT_MIN_CHARS = 50


def probe_pdf(file_bytes: bytes) -> dict:
    """
    Inspect a PDF and return a dict describing what kind it is:

        {
            "has_text":   bool,   # True if embedded text layer is usable
            "page_count": int,
            "char_count": int,    # total chars extracted across all pages
        }

    This is the routing decision point: callers use ``has_text`` to decide
    whether to run pdfplumber extraction or fall back to OCR via pdf2image.
    """
    char_count = 0
    page_count = 0
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text() or ""
                char_count += len(text)
    except Exception:
        pass
    return {
        "has_text":   char_count >= _PDF_TEXT_MIN_CHARS,
        "page_count": page_count,
        "char_count": char_count,
    }


def extract_all_lines_from_pdf(file_bytes: bytes) -> list[str]:
    """
    Extract every non-empty text line from a digital-native PDF using pdfplumber.
    Returns a flat list of stripped lines across all pages.
    """
    lines = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
    return lines


def extract_headings_from_pdf(file_bytes: bytes, size_ratio: float = 1.4) -> list[str]:
    """
    Attempt to extract only headline-sized text from a digital-native PDF.

    Strategy:
      1. Collect every character's font size from each page via pdfplumber.
      2. Compute the median font size for that page (proxy for body text size).
      3. Keep only characters whose size >= median * size_ratio.
      4. Group surviving characters into lines by rounding their top-coordinate
         to the nearest 2pt bucket, then sort buckets top-to-bottom.

    ``size_ratio`` is exposed as a dashboard slider (1.2 – 2.0).
    Returns a flat list of candidate headline strings.
    """
    headings = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            chars = page.chars  # list of char dicts with 'text', 'size', 'top'
            if not chars:
                continue

            sizes = [c["size"] for c in chars if c.get("size")]
            if not sizes:
                continue
            median_size = statistics.median(sizes)
            threshold   = median_size * size_ratio

            # Keep only large-font characters
            big_chars = [c for c in chars if c.get("size", 0) >= threshold]
            if not big_chars:
                continue

            # Group by vertical position (bucket to 2pt to merge same-line chars)
            buckets: dict[int, list] = {}
            for c in big_chars:
                bucket = round(c["top"] / 2) * 2
                buckets.setdefault(bucket, []).append(c)

            # Sort buckets top-to-bottom and assemble text
            for bucket_top in sorted(buckets):
                line_chars = sorted(buckets[bucket_top], key=lambda c: c["x0"])
                line_text  = "".join(c["text"] for c in line_chars).strip()
                if line_text:
                    headings.append(line_text)

    return headings


def render_pdf_pages_as_images(file_bytes: bytes, dpi: int = 150) -> list[Image.Image]:
    """
    Convert every page of a PDF to a PIL Image using pdf2image (poppler).
    Used both for OCR on scanned PDFs and for the side-by-side visual preview
    on digital-native PDFs.
    ``dpi=150`` is enough for screen display and reasonably fast to render.
    """
    return convert_from_bytes(file_bytes, dpi=dpi)


def ocr_images(images: list[Image.Image]) -> list[str]:
    """
    Run Tesseract OCR on a list of PIL Images and return a flat list of
    non-empty text lines across all images (pages).
    """
    lines = []
    for img in images:
        text = pytesseract.image_to_string(img)
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


# ---------------------------------------------------------------------------
# Image (PNG / JPG) helper
# ---------------------------------------------------------------------------

def extract_lines_from_image(source) -> tuple[Image.Image, list[str]]:
    """
    Open *source* (a file path, file-like object, or PIL Image) and run
    Tesseract OCR on it.

    Returns ``(pil_image, lines)`` so the caller can display the image
    alongside the extracted text for cross-referencing.
    """
    if isinstance(source, Image.Image):
        img = source
    else:
        img = Image.open(source)

    text  = pytesseract.image_to_string(img)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return img, lines


# ---------------------------------------------------------------------------
# CLI output helpers
# ---------------------------------------------------------------------------

def format_output(text, entities):
    print(f"{BOLD}Input:{RESET} {text}\n")

    if not entities:
        print("No named entities found.\n")
        return

    print(f"{'Word':<35} {'Entity':<8} {'Confidence'}")
    print("-" * 55)
    for e in entities:
        color = ENTITY_COLORS.get(e["entity"], "")
        print(f"{color}{e['word']:<35} {e['entity']:<8} {e['score']}{RESET}")

    print(f"\n{BOLD}Extracted Entities:{RESET}")
    print("-" * 55)
    grouped = {}
    for e in entities:
        grouped.setdefault(e["entity"], []).append(e["word"])
    for etype, words in grouped.items():
        color = ENTITY_COLORS.get(etype, "")
        print(f"  {color}[{etype}]{RESET}  {', '.join(words)}")
    print()


def save_results(text, entities, output_dir="results/english"):
    """Filename slug is sanitised with re.sub to strip all unsafe characters."""
    os.makedirs(output_dir, exist_ok=True)
    slug = re.sub(r"[^\w\-]", "_", text[:40].strip())
    slug = re.sub(r"_+", "_", slug).strip("_")   # collapse runs of underscores
    out_path = os.path.join(output_dir, f"{slug}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"input": text, "entities": entities}, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}\n")


def process_file(pipe, file_path, tokenizer=None, save=False):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"Processing {len(lines)} lines from {file_path}\n")
    print("=" * 60)
    for i, line in enumerate(lines, 1):
        print(f"[{i}/{len(lines)}]")
        entities = run_ner(pipe, line, tokenizer=tokenizer)
        format_output(line, entities)
        if save:
            save_results(line, entities)
        print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="English NER using dslim/bert-base-NER")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Any English text to analyse")
    group.add_argument("--file", type=str, help="Text file with one input per line")
    parser.add_argument("--save", action="store_true", help="Save results to results/english/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipe, tokenizer = load_model()

    if args.text:
        entities = run_ner(pipe, args.text, tokenizer=tokenizer)
        format_output(args.text, entities)
        if args.save:
            save_results(args.text, entities)
    elif args.file:
        process_file(pipe, args.file, tokenizer=tokenizer, save=args.save)
