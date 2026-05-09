"""
Streamlit dashboard for English Named Entity Recognition (NER)
using dslim/bert-base-NER model.

Run with: streamlit run dashboard_english.py
"""

import hashlib
import io

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from predict_english import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    ENTITY_HTML_COLORS,
    build_highlight_html,
    extract_all_lines_from_pdf,
    extract_headings_from_pdf,
    extract_lines_from_image,
    load_model,
    ocr_images,
    probe_pdf,
    render_pdf_pages_as_images,
    run_ner,
)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="English NER",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("🔍 English NER")
st.sidebar.markdown(
    """
    **About**

    This dashboard performs Named Entity Recognition on any English text
    using a pre-trained BERT model. Select a model below to switch between
    a faster news-optimised model and a larger general-purpose model.

    The model automatically identifies:
    - **PER** — Person names
    - **ORG** — Organisation names
    - **LOC** — Locations
    - **MISC** — Miscellaneous entities

    ---
    """
)

st.sidebar.markdown("### Model")
selected_model = st.sidebar.selectbox(
    "Choose NER model",
    options=list(AVAILABLE_MODELS.keys()),
    index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL),
    key="model_selector",
    label_visibility="collapsed",
)
model_meta = AVAILABLE_MODELS[selected_model]
st.sidebar.caption(model_meta["description"])
st.sidebar.markdown("---")

st.sidebar.markdown("### Entity Type Colors")
for entity_type, color in ENTITY_HTML_COLORS.items():
    col1, col2 = st.sidebar.columns([1, 4])
    with col1:
        st.markdown(
            f"<div style='background-color:{color}; padding:8px; border-radius:4px;'></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.write(f"**{entity_type}**")
st.sidebar.markdown("---")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Filter out predictions with confidence below this threshold",
)

# ============================================================================
# LOAD MODEL (cached across rerenders)
# ============================================================================
@st.cache_resource
def get_ner_model(model_name: str):
    """Load the NER model once per model_name and cache for the session."""
    return load_model(model_name)


try:
    ner_pipe, ner_tokenizer = get_ner_model(selected_model)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# ============================================================================
# SHARED HELPERS
# ============================================================================

def _text_hash(text: str) -> str:
    """Short hash used as a session_state cache key."""
    return hashlib.md5(text.encode()).hexdigest()


def render_entity_chart(entities: list):
    """Plotly bar chart coloured to match the project's entity palette."""
    counts = {}
    for e in entities:
        counts[e["entity"]] = counts.get(e["entity"], 0) + 1

    df = (
        pd.DataFrame(list(counts.items()), columns=["Type", "Count"])
        .sort_values("Count", ascending=False)
    )
    fig = px.bar(
        df,
        x="Type",
        y="Count",
        color="Type",
        color_discrete_map=ENTITY_HTML_COLORS,
        labels={"Type": "Entity Type", "Count": "Count"},
    )
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
    return df


def run_batch(lines: list[str]) -> list[dict]:
    """
    Native pipeline batching — all lines sent to the model in one call.
    """
    valid = [(i, l) for i, l in enumerate(lines) if l.strip()]
    if not valid:
        return [
            {"sentence": l, "raw_count": 0, "filtered_count": 0, "entities": []}
            for l in lines
        ]

    indices, texts = zip(*valid)
    batch_raw = ner_pipe(list(texts), batch_size=16)

    result_map = {}
    for idx, raw in zip(indices, batch_raw):
        entities = [
            {
                "entity": r["entity_group"],
                "word":   r["word"],
                "score":  round(float(r["score"]), 4),
                "start":  r["start"],
                "end":    r["end"],
            }
            for r in raw
        ]
        filtered = [e for e in entities if e["score"] >= confidence_threshold]
        result_map[idx] = {
            "sentence":       lines[idx],
            "raw_count":      len(entities),
            "filtered_count": len(filtered),
            "entities":       filtered,
        }

    return [
        result_map.get(
            i,
            {"sentence": lines[i], "raw_count": 0, "filtered_count": 0, "entities": []},
        )
        for i in range(len(lines))
    ]


def render_batch_results(results: list[dict]):
    """Per-sentence expanders + aggregate summary + chart."""
    all_entities = [e for r in results for e in r["entities"]]

    st.subheader("📋 Results by Sentence")
    for idx, result in enumerate(results, 1):
        n     = result["filtered_count"]
        label = f"Sentence {idx} — {n} entit{'y' if n == 1 else 'ies'}"
        with st.expander(label):
            st.write(f"**Text:** {result['sentence']}")
            if n > 0:
                st.markdown(
                    build_highlight_html(result["sentence"], result["entities"]),
                    unsafe_allow_html=True,
                )
                table_data = [
                    {
                        "Text":       e["word"],
                        "Type":       e["entity"],
                        "Confidence": f"{e['score']:.2%}",
                    }
                    for e in result["entities"]
                ]
                st.dataframe(
                    pd.DataFrame(table_data),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("*No entities detected above confidence threshold.*")

    st.subheader("📈 Aggregate Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sentences", len(results))
    c2.metric("Total Entities Found", len(all_entities))
    avg = len(all_entities) / len(results) if results else 0
    c3.metric("Avg Entities / Sentence", f"{avg:.1f}")

    if all_entities:
        render_entity_chart(all_entities)


def preview_and_run(
    lines: list[str],
    page_images: list | None,
    cache_key_prefix: str,
    run_btn_label: str = "🔍 Run NER",
    btn_key: str = "run_btn",
):
    """
    Side-by-side preview block used by every batch path.

    Left  — page image(s) for visual cross-referencing (if available).
    Right — editable text area seeded with the extracted lines.

    NER only runs on button click; result is cached in session_state
    keyed by the hash of whatever text the user has in the area.
    """
    st.markdown("---")
    st.markdown(
        "**Preview & edit extracted lines** before running NER. "
        "Remove page numbers, captions, or any artefacts you do not want analysed."
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        if page_images:
            st.markdown("**Page preview**")
            for i, img in enumerate(page_images, 1):
                if len(page_images) > 1:
                    st.caption(f"Page {i}")
                st.image(img, use_container_width=True)
        else:
            st.info("No visual preview available for this file.")

    with right:
        st.markdown("**Extracted text** (editable)")
        preview_text = st.text_area(
            label="lines",
            value="\n".join(lines),
            height=420,
            key=f"{cache_key_prefix}_preview",
            label_visibility="collapsed",
        )
        confirmed_lines = [l.strip() for l in preview_text.splitlines() if l.strip()]
        st.caption(f"{len(confirmed_lines)} headline(s) ready for analysis.")

    cache_key = f"{cache_key_prefix}_{_text_hash(preview_text)}"

    if st.button(run_btn_label, type="primary", key=btn_key):
        with st.spinner(f"Running NER on {len(confirmed_lines)} headline(s)..."):
            results = run_batch(confirmed_lines)
        st.session_state[cache_key] = results

    if cache_key in st.session_state:
        render_batch_results(st.session_state[cache_key])


# ============================================================================
# TABS
# ============================================================================
tab1, tab2 = st.tabs(["Single Headline", "Batch Mode"])

# Active model pill — visible in both tabs
st.sidebar.markdown(
    f"**Active model:** `{selected_model.split('/')[-1]}`"
)

# ============================================================================
# TAB 1 — SINGLE HEADLINE
# ============================================================================
with tab1:
    st.header("📝 Named Entity Recognition")

    sample_text = (
        "Elon Musk founded SpaceX in Hawthorne, California. "
        "The Nobel Peace Prize was awarded to the United Nations in 2001."
    )

    user_text = st.text_area(
        "Enter any English text to analyse:",
        value=sample_text,
        height=100,
        placeholder="Paste your headline here...",
    )

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        analyze_button = st.button("🔍 Analyse", key="analyze_single", type="primary")

    cache_key = f"ner_{_text_hash(user_text)}"

    if analyze_button:
        with st.spinner("Running NER analysis..."):
            raw = run_ner(ner_pipe, user_text, tokenizer=ner_tokenizer)
        st.session_state[cache_key] = raw

    if cache_key in st.session_state:
        raw_entities = st.session_state[cache_key]
        entities     = [e for e in raw_entities if e["score"] >= confidence_threshold]

        if not raw_entities:
            st.info("No entities detected in this headline.")
        else:
            if len(entities) < len(raw_entities):
                st.info(
                    f"Showing {len(entities)} of {len(raw_entities)} entities "
                    f"({len(raw_entities) - len(entities)} filtered by confidence threshold)."
                )

            if entities:
                st.subheader("✨ Highlighted Text")
                st.markdown(
                    build_highlight_html(user_text, entities),
                    unsafe_allow_html=True,
                )

                st.subheader("📊 Entity Details")
                table_data = [
                    {
                        "Entity Text":      e["word"],
                        "Type":             e["entity"],
                        "Confidence Score": f"{e['score']:.2%}",
                    }
                    for e in entities
                ]
                st.dataframe(
                    pd.DataFrame(table_data),
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("📈 Entity Type Distribution")
                render_entity_chart(entities)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Entities", len(entities))
                c2.metric("Unique Types", len({e["entity"] for e in entities}))
                c3.metric(
                    "Avg Confidence",
                    f"{sum(e['score'] for e in entities) / len(entities):.2%}",
                )
                c4.metric("Text Length", f"{len(user_text)} chars")

# ============================================================================
# TAB 2 — BATCH MODE
# ============================================================================
with tab2:
    st.header("📚 Batch Mode")
    st.markdown(
        "Upload a **plain-text file**, a **PDF** (digital-native or scanned), "
        "or an **image** (PNG / JPG) to run NER across multiple lines of text at once."
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf", "png", "jpg", "jpeg"],
        help=(
            "TXT: one headline per line. "
            "PDF: automatically detected as digital or scanned — OCR fallback if needed. "
            "Image: OCR is run to extract text."
        ),
    )

    if uploaded_file is not None:
        file_ext   = uploaded_file.name.rsplit(".", 1)[-1].lower()
        file_bytes = uploaded_file.read()

        # ==================================================================
        # PATH A — Plain-text file
        # ==================================================================
        if file_ext == "txt":
            st.markdown(f"**File type:** Plain text — `{uploaded_file.name}`")
            content = file_bytes.decode("utf-8")
            lines   = [l.strip() for l in content.splitlines() if l.strip()]
            st.success(f"✅ Loaded {len(lines)} headline(s).")
            preview_and_run(
                lines=lines,
                page_images=None,
                cache_key_prefix="txt",
                run_btn_label="🔍 Run NER",
                btn_key="run_txt",
            )

        # ==================================================================
        # PATH B — PDF
        # ==================================================================
        elif file_ext == "pdf":
            st.markdown(f"**File type:** PDF — `{uploaded_file.name}`")

            with st.spinner("Probing PDF..."):
                probe = probe_pdf(file_bytes)

            st.caption(
                f"Pages: {probe['page_count']} | "
                f"Characters extracted: {probe['char_count']} | "
                f"Text layer: {'✅ detected' if probe['has_text'] else '❌ not detected — OCR fallback'}"
            )

            # --------------------------------------------------------------
            # PATH B1 — Digital-native (embedded text layer found)
            # --------------------------------------------------------------
            if probe["has_text"]:
                st.info(
                    "Embedded text layer detected. "
                    "Choose whether to extract **all lines** or attempt "
                    "**headings only** via font-size analysis."
                )

                extract_mode = st.radio(
                    "Extraction mode",
                    options=["All lines", "Headings only (font-size heuristic)"],
                    horizontal=True,
                    key="pdf_extract_mode",
                    captions=["Extract every text line from the PDF", "Use font-size analysis to keep only large-font text"],
                )

                size_ratio = 1.4
                if extract_mode == "Headings only (font-size heuristic)":
                    size_ratio = st.slider(
                        "Heading size ratio",
                        min_value=1.2,
                        max_value=2.0,
                        value=1.4,
                        step=0.05,
                        help=(
                            "Characters whose font size ≥ (median page font size × ratio) "
                            "are treated as headings. Increase to be more selective."
                        ),
                        key="size_ratio_slider",
                    )

                with st.spinner("Extracting text from PDF..."):
                    if extract_mode == "All lines":
                        lines = extract_all_lines_from_pdf(file_bytes)
                    else:
                        lines = extract_headings_from_pdf(file_bytes, size_ratio=size_ratio)
                        if not lines:
                            st.warning(
                                "No text found above the font-size threshold. "
                                "Try lowering the heading size ratio, or switch to 'All lines'."
                            )
                            st.stop()

                    # Render pages for the visual preview sidebar
                    try:
                        page_images = render_pdf_pages_as_images(file_bytes)
                    except Exception:
                        page_images = None
                        st.caption(
                            "⚠️ Could not render page preview (Poppler may not be installed). "
                            "Text extraction still works normally."
                        )

                st.success(
                    f"✅ {len(lines)} line(s) extracted "
                    f"({'all text' if extract_mode == 'All lines' else 'large-font text only'})."
                )

                preview_and_run(
                    lines=lines,
                    page_images=page_images,
                    cache_key_prefix=f"pdf_native_{extract_mode}",
                    run_btn_label="🔍 Run NER on extracted lines",
                    btn_key="run_pdf_native",
                )

            # --------------------------------------------------------------
            # PATH B2 — Scanned / image-only PDF (OCR fallback)
            # --------------------------------------------------------------
            else:
                st.info(
                    "No embedded text detected — this appears to be a scanned PDF. "
                    "Rendering pages as images and running OCR (Tesseract)."
                )

                with st.spinner("Rendering pages and running OCR — this may take a moment..."):
                    try:
                        page_images = render_pdf_pages_as_images(file_bytes)
                    except Exception as ex:
                        st.error(
                            f"Could not render PDF pages: {ex}\n\n"
                            "Make sure **Poppler** is installed. See the README for instructions."
                        )
                        st.stop()

                    try:
                        lines = ocr_images(page_images)
                    except Exception as ex:
                        st.error(
                            f"OCR failed: {ex}\n\n"
                            "Make sure **Tesseract** is installed. See the README for instructions."
                        )
                        st.stop()

                st.success(
                    f"✅ OCR complete — {probe['page_count']} page(s), "
                    f"{len(lines)} line(s) extracted."
                )

                preview_and_run(
                    lines=lines,
                    page_images=page_images,
                    cache_key_prefix="pdf_scanned",
                    run_btn_label="🔍 Run NER on OCR output",
                    btn_key="run_pdf_scanned",
                )

        # ==================================================================
        # PATH C — Image (PNG / JPG / JPEG)
        # ==================================================================
        else:
            st.markdown(f"**File type:** Image — `{uploaded_file.name}`")

            with st.spinner("Running OCR on image..."):
                try:
                    pil_img, lines = extract_lines_from_image(io.BytesIO(file_bytes))
                except Exception as ex:
                    st.error(
                        f"OCR failed: {ex}\n\n"
                        "Make sure **Tesseract** is installed. See the README for instructions."
                    )
                    st.stop()

            st.success(f"✅ OCR complete — {len(lines)} line(s) extracted.")

            preview_and_run(
                lines=lines,
                page_images=[pil_img],
                cache_key_prefix="image",
                run_btn_label="🔍 Run NER on OCR output",
                btn_key="run_image",
            )
