"""Streamlit web UI for the document-intelligence RAG pipeline.
Upload a PDF, ask questions, get grounded answers with source citations.
Run with: streamlit run app.py
"""

import os
import re
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.error(
            "GEMINI_API_KEY not found. Add it to .env locally, "
            "or to Streamlit secrets when deployed."
        )
        st.stop()

from google import genai
from chroma_store import get_chroma_collection, query_collection
from generator import generate_answer
from ingest_text import ingest_file

CHROMA_PATH = "chroma_db"
CHUNK_SIZE = 500
OVERLAP = 100

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_source" not in st.session_state:
    st.session_state.indexed_source = None

DARK = {
    "bg": "#0B0D11",
    "surface": "#12151B",
    "surface2": "#161A21",
    "sidebar": "#090B0F",
    "border": "rgba(255,255,255,0.08)",
    "border_soft": "rgba(255,255,255,0.05)",
    "text": "#EDEFF2",
    "text_soft": "rgba(237,239,242,0.62)",
    "text_mute": "rgba(237,239,242,0.34)",
    "accent": "#6366F1",
    "accent_soft": "rgba(99,102,241,0.10)",
    "user_bg": "rgba(99,102,241,0.08)",
}

LIGHT = {
    "bg": "#FFFFFF",
    "surface": "#F7F8FA",
    "surface2": "#FFFFFF",
    "sidebar": "#FAFAFB",
    "border": "rgba(15,23,42,0.10)",
    "border_soft": "rgba(15,23,42,0.06)",
    "text": "#0F172A",
    "text_soft": "rgba(15,23,42,0.62)",
    "text_mute": "rgba(15,23,42,0.40)",
    "accent": "#5B4FE9",
    "accent_soft": "rgba(91,79,233,0.07)",
    "user_bg": "rgba(91,79,233,0.06)",
}


def inject_css(t: dict) -> None:
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp label, .stApp li, .stApp span, .stApp div,
.stApp input, .stApp textarea, .stApp button {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* Allow Streamlit to render its native material icons */
[data-testid="stIconMaterial"],
.material-symbols-rounded, .material-symbols-outlined, .material-icons,
span[class*="material-symbols"] {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Icons' !important;
    font-feature-settings: 'liga' !important;
}}

.stApp {{ background: {t['bg']} !important; }}
[data-testid="stAppViewContainer"] > .main .block-container {{
    padding-top: 2rem !important;
    padding-bottom: 7rem !important;
    max-width: 800px !important;
}}

[data-testid="stSidebar"] {{
    background: {t['sidebar']} !important;
    border-right: 1px solid {t['border_soft']} !important;
}}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{ 
    color: {t['text_soft']}; 
}}
[data-testid="stSidebar"] h3 {{
    font-size: 10.5px !important;
    font-weight: 600 !important;
    letter-spacing: 0.11em !important;
    text-transform: uppercase !important;
    color: {t['text_mute']} !important;
    margin: 20px 0 8px 0 !important;
}}

/* ---- Brand header ---- */
.brand {{ display: flex; align-items: center; gap: 11px; margin-bottom: 7px; }}
.brand-mark {{
    width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, {t['accent']}, #8B5CF6);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: 700; color: #fff;
    flex: 0 0 auto;
}}
.brand-name {{
    font-size: 21px; font-weight: 650; letter-spacing: -0.02em;
    color: {t['text']};
}}
.brand-sub {{
    font-size: 13.5px; color: {t['text_soft']};
    margin-bottom: 24px; line-height: 1.6;
}}

/* ---- Document status pill ---- */
.doc-row {{
    display: flex; align-items: center; gap: 10px;
    background: {t['accent_soft']};
    border: 1px solid {t['border']};
    border-radius: 10px; padding: 11px 15px;
    margin: 16px 0 22px 0;
    font-size: 13px; color: {t['text']};
    animation: fadeIn 0.35s ease-out;
}}
.doc-dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: #22C55E; box-shadow: 0 0 0 3px rgba(34,197,94,0.16);
    flex: 0 0 auto;
}}
.doc-meta {{ color: {t['text_mute']}; margin-left: auto; font-size: 12px; }}

.empty-hint {{
    text-align: center; padding: 34px 0 8px 0;
    color: {t['text_mute']}; font-size: 13px;
    animation: fadeIn 0.6s ease-out;
}}

/* ---- File uploader ---- */
[data-testid="stFileUploader"] section {{
    background: {t['surface']} !important;
    border: 1px dashed {t['border']} !important;
    border-radius: 12px !important;
    transition: border-color 0.18s ease, background 0.18s ease;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: {t['accent']} !important;
    background: {t['accent_soft']} !important;
}}

/* ---- Chat messages as cards ---- */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {{ display: none !important; }}

[data-testid="stChatMessage"] {{
    background: {t['surface2']} !important;
    border: 1px solid {t['border_soft']} !important;
    border-radius: 12px !important;
    padding: 15px 18px !important;
    margin-bottom: 14px !important;
    animation: msgIn 0.28s ease-out;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
    background: {t['user_bg']} !important;
    border-color: {t['border']} !important;
}}
[data-testid="stChatMessage"] p {{
    font-size: 14.5px !important;
    line-height: 1.7 !important;
    color: {t['text']} !important;
}}

@keyframes msgIn {{ from {{ opacity: 0; transform: translateY(7px); }}
                    to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

/* ---- Chat input ---- */
[data-testid="stChatInput"] {{
    border: 1px solid {t['border']} !important;
    border-radius: 12px !important;
    background: {t['surface2']} !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
}}
[data-testid="stChatInput"]:focus-within {{
    border-color: {t['accent']} !important;
    box-shadow: 0 0 0 3px {t['accent_soft']} !important;
}}
[data-testid="stChatInput"] textarea {{ color: {t['text']} !important; }}

/* ---- Sources ---- */
[data-testid="stExpander"] {{
    border: none !important; background: transparent !important;
    margin-top: 10px !important;
}}
[data-testid="stExpander"] summary {{
    font-size: 12.5px !important;
    color: {t['text_mute']} !important;
    padding-left: 0 !important;
    transition: color 0.15s ease;
}}
[data-testid="stExpander"] summary:hover {{ color: {t['accent']} !important; }}

.src-card {{
    background: {t['surface']};
    border: 1px solid {t['border_soft']};
    border-radius: 9px;
    padding: 11px 14px;
    margin-bottom: 9px;
}}
.src-head {{
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; font-weight: 600;
    color: {t['text']}; margin-bottom: 6px;
}}
.src-tag {{
    font-size: 10.5px; font-weight: 500;
    padding: 2px 7px; border-radius: 20px;
    background: {t['accent_soft']}; color: {t['accent']};
    margin-left: auto;
}}
.src-body {{
    font-size: 12.5px; line-height: 1.6;
    color: {t['text_soft']};
}}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_clients():
    genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    collection = get_chroma_collection()
    return genai_client, collection


def save_uploaded_pdf(uploaded_file) -> str:
    path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\(Source:[^)]*\)", "", text).strip()


def relevance_label(similarity: float) -> str:
    if similarity > 0.7:
        return "High"
    if similarity > 0.4:
        return "Medium"
    return "Low"


def render_sources(chunks: list[dict]) -> None:
    with st.expander(f"View {len(chunks)} sources"):
        for c in chunks:
            source = c["metadata"].get("source_file", "unknown")
            body = c["document"].replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f'<div class="src-card">'
                f'<div class="src-head">{source}'
                f'<span class="src-tag">{relevance_label(c["similarity"])}</span></div>'
                f'<div class="src-body">{body}</div></div>',
                unsafe_allow_html=True,
            )


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Appearance")
    theme = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.theme = theme

    st.markdown("### Document")
    uploaded = st.file_uploader(
        "Upload a PDF", type=["pdf"], label_visibility="collapsed"
    )

    st.markdown("### Session")
    st.caption(f"Loaded: {st.session_state.indexed_source or 'none'}")
    st.caption(
        f"Questions: {sum(1 for m in st.session_state.messages if m['role'] == 'user')}"
    )

    if st.session_state.messages and st.button(
        "Clear conversation", use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### About")
    st.caption(
        "Retrieval-Augmented Generation over your PDF. Answers come only from "
        "the uploaded document and cite the passages they came from. "
        "Text-based PDFs only."
    )

inject_css(DARK if st.session_state.theme == "Dark" else LIGHT)

# ---------- Brand header ----------
st.markdown(
    '<div class="brand"><div class="brand-mark">DI</div>'
    '<div class="brand-name">Document Intelligence</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="brand-sub">Chat with any PDF. Every answer is grounded in your '
    "document and shows the exact passages it came from — no invented facts.</div>",
    unsafe_allow_html=True,
)

# ---------- Ingest ----------
if uploaded is not None and st.session_state.indexed_source != uploaded.name:
    with st.status(f"Indexing {uploaded.name}...", expanded=False) as status:
        try:
            pdf_path = save_uploaded_pdf(uploaded)
            if os.path.isdir(CHROMA_PATH):
                shutil.rmtree(CHROMA_PATH)
            get_clients.clear()
            genai_client, _ = get_clients()
            ingest_file(genai_client, pdf_path, CHUNK_SIZE, OVERLAP, is_pdf=True)
            st.session_state.indexed_source = uploaded.name
            st.session_state.messages = []
            status.update(label=f"{uploaded.name} ready", state="complete")
        except SystemExit:
            status.update(label="Could not process this PDF", state="error")
            st.error(
                "This PDF could not be processed. Scanned or image-only PDFs "
                "are not supported."
            )
            st.stop()

if not st.session_state.indexed_source:
    st.markdown(
        '<div class="empty-hint">Upload a text-based PDF from the sidebar to begin.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

asked = sum(1 for m in st.session_state.messages if m["role"] == "user")
st.markdown(
    f'<div class="doc-row"><span class="doc-dot"></span>'
    f"<span>{st.session_state.indexed_source}</span>"
    f'<span class="doc-meta">{asked} question{"" if asked == 1 else "s"}</span></div>',
    unsafe_allow_html=True,
)

# ---------- Conversation ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            render_sources(msg["chunks"])

if not st.session_state.messages:
    st.markdown(
        '<div class="empty-hint">Ask your first question below.</div>',
        unsafe_allow_html=True,
    )

question = st.chat_input("Ask a question about this document")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    genai_client, collection = get_clients()
    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            chunks = query_collection(genai_client, collection, question, top_k=3)
            answer = strip_citations(generate_answer(question, chunks))
        st.markdown(answer)
        if chunks:
            render_sources(chunks)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "chunks": chunks}
    )
