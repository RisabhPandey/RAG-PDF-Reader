import os
import html
import tempfile
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------- Page Config ----------
st.set_page_config(
    page_title="RAG Chat with PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Images (remote, with gradient fallback in CSS) ----------
HERO_IMG = "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?auto=format&fit=crop&w=1600&q=70"
SIDEBAR_IMG = "https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=800&q=70"
USER_AVATAR = "https://api.dicebear.com/7.x/thumbs/svg?seed=reader&backgroundColor=a78bfa"
BOT_AVATAR = "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=ragbot&backgroundColor=60a5fa"

# ---------- Session State ----------
defaults = {
    "vectorstore": None,
    "messages": [],
    "processed": False,
    "theme": "dark",
    "doc_names": [],
    "chunk_count": 0,
    "pending_prompt": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def set_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


# ---------- Theme Tokens ----------
THEMES = {
    "dark": {
        "bg": "radial-gradient(1200px 600px at 10% -10%, rgba(124,58,237,0.25), transparent 60%),"
              "radial-gradient(900px 500px at 100% 0%, rgba(37,99,235,0.22), transparent 60%),"
              "linear-gradient(135deg, #0b0b12 0%, #12121f 50%, #0f172a 100%)",
        "text": "#e5e7eb",
        "muted": "#94a3b8",
        "card": "rgba(255,255,255,0.05)",
        "card_border": "rgba(255,255,255,0.10)",
        "sidebar": "linear-gradient(180deg, #0b0b12 0%, #151528 100%)",
        "source_bg": "rgba(0,0,0,0.35)",
        "source_text": "#cbd5e1",
        "shadow": "0 10px 30px rgba(0,0,0,0.35)",
        "title_grad": "linear-gradient(90deg, #c4b5fd, #93c5fd, #6ee7b7)",
    },
    "light": {
        "bg": "radial-gradient(1200px 600px at 10% -10%, rgba(167,139,250,0.30), transparent 60%),"
              "radial-gradient(900px 500px at 100% 0%, rgba(96,165,250,0.28), transparent 60%),"
              "linear-gradient(160deg, #fafaff 0%, #f1f5f9 100%)",
        "text": "#1e293b",
        "muted": "#64748b",
        "card": "rgba(255,255,255,0.85)",
        "card_border": "rgba(100,116,139,0.18)",
        "sidebar": "linear-gradient(180deg, #ffffff 0%, #f5f3ff 100%)",
        "source_bg": "#f1f5f9",
        "source_text": "#334155",
        "shadow": "0 10px 30px rgba(76,29,149,0.10)",
        "title_grad": "linear-gradient(90deg, #6d28d9, #2563eb, #0d9488)",
    },
}
t = THEMES[st.session_state.theme]

# ---------- Custom CSS ----------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; }}

.stApp {{ background: {t['bg']}; background-attachment: fixed; color: {t['text']}; }}

/* hide Streamlit chrome but keep the sidebar toggle usable */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* always keep the sidebar open/close toggle visible (covers old and new Streamlit versions) */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarHeader"] button {{
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    z-index: 999991;
}}
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {{
    background: linear-gradient(90deg, #7c3aed, #4f46e5) !important;
    color: #fff !important; border-radius: 10px !important;
}}

.block-container {{ padding-top: 1.5rem; max-width: 1200px; }}

/* ---------- Hero ---------- */
.hero {{
    position: relative;
    border-radius: 24px;
    overflow: hidden;
    padding: 3.2rem 2.6rem;
    margin-bottom: 1.6rem;
    background:
        linear-gradient(110deg, rgba(76,29,149,0.92) 0%, rgba(37,99,235,0.80) 55%, rgba(13,148,136,0.70) 100%),
        url('{HERO_IMG}') center/cover no-repeat,
        linear-gradient(135deg, #4c1d95, #2563eb);
    box-shadow: {t['shadow']};
    animation: fadeUp .7s ease both;
}}
.hero::after {{
    content: ""; position: absolute; right: -60px; top: -60px; width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(255,255,255,0.35), transparent 65%);
    border-radius: 50%; filter: blur(6px);
}}
.hero .badge {{
    display: inline-block; padding: .3rem .8rem; border-radius: 999px;
    background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.30);
    color: #fff; font-size: .8rem; font-weight: 600; letter-spacing: .04em;
    backdrop-filter: blur(8px); margin-bottom: .9rem;
}}
.hero h1 {{
    color: #fff; font-size: 2.8rem; font-weight: 800; line-height: 1.1; margin: 0 0 .6rem 0; padding: 0;
}}
.hero p {{ color: rgba(255,255,255,0.88); font-size: 1.1rem; max-width: 640px; margin: 0; }}

/* ---------- Cards ---------- */
.glass {{
    background: {t['card']}; border: 1px solid {t['card_border']};
    border-radius: 18px; padding: 1.2rem 1.3rem; box-shadow: {t['shadow']};
    backdrop-filter: blur(14px); height: 100%;
    transition: transform .25s ease, border-color .25s ease;
    animation: fadeUp .7s ease both;
}}
.glass:hover {{ transform: translateY(-4px); border-color: rgba(124,58,237,0.55); }}
.glass .icon {{ font-size: 1.9rem; margin-bottom: .4rem; }}
.glass h4 {{ margin: 0 0 .3rem 0; font-size: 1.05rem; font-weight: 700; color: {t['text']}; }}
.glass p {{ margin: 0; color: {t['muted']}; font-size: .92rem; }}

.stat {{ text-align: center; }}
.stat .num {{
    font-size: 2rem; font-weight: 800;
    background: {t['title_grad']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.stat .lbl {{ color: {t['muted']}; font-size: .85rem; font-weight: 500; }}

.section-header {{
    font-size: 1.4rem; font-weight: 700; color: {t['text']}; margin: 1.6rem 0 .9rem 0;
}}

/* ---------- Buttons ---------- */
.stButton > button {{
    background: linear-gradient(90deg, #7c3aed, #4f46e5) !important; color: #fff !important;
    border: none !important; border-radius: 12px !important; font-weight: 600 !important;
    padding: .55rem 1rem !important; transition: all .25s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.35) !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important; box-shadow: 0 8px 22px rgba(124,58,237,0.55) !important;
    filter: brightness(1.08);
}}

/* ---------- Uploader ---------- */
[data-testid="stFileUploader"] {{
    background: {t['card']}; border-radius: 18px; padding: 1rem;
    border: 2px dashed rgba(124,58,237,0.45); transition: border-color .25s ease;
}}
[data-testid="stFileUploader"]:hover {{ border-color: #7c3aed; }}
[data-testid="stFileUploader"] section {{ background: transparent !important; }}

/* ---------- Chat ---------- */
.stChatMessage {{
    background: {t['card']} !important; border: 1px solid {t['card_border']} !important;
    border-radius: 16px !important; padding: 1rem !important; margin-bottom: .6rem;
    box-shadow: {t['shadow']}; animation: fadeUp .4s ease both;
}}
.stChatMessage img {{ border-radius: 50%; }}
[data-testid="stChatInput"] {{ border-radius: 16px; }}
[data-testid="stChatInput"] textarea {{ color: {t['text']}; }}

.source-box {{
    background: {t['source_bg']}; border-left: 3px solid #7c3aed;
    padding: .7rem 1rem; border-radius: 0 10px 10px 0; margin: .5rem 0;
    font-size: .88rem; color: {t['source_text']};
}}
.source-box b {{ color: #a78bfa; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background: {t['sidebar']}; border-right: 1px solid {t['card_border']};
}}
.side-banner {{
    height: 120px; border-radius: 16px; margin-bottom: 1rem;
    background:
        linear-gradient(180deg, rgba(76,29,149,0.25), rgba(15,23,42,0.75)),
        url('{SIDEBAR_IMG}') center/cover no-repeat,
        linear-gradient(135deg, #4c1d95, #1d4ed8);
    display: flex; align-items: flex-end; padding: .8rem;
    color: #fff; font-weight: 700; font-size: 1.05rem;
}}
.doc-pill {{
    display: block; padding: .4rem .7rem; margin-bottom: .35rem; border-radius: 10px;
    background: {t['card']}; border: 1px solid {t['card_border']};
    font-size: .82rem; color: {t['text']}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}

@keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: none; }} }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Hero Header ----------
st.markdown(
    '<div class="hero">'
    '<span class="badge">✨ AI-POWERED · RAG</span>'
    "<h1>📚 Chat with your PDFs</h1>"
    "<p>Drop in one or more books, papers or notes and ask questions in plain English. "
    "Answers come with the exact passages they were pulled from.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------- Upload Section ----------
uploaded_files = st.file_uploader(
    "Upload your PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple PDF books at once",
)

col1, col2, col3 = st.columns([1.6, 1, 1])

with col1:
    process_btn = st.button("⚡ Process PDF(s)", use_container_width=True)

with col2:
    if st.session_state.processed:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

with col3:
    if st.session_state.processed:
        if st.button("🔄 Reset Document", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.processed = False
            st.session_state.messages = []
            st.session_state.doc_names = []
            st.session_state.chunk_count = 0
            st.rerun()

# ---------- Process PDFs ----------
if process_btn and uploaded_files:
    with st.spinner("🔄 Reading pages and building embeddings..."):
        all_chunks = []
        temp_paths = []

        try:
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                    temp_paths.append(tmp_path)

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            vectorstore = Chroma.from_documents(documents=all_chunks, embedding=embeddings)

            st.session_state.vectorstore = vectorstore
            st.session_state.processed = True
            st.session_state.messages = []
            st.session_state.doc_names = [f.name for f in uploaded_files]
            st.session_state.chunk_count = len(all_chunks)
            st.balloons()
            st.success(
                f"✅ Processed **{len(uploaded_files)}** PDF(s) → **{len(all_chunks)}** knowledge chunks created!"
            )

        except Exception as e:
            st.error(f"❌ Error processing PDF(s): {e}")
        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    os.unlink(path)

elif process_btn and not uploaded_files:
    st.warning("⚠️ Please upload at least one PDF first.")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="side-banner">📖 Your reading companion</div>', unsafe_allow_html=True)

    st.markdown("### 🎨 Theme")
    theme_label = "🌙 Switch to Dark Mode" if st.session_state.theme == "light" else "☀️ Switch to Light Mode"
    if st.button(theme_label, use_container_width=True):
        set_theme()
        st.rerun()

    if st.session_state.processed:
        st.markdown("---")
        st.markdown("### 📂 Loaded documents")
        for name in st.session_state.doc_names:
            st.markdown(f'<span class="doc-pill">📄 {html.escape(name)}</span>', unsafe_allow_html=True)
        st.success("Document(s) Ready ✓")

    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown(
        """
1. **Upload** one or more PDFs
2. Click **Process PDF(s)**
3. Ask questions in the chat
4. Expand **Sources** to see retrieved chunks
"""
    )

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown(
        """
- Ask specific questions
- Try “summarize the main idea”
- Use **Clear Chat** to start fresh
- Use **Reset Document** to upload new files
"""
    )


def render_sources(sources):
    with st.expander("📚 View Retrieved Sources"):
        for i, source in enumerate(sources, 1):
            safe = html.escape(source).replace("\n", "<br>")
            st.markdown(
                f'<div class="source-box"><b>Chunk {i}</b><br>{safe}</div>',
                unsafe_allow_html=True,
            )


# ---------- Chat Interface ----------
if st.session_state.processed and st.session_state.vectorstore is not None:

    # Stats row
    s1, s2, s3 = st.columns(3)
    stats = [
        (len(st.session_state.doc_names), "Documents"),
        (st.session_state.chunk_count, "Knowledge chunks"),
        (len([m for m in st.session_state.messages if m["role"] == "user"]), "Questions asked"),
    ]
    for col, (num, lbl) in zip((s1, s2, s3), stats):
        col.markdown(
            f'<div class="glass stat"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">💬 Chat with your document</div>', unsafe_allow_html=True)

    # Quick-start suggestions on an empty chat
    if not st.session_state.messages:
        st.caption("Try one of these to get started:")
        q1, q2, q3 = st.columns(3)
        suggestions = [
            (q1, "📝 Summarize the main idea", "Summarize the main idea of the document."),
            (q2, "🔑 List the key takeaways", "List the key takeaways from the document."),
            (q3, "❓ What topics are covered?", "What topics are covered in the document?"),
        ]
        for col, label, question in suggestions:
            if col.button(label, use_container_width=True, key=f"sugg_{label}"):
                st.session_state.pending_prompt = question
                st.rerun()

    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    prompt = st.chat_input("Ask anything about the document(s)...")
    if not prompt and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
        )
        docs = retriever.invoke(prompt)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.page_content for doc in docs]

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that answers questions based on the context provided. "
                    "If the context does not contain the answer, respond with "
                    "'I could not find the answer in the document.'",
                ),
                ("human", "{context}\n\nQuestion: {question}\nAnswer:"),
            ]
        )

        final_prompt = prompt_template.invoke({"context": context, "question": prompt})

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            with st.spinner("Thinking..."):
                try:
                    llm = ChatMistralAI(model_name="open-mistral-7b")
                    response = llm.invoke(final_prompt)
                    answer = response.content
                except Exception as e:
                    answer = f"Error calling Mistral AI: {e}"

            st.markdown(answer)
            if sources:
                render_sources(sources)

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

else:
    # ---------- Empty state: feature cards ----------
    st.info("👆 Upload one or more PDFs and click **Process PDF(s)** to begin.")

    st.markdown('<div class="section-header">Why you\'ll love it</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    features = [
        (f1, "🔍", "Smart retrieval", "MMR search pulls the most relevant and diverse passages from your documents."),
        (f2, "📎", "Cited sources", "Every answer ships with the exact chunks it was based on, so you can verify."),
        (f3, "📚", "Multi-PDF support", "Combine several books or papers and ask questions across all of them."),
    ]
    for col, icon, title, text in features:
        col.markdown(
            f'<div class="glass"><div class="icon">{icon}</div><h4>{title}</h4><p>{text}</p></div>',
            unsafe_allow_html=True,
        )