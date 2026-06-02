import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext,
    Settings, load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

PERSIST_DIR = "storage"
SYSTEM_PROMPT = (
    "You answer ONLY using the provided context from the filings. "
    "If the answer is not in the context, say you don't know. "
    "Be precise with numbers and cite what the filing says."
)

@st.cache_resource
def load_index():
    if not os.path.exists(PERSIST_DIR):
        docs = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index

st.set_page_config(page_title="WAB Filings RAG", page_icon="🏦")
st.title("Ask Western Alliance's Filings")
st.caption("RAG over WAB's 2025 10-K")

# --- Sidebar controls ---
st.sidebar.header("Settings")
model = st.sidebar.selectbox(
    "Answering model",
    ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    help="Sonnet is sharper; Haiku is cheaper and faster. Try both on the same question.",
)
top_k = st.sidebar.slider(
    "Passages retrieved (top-k)",
    min_value=2, max_value=8, value=4,
    help="More passages = broader coverage but more noise. Fewer = sharper but may miss context.",
)

Settings.llm = Anthropic(
    model=model,
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_tokens=1024,
    system_prompt=SYSTEM_PROMPT,
)

index = load_index()
query_engine = index.as_query_engine(similarity_top_k=top_k)

question = st.text_input("Ask about WAB's financials, risk factors, or strategy:")
if question:
    with st.spinner("Searching the filings..."):
        response = query_engine.query(question)
    st.markdown("### Answer")
    st.write(str(response))
    st.caption(f"Model: {model}  |  Passages retrieved: {top_k}")
    with st.expander("Sources used to answer"):
        for i, node in enumerate(response.source_nodes, 1):
            st.markdown(f"**Source {i}** (similarity: {node.score:.2f})")
            st.write(node.node.get_content()[:600] + "...")