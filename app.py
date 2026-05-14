import streamlit as st

from src.config import LLM_MODEL, TOP_K
from src.rag import GroceryRAG

st.set_page_config(page_title="GroceryGPT", page_icon=None, layout="wide")

st.title("GroceryGPT — Ask anything about 2,000+ UK grocery products")


@st.cache_resource(show_spinner="Loading RAG pipeline ...")
def load_rag() -> GroceryRAG:
    return GroceryRAG()


with st.sidebar:
    st.header("Settings")
    st.markdown(f"**Model:** `{LLM_MODEL}`")
    top_k = st.slider("Sources to retrieve (top-k)", min_value=1, max_value=10, value=TOP_K)
    if st.button("Reset chat"):
        st.session_state.messages = []
        st.rerun()

rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    name = src["metadata"].get("product_name", "Unknown")
                    score = 1 - src["distance"]
                    st.markdown(f"- **{name}** (similarity: {score:.3f})")

if prompt := st.chat_input("Ask about a UK grocery product ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating ..."):
            result = rag.answer(prompt, k=top_k)

        answer = result["answer"]
        sources = result["sources"]

        st.markdown(answer)
        with st.expander("Sources"):
            for src in sources:
                name = src["metadata"].get("product_name", "Unknown")
                score = 1 - src["distance"]
                st.markdown(f"- **{name}** (similarity: {score:.3f})")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
