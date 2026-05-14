import datetime
import json

import streamlit as st

from src.config import LLM_MODEL, LOGS_DIR, TOP_K
from src.health import check_health
from src.logger import get_logger
from src.rag import GroceryRAG

logger = get_logger(__name__)

st.set_page_config(page_title="GroceryGPT", page_icon=None, layout="wide")
st.title("GroceryGPT — Ask anything about ~1,900 UK grocery products")

_GRADE_COLOURS: dict[str, str] = {
    "a": "green",
    "b": "#6abf69",
    "c": "#f9c440",
    "d": "orange",
    "e": "red",
}
_ALL_GRADES = ["a", "b", "c", "d", "e"]


def _nutriscore_badge(grade: str) -> str:
    colour = _GRADE_COLOURS.get(grade.lower(), "grey") if grade else "grey"
    display = grade.upper() if grade else "?"
    return f'**Nutri-Score:** <span style="color:{colour}; font-weight:bold;">{display}</span>'


def _save_feedback(question: str, answer: str, grade: str) -> None:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "question": question,
            "answer": answer[:500],
            "grade": grade,
        }
        with open(LOGS_DIR / "feedback.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("Feedback saved: %s for query=%r", grade, question[:60])
    except Exception as exc:
        logger.warning("Failed to save feedback: %s", exc)


@st.cache_data(ttl=30)
def _get_health() -> dict:
    return check_health()


@st.cache_resource(show_spinner="Loading RAG pipeline ...")
def load_rag() -> GroceryRAG:
    return GroceryRAG()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    st.markdown(f"**Model:** `{LLM_MODEL}`")

    with st.expander("System status", expanded=True):
        health = _get_health()
        ollama_ok = health["ollama_online"]
        index_ok = health["index_built"]
        data_ok = health["data_ingested"]

        st.markdown(
            f"{'**Online**' if ollama_ok else '**Offline**'} — Ollama `{LLM_MODEL}`",
            help="Start with: ollama serve",
        )
        st.markdown(
            f"{'**Ready**' if index_ok else '**Not built**'} — "
            f"Index ({health['index_count']:,} products)",
            help="Build with: make index",
        )
        st.markdown(
            f"{'**Ingested**' if data_ok else '**Missing**'} — products.parquet",
            help="Fetch with: make ingest",
        )
        if st.button("Refresh status"):
            st.cache_data.clear()
            st.rerun()

    selected_grades = st.multiselect(
        "Nutri-Score filter (a = best)",
        options=_ALL_GRADES,
        default=_ALL_GRADES,
    )

    top_k = st.slider("Sources to retrieve (top-k)", min_value=1, max_value=10, value=TOP_K)

    if st.button("Reset chat"):
        st.session_state.messages = []
        st.rerun()

# ── Pre-flight checks ──────────────────────────────────────────────────────────
health = _get_health()

if not health["ollama_online"]:
    st.error(
        "Ollama is not running. Start it with `ollama serve` then click "
        "**Refresh status** in the sidebar."
    )
    st.stop()

if not health["index_built"]:
    st.warning(
        "ChromaDB index not found. Run `make index` (after `make ingest`) "
        "then click **Refresh status**."
    )
    st.stop()

# ── Load RAG ───────────────────────────────────────────────────────────────────
rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

_SUGGESTED_PROMPTS = [
    "Which biscuits are free from milk and egg allergens?",
    "What are the healthiest breakfast cereals — Nutri-Score A or B only?",
    "Tell me about Heinz products — what varieties do you have and what are they made of?",
    "I'm vegan — what snack options can you recommend?",
    "Which products contain palm oil and what categories are they from?",
]

# ── Suggested prompts (shown only on empty chat) ───────────────────────────────
if not st.session_state.messages:
    st.markdown("**Suggested prompts — click one to try it:**")
    col_a, col_b = st.columns(2)
    for idx, suggestion in enumerate(_SUGGESTED_PROMPTS):
        col = col_a if idx % 2 == 0 else col_b
        if col.button(suggestion, key=f"suggestion_{idx}", use_container_width=True):
            st.session_state.pending_prompt = suggestion
            st.rerun()
    st.divider()

# ── Chat history ───────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    name = src["metadata"].get("product_name", "Unknown")
                    grade = src["metadata"].get("nutriscore_grade", "")
                    score = 1 - src["distance"]
                    st.markdown(f"- **{name}** (similarity: {score:.3f})")
                    st.markdown(_nutriscore_badge(grade), unsafe_allow_html=True)

        if msg.get("latency_ms"):
            st.caption(f"Answered in {msg['latency_ms']:.0f} ms")

        if msg["role"] == "assistant":
            fb_key = f"fb_{i}"
            if not st.session_state.get(f"{fb_key}_saved"):
                fb = st.feedback("thumbs", key=fb_key)
                if fb is not None:
                    question = st.session_state.messages[i - 1]["content"] if i > 0 else ""
                    _save_feedback(question, msg["content"], "up" if fb == 1 else "down")
                    st.session_state[f"{fb_key}_saved"] = True
                    st.toast("Thanks for your feedback!")
                    st.rerun()
            else:
                st.caption("Feedback submitted")

# ── New query ──────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask about a UK grocery product ...")

# A clicked suggestion overrides the typed input
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    where_filter = (
        {"nutriscore_grade": {"$in": selected_grades}}
        if selected_grades and set(selected_grades) != set(_ALL_GRADES)
        else None
    )

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating ..."):
            try:
                result = rag.answer(prompt, k=top_k, where=where_filter)
            except Exception as exc:
                logger.error("RAG pipeline error: %s", exc)
                st.error(f"Something went wrong: {exc}")
                st.stop()

        answer = result["answer"]
        sources = result["sources"]
        latency_ms = result.get("latency_ms")

        st.markdown(answer)

        with st.expander("Sources"):
            for src in sources:
                name = src["metadata"].get("product_name", "Unknown")
                grade = src["metadata"].get("nutriscore_grade", "")
                score = 1 - src["distance"]
                st.markdown(f"- **{name}** (similarity: {score:.3f})")
                st.markdown(_nutriscore_badge(grade), unsafe_allow_html=True)

        if latency_ms:
            st.caption(f"Answered in {latency_ms:.0f} ms")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "latency_ms": latency_ms,
    })
