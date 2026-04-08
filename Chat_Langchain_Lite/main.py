from typing import List,Any, Dict
import streamlit as st
from backend.core import run_llm


def _format_sources(content_docs:List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (content_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]

st.set_page_config(page_title="Langchain Documentation helper", layout="centered")
st.title("Langchain Documentation Helper")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("session")
    if st.button("clear chat",use_container_width=True):
        st.session_state.pop("messages",None)
        st.rerun()

if "messages" in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about Langchain docs. I'll retreive context and cite sources",
            "sources": [],
        }

    ]

for msg in st.session_state.messages:
     with st.chat_message(msg["role"]):
         st.markdown(msg["content"])
         if msg.get("sources"):
             with st.expander("Sources"):
                 for s in msg["sources"]:
                     st.markdown(f"- {s}")


prompt = st.chat_input("Ask me anything about Langchain")
if prompt:
    st.session_state.messages.append({"role":"user", "content": prompt, "sources":[]})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retreiving docs and generating answer..."):
                result:Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context",[]))
            
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

        except Exception as e:
            st.error("Failed to generate a response")
            st.exception(e)

