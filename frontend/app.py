"""
Streamlit Frontend for NABARD Chatbot
Interactive UI for querying NABARD Annual Reports
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
import csv
from datetime import datetime
from backend.rag_pipeline import get_response
st.set_page_config(
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .top-tile {
        position: fixed;
        top: 3.5rem;           /* below Streamlit navbar */
        left: 224px;
        right: 0;
        z-index: 999;
        background: #111;      /* dark like your theme */
        padding: 0.75rem 2rem;
        text-align: center;    /* center content */
    }
    .top-tile-title {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    /* push rest of page below fixed tile */
    .block-container {
        padding-top: 8rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-tile">
  <div class="top-tile-title">NABARD Rural Finance & MSME Insight Assistant</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    
    st.header(" Sample Questions")
    example_queries = [
        "How is NABARD supporting climate and renewable energy projects?",
        "How much refinance was provided to RRBs?",
        "What are the key MSME schemes mentioned?",
        "What is the total income from financial services?",
        "Summarize NABARD's role in rural development"
    ]
    
    for query in example_queries:
        print(f"Adding example query button: {query}")
        if st.button(query, key=query):
            st.session_state.example_query = query

    

FEEDBACK_FILE = os.path.join(project_root, "Data", "feedback.csv")


def log_feedback(question: str, answer: str, rating: str) -> None:
    """Append a single feedback record to the feedback CSV file."""
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)

    file_exists = os.path.exists(FEEDBACK_FILE) and os.path.getsize(FEEDBACK_FILE) > 0

    try:
        with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "question", "answer", "rating"])
            writer.writerow([
                datetime.utcnow().isoformat(),
                question,
                answer,
                rating,
            ])
    except PermissionError:
        st.error(
            "Could not write feedback to file because 'Data/feedback.csv' is open "
            "or read-only. Please close it (e.g., in Excel/Notepad/VS Code) and try again."
        )


if "messages" not in st.session_state:
    print("Initializing chat session state...")
    st.session_state.messages = []
    print("Initialized chat session state.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        print(f"Displaying message from {msg['role']}")
        st.markdown(msg["content"])
        

        if "sources" in msg and msg["sources"]:
            print(f"Displaying {len(msg['sources'])} source excerpts")
            with st.expander("Source Excerpts from NABARD Reports"):
                print(f"Expanding source excerpts...")
                for i, src in enumerate(msg["sources"], 1):
                    print(f"Displaying source excerpt {i}")
                    st.caption(f"**Excerpt {i}:**")
                    st.text(src)
                    st.markdown("---")

example_prompt = None
if "example_query" in st.session_state:
    print(f"Example query detected: {st.session_state.example_query}")
    example_prompt = st.session_state.example_query
    del st.session_state.example_query
else:
    print("No example query detected. Waiting for user input.")

user_prompt = st.chat_input("Ask a question about NABARD reports...")
print("User prompt received (may be None if user didn't type).")

prompt = example_prompt if example_prompt is not None else user_prompt

if prompt:
    print(f"Processing user prompt: {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})
    print(f"User prompt: {prompt}")
    
    with st.chat_message("user"):
        print("Displaying user message...")
        st.markdown(prompt)
        print("User message displayed.")

    with st.chat_message("assistant"):
        print("Displaying assistant response...")
        with st.spinner(" analyzing NABARD reports..."):
            print("Generating response from RAG pipeline...")
            try:
                answer, sources = get_response(prompt)
                st.markdown(answer)
                if sources:
                    with st.expander("Source Excerpts from NABARD Reports"):
                        print(f"Displaying {len(sources)} source excerpts")
                        for i, src in enumerate(sources, 1):
                            print(f"Displaying source excerpt {i}")
                            st.caption(f"**Relevant excerpt {i}:**")
                            st.text(src)
                            st.markdown("---")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                print(f"Error generating response: {e}")
                error_msg = f" Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })

if st.session_state.messages:
    last_assistant_idx = None
    for i in range(len(st.session_state.messages) - 1, -1, -1):
        if st.session_state.messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is not None:
        last_assistant_msg = st.session_state.messages[last_assistant_idx]
        question_text = ""
        for j in range(last_assistant_idx - 1, -1, -1):
            if st.session_state.messages[j]["role"] == "user":
                question_text = st.session_state.messages[j]["content"]
                break

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Helpful", key="fb_up_latest"):
                    log_feedback(question_text, last_assistant_msg["content"], "helpful")
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("Not helpful", key="fb_down_latest"):
                    log_feedback(question_text, last_assistant_msg["content"], "not_helpful")
                    st.info("Feedback recorded. We'll use this to improve future responses.")

if st.session_state.messages:
    print("Adding Clear Chat History button...")
        


















