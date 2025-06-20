
import streamlit as st
import requests

API_URL = "http://localhost:8500/rag"  # Docker internal network; use localhost:8500 if running locally

st.set_page_config(page_title="RAG Observability Dashboard", layout="wide")
st.title("🔍 RAG Observability with Groq + Comet")

from langchain_huggingface import HuggingFaceEmbeddings


hf_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

query = st.text_input("Enter your query:", "What is LangChain?")

if st.button("Run RAG Pipeline"):
    with st.spinner("Processing your query..."):
        response = requests.post(API_URL, json={"query": query})

    if response.status_code == 200:
        result = response.json()

        st.success("✅ Answer Generated")
        st.markdown(f"**Answer:** {result['answer']}")

        st.markdown("### 📊 Evaluation Metrics")
        st.metric("Faithfulness", f"{result['evaluation']['faithfulness']:.2f}")
        st.metric("Answer Relevancy", f"{result['evaluation']['answer_relevancy']:.2f}")

        st.markdown(f"**Trace ID:** `{result['trace_id']}`")
    else:
        st.error(f"❌ Failed to get response. Status: {response.status_code}")
