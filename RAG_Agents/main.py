
from rag_pipeline import trace_id, retrieve_docs, generate_answer
from rag_evaluator import evaluate_rag
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=GROQ_API_KEY, model_name="deepseek-r1-distill-llama-70b")
retriever = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')).as_retriever()

query = "What is LangChain?"
trace = trace_id()

docs = retrieve_docs(query, retriever, trace)
answer = generate_answer(query, docs, llm, trace)

samples = [{
    "question": query,
    "ground_truth": "LangChain is a framework for building LLM applications with composability.",
    "contexts": [d.page_content for d in docs],
    "answer": answer
}]
evaluate_rag(samples, trace)
