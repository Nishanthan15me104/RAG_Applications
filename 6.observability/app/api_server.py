
from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import trace_id, retrieve_docs, generate_answer
from rag_evaluator import evaluate_rag
from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("✅ Loaded GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))


app = FastAPI()
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="deepseek-r1-distill-llama-70b")
retriever = FAISS.load_local(
    "faiss_index",
    HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'),
    allow_dangerous_deserialization=True
).as_retriever()

class RAGRequest(BaseModel):
    query: str

@app.post("/rag")
def run_rag(request: RAGRequest):
    trace = trace_id()
    query = request.query

    docs = retrieve_docs(query, retriever, trace)
    answer = generate_answer(query, docs, llm, trace)

    samples = [{
        "question": query,
        "ground_truth": "LangChain is a framework for building LLM applications with composability.",
        "contexts": [d.page_content for d in docs],
        "answer": answer
    }]
    eval_result = evaluate_rag(samples, trace)

    return {
        "trace_id": trace,
        "answer": answer,
        "evaluation": eval_result
    }
