import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# IMPORTANT: Ensure your OpenAI API key is set as an environment variable

# export OPENAI_API_KEY="your_openai_api_key_here"



# Path where the FAISS index will be saved INSIDE the app directory
FAISS_INDEX_PATH = "./app/faiss_index"

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Sample documents for the dummy index
documents = [
    Document(page_content="LangChain is a framework for developing applications powered by large language models."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="Comet ML is an MLOps platform for tracking, comparing, and optimizing models."),
    Document(page_content="Ragas helps evaluate Retrieval Augmented Generation (RAG) pipelines."),
    Document(page_content="FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+."),
    Document(page_content="Uvicorn is an ASGI server implementation for Python, used with FastAPI.")
]

print(f"Creating FAISS index with {len(documents)} dummy documents...")
vectorstore = FAISS.from_documents(documents, embeddings)

# Create the directory if it doesn't exist
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

# Save the FAISS index
vectorstore.save_local(FAISS_INDEX_PATH)
print(f"Dummy FAISS index saved to {FAISS_INDEX_PATH}")
