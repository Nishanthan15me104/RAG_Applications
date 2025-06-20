
# RAG Observability with HuggingFace + Groq + Comet

## 🚀 How to Run

### 1. Build & Start Containers

```
docker-compose up --build
```

- API will be live at: http://localhost:8500/rag
- Streamlit UI at: http://localhost:8501

### 2. Configure API Keys

Update the following in your `.env` or `docker-compose.yml`:
- `COMET_API_KEY`
- `GROQ_API_KEY`

## 📦 Components

- `rag-api`: FastAPI backend with HuggingFace embeddings and Groq model
- `streamlit-ui`: Frontend for querying and visualizing RAG evaluation metrics

## 🧪 Observability

All retrieval/generation metrics logged to [Comet ML](https://www.comet.com/).
