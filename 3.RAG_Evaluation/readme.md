# 📘 RAG Evaluation with LangChain, Groq, FAISS, and Ragas

This project demonstrates a basic **Retrieval Augmented Generation (RAG)** pipeline and evaluates its performance using the **Ragas** framework. It integrates:

* **LangChain** for orchestration
* **Groq** for fast LLM inference
* **FAISS** for vector similarity search
* **Ragas** for RAG metric evaluation

---

## 🚀 Features

* **Document Embedding**
  Uses `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2` to convert text documents into numerical vectors.

* **Vector Indexing**
  Employs `FAISS (IndexFlatIP)` for storing and efficiently searching document embeddings.

* **Retrieval**
  Fetches the most semantically similar document chunks based on a user's query.

* **Answer Generation (RAG)**
  Integrates Groq’s `gemma2-9b-it` LLM to generate answers grounded by the retrieved context.

* **Ragas Evaluation**
  Automatically assesses the RAG pipeline using LLM-based metrics:

  * `faithfulness`
  * `answer_relevancy`
  * `context_precision`
  * `context_recall`
  * `answer_correctness`

* **API Rate Limit Handling**
  Supports `max_retries`, `timeout`, and `num_workers=1` for Ragas evaluation to manage rate limits.

* **Ollama Integration (Optional)**
  Optional local evaluation with `qwen3:4b` for LLM and `BAAI/bge-small-en-v1.5` for embeddings.

---

## 🛠️ Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-eval.git
cd rag-eval
```

Or simply save the Python script as `rag_eval_script.py`.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install faiss-cpu numpy python-dotenv langchain-huggingface langchain-groq datasets ragas pandas
```

> ✅ Note: For GPU, replace `faiss-cpu` with `faiss-gpu` if CUDA is available.

### 4. Set up your Groq API key

1. Get your API key from the [Groq Console](https://console.groq.com/)
2. Create a `.env` file:

```env
GROQ_API_KEY=gsk_YOUR_ACTUAL_GROQ_API_KEY_HERE
```

---

## ▶️ Usage

Simply run:

```bash
python rag_eval_script.py
```

### The script will:

1. Initialize the embedding model and FAISS index.
2. Iterate through predefined `questions` and `ground_truths`.
3. For each question:

   * Retrieve relevant contexts via FAISS
   * Generate an answer using Groq LLM
   * Append results to a dataset
4. Evaluate the answers using **Ragas**
5. Output both:

   * The generated results as a table
   * Final RAG evaluation metrics as a Pandas DataFrame

---

## 🧠 Understanding Retrieval Issues

### Why is my model not retrieving the correct context?

Small, diverse document sets may confuse the embedding model due to:

* **Semantic nuance**: Embeddings like `MiniLM` may fail to rank truly relevant texts higher.
* **Sparse semantic space**: Few documents = fewer comparison opportunities = weak semantic matches.

### 🔍 Debug Output Example

```
Retrieval Debug for Query: 'What is the capital of France?'
FAISS Search Distances: [0.654321 0.456789]
FAISS Search Indices: [4 1]

Retrieved:
- Mike loves the color pink more than any other color.
- Jane Austen was an English novelist...
```

In this case, the expected document about **Paris** wasn't retrieved. This is due to a sparse and semantically varied corpus.

---

## ✨ Improving Retrieval Performance

* **Increase the number and coherence of documents**
* **Use better chunking** (e.g., 200–500 tokens with overlap)
* **Try different embedding models**:

  * `all-mpnet-base-v2`
  * `bge-small-en-v1.5`
* **Tune `k`**: Fewer retrieved documents (e.g., `k=1`) may reduce noise
* **Add re-ranking** with cross-encoder models (advanced)

---

## 📊 Ragas Metrics Explained

* **faithfulness**: Is the answer factually supported by the retrieved context?
* **answer\_relevancy**: Does the answer address the question directly?
* **context\_precision**: Are all retrieved docs actually relevant?
* **context\_recall**: Did the retrieval fetch everything needed to answer?
* **answer\_correctness**: How close is the answer to the ground truth?

---

## 🧪 Ollama Integration (Optional)

To evaluate locally without Groq:

### 1. Install and run [Ollama](https://ollama.com/)

```bash
ollama pull qwen3:4b
```

### 2. Install additional packages:

```bash
pip install langchain-ollama langchain-community
```

### 3. Uncomment the Ollama section in the script.

> This enables running RAG and Ragas **fully locally** (LLM + embeddings) depending on your hardware.

---
