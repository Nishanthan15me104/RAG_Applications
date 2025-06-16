# 🧠 Chat Memory Management Prototype with LangChain & Groq

This project demonstrates how to manage **short-term** and **long-term memory** in a conversational system using **LangChain**, **Groq LLM**, and **FAISS**.  
⚠️ It is **not a complete chatbot**—this is a memory management prototype meant for internal logic experimentation.

---

## 🔧 Setup Components

- **LLM**: [`gemma2-9b-it`](https://groq.com) via `ChatGroq`.
- **Embeddings**: HuggingFace's `all-MiniLM-L6-v2`.
- **Environment Config**: Uses `python-dotenv` to securely load `GROQ_API_KEY`.

---

## 🧾 Long-Term Memory (FAISS)

- Stores persistent facts about the user using FAISS vector store.
- Used for retrieving background knowledge across different sessions.
- Example entries:
  - `"User's name is Nishanthan and prefers analogies."`
  - `"User has experience with RAG-based systems."`

You can store new interactions like:
```python
vectorstore.add_documents([
    Document(page_content=f"User: {user_input}"),
    Document(page_content=f"Assistant: {response}")
])

## 💬 Short-Term Memory (ConversationSummaryBufferMemory)

Short-term memory is powered by LangChain’s `ConversationSummaryBufferMemory`.

- 🧠 **Automatically condenses** old messages when they exceed the token limit (e.g., 1000 tokens).
- 🔄 **Keeps recent context** alive for fluid conversation.
- ✨ **LLM-generated summaries** help retain coherence in long interactions.

```python
short_term_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True
)
### 🧪 Conversation Simulation
User interactions are simulated using either a list of inputs or a function call:

```python
chat_with_bot(user_input, store_to_long_term=False)
```

This function:
- ✅ Generates a response using the `ConversationChain`.
- ✅ Optionally stores the user-input and LLM response to **long-term memory** using FAISS.
- ✅ Displays updated **short-term memory buffer** (summarized + recent).

---

## 🧠 Optional Features (Commented for Now)

### 🔍 Long-Term Memory Retrieval
Retrieve user-specific context from the FAISS store:

```python
retrieved_docs = vectorstore.similarity_search("What do you know about the user?")
```

### 📚 Conversation Summarization
Manually summarize the short-term conversation memory:

```python
summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = summary_chain.run([
    Document(page_content=msg.content) for msg in short_term_memory.chat_memory.messages
])
```

---

## 🧵 Follow-up Interactions
Once the conversation starts, follow-up questions like:

```python
"Can you give me tips for solving ML case studies?"
"What should I focus on when answering business case questions in interviews?"
```

These will rely **only on short-term memory**, unless explicitly added to long-term memory by:

```python
chat_with_bot(user_input, store_to_long_term=True)
```
