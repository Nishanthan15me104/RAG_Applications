<<<<<<< HEAD
     
# Multi-Modal RAG with LangChain (PDF Q&A)

## Overview

This project implements a Multi-Modal Retrieval Augmented Generation (RAG) pipeline designed to answer questions by extracting and utilizing information from both text and image content within PDF documents. It leverages LangChain for orchestration, Unstructured.io for document parsing, HuggingFace embeddings for vectorization, ChromaDB for vector storage, and Groq's powerful language models for summarization and response generation. The pipeline allows the system to effectively understand and respond to queries that require understanding across different modalities (text, tables, and images).

## Tools Used

This project utilizes a suite of open-source and commercial tools to build the multi-modal RAG system:

* **Python:** The core programming language.
* **LangChain:** A framework for developing applications powered by language models. It's used for chaining components, managing prompts, and orchestrating the RAG pipeline.
* **Unstructured.io:** An open-source library used for robust PDF document partitioning. It intelligently extracts various elements like raw text, structured tables (with HTML representation), and images (in Base64 format) from complex PDF layouts, especially in "hi_res" mode.
    * **Dependencies for Unstructured.io (Linux/macOS):** `poppler-utils`, `tesseract-ocr`, `libmagic-dev`. (For Windows, `python-poppler` and `tesseract` binaries, plus `detectron2` setup, are critical).
* **HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`):** Used to convert text summaries into numerical vector representations, enabling semantic search and similarity calculations.
* **ChromaDB:** A lightweight, in-memory (or persistent) vector database used to store the embeddings of the document summaries and facilitate efficient retrieval.
* **Groq (`meta-llama/llama-4-scout-17b-16e-instruct`, `meta-llama/llama-4-maverick-17b-128e-instruct`):** High-performance language models hosted on the Groq API.
    * `meta-llama/llama-4-scout-17b-16e-instruct`: Used for concise summarization of text and table content.
    * `meta-llama/llama-4-maverick-17b-128e-instruct`: Used for the final answer generation in the RAG pipeline.
* **`python-dotenv`:** For securely loading API keys and other environment variables from a `.env` file.
* **`more-itertools`:** Specifically the `chunked` function, used for processing data in batches to manage API rate limits and improve efficiency.
* **`IPython.display` (for Jupyter/Colab environments):** Used to display Base64-encoded images directly within the notebook for verification.

## Process Followed

The multi-modal RAG pipeline is built through a series of sequential steps:

### 1. Environment Setup & API Keys

Before running the pipeline, essential environment variables are set up. This includes:

* `GROQ_API_KEY`: Your API key for accessing Groq's language models.
* `LANGCHAIN_API_KEY`: Your API key for LangSmith tracing (optional, but highly recommended for debugging).
* `LANGCHAIN_TRACING_V2`: Set to `true` to enable detailed tracing in LangSmith.
* `GOOGLE_API_KEY`: (Although commented out for image summarization, it's present in the setup for potential use with Google Generative AI models).

These keys are typically loaded from a `.env` file using `python-dotenv` for security and convenience.

### 2. Data Extraction from PDF

The core of the multi-modal extraction relies on `unstructured.partition_pdf`.

* **Input:** A PDF document (e.g., `attention.pdf`).
* **Method:** `partition_pdf` is called with `strategy="hi_res"` and `infer_table_structure=True` to enable advanced layout analysis.
* **Element Types:**
    * **Text:** Extracted as `CompositeElement`s, `NarrativeText`, `Title`, etc.
    * **Tables:** Identified and extracted as `Table` elements, with their structure often available in `metadata.text_as_html`.
    * **Images:** Detected and extracted as `Image` elements. `extract_image_block_to_payload=True` ensures their Base64 representation is included in the metadata, making them directly usable by multi-modal LLMs.
* **Separation:** After partitioning, the `chunks` are iterated through, and individual elements are categorized into separate lists: `texts`, `tables`, and `images_b64`.

### 3. Data Summarization

To create a concise representation for efficient retrieval, each extracted element (text, table, image) is summarized using an LLM.

* **Text and Table Summaries:**
    * A `ChatGroq` model (`meta-llama/llama-4-scout-17b-16e-instruct`) is used due to its speed and efficiency for text-based summarization.
    * A `ChatPromptTemplate` is defined to guide the model to provide concise, direct summaries without extra commentary.
    * Summarization is performed in **batches of 10 elements** (`chunked` from `more_itertools`) with a `time.sleep(2)` pause between batches to respect API rate limits. Tables are summarized using their HTML representation (`table.metadata.text_as_html`) to provide more context to the LLM.
* **Image Summaries:**
    * A separate model is needed because Groq's current models are text-only. The provided code snippet shows an attempt to use `GoogleGenerativeAI` or `ChatGoogleGenerativeAI` for image summarization, but then falls back to `ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")` for the image summarization chain.
    * **Note on Image Summarization Model:** For true multi-modal understanding, an image-capable model like `gpt-4o-mini` (from OpenAI) or `gemini-pro-vision` (from Google) would be necessary for accurate image descriptions. Using a text-only Groq model here will only process the `prompt_template`'s text, not the actual image data. The `image_url` part of the message will be ignored by `ChatGroq`.

### 4. Embedding & Knowledge Base Creation

The summaries are converted into numerical embeddings and stored in a vector database to enable semantic search.

* **Embedding Model:** `HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')` is used to generate dense vector representations of the summaries.
* **Vector Store:** `ChromaDB` is initialized as the vector store (`collection_name="multi_modal_rag"`).
* **Parent Document Storage:** An `InMemoryStore` is used as a `docstore` to hold the *original, full content* of the text, table, and image elements. This is crucial for retrieving the original context after a relevant summary is found.
* **MultiVectorRetriever:** This LangChain component orchestrates the RAG strategy:
    1.  It stores the **summaries** in the `vectorstore` (ChromaDB) for retrieval.
    2.  It stores the **original full documents/images** in the `docstore` (InMemoryStore), linked by a unique `doc_id`.
    3.  When a query comes in, it searches the `vectorstore` (using the embedded summaries), gets the `doc_id`s of the most relevant summaries, and then uses those `doc_id`s to fetch the *original, full content* from the `docstore`.

### 5. Retrieval & RAG Pipeline

The final RAG pipeline combines retrieval with a Generative AI model to answer user questions.

* **Retrieval:** The `MultiVectorRetriever` takes the user's question, embeds it, searches the summaries, and returns the full original text, table, and image content (or their Base64 representations) from the `docstore`.
* **Context Parsing (`parse_docs`):** A custom `RunnableLambda` function `parse_docs` separates the retrieved documents into two lists: Base64-encoded `images` and plain `texts` (which could include table text). This differentiation is critical for multi-modal LLMs.
* **Prompt Building (`build_prompt`):** Another custom `RunnableLambda` function `build_prompt` dynamically constructs a multi-modal prompt for the LLM.
    * It combines the retrieved text context into a single string.
    * It appends `{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}` messages for each retrieved image. This is the format required by multi-modal models like `gpt-4o-mini` to "see" the images.

* **`StrOutputParser()`:** Converts the LLM's response into a simple string.
* **Chain Execution:** The `chain.invoke()` method runs the entire pipeline, retrieving relevant context and generating an answer.
* **`chain_with_sources`:** This variant allows you to not only get the final response but also inspect the `context` (retrieved text and images) that was sent to the LLM, which is invaluable for debugging and understanding the RAG process.

This robust setup allows your RAG system to go beyond text and leverage visual information from your documents, providing more comprehensive and accurate answers to user queries.

### 6. Sample output for the Query from the RAG system

#### code

        response = chain_with_sources.invoke(
            "explain The Transformer- model architecture in 50 words and image"
        )

        print("Response:", response['response'])

        print("\n\nContext:")
        for text in response['context']['texts']:
            print(text.text)
            print("Page number: ", text.metadata.page_number)
            print("\n" + "-"*50 + "\n")
        for image in response['context']['images']:
            display_base64_image(image)

#### Output: 
Response: The Transformer model architecture is shown in the image, which consists of an encoder and a decoder. The encoder takes input embeddings and applies self-attention mechanisms, while the decoder generates output probabilities based on the encoder's output. The architecture relies entirely on attention mechanisms, allowing for parallelization and achieving state-of-the-art translation quality. 

Here is a 50-word explanation of the Transformer model architecture with reference to the image:

The Transformer model architecture consists of an encoder (left) and decoder (right). The encoder applies self-attention to input embeddings, while the decoder generates output probabilities. The architecture uses multi-head attention, feed-forward layers, and positional encoding, as shown in the image, to achieve state-of-the-art translation quality.


Context:
Figure 1: The Transformer - model architecture.
Page number:  3

--------------------------------------------------

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for signiﬁcantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.
Page number:  2

--------------------------------------------------

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.
Page number:  2

--------------------------------------------------

![moving_Average](Transformer_model_architecture.jpeg)


=======
This folder contains list of RAG_Application / models built
>>>>>>> 9330e897944327bda490ab4c3146e41109204aa6
