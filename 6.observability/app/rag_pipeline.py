
import uuid, time
from comet_setup import logger, log_metric

def trace_id():
    return str(uuid.uuid4())

def retrieve_docs(query, retriever, trace):
    start = time.time()
    docs = retriever.get_relevant_documents(query)
    latency = time.time() - start
    logger.info(f"[{trace}] Retrieved {len(docs)} docs in {latency:.2f}s")
    log_metric("retrieval_latency", latency)
    log_metric("retrieved_docs", len(docs))
    return docs

def generate_answer(query, docs, llm, trace):
    context = "\n".join([doc.page_content for doc in docs])
    start = time.time()
    answer = llm.invoke(f"Answer the question using context:\n{context}\n\nQ: {query}")
    latency = time.time() - start
    logger.info(f"[{trace}] Generated answer in {latency:.2f}s")
    log_metric("generation_latency", latency)
    return answer
