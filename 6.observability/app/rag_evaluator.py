import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from ragas.metrics import context_recall, context_precision, faithfulness, answer_relevancy # Adding other metrics
from ragas.evaluation import evaluate, EvaluationResult
from datasets import Dataset
from comet_setup import logger, log_metric # Assuming these are correctly defined

# --- RagasCompatibleChatGroq (Keep as is, needed for Ragas LLM) ---
class RagasCompatibleChatGroq(ChatGroq):
    def set_run_config(self, run_config):
        pass
# --- End of RagasCompatibleChatGroq ---


# Configure Ragas to use Groq LLM and Hugging Face Embeddings
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable not set. "
        "Please set it before running the application to enable Groq evaluation."
    )

ragas_eval_llm = RagasCompatibleChatGroq(api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")
ragas_eval_embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Assign the LLM and embeddings to all relevant metrics
context_recall.llm = ragas_eval_llm
context_recall.embeddings = ragas_eval_embeddings

context_precision.llm = ragas_eval_llm
context_precision.embeddings = ragas_eval_embeddings

faithfulness.llm = ragas_eval_llm # Added for faithfulness
answer_relevancy.llm = ragas_eval_llm # Added for answer_relevancy
answer_relevancy.embeddings = ragas_eval_embeddings # answer_relevancy also needs embeddings


# sanitize_sample function - This is crucial for preparing the data for Ragas
def sanitize_sample(sample):
    sanitized = {}
    
    # Ensure 'question' is a plain string
    question = sample.get("question")
    sanitized["question"] = question.content if hasattr(question, 'content') else str(question)

    # Ensure 'answer' is a plain string
    answer = sample.get("answer")
    sanitized["answer"] = answer.content if hasattr(answer, 'content') else str(answer)

    # Ensure 'contexts' is a list of plain strings
    contexts = sample.get("contexts", [])
    if not isinstance(contexts, list):
        contexts = [contexts] # Ensure it's a list

    sanitized_contexts = []
    for item in contexts:
        if hasattr(item, 'page_content'): # For LangChain Document
            sanitized_contexts.append(item.page_content)
        elif hasattr(item, 'content'): # For LangChain Message
            sanitized_contexts.append(item.content)
        else: # Assume it's already a string or convert to string
            sanitized_contexts.append(str(item))
    sanitized["contexts"] = sanitized_contexts

    # Ensure 'reference' (ground_truths) is a list of non-empty strings
    # This is THE MOST IMPORTANT FIELD for context_recall and context_precision
    ground_truths_input = sample.get("reference") or sample.get("ground_truths") # Accept either key

    temp_references = []
    if ground_truths_input: 
        if not isinstance(ground_truths_input, list):
            temp_references = [str(ground_truths_input)]
        else:
            temp_references = [str(item) for item in ground_truths_input if str(item).strip()] # Filter out empty/whitespace strings
    
    sanitized["reference"] = temp_references # Ragas uses 'reference' internally
    
    return sanitized


def evaluate_rag(samples, trace):
    """
    Evaluates RAG samples using Ragas metrics and logs to Comet.ml.
    Returns evaluation results (or None if no valid samples for evaluation).
    """
    try:
        cleaned_samples = []
        for sample in samples:
            cleaned_sample = sanitize_sample(sample)
            
            # CRITICAL FILTERING STEP: Only include samples that have all required data for chosen metrics
            # context_recall/precision need 'reference' (ground_truths)
            # faithfulness needs 'answer' and 'contexts'
            # answer_relevancy needs 'question' and 'answer'
            # If a sample doesn't meet the requirements for *any* metric, it's skipped for that metric.
            # To get context_recall/precision, 'reference' MUST be valid.
            if (
                cleaned_sample.get("question") and
                cleaned_sample.get("answer") and
                cleaned_sample.get("contexts") and
                cleaned_sample.get("reference") and all(cleaned_sample.get("reference")) # Check for ground_truths
            ):
                cleaned_samples.append(cleaned_sample)
            else:
                logger.warning(
                    f"[{trace}] Skipping evaluation for a sample due to missing or empty required fields. "
                    f"Question: {bool(cleaned_sample.get('question'))}, "
                    f"Answer: {bool(cleaned_sample.get('answer'))}, "
                    f"Contexts: {bool(cleaned_sample.get('contexts'))}, "
                    f"Reference (ground_truths): {bool(cleaned_sample.get('reference')) and all(cleaned_sample.get('reference'))}"
                )

        if not cleaned_samples:
            logger.warning(f"[{trace}] No valid samples to evaluate after filtering. Skipping Ragas evaluation.")
            # Return None for all metrics if no evaluation could be done
            return {
                "context_recall": None,
                "context_precision": None,
                "faithfulness": None, # Also return None for other metrics
                "answer_relevancy": None
            }

        dataset = Dataset.from_list(cleaned_samples)

        # Run Ragas evaluation with all metrics
        # NOTE: If a sample lacks info for a specific metric (e.g., empty contexts for faithfulness),
        # Ragas might still compute NaN for *that* specific metric for *that* sample,
        # but the overall run won't error out if other samples are fine.
        results: EvaluationResult = evaluate(
            dataset, 
            metrics=[context_recall, context_precision, faithfulness, answer_relevancy]
        )

        logger.info(f"[{trace}] RAG Evaluation Raw Object: {results}")

        # Extract scores, converting NaN to None for JSON compliance
        eval_scores = {}
        for metric_name in ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]:
            score = getattr(results, metric_name, float('nan'))
            score_for_json = score if (score == score) else None # Converts NaN to None
            eval_scores[metric_name] = score_for_json

            # Log to Comet.ml only if the score is not None
            if score_for_json is not None:
                log_metric(metric_name, score_for_json)
            else:
                logger.warning(f"[{trace}] Metric '{metric_name}' score is NaN/None. Not logging to Comet.ml for this trace.")

        logger.info(f"[{trace}] RAG Evaluation Scores: {eval_scores}")

        return eval_scores

    except Exception as e:
        logger.error(f"[{trace}] Error during RAG evaluation: {e}", exc_info=True)
        # Re-raise the exception to propagate it up to FastAPI
        raise