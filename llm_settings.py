LLM_MODEL_ID = "meta/meta-llama-3-70b-instruct"


def format_llm_model_label(model_id: str) -> str:
    """Return the sidebar label for the configured language model."""
    return f"LLM model: {model_id}"
