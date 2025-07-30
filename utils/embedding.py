import os


def get_embedding_model():
    """
    Returns the embedding model based on the environment variable EMBEDDING_PROVIDER.
    Defaults to 'openai' if not set.
    """
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

    if embedding_provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    elif embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
