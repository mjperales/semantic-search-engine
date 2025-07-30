from langchain_core.vectorstores import InMemoryVectorStore
from utils.loader import load_documents, split_documents


def create_or_load_vector_store(
    embeddings, file_path, chunk_size=1000, chunk_overlap=200
):
    """
    Creates or loads a vector store with the given embeddings and document splits.

    Args:
        embeddings: The embedding model to use.
        file_path: Path to the document file.
        chunk_size: Size of each document chunk.
        chunk_overlap: Overlap size between chunks.

    Returns:
        A vector store containing the document splits.
    """
    docs = load_documents(file_path)
    all_splits = split_documents(
        docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)

    return vector_store
