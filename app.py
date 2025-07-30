# Main entry point for the application
from dotenv import load_dotenv
from utils.embedding import get_embedding_model
from utils.vectorstore import create_or_load_vector_store

load_dotenv()

embeddings = get_embedding_model()
file_path = "./data/nke-10k-2023.pdf"
vector_store = create_or_load_vector_store(embeddings, file_path)

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])
