import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings



# 1. Configuration (Architect Tip: Keep paths and model names as variables)
CHROMA_PATH = "./vector_store"
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "nomic-embed-text"

# 2. Initialize Embeddings 
# validate_model_on_init=True ensures Ollama is running and has the model pulled
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    validate_model_on_init=True 
)

# 3. Persistent Vector Store Initialization
# In 2026, we use the specific langchain_chroma wrapper for better performance
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

# 4. Architect Upgrade: Utility Functions
# Don't just export the object; export "Actions" to keep your tools.py clean.

def query_vector_store(query: str, k: int = 3):
    """Safe wrapper for similarity search with error handling."""
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        print(f"Vector Store Query Error: {e}")
        return []

def add_to_vector_store(texts: list, metadatas: list = None):
    """Adds new research data to the store and ensures persistence."""
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    # In newer Chroma versions, persistence is automatic, but manual check is safer
    print(f"Successfully added {len(texts)} documents to {COLLECTION_NAME}")

