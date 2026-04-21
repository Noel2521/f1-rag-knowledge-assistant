# src/retriever.py

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_retriever(persist_directory: str = "chroma_db", k: int = 5, fetch_k: int = 15):
    """Load ChromaDB and return an MMR retriever."""
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k}
    )