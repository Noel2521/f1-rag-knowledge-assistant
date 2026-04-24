from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import shutil

def ingest_document(pdf_path:str):
    loader = PyPDFLoader(pdf_path) #load the pdf
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    #Chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)}chunks")

    #clear the old Chroma_db
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
        print("Old chroma_db cleared")

    #Embed and store
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    print(f"stored {vector_store._collection.count()} vectors")

    return vector_store

def load_vector_store():
    #Load existing ChromaDB from disk
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function = embedding_model
    )
    
    print(f"Loaded vector store with {vector_store._collection.count()} vectors")
    return vector_store




