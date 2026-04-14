from fastapi import FastAPI
from pydantic import BaseModel
from src.ingest import ingest_document, load_vector_store
from src.pipeline import build_rag_chain


app = FastAPI()


qa_chain = None

class IngestRequest(BaseModel):
    pdf_path: str

class AskRequest(BaseModel):
    question:str

@app.post("/ingest")
def ingest(request:IngestRequest):
    vector_store = ingest_document(request.pdf_path)
    global qa_chain
    qa_chain = build_rag_chain(vector_store)
    return {"status": "success","message": "Document ingested!"}

@app.post("/ask")
def ask(request: AskRequest):
    result = qa_chain.invoke({"query": request.question})
    return{"answer": result['result']}