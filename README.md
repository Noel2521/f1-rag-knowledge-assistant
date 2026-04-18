# f1-rag-knowledge-assistant

# 🏎️ F1 RAG Knowledge Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that answers 
questions about Formula 1 race data, built with LangChain, ChromaDB, 
HuggingFace Embeddings, and Ollama.

## 🧠 What is RAG?
RAG (Retrieval-Augmented Generation) combines document retrieval with 
LLM generation — instead of relying on the model's training data, it 
fetches relevant context from your own documents and generates grounded answers.

## 🏗️ Architecture
PDF Document
↓
[PyPDFLoader] → Load & extract text
↓
[RecursiveCharacterTextSplitter] → Chunk into 500-char pieces
↓
[HuggingFace all-MiniLM-L6-v2] → Convert chunks to vectors
↓
[ChromaDB] → Store & retrieve vectors
↓
[Ollama llama3.2] → Generate grounded answers
↓
[FastAPI] → Serve as REST API


## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| LangChain | RAG pipeline orchestration |
| ChromaDB | Vector database |
| HuggingFace Embeddings | Text → vector conversion |
| Ollama (llama3.2) | Local LLM, runs free |
| FastAPI | REST API framework |
| PyPDF | PDF loading |

## 📁 Project Structure
f1-rag-knowledge-assistant/
├── data/                    # F1 race report PDFs
├── src/
│   ├── ingest.py            # Load, chunk, embed, store
│   ├── pipeline.py          # LLM + retriever chain
│   └── api.py               # FastAPI endpoints
├── notebooks/
│   └── 01_ingest.ipynb      # Exploration notebook
├── requirements.txt
└── README.md


## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Noel2521/f1-rag-knowledge-assistant.git
cd f1-rag-knowledge-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama
Download Ollama from https://ollama.com and run:
```bash
ollama pull llama3.2
```

### 5. Start the API
```bash
uvicorn src.api:app --reload
```

### 6. Open the interactive docs
http://127.0.0.1:8000/docs


## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | `/health` | Check API status |
| POST | `/ingest` | Load a PDF into ChromaDB |
| POST | `/ask` | Ask a question |

### Example Usage

**Ingest a document:**
```json
POST /ingest
{
  "pdf_path": "data/F1_2024_BritishGP_RaceReport.pdf"
}
```

**Ask a question:**
```json
POST /ask
{
  "question": "Who won the 2024 British Grand Prix?"
}
```

**Response:**
```json
{
  "answer": "Lando Norris won the 2024 British Grand Prix."
}
```

## 🔑 Key Technical Decisions

- **Chunk size 500, overlap 100** — smaller chunks improve retrieval precision
- **MMR retrieval** — Maximum Marginal Relevance avoids returning duplicate 
  chunks and improves answer quality
- **fetch_k=15, k=5** — fetches 15 candidate chunks, selects best 5 diverse ones
- **all-MiniLM-L6-v2** — lightweight, fast embedding model that runs locally

## 👨‍💻 Author
**Noel Anthony** — AI/ML Engineer  
MSc Computer Science, University of South Wales  
[GitHub](https://github.com/Noel2521) | [LinkedIn](https://www.linkedin.com/in/noel-anthony/)