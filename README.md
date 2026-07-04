# MedRAG — Health Intelligence Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers general health questions using a curated set of medical documents (MedlinePlus articles, PubMedQA, MedQuAD). It combines local embeddings, a cloud vector database, and a fast LLM to ground its answers in real source material instead of hallucinating.

> ⚠️ **Disclaimer**: This project is for informational/educational purposes only and is **not** a substitute for professional medical advice, diagnosis, or treatment.

## How it works

1. **Ingestion** (`ingest.py`) — Loads source documents from `data/` (PDFs, PubMedQA JSON, MedQuAD XML), splits them into chunks, embeds them locally with Ollama, and stores them in a local Chroma vector store (`vectorstore/`).
2. **Migration** (`migrate_to_quadrant.py`) — Uploads the local Chroma vector store to a Qdrant Cloud collection (`medical_rag`), with batching/retry/resume support for large uploads.
3. **API** (`main.py`) — A FastAPI backend that embeds the incoming question, retrieves the top matching chunks from Qdrant, and passes them as context to a Groq-hosted LLM to generate a grounded answer, along with the source documents used.
4. **Frontend** (`Frontend/index.html`) — A single-page static UI that calls the API and displays the answer and sources.

## Tech stack

- **Backend**: FastAPI + Uvicorn
- **Orchestration**: LangChain / LangGraph
- **Embeddings**: Ollama (`nomic-embed-text`, run locally)
- **Vector store**: Qdrant Cloud (`langchain-qdrant`), with local Chroma used as a staging store during ingestion
- **LLM**: Groq (`llama-3.1-8b-instant` via `langchain-groq`)
- **Frontend**: Plain HTML/CSS/JS (no build step)

## Project structure

```
MedRag/
├── main.py                  # FastAPI app — the RAG chat API
├── ingest.py                 # Loads + chunks + embeds source docs into local Chroma
├── migrate_to_quadrant.py    # Migrates the local Chroma store to Qdrant Cloud
├── requirements.txt
├── data/                      # Source documents (PDFs, JSON, CSV) — not versioned
├── vectorstore/                # Local Chroma persistence directory — not versioned
├── Frontend/
│   └── index.html             # Static chat UI
└── .env                        # API keys / URLs (not versioned)
```

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally, with the embedding model pulled:
  ```bash
  ollama pull nomic-embed-text
  ```
- A [Qdrant Cloud](https://cloud.qdrant.io/) cluster (or a self-hosted Qdrant instance)
- A [Groq](https://console.groq.com/) API key

## Setup

1. **Clone and create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate       # Windows
   source venv/bin/activate    # macOS/Linux
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key
   QDRANT_URL=your_qdrant_cluster_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

4. **Add source documents**

   Place your source files under `data/` (PDFs, `ori_pqal.json` / `ori_pqaa.json` for PubMedQA, `medquad/` for MedQuAD XML files, `Symptom-severity.csv`).

5. **Build the local vector store**
   ```bash
   python ingest.py
   ```
   This reads everything in `data/`, chunks it, embeds it via Ollama, and writes a Chroma store to `vectorstore/`.

6. **Migrate embeddings to Qdrant Cloud**
   ```bash
   python migrate_to_quadrant.py
   ```
   This uploads the local Chroma collection to your Qdrant Cloud cluster under the `medical_rag` collection. It's safe to re-run — it resumes from where it left off.

## Running the API

```bash
uvicorn main:app --reload
```

The API starts on `http://localhost:8000`.

### Endpoints

| Method | Path    | Description                                  |
|--------|---------|-----------------------------------------------|
| GET    | `/`     | Health check                                   |
| POST   | `/chat` | Ask a question, get an answer + source list    |

**Example request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "What are the symptoms of dengue?"}'
```

**Example response:**
```json
{
  "question": "What are the symptoms of dengue?",
  "answer": "...",
  "sources": ["Dengue_ MedlinePlus.pdf"]
}
```

## Running the frontend

`Frontend/index.html` is a static file with no build step — just open it in a browser (or serve it) while the API is running. It expects the API at `http://localhost:8000/chat` (see `API_URL` in `index.html`); update that constant if you deploy the API elsewhere.

Note: Ollama embeddings run locally, so if you deploy the API to a host without Ollama installed, embedding generation for incoming queries will fail. Ensure the deployment environment either runs Ollama or the embedding step is swapped for a hosted embedding provider.

## Notes

- `data/`, `vectorstore/`, and `.env` are gitignored — you'll need to regenerate/provide them locally.
- The vector dimension is fixed at `768` (matching `nomic-embed-text`) in `migrate_to_quadrant.py`'s Qdrant collection config — change this if you switch embedding models.
