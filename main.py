from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import traceback
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load vector store ──────────────────────────────────────────
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="medical_rag",
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("✅ Vectorstore loaded successfully")
except Exception as e:
    traceback.print_exc()
    print(f"❌ Vectorstore failed to load: {e}")
    retriever = None

# ── LLM ───────────────────────────────────────────────────────
try:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    print("✅ LLM loaded successfully")
except Exception as e:
    traceback.print_exc()
    print(f"❌ LLM failed to load: {e}")
    llm = None

# ── Prompt ────────────────────────────────────────────────────
prompt = PromptTemplate.from_template("""
You are a helpful health awareness assistant.
Use the following context to answer the question.
If you don't know the answer from the context, say "I don't have enough information on this topic."
Keep the answer clear and simple.

Context:
{context}

Question: {question}

Answer:
""")

# ── RAG chain ─────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

try:
    if llm is not None and retriever is not None:
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("✅ RAG chain loaded successfully")
    else:
        rag_chain = None
        print("❌ RAG chain skipped — LLM or retriever is None")
except Exception as e:
    traceback.print_exc()
    print(f"❌ RAG chain failed to load: {e}")
    rag_chain = None

# ── Request model ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    text: str

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Medical RAG chatbot is running"}

@app.post("/chat")
def chat(request: QueryRequest):
    if rag_chain is None:
        return {"error": "RAG chain failed to initialize. Check startup logs."}
    try:
        # Get answer
        answer = rag_chain.invoke(request.text)

        # Get source documents separately
        docs = retriever.invoke(request.text)
        sources = list(set([
            doc.metadata.get("source", "unknown") for doc in docs
        ]))

        return {
            "question": request.text,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}