"""
migrate_to_qdrant.py - With retry logic and resume support
Run: python migrate_to_qdrant.py
"""

import time
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ── Config ─────────────────────────────────────────────────────
QDRANT_URL      = "your_cluster_url_here"
QDRANT_API_KEY  = "your_api_key_here"
COLLECTION_NAME = "medical_rag"
LOCAL_VECTORSTORE_PATH = "./vectorstore"

FETCH_BATCH  = 100   # reduced from 200 to avoid timeouts
UPLOAD_BATCH = 50    # reduced from 100 to avoid timeouts
MAX_RETRIES  = 5     # retry failed uploads this many times
RETRY_DELAY  = 5     # seconds to wait between retries

# ── Embeddings ─────────────────────────────────────────────────
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ── Step 1: Load local Chroma ──────────────────────────────────
print("⏳ Loading local Chroma vectorstore...")
local_vs = Chroma(
    persist_directory=LOCAL_VECTORSTORE_PATH,
    embedding_function=embeddings
)
total = local_vs._collection.count()
print(f"✅ Found {total} documents in local vectorstore")

if total == 0:
    print("❌ No documents found.")
    exit()

# ── Step 2: Connect to Qdrant ──────────────────────────────────
print("\n⏳ Connecting to Qdrant Cloud...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60  # increased timeout
)

if client.collection_exists(COLLECTION_NAME):
    already_uploaded = client.count(COLLECTION_NAME).count
    print(f"✅ Collection exists — {already_uploaded} docs already uploaded")
    start_offset = already_uploaded  # resume from where we left off
else:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created")
    start_offset = 0

if start_offset >= total:
    print("✅ All documents already uploaded. Nothing to do!")
    exit()

print(f"▶️  Resuming from offset {start_offset}/{total}")

# ── Step 3: Upload with retry logic ───────────────────────────
def upload_with_retry(docs, attempt=1):
    try:
        QdrantVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
            force_recreate=False
        )
        return True
    except Exception as e:
        if attempt <= MAX_RETRIES:
            print(f"    ⚠️  Upload failed (attempt {attempt}/{MAX_RETRIES}): {str(e)[:80]}")
            print(f"    ⏳ Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
            return upload_with_retry(docs, attempt + 1)
        else:
            print(f"    ❌ Failed after {MAX_RETRIES} attempts. Skipping batch.")
            return False

# ── Main upload loop ───────────────────────────────────────────
uploaded = start_offset
offset = start_offset

print(f"\n⏳ Uploading {total - start_offset} remaining docs...\n")

while offset < total:
    # Fetch from Chroma
    try:
        data = local_vs._collection.get(
            limit=FETCH_BATCH,
            offset=offset,
            include=["documents", "metadatas"]
        )
    except Exception as e:
        print(f"⚠️  Chroma fetch error at offset {offset}: {e}")
        break

    if not data["ids"]:
        break

    batch_docs = [
        Document(
            page_content=content,
            metadata=meta if meta else {}
        )
        for content, meta in zip(data["documents"], data["metadatas"])
    ]

    # Upload in sub-batches with retry
    for i in range(0, len(batch_docs), UPLOAD_BATCH):
        sub_batch = batch_docs[i:i + UPLOAD_BATCH]
        success = upload_with_retry(sub_batch)
        if success:
            uploaded += len(sub_batch)
        print(f"  ✅ Progress: {uploaded}/{total} docs ({int(uploaded/total*100)}%)")

    offset += FETCH_BATCH
    time.sleep(1)  # small pause between fetch batches

print(f"\n🎉 Migration complete!")
print(f"   Uploaded : {uploaded}/{total} documents")
print(f"   Collection: {COLLECTION_NAME}")
print(f"   Qdrant URL: {QDRANT_URL}")