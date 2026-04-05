import json
import os
import csv
import xml.etree.ElementTree as ET
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DATA_PATH        = "./data"
JSON_PATH_L      = "./data/ori_pqal.json"
JSON_PATH_A      = "./data/ori_pqaa.json"
SYMPTOM_SEV      = "./data/Symptom-severity.csv"
MEDQUAD_PATH     = "./data/medquad"
VECTORSTORE_PATH = "./vectorstore"

def load_pdfs():
    print("Loading PDFs...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()
    print(f"  Loaded {len(docs)} pages from PDFs")
    return docs

def load_pubmedqa_json(json_path):
    if not os.path.exists(json_path):
        print(f"  {json_path} not found, skipping...")
        return []
    print(f"Loading {json_path}...")
    docs = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for pubmed_id, entry in data.items():
        question    = entry.get("question", "")
        long_answer = entry.get("long_answer", "")
        if not long_answer.strip():
            continue
        content = f"Question: {question}\nAnswer: {long_answer}"
        docs.append(Document(
            page_content=content,
            metadata={
                "source": f"PubMedQA_{pubmed_id}",
                "pubmed_id": pubmed_id,
                "final_decision": entry.get("final_decision", "")
            }
        ))
    print(f"  Loaded {len(docs)} entries from {json_path}")
    return docs

def load_symptom_severity():
    if not os.path.exists(SYMPTOM_SEV):
        print("  Symptom-severity.csv not found, skipping...")
        return []
    print("Loading Symptom-severity.csv...")
    docs = []
    with open(SYMPTOM_SEV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symptom = (row.get("Symptom") or row.get("symptom") or row.get("Disease") or "").strip()
            weight  = (row.get("weight") or row.get("Weight") or row.get("Severity") or "").strip()
            if not symptom:
                continue
            content = f"Symptom: {symptom}"
            if weight:
                content += f"\nSeverity weight: {weight}"
            docs.append(Document(
                page_content=content,
                metadata={"source": "Kaggle_SymptomSeverity", "symptom": symptom}
            ))
    print(f"  Loaded {len(docs)} symptom severity entries")
    return docs

def load_medquad():
    if not os.path.exists(MEDQUAD_PATH):
        print("  MedQuAD folder not found, skipping...")
        return []
    print("Loading MedQuAD XML files...")
    docs = []
    for root_dir, dirs, files in os.walk(MEDQUAD_PATH):
        for filename in files:
            if not filename.endswith(".xml"):
                continue
            filepath = os.path.join(root_dir, filename)
            try:
                tree = ET.parse(filepath)
                root = tree.getroot()
                for qa in root.findall(".//QAPair"):
                    question = qa.findtext("Question", "").strip()
                    answer   = qa.findtext("Answer", "").strip()
                    if not answer:
                        continue
                    content = f"Question: {question}\nAnswer: {answer}"
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": "MedQuAD", "file": filename}
                    ))
            except Exception as e:
                print(f"  Skipping {filename}: {e}")
    print(f"  Loaded {len(docs)} QA pairs from MedQuAD")
    return docs

def ingest():
    pdf_docs     = load_pdfs()
    json_docs_l  = load_pubmedqa_json(JSON_PATH_L)
    json_docs_a  = load_pubmedqa_json(JSON_PATH_A)
    symptom_docs = load_symptom_severity()
    medquad_docs = load_medquad()

    all_docs = pdf_docs + json_docs_l + json_docs_a + symptom_docs + medquad_docs
    print(f"\nTotal documents loaded: {len(all_docs)}")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks created: {len(chunks)}")

    print("\nEmbedding chunks using Ollama (this will take several minutes)...\n")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )
    print(f"\nDone! Vector store saved to {VECTORSTORE_PATH}")

if __name__ == "__main__":
    ingest()