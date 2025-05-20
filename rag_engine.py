import os
import faiss
import pickle
import requests
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = "data"
EMBEDDINGS_DIR = "embeddings"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Switched to a standard HuggingFace embedding model

def embed_documents():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    docs = []
    for file in files:
        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            docs.append(f.read())
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    embeddings = embedder.embed_documents(docs)

    # Build FAISS index
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, os.path.join(EMBEDDINGS_DIR, "faiss.index"))

    # Save docs to disk for retrieval mapping
    with open(os.path.join(EMBEDDINGS_DIR, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)

def load_vectorstore():
    index_path = os.path.join(EMBEDDINGS_DIR, "faiss.index")
    docs_path = os.path.join(EMBEDDINGS_DIR, "docs.pkl")
    if not os.path.exists(index_path) or not os.path.exists(docs_path):
        raise FileNotFoundError("Please run embed_documents() first to create embeddings and index.")
    index = faiss.read_index(index_path)
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    return index, docs

def retrieve_similar_docs(query, top_k=3):
    index, docs = load_vectorstore()
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    query_embedding = embedder.embed_query(query)
    query_vector = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    return [docs[i] for i in indices[0]]

def generate_answer_ollama(context, query, model="mistral"):
    prompt = f"""You are a helpful technical assistant. Use the following context to answer the question:

Context:
{context}

Question:
{query}

Answer:"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    data = response.json()
    return data["response"].strip()
