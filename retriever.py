import json
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Load documents from file
with open("data/documents.json", "r") as f:
    documents = json.load(f)

# Load Hugging Face embedding model
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model)
model = AutoModel.from_pretrained(embedding_model)

def get_embedding(text):
    """Convert text to an embedding vector."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Create FAISS index
dimension = model.config.hidden_size
index = faiss.IndexFlatL2(dimension)
doc_embeddings = []

# Add document embeddings to FAISS index
for doc in documents:
    embedding = get_embedding(doc["text"])
    doc_embeddings.append(embedding)
    index.add(np.array([embedding], dtype=np.float32))

def retrieve_top_document(query, k=1):
    """Retrieve the most relevant document for the given query."""
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

