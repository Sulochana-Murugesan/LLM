pip install PyMuPDF requests beautifulsoup4 sentence-transformers faiss-cpu transformers


import fitz  # PyMuPDF for PDF processing
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Step 1: Data Ingestion Functions

def read_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def scrape_webpage(url):
    """Scrapes and extracts text from a webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return " ".join([p.text for p in soup.find_all("p")])

# Step 2: Embedding Generation

# Load a Hugging Face model for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(text):
    """Generates an embedding for the given text."""
    return embedder.encode(text)

# Step 3: FAISS Vector Store Setup

# Initialize FAISS index for 384-dimensional embeddings (based on MiniLM model)
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)
document_embeddings = []  # Store embeddings in a list for easy retrieval
metadata = []  # Store metadata for each document

def store_embedding(embedding, source):
    """Stores an embedding in the FAISS index with metadata."""
    document_embeddings.append(embedding)
    metadata.append({"source": source})
    index.add(np.array([embedding]))

# Step 4: Query Processing and Response Generation

def query_faiss(query_text, top_k=5):
    """Retrieves the top-k relevant documents from FAISS based on query text."""
    query_embedding = generate_embedding(query_text)
    _, indices = index.search(np.array([query_embedding]), top_k)
    results = [metadata[idx] for idx in indices[0]]
    return results

# Load a smaller model for text generation using transformers
generator = pipeline("text-generation", model="gpt2")

def generate_response(query, context_text):
    """Generates a response based on the context and query using a local LLM model."""
    prompt = f"Answer the following query based on the context:\n\nContext:\n{context_text}\n\nQuery:\n{query}\n\nAnswer:"
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]["generated_text"].split("Answer:")[1].strip()

# Step 5: Unified Function to Handle Queries

def chatbot_query(query_text):
    """Handles a user query by retrieving relevant documents and generating a response."""
    # Step 1: Retrieve relevant documents from FAISS
    matches = query_faiss(query_text)
    
    # Step 2: Compile text from matches for context
    context_text = "\n\n".join([match["source"] + ": " + match.get("text", "") for match in matches])
    
    # Step 3: Generate and return response based on the context
    response = generate_response(query_text, context_text)
    return response

# Example Document Ingestion and Storage
def ingest_and_store_document(file_path=None, url=None):
    """Processes and stores a document from either a PDF file or a URL."""
    if file_path:
        text = read_pdf(file_path)
        source = file_path
    elif url:
        text = scrape_webpage(url)
        source = url
    else:
        raise ValueError("Either file_path or url must be provided.")

    # Generate and store embedding
    embedding = generate_embedding(text)
    store_embedding(embedding, source)

# Ingesting sample documents
# Example usage (provide your own document paths or URLs):
# ingest_and_store_document(file_path="sample.pdf")
# ingest_and_store_document(url="https://example.com")

# Example User Query
user_query = "What is the main topic in the documents?"
print(chatbot_query(user_query))
