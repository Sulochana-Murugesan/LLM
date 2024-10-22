import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch 
torch.set_num_threads(1)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Directory containing your PDF and text documents
DATA_DIR = "./Documents"

# Load Hugging Face model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get embeddings from Hugging Face model
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to load and chunk documents using LangChain's RecursiveCharacterTextSplitter
def load_and_chunk_documents(data_dir, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    metadata = []

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        
        if file_name.endswith('.pdf'):
            # Extract text from PDF
            text = extract_text_from_pdf(file_path)
        elif file_name.endswith('.txt'):
            # Extract text from .txt files
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            continue  # Skip unsupported file types

        # Split the document into chunks
        file_chunks = text_splitter.split_text(text)
        chunks.extend(file_chunks)
        metadata.extend([{'filename': file_name, 'chunk': i} for i, _ in enumerate(file_chunks)])

    return chunks, metadata

# Load the documents and split them into chunks
chunks, metadata = load_and_chunk_documents(DATA_DIR)

# Create embeddings for each chunk
chunk_embeddings = get_embeddings(chunks)

# Initialize the FAISS index
embedding_dim = chunk_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Add chunk embeddings to the FAISS index
faiss_index.add(np.array(chunk_embeddings))

print(f"Indexed {len(chunks)} chunks from documents.")

# Function to query the vector store and get the most relevant chunks
def query_vector_store(query, top_k=5):
    # Convert the query to a vector
    query_embedding = get_embeddings([query])
    
    # Search for the most similar document chunks
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    
    # Retrieve the closest chunks with metadata
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        result = {
            'filename': metadata[idx]['filename'],
            'chunk_number': metadata[idx]['chunk'],
            'distance': dist,
            'content': chunks[idx][:2000]  # Show first 200 characters of the relevant chunk
        }
        results.append(result)
    
    return results

# Example query
query = "Channels to reach these audience"
results = query_vector_store(query, top_k=10)

# Print the results
for result in results:
    print(f"Document: {result['filename']}, Chunk: {result['chunk_number']}, Distance: {result['distance']}")
    print(f"Relevant Content: {result['content']}")
    print("-" * 80)
