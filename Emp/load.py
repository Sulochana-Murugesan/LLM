import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Initialize the Sentence Transformer model for creating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Directory containing your PDF and text documents
DATA_DIR = "./Documents"

# Function to extract text from PDF files by splitting into chunks (pages or sections)
def extract_chunks_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:  # Ensure there is text
            chunks.append(text)
    return chunks

# Function to load data from PDF and text files, splitting them into chunks
def load_documents(data_dir):
    chunks = []
    metadata = []

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)

        if file_name.endswith('.pdf'):
            # Extract chunks from PDF (each page as a chunk)
            pdf_chunks = extract_chunks_from_pdf(file_path)
            chunks.extend(pdf_chunks)
            metadata.extend([{'filename': file_name, 'chunk': i} for i in range(len(pdf_chunks))])
        
        elif file_name.endswith('.txt'):
            # Read text from .txt files and split into chunks (here by paragraphs)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                text_chunks = text.split('\n\n')  # Split by paragraphs
                chunks.extend(text_chunks)
                metadata.extend([{'filename': file_name, 'chunk': i} for i in range(len(text_chunks))])

    return chunks, metadata

# Load the documents and split them into chunks
chunks, metadata = load_documents(DATA_DIR)

# Create embeddings for each chunk
chunk_embeddings = model.encode(chunks)

# Initialize the FAISS index
embedding_dim = chunk_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Add chunk embeddings to the FAISS index
faiss_index.add(np.array(chunk_embeddings))

print(f"Indexed {len(chunks)} chunks from documents.")

# Function to query the vector store and get the most relevant chunks
def query_vector_store(query, top_k=5):
    # Convert the query to a vector
    query_embedding = model.encode([query])
    
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
