import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the Sentence Transformer model for creating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the saved FAISS index
faiss_index = faiss.read_index('faiss_db')

# Load the metadata and chunks
metadata = np.load('metadata.npy', allow_pickle=True).tolist()
chunks = np.load('chunks.npy', allow_pickle=True).tolist()

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
