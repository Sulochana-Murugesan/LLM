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

# Save the FAISS index
faiss.write_index(faiss_index, 'faiss_db')

# Save metadata and chunks for later access
np.save('metadata.npy', metadata)
np.save('chunks.npy', chunks)

print(f"Indexed {len(chunks)} chunks from documents.")
