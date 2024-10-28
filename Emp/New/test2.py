# Install dependencies if not already installed
# Uncomment these lines if you haven't installed the libraries
# !pip install haystack sentence-transformers faiss-cpu gradio PyMuPDF

import os
import fitz  # PyMuPDF for PDF parsing
from sentence_transformers import SentenceTransformer
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DenseRetriever, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline
import gradio as gr

# Initialize the document store (FAISS) for embeddings
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

# Initialize the embedding model (Sentence Transformers)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Path to the folder containing PDF files - update this to your PDF folder path
pdf_folder_path = "path/to/your/pdfs"

def parse_pdf(file_path):
    """Extracts text from a PDF and returns it as a single string."""
    text = ""
    with fitz.open(file_path) as pdf:
        for page_num in range(len(pdf)):
            text += pdf[page_num].get_text()
    return text

# Parse PDFs and store text chunks
docs = []
for pdf_file in os.listdir(pdf_folder_path):
    pdf_path = os.path.join(pdf_folder_path, pdf_file)
    content = parse_pdf(pdf_path)
    
    # Split content into chunks (500 words for example)
    chunks = [content[i:i+500] for i in range(0, len(content), 500)]
    for chunk in chunks:
        docs.append({"content": chunk, "meta": {"source": pdf_file}})

# Write documents to the FAISS document store
document_store.write_documents(docs)

# Initialize the retriever and update document store with embeddings
retriever = DenseRetriever(embedding_model=embedding_model, document_store=document_store)
document_store.update_embeddings(retriever)

# Load a basic QA pipeline with a transformer-based reader (smaller model for free usage)
reader = TransformersReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

# Function for chatbot response
def chatbot(query):
    # Perform a search in the pipeline
    prediction = pipeline.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})
    answer = prediction["answers"][0].answer if prediction["answers"] else "Sorry, I couldn't find an answer to that."
    return answer

# Gradio interface for the chatbot
iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="Document-based Chatbot",
    description="Ask questions based on the PDF documents in the database."
)

# Launch the Gradio app
iface.launch()
