pip install transformers==4.33.2 sentence-transformers==2.2.2 pdfminer.six==20221105 haystack==1.16.1 torch==2.0.1 gradio==3.38.0
import transformers
import sentence_transformers
import pdfminer
import haystack
import torch
import gradio

print("transformers:", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("pdfminer:", pdfminer.__version__)
print("haystack:", haystack.__version__)
print("torch:", torch.__version__)
print("gradio:", gradio.__version__)


import os
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DenseRetriever
from haystack.pipelines import FAQPipeline
from haystack.schema import Document
import gradio as gr

# Step 1: Extract text from PDFs
def pdf_to_text(pdf_path):
    return extract_text(pdf_path)

data_dir = "path/to/your/pdf_files"
text_data = [pdf_to_text(os.path.join(data_dir, f)) for f in os.listdir(data_dir) if f.endswith(".pdf")]

# Step 2: Chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

chunked_data = [chunk for text in text_data for chunk in chunk_text(text)]

# Step 3: Embed chunks
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
chunk_embeddings = embedder.encode(chunked_data, convert_to_tensor=True)

# Step 4: Set up FAISS document store and add documents
document_store = FAISSDocumentStore(embedding_dim=384)
docs = [Document(content=chunk) for chunk in chunked_data]
document_store.write_documents(docs)
retriever = DenseRetriever(document_store=document_store, embedding_model=embedder)
document_store.update_embeddings(retriever)

# Step 5: Create QA pipeline
qa_pipeline = FAQPipeline(retriever=retriever)

# Step 6: Create chatbot function
def chatbot(question):
    result = qa_pipeline.run(query=question)
    return result['answers'][0].answer if result['answers'] else "No relevant information found."

# Step 7: Deploy with Gradio
iface = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="PDF Document QA Bot")
iface.launch()
