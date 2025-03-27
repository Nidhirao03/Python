import gradio as gr
import faiss
import numpy as np
import nltk
import fitz  # PyMuPDF for PDF processing
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from sentence_transformers import SentenceTransformer

nltk.download('punkt')

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load pre-trained QA model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Global variables
chunks = []
index = None

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

# Function to process uploaded book file (PDF or TXT)
def process_file(file):
    global chunks, index  # Update global variables

    # Determine file type
    if file.name.endswith(".pdf"):
        book_text = extract_text_from_pdf(file.name)
    else:  # Assume TXT file
        with open(file.name, "r", encoding="utf-8") as f:
            book_text = f.read()

    # Split text into chunks
    sentences = sent_tokenize(book_text)
    chunk_size = 3
    chunks = [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

    # Create text embeddings
    embeddings = embed_model.encode(chunks)

    # Store embeddings in FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return "📘 Book uploaded and processed! Now, ask a question."

# Function to retrieve answer from processed book text
def retrieve_answer(query):
    global index, chunks
    if not chunks or index is None:
        return "⚠️ Please upload a book file first!"

    # Search for most relevant chunk
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    context = chunks[I[0][0]]

    # Get answer from QA model
    answer = qa_model(question=query, context=context)
    return answer["answer"]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📖 Book Q&A AI - Upload a book (TXT or PDF) and ask questions!")

    file_input = gr.File(label="Upload a `.txt` or `.pdf` file with book pages", type="filepath")
    upload_button = gr.Button("Process Book")

    question_input = gr.Textbox(label="Ask a question based on the book")
    answer_output = gr.Textbox(label="Answer")

    upload_button.click(process_file, inputs=file_input, outputs=answer_output)
    question_input.change(retrieve_answer, inputs=question_input, outputs=answer_output)

# Run Gradio app
demo.launch()
