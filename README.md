# 📄 RAG Chatbot (PDF-based)

A lightweight Retrieval-Augmented Generation (RAG) chatbot built using 
SentenceTransformers, FAISS, Ollama (LLaMA3), and Streamlit.

This project allows users to upload PDF documents and ask questions 
based strictly on the document content. The model is instructed to avoid 
using external knowledge and respond only from retrieved context.

---

## 🚀 Features

- Upload up to 3 PDF documents
- Automatic text extraction from PDFs
- Configurable chunking with overlap
- Embedding generation using SentenceTransformers
- Cosine similarity search using FAISS
- Structured prompt construction
- Windowed conversation memory
- Streamlit-based interactive UI

---

## 🏗️ Architecture Overview

1. **Document Upload**
   - PDFs are uploaded via Streamlit.
   - Text is extracted page-by-page using PyPDF.

   **Limitations at this stage:**
   - Only text-based PDFs are supported.
   - Scanned (image-based) PDFs are not processed.
   - Tables and complex layouts may not be parsed accurately.

2. **Chunking**
   - Extracted text is split into fixed-size chunks with overlap.
   - Metadata (source filename, page number, chunk ID) is stored for traceability.

3. **Embedding**
   - Each chunk is converted into dense vector embeddings using `all-MiniLM-L6-v2`.
   - Embeddings are L2-normalized to enable cosine similarity search.

4. **Indexing**
   - FAISS `IndexFlatIP` is used for vector similarity search.
   - The index is rebuilt when new or modified documents are uploaded.
   - The index is stored in memory and is not persisted between sessions.

5. **Retrieval**
   - Top-k relevant chunks are retrieved based on similarity score.
   - A minimum similarity threshold is applied.

6. **Prompt Construction**
   - Retrieved context is injected into a structured system prompt.
   - Chat history is limited to the last N turns to control context size.

7. **Answer Generation**
   - Ollama (LLaMA3) generates responses using only the retrieved context.
   - If relevant context is not found, the model is instructed to respond accordingly.

---

## 🛠️ Tech Stack

- Python
- SentenceTransformers (`all-MiniLM-L6-v2`)
- FAISS (`IndexFlatIP`)
- Ollama (LLaMA3)
- PyPDF
- Streamlit
- NumPy

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/swalinimore/pdf-rag-chatbot.git
cd pdf-rag-chatbot