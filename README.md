# Local RAG Assistant

This project is a 100% offline, local Retrieval-Augmented Generation (RAG) assistant that can read, understand, and answer questions based on your own documents without requiring any internet connection. With recent updates, the project has been upgraded to a "Medical Diagnostic Assistant" capable of processing large-scale datasets (e.g., Disease-Symptom databases) and providing professional medical insights based strictly on the provided data.

It serves as a highly secure AI search engine for companies, institutions, or individuals with strict data privacy concerns, ensuring that no data ever leaves the local environment.

## Features

- 100% Local Execution: None of your data is sent to the internet (OpenAI, Google, etc.). All processing happens entirely on your local machine.
- UI-Based File Management: Instead of manually placing documents in folders, you can upload documents directly through the Streamlit web interface and delete unwanted files with a single click. Supported formats include .txt, .md, .pdf, .docx, and .csv.
- Obsidian Integration (Graph View): Visualize all documents, chunks, and their relationships in a 3D Graph view using Obsidian. You can generate and open an Obsidian Vault directly from the application interface with one click.
- Dynamic RAM/VRAM Management: To preserve system memory, models are dynamically loaded and unloaded between the RAG retrieval process (Embedding) and the answer generation process (LLM). This ensures that heavy models run smoothly even on low-end systems (e.g., 8GB RAM).
- Smart Citations & Multi-Format Support: The assistant cites the exact source document it used to generate the answer. It is also highly capable of parsing complex CSV datasets using smart grouping algorithms.
- Anti-Hallucination Measures: If the answer to your question is not found in the provided documents, the system will explicitly state that it does not know, rather than making up information. (For the medical assistant use case, it is configured to logically present partial matches rather than failing silently).

## Installation

**1. Prerequisites**
- Python 3.10 or higher
- `pip` package manager
- Foundry Local SDK (for managing local models)

**2. Setup**
Create a virtual environment in the project directory and install the required libraries:
```bash
python -m venv venv
venv\Scripts\activate
pip install streamlit foundry-local-sdk pandas PyPDF2 python-docx
```

## How to Use

To launch the project with a single click, run the `Start_RAG_Assistant.bat` file in the main directory. Alternatively, you can use the following command in the terminal:

```bash
streamlit run app.py
```

### Uploading and Managing Documents
1. In the browser interface, use the "Upload Document" section in the left sidebar to transfer data from your computer to the system.
2. Uploaded documents are automatically analyzed in the background, split into chunks, converted into vectors, and saved to a local SQLite database named `knowledge_base.db`.
3. You can view all currently registered files under the "Manage Uploaded Documents" section in the sidebar and delete them from the database by clicking the "x" icon next to them.

### Visualizing with Obsidian
1. Click the "Create Obsidian Vault" button located in the left sidebar.
2. Once the process is complete, a "View in Obsidian" button will appear. By clicking this, you can explore the 3D Graph representation of your database directly within the Obsidian application.

## Project Structure
- `docs/`: A folder where documents uploaded via the UI are temporarily or permanently stored.
- `src/ingest.py`: The script responsible for reading documents (PDF, Word, CSV, TXT) and saving them to the vector database using smart algorithms (e.g., optimizing hundreds of thousands of CSV rows into unique groups).
- `src/retriever.py`: The engine that finds the closest matching documents to your query using hybrid search (FTS5 + Cosine Similarity).
- `app.py`: The modern, user-friendly main chat interface built with Streamlit.
- `knowledge_base.db`: The local database where all your data and chat history are securely stored.
