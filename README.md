# AnchorAI

AnchorAI is a Retrieval-Augmented Generation (RAG) system designed to interact with your personal knowledge base. It ingests markdown notes, indexes them into a vector database, and provides a chat interface to answer questions grounded in your data.

## Features

- **Semantic Ingestion**: intelligently chunks markdown notes based on semantic similarity using Hugging Face embeddings.
- **RAG Pipeline**: Retrieves relevant context from your notes to answer queries accurately.
- **Query Rewriting**: Automatically refines follow-up questions to be standalone and clear based on chat history.
- **Re-ranking**: Uses an LLM to re-rank retrieved documents for higher relevance before generating answers.
- **Interactive UI**: Clean and simple chat interface built with Streamlit.
- **API Backend**: Designed to work with the standard OpenAI API.

## Project Structure

- `app.py`: The main application entry point hosting the Streamlit chat interface.
- `ingestion.py`: Script to process markdown files, chunk them, and populate the Chroma vector database.
- `retrieval.py`: Handles logic for querying the vector database and re-ranking results.
- `Transformers/`: Directory where your source markdown notes should be placed.
- `chroma_db/`: Directory where the vector database is persisted.

## Prerequisites

- Python 3.8+
- An OpenAI API Key configured in your environment as `OPENAI_API_KEY`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd AnchorAI
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Prepare Your Data
Create a 'Knowledge_Base/' directory and place your markdown notes (`.md` files) inside that directory.

### 2. Ingest Data
Run the ingestion script to process your notes and creating the embeddings database.
```bash
python ingestion.py
```
This will create/update the `chroma_db` directory.

### 3. Run the Application
Start the chat interface:
```bash
python -m streamlit run app.py
```
Open the provided local URL (usually `http://localhost:8501`) in your browser to start chatting with your notes.

## Configuration

- **Model**: The default LLM model is set to `gpt-4o-mini` via the OpenAI API. You can change this in `app.py` and `retrieval.py` by modifying the `OPENAI_MODEL` environment variable.
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- **Database**: Data is persisted in `./chroma_db`.

## Tech Stack

- **LangChain**: For orchestration and vector store interactions.
- **ChromaDB**: For vector storage and retrieval.
- **Hugging Face**: For embeddings (`all-MiniLM-L6-v2`).
- **Streamlit**: For the clean, interactive web user interface.
- **OpenAI API**: For highly accurate LLM inference and query rewriting.
