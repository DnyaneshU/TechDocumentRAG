# TechDocRAG: Technical Document Retrieval-Augmented Generation

TechDocRAG is a system that leverages Retrieval-Augmented Generation (RAG) to answer questions based on local PDF documents. The system uses FAISS for vector search and Ollama for generating responses to user queries.

## Architecture

The system consists of three main components:

1. **Document Parsing**: PDF documents are parsed into text format using the `unstructured` library.
2. **Vector Embedding**: Text documents are embedded using HuggingFace embeddings and stored in a FAISS index.
3. **Question Answering**: User queries are processed by retrieving relevant documents and generating answers using Ollama's language models.

## Project Structure

- `app.py` - Main application entry point
- `rag_engine.py` - Core RAG functionality
- `parser.py` - PDF parsing functionality
- `data/` - Directory containing parsed text files
- `docs/` - Directory containing original PDF documents
- `embeddings/` - Directory for storing document embeddings and FAISS index
- `models/` - Directory for storing model files (if needed)

## Requirements

- Python 3.x
- FAISS
- HuggingFace Transformers
- LangChain
- Ollama (running locally)
- Unstructured (for PDF parsing)

## Setup

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Additional dependencies** (if not included in requirements.txt):

```bash
pip install pdfminer.six pi-heif unstructured-inference langchain-community tf-keras
```

3. **Install Ollama**:
   - Follow the instructions at [https://ollama.ai](https://ollama.ai) to install Ollama
   - Pull the Mistral model: `ollama pull mistral`

## Usage

1. **Parse documents** (if adding new PDFs):

```bash
python parser.py
```

2. **Create document embeddings**:

```bash
python -c "from rag_engine import embed_documents; embed_documents()"
```

3. **Run the QA application**:

```bash
python app.py
```

4. **Ask questions** in the interactive prompt to get answers based on your documents.

## How It Works

1. PDF documents are parsed and stored as text files in the `data/` directory.
2. Text documents are embedded using the HuggingFace `all-MiniLM-L6-v2` embedding model.
3. Embeddings are stored in a FAISS index for efficient similarity search.
4. When a question is asked, the system:
   - Embeds the question using the same embedding model
   - Searches for similar documents in the FAISS index
   - Retrieves the most relevant documents
   - Sends the question and relevant context to Ollama
   - Returns the generated answer to the user

## Customization

- Change the embedding model by modifying `EMBEDDING_MODEL_NAME` in `rag_engine.py`
- Adjust the number of retrieved documents by changing `top_k` in `retrieve_similar_docs()`
- Use a different LLM by changing the `model` parameter in `generate_answer_ollama()`

## Troubleshooting

- Ensure Ollama is running locally (`http://localhost:11434`)
- If you get embedding errors, make sure your embedding model is installed properly
- For parsing issues, ensure the proper dependencies for `unstructured` are installed
