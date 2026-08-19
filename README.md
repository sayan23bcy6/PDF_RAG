PDF RAG Chat Application

A professional Retrieval Augmented Generation (RAG) application for processing and querying multiple PDF documents with chat history persistence and session management.

Features

- Session-based management with automatic session ID generation
- Support for up to 5 PDFs per session
- Persistent chat history within sessions
- Chroma vector database with HNSW indexing for fast similarity search
- HuggingFace embeddings for document representation
- Groq LLM integration for natural language understanding
- Session recovery and management
- Document source tracking and retrieval
- Comprehensive logging system

Architecture

The application consists of the following components:

app.py
Main Streamlit application providing the user interface and session management.
Handles file uploads, session switching, and chat interaction.
Manages the overall workflow orchestration.

session_manager.py
Manages session lifecycle and persistence.
Handles session creation, deletion, and state management.
Maintains chat history and document metadata per session.
Stores session data in JSON format for easy recovery.

pdf_processor.py
Processes uploaded PDF files.
Implements document chunking with configurable chunk size and overlap.
Adds metadata to document chunks for tracking.
Handles multiple PDF processing with error handling.

rag_chain.py
Implements the RAG pipeline with LangChain.
Manages vector store creation and persistence using Chroma.
Implements retrieval-augmented generation with chat history awareness.
Provides conversation history management for multi-turn interactions.

utils.py
Centralized logging configuration.
Provides utility functions for common operations.

Setup Instructions

1. Prerequisites

- Python 3.10 or higher
- pip package manager
- Groq API key

2. Installation

Clone or download the application files to your project directory:

mkdir pdf-rag-app
cd pdf-rag-app

Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

3. Environment Configuration

Copy the EXAMPLE.env file to .env:

cp EXAMPLE.env .env

Edit .env with your configuration:

GROQ_API_KEY=your_actual_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
CHROMA_DB_PATH=./data/chroma_db
UPLOAD_DIR=./data/uploads
MAX_PDFS_PER_SESSION=5
CHUNK_SIZE=5000
CHUNK_OVERLAP=500
MODEL_NAME=mixtral-8x7b-32768
EMBEDDING_MODEL=all-MiniLM-L6-v2

4. Directory Structure Setup

The application will automatically create necessary directories:

pdf-rag-app/
├── app.py
├── session_manager.py
├── pdf_processor.py
├── rag_chain.py
├── utils.py
├── requirements.txt
├── README.md
├── .env
├── logs/
│   └── rag_application.log
└── data/
    ├── uploads/
    │   ├── sessions_metadata.json
    │   ├── session_1/
    │   ├── session_2/
    │   └── ...
    └── chroma_db/

Running the Application

Start the Streamlit server:

streamlit run app.py

The application will open in your default web browser at http://localhost:8501

Usage Guide

1. Session Management

New Session: Click the New Session button to create a fresh session with a new unique ID.
The session ID is automatically generated and displayed.
Previous sessions are retained and can be accessed from the session history list.

2. Uploading PDFs

Use the file uploader to select PDF files.
Up to 5 PDFs can be uploaded per session.
The system provides progress updates during processing.
Successfully processed PDFs are listed in the session info panel.

3. Asking Questions

Once PDFs are uploaded, type your question in the chat input box.
The system retrieves relevant information from the documents and generates answers.
Chat history is automatically maintained within the session.
All interactions are persisted and can be accessed later.

4. Session History

View previous sessions in the sidebar.
Click on a session to switch to it and continue your conversation.
The system loads all PDFs and chat history automatically.
Delete sessions that are no longer needed using the X button.

Configuration Parameters

GROQ_API_KEY
Your Groq API key for accessing the language model.
Required for the application to function.

HF_TOKEN
HuggingFace token for downloading embedding models.
Can be obtained from https://huggingface.co/settings/tokens

MAX_PDFS_PER_SESSION
Maximum number of PDFs that can be uploaded in a single session.
Default: 5

CHUNK_SIZE
Number of characters per document chunk.
Default: 5000
Larger values provide more context but use more tokens.

CHUNK_OVERLAP
Number of characters that overlap between consecutive chunks.
Default: 500
Ensures continuity between chunks.

MODEL_NAME
Groq model to use for language understanding.
Default: mixtral-8x7b-32768
Other options: llama2-70b-4096, gemma-7b-it

EMBEDDING_MODEL
HuggingFace model for generating document embeddings.
Default: all-MiniLM-L6-v2
Smaller and faster, suitable for most use cases.

Vector Store Configuration

The application uses Chroma with HNSW (Hierarchical Navigable Small World) indexing:

Similarity Search: Uses cosine similarity by default
Index Type: HNSW for O(log n) retrieval complexity
Retrieval: Returns top 5 most relevant documents for each query
Persistence: Vector stores are saved to disk per session

Logging

Application logs are stored in the logs/ directory:

logs/rag_application.log

Log levels:
- DEBUG: Detailed information for troubleshooting
- INFO: General application flow information
- ERROR: Error messages and exceptions

View logs in real-time or inspect the file for debugging.

Troubleshooting

Issue: GROQ_API_KEY not found

Solution: Ensure your .env file contains the correct API key.
Verify that python-dotenv is installed: pip install python-dotenv

Issue: Module not found errors

Solution: Reinstall dependencies: pip install -r requirements.txt

Issue: Vector store loading fails for old sessions

Solution: Sessions with corrupted vector stores may need to be deleted and recreated.
Use the Delete Session button to remove problematic sessions.

Issue: PDF processing takes too long

Solution: Reduce CHUNK_SIZE in .env if processing is slow.
Ensure your machine has sufficient memory for large PDFs.

Performance Considerations

Document Processing
Processing time depends on PDF size and complexity.
Large PDFs are automatically split into manageable chunks.
Multiple PDFs are processed sequentially with progress feedback.

Vector Store Performance
HNSW indexing provides fast similarity search.
Query performance is typically under 1 second.
Vector store is persisted to disk for instant loading.

Memory Usage
The application loads embeddings into memory.
Vector stores are persisted separately on disk.
Session data is stored as JSON files for minimal footprint.

API Limits
Groq API has rate limiting.
Monitor your API usage through the Groq dashboard.
Consider implementing rate limiting if needed for production use.

Advanced Usage

Custom Chunking Strategy

Modify the separators in pdf_processor.py:

self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=self.chunk_size,
    chunk_overlap=self.chunk_overlap,
    separators=["\n\n", "\n", " ", ""]  # Customize as needed
)

Modifying the RAG Prompt

Edit the system_prompt in rag_chain.py to customize the assistant behavior:

system_prompt = (
    "Your custom system prompt here..."
)

Changing the Language Model

Update MODEL_NAME in .env to use different Groq models:

Available models:
- mixtral-8x7b-32768 (Default, balanced)
- llama2-70b-4096 (Larger, more capable)
- gemma-7b-it (Smaller, faster)

Production Deployment

For production deployment, consider:

1. Use a production-grade database for session storage instead of JSON files
2. Implement user authentication and authorization
3. Add rate limiting and input validation
4. Use a production ASGI server instead of Streamlit's development server
5. Implement backup and recovery procedures
6. Monitor application performance and logs
7. Use SSL/TLS for secure communication

Security Considerations

Environment Variables
Never commit .env files to version control.
Use secure secret management systems in production.

File Uploads
PDFs are stored in the data/uploads directory.
Implement file size limits and validation for production use.

API Keys
Keep Groq API keys confidential.
Rotate keys periodically in production.

Session Data
Session metadata is stored in plain JSON.
Consider encrypting sensitive data in production.

Support and Troubleshooting

For issues with:

- PDF Processing: Check PDF file format and corruption
- LLM Responses: Verify Groq API key and rate limits
- Vector Store: Check disk space and file permissions
- Sessions: Review logs in logs/rag_application.log

License

This application is provided as-is for educational and commercial use.

Version

Current Version: 1.0.0

Last Updated: 2024
