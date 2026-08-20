# PDF RAG Chat Application (Gemini Edition)

A Retrieval Augmented Generation (RAG) app for chatting with multiple PDFs.
Both the chat model and the embeddings are powered entirely by **Google
Gemini** — no Groq, no HuggingFace.

## Pipeline

1. **Start** — open the app, a new chat session is created automatically.
2. **Upload PDF(s)** — drag in one or more PDFs; they're chunked and embedded
   into a per-session Chroma vector store.
3. **Ask questions** — chat with the uploaded PDFs using Gemini.
4. **Upload more PDFs** — add more PDFs to the *same* chat at any time; they
   get appended to the existing vector store (nothing is rebuilt from
   scratch).
5. **Ask again** — new questions are answered using *all* PDFs uploaded so
   far in that chat.

You can start a brand-new chat at any time from the sidebar, or switch back
to a previous one.

## Session naming

Each session's raw ID is just an internal identifier. What you actually see
in the sidebar is a short, auto-generated **summary of the first question
you asked** in that chat (similar to how ChatGPT/Claude name conversations),
e.g. a chat starting with "What were Q3 revenues in the report?" might show
up as "Q3 revenue summary". Until you've asked a first question, a chat is
labeled "New chat".

## Architecture

- **app.py** — Streamlit UI: session sidebar, PDF upload, chat interface.
- **session_manager.py** — session lifecycle, persistence (JSON), and the
  `display_name` (summary title) shown to users.
- **pdf_processor.py** — PDF loading and chunking.
- **rag_chain.py** — the RAG pipeline: `ChatGoogleGenerativeAI` for
  generation, `GoogleGenerativeAIEmbeddings` for the vector store, and a
  `generate_title()` helper used to summarize a session's first question.
- **utils.py** — logging setup.
- **advanced_features.py** — session analytics/search/export helpers.

## Setup

1. **Prerequisites**
   - Python 3.10+
   - A Gemini API key: https://aistudio.google.com/app/apikey

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**

   Copy `.env_example` to `.env` and fill in your key:

   ```bash
   cp .env_example .env
   ```

   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   CHROMA_DB_PATH=./data/chroma_db
   UPLOAD_DIR=./data/uploads
   MAX_PDFS_PER_SESSION=5
   CHUNK_SIZE=5000
   CHUNK_OVERLAP=500
   GEMINI_CHAT_MODEL=gemini-2.0-flash
   GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
   ```

4. **Run**

   ```bash
   streamlit run app.py
   ```

## Notes

- Sessions and messages persist to `data/uploads/sessions_metadata.json`.
- Each session's vector store lives under `data/uploads/<session_id>/vector_store`.
- Uploaded PDF files are saved under `data/uploads/<session_id>/uploads`.
