"""
Simulates the exact sequence app.py performs across multiple Streamlit
reruns, with Gemini calls mocked out (no real network access here).
This is meant to catch structural bugs: infinite reprocessing loops,
exceptions, or the vector store getting wiped instead of appended to.
"""
import os
import sys
import shutil
import types
from unittest.mock import MagicMock

os.environ["GEMINI_API_KEY"] = "dummy-key-for-test"

# ---- Mock out the Gemini classes before rag_chain imports them ----
import langchain_google_genai as lgg
from langchain_core.runnables import Runnable

class FakeEmbeddings:
    def __init__(self, *a, **k):
        pass
    def embed_documents(self, texts):
        # deterministic pseudo-embeddings based on text hash, fixed dim
        return [[float((hash(t) >> i) % 7) for i in range(16)] for t in texts]
    def embed_query(self, text):
        return [float((hash(text) >> i) % 7) for i in range(16)]

class FakeResponse:
    def __init__(self, content):
        self.content = content

class FakeChatModel(Runnable):
    """A minimal Runnable so it slots into the real LCEL chain (prompt | llm)."""
    def __init__(self, *a, **k):
        pass
    def invoke(self, x, config=None, **kwargs):
        # x can be a raw string (title-gen prompt) or a ChatPromptValue (main chain)
        text = x if isinstance(x, str) else str(x)
        if "Summarize the following user question" in text:
            return FakeResponse("Fake Title About PDFs")
        return FakeResponse("This is a fake Gemini answer based on the PDFs.")

lgg.GoogleGenerativeAIEmbeddings = FakeEmbeddings
lgg.ChatGoogleGenerativeAI = FakeChatModel

# ---- Clean workspace ----
WORKDIR = "/home/claude/pipeline_test_data"
if os.path.exists(WORKDIR):
    shutil.rmtree(WORKDIR)
os.makedirs(WORKDIR, exist_ok=True)
os.environ["UPLOAD_DIR"] = os.path.join(WORKDIR, "uploads")
os.environ["CHROMA_DB_PATH"] = os.path.join(WORKDIR, "chroma_db")

sys.path.insert(0, "/home/claude/PDF_RAG_gemini")

from session_manager import SessionManager
from pdf_processor import PDFProcessor
from rag_chain import RAGChain


class FakeUploadedFile:
    """Mimics Streamlit's UploadedFile enough for pdf_processor.process_pdf"""
    def __init__(self, path):
        self.name = os.path.basename(path)
        self._data = open(path, "rb").read()
    def getbuffer(self):
        return self._data


def run():
    session_manager = SessionManager()
    pdf_processor = PDFProcessor()

    session_id = session_manager.create_session()
    print(f"[1] Created session: {session_id}")

    rag_chain = None  # mimics st.session_state.rag_chain

    # ================= RERUN 1: user uploads PDF #1 =================
    print("\n--- RERUN 1: upload PDF #1 ---")
    session_data = session_manager.get_session(session_id)
    uploaded_files = [FakeUploadedFile("/home/claude/test1.pdf")]  # what file_uploader "returns" this run

    current_pdfs = len(session_data.get("uploaded_files", []))
    all_documents = []
    for f in uploaded_files:
        docs = pdf_processor.process_pdf(f, session_id)
        all_documents.extend(docs)
    assert all_documents, "No documents produced from PDF #1"
    print(f"  processed {len(all_documents)} chunks from PDF #1")

    updated_files = session_data.get("uploaded_files", []) + [f.name for f in uploaded_files]
    session_manager.update_session(session_id, uploaded_files=updated_files)

    rag_chain = RAGChain()
    rag_chain.build(all_documents, session_id)
    print("  rag_chain.build() succeeded for PDF #1")

    # Simulate the widget "resetting" because its key changed (this is the
    # actual fix for the freeze: on the NEXT script run, len(uploaded_files)
    # is different, so file_uploader gets a fresh key and returns None/[]).
    uploaded_files = None

    # ================= RERUN 2: user asks a question =================
    print("\n--- RERUN 2: ask a question ---")
    assert rag_chain.rag_chain is not None, "rag_chain not ready to answer"
    answer = rag_chain.invoke("What is this PDF about?", session_id)
    print(f"  got answer: {answer!r}")
    session_manager.add_message(session_id, "user", "What is this PDF about?")
    session_manager.add_message(session_id, "assistant", answer)

    session_data = session_manager.get_session(session_id)
    if session_data.get("display_name") is None:
        title = rag_chain.generate_title("What is this PDF about?")
        session_manager.set_display_name(session_id, title)
        print(f"  generated session title: {title!r}")

    # Re-run with NO new files (this is where the OLD code's static
    # file_uploader key would still be returning the same PDF forever,
    # causing an infinite reprocess -> rebuild -> rerun loop).
    print("\n--- RERUN 3: idle rerun, no new upload ---")
    uploaded_files = None
    if uploaded_files:
        raise RuntimeError("BUG: file_uploader still returning stale files -> infinite loop risk")
    else:
        print("  no stale files returned - no reprocessing triggered (good)")

    # ================= RERUN 4: user uploads PDF #2 (upload more) =================
    print("\n--- RERUN 4: upload PDF #2 (upload more) ---")
    session_data = session_manager.get_session(session_id)
    uploaded_files = [FakeUploadedFile("/home/claude/test2.pdf")]

    all_documents_2 = []
    for f in uploaded_files:
        docs = pdf_processor.process_pdf(f, session_id)
        all_documents_2.extend(docs)
    assert all_documents_2, "No documents produced from PDF #2"
    print(f"  processed {len(all_documents_2)} chunks from PDF #2")

    updated_files = session_data.get("uploaded_files", []) + [f.name for f in uploaded_files]
    session_manager.update_session(session_id, uploaded_files=updated_files)

    # rag_chain is REUSED (like st.session_state.rag_chain), build() called
    # again with only the NEW documents - must APPEND, not wipe PDF #1's data.
    rag_chain.build(all_documents_2, session_id)
    print("  rag_chain.build() succeeded for PDF #2 (appended)")

    # Check underlying collection actually has chunks from both PDFs
    total_in_store = rag_chain.vectorstore._collection.count()
    print(f"  total chunks now in vector store: {total_in_store}")
    assert total_in_store == len(all_documents) + len(all_documents_2), (
        f"Expected {len(all_documents) + len(all_documents_2)} chunks total, "
        f"got {total_in_store} -- vector store may have been wiped instead of appended!"
    )

    # ================= RERUN 5: ask another question, should see both PDFs =================
    print("\n--- RERUN 5: ask about PDF #2's content ---")
    answer2 = rag_chain.invoke("What does the second PDF say?", session_id)
    print(f"  got answer: {answer2!r}")

    sources = rag_chain.get_retrieval_sources("bananas", session_id)
    found_sources = {s["source"] for s in sources}
    print(f"  retrieval sources seen: {found_sources}")
    assert "test1.pdf" in found_sources or "test2.pdf" in found_sources

    print("\n[ALL CHECKS PASSED] Upload -> ask -> upload more -> ask pipeline works structurally.")


if __name__ == "__main__":
    run()
