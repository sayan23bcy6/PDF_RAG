import streamlit as st
import os
from dotenv import load_dotenv
from session_manager import SessionManager
from pdf_processor import PDFProcessor
from rag_chain import RAGChain
from utils import setup_logger

load_dotenv()


def ensure_directory(dir_path: str) -> bool:
    """
    Robustly create directory with comprehensive error handling for Windows.
    Returns True if successful, False otherwise.
    """
    try:
        abs_path = os.path.abspath(dir_path)

        if not os.path.exists(abs_path):
            os.makedirs(abs_path, mode=0o777, exist_ok=True)

        if not os.path.isdir(abs_path):
            print(f"Error: {abs_path} exists but is not a directory")
            return False

        return True
    except PermissionError:
        print(f"Permission denied creating {dir_path}")
        return False
    except Exception as e:
        print(f"Error creating {dir_path}: {e}")
        return False


# Create necessary directories
required_dirs = ["logs", "data", "data/uploads", "data/chroma_db"]
for dir_path in required_dirs:
    ensure_directory(dir_path)

# Setup logger
try:
    logger = setup_logger(__name__)
except Exception as e:
    print(f"Warning: Could not setup logger: {e}")
    logger = None


def initialize_app():
    """Initialize the Streamlit app state"""
    if 'session_manager' not in st.session_state:
        try:
            st.session_state.session_manager = SessionManager()
        except Exception as e:
            st.error(f"Failed to initialize SessionManager: {e}")
            st.stop()

    if 'current_session_id' not in st.session_state:
        try:
            st.session_state.current_session_id = st.session_state.session_manager.create_session()
        except Exception as e:
            st.error(f"Failed to create session: {e}")
            st.stop()

    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None

    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()


def get_or_create_rag_chain() -> RAGChain:
    """Reuse the RAGChain in session_state, creating one only if needed."""
    if st.session_state.rag_chain is None:
        st.session_state.rag_chain = RAGChain()
    return st.session_state.rag_chain


def switch_session(session_id: str, session_manager: SessionManager):
    """Switch the active session and load its vector store, if it has one."""
    st.session_state.current_session_id = session_id
    st.session_state.rag_chain = None

    session_data = session_manager.get_session(session_id)
    if session_data and session_data.get('uploaded_files'):
        try:
            rag_chain = RAGChain()
            rag_chain.load(session_id)
            st.session_state.rag_chain = rag_chain
        except Exception as e:
            if logger:
                logger.error(f"Could not load RAG chain for session {session_id}: {e}")


def main():
    st.set_page_config(
        page_title="PDF RAG Chat Application",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("PDF RAG Chat Application")
    st.write(
        "Upload PDFs, ask questions about them, then upload more PDFs and keep asking — "
        "all powered by Google Gemini."
    )

    initialize_app()

    session_manager = st.session_state.session_manager
    current_session_id = st.session_state.current_session_id

    # ============================================================
    # SIDEBAR — session switching, labeled by a summary of each
    # session's first question rather than the raw session id.
    # ============================================================
    with st.sidebar:
        st.header("Sessions")

        if st.button("+ New chat", use_container_width=True):
            new_id = session_manager.create_session()
            switch_session(new_id, session_manager)
            st.rerun()

        st.divider()
        st.subheader("Chat History")

        all_sessions = session_manager.list_sessions()

        if all_sessions:
            for session_id in all_sessions:
                session_data = session_manager.get_session(session_id)
                label = session_manager.get_display_name(session_id)
                pdf_count = len(session_data.get('uploaded_files', []))

                col1, col2 = st.columns([4, 1])
                with col1:
                    is_current = session_id == current_session_id
                    button_label = f"{'▶ ' if is_current else ''}{label} ({pdf_count} PDFs)"
                    if st.button(button_label, use_container_width=True, key=f"load_{session_id}"):
                        switch_session(session_id, session_manager)
                        st.rerun()

                with col2:
                    if st.button("✕", key=f"delete_{session_id}"):
                        session_manager.delete_session(session_id)
                        if session_id == current_session_id:
                            new_id = session_manager.create_session()
                            switch_session(new_id, session_manager)
                        st.rerun()
        else:
            st.info("No previous sessions found.")

        st.divider()
        st.subheader("Settings")
        st.caption(f"Max PDFs per session: {os.getenv('MAX_PDFS_PER_SESSION', 5)}")
        st.caption(f"Chat model: {os.getenv('GEMINI_CHAT_MODEL', 'gemini-3.6-flash')}")

    session_data = session_manager.get_session(current_session_id)

    # If the session was lost (e.g. metadata wiped after a crash + hot-reload),
    # create a fresh one so the app never crashes with AttributeError on None.
    if session_data is None:
        current_session_id = session_manager.create_session()
        st.session_state.current_session_id = current_session_id
        st.session_state.rag_chain = None
        session_data = session_manager.get_session(current_session_id)

    # ============================================================
    # STEP 1 / STEP 3: UPLOAD PDF(s) — available up-front, and
    # again at any point later to add more PDFs to the same chat.
    # ============================================================
    st.header("1. Upload PDF(s)")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files to add to this chat",
        type="pdf",
        accept_multiple_files=True,
        key=f"file_uploader_{current_session_id}_{len(session_data.get('uploaded_files', []))}"
    )

    max_pdfs = int(os.getenv('MAX_PDFS_PER_SESSION', 5))
    current_pdfs = len(session_data.get('uploaded_files', []))

    if uploaded_files:
        if current_pdfs + len(uploaded_files) > max_pdfs:
            st.error(f"Cannot upload more than {max_pdfs} PDFs per session. Current: {current_pdfs}")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            pdf_processor = st.session_state.pdf_processor
            all_documents = []

            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing file {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")

                    documents = pdf_processor.process_pdf(
                        uploaded_file,
                        current_session_id
                    )
                    all_documents.extend(documents)

                    progress_bar.progress((idx + 1) / len(uploaded_files))

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    if logger:
                        logger.error(f"Error processing PDF: {str(e)}")

            if all_documents:
                updated_files = session_data.get('uploaded_files', []) + [f.name for f in uploaded_files]
                session_manager.update_session(
                    current_session_id,
                    uploaded_files=updated_files
                )

                build_success = False
                try:
                    status_text.text("Embedding chunks with Gemini and updating the vector store...")
                    rag_chain = get_or_create_rag_chain()
                    # build() appends to the session's existing vector store
                    # when one is already present, so this works both for the
                    # very first upload and for any later "upload more" calls.
                    rag_chain.build(all_documents, current_session_id)
                    build_success = True

                except Exception as e:
                    st.error(f"Error building RAG chain: {str(e)}")
                    if logger:
                        logger.error(f"Error building RAG chain: {str(e)}")

                if build_success:
                    st.success(f"Successfully processed {len(uploaded_files)} PDF(s)")
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Uploaded PDFs", len(session_data.get('uploaded_files', [])))
    with col2:
        st.metric("Chat Messages", len(session_data.get('messages', [])))

    if session_data.get('uploaded_files'):
        with st.expander("View uploaded files"):
            for file in session_data.get('uploaded_files', []):
                st.write(f"- {file}")

    st.divider()

    # If we have PDFs but no RAG chain loaded yet (e.g. after a session
    # switch or app restart), load the existing vector store.
    if st.session_state.rag_chain is None and len(session_data.get('uploaded_files', [])) > 0:
        st.info("Loading this session's documents...")
        try:
            rag_chain = RAGChain()
            rag_chain.load(current_session_id)
            st.session_state.rag_chain = rag_chain
        except Exception as e:
            st.warning(f"Could not load existing documents: {str(e)}")

    # ============================================================
    # STEP 2 / STEP 4: ASK QUESTIONS
    # ============================================================
    st.header("2. Ask questions about your PDFs")

    if st.session_state.rag_chain:
        chat_container = st.container()

        with chat_container:
            session_messages = session_data.get('messages', [])

            for message in session_messages:
                with st.chat_message(message['role']):
                    st.write(message['content'])

        user_input = st.chat_input("Ask a question about your PDFs...")

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)

            try:
                with st.spinner("Generating response..."):
                    rag_chain = st.session_state.rag_chain

                    response = rag_chain.invoke(
                        user_input,
                        current_session_id
                    )

                    with st.chat_message("assistant"):
                        st.write(response)

                    session_manager.add_message(current_session_id, "user", user_input)
                    session_manager.add_message(current_session_id, "assistant", response)

                    # The very first question in a session becomes the
                    # session's human-facing label (a short Gemini-generated
                    # summary), shown in the sidebar instead of the raw id.
                    is_first_question = session_data.get('display_name') is None
                    if is_first_question:
                        try:
                            title = rag_chain.generate_title(user_input)
                            session_manager.set_display_name(current_session_id, title)
                        except Exception as e:
                            if logger:
                                logger.error(f"Could not generate session title: {e}")

                    st.rerun()

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                if logger:
                    logger.error(f"Error generating response: {str(e)}")
    else:
        st.info("Upload at least one PDF above to start chatting.")


if __name__ == "__main__":
    main()
