import sys
from unittest.mock import MagicMock

sys.modules.setdefault('onnxruntime', MagicMock())

import os
import time
import requests
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from session_manager import SessionManager
from utils import setup_logger

logger = setup_logger(__name__)

_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/{model}:embedContent"


# ── Pure-REST embeddings (zero gRPC, zero os._exit crashes) ───────────────────
class GeminiRESTEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001"):
        self.api_key = api_key
        self.model = model
        self._session = requests.Session()

    def _embed(self, text: str, task_type: str) -> List[float]:
        url = _EMBED_URL.format(model=self.model)
        for attempt in range(1, 4):
            try:
                r = self._session.post(
                    url,
                    json={"model": self.model,
                          "content": {"parts": [{"text": text}]},
                          "taskType": task_type},
                    params={"key": self.api_key},
                    timeout=30,
                )
                r.raise_for_status()
                return r.json()["embedding"]["values"]
            except Exception as e:
                logger.warning(f"Embed attempt {attempt}/3 failed: {e}")
                if attempt == 3:
                    raise RuntimeError(f"Embedding failed after 3 attempts: {e}") from e
                time.sleep(2 ** attempt)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = []
        for i, t in enumerate(texts):
            logger.info(f"  Embedding {i+1}/{len(texts)}...")
            result.append(self._embed(t, "RETRIEVAL_DOCUMENT"))
        return result

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, "RETRIEVAL_QUERY")


# ── Simple in-memory chat history ─────────────────────────────────────────────
class ChatHistory:
    def __init__(self):
        self.messages = []

    def add_user(self, text: str):
        self.messages.append(HumanMessage(content=text))

    def add_ai(self, text: str):
        self.messages.append(AIMessage(content=text))


# ── Main RAG chain ─────────────────────────────────────────────────────────────
class RAGChain:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self.session_manager = SessionManager()

        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash"),
            google_api_key=self.api_key,
            temperature=0.3,
        )

        self.embeddings = GeminiRESTEmbeddings(
            api_key=self.api_key,
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        )

        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self._histories: dict[str, ChatHistory] = {}

    # ── Vector store helpers ───────────────────────────────────────────────────
    def _faiss_path(self, session_id: str) -> str:
        base = self.session_manager.get_session_vector_store_path(session_id)
        os.makedirs(base, exist_ok=True)
        return base

    def _load_or_create_faiss(self, session_id: str, documents: List[Document] | None = None):
        path = self._faiss_path(session_id)
        index_file = os.path.join(path, "index.faiss")

        if os.path.exists(index_file):
            logger.info(f"Loading existing FAISS index for session {session_id}")
            vs = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            if documents:
                logger.info(f"Appending {len(documents)} new chunks to existing index")
                vs.add_documents(documents)
                vs.save_local(path)
        elif documents:
            logger.info(f"Creating new FAISS index with {len(documents)} chunks")
            vs = FAISS.from_documents(documents, self.embeddings)
            vs.save_local(path)
        else:
            raise ValueError(f"No FAISS index found and no documents provided for session {session_id}")

        return vs

    # ── Public API ─────────────────────────────────────────────────────────────
    def build(self, documents: List[Document], session_id: str):
        if not documents:
            raise ValueError("No documents provided")

        # Safety truncation — Gemini embedding limit is ~2048 tokens
        for doc in documents:
            if len(doc.page_content) > 6000:
                doc.page_content = doc.page_content[:6000]

        logger.info(f"Building FAISS index for session {session_id} ({len(documents)} chunks)...")
        self.vectorstore = self._load_or_create_faiss(session_id, documents)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self._build_chain()
        logger.info(f"RAG chain ready for session {session_id}")

    def load(self, session_id: str):
        path = self._faiss_path(session_id)
        if not os.path.exists(os.path.join(path, "index.faiss")):
            raise ValueError(f"No FAISS index found for session {session_id}")
        self.vectorstore = self._load_or_create_faiss(session_id)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self._build_chain()
        logger.info(f"Loaded RAG chain for session {session_id}")

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional assistant for answering questions about uploaded documents.\n\n"
             "Use the following retrieved context to answer the user's question.\n\n"
             "Retrieved context:\n{context}\n\n"
             "If the answer is not in the context, say so — do not make things up.\n"
             "Cite source document and page number when relevant."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        def fmt(docs):
            return "\n\n".join(
                f"Source: {d.metadata.get('source','?')}\n"
                f"Page: {d.metadata.get('page','?')}\n"
                f"Content: {d.page_content}"
                for d in docs
            )

        self.rag_chain = (
            {
                "context": (lambda x: x["input"]) | self.retriever | RunnableLambda(fmt),
                "chat_history": lambda x: x["chat_history"],
                "input": lambda x: x["input"],
            }
            | prompt
            | self.llm
        )

    def invoke(self, user_input: str, session_id: str) -> str:
        if not self.rag_chain:
            raise ValueError("RAG chain not built yet")

        hist = self._get_history(session_id)
        try:
            resp = self.rag_chain.invoke({"chat_history": hist.messages, "input": user_input})
            answer = resp.content if hasattr(resp, "content") else str(resp)
            hist.add_user(user_input)
            hist.add_ai(answer)
            return answer
        except Exception as e:
            logger.error(f"invoke error: {e}")
            return f"Error: {e}"

    def generate_title(self, question: str) -> str:
        fallback = (question.strip()[:47] + "...") if len(question.strip()) > 50 else question.strip()
        try:
            resp = self.llm.invoke(
                f"Summarize this question in 6 words or fewer, no quotes or period:\n\n{question}"
            )
            title = resp.content.strip().split("\n")[0].strip().strip('"\'')
            return title[:60] or fallback
        except Exception as e:
            logger.error(f"Title generation failed: {e}")
            return fallback

    def _get_history(self, session_id: str) -> ChatHistory:
        if session_id not in self._histories:
            h = ChatHistory()
            for msg in self.session_manager.get_session_messages(session_id):
                if msg["role"] == "user":
                    h.add_user(msg["content"])
                elif msg["role"] == "assistant":
                    h.add_ai(msg["content"])
            self._histories[session_id] = h
        return self._histories[session_id]