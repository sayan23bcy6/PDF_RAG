import os
from typing import List
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from session_manager import SessionManager
from utils import setup_logger

logger = setup_logger(__name__)

class RAGChain:
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.model_name = os.getenv('MODEL_NAME', 'mixtral-8x7b-32768')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.session_manager = SessionManager()
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=self.model_name,
            temperature=0.3
        )
        
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.conversation_history = {}
    
    def build(self, documents: List[Document], session_id: str):
        if not documents:
            raise ValueError("No documents provided to build RAG chain")
        
        vector_store_path = self.session_manager.get_session_vector_store_path(session_id)
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=vector_store_path,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        self.vectorstore.persist()
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        self._create_rag_chain()
        
        logger.info(f"Built RAG chain with {len(documents)} documents for session {session_id}")
    
    def load(self, session_id: str):
        vector_store_path = self.session_manager.get_session_vector_store_path(session_id)
        
        if not vector_store_path or not os.path.exists(vector_store_path):
            raise ValueError(f"Vector store not found for session {session_id}")
        
        self.vectorstore = Chroma(
            persist_directory=vector_store_path,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        self._create_rag_chain()
        
        logger.info(f"Loaded RAG chain for session {session_id}")
    
    def _create_rag_chain(self):
        system_prompt = (
            "You are a professional assistant for answering questions about documents. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer based on the provided context, clearly state that the information is not available in the documents. "
            "Provide accurate, concise, and helpful answers using no more than three sentences when possible. "
            "Always reference the source document when relevant."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {
                "context": self.retriever | RunnableLambda(format_docs),
                "chat_history": RunnablePassthrough(),
                "input": RunnablePassthrough(),
            }
            | prompt
            | self.llm
        )
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = ChatMessageHistory()
        
        stored_messages = self.session_manager.get_session_messages(session_id)
        
        if len(self.conversation_history[session_id].messages) == 0 and stored_messages:
            from langchain_core.messages import HumanMessage, AIMessage
            
            for msg in stored_messages:
                if msg['role'] == 'user':
                    self.conversation_history[session_id].add_user_message(msg['content'])
                elif msg['role'] == 'assistant':
                    self.conversation_history[session_id].add_ai_message(msg['content'])
        
        return self.conversation_history[session_id]
    
    def invoke(self, user_input: str, session_id: str) -> str:
        if not self.rag_chain:
            raise ValueError("RAG chain not built. Please build it first.")
        
        session_history = self._get_session_history(session_id)
        
        try:
            from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
            
            chat_history = session_history.messages
            
            response = self.rag_chain.invoke({
                "chat_history": chat_history,
                "input": user_input,
            })
            
            # Extract text from response
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error invoking RAG chain: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def get_retrieval_sources(self, query: str, session_id: str) -> List[dict]:
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        
        docs = self.retriever.invoke(query)
        
        sources = []
        for doc in docs:
            sources.append({
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
            })
        
        return sources