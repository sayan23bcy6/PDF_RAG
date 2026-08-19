from typing import List, Dict, Any
from rag_chain import RAGChain
from session_manager import SessionManager
from utils import setup_logger

logger = setup_logger(__name__)

class DocumentAnalytics:
    def __init__(self):
        self.session_manager = SessionManager()
    
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        session = self.session_manager.get_session(session_id)
        
        if not session:
            return {}
        
        messages = session.get('messages', [])
        uploaded_files = session.get('uploaded_files', [])
        
        user_messages = [m for m in messages if m['role'] == 'user']
        assistant_messages = [m for m in messages if m['role'] == 'assistant']
        
        stats = {
            'session_id': session_id,
            'total_messages': len(messages),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'uploaded_pdfs': len(uploaded_files),
            'pdf_names': uploaded_files,
            'created_at': session.get('created_at'),
            'updated_at': session.get('updated_at'),
            'average_response_length': self._calculate_avg_response_length(assistant_messages)
        }
        
        return stats
    
    def _calculate_avg_response_length(self, messages: List[Dict]) -> float:
        if not messages:
            return 0.0
        
        total_length = sum(len(m['content']) for m in messages)
        return total_length / len(messages)

class AdvancedSearch:
    def __init__(self, rag_chain: RAGChain):
        self.rag_chain = rag_chain
    
    def semantic_search(self, query: str, session_id: str, num_results: int = 5) -> List[Dict[str, Any]]:
        if not self.rag_chain.retriever:
            return []
        
        docs = self.rag_chain.retriever.invoke(query)
        
        results = []
        for i, doc in enumerate(docs[:num_results]):
            results.append({
                'rank': i + 1,
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'content': doc.page_content,
                'relevance_score': doc.metadata.get('relevance_score', 'N/A')
            })
        
        return results
    
    def multi_query_search(self, queries: List[str], session_id: str) -> Dict[str, List]:
        results = {}
        
        for query in queries:
            results[query] = self.semantic_search(query, session_id)
        
        return results

class ExportManager:
    def __init__(self):
        self.session_manager = SessionManager()
    
    def export_session_as_json(self, session_id: str) -> Dict[str, Any]:
        session = self.session_manager.get_session(session_id)
        
        if not session:
            return {}
        
        return {
            'session_id': session_id,
            'metadata': {
                'created_at': session.get('created_at'),
                'updated_at': session.get('updated_at'),
                'uploaded_files': session.get('uploaded_files', [])
            },
            'conversation': session.get('messages', [])
        }
    
    def export_session_as_markdown(self, session_id: str) -> str:
        session = self.session_manager.get_session(session_id)
        
        if not session:
            return ""
        
        markdown = f"# Session {session_id}\n\n"
        markdown += f"Created: {session.get('created_at')}\n"
        markdown += f"Updated: {session.get('updated_at')}\n"
        markdown += f"PDF Files: {', '.join(session.get('uploaded_files', []))}\n\n"
        markdown += "## Conversation\n\n"
        
        for message in session.get('messages', []):
            role = "User" if message['role'] == 'user' else "Assistant"
            markdown += f"**{role}:** {message['content']}\n\n"
        
        return markdown
