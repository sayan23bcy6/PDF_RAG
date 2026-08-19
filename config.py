import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    HF_TOKEN = os.getenv('HF_TOKEN')
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './data/chroma_db')
    UPLOAD_DIR = os.getenv('UPLOAD_DIR', './data/uploads')
    MAX_PDFS_PER_SESSION = int(os.getenv('MAX_PDFS_PER_SESSION', 5))
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 5000))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 500))
    MODEL_NAME = os.getenv('MODEL_NAME', 'mixtral-8x7b-32768')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    
    RETRIEVER_CONFIG = {
        'search_type': 'similarity',
        'search_kwargs': {
            'k': 5
        }
    }
    
    LLM_CONFIG = {
        'temperature': 0.3,
        'max_tokens': None,
        'top_p': 0.9
    }
    
    VECTOR_STORE_CONFIG = {
        'collection_metadata': {
            'hnsw:space': 'cosine'
        }
    }
    
    @classmethod
    def validate(cls):
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment variables")
        
        return True
