import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Gemini is now the single provider for both chat and embeddings
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './data/chroma_db')
    UPLOAD_DIR = os.getenv('UPLOAD_DIR', './data/uploads')
    MAX_PDFS_PER_SESSION = int(os.getenv('MAX_PDFS_PER_SESSION', 5))
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 5000))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 500))

    GEMINI_CHAT_MODEL = os.getenv('GEMINI_CHAT_MODEL', 'gemini-2.0-flash')
    GEMINI_EMBEDDING_MODEL = os.getenv('GEMINI_EMBEDDING_MODEL', 'models/gemini-embedding-001')

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
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment variables")

        return True
