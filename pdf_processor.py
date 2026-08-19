import os
from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from session_manager import SessionManager
from utils import setup_logger

logger = setup_logger(__name__)

class SimpleTextSplitter:
    """Simple text splitter without any ML dependencies"""
    
    def __init__(self, chunk_size=5000, chunk_overlap=500, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def split_documents(self, documents):
        """Split documents into chunks"""
        chunks = []
        for doc in documents:
            doc_chunks = self._split_text(doc.page_content, doc.metadata.copy())
            chunks.extend(doc_chunks)
        return chunks
    
    def _split_text(self, text, metadata):
        """Recursively split text into chunks using separators"""
        if len(text) <= self.chunk_size:
            return [Document(page_content=text, metadata=metadata)]
        
        good_splits = []
        separator = self.separators[-1]
        
        for sep in self.separators:
            if sep in text:
                separator = sep
                break
        
        splits = text.split(separator)
        result = []
        current_chunk = ""
        
        for split in splits:
            if not split:
                continue
                
            test_text = current_chunk + separator + split if current_chunk else split
            
            if len(test_text) <= self.chunk_size:
                current_chunk = test_text
            else:
                if current_chunk:
                    result.append(Document(page_content=current_chunk, metadata=metadata))
                    current_chunk = split
                else:
                    if len(split) > self.chunk_size:
                        result.extend(self._split_text(split, metadata))
                    else:
                        result.append(Document(page_content=split, metadata=metadata))
        
        if current_chunk:
            result.append(Document(page_content=current_chunk, metadata=metadata))
        
        return result

class PDFProcessor:
    def __init__(self):
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 5000))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 500))
        self.session_manager = SessionManager()
        self.text_splitter = SimpleTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Ensure directories exist
        Path("logs").mkdir(parents=True, exist_ok=True)
        Path("data/uploads").mkdir(parents=True, exist_ok=True)
        Path("data/chroma_db").mkdir(parents=True, exist_ok=True)
    
    def process_pdf(self, uploaded_file, session_id: str) -> List[Document]:
        upload_dir = self.session_manager.get_upload_dir_for_session(session_id)
        
        temp_pdf_path = upload_dir / uploaded_file.name
        
        with open(temp_pdf_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            loader = PyPDFLoader(str(temp_pdf_path))
            documents = loader.load()
            
            for doc in documents:
                doc.metadata['source'] = uploaded_file.name
                doc.metadata['session_id'] = session_id
            
            chunks = self.text_splitter.split_documents(documents)
            
            logger.info(f"Processed {uploaded_file.name}: {len(documents)} pages, {len(chunks)} chunks")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")
            raise
    
    def process_multiple_pdfs(self, uploaded_files: List, session_id: str) -> List[Document]:
        all_documents = []
        
        for uploaded_file in uploaded_files:
            try:
                documents = self.process_pdf(uploaded_file, session_id)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                continue
        
        return all_documents