import json
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

class SessionManager:
    def __init__(self):
        upload_dir = os.getenv('UPLOAD_DIR', './data/uploads')
        self.sessions_dir = Path(upload_dir)
        
        # Ensure directory exists with robust error handling
        self._ensure_directory_exists(str(self.sessions_dir))
        
        self.sessions_meta_file = self.sessions_dir / "sessions_metadata.json"
        self._load_sessions_metadata()
    
    @staticmethod
    def _ensure_directory_exists(directory_path: str):
        """
        Robustly create directory with comprehensive error handling for Windows.
        """
        try:
            # Convert to absolute path
            abs_path = os.path.abspath(directory_path)
            
            # Use os.makedirs which is more reliable on Windows
            if not os.path.exists(abs_path):
                os.makedirs(abs_path, mode=0o777, exist_ok=True)
                
            # Verify it was created
            if not os.path.isdir(abs_path):
                raise OSError(f"Failed to create directory: {abs_path}")
                
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied creating directory {directory_path}. "
                f"Check folder permissions or close any applications using this folder. Error: {e}"
            )
        except OSError as e:
            if "already exists" not in str(e):
                raise OSError(
                    f"Error creating directory {directory_path}. "
                    f"This may be due to antivirus software, file locking, or invalid path. Error: {e}"
                )
    
    def _load_sessions_metadata(self):
        """Load existing sessions metadata or create empty dict"""
        try:
            if self.sessions_meta_file.exists():
                with open(self.sessions_meta_file, 'r') as f:
                    self.sessions_metadata = json.load(f)
            else:
                self.sessions_metadata = {}
                self._save_sessions_metadata()
        except json.JSONDecodeError:
            print(f"Warning: Corrupted metadata file, creating new one")
            self.sessions_metadata = {}
            self._save_sessions_metadata()
        except Exception as e:
            print(f"Warning: Could not load sessions metadata: {e}. Creating new one.")
            self.sessions_metadata = {}
    
    def _save_sessions_metadata(self):
        """Save sessions metadata to JSON file with robust error handling"""
        try:
            # Ensure parent directory exists before writing
            parent_dir = str(self.sessions_meta_file.parent)
            self._ensure_directory_exists(parent_dir)
            
            # Write to temporary file first, then move (atomic operation)
            temp_file = str(self.sessions_meta_file) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(self.sessions_metadata, f, indent=2)
            
            # Move temp file to final location
            if os.path.exists(str(self.sessions_meta_file)):
                os.remove(str(self.sessions_meta_file))
            os.rename(temp_file, str(self.sessions_meta_file))
            
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied writing to {self.sessions_meta_file}. "
                f"Check file/folder permissions. Error: {e}"
            )
        except Exception as e:
            raise Exception(f"Error saving sessions metadata: {e}")
    
    def create_session(self) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())[:12]
        
        session_data = {
            'session_id': session_id,
            # display_name is the human-facing label for this session. It
            # starts as None and gets set to a short summary of the user's
            # first question (see set_display_name). Internally we always
            # keep using session_id as the real key/identifier.
            'display_name': None,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'uploaded_files': [],
            'messages': [],
            'vector_store_path': str(self.sessions_dir / session_id / 'vector_store')
        }
        
        self.sessions_metadata[session_id] = session_data
        self._save_sessions_metadata()
        
        # Create session directory
        session_dir = str(self.sessions_dir / session_id)
        self._ensure_directory_exists(session_dir)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by ID"""
        return self.sessions_metadata.get(session_id)
    
    def update_session(self, session_id: str, **kwargs):
        """Update session data"""
        if session_id in self.sessions_metadata:
            self.sessions_metadata[session_id].update(kwargs)
            self.sessions_metadata[session_id]['updated_at'] = datetime.now().isoformat()
            self._save_sessions_metadata()
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the session"""
        if session_id in self.sessions_metadata:
            if 'messages' not in self.sessions_metadata[session_id]:
                self.sessions_metadata[session_id]['messages'] = []
            
            message = {
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat()
            }
            
            self.sessions_metadata[session_id]['messages'].append(message)
            self.sessions_metadata[session_id]['updated_at'] = datetime.now().isoformat()
            self._save_sessions_metadata()
    
    def get_session_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Get all messages for a session"""
        session = self.get_session(session_id)
        if session:
            return session.get('messages', [])
        return []
    
    def set_display_name(self, session_id: str, display_name: str):
        """Set the human-facing display name (summary title) for a session"""
        if session_id in self.sessions_metadata:
            self.sessions_metadata[session_id]['display_name'] = display_name
            self._save_sessions_metadata()
    
    def get_display_name(self, session_id: str) -> str:
        """
        Get the label to show the user for this session: the summary title
        if one has been generated yet, otherwise a friendly placeholder.
        The raw session_id is never shown to the user.
        """
        session = self.get_session(session_id)
        if session and session.get('display_name'):
            return session['display_name']
        return "New chat"
    
    def delete_session(self, session_id: str):
        """Delete a session and all its data"""
        if session_id in self.sessions_metadata:
            del self.sessions_metadata[session_id]
            self._save_sessions_metadata()
            
            session_dir = self.sessions_dir / session_id
            if session_dir.exists():
                try:
                    shutil.rmtree(str(session_dir))
                except Exception as e:
                    print(f"Warning: Could not delete session directory: {e}")
    
    def list_sessions(self) -> List[str]:
        """List all session IDs sorted by update time"""
        return sorted(self.sessions_metadata.keys(), 
                     key=lambda x: self.sessions_metadata[x]['updated_at'], 
                     reverse=True)
    
    def get_session_vector_store_path(self, session_id: str) -> Optional[str]:
        """Get the vector store path for a session"""
        session = self.get_session(session_id)
        if session:
            return session.get('vector_store_path')
        return None
    
    def get_upload_dir_for_session(self, session_id: str) -> Path:
        """Get the upload directory for a session"""
        upload_dir = self.sessions_dir / session_id / 'uploads'
        self._ensure_directory_exists(str(upload_dir))
        return upload_dir