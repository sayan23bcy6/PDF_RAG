Quick Start Guide - PDF RAG Chat Application

Get up and running in 5 minutes.

Step 1: Prerequisites Check

Ensure you have:
- Python 3.10 or later installed
- pip package manager
- Groq API key (free from https://console.groq.com)
- HuggingFace token (optional, but recommended)

Check Python version:

python --version

Step 2: Clone or Download

Place all project files in a directory:

pdf-rag-app/
├── app.py
├── session_manager.py
├── pdf_processor.py
├── rag_chain.py
├── config.py
├── utils.py
├── advanced_features.py
├── EXAMPLE.env
├── requirements.txt
├── run.sh (or run.bat for Windows)
└── [documentation files]

Step 3: Setup Environment

Copy the example environment file:

cp EXAMPLE.env .env

Edit .env and add your credentials:

GROQ_API_KEY=your_actual_groq_api_key
HF_TOKEN=your_huggingface_token

Step 4: Install Dependencies

Linux/Mac:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Windows:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Step 5: Run Application

Option A - Using startup script (Linux/Mac):

chmod +x run.sh
./run.sh

Option B - Using startup script (Windows):

run.bat

Option C - Manual startup:

streamlit run app.py

The application will open in your browser at:

http://localhost:8501

Congratulations! You're ready to use the PDF RAG Chat Application.

First Run Walkthrough

1. Upload PDF

Click on file uploader in the main area
Select a PDF file from your computer
Wait for processing to complete (shows progress bar)

2. Ask Questions

Once PDF is uploaded, type a question in the chat box
Press Enter or click Send
The AI will read your PDF and answer based on its content

3. View Chat History

All your conversations are automatically saved
Switch between sessions in the sidebar
Your chat history persists even after closing the app

Basic Features Overview

Session Management

New Session: Creates fresh session with unique ID
Previous Sessions: Listed in sidebar with PDF count
Switch Sessions: Click any session to load it
Delete Session: Remove unwanted sessions

PDF Upload

Max 5 PDFs per session
Automatic processing with progress feedback
PDF names stored in session metadata

Chat Interface

Ask any questions about uploaded PDFs
Chat history maintained within session
Sources tracked for reference
Multi-turn conversations supported

Key Keyboard Shortcuts

Ctrl+C: Stop the running application
Ctrl+R: Refresh the Streamlit app
Ctrl+M: Access Streamlit menu

Troubleshooting Quick Fixes

Issue: Python command not found

Solution:
Use python3 instead:

python3 --version
python3 -m venv venv

Issue: Module not found error

Solution:
Reinstall requirements:

pip install -r requirements.txt

Issue: GROQ_API_KEY error

Solution:
1. Verify .env file exists
2. Check GROQ_API_KEY is set
3. Ensure no extra spaces in .env
4. Restart the application

Issue: PDF upload fails

Solution:
1. Check file is valid PDF
2. File size under system limits
3. Check disk space available
4. Try different PDF file

Issue: Slow responses

Solution:
1. Check internet connection
2. Verify Groq API is responding
3. Try smaller PDF files
4. Reduce number of PDFs

Directory Structure After First Run

pdf-rag-app/
├── app.py
├── [other source files]
├── .env                    # Your configuration (DO NOT commit)
│
├── logs/
│   └── rag_application.log # Application logs
│
└── data/
    ├── uploads/
    │   ├── sessions_metadata.json  # Session index
    │   └── [session_folders]/      # Auto-created per session
    └── chroma_db/                  # Vector database

Configuration Quick Reference

Critical Settings

GROQ_API_KEY: Your API key (required)
HF_TOKEN: HuggingFace token (optional but recommended)

Tuning Parameters

MAX_PDFS_PER_SESSION: Max 5 (can increase if needed)
CHUNK_SIZE: 5000 (larger = more context, higher cost)
CHUNK_OVERLAP: 500 (overlap between chunks)

Model Selection

MODEL_NAME: Current mixtral-8x7b-32768
EMBEDDING_MODEL: Current all-MiniLM-L6-v2

Advanced Features

Access advanced features in advanced_features.py:

- DocumentAnalytics: Get session statistics
- AdvancedSearch: Perform semantic searches
- ExportManager: Export conversations

Example:

from advanced_features import DocumentAnalytics

analytics = DocumentAnalytics()
stats = analytics.get_session_statistics("session_id")
print(stats)

Common Tasks

Task: Upload multiple PDFs at once

1. Click file uploader
2. Select multiple PDFs using Ctrl+Click (Cmd+Click on Mac)
3. Upload all at once
4. Wait for processing

Task: Export conversation

Add to app.py:

from advanced_features import ExportManager

export_manager = ExportManager()
session_data = export_manager.export_session_as_json(session_id)
session_md = export_manager.export_session_as_markdown(session_id)

Task: Change embedding model

Edit .env:

EMBEDDING_MODEL=all-mpnet-base-v2

Available models:
- all-MiniLM-L6-v2 (fast, default)
- all-mpnet-base-v2 (better quality)
- all-roberta-large-v1 (highest quality)

Task: Change LLM model

Edit .env:

MODEL_NAME=llama2-70b-4096

Available models:
- mixtral-8x7b-32768 (balanced)
- llama2-70b-4096 (more capable)
- gemma-7b-it (faster)

Common Questions

Q: How much does it cost to run?
A: Free tier available. Cost depends on Groq API usage.

Q: Can I run this locally without internet?
A: No, requires Groq API and HuggingFace model downloads.

Q: How long are conversations saved?
A: Until you delete the session manually.

Q: Can I upload very large PDFs?
A: Yes, but processing takes longer. Max depends on available disk space.

Q: How many concurrent users can it handle?
A: Single instance handles 1 user. Use deployment guide for scaling.

Q: Can I deploy this to production?
A: Yes, see DEPLOYMENT.md for detailed instructions.

Q: Is my data secure?
A: Data stored locally. Use HTTPS in production. See security section in README.md.

Next Steps

1. Read README.md for comprehensive documentation
2. Check TESTING.md for testing procedures
3. Review DEPLOYMENT.md for production setup
4. Explore advanced_features.py for additional capabilities
5. Monitor logs in logs/rag_application.log

Getting Help

Check these resources in order:

1. README.md - Comprehensive guide
2. TESTING.md - Testing and troubleshooting
3. PROJECT_STRUCTURE.md - Architecture details
4. Logs - Check logs/rag_application.log for errors
5. DEPLOYMENT.md - Advanced configuration

Performance Tips

For faster performance:

1. Reduce CHUNK_SIZE to 3000
2. Reduce chunk overlap to 300
3. Use all-MiniLM-L6-v2 embedding model
4. Upload smaller PDFs
5. Ask specific questions

For better accuracy:

1. Increase CHUNK_SIZE to 7000
2. Increase chunk overlap to 700
3. Use all-mpnet-base-v2 embedding model
4. Upload higher quality PDFs
5. Ask detailed questions

Production Readiness

Before deploying to production:

Required:
- [ ] .env configured with production credentials
- [ ] Database migration completed (from JSON)
- [ ] SSL/TLS certificates configured
- [ ] Backup and recovery tested
- [ ] Monitoring and alerts setup
- [ ] Rate limiting configured
- [ ] Security audit completed
- [ ] Load testing performed

See DEPLOYMENT.md for complete checklist.

Support

For issues:

1. Check logs: tail -f logs/rag_application.log
2. Restart application: Stop and run again
3. Reinstall dependencies: pip install -r requirements.txt
4. Clear cache: rm -rf .cache (after stopping app)

Uninstall

To remove the application:

1. Deactivate virtual environment:

deactivate

2. Remove directory:

rm -rf pdf-rag-app

3. Remove virtual environment:

rm -rf venv

Data Backup

Before major updates:

1. Backup sessions:

cp -r data/uploads ~/backup_sessions

2. Backup vector stores:

cp -r data/chroma_db ~/backup_chroma

3. Backup logs:

cp logs/rag_application.log ~/backup_logs/

Version Information

Current Version: 1.0.0
Python: 3.10+
Streamlit: 1.28.1+
LangChain: 0.1.14+

Recent Updates

V1.0.0 - Initial Release
- Complete RAG implementation
- Session management
- Multi-PDF support
- Vector store with HNSW
- Chat history persistence
- Streamlit UI

Happy coding!
