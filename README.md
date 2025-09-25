# RAG Chat Application

A simple and powerful RAG (Retrieval-Augmented Generation) chat application built with **Streamlit**, **LangChain**, and **Neo4j**.

## 🚀 Features

- **Simple Web Interface** - Clean Streamlit-based chat UI
- **RAG Pipeline** - Document processing and intelligent retrieval
- **Multiple LLM Support** - Azure OpenAI and AWS Bedrock integration
- **Vector Database** - Neo4j for knowledge graph storage
- **Document Upload** - Support for PDF, TXT, DOCX files
- **Real-time Chat** - Instant responses with source citations

## 📁 Project Structure

```
sundar_project/
├── streamlit_app/          # Main Streamlit application
│   ├── app.py             # Streamlit UI
│   ├── rag_service.py     # RAG logic
│   ├── config.py          # Configuration
│   └── requirements.txt   # Dependencies
├── backend/               # Optional FastAPI backend
├── docs/                  # Sample documents for testing
└── run_streamlit.py       # Easy startup script
```

## 🎯 Use Cases

1. **Customer Support Assistant** - Answer questions using company knowledge base
2. **Document Q&A System** - Upload documents and ask questions
3. **Research Assistant** - Analyze research papers and documents
4. **Technical Documentation Helper** - Help with code/docs
5. **Legal Document Analyzer** - Analyze contracts and legal documents

## ⚡ Quick Start

### 1. Install Dependencies
```bash
cd streamlit_app
pip install -r requirements.txt
```

### 2. Configure Environment
Copy the example environment file and add your credentials:
```bash
cp .env.example .env
# Edit .env file with your API keys
```

### 3. Run the Application
```bash
# From project root
python run_streamlit.py

# OR directly
cd streamlit_app
streamlit run app.py
```

### 4. Access the Application
- Open your browser to: http://localhost:8501
- Upload documents in the sidebar
- Start chatting!

## 🔧 Configuration

The application uses:
- **Azure OpenAI** - Primary LLM (GPT-4)
- **AWS Bedrock** - Fallback LLM (Claude 3.5 Sonnet)
- **Neo4j** - Vector database for document storage
- **Cohere Rerank** - Document reranking for better relevance

## 📝 Sample Documents

Upload any of these file types:
- **PDF** - Research papers, manuals, reports
- **TXT** - Plain text documents
- **DOCX/DOC** - Microsoft Word documents

## 🎮 How to Use

1. **Upload Documents**: Use the sidebar to upload your documents
2. **Start Chatting**: Type questions about your uploaded documents
3. **View Sources**: Click on "Sources" to see which documents were used
4. **Clear Chat**: Use the sidebar button to start a new conversation

## 🔑 Your Credentials

Your API keys are already configured in the code:
- ✅ Azure OpenAI GPT-4.1
- ✅ AWS Bedrock Claude 3.5 Sonnet  
- ✅ Cohere Rerank v3.5
- ⚠️ Neo4j - Update connection details as needed



User uploads document → app.py → rag_service.py → FAISS vector store
User asks question → app.py → rag_service.py → Vector search → LLM → Answer + Sources
✅ Azure OpenAI GPT-4.1 for chat responses
✅ HuggingFace Embeddings for document processing
✅ FAISS Vector Store for document search
✅ Streamlit UI for user interaction



📤 Upload Document → �� Extract Text → ✂️ Split into Chunks → 🔢 Create Embeddings → �� Store in Vector DB
📤 You upload a PDF/TXT/DOCX file
📄 Text extraction happens instantly (PyPDF2, python-docx, etc.)
✂️ Text splitting - Document is split into chunks (1000 chars each, 200 char overlap)
🔢 Embedding creation - Each chunk gets converted to vector embeddings using HuggingFace
💾 Vector storage - Embeddings stored in FAISS vector database
✅ Ready for queries - You can immediately ask questions



❓ Ask Question → �� Search Similar Chunks → 🤖 Generate Answer → 📚 Show Sources
❓ You ask a question
�� Vector search - Finds most similar document chunks
�� LLM processing - Azure OpenAI combines question + relevant chunks
�� Answer generation - Creates response with source citations
📚 Source display - Shows which documents/chunks were used


🗑️ Document Management
Current Limitations:
❌ No individual document deletion - All uploaded documents stay in memory
❌ No document list management - Can't see what's uploaded
❌ Documents persist - They remain until app restart


Current Setup:
FAISS Vector Store - In-memory (lost on restart)
Chat History - Session-based (cleared with button)
Documents - Persistent until restart
