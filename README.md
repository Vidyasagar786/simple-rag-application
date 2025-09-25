# RAG Chat Application

A simple and powerful RAG (Retrieval-Augmented Generation) chat application built with **Streamlit**, **LangChain**, and **FAISS**.

## 🚀 Features

- **Simple Web Interface** - Clean Streamlit-based chat UI
- **RAG Pipeline** - Document processing and intelligent retrieval
- **Multiple LLM Support** - Azure OpenAI and AWS Bedrock integration
- **Vector Database** - FAISS for fast document search
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
├── docs/                  # Sample documents for testing
├── .env.example           # Environment template
├── .gitignore            # Git exclusions
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
- **Azure OpenAI** - Primary LLM (GPT-4.1)
- **AWS Bedrock** - Fallback LLM (Claude 3.5 Sonnet)
- **HuggingFace Embeddings** - Local document processing
- **FAISS Vector Store** - In-memory document storage

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

## 🔄 Workflow

```
User uploads document → app.py → rag_service.py → FAISS vector store
User asks question → app.py → rag_service.py → Vector search → LLM → Answer + Sources
```

## ✅ Current Features

- ✅ Azure OpenAI GPT-4.1 for chat responses
- ✅ HuggingFace Embeddings for document processing
- ✅ FAISS Vector Store for document search
- ✅ Streamlit UI for user interaction
- ✅ Environment variable configuration for security
- ✅ Source citations for all responses

## 🔒 Security

- All API keys are stored in `.env` file (not committed to git)
- Use `.env.example` as a template
- Never commit actual credentials

## 📊 Document Processing Flow

1. **Upload**: Document is uploaded via Streamlit interface
2. **Extract**: Text is extracted using PyPDF2, python-docx, etc.
3. **Split**: Document is split into chunks (1000 chars, 200 overlap)
4. **Embed**: Chunks are converted to vectors using HuggingFace
5. **Store**: Vectors are stored in FAISS for fast retrieval
6. **Query**: Questions search for similar chunks and generate answers

## 🚀 Performance

- **Upload Time**: 2-30 seconds depending on file size
- **Query Time**: 2-15 seconds for responses
- **Memory**: Documents persist until app restart
- **Storage**: In-memory FAISS (fast but temporary)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.