import streamlit as st
import time
from datetime import datetime
from rag_service import RAGService
from config import Config

# Page configuration
st.set_page_config(
    page_title="RAG Chat Bot Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        max-width: 80%;
    }
    
    .user-message {
        background-color: #3b82f6;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    
    .assistant-message {
        background-color: #f3f4f6;
        color: #1f2937;
        margin-right: auto;
    }
    
    .source-card {
        background-color: #afb3b6;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .status-success {
        color: #059669;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc2626;
        font-weight: bold;
    }
    
    .sidebar-section {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_service" not in st.session_state:
    st.session_state.rag_service = RAGService()
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

def initialize_rag_service():
    """Initialize the RAG service"""
    if st.session_state.initialized:
        return
    
    with st.spinner("Initializing RAG service..."):
        # Initialize LLM
        success, message = st.session_state.rag_service.initialize_llm()
        if not success:
            st.error(f"❌ LLM Initialization Failed: {message}")
            return
        
        st.success(f"✅ {message}")
        
        # Initialize Embeddings
        success, message = st.session_state.rag_service.initialize_embeddings()
        if not success:
            st.error(f"❌ Embeddings Initialization Failed: {message}")
            return
        
        st.success(f"✅ {message}")
        
        # Initialize Vector Store
        success, message = st.session_state.rag_service.initialize_vector_store()
        if not success:
            st.error(f"❌ Vector Store Initialization Failed: {message}")
            return
        
        st.success(f"✅ {message}")
        
        st.session_state.initialized = True
        st.success("🎉 RAG Service initialized successfully!")

def display_chat_message(role, content, sources=None):
    """Display a chat message"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources if available
        if sources and len(sources) > 0:
            with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>Source {i}:</strong> {source['filename']}<br>
                        <em>{source['content']}</em>
                    </div>
                    """, unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<div class="main-header">🤖 RAG Chat Application</div>', unsafe_allow_html=True)
    
    # Sidebar for document upload and configuration
    with st.sidebar:
        st.markdown("## 📁 Document Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'txt', 'docx', 'doc'],
            help="Upload PDF, TXT, DOCX, or DOC files for RAG processing"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    result = st.session_state.rag_service.process_document(uploaded_file)
                    
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.session_state.uploaded_documents.append({
                            "name": uploaded_file.name,
                            "chunks": result["chunks_count"],
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    else:
                        st.error(f"❌ {result['message']}")
        
        # Display uploaded documents
        if st.session_state.uploaded_documents:
            st.markdown("### 📋 Uploaded Documents")
            for doc in st.session_state.uploaded_documents:
                st.markdown(f"""
                <div style="background-color: #f9fafb; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <strong>{doc['name']}</strong><br>
                    <small>{doc['chunks']} chunks • {doc['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # Configuration section
        st.markdown("### ⚙️ Configuration")
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin: 1rem 0;">
        <strong>LLM Provider:</strong> {Config.AZURE_OPENAI_DEPLOYMENT}<br>
        <strong>Vector Store:</strong> FAISS (Local)<br>
        <strong>Embeddings:</strong> HuggingFace all-MiniLM-L6-v2 model <br>
        <strong>Chunk Size:</strong> {Config.CHUNK_SIZE}<br>
        <strong>Top K Results:</strong> {Config.TOP_K_RESULTS}
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize RAG service
    initialize_rag_service()
    
    # Main chat interface
    st.markdown("## 💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(
            message["role"], 
            message["content"], 
            message.get("sources", [])
        )
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your uploaded documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        display_chat_message("user", prompt)
        
        # Get AI response
        with st.spinner("Thinking..."):
            chat_history = [msg for msg in st.session_state.messages if msg["role"] in ["user", "assistant"]]
            response = st.session_state.rag_service.get_chat_response(prompt, chat_history)
            
            if response["success"]:
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["answer"],
                    "sources": response["sources"]
                })
                
                # Display assistant message
                display_chat_message("assistant", response["answer"], response["sources"])
            else:
                st.error(f"❌ Error: {response['answer']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.8rem;">
        Powered by LangChain, Azure OpenAI, and FAISS Vector Store , HuggingFace Embeddings (all-MiniLM-L6-v2) | 
        <a href="https://github.com/Vidyasagar786" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()