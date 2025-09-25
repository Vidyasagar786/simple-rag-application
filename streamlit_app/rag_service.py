from langchain_openai import AzureChatOpenAI
from langchain_aws import ChatBedrock
try:
    from langchain_community.vectorstores import Neo4jVector
    from langchain_community.embeddings import OpenAIEmbeddings
except ImportError:
    from langchain.vectorstores import Neo4jVector
    from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import Config
import uuid
import tempfile
import os
from typing import List, Dict, Any
from datetime import datetime
import streamlit as st

class RAGService:
    """Streamlit RAG service for document processing and chat"""
    
    def __init__(self):
        self.llm = None
        self.vector_store = None
        self.embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
        
    def initialize_llm(self):
        """Initialize the language model"""
        try:
            # Try Azure OpenAI first
            self.llm = AzureChatOpenAI(
                azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_version=Config.AZURE_OPENAI_API_VERSION,
                api_key=Config.AZURE_OPENAI_API_KEY,
                temperature=Config.AZURE_OPENAI_TEMPERATURE,
                top_p=Config.AZURE_OPENAI_TOP_P,
                model_name=Config.AZURE_OPENAI_MODEL
            )
            return True, "Azure OpenAI initialized successfully"
            
        except Exception as azure_error:
            try:
                # Try AWS Bedrock as fallback
                self.llm = ChatBedrock(
                    model_id=Config.BEDROCK_MODEL,
                    region_name=Config.AWS_REGION,
                    aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
                    model_kwargs={
                        "max_tokens": Config.BEDROCK_MAX_TOKENS,
                        "temperature": Config.BEDROCK_TEMPERATURE,
                        "top_p": Config.BEDROCK_TOP_P
                    }
                )
                return True, "AWS Bedrock initialized successfully"
            except Exception as bedrock_error:
                return False, f"Both LLM providers failed. Azure: {azure_error}, Bedrock: {bedrock_error}"
    
    def initialize_embeddings(self):
        """Initialize embeddings model"""
        try:
            # Use HuggingFace embeddings
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name='all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu'}
            )
            return True, "HuggingFace embeddings initialized"
                
        except Exception as e:
            return False, f"Error initializing embeddings: {e}"
    
    def initialize_vector_store(self):
        """Initialize vector store (try Neo4j first, fallback to FAISS)"""
        try:
            # Try Neo4j first if password is provided
            if Config.NEO4J_PASSWORD:
                try:
                    # Try to connect to existing vector store
                    self.vector_store = Neo4jVector.from_existing_index(
                        embedding=self.embeddings,
                        url=Config.NEO4J_URI,
                        username=Config.NEO4J_USER,
                        password=Config.NEO4J_PASSWORD,
                        index_name="document_embeddings"
                    )
                    return True, "Connected to existing Neo4j vector store"
                except Exception as e:
                    try:
                        # Create new vector store if it doesn't exist
                        self.vector_store = Neo4jVector.from_documents(
                            documents=[],
                            embedding=self.embeddings,
                            url=Config.NEO4J_URI,
                            username=Config.NEO4J_USER,
                            password=Config.NEO4J_PASSWORD,
                            index_name="document_embeddings"
                        )
                        return True, "Created new Neo4j vector store"
                    except Exception as create_error:
                        print(f"Neo4j failed: {create_error}")
                        # Fall through to FAISS
            
            # Fallback to FAISS (in-memory vector store)
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            import tempfile
            import os
            
            # Create HuggingFace embeddings wrapper
            hf_embeddings = HuggingFaceEmbeddings(
                model_name='all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu'}
            )
            
            # Create a temporary FAISS index
            self.vector_store = FAISS.from_texts(
                texts=["Sample document for initialization"],
                embedding=hf_embeddings
            )
            return True, "Using FAISS vector store (in-memory)"
            
        except Exception as e:
            return False, f"Error initializing vector store: {e}"
    
    def process_document(self, uploaded_file) -> Dict[str, Any]:
        """Process an uploaded document"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load document based on file type
            documents = self._load_document(tmp_file_path, uploaded_file.name)
            
            if not documents:
                os.unlink(tmp_file_path)
                return {
                    "success": False,
                    "message": "Could not extract text from document"
                }
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata to chunks
            document_id = str(uuid.uuid4())
            for chunk in chunks:
                chunk.metadata.update({
                    "document_id": document_id,
                    "filename": uploaded_file.name,
                    "upload_date": datetime.now().isoformat(),
                    "file_size": uploaded_file.size
                })
            
            # Store in vector database
            if self.vector_store:
                if hasattr(self.vector_store, 'add_documents'):
                    self.vector_store.add_documents(chunks)
                else:
                    # For FAISS, we need to recreate the index
                    texts = [chunk.page_content for chunk in chunks]
                    metadatas = [chunk.metadata for chunk in chunks]
                    self.vector_store.add_texts(texts, metadatas)
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            return {
                "success": True,
                "message": f"Document processed successfully. Created {len(chunks)} chunks.",
                "chunks_count": len(chunks),
                "document_id": document_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing document: {str(e)}"
            }
    
    def _load_document(self, file_path: str, filename: str) -> List[Document]:
        """Load document based on file type"""
        try:
            file_extension = "." + filename.split(".")[-1].lower() if "." in filename else ""
            
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
                return loader.load()
            
            elif file_extension in [".txt"]:
                loader = TextLoader(file_path)
                return loader.load()
            
            elif file_extension in [".docx", ".doc"]:
                loader = Docx2txtLoader(file_path)
                return loader.load()
            
            else:
                # Try as text file
                loader = TextLoader(file_path)
                return loader.load()
                
        except Exception as e:
            st.error(f"Error loading document {filename}: {e}")
            return []
    
    def get_chat_response(self, question: str, chat_history: List[Dict]) -> Dict[str, Any]:
        """Get RAG-enhanced chat response"""
        try:
            if not self.llm or not self.vector_store:
                # Simple LLM response without RAG
                response = self.llm.invoke(question) if self.llm else "LLM not initialized"
                return {
                    "answer": response.content if hasattr(response, 'content') else str(response),
                    "sources": [],
                    "success": True
                }
            
            # Create retrieval chain without memory to avoid the output_key issue
            try:
                from langchain.chains import RetrievalQA
                
                # Create a simple RetrievalQA chain instead
                retrieval_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(
                        search_kwargs={"k": Config.TOP_K_RESULTS}
                    ),
                    return_source_documents=True,
                    verbose=False
                )
                
                # Build context from chat history
                context = ""
                if chat_history:
                    context = "Previous conversation:\n"
                    for msg in chat_history[-5:]:  # Last 5 messages
                        if msg["role"] == "user":
                            context += f"Human: {msg['content']}\n"
                        elif msg["role"] == "assistant":
                            context += f"Assistant: {msg['content']}\n"
                    context += "\nCurrent question: "
                
                # Process the question with context
                full_question = context + question
                result = retrieval_chain.invoke({"query": full_question})
                
            except Exception as chain_error:
                print(f"Chain creation error: {chain_error}")
                raise chain_error
            
            # Extract sources
            sources = []
            if result.get("source_documents"):
                for doc in result["source_documents"]:
                    sources.append({
                        "content": doc.page_content[:200] + "...",
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "metadata": doc.metadata
                    })
            
            return {
                "answer": result.get("result", result.get("answer", "No answer generated")),
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "sources": [],
                "success": False
            }