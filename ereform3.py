"""
Enhanced RAG Document Query Application
=====================================

A Streamlit-based Retrieval-Augmented Generation (RAG) application that allows users
to query documents using vector similarity search backed by ChromaDB and LLaMA.

Features:
- Document chunking and vectorization
- Semantic search with ChromaDB
- LLaMA-powered response generation
- Clean, user-friendly interface
"""

import streamlit as st
import sys
from pathlib import Path
import logging
from typing import List, Optional, Tuple
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import replicate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLite3 workaround for ChromaDB compatibility
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    logger.warning("pysqlite3 not available, using default sqlite3")

# --- CONFIGURATION ---
class Config:
    """Application configuration constants."""
    CHROMA_DATA_PATH = "chroma_data/"
    EMBED_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "ereform_docs"
    DOC_FILE = "Factsheet by user type - summary4.txt"
    
    # Document processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 20
    
    # Query settings  
    DEFAULT_RESULTS = 10
    CONTEXT_RESULTS = 3
    
    # LLaMA settings
    MAX_TOKENS = 512
    TEMPERATURE = 0.6
    TOP_P = 0.9

class VectorDBManager:
    """Manages ChromaDB operations and document vectorization."""
    
    def __init__(self, config: Config):
        self.config = config
        self._collection = None
        
    @st.cache_resource
    def get_collection(_self):
        """Initialize and return ChromaDB collection with caching."""
        try:
            client = chromadb.PersistentClient(path=_self.config.CHROMA_DATA_PATH)
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=_self.config.EMBED_MODEL
            )
            collection = client.get_or_create_collection(
                name=_self.config.COLLECTION_NAME,
                embedding_function=embedding_func,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB collection '{_self.config.COLLECTION_NAME}' initialized")
            return collection
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            st.error("Failed to initialize vector database. Please check your setup.")
            return None
    
    @property
    def collection(self):
        """Lazy loading of collection."""
        if self._collection is None:
            self._collection = self.get_collection()
        return self._collection
    
    def populate_if_empty(self, chunks: List[str]) -> bool:
        """Populate vector database if empty."""
        try:
            if not self.collection.count():
                ids = [f"doc_chunk_{i}" for i in range(len(chunks))]
                self.collection.add(documents=chunks, ids=ids)
                logger.info(f"Added {len(chunks)} chunks to vector database")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to populate vector database: {e}")
            return False
    
    def reindex(self, chunks: List[str]) -> bool:
        """Completely reindex the vector database."""
        try:
            self.collection.delete(where={})
            ids = [f"doc_chunk_{i}" for i in range(len(chunks))]
            self.collection.add(documents=chunks, ids=ids)
            logger.info(f"Reindexed {len(chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to reindex vector database: {e}")
            return False
    
    def query(self, query_text: str, n_results: int = None) -> List[str]:
        """Query vector database for similar documents."""
        if n_results is None:
            n_results = self.config.DEFAULT_RESULTS
            
        try:
            results = self.collection.query(
                query_texts=[query_text],
                include=["documents"],
                n_results=n_results
            )
            documents = results.get("documents", [[]])[0]
            logger.info(f"Retrieved {len(documents)} similar documents")
            return documents
        except Exception as e:
            logger.error(f"Failed to query vector database: {e}")
            return []

class DocumentProcessor:
    """Handles document loading and chunking operations."""
    
    def __init__(self, config: Config):
        self.config = config
    
    @st.cache_data
    def load_and_chunk_document(_self, doc_path: str) -> Optional[List[str]]:
        """Load document and split into chunks with caching."""
        try:
            doc_file = Path(doc_path)
            if not doc_file.exists():
                logger.error(f"Document file not found: {doc_path}")
                return None
                
            with open(doc_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            if not content.strip():
                logger.warning("Document is empty")
                return []
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=_self.config.CHUNK_SIZE,
                chunk_overlap=_self.config.CHUNK_OVERLAP,
                length_function=len
            )
            
            chunks = splitter.split_text(content)
            logger.info(f"Document split into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            return None

class LLaMAClient:
    """Handles LLaMA API interactions."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def generate_response(self, prompt: str, context: str) -> str:
        """Generate response using LLaMA with context."""
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
        Use the context to provide accurate, relevant answers. If the context doesn't contain enough information 
        to answer the question, say so clearly."""
        
        full_prompt = f"""Context: {context}

Question: {prompt}

Please provide a helpful answer based on the context above."""

        try:
            result = ""
            for event in replicate.stream(
                "meta/meta-llama-3-70b-instruct",
                input={
                    "top_k": 50,
                    "top_p": self.config.TOP_P,
                    "prompt": full_prompt,
                    "max_tokens": self.config.MAX_TOKENS,
                    "min_tokens": 0,
                    "temperature": self.config.TEMPERATURE,
                    "prompt_template": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{{prompt}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    "presence_penalty": 1.15,
                    "frequency_penalty": 0.2
                }
            ):
                result += str(event)
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"LLaMA API error: {e}")
            return f"Sorry, I encountered an error while generating the response: {str(e)}"

class RAGApp:
    """Main RAG application class."""
    
    def __init__(self):
        self.config = Config()
        self.doc_processor = DocumentProcessor(self.config)
        self.vector_db = VectorDBManager(self.config)
        self.llama_client = LLaMAClient(self.config)
        self._initialize_app()
    
    def _initialize_app(self):
        """Initialize the application components."""
        # Load and process document
        self.chunks = self.doc_processor.load_and_chunk_document(self.config.DOC_FILE)
        
        if self.chunks is None:
            st.error(f"Failed to load document: {self.config.DOC_FILE}")
            st.stop()
        
        if not self.chunks:
            st.warning("Document is empty or couldn't be processed")
            st.stop()
            
        # Initialize vector database
        if self.vector_db.collection is None:
            st.error("Failed to initialize vector database")
            st.stop()
            
        # Populate vector database if needed
        was_populated = self.vector_db.populate_if_empty(self.chunks)
        if was_populated:
            st.success("Vector database initialized with document chunks!")
    
    def render_sidebar(self):
        """Render sidebar with admin controls."""
        with st.sidebar:
            st.header("📚 Document Management")
            
            st.info(f"**Document:** {self.config.DOC_FILE}")
            st.info(f"**Chunks:** {len(self.chunks)}")
            st.info(f"**Collection:** {self.config.COLLECTION_NAME}")
            
            if st.button("🔄 Reindex Document", help="Rebuild the vector database"):
                with st.spinner("Reindexing document..."):
                    success = self.vector_db.reindex(self.chunks)
                
                if success:
                    st.success("✅ Document reindexed successfully!")
                else:
                    st.error("❌ Failed to reindex document")
    
    def render_main_interface(self):
        """Render main chat interface."""
        st.title("📊 Document Q&A Assistant")
        st.markdown("Ask questions about the reform documentation and get AI-powered answers!")
        
        # Query input
        prompt = st.text_area(
            "What would you like to know?",
            height=100,
            placeholder="Type your question here..."
        )
        
        # Query button
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🤖 Ask Assistant", type="primary")
        
        if ask_button:
            if not prompt.strip():
                st.warning("⚠️ Please enter a question to get started!")
                return
            
            self._process_query(prompt.strip())
    
    def _process_query(self, prompt: str):
        """Process user query and generate response."""
        with st.spinner("🔍 Searching for relevant information..."):
            # Retrieve similar documents
            similar_docs = self.vector_db.query(prompt, self.config.DEFAULT_RESULTS)
            
            if not similar_docs:
                st.info("📝 No relevant information found in the document.")
                return
            
            # Show retrieved context (top 3)
            context_docs = similar_docs[:self.config.CONTEXT_RESULTS]
            
            with st.expander("📋 Retrieved Context", expanded=False):
                for i, doc in enumerate(context_docs, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(doc[:200] + "..." if len(doc) > 200 else doc)
                    st.divider()
        
        with st.spinner("🤖 Generating response..."):
            # Generate response using LLaMA
            context = '\n\n---\n\n'.join(context_docs)
            response = self.llama_client.generate_response(prompt, context)
            
            # Display response
            st.subheader("💬 Assistant Response:")
            st.markdown(response)
            
            # Show query statistics
            with st.expander("📊 Query Details"):
                st.write(f"**Documents retrieved:** {len(similar_docs)}")
                st.write(f"**Context chunks used:** {len(context_docs)}")
                st.write(f"**Total context length:** {len(context)} characters")

def main():
    """Main application entry point."""
    # Configure Streamlit page
    st.set_page_config(
        page_title="Document Q&A Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        # Initialize and run application
        app = RAGApp()
        app.render_sidebar()
        app.render_main_interface()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
