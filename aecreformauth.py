"""
Enhanced RAG Document Query Application with User Authentication
===============================================================

A Streamlit-based Retrieval-Augmented Generation (RAG) application that allows users
to query documents using vector similarity search backed by ChromaDB and LLaMA.

Features:
- User authentication and session management
- Query logging to JSON file
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
import json
from datetime import datetime
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
    DOC_FILE = "24146b01_Electoral_Reform.txt"
    QUERY_LOG_FILE = "user_queries.json"
    
    # Document processing
    CHUNK_SIZE = 800      # Larger chunks for more context per piece
    CHUNK_OVERLAP = 100   # More overlap to preserve context between chunks
    
    # Query settings  
    DEFAULT_RESULTS = 20  # Retrieve more documents for better coverage
    CONTEXT_RESULTS = 8   # Use more context chunks for better answers
    
    # LLaMA settings
    MAX_TOKENS = 800      # Moderate length for clear, concise answers
    TEMPERATURE = 0.4     # Balanced for accuracy and natural language
    TOP_P = 0.8

class QueryLogger:
    """Handles logging of user queries and responses."""
    
    def __init__(self, config: Config):
        self.config = config
        self.log_file = Path(config.QUERY_LOG_FILE)
        self._ensure_log_file_exists()
    
    def _ensure_log_file_exists(self):
        """Create log file if it doesn't exist."""
        if not self.log_file.exists():
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
    
    def log_query(self, username: str, query: str, response: str, context_chunks: int):
        """Log a user query and response."""
        try:
            # Load existing logs
            with open(self.log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            # Create new log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "username": username,
                "query": query,
                "response": response,
                "context_chunks": context_chunks,
                "session_id": st.session_state.get('session_id', 'unknown')
            }
            
            # Append new entry
            logs.append(log_entry)
            
            # Save updated logs
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Logged query for user: {username}")
            
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
    
    def get_user_query_count(self, username: str) -> int:
        """Get the number of queries for a specific user."""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            return len([log for log in logs if log.get('username') == username])
            
        except Exception as e:
            logger.error(f"Failed to get user query count: {e}")
            return 0
    
    def get_recent_queries(self, username: str, limit: int = 5) -> List[dict]:
        """Get recent queries for a specific user."""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            user_logs = [log for log in logs if log.get('username') == username]
            return sorted(user_logs, key=lambda x: x['timestamp'], reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recent queries: {e}")
            return []

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
        system_prompt = """You are a helpful assistant that explains reform documentation in simple, easy-to-understand language. Your goal is to make complex information accessible to everyday people.

Guidelines:
1. Use simple, everyday language - avoid jargon and technical terms
2. Break down complex ideas into easy-to-follow points
3. Use examples and analogies when helpful
4. Keep sentences short and clear
5. If you must use technical terms, explain them in plain English
6. Structure your answer with clear headings or bullet points
7. Focus on what matters most to the person asking
8. Write like you're explaining to a friend or family member"""
        
        full_prompt = f"""Based on the following information from reform documents, please answer the user's question in simple, clear language that anyone can understand.

CONTEXT:
{context}

USER QUESTION: {prompt}

INSTRUCTIONS: 
- Explain this in simple, everyday language
- Avoid technical jargon - if you must use it, explain what it means
- Break down complex ideas into easy steps
- Use examples if helpful
- Keep your answer clear and to the point
- Think about what the person really needs to know"""

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
        self.query_logger = QueryLogger(self.config)
        self._initialize_session_state()
        self._initialize_app()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if 'username' not in st.session_state:
            st.session_state.username = None
        
        if 'user_authenticated' not in st.session_state:
            st.session_state.user_authenticated = False
    
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
    
    def render_user_authentication(self):
        """Render user authentication interface."""
        st.title("📊 Wazzup and Welcome to your EReform Assistant")
        st.markdown("Please enter your name to get started!")
        
        with st.form("user_auth_form"):
            username = st.text_input(
                "Your Name:",
                placeholder="Enter your full name",
                help="This will be used to track your queries"
            )
            
            submitted = st.form_submit_button("Continue", type="primary")
            
            if submitted:
                if username.strip():
                    st.session_state.username = username.strip()
                    st.session_state.user_authenticated = True
                    st.success(f"Welcome, {username}! 🎉")
                    st.rerun()
                else:
                    st.error("Please enter your name to continue.")
    
    def render_sidebar(self):
        """Render sidebar with admin controls and user info."""
        with st.sidebar:
            # User info section
            if st.session_state.user_authenticated:
                st.header(f"👋 Hello, {st.session_state.username}!")
                
                # Show user statistics
                query_count = self.query_logger.get_user_query_count(st.session_state.username)
                st.metric("Your Queries", query_count)
                
                # Show recent queries
                if query_count > 0:
                    with st.expander("📋 Your Recent Queries"):
                        recent_queries = self.query_logger.get_recent_queries(st.session_state.username, 3)
                        for i, query_log in enumerate(recent_queries, 1):
                            st.write(f"**{i}.** {query_log['query'][:50]}...")
                            st.caption(f"Asked: {query_log['timestamp'][:19]}")
                
                st.divider()
            
            # Document management section
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
            
            # Logout option
            if st.session_state.user_authenticated:
                st.divider()
                if st.button("🚪 Switch User"):
                    st.session_state.user_authenticated = False
                    st.session_state.username = None
                    st.rerun()
    
    def render_main_interface(self):
        """Render main chat interface."""
        if not st.session_state.user_authenticated:
            self.render_user_authentication()
            return
            
        st.title("📊 Document Q&A Assistant")
        st.markdown(f"Welcome back, **{st.session_state.username}**! Ask me any questions about reforms.")
        
        # Add helpful examples
        with st.expander("💡 Example Questions", expanded=False):
            st.markdown("""
            **Try asking questions like:**
            - What are the main changes in the reform?
            - How will this affect me?
            - What do I need to do?
            - When do these changes start?
            - Who can help me with this?
            - What are the benefits?
            """)
        
        # Query input
        prompt = st.text_area(
            "What would you like to know?",
            height=100,
            placeholder="Ask your question in plain English - no need for technical terms!"
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
            
            with st.expander("📋 Retrieved Context", expanded=True):  # Expand by default for debugging
                for i, doc in enumerate(context_docs, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(doc)  # Show full chunks instead of truncated
                    st.divider()
        
        with st.spinner("🤖 Generating response..."):
            # Generate response using LLaMA
            context = '\n\n---\n\n'.join(context_docs)
            response = self.llama_client.generate_response(prompt, context)
            
            # Log the query and response
            self.query_logger.log_query(
                username=st.session_state.username,
                query=prompt,
                response=response,
                context_chunks=len(context_docs)
            )
            
            # Display response
            st.subheader("💬 Here's what I found:")
            
            # Add a friendly intro
            st.markdown("*Let me explain this in simple terms:*")
            st.markdown(response)
            
            # Add helpful follow-up suggestions
            st.markdown("---")
            st.markdown("**Need more help?** Try asking:")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Can you give me an example?"):
                    st.session_state['follow_up'] = f"Can you give me a specific example related to: {prompt}"
                if st.button("🔍 Tell me more details"):
                    st.session_state['follow_up'] = f"Can you explain more details about: {prompt}"
            with col2:
                if st.button("❓ How does this affect me?"):
                    st.session_state['follow_up'] = f"How does this affect regular people: {prompt}"
                if st.button("📅 When does this happen?"):
                    st.session_state['follow_up'] = f"What are the timelines and dates for: {prompt}"
            
            # Show follow-up question if clicked
            if 'follow_up' in st.session_state:
                st.info(f"💡 **Follow-up question:** {st.session_state['follow_up']}")
                if st.button("Ask this follow-up question"):
                    self._process_query(st.session_state['follow_up'])
                    del st.session_state['follow_up']
            
            # Show query statistics
            with st.expander("📊 Query Details"):
                st.write(f"**Documents retrieved:** {len(similar_docs)}")
                st.write(f"**Context chunks used:** {len(context_docs)}")
                st.write(f"**Total context length:** {len(context)} characters")
                st.write(f"**Average chunk length:** {len(context) // len(context_docs) if context_docs else 0} characters")
                st.write(f"**Query logged:** ✅ Yes")
                
                # Show similarity scores if available (for debugging)
                if hasattr(self.vector_db.collection, 'query'):
                    try:
                        results_with_scores = self.vector_db.collection.query(
                            query_texts=[prompt],
                            include=["documents", "distances"],
                            n_results=len(context_docs)
                        )
                        if "distances" in results_with_scores:
                            distances = results_with_scores["distances"][0]
                            st.write("**Similarity scores (lower = more similar):**")
                            for i, distance in enumerate(distances[:3]):
                                st.write(f"Chunk {i+1}: {distance:.3f}")
                    except:
                        pass

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
