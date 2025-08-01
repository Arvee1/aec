import streamlit as st
import sys
import os
import logging
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# SQLite3 workaround for deployment environments
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Use system sqlite3 if pysqlite3 not available

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import replicate

# --- CONFIGURATION ---
@dataclass
class Config:
    """Application configuration settings"""
    CHROMA_DATA_PATH: str = os.getenv("CHROMA_DATA_PATH", "chroma_data/")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "ereform_docs")
    DOC_FILE: str = os.getenv("DOC_FILE", "Factsheet by user type - summary4.txt")
    
    # Chunking parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    # Query parameters
    DEFAULT_RESULTS: int = int(os.getenv("DEFAULT_RESULTS", "10"))
    CONTEXT_RESULTS: int = int(os.getenv("CONTEXT_RESULTS", "3"))
    
    # LLM parameters
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.6"))
    
    # Replicate model and timeout settings
    REPLICATE_MODEL: str = os.getenv("REPLICATE_MODEL", "meta/meta-llama-3-70b-instruct")
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "120"))  # 2 minutes
    STREAM_TIMEOUT: int = int(os.getenv("STREAM_TIMEOUT", "300"))   # 5 minutes
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY: int = int(os.getenv("RETRY_DELAY", "2"))  # seconds

config = Config()

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- UTILITY FUNCTIONS ---
def validate_inputs() -> bool:
    """Validate required environment variables and files"""
    try:
        # Check if document file exists
        if not os.path.exists(config.DOC_FILE):
            st.error(f"Document file not found: {config.DOC_FILE}")
            return False
        
        # Check if Replicate API key is set
        if not os.getenv("REPLICATE_API_TOKEN"):
            st.error("REPLICATE_API_TOKEN environment variable not set")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Validation error: {e}")
        st.error(f"Configuration error: {e}")
        return False

def sanitize_input(text: str) -> str:
    """Basic input sanitization"""
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace and limit length
    sanitized = text.strip()[:2000]  # Reasonable length limit
    
    # Basic security: remove potential script injections
    dangerous_patterns = ['<script', 'javascript:', 'data:']
    for pattern in dangerous_patterns:
        sanitized = sanitized.replace(pattern, '')
    
    return sanitized

# --- VECTOR DATABASE OPERATIONS ---
@st.cache_resource
def get_chroma_collection():
    """Initialize and return ChromaDB collection with error handling"""
    try:
        logger.info("Initializing ChromaDB collection")
        client = chromadb.PersistentClient(path=config.CHROMA_DATA_PATH)
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBED_MODEL
        )
        collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB collection initialized with {collection.count()} documents")
        return collection
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        st.error(f"Database initialization failed: {e}")
        return None

@st.cache_data
def chunk_document(doc_file: str) -> Optional[List[str]]:
    """Load and chunk document with error handling"""
    try:
        logger.info(f"Loading and chunking document: {doc_file}")
        with open(doc_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        if not content.strip():
            logger.warning("Document is empty")
            return []
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len
        )
        chunks = splitter.split_text(content)
        logger.info(f"Document chunked into {len(chunks)} pieces")
        return chunks
        
    except FileNotFoundError:
        logger.error(f"Document file not found: {doc_file}")
        st.error(f"Document file not found: {doc_file}")
        return None
    except Exception as e:
        logger.error(f"Failed to chunk document: {e}")
        st.error(f"Document processing failed: {e}")
        return None

def populate_vectordb_if_empty(collection, chunks: List[str]) -> bool:
    """Populate vector database if empty"""
    try:
        if not collection or not chunks:
            return False
            
        if collection.count() == 0:
            logger.info("Populating empty vector database")
            ids = [f"doc_chunk_{i}" for i in range(len(chunks))]
            collection.add(documents=chunks, ids=ids)
            logger.info(f"Added {len(chunks)} chunks to vector database")
        return True
    except Exception as e:
        logger.error(f"Failed to populate vector database: {e}")
        st.error(f"Database population failed: {e}")
        return False

def reindex_vectordb(collection, chunks: List[str]) -> bool:
    """Reindex vector database with new chunks"""
    try:
        if not collection or not chunks:
            return False
            
        logger.info("Reindexing vector database")
        collection.delete(where={})  # Clear all documents
        ids = [f"doc_chunk_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)
        logger.info(f"Reindexed {len(chunks)} chunks")
        return True
    except Exception as e:
        logger.error(f"Failed to reindex vector database: {e}")
        st.error(f"Database reindexing failed: {e}")
        return False

def query_vectordb(collection, query: str, n_results: int = None) -> Optional[List[str]]:
    """Query vector database with error handling"""
    try:
        if not collection or not query.strip():
            return None
            
        n_results = n_results or config.DEFAULT_RESULTS
        logger.info(f"Querying vector database for: {query[:50]}...")
        
        results = collection.query(
            query_texts=[query],
            include=["documents"],
            n_results=n_results
        )
        
        docs = results.get("documents", [[]])[0]
        logger.info(f"Retrieved {len(docs)} relevant documents")
        return docs
        
    except Exception as e:
        logger.error(f"Vector database query failed: {e}")
        st.error(f"Search failed: {e}")
        return None

# --- LLM OPERATIONS ---
def ask_llama_with_retry(prompt: str, context: str, max_retries: int = None) -> Optional[str]:
    """Query Llama model with retry logic and timeout handling"""
    max_retries = max_retries or config.MAX_RETRIES
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Querying Llama model (attempt {attempt + 1}/{max_retries})")
            result = ask_llama_single_attempt(prompt, context)
            if result:
                return result
                
        except (requests.exceptions.Timeout, 
                requests.exceptions.ReadTimeout,
                replicate.exceptions.ReplicateError) as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:  # Not the last attempt
                wait_time = config.RETRY_DELAY * (attempt + 1)  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                st.info(f"Request timed out. Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed")
                st.error("AI service is experiencing issues. Please try again with a shorter question or check back later.")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:  # Last attempt
                st.error(f"AI processing failed: {e}")
                return None
    
    return None

def ask_llama_single_attempt(prompt: str, context: str) -> Optional[str]:
    """Single attempt to query Llama model with timeout handling"""
    if not prompt.strip():
        return None
    
    # Construct full prompt
    full_prompt = f"""Based on the following context, please answer the question accurately and concisely.

Context:
{context}

Question: {prompt}

Answer:"""

    result_ai = ""
    start_time = time.time()
    
    try:
        # Configure Replicate client with timeout
        client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
        
        # Stream response with timeout monitoring
        stream = client.stream(
            config.REPLICATE_MODEL,
            input={
                "top_k": 50,
                "top_p": 0.9,
                "prompt": full_prompt,
                "max_tokens": config.MAX_TOKENS,
                "min_tokens": 0,
                "temperature": config.TEMPERATURE,
                "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant that answers questions based on provided context.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "presence_penalty": 1.15,
                "frequency_penalty": 0.2
            }
        )
        
        # Process stream with timeout monitoring
        for event in stream:
            current_time = time.time()
            if current_time - start_time > config.STREAM_TIMEOUT:
                logger.error(f"Stream timeout after {config.STREAM_TIMEOUT} seconds")
                raise requests.exceptions.ReadTimeout("Stream processing timeout")
            
            result_ai += str(event)
            
            # Optional: Show progress to user
            if len(result_ai) % 100 == 0 and len(result_ai) > 0:  # Every 100 characters
                elapsed = current_time - start_time
                st.info(f"Generating response... ({len(result_ai)} characters, {elapsed:.1f}s)")
        
        logger.info(f"Llama model response received in {time.time() - start_time:.2f}s")
        return result_ai.strip()
        
    except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
        elapsed = time.time() - start_time
        logger.error(f"Request timeout after {elapsed:.2f}s: {e}")
        raise  # Re-raise for retry logic
        
    except replicate.exceptions.ReplicateError as e:
        logger.error(f"Replicate API error: {e}")
        raise  # Re-raise for retry logic
        
    except Exception as e:
        logger.error(f"Unexpected error in LLM query: {e}")
        raise

def ask_llama(prompt: str, context: str) -> Optional[str]:
    """Main LLM query function with fallbacks"""
    # Try with retry logic first
    result = ask_llama_with_retry(prompt, context)
    
    if not result:
        # Fallback: Try with shorter context
        logger.info("Retrying with shorter context")
        short_context = context[:1000] + "..." if len(context) > 1000 else context
        result = ask_llama_with_retry(prompt, short_context, max_retries=1)
        
        if result:
            st.warning("⚠️ Response generated with reduced context due to timeout issues.")
    
    return result

# --- INITIALIZATION ---
def initialize_app() -> Tuple[Optional[object], Optional[List[str]]]:
    """Initialize application components"""
    if not validate_inputs():
        return None, None
    
    # Initialize vector database
    collection = get_chroma_collection()
    if not collection:
        return None, None
    
    # Load and chunk document
    chunks = chunk_document(config.DOC_FILE)
    if not chunks:
        return None, None
    
    # Populate database if empty
    if not populate_vectordb_if_empty(collection, chunks):
        return None, None
    
    return collection, chunks

# --- STREAMLIT UI ---
def main():
    """Main application function"""
    st.set_page_config(
        page_title="Reform Document Q&A",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Reform Document Q&A Assistant")
    st.markdown("Ask me anything about the reform documents and I'll provide context-aware answers!")
    
    # Initialize app
    with st.spinner("Initializing application..."):
        collection, chunks = initialize_app()
    
    if not collection or not chunks:
        st.error("Application initialization failed. Please check your configuration.")
        st.stop()
    
    # Sidebar for configuration (optional)
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info(f"Document: {os.path.basename(config.DOC_FILE)}")
        st.info(f"Total chunks: {len(chunks)}")
        st.info(f"Database entries: {collection.count()}")
        
        if st.button("🔄 Re-index Document", help="Reload and re-index the document"):
            with st.spinner("Re-indexing document..."):
                if reindex_vectordb(collection, chunks):
                    st.success("✅ Document re-indexed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Re-indexing failed")
    
    # Main query interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area(
            "What would you like to know?",
            placeholder="Enter your question about the reform documents...",
            height=100
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        ask_button = st.button("🤖 Ask Assistant", type="primary", use_container_width=True)
    
    # Handle query
    if ask_button:
        sanitized_prompt = sanitize_input(prompt)
        
        if not sanitized_prompt:
            st.warning("⚠️ Please enter a valid question!")
        else:
            # Create a progress container
            progress_container = st.empty()
            status_container = st.empty()
            
            with progress_container:
                st.info("🔍 Searching documents...")
            
            # Query vector database
            docs = query_vectordb(collection, sanitized_prompt, config.DEFAULT_RESULTS)
            
            if not docs:
                progress_container.empty()
                st.info("ℹ️ No relevant context found in the documents.")
            else:
                # Show retrieved context (expandable)
                with st.expander("📄 Retrieved Context", expanded=False):
                    for i, doc in enumerate(docs[:config.CONTEXT_RESULTS], 1):
                        st.markdown(f"**Context {i}:**")
                        st.info(doc)
                
                with progress_container:
                    st.info("🤖 Generating AI response...")
                
                with status_container:
                    st.info("⏳ This may take up to 2-3 minutes. Please be patient...")
                
                # Generate answer with timeout handling
                context = '\n\n---\n\n'.join(docs[:config.CONTEXT_RESULTS])
                result = ask_llama(sanitized_prompt, context)
                
                # Clear progress indicators
                progress_container.empty()
                status_container.empty()
                
                if result:
                    st.subheader("🎯 Answer:")
                    st.markdown(result)
                    
                    # Show response time info
                    st.caption("💡 Tip: For faster responses, try shorter questions or break complex questions into parts.")
                else:
                    st.error("❌ Failed to generate answer after multiple attempts. Please try again with a shorter question.")
                    
                    # Provide helpful suggestions
                    with st.expander("💡 Troubleshooting Tips", expanded=True):
                        st.markdown("""
                        **If you're experiencing timeouts:**
                        - Try breaking your question into smaller parts
                        - Use simpler, more direct questions
                        - Check if the AI service is experiencing high load
                        - Wait a few minutes before trying again
                        
                        **Example of a good question format:**
                        - "What are the main benefits of the reform?"
                        - "How does this affect small businesses?"
                        - "What are the implementation timelines?"
                        """)

if __name__ == "__main__":
    main()
