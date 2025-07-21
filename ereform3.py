import streamlit as st
import sys
import os
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# sqlite3 workaround with error handling
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    logger.info("Successfully applied pysqlite3 workaround")
except ImportError as e:
    logger.error(f"Failed to import pysqlite3: {e}")
    st.error("Database setup failed. Please check your environment.")
    st.stop()

try:
    import chromadb
    from chromadb.utils import embedding_functions
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import replicate
    from audiorecorder import audiorecorder
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    st.error(f"Missing required package: {e}. Please install all dependencies.")
    st.stop()

# --- CONFIGURATION ---
CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ereform_docs"
DOC_FILE = "Factsheet by user type - summary4.txt"

# Configurable parameters (moved from hard-coded values)
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 20))
DEFAULT_N_RESULTS = int(os.getenv('DEFAULT_N_RESULTS', 15))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 512))

# --- VALIDATION FUNCTIONS ---
def validate_environment() -> bool:
    """Validate that all required files and environment variables exist."""
    try:
        # Check if document file exists
        if not os.path.exists(DOC_FILE):
            st.error(f"📄 Document file not found: {DOC_FILE}")
            logger.error(f"Document file missing: {DOC_FILE}")
            return False
        
        # Check if Replicate API key is available
        replicate_token = os.getenv('REPLICATE_API_TOKEN')
        if not replicate_token:
            st.warning("⚠️ REPLICATE_API_TOKEN not found. AI responses may fail.")
            logger.warning("Replicate API token not configured")
        
        # Create chroma directory if it doesn't exist
        os.makedirs(CHROMA_DATA_PATH, exist_ok=True)
        logger.info("Environment validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        st.error(f"Environment setup failed: {e}")
        return False

# --- SETUP EMBEDDING AND VECTOR DB ---
@st.cache_resource
def get_chroma_collection():
    """Initialize ChromaDB collection with error handling."""
    try:
        logger.info("Initializing ChromaDB collection...")
        client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' initialized successfully")
        return collection
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        st.error(f"Database initialization failed: {e}")
        st.stop()

@st.cache_data
def chunk_document(doc_file: str) -> List[str]:
    """Load and chunk the document with error handling."""
    try:
        logger.info(f"Loading and chunking document: {doc_file}")
        with open(doc_file, "r", encoding='utf-8') as f:
            hansard = f.read()
        
        if not hansard.strip():
            logger.warning("Document is empty")
            st.warning("📄 Document appears to be empty")
            return []
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP, 
            length_function=len
        )
        chunks = splitter.split_text(hansard)
        logger.info(f"Document chunked into {len(chunks)} pieces")
        return chunks
        
    except FileNotFoundError:
        logger.error(f"Document file not found: {doc_file}")
        st.error(f"📄 Document file not found: {doc_file}")
        return []
    except UnicodeDecodeError as e:
        logger.error(f"Failed to read document (encoding issue): {e}")
        st.error("📄 Document encoding issue. Please check file format.")
        return []
    except Exception as e:
        logger.error(f"Failed to chunk document: {e}")
        st.error(f"📄 Failed to process document: {e}")
        return []

def populate_vectordb_if_empty(collection, chunks: List[str]) -> bool:
    """Populate vector database if empty, with error handling."""
    try:
        if not chunks:
            logger.warning("No chunks to populate vectordb")
            return False
            
        count = collection.count()
        logger.info(f"Current vectordb count: {count}")
        
        if not count:  # no docs stored yet
            logger.info("Populating empty vectordb...")
            ids = [f"id{i}" for i in range(len(chunks))]
            collection.add(documents=chunks, ids=ids)
            logger.info(f"Vectordb populated with {len(chunks)} documents")
            return True
        else:
            logger.info("Vectordb already populated")
            return False
            
    except Exception as e:
        logger.error(f"Failed to populate vectordb: {e}")
        st.error(f"🔍 Database population failed: {e}")
        return False

def reindex_vectordb(collection, chunks: List[str]) -> bool:
    """Reindex the vector database with error handling."""
    try:
        if not chunks:
            logger.error("No chunks available for reindexing")
            st.error("No document chunks available for reindexing")
            return False
            
        logger.info("Reindexing vectordb...")
        collection.delete(where={})  # Wipe all docs
        ids = [f"id{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)
        logger.info(f"Vectordb reindexed with {len(chunks)} documents")
        return True
        
    except Exception as e:
        logger.error(f"Failed to reindex vectordb: {e}")
        st.error(f"🔍 Database reindexing failed: {e}")
        return False

def query_vectordb(collection, query: str, n_results: int = DEFAULT_N_RESULTS) -> List[str]:
    """Query vector database with error handling and logging."""
    try:
        if not query.strip():
            logger.warning("Empty query provided")
            return []
            
        logger.info(f"Querying vectordb: '{query[:50]}...' (n_results={n_results})")
        results = collection.query(
            query_texts=[query],
            include=["documents"],
            n_results=n_results
        )
        
        docs = results.get("documents", [[]])[0]
        logger.info(f"Found {len(docs)} relevant documents")
        
        # Log first few characters of each result for debugging
        for i, doc in enumerate(docs[:3]):
            logger.debug(f"Result {i+1}: {doc[:100]}...")
            
        return docs
        
    except Exception as e:
        logger.error(f"Failed to query vectordb: {e}")
        st.error(f"🔍 Search failed: {e}")
        return []

def ask_llama(prompt: str, context: str) -> str:
    """Query Llama model with comprehensive error handling."""
    try:
        if not prompt.strip():
            return "Please provide a valid question."
            
        logger.info(f"Asking AI: '{prompt[:50]}...'")
        full_prompt = f"Prompt: {prompt}\nContext: {context}"
        
        # Validate context isn't too long
        if len(full_prompt) > 10000:  # Rough token limit check
            logger.warning("Context too long, truncating...")
            context = context[:8000] + "...[truncated]"
            full_prompt = f"Prompt: {prompt}\nContext: {context}"
        
        result_ai = ""
        
        # Stream response with timeout handling
        try:
            event_count = 0
            for event in replicate.stream(
                "meta/meta-llama-3-70b-instruct",
                input={
                    "top_k": 50,
                    "top_p": 0.9,
                    "prompt": full_prompt,
                    "max_tokens": MAX_TOKENS,
                    "min_tokens": 0,
                    "temperature": 0.6,
                    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    "presence_penalty": 1.15,
                    "frequency_penalty": 0.2
                }
            ):
                result_ai += str(event)
                event_count += 1
                
                # Prevent infinite loops
                if event_count > 1000:
                    logger.warning("Too many events from Replicate, breaking...")
                    break
                    
        except Exception as stream_error:
            logger.error(f"Streaming error: {stream_error}")
            raise
            
        if not result_ai.strip():
            logger.warning("Empty response from AI")
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
        logger.info(f"AI response generated ({len(result_ai)} characters)")
        return result_ai.strip()
        
    except replicate.exceptions.ReplicateError as e:
        logger.error(f"Replicate API error: {e}")
        return f"🤖 AI service temporarily unavailable: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in ask_llama: {e}")
        return f"🤖 Sorry, I encountered an error: {str(e)}"

# --- INITIALIZE (WITH VALIDATION) ---
def initialize_app():
    """Initialize the application with proper error handling."""
    try:
        # Validate environment first
        if not validate_environment():
            st.stop()
            
        # Initialize components
        collection = get_chroma_collection()
        hansard_chunks = chunk_document(DOC_FILE)
        
        if not hansard_chunks:
            st.error("❌ Failed to load document. Please check the file.")
            st.stop()
            
        # Populate database
        populated = populate_vectordb_if_empty(collection, hansard_chunks)
        if populated:
            st.success(f"✅ Loaded {len(hansard_chunks)} document chunks into database")
            
        return collection, hansard_chunks
        
    except Exception as e:
        logger.error(f"App initialization failed: {e}")
        st.error(f"❌ App initialization failed: {e}")
        st.stop()

# Initialize the app
collection, hansard_chunks = initialize_app()

# --- APP UI ---
st.title("📊 WAZZUP!!! Ask me anything about reforms")

# Debug info in sidebar
with st.sidebar:
    st.subheader("🔧 Debug Info")
    if collection:
        try:
            doc_count = collection.count()
            st.metric("Documents in DB", doc_count)
        except:
            st.metric("Documents in DB", "Error")
    
    st.metric("Chunk Size", CHUNK_SIZE)
    st.metric("Max Tokens", MAX_TOKENS)
    
    # Re-index button
    if st.button("🔄 Re-index Document"):
        with st.spinner("Re-indexing..."):
            success = reindex_vectordb(collection, hansard_chunks)
        if success:
            st.success("✅ Re-indexed successfully!")
        else:
            st.error("❌ Re-indexing failed!")

# Main interface
prompt = st.text_area("What do you want to know?", height=100)

if st.button("Ask Arvee", type="primary"):
    if not prompt.strip():
        st.warning("You need to type a question! I can't read your mind...yet 🙃")
    else:
        with st.spinner("🔍 Retrieving info and asking the AI..."):
            try:
                # Query the database
                docs = query_vectordb(collection, prompt, n_results=10)
                
                if docs:
                    # Show retrieved context
                    with st.expander("📖 Retrieved Context", expanded=False):
                        for i, d in enumerate(docs[:3], 1):
                            st.info(f"**Context {i}:** {d}")
                    
                    # Get AI response
                    context_for_ai = '\n---\n'.join(docs[:3])
                    result = ask_llama(prompt, context_for_ai)
                    
                    # Display result
                    st.subheader("🤖 Arvee says:")
                    st.write(result)
                    
                    # Debug info
                    with st.expander("🔍 Debug Info"):
                        st.write(f"**Query:** {prompt}")
                        st.write(f"**Retrieved {len(docs)} documents**")
                        st.write(f"**Response length:** {len(result)} characters")
                        
                else:
                    st.info("🤷 No relevant context found in the documents.")
                    logger.warning(f"No results for query: {prompt}")
                    
            except Exception as e:
                logger.error(f"Query processing failed: {e}")
                st.error(f"❌ Something went wrong: {e}")

# Footer with status
st.markdown("---")
st.markdown("💡 **Tip:** Check the sidebar for debug information and re-indexing options.")
