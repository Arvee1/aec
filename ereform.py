import streamlit as st
import sys

# sqlite3 workaround
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import replicate
from audiorecorder import audiorecorder

# --- CONFIGURATION ---
CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ereform_docs"
DOC_FILE = "24146b01_Electoral_Reform.txt"

# --- SETUP EMBEDDING AND VECTOR DB ---
@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

@st.cache_data
def chunk_document(doc_file):
    with open(doc_file, "r") as f:
        hansard = f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=20, length_function=len
    )
    chunks = splitter.split_text(hansard)
    return chunks

def populate_vectordb_if_empty(collection, chunks):
    if not collection.count():  # no docs stored yet
        ids = [f"id{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)

def reindex_vectordb(collection, chunks):
    collection.delete(where={})  # Wipe all docs
    ids = [f"id{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

def query_vectordb(collection, query, n_results=15):
    results = collection.query(
        query_texts=[query],
        include=["documents"],
        n_results=n_results
    )
    docs = results.get("documents", [[]])[0]
    return docs

def ask_llama(prompt, context):
    full_prompt = f"Prompt: {prompt}\nContext: {context}"
    result_ai = ""
    for event in replicate.stream(
        "meta/meta-llama-3-70b-instruct",
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": full_prompt,
            "max_tokens": 512,
            "min_tokens": 0,
            "temperature": 0.6,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15,
            "frequency_penalty": 0.2
        }
    ):
        result_ai += str(event)
    return result_ai

# --- INITIALIZE (ONLY ONCE, FAST ON RELOADS) ---
collection = get_chroma_collection()
hansard_chunks = chunk_document(DOC_FILE)
populate_vectordb_if_empty(collection, hansard_chunks)

# --- APP UI ---
st.title("📊 WAZZUP!!! Ask me anything about reforms")
# with st.sidebar:
#     if st.button("Re-index Document into VectorDB"):
#         with st.spinner("Re-indexing..."):
#             reindex_vectordb(collection, hansard_chunks)
#         st.success("Re-indexed! (All embeddings reloaded)")

prompt = st.text_area("What do you want to know?")
if st.button("Ask Arvee", type="primary"):
    if not prompt.strip():
        st.warning("You need to type a question! I can't read your mind...yet :)")
    else:
        with st.spinner("Retrieving info and asking the AI..."):
            docs = query_vectordb(collection, prompt, n_results=10)
            if docs:
                st.markdown("**Top retrieved context:**")
                for d in docs[:3]: st.info(d)
                result = ask_llama(prompt, '\n---\n'.join(docs[:3]))
                st.subheader("Arvee says:")
                st.write(result)
            else:
                st.info("No relevant context found.")

