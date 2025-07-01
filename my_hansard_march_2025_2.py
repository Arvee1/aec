import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# 1. CONFIGURATION
CHROMA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"  # Sentence Transformers
COLLECTION_NAME = "mydocs"
TEXTFILE = "your_text_file.txt"   # <--- change to your file

# 2. VECTOR DB SETUP (Cached)
@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )
    return collection

@st.cache_data
def get_chunks(textfile):
    with open(textfile, "r") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=20, length_function=len
    )
    return splitter.split_text(text)

def add_chunks_to_db(collection, chunks):
    if not collection.count():
        ids = [f"id{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)

# 3. MAIN LOGIC
collection = get_chroma_collection()

if not os.path.exists(TEXTFILE):
    st.error(f"Text file '{TEXTFILE}' not found! Please upload or specify a valid file.")
    st.stop()

chunks = get_chunks(TEXTFILE)

# Offer (re)load
if st.sidebar.button("Add text file to VectorDB"):
    add_chunks_to_db(collection, chunks)
    st.sidebar.success("Text loaded into vector DB.")

add_chunks_to_db(collection, chunks)  # Auto add on load if empty

st.title("🔎 Ask Questions from Your Text File")
st.markdown(f"Loaded and indexed: `{TEXTFILE}` (`{len(chunks)}` chunks)")

# 4. QUESTION/SEARCH
query = st.text_input("Ask a question or search for something:")
if st.button("Search"):
    with st.spinner("Searching..."):
        results = collection.query(
            query_texts=[query],
            include=["documents"],
            n_results=5
        )
        docs = results["documents"][0]
        if docs:
            st.subheader("Top relevant context:")
            for doc in docs:
                st.info(doc)
        else:
            st.warning("No relevant context found for your query.")

# 5. OPTIONAL: UPLOAD YOUR OWN FILE
with st.expander("Upload a different text file (.txt)"):
    upload = st.file_uploader("Upload .txt", type="txt")
    if upload:
        with open("uploaded.txt", "wb") as f:
            f.write(upload.read())
        st.success("Uploaded as uploaded.txt. Change TEXTFILE in code for persistent use!")
