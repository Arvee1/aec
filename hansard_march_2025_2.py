import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Settings ---
CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "hansard_docs"
DOC_FILE = "hansard_march_2025.txt"

# --- Vector DB and Embedding Setup ---
@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
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
def process_file_and_store(file_path):
    # Read and chunk the document
    with open(file_path, "r") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=20, length_function=len
    )
    chunks = splitter.split_text(text)
    # Only add new chunks (avoid duplicates)
    collection = get_chroma_collection()
    existing = collection.get()
    if not existing["ids"]:  # Only add if DB is empty
        ids = [f"id{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)
    return len(chunks)

# Call on start
num_chunks = process_file_and_store(DOC_FILE)
collection = get_chroma_collection()

# --- UI ---
st.title("📄 Hansard Vector Search QA")

st.markdown(f"Loaded `{DOC_FILE}` into database as {num_chunks} chunks.")

prompt = st.text_area("Ask anything about the Hansard:")

if st.button("Query the Vector Database"):
    if not prompt.strip():
        st.warning("Please enter a question.")
    else:
        results = collection.query(
            query_texts=[prompt], include=["documents"], n_results=5
        )
        docs = results["documents"][0]
        st.subheader("Top relevant context from your docs:")
        for doc in docs:
            st.info(doc)
        # Below: call your LLM with the retrieved context (`docs`)
        # This is a stub -- replace below section with your call to Replicate or OpenAI as you had before
        st.write("_Now call your LLM to generate an answer using the above content as context._")

# Optionally: Support file uploads
with st.expander("💾 Upload a new text document"):
    uploaded = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded is not None:
        # Save and process new file
        with open("uploaded.txt", "wb") as f:
            f.write(uploaded.read())
        n_chunks = process_file_and_store("uploaded.txt")
        st.success(f"{n_chunks} new chunks added from uploaded document.")
