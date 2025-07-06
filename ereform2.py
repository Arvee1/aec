import streamlit as st
import sys
import json
import os

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
FEEDBACK_FILE = "user_feedback.jsonl"

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

def get_user_feedback(question):
    import json
    if not os.path.exists(FEEDBACK_FILE):
        return None
    with open(FEEDBACK_FILE, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["prompt"].strip().lower() == question.strip().lower():
                return data
    return None

def load_all_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return []
    with open(FEEDBACK_FILE, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

with st.sidebar:
    st.header("📝 User Feedback History")
    feedback_entries = load_all_feedback()
    if feedback_entries:
        # Show the last N entries (change N as desired)
        N = min(len(feedback_entries), 10)
        for entry in reversed(feedback_entries[-N:]):
            st.markdown("---")
            st.markdown(f"**Prompt:** {entry['prompt']}")
            st.markdown(f"**Rating:** {entry['user_feedback']}")
            if entry['user_correction']:
                st.markdown(f"**Correction:** {entry['user_correction']}")
            # Optional: Show part of AI answer
            # st.markdown(f"<span style='font-size:small;'>{entry['ai_answer'][:100]}</span>",
            #             unsafe_allow_html=True)

        if len(feedback_entries) > N:
            st.info(f"Showing last {N} of {len(feedback_entries)} feedback entries.")
    else:
        st.info("No feedback submitted yet!")

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
            # Make sure feedback_data is defined at the top of this block!
            feedback_data = get_user_feedback(prompt)
            docs = query_vectordb(collection, prompt, n_results=10)
            if docs:
                st.markdown("**Top retrieved context:**")
                for d in docs[:3]: st.info(d)

                # Use user correction if available and marked as such
                user_context = ""
                if feedback_data and feedback_data["user_feedback"] in ["Bad", "Can be improved"] and feedback_data["user_correction"]:
                    user_context = f"Note: A user suggested this correction:\n{feedback_data['user_correction']}"

                # Compose context with possible user-contributed corrections
                final_context = '\n---\n'.join(docs[:3])
                if user_context:
                    final_context = final_context + "\n" + user_context

                result = ask_llama(prompt, final_context)
                st.subheader("Arvee says:")
                st.write(result)
                
                # result = ask_llama(prompt, '\n---\n'.join(docs[:3]))
                # st.subheader("Arvee says:")
                # st.write(result)
                
                # --- Feedback UI ---
                st.markdown("### Was this answer helpful?")
                fb = st.radio("Your rating:", ["Good", "Bad", "Can be improved"], key="feedback_radio")
                user_correction = ""
                if fb in ["Bad", "Can be improved"]:
                    user_correction = st.text_area("What would have been a better answer? (optional)")
                
                if st.button("Submit Feedback"):
                    # Store feedback to file
                    import json
                    feedback = {
                        "prompt": prompt,
                        "context": docs[:3],
                        "ai_answer": result,
                        "user_feedback": fb,
                        "user_correction": user_correction
                    }
                    with open(FEEDBACK_FILE, "a") as f:
                        f.write(json.dumps(feedback) + "\n")
                    st.success("Feedback received - thank you!")            
            else:
                st.info("No relevant context found.")
