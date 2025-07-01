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
COLLECTION_NAME = "hansard_docs"
DOC_FILE = "Finance_2025_03_27.txt" 
