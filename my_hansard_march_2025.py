import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.text_splitter import RecursiveCharacterTextSplitter

from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import speech_recognition as sr
import replicate
