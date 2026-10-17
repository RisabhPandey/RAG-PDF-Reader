from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Note: Added 'r' before the string to make it a raw string. 
# This prevents Windows path issues with backslashes (like \D in \Document).
data = PyPDFLoader(r"E:\Learning\SELF\INCOMPLETE\RAG\Document loaders\Deep Learning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

# Updated to match main.py
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents = chunks, 
    embedding = embeddings, 
    persist_directory="chroma-db"
)