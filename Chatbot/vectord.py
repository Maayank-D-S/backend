# Step 1: Read text from DOCX
import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Load DOCX file ===
loader = UnstructuredFileLoader(r"C:\Users\vaibh\Downloads\Riverside.docx")
raw_docs = loader.load()

# === Split documents ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(raw_docs)

# === Create embeddings and FAISS index ===
embedding = OpenAIEmbeddings()
vector_db = FAISS.from_documents(docs, embedding)

# === Save the FAISS vectorstore ===
vector_db.save_local("riverside_faiss")

print("✅ FAISS vector DB created and saved as 'riverside_faiss_db'")
