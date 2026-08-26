#load pdf document
from langchain_community.document_loaders import PyPDFLoader
#split the document into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
#for creating embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
#for creating vectorstore
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


data_pdf = PyPDFLoader("document loaders/example.pdf").load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    
)
chunks=splitter.split_documents(data_pdf)
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)