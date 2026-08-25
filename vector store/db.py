from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
texts = ["Hello world", "How are you?", "I am fine, thank you."]
vectors = embeddings.embed_documents(texts)
print(vectors)

from langchain_core.documents import Document
docs = [Document(page_content="Hello world", metadata={"source": "example1.txt"}),
        Document(page_content="How are you?", metadata={"source": "example2.txt"})]
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2" 
)
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory="chroma_db"
    )


