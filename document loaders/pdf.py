from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter



data = PyPDFLoader("document loaders/example.pdf").load()

print(data[10])