from langchain_community.document_loaders import TextLoader
data = TextLoader("document loaders/example.txt").load()
print(data)