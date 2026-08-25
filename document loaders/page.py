from langchain_community.document_loaders import WebBaseLoader

url='https://www.apple.com/in/iphone/'
data= WebBaseLoader(url).load()
print(data[0].page_content)