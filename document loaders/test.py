#this is the orginal code but due to error not able to run the code 
# so i have changed the code to this one and it is working fine
#from langchain_community.document_loaders import TextLoader
#data = TextLoader("example.txt").load()
#print(data)


from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=2
)

data = TextLoader(
	str(Path(__file__).parent / "example.txt"),
	encoding="utf-8",
).load()

splits = splitter.split_documents(data)
print(splits[0].page_content)

for i in splits:
	print(i.page_content)
	print("--------------------------------------------------")
