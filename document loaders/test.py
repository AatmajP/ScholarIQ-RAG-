#this is the orginal code but due to error not able to run the code 
# so i have changed the code to this one and it is working fine
#from langchain_community.document_loaders import TextLoader
#data = TextLoader("example.txt").load()
#print(data)


from pathlib import Path

from langchain_community.document_loaders import TextLoader

data = TextLoader(
	str(Path(__file__).parent / "example.txt"),
	encoding="utf-8",
).load()
print(data[0].page_content  )


