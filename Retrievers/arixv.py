#thsi is data source type retriever for arxiv papers
#where we can query arxiv papers and get the results in the form of 
# documents with metadata


from langchain_community.retrievers import ArxivRetriever

# create the retriever
retriever = ArxivRetriever(
    load_max_docs=2,      # number of papers to retrieve
    load_all_available_meta=True
)

# query arxiv
docs = retriever.invoke("large language models")

# print results
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc[500].page_content)  # print first 500 characters