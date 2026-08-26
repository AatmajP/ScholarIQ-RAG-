from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the document
#data = TextLoader(
 #   str(Path(__file__).parent / "document loaders" / "example.txt"),
 #   encoding="utf-8",
#).load()

# Load the PDF document





# Create a prompt template for assigining role to the model
template=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a ai that summarizes text."),
        ("user", "{data}"),
    ]
)   






# Invoke the model with the prompt
model = ChatMistralAI(model='mistral-small-2603')


