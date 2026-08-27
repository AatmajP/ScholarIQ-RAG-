import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ScholarIQ",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# SIMPLE UI
# =========================================================

st.markdown("""
<style>

    .main {
        background-color: #f8fafc;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 17px;
        margin-bottom: 30px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">📚 ScholarIQ</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload your study material and ask questions from it'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# LLM
# =========================================================

llm = ChatMistralAI(
    model="mistral-small-2603"
)


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """
Context:

{context}

Question:

{question}
"""
        )
    ]
)


# =========================================================
# SIDEBAR - PDF UPLOAD
# =========================================================

with st.sidebar:

    st.header("📄 Upload Study Material")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    # Check whether this is a new uploaded file
    if (
        "uploaded_file_name" not in st.session_state
        or st.session_state.uploaded_file_name
        != uploaded_file.name
    ):

        with st.spinner("Processing your PDF..."):

            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                temp_pdf_path = temp_file.name


            # Load PDF
            data_pdf = PyPDFLoader(
                temp_pdf_path
            ).load()


            # Split document
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(
                data_pdf
            )


            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                collection_name="scholariq",
                persist_directory="chroma_db"
            )


            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )


            # Store in session
            st.session_state.retriever = retriever
            st.session_state.uploaded_file_name = (
                uploaded_file.name
            )

            # Remove temporary file
            os.remove(temp_pdf_path)


        st.sidebar.success(
            f"✓ {uploaded_file.name}"
        )

        st.sidebar.caption(
            f"{len(chunks)} chunks created"
        )


# =========================================================
# CHAT
# =========================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# Display previous messages

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# =========================================================
# USER QUESTION
# =========================================================

query = st.chat_input(
    "Ask something about your document..."
)


if query:

    # Check PDF
    if "retriever" not in st.session_state:

        st.warning(
            "Please upload a PDF before asking a question."
        )

    else:

        # Display user message
        with st.chat_message("user"):

            st.write(query)


        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": query
            }
        )


        # Retrieve relevant chunks
        with st.spinner("Searching the document..."):

            docs = st.session_state.retriever.invoke(
                query
            )


            # Create context
            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )


            # Create prompt
            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )


            # Ask Mistral
            response = llm.invoke(
                final_prompt
            )


        # Display AI response
        with st.chat_message("assistant"):

            st.write(
                response.content
            )


        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response.content
            }
        )