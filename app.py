
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

st.title("Multi-Document with pdf")

api_key = os.getenv("GROQ_API_KEY")

if "db" not in st.session_state:
    st.session_state.db = None

if "llm" not in st.session_state:
    st.session_state.llm = None

files = st.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Upload Files"):

    if not files:
        st.warning("Please upload PDF files.")

    elif not api_key:
        st.error("GROQ_API_KEY is missing")

    else:
        try:
            documents = []

            with st.spinner("Reading PDF files."):
                for file in files:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as temp:
                        temp.write(file.getvalue())
                        temp_path = temp.name

                    try:
                        loader = PyPDFLoader(temp_path)
                        documents.extend(loader.load())
                    finally:
                        os.remove(temp_path)

            if not documents:
                st.error("No text found in the uploaded PDFs.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            st.session_state.llm = ChatGroq(
                model="openai/gpt-oss-20b",
                api_key=api_key,
                temperature=0
            )

            st.success("PDFs processed successfully!")

        except Exception as e:
            st.error(f"Error processing PDFs: {e}")

if st.session_state.db is not None:

    question = st.text_input("Ask a question about your PDFs")

    if st.button("Get Answer"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner("Finding the answer..."):
                    docs = st.session_state.db.similarity_search(
                        question,
                        k=5
                    )

                    if not docs:
                        st.warning("No relevant information found.")
                    else:
                        context = "\n\n".join(
                            doc.page_content for doc in docs
                        )

                        prompt = f"""
You are a helpful assistant that answers questions about uploaded PDFs.


PDF Context:
{context}

User Question:
{question}

Answer:
"""
                        response = st.session_state.llm.invoke(prompt)

                        st.subheader("Answer")
                        st.write(response.content)

                        with st.expander("View Retrieved PDF Content"):
                            for i, doc in enumerate(docs, start=1):
                                page = doc.metadata.get("page", 0)
                                st.markdown(
                                    f"**Source {i} — Page {page + 1}**"
                                )
                                st.write(doc.page_content)

            except Exception as e:
                st.error(f"Error generating answer: {e}")
