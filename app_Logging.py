import os
import logging
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Setup logging
logging.basicConfig(
    filename="app.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model='text-embedding-3-large', openai_api_key=OPENAI_API_KEY)

# Creating Db path
db_path = r".\faiss_index"

custom_prompt_template = """
    You are an helpful assistant for the question answering system. Use the following pieces of retrieved context to answer the questions.
    IF you don’t know the answer, say "I don't know". Use four to five sentences to answer the question and keep the answers concise.

    context: {context}
    question: {question}

    answer:
"""

def load_data(file_path):
    logging.info(f"Loading PDF file from {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    logging.info(f"Loaded {len(docs)} documents from PDF")
    return docs

def create_splitter(docs):
    logging.info("Splitting documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(docs)
    logging.info(f"Created {len(split_docs)} chunks from documents")
    return split_docs

def create_vector_store(docs):
    logging.info("Creating FAISS vector store")
    vector_store = FAISS.from_documents(
        docs,
        embeddings,
    )
    vector_store.save_local(db_path)
    logging.info(f"Vector store saved to {db_path}")
    return vector_store

def retrieve_answer(query):
    logging.info(f"Retrieving documents for query: {query}")
    vector_store = FAISS.load_local(db_path, embeddings)
    docs = vector_store.similarity_search(query, k=3)
    logging.info(f"Retrieved {len(docs)} documents")
    return docs

def generate_answer(docs, query):
    logging.info("Generating answer using language model")
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )
    context = docs[0].page_content if docs else ""
    question = query
    formatted_prompt = prompt.format(context=context, question=question)

    response = model(formatted_prompt)
    logging.info("Answer generated")
    return response

def main():
    st.title("PDF Question Answering System")
    st.subheader("Upload your PDF file")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        os.makedirs("temp", exist_ok=True)
        temp_file_path = os.path.join("temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logging.info(f"Uploaded file saved to {temp_file_path}")

        docs = load_data(temp_file_path)
        split_docs = create_splitter(docs)

        if not os.path.exists(db_path):
            vector_store = create_vector_store(split_docs)
            st.success("Vector store created and saved.")
        else:
            vector_store = FAISS.load_local(db_path, embeddings)
            st.success("Vector store loaded.")
            logging.info("Vector store loaded from disk")

        query = st.text_input("Ask a question about the PDF:")
        if query:
            docs = retrieve_answer(query)
            response = generate_answer(docs, query)
            st.write(response.content)

if __name__ == "__main__":
    main()
