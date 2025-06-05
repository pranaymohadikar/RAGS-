from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys
import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma

db = "./new_db_csv"
loader = CSVLoader(file_path="customer_service_messages.csv")
data = loader.load()
#print(data)

# Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
text_chunks = text_splitter.split_documents(data)

print(len(text_chunks))

# Download Sentence Transformers Embedding From Hugging Face
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
#Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./new_db")


llm = ChatOllama(model = "llama3.2")

vector_store = Chroma(persist_directory="./new_db_csv", embedding_function=embeddings)
    
retriever = vector_store.as_retriever()

qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

while True:
    chat_history = []
    query = input(f"Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa({"question":query, "chat_history":chat_history})
    print("Response: ", result['answer'])