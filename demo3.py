import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma

# Ingest the PDF document and store it in a vector store (Chroma)
def ingest():
    loader = PyPDFLoader("Joel Grus - Data Science from Scratch_ First Principles with Python-O'Reilly Media (2015).pdf")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(pages)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./new_db")
    print(f"Documents ingested. Total chunks: {len(chunks)}")

# Initialize the retrieval-augmented generation (RAG) chain
def rag():
    model = ChatOllama(model="llama3.2", temperature=0)
    
    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a friendly assistant. Answer the question based only on the following context. 
        If you don't know the answer, then reply, No Context available for this question {input}. [/Instructions] </s> 
        [Instructions] Question: {input} 
        #Context: {context} 
        Answer: [/Instructions]
        """
    )
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    vector_store = Chroma(persist_directory="./new_db", embedding_function=embeddings)
    
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.1
        }
    )
    
    document_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    return chain

# Function to summarize the conversation history
def summarize(history: str):
    chain = rag()
    query = f"Please summarize the following conversation: {history}"
    result = chain.invoke({"input": query, "history": history})
    
    # Print the summary of the conversation
    print("Conversation Summary:")
    print(result["answer"])

# Function to interact with the chatbot
def ask(query: str, history: str):
    # Pass the query and history to the chain
    chain = rag()
    result = chain.invoke({"input": query, "history": history})
    
    # Print the context (sources) for the answer
    print(f"Context: {result['context']}")
    
    # Print the generated answer
    print(f"Answer: {result['answer']}")
    
    # Update history with the current query and answer
    history += f"Question: {query}\nAnswer: {result['answer']}\n"
    return history

# Function to run the chatbot interaction
def run_chatbot():
    # Initialize the history variable
    history = ""
    
    print("Welcome to the chatbot! Type 'exit' to quit or 'summarize' to summarize the conversation.")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # If the user types 'exit', break the loop
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        # If the user asks for a summary of the conversation
        elif user_input.lower() == "summarize":
            summarize(history)
        else:
            # Pass the input along with the history to the ask function
            history = ask(user_input, history)

if __name__ == "__main__":
    # Optionally, ingest the data first (can be skipped if data is already ingested)
    # ingest()

    # Start the chatbot interaction
    run_chatbot()
