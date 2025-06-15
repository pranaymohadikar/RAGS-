
# import pandas as pd
# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate

# # Function to process CSV documents
# def get_csv_text(csv_docs):
#     text = ""
#     for csv in csv_docs:
#         df = pd.read_csv(csv)
#         for _, row in df.iterrows():
#             text += " ".join(str(cell) for cell in row) + "\n"
#     return text

# # Function to split the text into chunks
# def get_text_chunks_from_csv(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
#     chunks = text_splitter.split_text(text)
#     return chunks   

# # Function to process and save vector store
# def get_vector_store_from_csv(text_chunks):
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Function to create and load the QA chain
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. If the answer is not available in the context, say "Answer is not available in the context."
#     Context: {context}
#     Question: {question}
#     Answer:
#     """

#     llm = ChatOllama(model="llama3.2")
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

#     return chain

# # Function to handle user input and provide response
# def user_input(user_question):
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
#     # Load the FAISS vector store
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     # Search for relevant documents
#     docs = new_db.similarity_search(user_question)

#     # Get the QA chain
#     chain = get_conversational_chain()

#     # Get the answer from the chain using the documents found by the similarity search
#     response = chain(
#         {"input_documents": docs, "question": user_question},
#         return_only_outputs=True
#     )

#     # Return the chatbot response
#     return response["output_text"]

# # Main function for the Streamlit app
# def main():
#     st.set_page_config("Chat CSV")
#     st.header("Chat with CSV using Ollama")

#     # Initialize session state for conversation history
#     if "history" not in st.session_state:
#         st.session_state.history = []

#     # Sidebar for file upload and processing
#     with st.sidebar:
#         st.title("Menu:")
#         csv_docs = st.file_uploader("Upload your CSV Files", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_csv_text(csv_docs)
#                 text_chunks = get_text_chunks_from_csv(raw_text)
#                 get_vector_store_from_csv(text_chunks)
#                 st.success("CSV processed and vector store created!")

#     # Main Chat Interface
#     user_question = st.text_input("Ask a Question from the CSV Files:")

#     if user_question:
#         # Add the user's question to the conversation history
#         st.session_state.history.append(f"User: {user_question}")

#         # Get the chatbot response
#         answer = user_input(user_question)

#         # Add the response to the conversation history
#         st.session_state.history.append(f"Chatbot: {answer}")

#         # Display the conversation history
#         for message in st.session_state.history:
#             st.write(message)

#     # Option to clear the conversation history
#     if st.button("Clear Conversation"):
#         st.session_state.history = []
#         st.rerun()

# if __name__ == "__main__":
#     main()
    

import pandas as pd
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from collections import Counter

# Function to process CSV documents
def get_csv_text(csv_docs):
    text = ""
    for csv in csv_docs:
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            text += " ".join(str(cell) for cell in row) + "\n"
    return text

# Function to split the text into chunks
def get_text_chunks_from_csv(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to process and save vector store
def get_vector_store_from_csv(text_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create and load the QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, say "Answer is not available in the context."
    Context: {context}
    Question: {question}
    Answer:
    """
    llm = ChatOllama(model="llama3.2")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and provide response
def user_input(user_question):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

# Function to find peak dates and summarize messages
def find_peak_dates_and_summary(csv_docs, top_n=3):
    all_data = pd.concat([pd.read_csv(csv) for csv in csv_docs], ignore_index=True)

    date_col = "Created On (Posted On)"

    if date_col not in all_data.columns or "Message" not in all_data.columns:
        return []

    # Get top N dates with highest message counts
    date_counts = all_data[date_col].value_counts().nlargest(top_n)

    summaries = []
    for date, count in date_counts.items():
        messages_df = all_data[all_data[date_col] == date]

        # Treat each message as a separate document
        documents = [Document(page_content=msg) for msg in messages_df['Message'].astype(str).tolist()]

        # Summarize using LLM
        llm = ChatOllama(model="llama3.2")
        prompt_template = """
        You are given a list of user messages from a specific date.
        Your task is to summarize the key themes, topics, and sentiments expressed.

        Messages:
        {context}

        Summary:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        summary = chain({"input_documents": documents}, return_only_outputs=True)["output_text"]

        summaries.append((date, count, summary))

    return summaries

# Streamlit App
def main():
    st.set_page_config("Chat CSV")
    st.header("📊 Chat with CSV using Ollama")

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.title("📁 Menu")
        csv_docs = st.file_uploader("Upload your CSV Files", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if csv_docs:
                with st.spinner("Processing CSV files..."):
                    raw_text = get_csv_text(csv_docs)
                    text_chunks = get_text_chunks_from_csv(raw_text)
                    get_vector_store_from_csv(text_chunks)
                    st.success("✅ CSV processed and vector store created!")

        if st.button("Find Peak Dates"):
            if csv_docs:
                with st.spinner("Analyzing peak dates..."):
                    peaks = find_peak_dates_and_summary(csv_docs)
                    for date, count, summary in peaks:
                        st.markdown(f"**📅 Date:** {date}")
                        st.markdown(f"- 🧮 Count: {count}")
                        st.markdown(f"- 📝 Summary:**\n{summary}")
                        st.markdown("---")

    user_question = st.text_input("Ask a Question from the CSV Files:")

    if user_question:
        st.session_state.history.append(f"🧑‍💬 User: {user_question}")
        answer = user_input(user_question)
        st.session_state.history.append(f"🤖 Chatbot: {answer}")

    for message in st.session_state.history:
        st.write(message)

    if st.button("Clear Conversation"):
        st.session_state.history = []
        st.rerun()

if __name__ == "__main__":
    main()
   
    


