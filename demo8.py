# import pandas as pd
# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain.docstore.document import Document

# # # Function to process CSV documents into raw text
# # def get_csv_text(csv_docs):
# #     text = ""
# #     for csv in csv_docs:
# #         df = pd.read_csv(csv)
# #         for _, row in df.iterrows():
# #             text += " ".join(str(cell) for cell in row) + "\n"
# #     return text


# def get_documents_from_csv(csv_docs):
#     documents = []
#     for csv in csv_docs:
#         df = pd.read_csv(csv)
#         for _, row in df.iterrows():
#             content = str(row["Message"])
#             metadata = {
#                 "date": row.get("Created On (Posted On)", "Unknown"),
#                 "location": row.get("Location", "Unknown"),
#                 "sentiment": row.get("Sentiment", "Unknown")
#             }
#             documents.append(Document(page_content=content, metadata=metadata))
#     return documents

# # Function to split text into manageable chunks
# def get_text_chunks_from_csv(doc):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return text_splitter.split_text(doc)

# # Store vector embeddings locally
# def get_vector_store_from_csv(text_chunks):
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Build the QA chain
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context.
#     If the answer is not available in the context, say "Answer is not available in the context."
    
#     Context: {context}
#     Question: {question}
#     Answer:
#     """
#     llm = ChatOllama(model="llama3.2")
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
#     return chain

# # Detect peak date queries and return summaries
# def find_peak_dates_and_summary(csv_docs, top_n=3):
#     all_data = pd.concat([pd.read_csv(csv) for csv in csv_docs], ignore_index=True)
#     date_col = "Created On (Posted On)"

#     if date_col not in all_data.columns or "Message" not in all_data.columns:
#         return []

#     # Normalize date format
#     all_data[date_col] = pd.to_datetime(all_data[date_col], errors="coerce").dt.date

#     date_counts = all_data[date_col].value_counts().nlargest(top_n)

#     summaries = []
#     for date, count in date_counts.items():
#         messages_df = all_data[all_data[date_col] == date]
#         documents = [Document(page_content=msg) for msg in messages_df["Message"].astype(str).tolist()]

#         llm = ChatOllama(model="llama3.2")
#         prompt_template = """
#         You are given a list of user messages from a specific date.
#         Your task is to summarize the key themes, topics, and sentiments expressed.

#         Messages:
#         {context}

#         Summary:
#         """
#         prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
#         chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
#         summary = chain({"input_documents": documents}, return_only_outputs=True)["output_text"]

#         summaries.append((date, count, summary))

#     return summaries

# # Handle user input intelligently
# def user_input(user_question, csv_docs=None):
#     peak_keywords = ["peak date", "top date", "most messages", "highest message", "most active day"]
#     if any(keyword in user_question.lower() for keyword in peak_keywords):
#         if csv_docs:
#             peaks = find_peak_dates_and_summary(csv_docs)
#             if not peaks:
#                 return "Couldn't find peak dates due to missing data."
#             response = "📊 **Peak Dates with Highest Message Counts:**\n\n"
#             for date, count, summary in peaks:
#                 response += f"📅 **Date:** {date}\n🧮 **Count:** {count}\n📝 **Summary:** {summary}\n\n---\n"
#             return response
#         else:
#             return "⚠️ Please upload CSV files first to analyze peak dates."

#     # Otherwise, use RAG
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain(
#         {"input_documents": docs, "question": user_question},
#         return_only_outputs=True
#     )
#     return response["output_text"]

# # Streamlit interface
# def main():
#     st.set_page_config("Chat CSV")
#     st.header("📄 Chat with Your CSV using Ollama 🧠")

#     if "history" not in st.session_state:
#         st.session_state.history = []

#     csv_docs = st.file_uploader("📁 Upload your CSV Files", accept_multiple_files=True)

#     if st.button("Submit & Process"):
#         if csv_docs:
#             with st.spinner("🔍 Processing CSV files..."):
#                 raw_text = get_documents_from_csv(csv_docs)
#                 text_chunks = get_text_chunks_from_csv(raw_text)
#                 get_vector_store_from_csv(text_chunks)
#                 st.success("✅ CSV processed and vector store created!")

#     user_question = st.text_input("💬 Ask a Question from the CSV Files:")

#     if user_question:
#         st.session_state.history.append(f"🧑‍💬 User: {user_question}")
#         answer = user_input(user_question, csv_docs)
#         st.session_state.history.append(f"🤖 Chatbot: {answer}")

#     for message in st.session_state.history:
#         st.write(message)

#     if st.button("🗑️ Clear Conversation"):
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

# -------------------------------
# Load and prepare documents with metadata
# -------------------------------
def get_documents_from_csv(csv_docs):
    documents = []
    for csv in csv_docs:
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            content = str(row.get("Message", ""))
            metadata = {
                "date": row.get("Created On (Posted On)", "Unknown"),
                "location": row.get("Administrative Area", "Unknown"),
                "sentiment": row.get("Sentiment", "Unknown")
            }
            documents.append(Document(page_content=content, metadata=metadata))
    return documents

# -------------------------------
# Create FAISS vector store from documents
# -------------------------------
def get_vector_store_from_documents(documents):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")

# -------------------------------
# Load LLM and QA Chain
# -------------------------------
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. and if someone ask you to summarize sentiment wise, use the sentiment provided in the data. dont assume sentiment itself.
    If the answer is not available in the context, say "Answer is not available in the context."
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    llm = ChatOllama(model="llama3.2")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# -------------------------------
# Find peak message dates and summarize them
# -------------------------------
def find_peak_dates_and_summary(csv_docs, top_n=3):
    all_data = pd.concat([pd.read_csv(csv) for csv in csv_docs], ignore_index=True)
    date_col = "Created On (Posted On)"

    if date_col not in all_data.columns or "Message" not in all_data.columns:
        return []

    date_counts = all_data[date_col].value_counts().nlargest(top_n)

    summaries = []
    for date, count in date_counts.items():
        messages_df = all_data[all_data[date_col] == date]
        documents = [Document(page_content=msg) for msg in messages_df['Message'].astype(str).tolist()]

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

# -------------------------------
# Handle user input & integrate metadata + peak date detection
# -------------------------------
def user_input(user_question, csv_docs=None):
    peak_keywords = ["peak date", "top date", "most messages", "highest message", "most active day"]
    if any(keyword in user_question.lower() for keyword in peak_keywords):
        if csv_docs:
            peaks = find_peak_dates_and_summary(csv_docs)
            if not peaks:
                return "Couldn't find peak dates due to missing data."
            response = "📊 **Peak Dates with Highest Message Counts:**\n\n"
            for date, count, summary in peaks:
                response += f"📅 **Date:** {date}\n🧮 **Count:** {count}\n📝 **Summary:** {summary}\n\n---\n"
            return response
        else:
            return "⚠️ Please upload CSV files first to analyze peak dates."

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k = 10)

    # Metadata visualization
    meta_info = "\n\n".join(
        f"📍 **Location:** {doc.metadata.get('location')} | 📅 **Date:** {doc.metadata.get('date')} | 😊 **Sentiment:** {doc.metadata.get('sentiment')}\n➡️ {doc.page_content}"
        for doc in docs
    )

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)["output_text"]

    #return f"**🔎 Retrieved Info:**\n\n{meta_info}\n\n**🤖 Answer:**\n{response}"
    return response
# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.set_page_config("Chat CSV with Metadata")
    st.header("🧠 Chat with CSV using Ollama + Metadata")

    if "history" not in st.session_state:
        st.session_state.history = []

    csv_docs = st.file_uploader("📁 Upload your CSV Files", accept_multiple_files=True)

    if st.button("Process CSV Files"):
        if csv_docs:
            with st.spinner("🔄 Processing CSV..."):
                documents = get_documents_from_csv(csv_docs)
                get_vector_store_from_documents(documents)
                st.success("✅ CSV processed and vector store created!")

    user_question = st.text_input("💬 Ask a Question from the CSV Files:")

    if user_question:
        st.session_state.history.append(f"🧑‍💬 **User:** {user_question}")
        answer = user_input(user_question, csv_docs=csv_docs)
        st.session_state.history.append(f"🤖 **Bot:** {answer}")

    for message in st.session_state.history:
        st.markdown(message)

    if st.button("🗑️ Clear Conversation"):
        st.session_state.history = []
        st.rerun()

if __name__ == "__main__":
    main()
