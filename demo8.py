

import pandas as pd
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from dateutil import parser
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
def find_peak_dates_and_summary(csv_docs, top_n=1):
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
        Your task is to summarize the key themes, topics, and sentiments expressed provide me a consice summary of the messages.

        Messages:
        {context}

        Summary:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        summary = chain({"input_documents": documents}, return_only_outputs=True)["output_text"]

        summaries.append((date, count, summary))

    return summaries

#=================================Added function for the particular date for the summary added on 16-6-25========================================
def get_date_summary( csv_docs):
    all_data = pd.concat([pd.read_csv(csv) for csv in csv_docs], ignore_index=True)
    date_col = "Created On (Posted On)"
    if date_col not in all_data.columns or "Message" not in all_data.columns:
        return "No data available for the specified date."
    #date = pd.to_datetime(date, errors="coerce").date()
    date_counts = all_data[date_col]
    #print(date_counts)
    for date in date_counts:
        #print(date)
        
        messages_df = all_data[all_data[date_col] == date]
        print(len(messages_df))
        if messages_df.empty:
            return "no messages found"
        documents = [Document(page_content=msg) for msg in messages_df['Message'].astype(str).tolist()]

        llm = ChatOllama(model = "llama3.2")
        prompt_template = """
        You are given a list of user messages from a specific date.
        Your task is to summarize the key themes, topics, and sentiments expressed provide me a consice summary of the messages.
        Messages: {context}

        Summary: 
    """

        prompt = PromptTemplate(template = prompt_template, input_variables=["context"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt = prompt)
        summary = chain({"input_documents": documents}, return_only_outputs=True)['output_text']
        return summary if summary else "no summary found"


#=================================Added function for the particular date for the summary added on 16-6-25========================================
# -------------------------------
# Handle user input & integrate other functions to handle things
# -------------------------------
def user_input(user_question, csv_docs=None):
    peak_keywords = ["peak date", "top date", "most messages", "highest message", "most active day"]
    if any(keyword in user_question.lower() for keyword in peak_keywords):
        if csv_docs:
            peaks = find_peak_dates_and_summary(csv_docs)
            if not peaks:
                return "Couldn't find peak dates due to missing data."
            response = "**Peak Dates with Highest Message Counts:**\n\n"
            for date, count, summary in peaks:
                response += f"**Date:** {date}\n **Count:** {count}\n **Summary:** {summary}\n\n---\n"
            return response
        else:
            return "Please upload CSV files first to analyze peak dates."
        
    
#==============================Added logic to get the date wise summary on 16-6-25========================================
    date_summary_keywords = [
    "summary on", "summary for", "recap on", "recap for",
    "events on", "updates on", "daily summary", "day summary",
    "what happened on", "highlights of", "notes from", "log for",
    "activity on", "meeting notes", "status update", "timeline for"]
    if any(keyword in user_question.lower() for keyword in date_summary_keywords):
    #     if csv_docs:
    # # Extract date from user question
    #         try:
    #     # Extract the last few words and attempt to parse them as a date
    #             words = user_question.lower().split()
    #     # Try parsing using full question (or a fallback slice of last 3 words)
    #             date_str = " ".join(words[-3:])  # e.g., "31st May", "May 31", "June 1st"
    #             date = parser.parse(date_str, fuzzy=True).date()
    #         except Exception:
    #             return "Please provide a valid date, e.g., 'May 31' or '31st May'."

    #         daily_summary = get_date_summary(date, csv_docs)
    #         if not daily_summary:
    #             return f"Couldn't find messages for the specified date: {date}."
    
    #         response = f"**Summary of {date.strftime('%B %d, %Y')}**\n\n{daily_summary}\n\n---\n"
    #         return response
            if csv_docs:
                date =  user_question.split()[-1]
                date = date.strip()
                if not pd.to_datetime(date, errors="coerce"):
                    return "Please provide a valid date in the format YYYY-MM-DD."
                #date = pd.to_datetime(date, errors="coerce").date()
                
                daily_summary = get_date_summary(csv_docs)
                if not daily_summary:
                    return f"Couldn't find the messages for the specified date."
                response = f"**Summary of {date}**\n\n"
                for  summary in daily_summary:
                    response = f"**Summary:** {summary}\n\n---\n"
                return response
            else:
                return "Please upload CSV files first to analyze day."
            
 #==============================Added logic to get the date wise summary on 16-6-25========================================       

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k = 10)

    # # Metadata visualization
    # meta_info = "\n\n".join(
    #     f" **Location:** {doc.metadata.get('location')} |  **Date:** {doc.metadata.get('date')} |  **Sentiment:** {doc.metadata.get('sentiment')}\n {doc.page_content}"
    #     for doc in docs
    # )

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)["output_text"]

    #return f"** Retrieved Info:**\n\n{meta_info}\n\n** Answer:**\n{response}"
    return response
# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.set_page_config("Chat CSV")
    st.header("Chat with CSV using Ollama")

    if "history" not in st.session_state:
        st.session_state.history = []

    csv_docs = st.file_uploader("Upload your CSV Files", accept_multiple_files=True)

    if st.button("Process CSV Files"):
        if csv_docs:
            with st.spinner("Processing CSV..."):
                documents = get_documents_from_csv(csv_docs)
                get_vector_store_from_documents(documents)
                st.success(" CSV processed and vector store created!")

    user_question = st.text_input(" Ask a Question from the CSV Files:")

    if user_question:
        st.session_state.history.append(f" **User:** {user_question}")
        answer = user_input(user_question, csv_docs=csv_docs)
        st.session_state.history.append(f" **Bot:** {answer}")

    for message in st.session_state.history:
        st.markdown(message)

    if st.button(" Clear Conversation"):
        st.session_state.history = []
        st.rerun()

if __name__ == "__main__":
    main()