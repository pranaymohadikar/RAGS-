import pandas as pd
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from dateutil import parser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", default=None)


##==========================converted for the excel files on 18-6-25========================================== 
# # -------------------------------
# Load and prepare documents with metadata
# -------------------------------
def get_documents_from_excel(excel_docs):
    documents = []
    for excel in excel_docs:
        df = pd.read_excel(excel)
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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
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
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# -------------------------------
# Find peak message dates and summarize them
# -------------------------------
def find_peak_dates_and_summary(excel_docs, top_n=1):
    all_data = pd.concat([pd.read_excel(excel) for excel in excel_docs], ignore_index=True)
    date_col = "Created On (Posted On)"

    if date_col not in all_data.columns or "Message" not in all_data.columns:
        return []

    date_counts = all_data[date_col].value_counts().nlargest(top_n)

    summaries = []
    for date, count in date_counts.items():
        messages_df = all_data[all_data[date_col] == date]
        documents = [Document(page_content=msg) for msg in messages_df['Message'].astype(str).tolist()]

        llm = ChatOpenAI(model="gpt-4o-mini")
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
def get_date_summary( date, excel_docs):
    all_data = pd.concat([pd.read_excel(excel) for excel in excel_docs], ignore_index=True)
    date_col = "Created On (Posted On)"
    if date_col not in all_data.columns or "Message" not in all_data.columns:
        return "No data available for the specified date."
    
    #added logic to get the latest year summary of any data availsable in the data on 21-6-25
    
    all_data[date_col] = pd.to_datetime(all_data[date_col], errors="coerce", dayfirst=True)
    #print(all_data[date_col])
    
    date = pd.to_datetime(date, errors="coerce", dayfirst=True)
    
    target_day = date.day
    target_month = date.month
    
    # Add helper columns 
    all_data["day"] = all_data[date_col].dt.day
    all_data["month"] = all_data[date_col].dt.month
    all_data["year"] = all_data[date_col].dt.year
    
    day_month_df = all_data[(all_data['day']==target_day) & (all_data['month'] == target_month)]
    
    if day_month_df.empty:
        return f" No messages found in any year."
    
    available_years = sorted(day_month_df["year"].unique())

    year_use = max(available_years)
    #date_counts = all_data[date_col]
    #print(date_counts)
    print(date)

####============================================need to remove this for loop 18-6-25==============================    
    #for date in date_counts:
        #print(date)
####============================================need to remove this for loop 18-6-25==============================            
    messages_df = day_month_df[day_month_df["year"] == year_use]  #just needed to cconvert the date to string
    print(len(messages_df))
    #print(date)
    if messages_df.empty:
        return "no messages found"
    documents = [Document(page_content=msg) for msg in messages_df['Message'].astype(str).tolist()]

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt_template = """
    You are given a list of user messages from a specific date and consider the latest year mentioned in the given data.
    Your task is to summarize the key themes, topics, and sentiments expressed provide me a consice summary of the messages.
    Messages: {context}

    Summary: 
"""

    prompt = PromptTemplate(template = prompt_template, input_variables=["context"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt = prompt)
    summary = chain({"input_documents": documents}, return_only_outputs=True)['output_text']
    return summary if summary else "no summary found"


#=================================Added function for the summarization for the date range for the summary added on 19-6-25========================================


def get_date_range_summary( start_date, end_date, excel_docs):
    all_data = pd.concat([pd.read_excel(excel) for excel in excel_docs], ignore_index=True)
    date_col = "Created On (Posted On)"
    if date_col not in all_data.columns or "Message" not in all_data.columns:
        return "No data available for the specified date."
    start_date = pd.to_datetime(start_date, errors="coerce", dayfirst=True).date()
    end_date = pd.to_datetime(end_date, errors="coerce", dayfirst=True).date()
    #date_counts = all_data[date_col]
    #print(date_counts)
    #print(start_date, end_date)

    masking = (all_data[date_col] >= str(start_date)) & (all_data[date_col] <= str(end_date))
    messages_df =all_data[masking]
    #messages_df = all_data[all_data[date_col] == str(date)]  #just needed to cconvert the date to string
    #print(len(messages_df))
    #print(date)
    #print(start_date, end_date)
    if messages_df.empty:
        return "no messages found"
    documents = [Document(page_content=msg) for msg in messages_df['Message'].astype(str).tolist()]

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt_template = """
    You are given a list of user messages from a specific date and consider the latest year mentioned in the given data.
    Your task is to summarize the key themes, topics, and sentiments expressed provide me a consice summary of the messages.
    Messages: {context}

    Summary: 
"""

    prompt = PromptTemplate(template = prompt_template, input_variables=["context"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt = prompt)
    summary = chain({"input_documents": documents}, return_only_outputs=True)['output_text']
    return summary if summary else "no summary found"


#=================================Added function for the summarization for the date range for the summary added on 19-6-25========================================


# -------------------------------
# Handle user input  + peak date detection+ other things
# -------------------------------
def user_input(user_question, excel_docs=None):
    peak_keywords = ["peak date", "top date", "most messages", "highest message", "most active day"]
    if any(keyword in user_question.lower() for keyword in peak_keywords):
        if excel_docs:
            peaks = find_peak_dates_and_summary(excel_docs)
            if not peaks:
                return "Couldn't find peak dates due to missing data."
            response = "**Peak Dates with Highest Message Counts:**\n\n"
            for date, count, summary in peaks:
                response += f"**Date:** {date}\n **Count:** {count}\n **Summary:** {summary}\n\n---\n"
            return response
        else:
            return "Please upload excel files first to analyze peak dates."
        
    
#==============================Added logic to get the date wise summary on 16-6-25========================================
    date_summary_keywords = [
    "summary on", "summary for", "recap on", "recap for",
    "events on", "updates on", "daily summary", "day summary",
    "what happened on", "highlights of", "notes from", "log for",
    "activity on", "meeting notes", "status update", "timeline for"]
#==============================New logic to get the date wise summary on 18-6-25========================================  

    if any(keyword in user_question.lower() for keyword in date_summary_keywords):
        if excel_docs:
    # Extract date from user question
            try:
        # Extract the last few words and attempt to parse them as a date
                words = user_question.lower().split()
        # Try parsing using full question (or a fallback slice of last 3 words)
                date_str = " ".join(words[-3:])  # e.g., "31st May", "May 31", "June 1st"
                date = parser.parse(date_str, fuzzy=True,  dayfirst=True).date()
                #print(date)
            except Exception:
                return "Please provide a valid date, e.g., 'May 31' or '31st May'."
            if date:
                daily_summary = get_date_summary( date, excel_docs)
            if not daily_summary:
                return f"Couldn't find messages for the specified date: {date}."
    
            response = f"**Summary of {date.strftime('%B %d')}**\n\n{daily_summary}\n\n---\n"
            return response
        
#==============================New logic to get the date wise summary on 18-6-25======================================== 

#======================================new logic to get summarization of rangewise summary on 19-6-25============================

    date_range_summary_keywords = [
    "summary from", "summary between", "recap from", "recap between",
    "summary for the week", "summary for the month", 
    "summarize messages between", "summarize feedback from", 
    "feedback from", "update from", "insights between", "activity from", 
    "events between", "review from", "data between", 
    "summarize last week", "summarize last month", 
    "overview from", "overview between", "messages from"
    ]


    if any(keyword in user_question.lower() for keyword in date_range_summary_keywords):
        if excel_docs:
            try:
                #extract the last few words and attempt TO PARSE THEN AS AS DATE
                words = user_question.lower().split()
                date_str = " ".join(words[-5:])
                date_range = date_str.split("to") if "to" in date_str else date_str.split("between")
                start_date = parser.parse(date_range[0].strip(), fuzzy = True,  dayfirst=True).date()
                end_date = parser.parse(date_range[1].strip(), fuzzy = True, dayfirst=True).date()
                #print(start_date, end_date)

                if start_date and end_date:
                    range_summary = get_date_range_summary(start_date= start_date, end_date=end_date, excel_docs=excel_docs)
                    if not range_summary:
                        return f"Couldn't find messages for the specified date range: {start_date} to {end_date}."

            except Exception as e:
                return "Please provide a valid date range, e.g., 'from May 1 to May 31' or 'between June 1 and June 7'."
            
            
    
            response = f"**Summary between {start_date.strftime('%B %d, %Y')} and {end_date.strftime('%B %d, %Y')}**\n\n{range_summary}\n\n---\n"
            return response

#======================================new logic to get summarization of rangewise summary on 19-6-25============================

      
######==============================Obsolete logic to get the date wise summary on 18-6-25========================================        
            # if csv_docs:
            #     date =  user_question.split()[-1]
            #     date = date.strip()
            #     if not pd.to_datetime(date, errors="coerce"):
            #         return "Please provide a valid date in the format YYYY-MM-DD."
            #     date = pd.to_datetime(date, errors="coerce").date()
                
            #     daily_summary = get_date_summary(csv_docs)
            #     if not daily_summary:
            #         return f"Couldn't find the messages for the specified date."
            #     response = f"**Summary of {date}**\n\n"
            #     for  summary in daily_summary:
            #         response = f"**Summary:** {summary}\n\n---\n"
            #     return response
            # else:
            #     return "Please upload CSV files first to analyze day."
######==============================Obsolete logic to get the date wise summary on 18-6-25========================================  


            
 #==============================Added logic to get the date wise summary on 16-6-25========================================       

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
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
    st.set_page_config("Chat Excel")
    st.header("Chat with Excel using Ollama")

    if "history" not in st.session_state:
        st.session_state.history = []

    excel_docs = st.file_uploader("Upload your excel Files", accept_multiple_files=True)

    if st.button("Process Excel Files"):
        if excel_docs:
            with st.spinner("Processing excel..."):
                documents = get_documents_from_excel(excel_docs)
                get_vector_store_from_documents(documents)
                st.success(" CSV processed and vector store created!")

    user_question = st.text_input(" Ask a Question from the Excel Files:")

    if user_question:
        st.session_state.history.append(f" **User:** {user_question}")
        answer = user_input(user_question, excel_docs=excel_docs)
        st.session_state.history.append(f" **Bot:** {answer}")

    for message in st.session_state.history:
        st.markdown(message)

    if st.button(" Clear Conversation"):
        st.session_state.history = []
        st.rerun()

if __name__ == "__main__":
    main()