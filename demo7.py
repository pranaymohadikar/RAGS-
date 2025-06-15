from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM with strict mode
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Load your CSV (replace with your file path)
df = pd.read_csv("ICICI_Bank_GBR.csv")  # Ensure columns: message, sentiment, date, etc.

def strict_csv_qa(question):
    """Answers questions about the CSV with ONLY factual responses"""
    response = llm.invoke([
        HumanMessage(content=f"""
        You are a CSV data lookup tool. Follow these rules:
        
        1. Data Context:
        - Columns: {df.columns.tolist()}
        - {df.head().to_csv(index=False)}
        
        
        2. Strict Rules:
        - Answer ONLY what is explicitly asked
        - No summaries, analysis, or extra information
        - Return raw data or simple counts
        - Format: "Answer: [response]"
        
        3. Examples:
        Question: "Messages on 2024-01-15"
        Answer: "Great product", "Late delivery"
        
        Question: "Count of positive sentiments"
        Answer: 42
        
        Current Question: {question}
        Answer: 
        """)
    ])
    return response.content

# Interactive session
print("CSV Q&A Tool (type 'exit' to quit)")
while True:
    user_input = input("\nYour question about the data: ")
    if user_input.lower() == 'exit':
        break
    answer = strict_csv_qa(user_input)
    print(answer)