from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import pandas as pd

# 1. Load and prepare data
df = pd.read_csv("sentiment.csv")  # Contains 'messages' and 'sentiment' columns
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db_location = "./sentiment_analysis_db"

# 2. Create/load vector store
if not os.path.exists(db_location):
    documents = [
        Document(
            page_content=row["tweets"],
            metadata={"sentiment": row["sentiment"], 
                      "id": str(i)}
        )
        for i, row in df.iterrows()
    ]
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_location
    )
else:
    vector_store = Chroma(
        persist_directory=db_location,
        embedding_function=embeddings
    )

# 3. Summarization setup
llm = OllamaLLM(model="llama3.2")
summarize_template = """
Analyze these customer messages and provide:
1. Overall sentiment summary (positive/neutral/negative distribution)
2. Key themes mentioned
3. Notable specific feedback

Messages:
{formatted_messages}

Provide your analysis in this format:
[Overall Sentiment]: 
[Key Themes]: 
[Notable Feedback]: 
"""
summarize_prompt = ChatPromptTemplate.from_template(summarize_template)
summarize_chain = summarize_prompt | llm | StrOutputParser()

# 4. Analysis function
def analyze_conversation(query=None):
    # Get all messages (or filtered by query)
    if query:
        docs = vector_store.similarity_search(query, k=20)
    else:
        docs = vector_store.get()["documents"]
        docs = [Document(page_content=doc, metadata={}) for doc in docs]
    
    # Calculate sentiment distribution
    sentiments = [doc.metadata.get("sentiment", "unknown") for doc in docs]
    sentiment_counts = {
        "positive": sentiments.count("positive"),
        "neutral": sentiments.count("neutral"),
        "negative": sentiments.count("negative")
    }
    
    # Format messages for summarization
    formatted_msgs = "\n".join(
        f"- {doc.page_content} (Sentiment: {doc.metadata.get('sentiment', 'unknown')})"
        for doc in docs[:50]  # Limit to first 50 for token constraints
    )
    
    # Generate summary
    summary = summarize_chain.invoke({"formatted_messages": formatted_msgs})
    
    return {
        "sentiment_distribution": sentiment_counts,
        "summary": summary,
        "sample_messages": [doc.page_content for doc in docs[:3]]  # Sample messages
    }

# 5. Example usage
if __name__ == "__main__":
    # Get overall summary
    print("\n=== FULL CONVERSATION SUMMARY ===")
    full_analysis = analyze_conversation()
    print(f"Sentiment Distribution: {full_analysis['sentiment_distribution']}")
    print(f"\nSummary:\n{full_analysis['summary']}")
    print(f"\nSample Messages: {full_analysis['sample_messages']}")
    
    # Get summary about specific topic
    print("\n=== DELIVERY-RELATED SUMMARY ===")
    delivery_analysis = analyze_conversation("delivery")
    print(f"Delivery Sentiment: {delivery_analysis['sentiment_distribution']}")
    print(f"\nSummary:\n{delivery_analysis['summary']}")