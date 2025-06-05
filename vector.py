# Import required libraries
from langchain_ollama import OllamaEmbeddings  # For generating embeddings using Ollama
from langchain_chroma import Chroma  # Vector store (ChromaDB) for storing and retrieving embeddings
from langchain_core.documents import Document  # LangChain Document class for structured text data
import os  # For file/directory operations
import pandas as pd  # For reading CSV files

# Load the dataset containing restaurant reviews
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Initialize Ollama embeddings with the "nomic-embed-text" model (used to convert text into vectors)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Define the directory where ChromaDB will store the vector data
db_location = "./chrome_langchain_db"

# Check if the database already exists (if not, we'll add documents)
add_documents = not os.path.exists(db_location)

# If the database doesn't exist, prepare documents for storage
if add_documents:
    documents = []  # List to store LangChain Document objects
    ids = []  # List to store unique IDs for each document
    
    # Iterate through each row in the DataFrame
    for i, row in df.iterrows():
        # Create a Document object containing:
        # - page_content: Combined "Title" and "Review" text
        # - metadata: Additional info like "Rating" and "Date"
        # - id: A unique identifier (using row index)
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))  # Store the ID
        documents.append(document)  # Store the Document

# Initialize Chroma vector store:
# - collection_name: Name of the collection in ChromaDB
# - persist_directory: Where the database is stored
# - embedding_function: The embedding model (Ollama in this case)
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# If this is a new database, add the documents to Chroma
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)  # Insert documents with their IDs

# Create a retriever for querying the vector store:
# - search_kwargs={"k": 5} means it retrieves the top 5 most relevant documents
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)