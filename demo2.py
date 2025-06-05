import pandas as pd
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama


def read_data(file):
    if file.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


# Initialize chat history
chat_history = []

# Initialize df
df = None


def process_query(file_path, user_prompt):
    # Read the file
    global df
    df = read_data(file_path)
    
    # Add user's message to chat history
    chat_history.append({"role": "user", "content": user_prompt})
    
    # Load the LLM
    llm = ChatOllama(model="ollama3.2", temperature=0)

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        *chat_history
    ]

    response = pandas_df_agent.invoke(messages)

    assistant_response = response["output"]

    chat_history.append({"role": "assistant", "content": assistant_response})
    
    return assistant_response


# Example usage:
response = process_query("E:\KaaM\Langchain_RAG\CSV_RAG\sentiment.csv", "summarize the positive conversaions")
print(response)