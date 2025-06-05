
#llms got older and now we should use chat models

# from langchain_ollama import OllamaLLM

# llm = OllamaLLM(model = "llama3.2")


# result = llm.invoke("what is the capital of India?")

# print(result)


#now for chat models

# from langchain_ollama import ChatOllama

# model = ChatOllama(model = "llama3.2", temperature = 1.8, max_completion_tokens = 100)#temperature is the parameter that control the cretivity of the result. 0-->lowerrst creativity(factual answers) and 2 is the hgh form of creativity(strorytellin and all)

# #max tokens limit the words for the results and help to reduce the cost of closed source model apis'

# result = model.invoke("suggest me a poem on rain in 5 lines")

# print(result.content)


#now fofr embeddings

# from langchain_ollama import OllamaEmbeddings

# embedding = OllamaEmbeddings(model = "nomic-embed-text")# it denotes that how much context you want to capture. more the dim more the capture of context it is not needed inn ollama but in open ai

# result = embedding.embed_query("delhi is capital of india")

# print(result)

#can be used for the docs also

from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model = "nomic-embed-text")# it denotes that how much context you want to capture. more the dim more the capture of context it is not needed inn ollama but in open ai

documents = [
    "delhi is capital of india",
    "im a good boy"
]

result = embedding.embed_documents(documents)

print(result)