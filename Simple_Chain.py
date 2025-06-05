from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model = "llama3.2")

prompt = PromptTemplate(
    template = "generate 5 facts about the {topic}"
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic":"blackhole"})

print(result)


chain.get_graph().print_ascii()