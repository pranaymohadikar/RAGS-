
#strinig output parser is used to extract the string data from the llm output and get the chains work in a single go for the different templates 

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model = "llama3.2")

#1st prompt for the detailed information


template1 = PromptTemplate(
    template = "write a detailed report on {topic}",
    input_variable = ["topic"]
)

#2nd promt for the summary
template2 = PromptTemplate(
    template = "write a 5 line summary on {text}",
    input_variable = ["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({ "topic":"black hole"
})

print(result)



# we have another things json output parser which will give data in json format but we cant force the user defined  schema so to overcome this we use structured output parser.



