from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable  import RunnableParallel

model1 = ChatOllama(model = "llama3.2")
model2 = ChatOllama(model = "llama3.2")


prompt1 = PromptTemplate(
    template = "generate a short and simplae notes from the followiwng text \n {text}",
    input_variable = ['text']
)

prompt2 = PromptTemplate(
    template = "generate a 5 short question and answers from the followiwng text \n {text}",
    input_variable = ['text']
)

prompt3 = PromptTemplate(
    template = "merge the provided notes and quiz into the single doc \n notes-->{notes} and quiz-->{quiz}",
    input_variable = ['notes','quiz']
    
)


parser = StrOutputParser()


#we need to create a paralle chain with the help of runnable parallel

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})


merge_chain = prompt3 | model1 | parser


chain = parallel_chain | merge_chain

result = chain.invoke({'text': '''
              A black hole is a massive, compact astronomical object so dense that its gravity prevents anything from escaping, even light. Albert Einstein's theory of general relativity predicts that a sufficiently compact mass will form a black hole.[2] The boundary of no escape is called the event horizon. A black hole has a great effect on the fate and circumstances of an object crossing it, but has no locally detectable features according to general relativity.[3] In many ways, a black hole acts like an ideal black body, as it reflects no light.[4][5] Quantum field theory in curved spacetime predicts that event horizons emit Hawking radiation, with the same spectrum as a black body of a temperature inversely proportional to its mass. This temperature is of the order of billionths of a kelvin for stellar black holes, making it essentially impossible to observe directly.
              '''})


print(result)

chain.get_graph().print_ascii()



