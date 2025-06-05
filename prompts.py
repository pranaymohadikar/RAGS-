from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

st.header('research tool')

paper_input = st.selectbox("select the research paper name",[
    "select....","Attention is all you need","BERT","GPT-3",
    "diffusion models","WAQF Bill"
])

style_input = st.selectbox("select the explanation style",[
    "Beginner-friendly","Technical","case-oriented",
    "Mathematical"
])

length_input = st.selectbox("select the explanation length",[
    "1-2 paragraph","4-5 paragraphs","detailed explanation"
])

#template

template = load_prompt("template.json")

# prompt = template.invoke({
#     "paper_input":paper_input,
#     "style_input": style_input,
#     "length_input": length_input
# })

model = ChatOllama(model = "llama3.2")
#user_input = st.text_input("enter as prompt")

# if st.button("Summarize"):
#     result = model.invoke(prompt)
#     st.write(result.content)
    
    
    
#here im invoking prompt and model differently


#use chain

chain = template | model

if st.button("Summarize"):
    result = chain.invoke({
    "paper_input":paper_input,
    "style_input": style_input,
    "length_input": length_input
}) 
    st.write(result.content)
