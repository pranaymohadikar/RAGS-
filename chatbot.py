from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatOllama(model = "llama3.2")

#       
    
# it doesnot recognize the context of the previous chat so we need to add the chatbot history as a list to get the context

# while True:
#     user_input = input("You: ")
#     if user_input == "exit":
#         break
#     result = model.invoke(user_input)
#     print("AI: ",result.content)

#now we will add chat history to add context

chat_history = [
    SystemMessage(content = "You are a chatbot")
]


while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content = user_input))
    if user_input == "exit":
        break
    
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("AI: ",result.content)


print(chat_history)

'''
['hi', 'How can I help you today?', 'which is greater 2 or 4', "Hi there! I'm happy to chat with you.\n\nAs for your question, the number 4 is greater than 2. Would you like to ask another question or need help with something else?", 'multiply the greater one with 5', "I'd be happy to chat with you too!\n\nOkay, so if 4 is greater than 2, and we multiply 4 by 5... that would be:\n\n4 × 5 = 20\n\nSo the result is 20! Is there anything else I can help you with?", '\texit', "Nice math exercise!\n\nI'm glad I could start a fun conversation. To answer your exit command, I'll just say goodbye and thank you for chatting with me.\n\nGoodbye for now!", 'exit'] '''

# here we dont know whose message is from user and whcoh message is from ai





