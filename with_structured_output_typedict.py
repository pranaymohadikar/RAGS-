from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated

model = ChatOllama(model = "llama3.2")


#schema to define to get the structured output in a defined way

class Review(TypedDict):
    summary: Annotated[str, "detailed summary of the review"]
    sentiment: Annotated[str, "return the sentiment of the review"]
    key_themes: Annotated[list[str], "write down all the key themes of te review"]    
    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke('''
                                 I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.Pros:Insanely powerful processor (great for gaming and productivity)Stunning 200MP camera with incredible zoom capabilitiesLong battery life with fast chargingS-Pen support is unique and useful
                                 
Review by Pranay
                                 ''')


print(result)

#here we ar egettign results without any prompt becuase behind the scenes it wil generate a prompt and generate summary and provide the sentimment and return a json output
