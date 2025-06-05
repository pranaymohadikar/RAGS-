from langchain_ollama import ChatOllama
import pandas as pd
from langchain.prompts import PromptTemplate


df = pd.read_excel("conversations_tory.xlsx")
model = ChatOllama(model = "llama3.2", temperature = 0.1)


chat_prompt = PromptTemplate(
    
    template = """
        
        You are analyzing a social media message about a fashion/luxury product.
Choose the best general category that fits the message context. Choose only one from:
- Size
- Style
- Availability
- Product Quality
- Appreciation
- Criticism

Message: "{conversations}"
Respond with only the best category name.


Choose from the options or suggest a concise, analytics-friendly subcategory.

Availability:
- Out of Stock, Restock Updates, Limited Size Availability, Online vs In-store Discrepancy
Criticism:
- Uncomfortable Fit, Poor Durability, Bad Sizing/Fit, Low Quality Material, Lack of Originality, Negative Purchase Experience, expensive or overpriced
Product Quality:
- Discomfort/Wear Issues, Outdated Style, Durability Issues, Low Quality Material
Size:
- Narrow Fit, Wrong Size, Limited Size Options, Lack of Half Sizes
Style:
- Casual, Office, Chic, Sporty, Occasion, Party, Vintage
Appreciation:
- Product Quality, Style/Design, Comfort, Fashion Appeal, Value for Price, Luxury Appeal

Message: "{conversations}"

Format your output as:
    Category: <category>
    Sub-Category: <sub-category>
"""
        
    
)

chain = chat_prompt | model



def get_categories(convo):
    output = chain.invoke({"conversations": convo}).content
    print("RAW OUTPUT:\n", output)
    try:
        category = output.split("Category:")[1].split("Sub-Category:")[0].strip()
        sub_category = output.split("Sub-Category:")[1].strip()
        return category, sub_category
    except:
        return "Unknown", "Unknown"
    

df[['Category', 'Sub-Category']] = df['Mesage'].apply(
    lambda x: pd.Series(get_categories(x))
)

df.to_excel("categorized_conversations_tory.xlsx", index=False)