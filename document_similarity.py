from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = OllamaEmbeddings(model = "nomic-embed-text")

documents = [
    "virat kohli is god",
    "virat kohli is good",
    "ranveer is bad",
    "raka is goat"
]

query = "tell me about rakazone"
doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)


#now need to check the similarity we need to to have 2 2d lists as input

result = cosine_similarity([query_embedding],doc_embedding)[0] #convert into 1d list

index, score = sorted(list(enumerate(result)), key = lambda x:x[1])[-1]
''' so tha it will give me the index to the score and easy for us to sort the score the document else if we sort without it the popsitioning will become here and there. here -1 for getting the higheest as the sorting is ascending'''

print(f"for the query : {query}\n {documents[index]} \n with similarity score : {score}")



#when you get the answers it will show the list with the similarity score with every doc

#[[0.8355693  0.44323146 0.43014553 0.40653309]] so here it is showng that the query is most similar to the first doc with 83%

