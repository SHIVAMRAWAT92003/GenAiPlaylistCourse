from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [

    "Delhi is capital of India",
    "Paris is capital of France",
    "Beijing is capital of China",
    "Washington.DC is capital of United State"

]

querry = "Tell me about Delhi"


docs_embedding = embedding.embed_documents(documents)
querry_embedding = embedding.embed_query(querry)


scores = cosine_similarity([querry_embedding],docs_embedding)[0] 

index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(querry)
print(documents[index])
print("Similarity Score is ",score)
