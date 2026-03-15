from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()
embedding = OpenAIEmbeddings(model='',dimensions=32)

documents = [
    "Delhi is capital of India",
    "Paris is capital of France",
    "Beijing is capital of China",
    "Washington.DC is capital of United State"
]


result=embedding.embed_documents(documents)

print(str(result))