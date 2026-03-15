from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
documents =[
    "Delhi is capital of India",
    "Paris is capital of France",
    "Beijing is capital of China",
    "Washington.DC is capital of United State"
]

result = embedding.embed_documents(documents)
print(str(result))

