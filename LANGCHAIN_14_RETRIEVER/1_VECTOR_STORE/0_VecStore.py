from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# Step 1: Your source documents
docs = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]


model_embedding = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=model_embedding,
    collection_name="my_collection"
)

retriever = vectorstore.as_retriever(search_kwargs={"k":1})

querry = "What is Chroma used for"

result =retriever.invoke(querry)

for i,doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n {doc.page_content} ---")




