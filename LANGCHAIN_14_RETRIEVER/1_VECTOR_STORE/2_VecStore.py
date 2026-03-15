from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import ChatHuggingFace ,HuggingFaceEmbeddings ,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()


all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]



embedding_model = HuggingFaceEmbeddings()

vectorstore = FAISS.from_documents(documents=all_docs,embedding=embedding_model)

# Simple Retriever 
similarity_retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})


hf_endpoint=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)


# Multiquery Retriever 
multiquerry_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k":3}),
    llm = ChatHuggingFace(llm = hf_endpoint)
)


querry = "How to improve energy levels and maintain balance?"

similarity_result = similarity_retriever.invoke(querry)
multiquerry_result = multiquerry_retriever.invoke(querry)


for i ,doc in enumerate(similarity_result):
    print(f"\n---Result {i+1}---")
    print(doc.page_content)


print("*"*150)

for i ,doc in enumerate(multiquerry_result):
    print(f"\n---Result {i+1}---")
    print(doc.page_content)














