from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=5, lang='en')

querry = 'The geopolitical history of India and Pakisthan from the perspective of India'

docs = retriever.invoke(querry)

# print(docs)

for i,doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n {doc.page_content} ---")


