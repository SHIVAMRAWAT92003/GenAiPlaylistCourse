import shutil, os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# -----------------------------
# Clean previous DB
# -----------------------------
if os.path.exists("my_chroma_db"):
    shutil.rmtree("my_chroma_db")

# -----------------------------
# Create documents
# -----------------------------
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )
]

ids = ["kohli", "rohit", "dhoni", "bumrah", "jadeja"]

# -----------------------------
# Initialize Chroma
# -----------------------------
vector_store = Chroma(
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    persist_directory="my_chroma_db",
    collection_name="Sample"
)

# -----------------------------
# Add documents
# -----------------------------
vector_store.add_documents(docs, ids=ids)

# -----------------------------
# View documents with embeddings
# -----------------------------
all_docs = vector_store.get(include=['embeddings','documents','metadatas'])
print("\nAll documents in DB with embeddings (first 5 values shown):")
for embed, doc, meta in zip(all_docs['embeddings'], all_docs['documents'], all_docs['metadatas']):
    print(f"Embedding: {embed[:5]} ...\nContent: {doc}\nMetadata: {meta}\n")

# -----------------------------
# Similarity search
# -----------------------------
print("\nSimilarity Search for 'Who among these are a bowler?':")
for r in vector_store.similarity_search("Who among these are a bowler?", k=1):
    print(f"Content: {r.page_content}\nMetadata: {r.metadata}\n")

# -----------------------------
# Similarity search with score
# -----------------------------
print("\nSimilarity Search with Score:")
for r, score in vector_store.similarity_search_with_score("Who among these are a bowler?", k=1):
    print(f"Score: {score:.4f}\nContent: {r.page_content}\nMetadata: {r.metadata}\n")

# -----------------------------
# Metadata filtering
# -----------------------------
print("\nMetadata Filtered Search for 'Who is the captain of CSK?':")
for doc, score in vector_store.similarity_search_with_score(
        "Who is the captain?", filter={"team": "Chennai Super Kings"}):
    print(f"Score: {score:.4f}\nContent: {doc.page_content}\nMetadata: {doc.metadata}\n")

# -----------------------------
# Update a document
# -----------------------------
vector_store.delete(ids=["kohli"])
vector_store.add_documents(
    [Document(
        page_content="Virat Kohli, former RCB captain, holds IPL run records.",
        metadata={"team": "Royal Challengers Bangalore"}
    )],
    ids=["kohli"]
)

# -----------------------------
# View all documents after update
# -----------------------------
all_docs = vector_store.get(include=['embeddings','documents','metadatas'])
print("\nAll documents after update (first 5 embedding values shown):")
for embed, doc, meta in zip(all_docs['embeddings'], all_docs['documents'], all_docs['metadatas']):
    print(f"Embedding: {embed[:5]} ...\nContent: {doc}\nMetadata: {meta}\n")

# -----------------------------
# Delete a document
# -----------------------------
vector_store.delete(ids=["kohli"])

# -----------------------------
# Remaining documents
# -----------------------------
all_docs = vector_store.get(include=['embeddings','documents','metadatas'])
print("\nRemaining documents after deletion (first 5 embedding values shown):")
for embed, doc, meta in zip(all_docs['embeddings'], all_docs['documents'], all_docs['metadatas']):
    print(f"Embedding: {embed[:5]} ...\nContent: {doc}\nMetadata: {meta}\n")
