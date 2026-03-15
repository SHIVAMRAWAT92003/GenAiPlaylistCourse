from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace ,HuggingFaceEmbeddings ,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


load_dotenv()

# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]



embedding_model = HuggingFaceEmbeddings()

vectorstore = FAISS.from_documents(documents=docs,embedding=embedding_model)

base_retriever = vectorstore.as_retriever(search_kwargs={"k":2})



hf_endpoint=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

llm = ChatHuggingFace(llm = hf_endpoint)

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)


query= "What is photsynthesis?"
compressed_result = compression_retriever.invoke(query)


for i ,doc in enumerate(compressed_result):
    print(f"\n--- Result {i+1}---")
    print(doc.page_content)






















