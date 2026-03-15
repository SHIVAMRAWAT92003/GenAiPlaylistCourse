from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


video_id = "Gs8ZPKCFlTc"

try:
    api = YouTubeTranscriptApi()  
    transcript_list = api.fetch(video_id, languages=["en"])  
    transcript = " ".join(chunk.text for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

# print(transcript_list)


# SPLITER
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.create_documents([transcript])
# print(len(chunks))



# VECTORSTORE
embeddings_model = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(documents=chunks,embedding=embeddings_model)
# print(vectorstore.index_to_docstore_id)



# RETRIEVALS
retriever = vectorstore.as_retriever(search_type= 'similarity', search_kwargs={"k":3} )



# AUGMENTATION
prompt= PromptTemplate(
    template="""
    Your are a helpful assistant
    Answer only from the provided transcript context.
    If the context is insufficient,just say don't know.
    {context}
    Question: {question}
""",
input_variables=['context','question']
)

question = "What is functional Interface ?"
retrieved_docs = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print(context_text)
final_prompt = prompt.invoke({"context":context_text,"question":question})
# print(final_prompt)



# GENERATION
hf_endpoint= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

chatModel = ChatHuggingFace(llm =hf_endpoint)
answer = chatModel.invoke(final_prompt)
print(answer.content)













