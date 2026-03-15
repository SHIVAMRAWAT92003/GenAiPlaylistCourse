# ChatBot without ChatHistory

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

hf_endpoint =HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

ChatModel = ChatHuggingFace(llm = hf_endpoint)


while True:
    user_input = input("Ask me Anything: ")
    if user_input.lower() == "exit":
        break
    
    result = ChatModel.invoke(user_input)
    print("AI Response: ",result.content)

