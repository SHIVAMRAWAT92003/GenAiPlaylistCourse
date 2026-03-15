# ChatBot with ChatHistory

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
    
)
chatmodel = ChatHuggingFace(llm=hf_endpoint)

chatHistory=[
     
]

while True:
    user_input = input("Ask me Anything: ")
    if user_input.lower() ==  "exit":
        break


    #Add user input mess in chat history
    chatHistory.append(user_input) 

    result=chatmodel.invoke(chatHistory)

    #Add AI output mess in chat history
    chatHistory.append(result.content)

    print("\nAI:",result.content,"\n")


print("\n---- Generating Chat History ----\n")
for msg in chatHistory:
    print(msg)