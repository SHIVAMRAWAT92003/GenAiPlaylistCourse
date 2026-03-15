# ChatBot with ChatHistory and types of messages in langchain.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
    
)
chatmodel = ChatHuggingFace(llm=hf_endpoint)

chatHistory=[
     SystemMessage(content="I'm an expert Assistant, How can I assist you."),
]
 
while True:
    user_input = input("Ask me Anything: ")
    if user_input.lower() ==  "exit":
        break


    #Add user input mess in chat history
    chatHistory.append(HumanMessage(content=user_input)) 

    result=chatmodel.invoke(chatHistory)

    #Add AI output mess in chat history
    chatHistory.append(AIMessage(content=result.content))

    print("\nAI:",result.content,"\n")


print("\n---- Chat History ----\n")
for msg in chatHistory:
    role = msg.__class__.__name__.replace("Message","")
    print(f"{role}: {msg.content}\n")
