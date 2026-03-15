from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
    
)
chatmodel = ChatHuggingFace(llm=llm)

chatHistory=[
    #Add SystemMessage in chat history
    SystemMessage(content='You are a helpful assistant'),
   
]


#Add HumanMessage in chat history
chatHistory.append(HumanMessage(content="Tell me about Langchian in 3 line.")) 

result = chatmodel.invoke(chatHistory)

#Add AIMessage in chat history
chatHistory.append(AIMessage(content=result.content))

print(chatHistory)






