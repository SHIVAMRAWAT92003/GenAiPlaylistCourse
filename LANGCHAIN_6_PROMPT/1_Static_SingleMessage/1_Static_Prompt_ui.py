# Static Prompt with streamlit ui. 

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

chatmodel = ChatHuggingFace(llm=hf_endpoint)


st.header('Research tools')
user_input = st.text_input('Enter your prompt')

if st.button('Summarize'):
    
    result = chatmodel.invoke(user_input)
    st.write(result.content)
    









