# Dynamic Prompt with seperate prompt template file and streamlit ui. 

from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt

load_dotenv()
llm=HuggingFaceEndpoint(
     repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

chatmodel = ChatHuggingFace(llm=llm) 


st.header("Research Tool")
selected_paper = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

selected_style = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

selected_length = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


summary_template = load_prompt("2_Template.json")


final_prompt = summary_template.invoke({
    'paper_input':selected_paper,
    'style_input':selected_style,
    'length_input':selected_length
})



if st.button('Summarize'):
    result = chatmodel.invoke(final_prompt)
    st.write(result.content)



