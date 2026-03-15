# Static Prompt without streamlit ui. 


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

# Initialize HuggingFace LLM
hf_endpoint=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

chatModel = ChatHuggingFace(llm=hf_endpoint)

print("=====Research Tools====")
user_input = input("Enter your input:\n")
if user_input.strip():
    result = chatModel.invoke(user_input)
    print(result.content)
else:
    print("Not input available")