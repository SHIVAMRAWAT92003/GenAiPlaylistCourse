from typing import TypedDict
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import load_dotenv

load_dotenv()


hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=200
)

chatModel = ChatHuggingFace(llm =hf_endpoint)

# Schema
class Review(TypedDict):
    summary:str
    sentiment:str


structured_model = chatModel.with_structured_output(Review)



result = structured_model.invoke("""
The hardware is great but the software feels bloated.There are too many pre installed aaps that I can't remove .Also, the UI looks outdated compare to other brands. Hpoing for a software update to fix this.
""")

print(result)

