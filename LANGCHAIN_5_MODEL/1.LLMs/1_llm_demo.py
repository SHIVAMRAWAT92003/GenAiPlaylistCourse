# LLMs Closed Source Model

from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = OpenAI(model='openai/gpt-oss-20b:free')


# Test query
result = llm.invoke("What is the capital of India?")
print(result)

