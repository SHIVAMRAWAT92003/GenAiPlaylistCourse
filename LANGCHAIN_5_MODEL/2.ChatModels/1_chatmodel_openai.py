from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

chatModel = ChatOpenAI(model='gpt-4', temperature=0, max_completion_tokens = 10)

result = chatModel.invoke("What is capital of India?")

print(result.content)