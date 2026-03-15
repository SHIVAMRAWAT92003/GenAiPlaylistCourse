from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

chatModel = ChatAnthropic(model='claude-3-5-sonnet-20241022')

result = chatModel.invoke("What is the capital of India?")

print(result.content)
