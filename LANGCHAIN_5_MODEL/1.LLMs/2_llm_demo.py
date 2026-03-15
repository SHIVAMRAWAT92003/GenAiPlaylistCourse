# LLMs Open Source Model:OpenRouter


from openai import OpenAI
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file!")

# Initialize OpenRouter client
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Test query
response = client.chat.completions.create(
    model="openai/gpt-oss-20b:free",
    messages=[{"role": "user", "content": "What is the capital of India?"}]
)

print(response.choices[0].message.content)
