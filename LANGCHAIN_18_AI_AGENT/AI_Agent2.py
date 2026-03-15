from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()



search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=de015b022840a689085bb9f094daf914&query={city}'
  response = requests.get(url) 
  return response.json()



hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
chatModel = ChatHuggingFace(llm = hf_endpoint)



from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

prompt = hub.pull("hwchase17/react")


# React Agent
agent = create_react_agent(
    llm=chatModel,
    tools=[search_tool,get_weather_data],
    prompt=prompt
)

# React Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool,get_weather_data],
    verbose=True
)

response = agent_executor.invoke({"input":"Find the capital of Madhya Pradesh, then find it's current weather condition , plan 3 day itneary for visit"})
print(response)
print(response['output'])