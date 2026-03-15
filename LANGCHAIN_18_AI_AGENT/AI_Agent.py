from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()


hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
chatModel = ChatHuggingFace(llm = hf_endpoint)


search_tool = DuckDuckGoSearchRun()
result = search_tool.invoke('Top News in India?')


from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

prompt = hub.pull("hwchase17/react")


# React Agent
agent = create_react_agent(
    llm=chatModel,
    tools=[search_tool],
    prompt=prompt
)

# React Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True
)

response = agent_executor.invoke({"input":"3 ways to reach goa from delhi"})
print(response)