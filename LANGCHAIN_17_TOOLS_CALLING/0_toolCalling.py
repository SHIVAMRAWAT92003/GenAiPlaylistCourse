from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests


# Tool Creation
@tool
def multiply(a:int, b:int)->int:
    """"Given 2 number a nd b this tools return their products"""
    return a*b
@tool
def add(a:int, b:int)->int:
    """"Given 2 number a nd b this tools return their addition"""
    return a+b



# Tool Binding
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([multiply,add])



# Tool Calling
llm_with_tools.invoke('Hi how are you' )
result=llm_with_tools.invoke('can you multiply 3 with 1000')
result.tool_calls[0]


# Tool Execution
multiply.invoke(result.tool_calls[0])







