from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

#Tool Creation
@tool
def Multiply(a:int , b:int)->a:
    """"Given 2 number a nd b this tools return their products"""
    return a*b


# Tool Binding
llm = ChatOpenAI
llm_with_tool = llm.bind_tools([Multiply])


# query
query = HumanMessage('Can you multiply 3 with 100')
messages = [query]


# Tool Calling
result= llm_with_tool.invoke(messages)


# Storing Conversation
messages.append(result)

# Tool Calling
tool_result = Multiply.invoke(result.tool_call[0])


# Storing Conversation
messages.append(tool_result)


print(llm_with_tool.invoke(messages).content)
