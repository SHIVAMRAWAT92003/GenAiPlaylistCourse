from langchain_core.tools import tool

@tool
def add(a:int, b:int)->int:
    """"Add two number"""
    return a+b
@tool
def sub(a:int, b:int)->int:
    """"Sub two number"""
    return a-b
@tool
def multiply(a:int, b:int)->int:
    """"Multiply two number"""
    return a*b 
@tool
def divide(a:int,b:int)->int:
    """"Divide two number"""
    return a/b 


class MathToolKit:
    def get_tools(self):
        return[add,sub,multiply,divide]


toolKit = MathToolKit()
tools = toolKit.get_tools()

for tool in tools:
    print(tool.name,"==>",tool.description)

