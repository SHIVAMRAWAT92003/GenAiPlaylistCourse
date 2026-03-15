from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
result = search_tool.invoke("Tell me about operation sindoor? ")
print(result) 
print(search_tool.name)
print(search_tool.description)
print(search_tool.args)



# from langchain_community.tools import ShellTool
# shell_tools =  ShellTool()
# result = shell_tools.invoke('Whoami')
# print(result)