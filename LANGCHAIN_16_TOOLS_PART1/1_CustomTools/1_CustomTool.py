from langchain_core.tools import tool

#Step1: Create a function
def multiply(a,b):
    """Multiply two numbers"""
    return a*b

#Step2: Add type hints
def multiply(a:int, b:int)->int:
    """Multiply two numbers"""
    return a*b

#Step3: Add tool decoator
@tool
def multiply(a:int, b:int)->int:
    """Multiply two numbers"""
    return a*b
    


result = multiply.invoke({"a":3,"b":7})

print(result)


print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema())













