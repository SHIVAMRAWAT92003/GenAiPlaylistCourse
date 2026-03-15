# Exchange Rate API ::  https://v6.exchangerate-api.com/v6/{YOUR API KEY}/pair/{SOURCE}/{TARGET}

from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import requests


@tool
def get_conversion_factor(base_curr:str,target_curr:str)->float:
    """ This functions fetches the currency conversion factor between a given base and a target currency."""
    url = f'https://v6.exchangerate-api.com/v6/024ea732bc4f7b7d02917c14/pair/{base_curr}/{target_curr}'
    response = requests.get(url)
    return response.json()

result1 = get_conversion_factor.invoke({'base_curr':'USD','target_curr':'INR'})
print(result1)


@tool
def convert(base_curr_val:int, conversion_factor:Annotated[float,InjectedToolArg])->float:
    """
    given a currency conversion rate this function calculates the target currency value from a given base currency value 
    """
    return base_curr_val*conversion_factor

result2 = convert.invoke({'base_curr_val':10,'conversion_factor':85.16})


# Tool bInding for openAi.
# llm =ChatOpenAI()
# llm_with_tools = llm.bind_tools([get_conversion_factor,convert])

# messages =[ HumanMessage('What is the conversion factor btw USD and INR, and bassed on that can you convert 10 usd to inr')]
# ai_message = llm_with_tools.invoke(messages)






hf_endpoint= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
chatModel =ChatHuggingFace(llm = hf_endpoint)

# Tool binding for HuggingFace.
chatModel_with_tools = chatModel.bind_tools([get_conversion_factor,convert])



messages =[ HumanMessage('What is the conversion factor btw USD and INR, and bassed on that can you convert 10 usd to inr')]
ai_message = chatModel_with_tools.invoke(messages)
messages.append(ai_message)


import json
for tool_call in ai_message.tool_calls:
    # execute the 1st tool and get the value of conversion rat
    if tool_call['name'] == 'get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']
        messages.append(tool_message1)
    # execute the 2nd tool using the conversion rate from tool 1
    if tool_call['name'] == 'convert':
        # fetch the current arg
        tool_call['args']['conversion_factor'] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)


print(chatModel_with_tools.invoke(messages).content)

