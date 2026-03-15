
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core. prompts import PromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=40
) 

chatModel = ChatHuggingFace(llm = hf_endpoint)

class Person(BaseModel):

    name:str = Field(description='Name of the person')
    age:int = Field(gt=18,description='Age of the person')
    city:str = Field(description='City of the person belongs to')


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the 3 name, age, city of a fictional {place} person \n {format_instruction} ",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


chain =  template | chatModel | parser

final_result = chain.invoke({'place':'British'})


print(final_result)





















