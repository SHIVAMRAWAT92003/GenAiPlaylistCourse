from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=100
)

chatModel = ChatHuggingFace(llm = hf_endpoint)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 fact about {topic} \n {format_instruction} ',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# Chaining Method
chain = template | chatModel | parser

result = chain.invoke({'topic':'Black Hole'})

print(result)










