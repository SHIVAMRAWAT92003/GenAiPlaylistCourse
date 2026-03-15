from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core. prompts import PromptTemplate 



hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=40

) 

chatModel = ChatHuggingFace(llm = hf_endpoint)


schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),

]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me 3 fact about {topic} \n {format_instruction} ',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'BlackHole'})

result = chatModel.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
















