from langchain_huggingface import  ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



load_dotenv()

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=200
)

chatModel = ChatHuggingFace(llm = hf_endpoint)

parser = StrOutputParser()

# temp1 
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)


# temp2
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. {text}',
    input_variables=['text']

)


# Chaining
chain = template1 | chatModel | parser | template2 | chatModel | parser

result = chain.invoke({'topic':'Black Hole'})
print(result)





















