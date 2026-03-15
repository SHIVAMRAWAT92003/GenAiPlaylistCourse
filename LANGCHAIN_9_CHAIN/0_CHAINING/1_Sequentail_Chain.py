from langchain_huggingface import  ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv 
from langchain_core. prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

hf_endpoint=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

chatModel = ChatHuggingFace(llm = hf_endpoint)

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic} ",
    input_variables=['topic']
    )

prompt2 = PromptTemplate(
    template="Generate a 5 line short summary on text \n {text} ",
    input_variables=['text']
    )


parser = StrOutputParser()


chain = prompt1 | chatModel | parser | prompt2 | chatModel | parser

result = chain.invoke({'topic','Langchain'})
print(result)
chain.get_graph().print_ascii()






















































