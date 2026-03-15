from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    
)

chatModel = ChatHuggingFace(llm = hf_endpoint)

prompt = PromptTemplate(
    template="Tell me 5 Detail point about the {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | chatModel | parser

result = chain.invoke({'topic':'cricket'})

print(result)
chain.get_graph().print_ascii()   #pip install grandalf
