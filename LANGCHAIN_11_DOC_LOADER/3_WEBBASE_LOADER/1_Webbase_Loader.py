import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader , WebBaseLoader
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

hf_endpoint = HuggingFaceEndpoint (
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

chatModel = ChatHuggingFace(llm = hf_endpoint)

parser = StrOutputParser()

url='https://www.amazon.in/dp/B0FQFYXCC4'

loader= WebBaseLoader(url)

docs = loader.load()

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text \n {text}",
    input_variables=['question','text']
)

chain = prompt|chatModel|parser

result = chain.invoke({'question':'What is the batery backup of this product?','text':docs[0].page_content})

print(result)

