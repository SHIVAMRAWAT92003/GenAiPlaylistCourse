# Lazy Loader are used when we have to deal with large number of files or documents


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

hf_endpoint = HuggingFaceEndpoint (
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

chatModel = ChatHuggingFace(llm = hf_endpoint)

parser = StrOutputParser()

loader = TextLoader('Cricket.txt',encoding='utf-8')

docs= loader.lazy_load()

prompt = PromptTemplate(
    template="Write a 5 line summary of the following poem {poem}",
    input_variables=['poem']
)

chain  = prompt | chatModel | parser

result = chain.invoke({'poem':docs[0].page_content})

print("Poem :",docs[0].page_content)

print("\n Summary:",result)
