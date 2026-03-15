from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv 
from langchain.schema.runnable import RunnableBranch ,RunnableSequence ,RunnableParallel ,RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

chatModel = ChatHuggingFace(llm = hf_endpoint)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="Summarize the report {text}",
    input_variables=['text']
)

report_generation_chain = RunnableSequence(prompt1,chatModel,parser)

branch_chain =  RunnableBranch(
    (lambda x: len(x.split())>500,RunnableSequence(prompt2,chatModel,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generation_chain,branch_chain)

result =final_chain.invoke({'topic','cricket'})

print(result)










