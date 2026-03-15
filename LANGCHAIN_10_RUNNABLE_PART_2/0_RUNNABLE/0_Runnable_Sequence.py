from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

chatModel = ChatHuggingFace(llm =hf_endpoint)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
) 

prompt2 = PromptTemplate(
    template="Explain the following joke {text}",
    input_variables=['text']
)

chain = RunnableSequence(prompt1,chatModel,parser,prompt2,chatModel,parser)

print(chain.invoke({'topic':'AI'}))














