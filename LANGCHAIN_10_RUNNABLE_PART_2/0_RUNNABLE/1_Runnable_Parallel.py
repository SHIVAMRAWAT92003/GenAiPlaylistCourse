from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel , RunnableSequence

load_dotenv()

hf_endpoint= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

chatModel = ChatHuggingFace(llm = hf_endpoint)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a tweet about the {topic} ",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="Generate a linkedin about the {topic} ",
    input_variables=['topic']
)

parallel_chain  = RunnableParallel({

    'tweet':RunnableSequence(prompt1,chatModel,parser),
    'linkedIn':RunnableSequence(prompt2,chatModel,parser)
})

result=parallel_chain.invoke({'topic':'AI'})

print(result['tweet'])
print(result['linkedIn'])



