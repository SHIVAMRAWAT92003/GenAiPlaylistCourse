from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence ,RunnableLambda , RunnableParallel ,RunnablePassthrough
from dotenv import load_dotenv 
from langchain_core.output_parsers import StrOutputParser

def word_counter(text):
    return len(text.split())

load_dotenv()

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"

)

chatModel = ChatHuggingFace(llm = hf_endpoint)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a joke on {topic} ",
    input_variables=['topic']
)

generate_joke =RunnableSequence(prompt1,chatModel,parser)

parallel_chain = RunnableParallel({ 

    'joke':RunnablePassthrough(),
    'count':RunnableLambda(word_counter)

})
 
finalChain = RunnableSequence(generate_joke,parallel_chain)

result = finalChain.invoke({'topic':'AI'})

print(result)
 
































