from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel , RunnableSequence ,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

hf_endpoint= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

chatModel = ChatHuggingFace(llm = hf_endpoint)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Give me a explanation of joke {text} in 30 words.',
    input_variables=['text']
)

joke_generator_chain = RunnableSequence(prompt1,chatModel,parser)

parallel_chain = RunnableParallel({
'joke':RunnablePassthrough(),
'explanation':RunnableSequence(prompt2,chatModel,parser)
})

final_chain = RunnableSequence(joke_generator_chain,parallel_chain)

result = final_chain.invoke({'topic':'AI'})

print(result)

print('*********************************************')

print(result['joke'])
print(result['explanation'])





























































