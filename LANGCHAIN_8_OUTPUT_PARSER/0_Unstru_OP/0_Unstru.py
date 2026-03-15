from langchain_huggingface import  ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate




load_dotenv()

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=200
)

chatModel = ChatHuggingFace(llm = hf_endpoint)



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


prompt1 = template1.invoke({'topic':'blackhole'})
step_1_result = chatModel.invoke(prompt1)


prompt2 = template2.invoke({'text':step_1_result.content})
step_2_result = chatModel.invoke(prompt2)


print(step_2_result.content)



























