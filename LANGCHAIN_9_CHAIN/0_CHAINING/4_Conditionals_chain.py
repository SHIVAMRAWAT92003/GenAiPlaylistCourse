from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field, BaseModel
from typing import Literal



load_dotenv()

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

chatModel = ChatHuggingFace(llm = hf_endpoint)

parser1 =StrOutputParser()

class FeedBack(BaseModel):
    sentiment :Literal['Positive','Negative']=Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=FeedBack)

prompt1 = PromptTemplate(
    template="classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

clasifier_chain = prompt1 | chatModel | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback in 10 words\n {feednack}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback in 10 words\n {feednack}',
    input_variables=['feedback']
)





branch_chain = RunnableBranch(
    (lambda x:x.sentiment =='Positive', prompt2 | chatModel | parser1),
    (lambda x:x.sentiment =='Negative', prompt3|chatModel|parser1),
    RunnableLambda(lambda x: "Could not find the sentiment...")
)

chain = clasifier_chain | branch_chain

result = chain.invoke({'feedback':'They connect people globally, provide access to education, work, and entertainment, and have become essential tools in daily life.'})

print(result)






























