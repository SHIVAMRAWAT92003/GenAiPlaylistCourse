from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
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

class Feedback(BaseModel):
    sentiment : Literal['positive','negative'] = Field (description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

Prompt1 = PromptTemplate(
    template="classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

clasifier_chain = Prompt1 | chatModel | parser2

sentiment_status = clasifier_chain.invoke({'feedback':'Overuse can lead to addiction, reduced face-to-face interaction, and privacy/security concerns.'})
print(sentiment_status)

result = clasifier_chain.invoke({'feedback':'Overuse can lead to addiction, reduced face-to-face interaction, and privacy/security concerns.'}).sentiment
print(result)




















