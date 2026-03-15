from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system','You are a Professional {domain} expert.'),
    ('human','Explain what is {topic} in {style} term.')

])
prompt = chat_template.invoke({'domain':'Information Security','topic':' Phising','style':'Technical'})
print(prompt)

