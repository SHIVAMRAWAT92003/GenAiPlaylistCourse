from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()

# Initialize HuggingFace model with token limit
hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=150  # Hard stop (~100–120 words)
) 

chatModel = ChatHuggingFace(llm=hf_endpoint)

# Define template with word limit instruction
chat_template = ChatPromptTemplate([
    ('system', 'You are a Professional {domain} expert.'),
    ('human', 'Explain what is {topic} in {style} terms in less than {word_limit} words.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# Initialize history
chat_history = []

# Try loading existing chat history
try:
    with open('chatbot_history.txt') as f:
        chat_history.extend(f.readlines())
except FileNotFoundError: 
    pass

# Ask for setup info only if history is empty
if not chat_history:
    domain = input("Enter domain (e.g. Finance, Tech, Medical...): ")
    topic = input("Enter topic: ")
    style = input("Enter style (e.g. simple, professional, layman): ")
    word_limit = input("Enter word limit (e.g. 50, 100, 200): ")
else:
    # Default fallback values
    domain, topic, style, word_limit = "General", "Conversation", "simple", "100"

# Chatbot loop
while True:
    query = input("Ask me Anything (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Exiting chatbot... Goodbye!")
        break

    # Fill in template variables
    prompt = chat_template.invoke({
        'domain': domain,
        'topic': topic,
        'style': style,
        'word_limit': word_limit,
        'chat_history': chat_history,
        'query': query
    })
    

    # Get model response
    response = chatModel.invoke(prompt)

    # Print response
    print("AI:", response.content)

    # Save chat to history
    chat_history.append(f"User: {query}")
    chat_history.append(f"Bot: {response.content}")

    with open("chatbot_history.txt", "w") as f:
        f.write("\n".join(chat_history))
