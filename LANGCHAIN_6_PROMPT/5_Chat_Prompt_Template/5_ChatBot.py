from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# Load env
load_dotenv()

# Initialize model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
chatmodel = ChatHuggingFace(llm=llm)

# Create a template with placeholders
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "{user_input}")
])

# Dynamic chat history
chatHistory = []
domain = input("Enter the domain you want to ask (e.g cricket, math, medicine): ")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Format prompt with domain + latest user input
    formatted_prompt = chat_template.invoke({
        "domain": domain,
        "user_input": user_input
    })

    # Convert template output + history into a full conversation
    conversation = chatHistory + formatted_prompt.to_messages()

    # Get model response
    result = chatmodel.invoke(conversation)

    # Save interaction to history
    chatHistory.append(HumanMessage(content=user_input))
    chatHistory.append(AIMessage(content=result.content))

    # Show response
    print("\nAI:", result.content, "\n")

# Print full chat log
print("\n---- Chat History ----\n")
for msg in chatHistory:
    role = msg.__class__.__name__.replace("Message","")
    print(f"{role}: {msg.content}\n")
