
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import datetime

# Load environment variables
load_dotenv()

# Initialize HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
chatmodel = ChatHuggingFace(llm=llm)

# Streamlit Page Config
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title("🤖 LangChain Chatbot with HuggingFace")

# Persona options
persona_options = {
    "🎓 Teacher": "You are a knowledgeable teacher. Explain concepts simply and clearly.",
    "🧑‍💻 Coding Assistant": "You are an expert coding assistant. Provide clean, efficient code with explanations.",
    "📚 Research Assistant": "You are a research assistant. Provide detailed, fact-based answers with references where possible.",
    "😎 Casual Friend": "You are a friendly conversational partner. Keep responses short, casual, and fun."
}

# Persona selector
persona_choice = st.selectbox("Choose your assistant's style:", list(persona_options.keys()))

# Initialize session state for chat history
if "chatHistory" not in st.session_state or st.session_state.persona != persona_choice:
    st.session_state.chatHistory = [SystemMessage(content=persona_options[persona_choice])]
    st.session_state.persona = persona_choice

# Chat display
for msg in st.session_state.chatHistory:
    role = msg.__class__.__name__.replace("Message", "")
    if role == "Human":
        st.chat_message("user").markdown(msg.content)
    elif role == "AI":
        st.chat_message("assistant").markdown(msg.content)

# User input box
user_input = st.chat_input("Type your message...")

if user_input:
    # Add human message
    st.session_state.chatHistory.append(HumanMessage(content=user_input))
    st.chat_message("user").markdown(user_input)

    # Generate AI response
    with st.spinner("Thinking..."):
        result = chatmodel.invoke(st.session_state.chatHistory)
    st.session_state.chatHistory.append(AIMessage(content=result.content))

    # Display AI response
    st.chat_message("assistant").markdown(result.content)

# --- Stylish End Conversation Button at Bottom ---
st.markdown(
    """
    <style>
    .btn-container {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-top: 20px;
    }
    .custom-btn button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 8px 18px;
        border-radius: 6px;
        font-size: 14px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .custom-btn button:hover {
        background-color: #e03e3e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Buttons (End + Download) ---
st.markdown('<div class="btn-container">', unsafe_allow_html=True)

# End Conversation
if st.button("End Conversation", key="end_btn"):
    st.session_state.chatHistory = [SystemMessage(content=persona_options[persona_choice])]
    st.success("✅ Conversation ended. Start a new one!")

# Download Chat History
if st.session_state.chatHistory:
    # Convert chat history to text
    chat_text = f"Persona: {persona_choice}\n\n"
    for msg in st.session_state.chatHistory:
        role = msg.__class__.__name__.replace("Message", "")
        chat_text += f"{role}: {msg.content}\n\n"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="⬇️ Download Chat",
        data=chat_text,
        file_name=f"chat_history_{timestamp}.txt",
        mime="text/plain",
    )

st.markdown('</div>', unsafe_allow_html=True)
