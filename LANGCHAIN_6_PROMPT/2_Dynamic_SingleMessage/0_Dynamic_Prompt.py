# Dynamic Prompt without streamlit ui. 


from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

chatModel = ChatHuggingFace(llm=hf_endpoint)


# User input via console
print("=== Research Tool (Console Version) ===\n")

papers=[
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beat GANs on Image Synthesis"
]

styles = ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]

lengths = ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]


# Select options
print("Available Research Papers:")
for i,p in enumerate(papers,1):
    print(f"{i}.{p}")
paper_choice = int(input("Select a paper (1-4): "))
selected_paper = papers[paper_choice-1]

print("\nAvailable Explanation Styles:")
for i,s in enumerate(styles):
    print(f"{i}.{s}")
style_choice = int(input("Select a style (1-4): "))     
selected_style = styles[style_choice-1]

print("\nAvailable Explanation Lengths:")
for i, l in enumerate(lengths, 1):
    print(f"{i}. {l}")
length_choice = int(input("Select a length (1-3): "))
selected_length = lengths[length_choice - 1]



summary_template = PromptTemplate(
    template="""

Please summarize the research paper titled "{paper}" with the following specifications:
Explanation Style: {style}  
Explanation Length: {length}  

1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  

2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  

If certain information is not available in the paper, respond with: 
"Insufficient information available" instead of guessing.  

Ensure the summary is clear, accurate, and aligned with the provided style and length.

""",
input_variables=["paper","style","length"],
validate_template=True

)


final_prompt = summary_template.invoke({
    "paper":selected_paper,
    "style":selected_style,
    "length":selected_length
})


# Generate response
print("\n--- Generating Summary ---\n")
response = chatModel.invoke(final_prompt)
print(response.content)
