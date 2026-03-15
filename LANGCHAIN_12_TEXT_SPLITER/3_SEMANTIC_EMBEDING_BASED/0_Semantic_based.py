from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_spliter = SemanticChunker(
    HuggingFaceEmbeddings(),breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.5
)

text =""""

Terrorism, in its broadest sense, is the use of violence against non-combatants to achieve political or ideological aims.The term is used in this regard primarily to refer to intentional violence during peacetime or in the context of war against non-combatants.Cricket is a bat-and-ball game that is played between two teams of eleven players on a field, at the centre of which is a 22-yard (20-metre; 66-foot) pitch with a wicket at each end, each comprising two bails (small sticks) balanced on three stumps. 

A farmer is a person engaged in agriculture, raising living organisms for food or raw materials. The term usually applies to people who do some combination of raising field crops, orchards, vineyards, poultry, or other livestock.



"""

docs = text_spliter.create_documents([text])

print(len(docs))

print(docs)






# pip install sentence-transformers