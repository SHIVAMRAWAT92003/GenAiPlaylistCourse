from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Langchain1.pdf')

docs = loader.load()



spliter =  CharacterTextSplitter(
    chunk_size =100,
    chunk_overlap=0,
    separator=''
)

chunks = spliter.split_documents(docs)


print(chunks )

print(chunks [0].page_content)







