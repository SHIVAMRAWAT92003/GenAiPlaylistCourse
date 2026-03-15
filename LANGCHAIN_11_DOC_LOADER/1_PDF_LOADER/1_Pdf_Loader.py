# Lazy Loader are used when we have to deal with large number of files or documents

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Sustainable_Development.pdf')

docs = loader.lazy_load()

# print(docs)

# print(len(docs))

# print(docs[0])

print(docs[0].page_content)

print(docs[0].metadata)