# Lazy Loader are used when we have to deal with large number of files or documents

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
    UnstructuredImageLoader,
)

# mapping of extensions to loaders
loader_mapping = {
    "*.txt": TextLoader,
    "*.csv": CSVLoader,
    "*.pdf": PyPDFLoader,
    "*.docx": UnstructuredWordDocumentLoader,
    "*.xlsx": UnstructuredExcelLoader,
    "*.pptx": UnstructuredPowerPointLoader,
    "*.md": UnstructuredMarkdownLoader,
    "*.png": UnstructuredImageLoader,
    "*.jpg": UnstructuredImageLoader,
    "*.jpeg": UnstructuredImageLoader,
}

docs = []

for pattern, loader_cls in loader_mapping.items():
    loader = DirectoryLoader(
        path="books",
        glob=pattern,
        loader_cls=loader_cls
    )
    docs.extend(loader.lazy_load())

print(len(docs))
