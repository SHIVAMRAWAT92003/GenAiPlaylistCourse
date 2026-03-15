from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text=""""

# Welcome to Markdown Editor

This is a **live preview** editor. Try editing this text or use the toolbar buttons above!

## Features
- *Italic* and **bold** text formatting
- Lists and numbered lists
- [Links](https://example.com)
- And much more!

"""

spliter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=100,
    chunk_overlap=0

)

chunks = spliter.split_text(text)

print(chunks[2])