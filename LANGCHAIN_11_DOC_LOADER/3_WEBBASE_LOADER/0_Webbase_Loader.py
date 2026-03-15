import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
from langchain_community.document_loaders import WebBaseLoader

url='https://www.amazon.in/dp/B0FQFYXCC4'

loader= WebBaseLoader(url)

docs = loader.load()

print(len(docs))

print(docs[0].page_content)


# step1
# for web-scraping you need to install beautifulsoup and request
# pip install beautifulsoup4 requests


# step2
# USER_AGENT environment variable not set, consider setting it to identify your requests.
# Always import on the top of the file otherwise it will create warning.
# import os
# os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
