
# # app.py
# import os
# import streamlit as st
# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv

# load_dotenv()

# # --- WIDE PAGE LAYOUT ---
# st.set_page_config(layout="wide")
# st.title("YouTube Transcript Q&A with FAISS & LLM")

# # --- TWO COLUMNS LAYOUT ---
# col1, col2 = st.columns([1, 1])  # Left: video, Right: chat

# # --- LEFT COLUMN: VIDEO ---
# with col1:
#     video_id = st.text_input("Enter YouTube Video ID", "")
#     if video_id:
#         # Larger video using iframe
#         video_html = f"""
#         <iframe width="100%" height="600" 
#         src="https://www.youtube.com/embed/{video_id}" 
#         frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
#         allowfullscreen></iframe>
#         """
#         st.markdown(video_html, unsafe_allow_html=True)

# # --- RIGHT COLUMN: QUESTION INPUT & ANSWER ---
# with col2:
#     question = st.text_input("Ask a question about this video")
#     if st.button("Get Answer"):

#         # --- FETCH TRANSCRIPT ---
#         try:
#             api = YouTubeTranscriptApi()
#             transcript_list = api.fetch(video_id, languages=["en"])
#             transcript = " ".join(chunk.text for chunk in transcript_list)
#         except TranscriptsDisabled:
#             st.warning("No captions available for this video.")
#             transcript = ""

#         if transcript:
#             # --- EMBEDDINGS MODEL ---
#             embeddings_model = HuggingFaceEmbeddings()

#             # --- LOAD OR CREATE FAISS INDEX ---
#             index_path = f"faiss_index_{video_id}"
#             if os.path.exists(index_path):
#                 st.info(f"Loading existing FAISS index for video {video_id}")
#                 vectorstore = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
#             else:
#                 st.info(f"Creating new FAISS index for video {video_id}")
#                 splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#                 chunks = splitter.create_documents([transcript])
#                 vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
#                 vectorstore.save_local(index_path)

#             # --- RETRIEVAL ---
#             retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 3})

#             # --- AUGMENTATION ---
#             prompt = PromptTemplate(
#                 template="""
#                 You are a helpful assistant.
#                 Answer only from the provided transcript context.
#                 If the context is insufficient, just say "don't know".
#                 {context}
#                 Question: {question}
#                 """,
#                 input_variables=['context', 'question']
#             )

#             # --- GENERATION ---
#             hf_endpoint = HuggingFaceEndpoint(
#                 repo_id="meta-llama/Llama-3.1-8B-Instruct",
#                 task="text-generation",
#             )
#             chatModel = ChatHuggingFace(llm=hf_endpoint)

#             # --- PARSER ---
#             parser = StrOutputParser()

#             def format_docs(retrieved_docs):
#                 return "\n\n".join(doc.page_content for doc in retrieved_docs)

#             parallel_chain = RunnableParallel({
#                 'context': retriever | RunnableLambda(format_docs),
#                 'question': RunnablePassthrough()
#             })

#             main_chain = parallel_chain | prompt | chatModel | parser
#             result = main_chain.invoke(question)

#             st.subheader("Answer:")
#             st.write(result)


















# app.py
import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# --- WIDE PAGE LAYOUT ---
st.set_page_config(layout="wide")
st.title("YouTube Transcript Q&A with FAISS & LLM")

# --- TWO COLUMNS LAYOUT ---
col1, col2 = st.columns([1, 1])  # Left: video, Right: chat

# --- LEFT COLUMN: VIDEO ---
with col1:
    video_id = st.text_input("Enter YouTube Video ID", "")
    if video_id:
        # Larger video using iframe
        video_html = f"""
        <iframe width="100%" height="600" 
        src="https://www.youtube.com/embed/{video_id}" 
        frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
        allowfullscreen></iframe>
        """
        st.markdown(video_html, unsafe_allow_html=True)

# --- RIGHT COLUMN: QUESTION INPUT & ANSWER ---
with col2:
    question = st.text_input("Ask a question about this video")
    if st.button("Get Answer"):

        # --- FETCH TRANSCRIPT ---
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.fetch(video_id, languages=["en"])
            transcript = " ".join(chunk.text for chunk in transcript_list)
        except TranscriptsDisabled:
            st.warning("No captions available for this video.")
            transcript = ""

        if transcript:
            # --- SHOW LOADING SPINNER WHILE GENERATING ANSWER ---
            with st.spinner("Generating answer..."):
                # --- EMBEDDINGS MODEL ---
                embeddings_model = HuggingFaceEmbeddings()

                # --- LOAD OR CREATE FAISS INDEX ---
                index_path = f"faiss_index_{video_id}"
                if os.path.exists(index_path):
                    st.info(f"Loading existing FAISS index for video {video_id}")
                    vectorstore = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
                else:
                    st.info(f"Creating new FAISS index for video {video_id}")
                    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
                    chunks = splitter.create_documents([transcript])
                    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
                    vectorstore.save_local(index_path)

                # --- RETRIEVAL ---
                retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 3})

                # --- AUGMENTATION ---
                prompt = PromptTemplate(
                    template="""
                    You are a helpful assistant.
                    Answer only from the provided transcript context.
                    If the context is insufficient, just say "don't know".
                    {context}
                    Question: {question}
                    """,
                    input_variables=['context', 'question']
                )

                # --- GENERATION ---
                hf_endpoint = HuggingFaceEndpoint(
                    repo_id="meta-llama/Llama-3.1-8B-Instruct",
                    task="text-generation",
                )
                chatModel = ChatHuggingFace(llm=hf_endpoint)

                # --- PARSER ---
                parser = StrOutputParser()

                def format_docs(retrieved_docs):
                    return "\n\n".join(doc.page_content for doc in retrieved_docs)

                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })

                main_chain = parallel_chain | prompt | chatModel | parser
                result = main_chain.invoke(question)

            st.subheader("Answer:")
            st.write(result)






















