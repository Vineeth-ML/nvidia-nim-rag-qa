import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/vr7/Desktop/GenAI/nimnvidia/ven1/.env")


### load the Nvidia API Key
os.environ['NVIDIA_API_KEY']=os.getenv('NVIDIA_API_KEY')


llm=ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

def vector_embedding():
    if "vector" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./us_census")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) # Splittings
        print("Hello")
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) ## vector openAI embeddings

st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); 
    padding: 20px; border-radius: 12px; border-left: 5px solid #76b900;'>
        <h1 style='color: #76b900; margin: 0;'>⚡ RAG Document Q&A</h1>
        <p style='color: #aaa; margin: 5px 0 0;'>Powered by NVIDIA NIM · Retrieval-Augmented Generation</p>
    </div>
""", unsafe_allow_html=True)
st.caption("Retrieval-Augmented Generation · Ask questions about your documents instantly")
llm=ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""    
)

prompt1=st.text_input("Enter Your Question From Documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector Store DB IS Ready")

import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retriever_chain.invoke({'input':prompt1})
    print("Response Time :", time.process_time()-start)
    st.write(response['answer'])

    ### With a streamlit Expander

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------")
            st.snow()