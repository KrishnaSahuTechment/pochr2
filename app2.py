import streamlit as st
import os
import PyPDF2 as pdf
# from dotenv import load_dotenv
import json




import streamlit as st
import json
import time
import sqlite3
from datetime import date
from datetime import datetime, timedelta
import os


from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory.buffer import ConversationBufferMemory
from langchain import PromptTemplate, FewShotPromptTemplate

from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

import oci
# load_dotenv() ## load all our environment variables
CONFIG_PROFILE = st.secrets["CONFIG_PROFILE"] 
NAMESPACE = st.secrets["NAMESPACE"] 
BUCKET_NAME = st.secrets["BUCKET_NAME"] 
OBJECT_NAME = st.secrets["OBJECT_NAME"] 
COMPARTMENT_ID = st.secrets["COMPARTMENT_ID"] 
user = st.secrets["user"] 
fingerprint = st.secrets["fingerprint"] 
key_file = st.secrets["key_file"] 
tenancy = st.secrets["tenancy"] 
region = st.secrets["region"] 

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
config = {
            "user":user,
            "fingerprint":fingerprint,       
            "key_file":key_file, 
            "tenancy":tenancy,        
            "region":region
        } 

def initialize_llm(temperature=0.75,top_p=0,top_k=0,max_tokens=200):
    print(f"Temperature: {temperature}")
    print(f"Top_p: {top_p}")
    print(f"Top_k: {top_k}")
    print(f"Max_tokens: {max_tokens}")
    try:
        client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config)    
        
        llm =  OCIGenAI(
            model_id="cohere.command",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id=COMPARTMENT_ID,
            model_kwargs={"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_tokens": max_tokens},     
            client=client
        )
        print("LLM initialized successfully")
    except Exception as e:
        print(f"Error initializing OCIGenAI: {e}")
        raise e

    return llm

def get_model_response(llm, text, jd):
    template = f"""
    Hey Act Like a skilled or very experienced ATS (Application Tracking System)
    with a deep understanding of the tech field, software engineering, data science, data analysis,
    and big data engineering. Your task is to evaluate the resume based on the given job description.
    You must consider the job market is very competitive and you should provide 
    the best assistance for improving the resumes. Assign the percentage Matching based 
    on JD and
    the missing keywords with high accuracy
    resume: {{resume}}
    description: {{description}}

    I want the response in one single string having the structure
    {{"JD Match":,"MissingKeywords:","Profile Summary":""}}
    """

    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"resume": text, "description": jd})
    return response
def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)

    st.write("reader",reader)
    text=""

    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text()) + "\n"

    return text
    

#Prompt Template



## streamlit app
st.title("Smart ATS")
st.text("Improve Your Resume ATS")
jd=st.text_area("Paste the Job Description")
uploaded_file=st.file_uploader("Upload Your Resume",type="pdf",help="Please uplaod the pdf")

submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        llm = initialize_llm()
        text = input_pdf_text(uploaded_file)
        response = get_model_response(llm, text, jd)
        st.subheader("Response")
        st.write(response)