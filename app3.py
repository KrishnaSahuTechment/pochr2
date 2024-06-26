import streamlit as st
import os
import PyPDF2 as pdf
import json
import oci
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OCIGenAI

# Load secrets from Streamlit
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

# OCI configuration
config = {
    "user": user,
    "fingerprint": fingerprint,
    "key_file": key_file,
    "tenancy": tenancy,
    "region": region
}

def initialize_llm(temperature=0.75, top_p=0, top_k=0, max_tokens=2000):
    print(f"Temperature: {temperature}")
    print(f"Top_p: {top_p}")
    print(f"Top_k: {top_k}")
    print(f"Max_tokens: {max_tokens}")
    try:
        client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config)
        
        llm = OCIGenAI(
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
    template = """
    Act like a skilled or very experienced ATS (Application Tracking System)
    with a deep understanding of the tech field, software engineering, data science, data analysis,
    and big data engineering. Your task is to evaluate the resume based on the given job description.
    You must consider the job market is very competitive and you should provide
    the best assistance for improving the resumes. Assign the percentage Matching based
    on JD and
    the missing keywords with high accuracy
    resume: {resume}
    description: {description}

    I want the response in one single string having the structure:
    Job description Match: %,

    \n\n
    MissingKeywords:[,],

    \n\n
    Profile Summary: "in bullet points"
    """

    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"resume": text, "description": jd})
    return response

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""

    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text()) + "\n"

    return text

# Streamlit app
st.title("Smart ATS")
st.text("Improve Your Resume ATS")
jd = st.text_area("Paste the Job Description")
uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the pdf")

submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        llm = initialize_llm()
        text = input_pdf_text(uploaded_file)
        response = get_model_response(llm, text, jd)
        st.subheader("Response")
        st.write(response["text"])
