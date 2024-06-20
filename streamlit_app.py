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

# Global Variables
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

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] =  f"HR_POC_staging"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

SESSION_ID = "abc12345"
DATABASE_NAME = "chat_history_table_session"

pdf_bucket_name = st.secrets["pdf_bucket_name"] #"demo_text_labeling"
history = StreamlitChatMessageHistory(key="history")
memory = ConversationBufferMemory(chat_memory=history)
store = {}

#set config file
config = {
            "user":user,
            "fingerprint":fingerprint,       
            "key_file":key_file, 
            "tenancy":tenancy,        
            "region":region
        } 

object_storage = oci.object_storage.ObjectStorageClient(config)

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

def get_example(objects):
    examples = []
    for obj in objects:
        if obj.name.endswith('.pdf'):  # Assuming resumes are in PDF format
            par_url = create_preauthenticated_request(obj.name)
            examples.append({"input": f"What is the resume link of {obj.name}?","answer": f"You can view resume here: [{obj.name}]({par_url})"})
            examples.append({"input": f"Can I get the link to {obj.name}? resume?", "answer": f"You can view resume here: [{obj.name}]({par_url})"})
            examples.append({"input": f"Can you drop the link to {obj.name} resume?", "answer": f"You can view resume here: [{obj.name}]({par_url})"})
            examples.append({"input": f"Can you please provide the link to  {obj.name} resume?", "answer": f"You can view resume here: [{obj.name}]({par_url})"})
            examples.append({"input": f"Could you share {obj.name} resume link with me?","answer": f"You can view resume here: [{obj.name}]({par_url})"})
            examples.append({"input": f"I'd like to see  {obj.name} resume. Can you provide the link?", "answer": f"You can view resume here:[{obj.name}]({par_url})"})
            examples.append({"input": f"Would you be able to share {obj.name} resume link with me?","answer": f"You can view resume here: [{obj.name}]({par_url})"})
            examples.append({"input": f"Would you be able to share {obj.name} resume link with me?","answer": f"You can view resume here:[{obj.name}]({par_url})"})
    return examples


def initialize_object_storage_client():
    try: 
        return oci.object_storage.ObjectStorageClient(config)
    except Exception as e:
        print(f"Error loading OCI config: {e}")
        raise e

def get_object_content(object_storage_client):
    try:
        get_object_response = object_storage_client.get_object(NAMESPACE, BUCKET_NAME, OBJECT_NAME)
        return get_object_response.data.content
    except oci.exceptions.ServiceError as e:
        print("Error getting object:", e)
        return None

def save_and_load_json(data_string):
    rows = [row.split(',') for row in data_string.strip().split('\n')]

    with open('test.json', 'w', newline='', encoding='utf-8') as jsonfile:
        json.dump(rows, jsonfile)

    with open('test.json', 'r', encoding='utf-8') as jsonfile:
        return json.load(jsonfile)

def create_vectorstore(docs):
    client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config)       
    
    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-english-v3.0",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id=COMPARTMENT_ID,
        model_kwargs={"temperature": 0, "top_p": 0, "max_tokens": 512},
        client=client
    )
    return FAISS.from_documents(docs, embeddings)


def few_shot_data(llm,examples):
    # st.write(examples)
    # print(examples)
    example_formatter_template = """Question: {question}
                                Answer: {answer}
                                """

    example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=example_formatter_template,
        )
    
    few_shot_prompt = FewShotPromptTemplate(
        # These are the examples we want to insert into the prompt.
        examples=examples,
        # This is how we want to format the examples when we insert them into the prompt.
        example_prompt=example_prompt,
        # The prefix is some text that goes before the examples in the prompt.
        # Usually, this consists of intructions.
        prefix="Give the answer of every input\n",
        # The suffix is some text that goes after the examples in the prompt.
        # Usually, this is where the user input will go
        suffix="Question: {input}\nAnswer: ",
        # The input variables are the variables that the overall prompt expects.
        input_variables=["input"],
        # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
        example_separator="\n",
        )

    chain=LLMChain(llm=llm,prompt=few_shot_prompt)
    response = (chain({'input':"What is the resume link of Krishna Sahu?"}))
    # st.write(response)
    st.write(response["text"])

    # st.markdown( response["text"])



def create_chains(llm, retriever):    
    objects = list_objects_in_bucket(NAMESPACE, pdf_bucket_name)
    examples = get_example(objects)
    print(examples)
    examples_str = "\n".join([f"\n Q: {ex['input']} \n A: {ex['answer']}" for ex in examples])

    st.write(examples_str)
    st.write("examples",examples)

    example_prompt = (
        HumanMessagePromptTemplate.from_template("{input}")
        + AIMessagePromptTemplate.from_template("{answer}")
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )

    # chain=LLMChain(llm=llm,prompt=few_shot_prompt)
    contextualize_q_system_prompt = (
        f"""Your name is Wall-E,You are an assistant for question-answering tasks.You are an AI language model trained to provide precise answers based on given examples. Follow the format and style of the provided examples exactly. Do not add any additional information or deviate from the structure. Always ensure the answer is in the same format as shown in the examples. Do not modify the links provided in the examples. and also take examples which I provided to you
        ### Examples:

        Q: What is the resume link of Krishna Sahu.pdf?
        A: You can view the resume here: [Krishna Sahu.pdf](https://objectstorage.us-chicago-1.oraclecloud.com/p/4IRwmMhlvWGXfti7Mxir5BvelyCSElCCjC4ICzqCHqgLsrpROqP8t1aK_uj8ZZVL/n/axbpjkug04ct/b/demo_text_labeling/o/Krishna%20Sahu.pdf)

        Q: Can I get the link to Parul Paul.pdf? resume?
        A: You can view resume here: [Parul Paul.pdf](https://objectstorage.us-chicago-1.oraclecloud.com/p/mOKtcBmHubbGGRXAFwUpLmIbwfChft-nGJru-77mQK-E4G1dNNtqi9rePD_DvhZC/n/axbpjkug04ct/b/demo_text_labeling/o/Parul%20Paul.pdf)

        {examples_str}
        
        ### End of Examples

        ### Instructions:

        Based on the examples, provide the exact format of the answer for the given input question. Do not modify any links provided in the examples.
        """
        """

        Input: {input}
        Answer: 
        """         
            
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        few_shot_prompt,        
        ('human', '{input}'),
        ("ai", '{answer}'),
    ]
    )    
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt,
    )

    print("history_aware_retriever created ")
    
    system_prompt = (                
        """First you need to get the answers from few shot template
        if answer is not available to few shot shot template then
        you can also use the following pieces of retrieved context to answer
        the question.     
        \n\n
        {context}
        """       
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt, 
            MessagesPlaceholder("history"),
            ("human", "{input}"),           
        ]
    )    

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)   
    print("question_answer_chain",question_answer_chain)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

# def get_session_history(store, session_id):
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def response_generator(response):
    lines = response.splitlines(keepends=True)
    for line in lines:
        for word in line.split():
            yield word + " "
            time.sleep(0.08)
        yield "\n"  # Yield a newline character to preserve line breaks

def display_chat_history(history):
    for msg in history.messages:
        st.chat_message(msg.type).write(msg.content)

@st.cache_data(ttl=3600)
def list_objects_in_bucket(NAMESPACE, pdf_bucket_name):
    response = object_storage.list_objects(NAMESPACE, pdf_bucket_name)
    return response.data.objects


def create_preauthenticated_request(object_name, expiration_days=7):
    par_details = oci.object_storage.models.CreatePreauthenticatedRequestDetails(
        name=f"par_for_{object_name}",
        object_name=object_name,
        access_type='ObjectRead',
        time_expires=datetime.utcnow() + timedelta(days=expiration_days)
    )
    response = object_storage.create_preauthenticated_request(
        namespace_name=NAMESPACE,
        bucket_name=pdf_bucket_name,
        create_preauthenticated_request_details=par_details
    )
    par_url = f"https://objectstorage.{config['region']}.oraclecloud.com{response.data.access_uri}"
    return par_url

# Connect to SQLite database
conn = sqlite3.connect(f'{DATABASE_NAME}.db')
c = conn.cursor()

def get_chatbot():    
    temperature  = st.sidebar.slider("Tempreture:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    top_p = st.sidebar.slider("Top_p:", min_value=0.00, max_value=1.00, value=0.00, step=0.01)
    max_tokens = st.sidebar.slider("Max Tokens:", min_value=10, max_value=4000, value=512, step=1)
    top_k = st.sidebar.slider("Top_k:", min_value=0.00, max_value=1.00, value=0.00, step=0.01)
    
    llm = initialize_llm(temperature,top_p,top_k,max_tokens)
    object_storage_client = initialize_object_storage_client()
    part_file_content = get_object_content(object_storage_client)

    if part_file_content is None:
        return

    data_string = part_file_content.decode('utf-8')  # Adjust encoding if needed
    data = save_and_load_json(data_string)

    loader = TextLoader("test.json")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)      

    vectorstore = create_vectorstore(docs)
    retriever = vectorstore.as_retriever()  

    rag_chain = create_chains(llm, retriever)
    # few_shot_data(llm,examples)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )    

    # Create a table to store chat messages if not exists
    c.execute(
        f'''CREATE TABLE IF NOT EXISTS {DATABASE_NAME} (session_id TEXT, AI_message TEXT, Human_message TEXT, date_val TEXT) ''')

    today = date.today()
    str_today = str(today)

    display_chat_history(history)

    # Input from the user
    if prompt := st.chat_input():
        st.chat_message("human").write_stream(response_generator(prompt))
        

        # Invoke the conversational RAG chain
        response = conversational_rag_chain.invoke(
            {"input": f"{prompt}"},
            config={
                "configurable": {"session_id": SESSION_ID}
            },
        )

        st.write("response",response)

        # Update the history with the new human message
        history.add_user_message(prompt)

        # Update the history with the new AI response
        history.add_ai_message(response["answer"])

        # Display the AI response
        st.chat_message("ai").write_stream(response_generator(response["answer"]))

        # Save chat message to the database
        c.execute(
            f"INSERT INTO {DATABASE_NAME} (session_id, AI_message, Human_message, date_val) VALUES (?,?,?,?)",
            (SESSION_ID, response["answer"],prompt, str_today))
        conn.commit()


def main():
    # Define session history and other tabs
    tabs = ["Chatbot", "Session History","Resume downloader"]
    selected_tab = st.sidebar.radio("Select Tab", tabs)

    # Display content based on selected tab
    if selected_tab == "Chatbot":
        # Display current chat history
        st.title("Welcome to Techment chatbot")
        get_chatbot()        

    elif selected_tab == "Session History":
        # Create a sidebar to display session history        
        # session_history = st.sidebar.expander("Session History")

        # Display chat history from the database
        st.write("Chat History:")
        unique_dates = c.execute(
            f"SELECT DISTINCT date_val FROM {DATABASE_NAME} where session_id='{SESSION_ID}'").fetchall()

        for date_value in unique_dates:
            with st.expander(date_value[0]):
                chat_history_date = c.execute(
                    f"SELECT AI_message, Human_message, date_val FROM {DATABASE_NAME} where session_id='{SESSION_ID}' and date_val='{date_value[0]}'"
                ).fetchall()

                for message_date in chat_history_date:
                    st.chat_message("human").write(message_date[1])
                    st.chat_message("ai").write(message_date[0])
                    st.markdown("<hr>", unsafe_allow_html=True)

        # all_data = c.execute(
        #             f"SELECT session_id,AI_message, Human_message, date_val FROM {DATABASE_NAME} where session_id='{SESSION_ID}'"
        #         ).fetchall()
        # st.write(all_data)
        # conn.close()

    elif selected_tab == "Resume downloader":
        st.title("Resume Downloader")
        st.write("Fetching resumes from OCI Object Storage...")
        objects = list_objects_in_bucket(NAMESPACE, pdf_bucket_name)       
        resume_links = {}
        for obj in objects:
            if obj.name.endswith('.pdf'):  # Assuming resumes are in PDF format
                par_url = create_preauthenticated_request(obj.name)
                resume_links[obj.name] = par_url
      
        st.write("### Available Resumes")
        for resume, link in resume_links.items():
            st.write(f"[{resume}]({link})")
if __name__ == "__main__":
    main()



