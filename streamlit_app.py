import streamlit as st
import pexpect
import os
import json
import time
import sqlite3
from datetime import date
import subprocess
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory.buffer import ConversationBufferMemory

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
import oci

# Global Variables
CONFIG_PROFILE = "DEFAULT"
NAMESPACE = st.secrets["NAMESPACE"] 
BUCKET_NAME = st.secrets["BUCKET_NAME"] 
OBJECT_NAME = st.secrets["OBJECT_NAME"] 
COMPARTMENT_ID = st.secrets["COMPARTMENT_ID"] 
SESSION_ID = "abc123"
DATABASE_NAME = "chat_history_table_session"
os.environ['OCI_USER'] = 'ocid1.user.oc1..aaaaaaaawxbz5prkm6y3ja5ambupqdfgqn6ggp5zbzojpq7pirvbyqas6dgq'
os.environ['OCI_FINGERPRINT'] = 'e4:64:6a:9e:1a:fa:0d:2f:7a:f8:36:d8:8a:18:83:fd'
os.environ['OCI_KEY_FILE'] = 'krishna.sahu@techment.com_2024-04-24T10_13_19.206Z.pem'
os.environ['OCI_TENANCY'] = 'ocid1.tenancy.oc1..aaaaaaaauevhkihjbrur3awjyepvnvkelbtw5qss6cjuxhwop4etveapxoja'
os.environ['OCI_REGION'] = 'us-chicago-1'


#set config file

# import streamlit as st
# import pexpect

# Define the command to run
cmd = "oci os ns get"  # Example OCI command to get the Object Storage namespace

# Define the user OCID, tenancy OCID, and region (replace with your actual values)
user_ocid = 'ocid1.user.oc1..aaaaaaaawxbz5prkm6y3ja5ambupqdfgqn6ggp5zbzojpq7pirvbyqas6dgq'
tenancy_ocid = 'ocid1.tenancy.oc1..aaaaaaaauevhkihjbrur3awjyepvnvkelbtw5qss6cjuxhwop4etveapxoja'
region = "us-chicago-1"

import streamlit as st
import pexpect
import json

# Define the command to run
cmd = "oci os ns get"  # Example OCI command to get the Object Storage namespace

# Define the user OCID, tenancy OCID, and region (replace with your actual values)
user_ocid = 'ocid1.user.oc1..aaaaaaaawxbz5prkm6y3ja5ambupqdfgqn6ggp5zbzojpq7pirvbyqas6dgq'
tenancy_ocid = 'ocid1.tenancy.oc1..aaaaaaaauevhkihjbrur3awjyepvnvkelbtw5qss6cjuxhwop4etveapxoja'
region = "us-chicago-1"

try:
    # Use pexpect to handle interactive input
    child = pexpect.spawn(cmd, timeout=300)

    # Check if the output contains the "data" dictionary
    index = child.expect([pexpect.EOF, '{"data":'])

    if index == 1:  # "data" dictionary found in the output
        # Extract the JSON data and parse it
        data_output = child.before.decode()
        data_dict = json.loads(data_output)

        if "data" in data_dict and isinstance(data_dict["data"], dict):
            st.write("Found 'data' dictionary in the output:", data_dict["data"])
            # You can perform additional actions or skip the remaining prompts here
        else:
            st.error("Invalid output format. 'data' dictionary not found or invalid format.")
    else:
        # "data" dictionary not found in the output, continue with the prompts
        st.write("Expected output not found. Continuing with prompts...")

        # Handle the first prompt
        child.expect("Do you want to create a new config file? [Y/n]:")
        child.sendline("y")

        # Handle the second prompt
        child.expect("Do you want to create your config file by logging in through a browser? [Y/n]:")
        child.sendline("n")

        # Handle the third prompt
        child.expect("Enter a location for your config")
        child.sendline("")  # Sending an empty line to accept the default

        # Handle the fourth prompt
        child.expect("Enter a user OCID:")
        child.sendline(user_ocid)

        # Handle the fifth prompt
        child.expect("Enter a tenancy OCID:")
        child.sendline(tenancy_ocid)

        # Handle the sixth prompt
        child.expect("Enter a region by index or name")
        child.sendline(region)

        # Handle the 7th prompt
        child.expect("Do you want to generate a new API Signing RSA key pair? (If you decline you will be asked to supply the path to an existing key.) [Y/n]")
        child.sendline("n")

        # Handle the 8th prompt
        child.expect("Enter the location of your API Signing private key file")
        child.sendline("/workspaces/pochr2/krishna.sahu@techment.com_2024-04-24T10_13_19.206Z.pem")

        # Handle the 9th prompt
        child.expect("Fingerprint")
        child.sendline("e4:64:6a:9e:1a:fa:0d:2f:7a:f8:36:d8:8a:18:83:fd")

        # Capture the output
        child.expect(pexpect.EOF)
        result = child.before.decode()

        # Display the command output
        st.code(result)
except pexpect.ExceptionPexpect as e:
    # Display the error if the command fails
    st.error(f"Command failed with error: {str(e)}")

# def initialize_llm(temperature=0.75,top_p=0,top_k=0,max_tokens=200):
#     print(f"Temperature: {temperature}")
#     print(f"Top_p: {top_p}")
#     print(f"Top_k: {top_k}")
#     print(f"Max_tokens: {max_tokens}")
#     try:
#         llm =  OCIGenAI(
#             model_id="cohere.command",
#             service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
#             compartment_id=COMPARTMENT_ID,
#             model_kwargs={"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_tokens": max_tokens}        
#         )
#         print("LLM initialized successfully")
#     except Exception as e:
#         print(f"Error initializing OCIGenAI: {e}")
#         raise e

#     return llm
     


# def initialize_object_storage_client():
#     try:
#         CONFIG_PROFILE = "DEFAULT" 
#         config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)  
        
#         config = {
#             "user":"ocid1.user.oc1..aaaaaaaawxbz5prkm6y3ja5ambupqdfgqn6ggp5zbzojpq7pirvbyqas6dgq",
#             "fingerprint":"e4:64:6a:9e:1a:fa:0d:2f:7a:f8:36:d8:8a:18:83:fd",       
#             "key_file":"krishna.sahu@techment.com_2024-04-24T10_13_19.206Z.pem", 
#             "tenancy":"ocid1.tenancy.oc1..aaaaaaaauevhkihjbrur3awjyepvnvkelbtw5qss6cjuxhwop4etveapxoja",        
#             "region":"us-chicago-1"
#         }   
#         print(f"Loaded OCI config: {config}")
#         return oci.object_storage.ObjectStorageClient(config)
#     except Exception as e:
#         print(f"Error loading OCI config: {e}")
#         raise e

#     # return oci.object_storage.ObjectStorageClient(config)


# def get_object_content(object_storage_client):
#     try:
#         get_object_response = object_storage_client.get_object(NAMESPACE, BUCKET_NAME, OBJECT_NAME)
#         return get_object_response.data.content
#     except oci.exceptions.ServiceError as e:
#         print("Error getting object:", e)
#         return None


# def save_and_load_json(data_string):
#     rows = [row.split(',') for row in data_string.strip().split('\n')]

#     with open('test.json', 'w', newline='', encoding='utf-8') as jsonfile:
#         json.dump(rows, jsonfile)

#     with open('test.json', 'r', encoding='utf-8') as jsonfile:
#         return json.load(jsonfile)


# def create_vectorstore(docs):
#     embeddings = OCIGenAIEmbeddings(
#         model_id="cohere.embed-english-v3.0",
#         service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
#         compartment_id=COMPARTMENT_ID,
#         model_kwargs={"temperature": 0, "top_p": 0, "max_tokens": 512}
#     )
#     return FAISS.from_documents(docs, embeddings)


# def create_chains(llm, retriever):
#     contextualize_q_system_prompt = (
#         "Given a chat history and the latest user question "
#         "which might reference context in the chat history, "
#         "formulate a standalone question which can be understood "
#         "without the chat history. Do NOT answer the question, "
#         "just reformulate it if needed and otherwise return it as is."
#     )
#     contextualize_q_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", contextualize_q_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )
#     history_aware_retriever = create_history_aware_retriever(
#         llm, retriever, contextualize_q_prompt,
#     )

#     system_prompt = (
#         "Your name is Wall-E,You are an assistant for question-answering tasks. "
#         "Use the following pieces of retrieved context to answer "
#         "the question. If you don't know the answer, say that you "
#         "don't know. Use three sentences maximum and keep the "
#         "answer concise."
#         "\n\n"
#         "{context}"
#     )
#     qa_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )
#     question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#     return rag_chain


# def get_session_history(store, session_id):
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]


# def response_generator(response):
#     for word in response.split():
#         yield word + " "
#         time.sleep(0.08)


# def display_chat_history(history):
#     for msg in history.messages:
#         st.chat_message(msg.type).write(msg.content)


# def main():
#     temperature  = st.sidebar.slider("Tempreture:", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
#     top_p = st.sidebar.slider("Top_p:", min_value=0.00, max_value=1.00, value=0.00, step=0.01)
#     max_tokens = st.sidebar.slider("Max Tokens:", min_value=10, max_value=4000, value=512, step=1)
#     top_k = st.sidebar.slider("Top_k:", min_value=0.00, max_value=1.00, value=0.00, step=0.01)
    
#     llm = initialize_llm(temperature,top_p,top_k,max_tokens)
#     object_storage_client = initialize_object_storage_client()
#     part_file_content = get_object_content(object_storage_client)

#     if part_file_content is None:
#         return

#     data_string = part_file_content.decode('utf-8')  # Adjust encoding if needed
#     data = save_and_load_json(data_string)

#     loader = TextLoader("test.json")
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
#     docs = text_splitter.split_documents(documents)      

#     vectorstore = create_vectorstore(docs)
#     retriever = vectorstore.as_retriever()

#     history = StreamlitChatMessageHistory(key="chat_history")
#     memory = ConversationBufferMemory(chat_memory=history)

#     rag_chain = create_chains(llm, retriever)
#     store = {}

#     conversational_rag_chain = RunnableWithMessageHistory(
#         rag_chain,
#         lambda session_id: get_session_history(store, session_id),
#         input_messages_key="input",
#         history_messages_key="chat_history",
#         output_messages_key="answer",
#     )

#     # Connect to SQLite database
#     conn = sqlite3.connect(f'{DATABASE_NAME}.db')
#     c = conn.cursor()

#     # Create a table to store chat messages if not exists
#     c.execute(
#         f'''CREATE TABLE IF NOT EXISTS {DATABASE_NAME} (session_id TEXT, AI_message TEXT, Human_message TEXT, date_val TEXT) ''')

#     today = date.today()
#     str_today = str(today)


#     # Define session history and other tabs
#     tabs = ["Chatbot", "Session History"]
#     selected_tab = st.sidebar.radio("Select Tab", tabs)

#     # Display content based on selected tab
#     if selected_tab == "Chatbot":
#         # Display current chat history
#         st.title("Welcome to Techment chatbot")

#         display_chat_history(history)

#         # Input from the user
#         if prompt := st.chat_input():
#             st.chat_message("human").write_stream(response_generator(prompt))
            

#             # Invoke the conversational RAG chain
#             response = conversational_rag_chain.invoke(
#                 {"input": f"{prompt}"},
#                 config={
#                     "configurable": {"session_id": SESSION_ID}
#                 },
#             )
#             # Update the history with the new human message
#             history.add_user_message(prompt)

#             # Update the history with the new AI response
#             history.add_ai_message(response["answer"])

#             # Display the AI response
#             st.chat_message("ai").write_stream(response_generator(response["answer"]))

#             # Save chat message to the database
#             c.execute(
#                 f"INSERT INTO {DATABASE_NAME} (session_id, AI_message, Human_message, date_val) VALUES (?,?,?,?)",
#                 (SESSION_ID, response["answer"],prompt, str_today))
#             conn.commit()

#     elif selected_tab == "Session History":
#         # Create a sidebar to display session history
#         st.sidebar.subheader("History")
#         # session_history = st.sidebar.expander("Session History")

#         # Display chat history from the database
#         st.write("Chat History:")
#         unique_dates = c.execute(
#             f"SELECT DISTINCT date_val FROM {DATABASE_NAME} where session_id='{SESSION_ID}'").fetchall()

#         for date_value in unique_dates:
#             with st.expander(date_value[0]):
#                 chat_history_date = c.execute(
#                     f"SELECT AI_message, Human_message, date_val FROM {DATABASE_NAME} where session_id='{SESSION_ID}' and date_val='{date_value[0]}'"
#                 ).fetchall()

#                 for message_date in chat_history_date:
#                     st.chat_message("human").write(message_date[1])
#                     st.chat_message("ai").write(message_date[0])
#                     st.markdown("<hr>", unsafe_allow_html=True)

#         conn.close()


# if __name__ == "__main__":
#     main()
