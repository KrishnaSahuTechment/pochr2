import streamlit as st
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

# Define the command you want to run in the subprocess
command = "oci os ns get"  # For example, listing files in the current directory

# Run the command as a subprocess using the subprocess module
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output, error = process.communicate()

# Display the output in Streamlit
st.write("Output:")
st.code(output.decode('utf-8'))

st.write("Error:")
st.code(error.decode('utf-8'))

def initialize_llm(temperature=0.75,top_p=0,top_k=0,max_tokens=200):
    return OCIGenAI(
        model_id="cohere.command",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id=COMPARTMENT_ID,
        model_kwargs={"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_tokens": max_tokens}        
    )


def initialize_object_storage_client():
    config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)   
    # config = {
    #     "user":st.secrets["user"] ,
    #     "fingerprint":st.secrets["fingerprint"],
    #     "tenancy":st.secrets["tenancy"],
    #     "region":st.secrets["region"],
    #     "key_file":st.secrets["key_file"] # TODO
    # }
    # validate the default config file
    # signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    # config_response = oci.config.validate_config(config)
    # print("config_response",config_response)

    return oci.object_storage.ObjectStorageClient(config)


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
    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-english-v3.0",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id=COMPARTMENT_ID,
        model_kwargs={"temperature": 0, "top_p": 0, "max_tokens": 512}
    )
    return FAISS.from_documents(docs, embeddings)


def create_chains(llm, retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt,
    )

    system_prompt = (
        "Your name is Wall-E,You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def get_session_history(store, session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.08)


def display_chat_history(history):
    for msg in history.messages:
        st.chat_message(msg.type).write(msg.content)


def main():
    temperature  = st.sidebar.slider("Tempreture:", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
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

    history = StreamlitChatMessageHistory(key="chat_history")
    memory = ConversationBufferMemory(chat_memory=history)

    rag_chain = create_chains(llm, retriever)
    store = {}

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: get_session_history(store, session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Connect to SQLite database
    conn = sqlite3.connect(f'{DATABASE_NAME}.db')
    c = conn.cursor()

    # Create a table to store chat messages if not exists
    c.execute(
        f'''CREATE TABLE IF NOT EXISTS {DATABASE_NAME} (session_id TEXT, AI_message TEXT, Human_message TEXT, date_val TEXT) ''')

    today = date.today()
    str_today = str(today)


    # Define session history and other tabs
    tabs = ["Chatbot", "Session History"]
    selected_tab = st.sidebar.radio("Select Tab", tabs)

    # Display content based on selected tab
    if selected_tab == "Chatbot":
        # Display current chat history
        st.title("Welcome to Techment chatbot")

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

    elif selected_tab == "Session History":
        # Create a sidebar to display session history
        st.sidebar.subheader("History")
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

        conn.close()


if __name__ == "__main__":
    main()
