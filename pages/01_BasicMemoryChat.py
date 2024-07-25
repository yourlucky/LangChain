from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

import streamlit as st

st.set_page_config(
    page_title="BasicChatGPT",
    page_icon="🗣️",
)

st.title("Basic Chat")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask any questions!
This chatbot can remember conversations and provide responses based on past interactions

"""
)

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

llm = ChatOpenAI(temperature=0.1)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         You are a helpful Aptamigo employee. Please chat with the human using the following context. 
         Context: {context}
         """,
        ),
        ("human", "{question}"),
    ]
)

message = st.chat_input("Type a message...")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if message:
    paint_history()
    send_message(message, "human")
    history = "\n".join([f"{msg['role']}: {msg['message']}" for msg in st.session_state['messages']])
   
    chain = prompt|llm
    response = chain.invoke( {"context": history,"question": message,}).content
    send_message(response, "ai")

    with st.sidebar:
        st.write(st.session_state)


