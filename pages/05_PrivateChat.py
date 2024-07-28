from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings, OllamaEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from login_f import check_password

if not check_password():
    st.stop()


st.set_page_config(
    page_title="Private GPT",
    page_icon="🔒",
)

st.title("Basic Chat")

st.markdown(
    """
Welcome!
            
Use this chatbot by downloading the file to your local PC or private server! 
The chatbot remembers conversations and provides responses based on past interactions

"""
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
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
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

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
    model = st.selectbox("Choose Your model", ("mistral","llama2"))

    if model == "mistral":
        llm = ChatOllama(
        model="mistral:latest",
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler(),
        ],
        )
    else:
        llm = ChatOllama(
        model="llama2:latest",
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler(),],
        )
