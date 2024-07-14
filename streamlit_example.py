import streamlit as st
from langchain.prompts import PromptTemplate

from datetime import datetime


# today = datetime.today().strftime("%H:%M:%S")

# st.title(today)

# st.title("Hello world!")

# st.subheader("Welcom to Streamlit!")

# st.markdown("""This is a simple example of some **markdown** text.""")

# st.write("This is a simple example of some text.")

# st.write(['A','B','C','D'])

# p = PromptTemplate.from_template("xxxxxxx")

# st.write(p)

# model=st.selectbox("chooose your model",("GPT-3","GPT-4"))

# if model == "GPT-3":
#     st.write("You have chosen GPT-3")
# else:
#     st.write("You have chosen GPT-4")

#     value = st.slider("temperature", min_value=0.1, max_value=1.0,)
#     st.write(value)

st.title("Title")

with st.sidebar:
    st.sidebar.title("sidebar title")
    st.sidebar.text_input("xxxxxxx")


st.title("Title!!2")

st.tabs(["A","B","C","D"])