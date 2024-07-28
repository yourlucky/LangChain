import streamlit as st
from login_f import check_password

if not check_password():
    st.stop()

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖",
)



st.markdown('# Hey There! 👋')
st.markdown(
    """
Welcome to my GPT Portfolio! 🎉

Explore more of my projects on my website: yoonna.net

Here are the apps I've developed:

- Memory Chat 💬  
This chatbot can remember conversations and provide responses based on them.  
It utilizes basic LLM model interactions for input and output functionalities.

- Document Chat 📄  
This chatbot can answer questions based on the context of an uploaded document.  
It acts like reading reports and answering questions from your boss for you.

- SiteChat Chat 🌐  
This chatbot generates questions based on the content read from a provided URL (sitemap).  
It gathers information through crawling and responds accordingly.

- Quiz Chat ❓  
This chatbot generates questions based on the context of an uploaded document.  
It forces the LLM's output into a JSON format to show extensibility.

- Private Chat 🔒  
Using a free open-source model, not ChatGPT (OpenAI), this chatbot can remember conversations and provide responses.  
It operates independently of OpenAI with an open-source model.



My summer vacation is coming!🌞  
(🇰🇷 Aug. 23 - Sep. 7)

     
"""
)