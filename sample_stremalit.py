import streamlit as st
from datetime import datetime

tab_one,tab_two, tab_three = st.tabs(["A","B","C"])

with tab_one:
    st.write("This is tab one")

with tab_two:
    st.write("This is tab two")

with tab_three:
    st.write("This is tab three")

    

