import streamlit as st
st.set_page_config(page_title="LLM Tracing", layout="wide")

from src.LLM_Utility.Sidemenu_initialization import All_Initialization,CSS_Property
from src.LLM_source_code.LLM_RAG.LLM_RAG import LLM_RAG_Impl

if __name__=='__main__':
    vAR_hide_footer = """<style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(vAR_hide_footer, unsafe_allow_html=True)
    # Applying CSS properties for web page
    CSS_Property("src/LLM_Utility/LLM_style.css")
    # Initializing Basic Componentes of Web Page
    choice = All_Initialization()
 
    if choice=="LLM Tracing & Evaluation":

        LLM_RAG_Impl()

        

        

        
        
    else:
        pass