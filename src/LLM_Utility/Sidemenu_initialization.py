
import streamlit as st
from PIL import Image


def CSS_Property(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def All_Initialization():
    '''
    To make the selection in the application and exhibit the libraries in scope, and Model used.

    return:
        Return vAR_choice1 which is to select the type of application
    '''


    
    image = Image.open('src/LLM_Utility/Logo_final.png')
    st.image(image)

    st.markdown("<h2 style='text-align: center; color: #454545; font-size:20px;'>Enterprise LLM Platform with Accurate Response, Observability, and Evaluation:<br><span style='color: #0000FF;'>Open AI, Bedrock, and Google</span></h2><br><br>", unsafe_allow_html=True)
    
    # st.markdown("<h2 style='text-align: center; color: blue; font-size:15px;position: relative;'>LLMAtScale Trace and Evaluate Real-time Answers Generated By LLMs</h2>",unsafe_allow_html=True)
    st.markdown("""
    <hr style="width:100%;height:3px;background-color:gray;border-width:10;position:relative; bottom:30px;">
    """, unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='text-align: center; color: white; font-size:20px;'>Solution Scope</h2>", unsafe_allow_html=True)
    
    vAR_choice2 =  st.sidebar.selectbox(" ",('Home','About Us'))
    vAR_choice1 =  st.sidebar.selectbox(" ",('Select Application','LLM Guardrail','LLM as a Judge','LLM Human in the Feedback Loop','LLM Human in the Feedback Loop V2','LLM Agent')) #,'Policy Analysis - Dialogflow','Policy Analysis - Evaluation'
    vAR_choice3 =  st.sidebar.selectbox(" ",('Libraries in Scope','OpenAI','Streamlit'))
    vAR_choice4 =  st.sidebar.selectbox(" ",('Models Used','GPT4', 'GPT3.5','Claude','Llama','Titan'))
    menu = ["Google Cloud Services in Scope","Cloud Storage", "Cloud Run", "Cloud Function", "Secret Manager","AWS-OpenSearch(Serverless VectorStore)"]
    vAR_choice = st.sidebar.selectbox(" ",menu)
    st.sidebar.write('')
    st.sidebar.write('')
    href = """<form action="#">
    <input style='width: 100%;
        border-radius: 5px;
        padding: 7px;
        background-color: #32CD32;
        border: none;' type="submit" value="Clear/Reset" />
        </form>"""
    st.sidebar.markdown(href, unsafe_allow_html=True)
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.markdown("<p style='text-align: center; color: White; font-size:20px;'>Build & Deployed on<span style='font-weight: bold'></span></p>", unsafe_allow_html=True)
    st.sidebar.write('')
    
    with st.sidebar:
        col1,col2,col3 = st.columns([1.5,1.5,1.5])
        with col1:
            st.image('src/LLM_Utility/openai-logo.png')
        with col2:
            st.image('src/LLM_Utility/aws_logo.png')
        with col3:
            st.image('src/LLM_Utility/Google-Cloud-Platform-GCP-logo.png')
            
    
    return vAR_choice1

