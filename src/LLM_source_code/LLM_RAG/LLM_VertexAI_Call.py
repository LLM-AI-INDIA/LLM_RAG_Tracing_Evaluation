import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from src.LLM_source_code.LLM_RAG.LLM_Bedrock_Call_KB import generate_random_string
import streamlit as st

def generate(input,vAR_model):



    if "vertex_request_id" not in st.session_state:
        st.session_state.vertex_request_id = 0
        st.session_state.vertex_response_id = 0
        st.session_state.vertex_session_id = generate_random_string()

    vAR_retrieved_text = ""
    document1 = Part.from_uri(
        mime_type="application/pdf",
        uri="gs://dmv_elp_project/DMV FAQ.pdf",
    )

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
    ]
    vAR_system_instruction = """You are a helpful DMV customer support assistant. Your job is to respond to user questions based on document provided. If you don\'t know the answer from the document provided, please respond \"I don\'t find any relevant details in the provided document\". Don\'t try to create answer apart from document provided."""
    vertexai.init(project="elp-2022-352222", location="us-central1")
    model = GenerativeModel(
        "gemini-1.5-flash-002",
        system_instruction=[vAR_system_instruction]
    )
    responses = model.generate_content(
        [document1, input],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    
    result = ""
    for response in responses:
        print("Vertext AI Raw response - ",response)
        result = result+response.text

    print("Vertext result in generate - ",result)

    
    st.session_state.vertex_request_id = st.session_state.vertex_response_id+1
    st.session_state.vertex_response_id = st.session_state.vertex_response_id+2

    return result,st.session_state.vertex_session_id,st.session_state.vertex_request_id,st.session_state.vertex_response_id,vAR_retrieved_text

