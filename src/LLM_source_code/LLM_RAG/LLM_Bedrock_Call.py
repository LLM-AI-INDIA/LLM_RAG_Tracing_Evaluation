import os

import boto3
import streamlit as st
import random
import string
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.bedrock import BedrockInstrumentor

client = boto3.client("bedrock-agent-runtime",aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
                      aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],region_name='us-west-2')


def retrieve_generated(input):

    

    if "bedrock_request_id" not in st.session_state:
        st.session_state.bedrock_request_id = 0
        st.session_state.bedrock_response_id = 0
        st.session_state.vAR_session_id = generate_random_string()

    response = client.retrieve_and_generate(
        input={"text":input},
        retrieveAndGenerateConfiguration={
            
            "type":"KNOWLEDGE_BASE","knowledgeBaseConfiguration":{"knowledgeBaseId":os.environ["KNOWLEDGEBASE_ID"],"modelArn":os.environ["MODEL_ARN"],"retrievalConfiguration":{"vectorSearchConfiguration":{"numberOfResults":5}},"generationConfiguration":{"inferenceConfig":{"textInferenceConfig":{"temperature":0,"topP":1,"maxTokens":2048,"stopSequences":["\nObservation"]}}}}
        }
    )

    print("bedrock response - ",response)

    st.session_state.bedrock_request_id = st.session_state.bedrock_response_id+1
    st.session_state.bedrock_response_id = st.session_state.bedrock_response_id+2

    

    return response["output"]["text"],st.session_state.vAR_session_id,st.session_state.bedrock_request_id,st.session_state.bedrock_response_id


def generate_random_string(length=15):
    characters = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    random_string = ''.join(random.choices(characters, k=length))
    return random_string