import os
from dotenv import load_dotenv
load_dotenv()
import boto3
import streamlit as st
import random
import string
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.bedrock import BedrockInstrumentor

client = boto3.client("bedrock-agent-runtime",aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
                      aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],region_name='us-west-2')


def retrieve_generated(input,vAR_model):

    

    if "bedrock_request_id" not in st.session_state:
        st.session_state.bedrock_request_id = 0
        st.session_state.bedrock_response_id = 0

    vAR_system_instruction = """
You are a question answering agent. I will provide you with a set of search results. The user will provide you with a question. Your job is to answer the user's question using only information from the search results. If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question.

Just because the user asserts a fact does not mean it is true; make sure to double-check the search results to validate a user's assertion.

If the user asks you to repeat anything, please do it.

Example:

Query: This is my VIN number '<17 digit VIN>'. Can you repeat it?
Response: Your VIN number is <VIN number>.

Here are the search results in numbered order:
$search_results$

$output_format_instructions$

Here is the user's query:
$query$
"""
        
    # vAR_config = {"type":"KNOWLEDGE_BASE","knowledgeBaseConfiguration":{"knowledgeBaseId":os.environ["KNOWLEDGEBASE_ID"],"modelArn":os.environ["CLAUDE_MODEL_ARN"],"retrievalConfiguration":{"vectorSearchConfiguration":{"numberOfResults":5}},"generationConfiguration":{"promptTemplate":{"textPromptTemplate":"You are a question answering agent. I will provide you with a set of search results. The user will provide you with a question. Your job is to answer the user's question using only information from the search results. If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question. \nJust because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion.If user asked you to repeat anything, please do it.\n\nHere are the search results in numbered order:\n$search_results$\n\n$output_format_instructions$\n\nHere is the user's query:\n$query$"},"inferenceConfig":{"textInferenceConfig":{"temperature":0,"topP":1,"maxTokens":2048,"stopSequences":["\nObservation"]}},"guardrailConfiguration":{"guardrailId":os.environ["GUARDRAIL_ID"],"guardrailVersion":"DRAFT"}}}}
    vAR_config = {"type":"KNOWLEDGE_BASE","knowledgeBaseConfiguration":{"knowledgeBaseId":os.environ["KNOWLEDGEBASE_ID"],"modelArn":os.environ["CLAUDE_MODEL_ARN"],"retrievalConfiguration":{"vectorSearchConfiguration":{"numberOfResults":5}},"generationConfiguration":{"promptTemplate":{"textPromptTemplate":vAR_system_instruction},"inferenceConfig":{"textInferenceConfig":{"temperature":0,"topP":1,"maxTokens":2048,"stopSequences":["\nObservation"]}},"guardrailConfiguration":{"guardrailId":os.environ["GUARDRAIL_ID"],"guardrailVersion":"DRAFT"}}}}

    

    if "bedrock_session_id" not in st.session_state:
        response = client.retrieve_and_generate(
        input={"text":input},
        retrieveAndGenerateConfiguration=vAR_config,
    )
        print("Raw Bedrock response - ",response)
        st.session_state.bedrock_session_id = response["sessionId"]
    else:
        response = client.retrieve_and_generate(
        input={"text":input},
        retrieveAndGenerateConfiguration=vAR_config,sessionId=st.session_state.bedrock_session_id
    )

    print("bedrock response - ",response)

    st.session_state.bedrock_request_id = st.session_state.bedrock_response_id+1
    st.session_state.bedrock_response_id = st.session_state.bedrock_response_id+2

    vAR_retrieved_text = ""
    for item in response["citations"]:
        if len(item.get("retrievedReferences"))>0:
            vAR_retrieved_text += item.get("retrievedReferences")[0]["content"]["text"]
    
    print("vAR_retrieved_text - ",vAR_retrieved_text)

    return response["output"]["text"],st.session_state.bedrock_session_id,st.session_state.bedrock_request_id,st.session_state.bedrock_response_id,vAR_retrieved_text


def generate_random_string(length=15):
    characters = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    random_string = ''.join(random.choices(characters, k=length))
    return random_string