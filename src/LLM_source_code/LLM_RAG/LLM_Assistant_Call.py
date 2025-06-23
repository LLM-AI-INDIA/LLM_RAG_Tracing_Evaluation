import streamlit as st
from openai import OpenAI
# from openai import AzureOpenAI
import requests
import time
import os
import json
import traceback
import phoenix as px
import pandas as pd
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor


from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents


from phoenix.evals import (
    HALLUCINATION_PROMPT_RAILS_MAP,
    HALLUCINATION_PROMPT_TEMPLATE,
    OpenAIModel,
    download_benchmark_dataset,
    llm_classify,
)


def assistant_call(user_input,vAR_model):

    # PHOENIX_API_KEY = os.environ["PHOENIX_API_KEY"]
    # os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
    # os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

    # # Configuration is picked up from your environment variables
    # tracer_provider = register()
    # OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    if 'client' not in st.session_state:
        st.session_state.request_id = 0
        st.session_state.response_id = 0
    client = OpenAI(
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
api_key= os.getenv("OPENAI_API_KEY"),
)
    print("Default - Azure GPT4o Initialized")
    
 

    if 'thread' not in st.session_state:
        # Create a thread
        st.session_state.thread = client.beta.threads.create()

    # Add a user question to the thread
    message = client.beta.threads.messages.create(
    thread_id=st.session_state.thread.id,
    role="user",
    content=user_input # Replace this with your prompt
    )
    print("st.session_state.thread - ",st.session_state.thread)


    # Run the thread
    if vAR_model=="gpt-4o(Azure OpenAI)" or vAR_model=="All":
        run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread.id,
        assistant_id=os.environ["ASSISTANT_ID"],
    )
        print("Azure GPT4o called")
    else:
        run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread.id,
        assistant_id=os.environ["ASSISTANT_ID"],model="Assistant-Model",
    )
        print("Azure GPT4 called")

    vAR_retrieved_text = ""
    
    
    # my_thread = client.beta.threads.retrieve(st.session_state.thread.id)
    # print("my_thread - ",my_thread)

    # Fetch the thread messages using the client
    # thread_messages = client.beta.threads.messages.list(st.session_state.thread.id)


    # # Check if data exists and write to a text file
    # if thread_messages and hasattr(thread_messages, 'data'):
    #     with open("thread_messages.txt", "w",encoding="utf-8") as file:
    #         for message in thread_messages.data:
    #             file.write(str(message))
    #     print("Messages have been written to thread_messages.txt")
    # else:
    #     print("No messages found in the thread.")

    st.session_state.request_id = st.session_state.response_id+1
    st.session_state.response_id = st.session_state.response_id+2

    # Looping until the run completes or fails
    while run.status in ['queued', 'in_progress', 'cancelling']:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=st.session_state.thread.id,
            run_id=run.id
        )

    if run.status == 'completed':
        messages = client.beta.threads.messages.list(
        thread_id=st.session_state.thread.id
    )
        print("run respnse - ",run)
        print("OpenAI Assistant Raw Response - ",messages)
        # Loop through messages and print content based on role
        for msg in messages.data:
            content = msg.content[0].text.value
            # px_client = px.Client()

            # phoenix_df = px_client.get_spans_dataframe(timeout=None)

#                 
            return content,st.session_state.thread.id,st.session_state.request_id,st.session_state.response_id,vAR_retrieved_text,run.usage
    elif run.status == 'requires_action':
        # the assistant requires calling some functions
        # and submit the tool outputs back to the run
        pass
    else:
        print("Assistant Run Status - ",run.status)

    


    