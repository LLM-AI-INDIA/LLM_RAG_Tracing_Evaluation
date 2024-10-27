import streamlit as st
from openai import OpenAI
import time
import os
import json
import traceback
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor


def assistant_call(user_input):

    # PHOENIX_API_KEY = os.environ["PHOENIX_API_KEY"]
    # os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
    # os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

    # # Configuration is picked up from your environment variables
    # tracer_provider = register()
    # OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    
    
    
    if 'client' not in st.session_state:
        st.session_state.request_id = 0
        st.session_state.response_id = 0
        st.session_state.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])



        file_create_response = st.session_state.client.files.create(
    file=open("DMV-FAQ.pdf","rb"),
    purpose="assistants"
    )       

    if 'thread' not in st.session_state:
        st.session_state.thread = st.session_state.client.beta.threads.create(messages=[
    {
      "role": "user",
      "content": user_input,
      # Attach the new file to the message.
      "attachments": [
        { "file_id": file_create_response.id, "tools": [{"type": "file_search"}] }
      ],
    }
  ])

        print("st.session_state.thread - ",st.session_state.thread)
    # message = st.session_state.client.beta.threads.messages.create(thread_id=st.session_state.thread.id,role="user",content=user_input)
    run = st.session_state.client.beta.threads.runs.create(thread_id=st.session_state.thread.id,assistant_id=os.environ["ASSISTANT_ID"],max_completion_tokens=2000)
    
    st.session_state.request_id = st.session_state.response_id+1
    st.session_state.response_id = st.session_state.response_id+2

    while True:
        time.sleep(2)
        
        # Retrieve the run status
        run_status = st.session_state.client.beta.threads.runs.retrieve(thread_id=st.session_state.thread.id,run_id=run.id)
        
        print('run status - ',run_status.model_dump_json(indent=4))

        # If run is completed, get messages
        if run_status.status == 'completed':
            messages = st.session_state.client.beta.threads.messages.list(
                thread_id=st.session_state.thread.id
            )
            # Loop through messages and print content based on role
            for msg in messages.data:
                role = msg.role
                content = msg.content[0].text.value
                print(f"{role.capitalize()}: {content}")
                # px_client = px.Client()

                # phoenix_df = px_client.get_spans_dataframe()

                return content,st.session_state.thread.id,st.session_state.request_id,st.session_state.response_id
        
        else:
            print("Waiting for the Assistant to process...")
            time.sleep(2)

    


    