import boto3
import os
import streamlit as st
from src.LLM_source_code.LLM_RAG.LLM_Bedrock_Call import generate_random_string
import traceback
import json
from streamlit_chat import message


def agent_call(vAR_user_input):

    if "agent_session_id" not in st.session_state:
        st.session_state.agent_session_id=generate_random_string()


    try:
        client = boto3.client("bedrock-agent-runtime",aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
                        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],region_name='us-west-2')

        response = client.invoke_agent(
        agentId=os.environ["AGENT_ID"],
        inputText=vAR_user_input,
        agentAliasId=os.environ["AGENT_ALIAS_ID"],
        sessionId=st.session_state.agent_session_id)
        
        print("Agent response - ",response)

        event_stream = response['completion']
        final_answer = None
        try:
            for event in event_stream:
                if 'chunk' in event:
                    data = event['chunk']['bytes']
                    final_answer = data.decode('utf8')
                    print(f"Final answer ->\n{final_answer}")
                    end_event_received = True
                elif 'trace' in event:
                    print(json.dumps(event['trace'], indent=2))
                else: 
                    print("event - ",event)
                    raise Exception("unexpected event.", event)
        except Exception as e:
            raise Exception("unexpected event.",e)

    except BaseException as e:
        print("Agent error - ",str(e))
        print(traceback.format_exc())

    return final_answer
    



def bedrock_agent_chat():
    # Initialize session state
    if 'history_agent' not in st.session_state:
        st.session_state['history_agent'] = []

    if 'generated_agent' not in st.session_state:
        st.session_state['generated_agent'] = ["Greetings! I am LLMAI Live Agent. How can I help you?"]

    if 'past_agent' not in st.session_state:
        st.session_state['past_agent'] = ["We are delighted to have you here in the LLMAI Live Agent Chat room!"]

    # Container for the chat history
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_agent_form', clear_on_submit=True):
            vAR_user_input = st.text_input("Prompt:", placeholder="How can I help you?", key='agent_input')
            submit_button = st.form_submit_button(label='Interact with LLM')



        if submit_button and vAR_user_input and vAR_user_input!='':
            vAR_agent_response = agent_call(vAR_user_input)

            st.session_state['past_agent'].append(vAR_user_input)
            print("vAR_agent_response - ",vAR_agent_response)
            print("type vAR_agent_response - ",type(vAR_agent_response))
            st.session_state['generated_agent'].append(vAR_agent_response)

        if st.session_state['generated_agent']:
            with response_container:
                st.write("")
                st.write("")
                for i in range(len(st.session_state['generated_agent'])):
                    message(st.session_state["past_agent"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated_agent"][i], key=str(i+55), avatar_style="thumbs")

                    # Skip feedback for predefined messages
                    if st.session_state["past_agent"][i] != st.session_state["past_agent"][0] and st.session_state["generated_agent"][i] != st.session_state["generated_agent"][0]:
                        feedback = st.feedback("thumbs", key=f"feedback_{i}")
                        if feedback is not None:
                            feedback_text = "thumbs up" if feedback == 1 else "thumbs down"
                            feedback_class = "thumbs-up" if feedback == 1 else "thumbs-down"
                            st.markdown(f'<div class="{feedback_class}">Thank you for your feedback! You rated this response with a {feedback_text}.</div>', unsafe_allow_html=True)

