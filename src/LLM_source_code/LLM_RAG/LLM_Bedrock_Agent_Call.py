import boto3
import os
import streamlit as st
from src.LLM_source_code.LLM_RAG.LLM_Bedrock_Call_KB import generate_random_string
import traceback
import json
from streamlit_chat import message
import matplotlib.pyplot as plt
import io


def agent_call(vAR_user_input,vAR_file_obj,vAR_source):

    if "agent_session_id" not in st.session_state:
        st.session_state.agent_session_id=generate_random_string()


    try:
        client = boto3.client("bedrock-agent-runtime",aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
                        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],region_name='us-west-2')

        if vAR_file_obj and vAR_source=="UPLOAD":

            response = client.invoke_agent(
        agentId=os.environ["AGENT_ID"],
        enableTrace = True,
        inputText=vAR_user_input,
        agentAliasId=os.environ["AGENT_ALIAS_ID"],
        sessionId=st.session_state.agent_session_id,
        sessionState={
        'files': [
            {
                'name': vAR_file_obj.name,
                'source': {
                    'byteContent': {
                        'data': vAR_file_obj.read(),
                        'mediaType': vAR_file_obj.type
                    },
                    'sourceType': 'BYTE_CONTENT'
                },
                'useCase': 'CODE_INTERPRETER'
            }
        ]
        }
        )

        elif vAR_file_obj and vAR_source=="S3":
            print("s3 filename - ",vAR_file_obj.split("/")[-1])
            response = client.invoke_agent(
        agentId=os.environ["AGENT_ID"],
        enableTrace = True,
        inputText=vAR_user_input,
        agentAliasId=os.environ["AGENT_ALIAS_ID"],
        sessionId=st.session_state.agent_session_id,
        
        sessionState={
        'files': [
            {
                'name': vAR_file_obj.split("/")[-1],
                'source': {
                    's3Location':{
                        'uri':vAR_file_obj
                    },
                    
                    'sourceType': 'S3'
                },
                'useCase': 'CODE_INTERPRETER'
            }
        ]
        }
        )
        else:
            response = client.invoke_agent(
        agentId=os.environ["AGENT_ID"],
        enableTrace = True,
        inputText=vAR_user_input,
        agentAliasId=os.environ["AGENT_ALIAS_ID"],
        sessionId=st.session_state.agent_session_id)
        
        print("Agent response - ",response)

        event_stream = response['completion']
        final_answer = None
        vAR_trace_list = []
        vAR_file_generated = False
        file_name = None
        idx = 0
        idx2 = 0
        try:
            for event in event_stream:
                idx +=1
                print("idx - ",idx)
                if 'chunk' in event:
                    data = event['chunk']['bytes']
                    final_answer = data.decode('utf8')
                    print(f"Final answer ->\n{final_answer}")
                    end_event_received = True
                if 'trace' in event:
                    vAR_trace_obj = json.dumps(event['trace'])
                    vAR_trace_list.append(vAR_trace_obj)

                # files contains intermediate response for code interpreter if any files have been generated.
                if 'files' in event:
                    files = event['files']['files']
                    for file in files:
                        file_name = file['name']
                        type = file['type']
                        bytes_data = file['bytes']
                        
                        # It the file is a PNG image then we can display it...
                        if type == 'image/png':
                            # Here, we are facing some problem with plotting graphs. So, currently keeping 2 as static value, later this can be changed
                            idx2 +=1
                            print("idx2 - ",idx2)
                            if idx2 ==2:
                                # Display PNG image using Matplotlib in Streamlit
                                img = plt.imread(io.BytesIO(bytes_data))
                                fig, ax = plt.subplots(figsize=(5, 5))
                                ax.imshow(img)
                                ax.axis('off')
                                ax.set_title(file_name)
                                st.pyplot(fig)  # Use st.pyplot() instead of plt.show()
                                plt.close(fig)  # Close the figure to free up memory
                            
                        # If the file is NOT a PNG then we save it to disk...
                        else:
                            # Save other file types to local disk
                            vAR_file_generated = True
                            with open(file_name, 'wb') as f:
                                f.write(bytes_data)
                            print(f"File '{file_name}' saved to disk.")
                    
        except Exception as e:
            raise Exception("unexpected event.",e)

    except BaseException as e:
        print("Agent error - ",str(e))
        print(traceback.format_exc())

    return final_answer,vAR_trace_list,vAR_file_generated,file_name
        
    



def bedrock_agent_chat(vaR_file_obj,vAR_source):
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
    vAR_file_generated = False

    if "vAR_trace_list" not in st.session_state:
        st.session_state.vAR_trace_list = []

    with container:
        with st.form(key='my_agent_form', clear_on_submit=True):
            vAR_user_input = st.text_input("Prompt:", placeholder="How can I help you?", key='agent_input')
            submit_button = st.form_submit_button(label='Interact with LLM')



        if submit_button and vAR_user_input and vAR_user_input!='':
            vAR_agent_response,vAR_trace_obj_list,vAR_file_generated,file_name = agent_call(vAR_user_input,vaR_file_obj,vAR_source)

            st.session_state.vAR_trace_list = vAR_trace_obj_list

            st.session_state['past_agent'].append(vAR_user_input)
            print("vAR_agent_response - ",vAR_agent_response)
            print("type vAR_agent_response - ",type(vAR_agent_response))
            print("st.session_state.vAR_trace_list - ",st.session_state.vAR_trace_list)
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
                            st.write("")
                if len(st.session_state['generated_agent'])>1:
                    if st.button("Click Here for Trace!",key="button_key"):
                        st.write("Tracing Details : ")
                        st.json({"Trace Details" : st.session_state.vAR_trace_list})
                        st.write("")
                if vAR_file_generated:
                    # Read the file content
                    with open(file_name, "rb") as file:
                        file_content = file.read()

                    # Button to download the file
                    st.download_button(
                        label="Download File",             # Label on the button
                        data=file_content,                 # File content to download
                        file_name=file_name,               # Default file name
                        mime="text/plain"                  # MIME type
                    )


