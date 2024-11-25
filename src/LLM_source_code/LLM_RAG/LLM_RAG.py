import streamlit as st
from streamlit_chat import message
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from src.LLM_source_code.LLM_RAG.LLM_Assistant_Call import  assistant_call
from src.LLM_source_code.LLM_RAG.LLM_Bedrock_Call import  retrieve_generated
from src.LLM_source_code.LLM_RAG.LLM_VertexAI_Call import generate
import datetime
from google.cloud import bigquery
import concurrent.futures
import pandas as pd
import webbrowser
import traceback
from src.LLM_source_code.LLM_RAG.LLM_Evaluation import process_single_response
from src.LLM_source_code.LLM_RAG.LLM_Guardrails_Call import guardrails_call
import multiprocessing as mp
from functools import partial
from multiprocessing import Pool
from src.LLM_source_code.LLM_RAG.LLM_Bedrock_Agent_Call import bedrock_agent_chat




from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from src.LLM_source_code.LLM_RAG.LLM_Custom_Eval import Custom_Eval_Context_Precision,Custom_Eval_Context_Recall





def ConversationWoEval(vAR_model):
    
    # Initialize session state
    if 'history_woeval' not in st.session_state:
        st.session_state.vAR_bedrock_response_list_wo_eval = []
        st.session_state['history_woeval'] = []

    if 'generated_woeval' not in st.session_state:
        st.session_state['generated_woeval'] = ["Greetings! I am LLMAI Live Agent. How can I help you?"]

    if 'past_woeval' not in st.session_state:
        st.session_state['past_woeval'] = ["We are delighted to have you here in the LLMAI Live Agent Chat room!"]

    # Container for the chat history
    response_container = st.container()
    container = st.container()
    vAR_response = None
    vAR_response_bedrock = None


    with container:
        with st.form(key='my_form', clear_on_submit=True):
            vAR_user_input = st.text_input("Prompt:", placeholder="How can I help you?", key='input')
            submit_button = st.form_submit_button(label='Interact with LLM')



        if submit_button and vAR_user_input and vAR_user_input!='':
            vAR_guard_input_response = guardrails_call(vAR_user_input,'INPUT')
            

            # Generate response from the agent
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # vAR_assistant = executor.submit(assistant_call,vAR_user_input,vAR_model)
                vAR_bedrock = executor.submit(retrieve_generated,vAR_user_input,vAR_model)
                # vAR_vertex = executor.submit(generate,vAR_user_input)

                # vAR_assistant_response,assistant_thread_id,assistant_request_id,assistant_response_id,vAR_retrieved_text_openai,vAR_run_usage = vAR_assistant.result()
                vAR_response_bedrock,bedrock_thread_id,bedrock_request_id,bedrock_response_id,vAR_retrieved_text_bedrock = vAR_bedrock.result()
                # vAR_response_vertex,vertex_thread_id,vertex_request_id,vertex_response_id,vAR_retrieved_text_gemini = vAR_vertex.result()

                vAR_guard_output_response = guardrails_call(vAR_response_bedrock,'OUTPUT')

                if vAR_guard_output_response.get("action")=="GUARDRAIL_INTERVENED":
                    # st.warning("Guardrail Intervened in Below Policy! for model response")
                    st.write("")
                    # st.json(vAR_guard_input_response.get("assessments"))
                    vAR_response_bedrock = vAR_guard_output_response.get("outputs")[0]["text"]

                if vAR_guard_input_response.get("action")=="GUARDRAIL_INTERVENED":
                    # st.warning("Guardrail Intervened in Below Policy! for user prompt")
                    st.write("")
                    # st.json(vAR_guard_input_response.get("assessments"))
                    vAR_response_bedrock = vAR_guard_input_response.get("outputs")[0]["text"]

                # print("vAR_assistant_result - ",vAR_assistant_response)
                print("vAR_bedrock_result - ",vAR_response_bedrock)
                # print("vAR_vertex_result - ",vAR_response_vertex)
            

            # st.session_state.vAR_assistant_response_list.append(vAR_assistant_response)
            st.session_state.vAR_bedrock_response_list_wo_eval.append(vAR_response_bedrock)
            # st.session_state.vAR_vertex_response_list.append(vAR_response_vertex)
            
            # vAR_final_df = pd.DataFrame({"OpenAI":st.session_state.vAR_assistant_response_list,"Bedrock":st.session_state.vAR_bedrock_response_list,"Google":st.session_state.vAR_vertex_response_list})
            vAR_final_df = pd.DataFrame({"Bedrock":st.session_state.vAR_bedrock_response_list_wo_eval})

            st.write("")
            col1,col2,col3,col4 = st.columns([7,3,5,1])
            with col1:
                st.markdown("<h3 style='font-size:18px;'>LLM Response Summary</h3>", unsafe_allow_html=True)
            with col2:
                st.link_button("Report View","https://lookerstudio.google.com/reporting/f7586dea-e417-44c9-bc6b-f5ba3dee09ee")

            # vAR_final_df = vAR_final_df.reset_index().rename(columns={'index': '#'})
            
            # vAR_final_df.style.hide(axis="index")

            st.table(vAR_final_df)



    
            

            # Bigquery Insert
            # if vAR_model=="All" or vAR_model=="gpt-4o(Azure OpenAI)":
            #     Bigquery_Insert(assistant_thread_id,vAR_user_input,vAR_assistant_response,assistant_request_id,assistant_response_id,"OpenAI-Assistant-GPT-4o")
            # elif vAR_model=="gpt-4(Azure OpenAI)":
            #     Bigquery_Insert(assistant_thread_id,vAR_user_input,vAR_assistant_response,assistant_request_id,assistant_response_id,"OpenAI-Assistant-GPT-4")
            
            # Below code can be changes later
            Bigquery_Insert("001",vAR_user_input,vAR_response_bedrock,1,2,"Anthropic-Claude-3.5-Sonnet")
            # Bigquery_Insert(vertex_thread_id,vAR_user_input,vAR_response_vertex,vertex_request_id,vertex_response_id,"Gemini-1.5-Flash")
            
            # # Eval Insertion
            # Bigquery_Eval_Insert(vAR_final_eval_df2)

            st.session_state['past_woeval'].append(vAR_user_input)
            st.session_state['generated_woeval'].append(vAR_response_bedrock)
        
        # Thumbs color changes
        custom_css = """
    <style>
    """
        for i in range(1,52):
            custom_css += f"""#root > div:nth-child(1) > div.withScreencast > div > div > div > section.stMain.st-emotion-cache-bm2z3a.ea3mdgi8 > div.stMainBlockContainer.block-container.st-emotion-cache-1jicfl2.ea3mdgi5 > div > div > div > div:nth-child(12) > div > div > div.stElementContainer.element-container.st-key-feedback_{i}.st-emotion-cache-9p65df.e1f1d6gn4 > div > div > button:nth-child(1) > span > span"""
            # Add a comma unless it's the last iteration
            if i < 50:
                custom_css += ",\n"
            elif i>50:
                custom_css+="\n"

        custom_css += """{
    color: green !important;
    font-size: x-large !important;
}
"""

        # Loop again for button:nth-child(2)
        for i in range(1, 52):
            # Append the CSS selectors for button:nth-child(2)
                          
            custom_css += f"""#root > div:nth-child(1) > div.withScreencast > div > div > div > section.stMain.st-emotion-cache-bm2z3a.ea3mdgi8 > div.stMainBlockContainer.block-container.st-emotion-cache-1jicfl2.ea3mdgi5 > div > div > div > div:nth-child(12) > div > div > div.stElementContainer.element-container.st-key-feedback_{i}.st-emotion-cache-9p65df.e1f1d6gn4 > div > div > button:nth-child(2) > span > span"""
            
            # Add a comma unless it's the last iteration
            if i < 50:
                custom_css += ",\n"
            elif i>50:
                custom_css+="\n"

        # Start the CSS rule for button:nth-child(2)
        custom_css += """{
            color: red !important;
            font-size: x-large !important;
            padding: 10px;
        }
        </style>"""

        st.markdown(custom_css,unsafe_allow_html=True)


        if st.session_state['generated_woeval']:
            with response_container:
                st.write("")
                st.write("")
                for i in range(len(st.session_state['generated_woeval'])):
                    message(st.session_state["past_woeval"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated_woeval"][i], key=str(i+55), avatar_style="thumbs")

                    # Skip feedback for predefined messages
                    if st.session_state["past_woeval"][i] != st.session_state["past_woeval"][0] and st.session_state["generated_woeval"][i] != st.session_state["generated_woeval"][0]:
                        feedback = st.feedback("thumbs", key=f"feedback_{i}")
                        if feedback is not None:
                            feedback_text = "thumbs up" if feedback == 1 else "thumbs down"
                            feedback_class = "thumbs-up" if feedback == 1 else "thumbs-down"
                            st.markdown(f'<div class="{feedback_class}">Thank you for your feedback! You rated this response with a {feedback_text}.</div>', unsafe_allow_html=True)
    return vAR_response








def ConversationWithEval(vAR_model):
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.vAR_assistant_response_list = []
        st.session_state.vAR_bedrock_response_list = []
        st.session_state.vAR_vertex_response_list = []
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Greetings! I am LLMAI Live Agent. How can I help you?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["We are delighted to have you here in the LLMAI Live Agent Chat room!"]

    # Container for the chat history
    response_container = st.container()
    container = st.container()

    # Initialize variables
    vAR_response = None
    vAR_assistant_response = None
    vAR_response_bedrock = None
    vAR_response_vertex = None
    assistant_thread_id = None
    bedrock_thread_id = None
    vertex_thread_id = None
    assistant_request_id = None
    bedrock_request_id = None
    vertex_request_id = None
    assistant_response_id = None
    bedrock_response_id = None
    vertex_response_id = None
    vAR_retrieved_text_openai = None
    vAR_retrieved_text_bedrock = None
    vAR_retrieved_text_gemini = None
    vAR_run_usage = None


    with container:
        with st.form(key='my_form_with_eval', clear_on_submit=True):
            vAR_user_input = st.text_input("Prompt:", placeholder="How can I help you?", key='input_with_eval')
            submit_button = st.form_submit_button(label='Interact with LLM')



        if submit_button and vAR_user_input and vAR_user_input!='':            

            # Generate response from the agent
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []

                if vAR_model in ["All", "gpt-4o(Azure OpenAI)"]:
                    futures.append(('assistant', executor.submit(assistant_call, vAR_user_input, vAR_model)))
                if vAR_model in ["gpt-4(Azure OpenAI)"]:
                    futures.append(('assistant', executor.submit(assistant_call, vAR_user_input, vAR_model)))
                if vAR_model in ["All", "claude-3.5-sonnet(Bedrock)"]:
                    futures.append(('bedrock', executor.submit(retrieve_generated, vAR_user_input, vAR_model)))
                if vAR_model in ["All", "gemini-1.5(Vertex AI)"]:
                    futures.append(('vertex', executor.submit(generate, vAR_user_input,vAR_model)))

                
                # Collect results
                for model_type, future in futures:
                    if model_type == 'assistant':
                        vAR_assistant_response, assistant_thread_id, assistant_request_id, assistant_response_id, vAR_retrieved_text_openai, vAR_run_usage = future.result()
                    elif model_type == 'bedrock':
                        vAR_response_bedrock, bedrock_thread_id, bedrock_request_id, bedrock_response_id, vAR_retrieved_text_bedrock = future.result()
                    elif model_type == 'vertex':
                        vAR_response_vertex, vertex_thread_id, vertex_request_id, vertex_response_id, vAR_retrieved_text_gemini = future.result()


    
                

                print("vAR_assistant_result - ",vAR_assistant_response)
                print("vAR_bedrock_result - ",vAR_response_bedrock)
                print("vAR_vertex_result - ",vAR_response_vertex)

            # Append responses to session state based on model selection
            st.session_state.vAR_assistant_response_list.append(vAR_assistant_response)
            st.session_state.vAR_bedrock_response_list.append(vAR_response_bedrock)
            st.session_state.vAR_vertex_response_list.append(vAR_response_vertex)
            

            # Create DataFrame based on available responses
            responses_dict = {}
            if vAR_model in ["All", "gpt-4o(Azure OpenAI)"]:
                responses_dict["OpenAI"] = st.session_state.vAR_assistant_response_list
            if vAR_model in ["gpt-4(Azure OpenAI)"]:
                responses_dict["OpenAI"] = st.session_state.vAR_assistant_response_list
            if vAR_model in ["All", "claude-3.5-sonnet(Bedrock)"]:
                responses_dict["Bedrock"] = st.session_state.vAR_bedrock_response_list
            if vAR_model in ["All", "gemini-1.5(Vertex AI)"]:
                responses_dict["Google"] = st.session_state.vAR_vertex_response_list

            vAR_final_df = pd.DataFrame(responses_dict)


            st.write("")
            col1,col2,col3,col4 = st.columns([7,3,5,1])
            with col1:
                st.markdown("<h3 style='font-size:18px;'>LLM Response Summary</h3>", unsafe_allow_html=True)
            with col2:
                st.link_button("Report View","https://lookerstudio.google.com/reporting/f7586dea-e417-44c9-bc6b-f5ba3dee09ee")

            # vAR_final_df = vAR_final_df.reset_index().rename(columns={'index': '#'})
            
            # vAR_final_df.style.hide(axis="index")

            st.table(vAR_final_df)



            # Prepare responses for evaluation
            vAR_responses = []
            if vAR_assistant_response:
                vAR_responses.append(("GPT", vAR_assistant_response, assistant_thread_id, vAR_retrieved_text_openai))
            if vAR_response_bedrock:
                vAR_responses.append(("Claude", vAR_response_bedrock, bedrock_thread_id, vAR_retrieved_text_bedrock))
            if vAR_response_vertex:
                vAR_responses.append(("Gemini", vAR_response_vertex, vertex_thread_id, vAR_response_vertex))

            if vAR_responses:
                process_func = partial(
            process_single_response,
            vAR_user_input=vAR_user_input,
            assistant_request_id=assistant_request_id,
            assistant_response_id=assistant_response_id
        )
        
                # Create a pool of workers
                with Pool(processes=4) as pool:
                    # Map the processing function to all responses
                    results = pool.map(process_func, vAR_responses)
                
                # Separate results into two lists
                vAR_eval_df_list = []
                vAR_eval_df_list2 = []
                
                for merged_df, merged_df2 in results:
                    if merged_df is not None and merged_df2 is not None:
                        vAR_eval_df_list.append(merged_df)
                        vAR_eval_df_list2.append(merged_df2)

                vAR_final_eval_df = pd.concat(vAR_eval_df_list,ignore_index=True)
                # vAR_final_eval_df = vAR_final_eval_df.reset_index().rename(columns={'index': '#'})
                # vAR_final_eval_df.style.hide(axis="index")

                vAR_final_eval_df2 = pd.concat(vAR_eval_df_list2,ignore_index=True)


                st.write("")
                col1,col2,col3,col4 = st.columns([7,3,5,1])
                with col1:
                    st.markdown("<h3 style='font-size:18px;'>Judge (GPT4) Reasoning and Evaluation Score</h3>", unsafe_allow_html=True)
                with col2:
                    st.link_button("Report View","https://lookerstudio.google.com/reporting/f7586dea-e417-44c9-bc6b-f5ba3dee09ee/page/p_55e0w4zfmd")

                st.table(vAR_final_eval_df)

                
                # Bigquery Insert based on model selection
                if vAR_model in ["All", "gpt-4(Azure OpenAI)","gpt-4o(Azure OpenAI)"]:
                    if vAR_model == "All" or vAR_model=="gpt-4o(Azure OpenAI)":
                        Bigquery_Insert(assistant_thread_id, vAR_user_input, vAR_assistant_response, 
                                        assistant_request_id, assistant_response_id, "OpenAI-Assistant-GPT-4o")
                    else:
                        Bigquery_Insert(assistant_thread_id, vAR_user_input, vAR_assistant_response, 
                                        assistant_request_id, assistant_response_id, "OpenAI-Assistant-GPT-4")
                
                if vAR_model in ["All", "claude-3.5-sonnet(Bedrock)"]:
                    Bigquery_Insert(bedrock_thread_id, vAR_user_input, vAR_response_bedrock, 
                                    bedrock_request_id, bedrock_response_id, "Anthropic-Claude-3.5-Sonnet")
                
                if vAR_model in ["All", "gemini-1.5(Vertex AI)"]:
                    Bigquery_Insert(vertex_thread_id, vAR_user_input, vAR_response_vertex, 
                                    vertex_request_id, vertex_response_id, "Gemini-1.5-Flash")

                # Eval Insertion
                Bigquery_Eval_Insert(vAR_final_eval_df2)




            st.session_state['past'].append(vAR_user_input)

            # Select appropriate response for chat display
            if (vAR_model == "gpt-4(Azure OpenAI)" or vAR_model == "gpt-4o(Azure OpenAI)") and vAR_assistant_response:
                chat_response = vAR_assistant_response
            elif vAR_model == "claude-3.5-sonnet(Bedrock)" and vAR_response_bedrock:
                chat_response = vAR_response_bedrock
            elif vAR_model == "gemini-1.5(Vertex AI)" and vAR_response_vertex:
                chat_response = vAR_response_vertex
            else:  # For "All" model or fallback
                chat_response = vAR_response_bedrock

            st.session_state['generated'].append(chat_response)

        
        # Thumbs color changes
        custom_css = """
    <style>
    """
        for i in range(1,52):
            custom_css += f"""#root > div:nth-child(1) > div.withScreencast > div > div > div > section.stMain.st-emotion-cache-bm2z3a.ea3mdgi8 > div.stMainBlockContainer.block-container.st-emotion-cache-1jicfl2.ea3mdgi5 > div > div > div > div:nth-child(12) > div > div > div.stElementContainer.element-container.st-key-feedback_{i}.st-emotion-cache-9p65df.e1f1d6gn4 > div > div > button:nth-child(1) > span > span"""
            # Add a comma unless it's the last iteration
            if i < 50:
                custom_css += ",\n"
            elif i>50:
                custom_css+="\n"

        custom_css += """{
    color: green !important;
    font-size: x-large !important;
}
"""

        # Loop again for button:nth-child(2)
        for i in range(1, 52):
            # Append the CSS selectors for button:nth-child(2)
                          
            custom_css += f"""#root > div:nth-child(1) > div.withScreencast > div > div > div > section.stMain.st-emotion-cache-bm2z3a.ea3mdgi8 > div.stMainBlockContainer.block-container.st-emotion-cache-1jicfl2.ea3mdgi5 > div > div > div > div:nth-child(12) > div > div > div.stElementContainer.element-container.st-key-feedback_{i}.st-emotion-cache-9p65df.e1f1d6gn4 > div > div > button:nth-child(2) > span > span"""
            
            # Add a comma unless it's the last iteration
            if i < 50:
                custom_css += ",\n"
            elif i>50:
                custom_css+="\n"

        # Start the CSS rule for button:nth-child(2)
        custom_css += """{
            color: red !important;
            font-size: x-large !important;
            padding: 10px;
        }
        </style>"""

        st.markdown(custom_css,unsafe_allow_html=True)


        if st.session_state['generated']:
            with response_container:
                st.write("")
                st.write("")
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i+55), avatar_style="thumbs")

                    # Skip feedback for predefined messages
                    if st.session_state["past"][i] != st.session_state["past"][0] and st.session_state["generated"][i] != st.session_state["generated"][0]:
                        feedback = st.feedback("thumbs", key=f"feedback_{i}")
                        if feedback is not None:
                            feedback_text = "thumbs up" if feedback == 1 else "thumbs down"
                            feedback_class = "thumbs-up" if feedback == 1 else "thumbs-down"
                            st.markdown(f'<div class="{feedback_class}">Thank you for your feedback! You rated this response with a {feedback_text}.</div>', unsafe_allow_html=True)
    return vAR_response








def LLM_RAG_Impl(choice):
    
    col1,col2,col3,col4,col5 = st.columns([1,5,1,5,1])
    col21,col22,col23,col24,col25 = st.columns([1,5,1,5,1])
    col16,col17,col18,col19,col20 = st.columns([1,5,1,5,1])


    col6,col7,col8,col9,col10 = st.columns([1,5,1,5,1])
    col11,col12,col13,col14,col15 = st.columns([1,5,1,5,1])
    col26,col27,col28,col29,col30 = st.columns([1,5,1,5,1])


    vAR_model = None
    vAR_questions = None
    vAR_knowledge_base = None
    vAR_guard_2ndlvl_category = None
    vAR_guard_category = None
    vAR_eval_llm = None
    vAR_eval_type = None

    with col2:
        st.write("")
        st.markdown("<h3 style='font-size:16px;'>Select Use Case</h3>", unsafe_allow_html=True)
    with col4:
        vAR_usecase = st.selectbox(" ",("Policy Guru","Multimodal RAG"))
        st.write("")

    if vAR_usecase=="Policy Guru" and choice=="LLM Guardrail":

        with col17:
            st.write("")
            st.markdown("<h3 style='font-size:16px;'>Select LLM</h3>", unsafe_allow_html=True)
        with col19:
            vAR_model = st.selectbox(" ",("claude-3.5-sonnet(Bedrock)"))
            st.write("")

        with col7:
            st.write("")
            st.markdown("<h3 style='font-size:16px;'>Select Platform</h3>", unsafe_allow_html=True)
        with col9:
            if vAR_model=="claude-3.5-sonnet(Bedrock)":
                vAR_platform = st.selectbox(" ",("AWS Bedrock"))
                st.write("")

        with col12:
                st.write("")
                st.markdown("<h3 style='font-size:16px;'>Select Guardrails Function</h3>", unsafe_allow_html=True)

        with col14:
            vAR_guard_category = st.selectbox(" ",("Sensitive Information filters","Word filters","Phrases filters","Denied Topics","Content filters","Prompt attacks"))
            st.write("")
        
        if vAR_guard_category=="Sensitive Information filters":
            with col27:
                st.write("")
                st.markdown("<h3 style='font-size:16px;'>Select Sensitive Data</h3>", unsafe_allow_html=True)

            with col29:
                vAR_guard_2ndlvl_category = st.selectbox(" ",("US Passport Number(Mask)","Vehicle Identification Number(VIN)(Mask)","US Social Security Number(SSN)(Block)","Password(Block)"))
                st.write("")
        elif vAR_guard_category=="Word filters":
            with col27:
                st.write("")
                st.markdown("<h3 style='font-size:16px;'>Select Word Filter Data</h3>", unsafe_allow_html=True)

            with col29:
                vAR_guard_2ndlvl_category = st.selectbox(" ",("LICENSE PLATE","SSN","VIN","DL","RN"))
                st.write("")
        elif vAR_guard_category=="Denied Topics":
            with col27:
                st.write("")
                st.markdown("<h3 style='font-size:16px;'>Select Denied Topic</h3>", unsafe_allow_html=True)

            with col29:
                vAR_guard_2ndlvl_category = st.selectbox(" ",("Terrorism","Racism","Age"))
                st.write("")
        elif vAR_guard_category=="Phrases filters":
            with col27:
                st.write("")
                st.markdown("<h3 style='font-size:16px;'>Select Phrase Filters</h3>", unsafe_allow_html=True)

            with col29:
                vAR_guard_2ndlvl_category = st.selectbox(" ",("Speeding Drivers","Return to India","Indefatigable India"))
                st.write("")
        elif vAR_guard_category=="Prompt attacks":
            with col27:
                st.write("")
                st.markdown("<h3 style='font-size:16px;'>Select Prompt Attack Type</h3>", unsafe_allow_html=True)

            with col29:
                vAR_guard_2ndlvl_category = st.selectbox(" ",("Role Playing Exploit","Malicious Input","Jailbreaking"))
                st.write("")

        elif vAR_guard_category=="Content filters":
            with col27:
                st.write("")
                st.markdown("<h3 style='font-size:16px;'>Select Prompt Attack Type</h3>", unsafe_allow_html=True)

            with col29:
                vAR_guard_2ndlvl_category = st.selectbox(" ",("Hate","Insult","Sexual","Violence","Misconduct"))
                st.write("")

        print("vAR_guard_category - ",vAR_guard_category)
        print("vAR_guard_2ndlvl_category - ",vAR_guard_2ndlvl_category)
        if vAR_guard_category and vAR_guard_2ndlvl_category:
            vAR_response = ConversationWoEval(vAR_model)
        else:
            st.warning("Please select proper Guardrails Options!")








    elif vAR_usecase=="Policy Guru" and choice=="LLM as a Judge":
        
        with col17:
            st.write("")
            st.markdown("<h3 style='font-size:16px;'>Select LLM</h3>", unsafe_allow_html=True)
        with col19:
            vAR_model = st.selectbox(" ",("All","gpt-4o(Azure OpenAI)","gpt-4(Azure OpenAI)","claude-3.5-sonnet(Bedrock)","gemini-1.5(Vertex AI)"))
            st.write("")

        with col7:
            st.write("")
            st.markdown("<h3 style='font-size:16px;'>Select Platform</h3>", unsafe_allow_html=True)
        with col9:
            if vAR_model=="All":
                vAR_platform = st.selectbox(" ",("All","Assistant(Azure OpenAI)","AWS Bedrock","Vertex AI(Gemini)"))
                st.write("")
            elif vAR_model=="gpt-4o(Azure OpenAI)" or vAR_model=="gpt-4(Azure OpenAI)":
                vAR_platform = st.selectbox(" ",("Assistant(Azure OpenAI)"))
                st.write("")
            elif vAR_model=="claude-3.5-sonnet(Bedrock)":
                vAR_platform = st.selectbox(" ",("AWS Bedrock"))
                st.write("")
            elif vAR_model=="gemini-1.5(Vertex AI)":
                vAR_platform = st.selectbox(" ",("Vertex AI(Gemini)"))
                st.write("")
        
        with col12:
            st.write("")
            st.markdown("<h3 style='font-size:16px;'>Select LLM as a Judge Model(Evaluator)</h3>", unsafe_allow_html=True)

        with col14:
            vAR_eval_llm = st.selectbox(" ",("GPT(Default)","Claude","Gemini"))
            st.write("")

        with col27:
                st.write("")
                st.markdown("<h3 style='font-size:16px;'>Select LLM as a Judge Type</h3>", unsafe_allow_html=True)

        with col29:
            vAR_eval_type = st.selectbox(" ",("Pairwise Comparison(Default)","Evaluation by criteria with reference","Evaluation by criteria without reference"))
            st.write("")

        if vAR_eval_llm and vAR_eval_type:
            vAR_response = ConversationWithEval(vAR_model)
        else:
            st.warning("Please select proper LLM as Judge Options!")

    elif vAR_usecase=="Policy Guru" and choice=="LLM Agent":

        with col17:
            st.write("")
            st.markdown("<h3 style='font-size:16px;'>Select LLM</h3>", unsafe_allow_html=True)
        with col19:
            vAR_model = st.selectbox(" ",("claude-3.5-sonnet(Bedrock)"))
            st.write("")

        with col7:
            st.write("")
            st.markdown("<h3 style='font-size:16px;'>Select Platform</h3>", unsafe_allow_html=True)
        with col9:
            if vAR_model=="claude-3.5-sonnet(Bedrock)":
                vAR_platform = st.selectbox(" ",("AWS Bedrock"))
                st.write("")

        
        bedrock_agent_chat()
            
            
  







def Bigquery_Insert(thread_id,vAR_user_input,vAR_response,request_id,response_id,model_name):

    if vAR_response:
        vAR_response = vAR_response.replace("\n","").replace("\"","'")
    vAR_user_input = vAR_user_input.replace("\n","").replace("\"","'")
    vAR_request_dict = {
        "THREAD_ID":thread_id,
        "REQUEST_ID":request_id,
        "REQUEST_DATE_TIME":datetime.datetime.utcnow(),
        "PROMPT":vAR_user_input,
        "MODEL_NAME":model_name,
        "KNOWLEDGEBASE":"DMV-FAQ.PDF",
        "CREATED_DT":datetime.datetime.utcnow(),
        "CREATED_USER":"LLM_USER",
        "UPDATED_DT":datetime.datetime.utcnow(),
        "UPDATED_USER":"LLM_USER",

    }
    vAR_response_dict = {
        "THREAD_ID":thread_id,
        "REQUEST_ID":request_id,
        "RESPONSE_ID":response_id,
        "REQUEST_DATE_TIME":datetime.datetime.utcnow(),
        "PROMPT":vAR_user_input,
        "MODEL_NAME":model_name,
        "MODEL_RESPONSE":vAR_response,
        "CREATED_DT":datetime.datetime.utcnow(),
        "CREATED_USER":"LLM_USER",
        "UPDATED_DT":datetime.datetime.utcnow(),
        "UPDATED_USER":"LLM_USER",
    }



    client = bigquery.Client(project="elp-2022-352222")
    request_table = "DMV_RAG_EVALUATION"+'.LLM_REQUEST'
    response_table = "DMV_RAG_EVALUATION"+'.LLM_RESPONSE'


    vAR_request_query = """
insert into `{}` values("{}",{},"{}","{}","{}","{}","{}","{}","{}","{}")

""".format(request_table,vAR_request_dict["THREAD_ID"],vAR_request_dict["REQUEST_ID"],vAR_request_dict["REQUEST_DATE_TIME"],
           vAR_request_dict["PROMPT"],vAR_request_dict["MODEL_NAME"],vAR_request_dict["KNOWLEDGEBASE"],vAR_request_dict["CREATED_DT"],vAR_request_dict["CREATED_USER"],vAR_request_dict["UPDATED_DT"],vAR_request_dict["UPDATED_USER"])
    

    vAR_response_query = """
insert into `{}` values("{}",{},{},"{}","{}","{}","{}","{}","{}","{}","{}")

""".format(response_table,vAR_response_dict["THREAD_ID"],vAR_response_dict["REQUEST_ID"],vAR_response_dict["RESPONSE_ID"],vAR_response_dict["REQUEST_DATE_TIME"],
           vAR_response_dict["PROMPT"],vAR_response_dict["MODEL_NAME"],vAR_response_dict["MODEL_RESPONSE"],vAR_response_dict["CREATED_DT"],vAR_response_dict["CREATED_USER"],vAR_response_dict["UPDATED_DT"],vAR_response_dict["UPDATED_USER"])
    
    
    print('Insert request table query - ',vAR_request_query)
    print('Insert response table query - ',vAR_response_query)

    
    vAR_job = client.query(vAR_request_query)
    vAR_job.result()
    vAR_num_of_inserted_row = vAR_job.num_dml_affected_rows
    print('Number of rows inserted into request table - ',vAR_num_of_inserted_row)


    vAR_job = client.query(vAR_response_query)
    vAR_job.result()
    vAR_num_of_inserted_row = vAR_job.num_dml_affected_rows
    print('Number of rows inserted into response table - ',vAR_num_of_inserted_row)

    print("Records Successfully inserted into bigquery table")


def Bigquery_Eval_Insert(vAR_eval_df):
    vAR_eval_df.rename(columns={'label': 'EVALUATION_PARAM', 'score': 'EVALUATION_PARAM_VALUE','explanation':'EVALUATION_EXPLANATION','Metrics Category':'METRICS_CATEGORY'}, inplace=True)
    vAR_eval_df['EVALUATION_PARAM_VALUE'] = vAR_eval_df['EVALUATION_PARAM_VALUE'].astype("string")
    vAR_eval_df["EVALUATION_ID"] = vAR_eval_df["REQUEST_ID"]
    vAR_eval_df["EVALUATION_DATE_TIME"] = len(vAR_eval_df)*[datetime.datetime.utcnow()]
    vAR_eval_df["CREATED_DT"] = len(vAR_eval_df)*[datetime.datetime.utcnow()]
    vAR_eval_df["UPDATED_DT"] = len(vAR_eval_df)*[datetime.datetime.utcnow()]
    vAR_eval_df["CREATED_USER"] = len(vAR_eval_df)*["LLM_USER"]
    vAR_eval_df["UPDATED_USER"] = len(vAR_eval_df)*["LLM_USER"]

    client = bigquery.Client(project="elp-2022-352222")

    table = "DMV_RAG_EVALUATION"+'.'+'LLM_EVALUATION'
    job_config = bigquery.LoadJobConfig(autodetect=True,write_disposition="WRITE_APPEND")
    # job_config = bigquery.LoadJobConfig(autodetect=True,write_disposition="WRITE_APPEND",source_format=bigquery.SourceFormat.CSV,max_bad_records=vAR_number_of_configuration,allowJaggedRows=True)
    job = client.load_table_from_dataframe(vAR_eval_df, table,job_config=job_config)

    job.result()  # Wait for the job to complete.
    table_id = "elp-2022-352222"+'.'+table
    table = client.get_table(table_id)  # Make an API request.
    print(
            "Evaluation Table Loaded {} rows and {} columns to {}".format(
                table.num_rows, len(table.schema), table_id
            )
        )
   

def open_report_link():
    report_link = "https://lookerstudio.google.com/reporting/f7586dea-e417-44c9-bc6b-f5ba3dee09ee"
    webbrowser.open(report_link)

def token_analysis(data):
    print("Data for graph - ",data)

    # Display the plot in Streamlit
    # Create figure and axis objects
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot first set of bars
    x = np.arange(len(data['query']))
    bar_width = 0.35
    bars1 = ax1.bar(x - bar_width/2, data['prompt_tokens'], bar_width, 
                    label='Count(prompt_tokens)', color='skyblue')
    
    # Plot second set of bars
    bars2 = ax1.bar(x + bar_width/2, data['response_tokens'], bar_width,
                    label='Count(response_tokens)', color='lightgreen')
    
    # Customize the plot
    ax1.set_xlabel('Query')
    ax1.set_ylabel('Counts')
    ax1.set_title('Comparison of Counts by Query')
    ax1.set_xticks(x)
    ax1.set_xticklabels(data['query'])
    ax1.legend()
    
    # Add value labels on top of bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    st.pyplot(fig)

