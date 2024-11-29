import streamlit as st
import pandas as pd
from streamlit_feedback import streamlit_feedback
from src.LLM_source_code.LLM_HumanFeedbackLoop_v2.query import get_expected_response
from src.LLM_source_code.LLM_HumanFeedbackLoop.model import Azure_model_for_human_in_the_loop
from src.LLM_source_code.LLM_HumanFeedbackLoop.annotation_llm import assistant_call_with_annotation
from src.LLM_source_code.LLM_HumanFeedbackLoop_v2.reset import Update_vector
import os
import re
import shutil
from docx import Document
import time

def updated_text_based():
    w1, col1, w2, col2, w3 = st.columns([1, 5, 1, 5, 1]) 
    m1, m2, m3 = st.columns([0.2, 8, 0.2])
    # Update_vector()
    # Row 1: Select Use Case
    with col1:
        st.markdown("<div style='height:4rem; line-height:4rem; font-weight: bold;'>Select Use Case</div>", unsafe_allow_html=True)
    with col2:
        vAR_usecase = st.selectbox(" ", ("Policy Guru", "Multimodal RAG"))

    # Row 2: Select LLM
    with col1:
        st.markdown("<div style='height:5.3rem; line-height:5.3rem; font-weight: bold;'>Select LLM</div>", unsafe_allow_html=True)
    with col2:
        vAR_model = st.selectbox(" ", ("gpt-4o(Azure OpenAI)"  ))

    # Row 3: Select Platform
    with col1:
        st.markdown("<div style='height:4rem; line-height:4rem; font-weight: bold;'>Select Platform</div>", unsafe_allow_html=True)
    with col2:
        if vAR_model == "All":
            vAR_platform = st.selectbox(" ", ("All", "Assistant(Azure OpenAI)", "AWS Bedrock", 
                                             "Vertex AI(Gemini)", "Azure OpenAI(Langchain)"))
        elif vAR_model in ["gpt-4o(Azure OpenAI)", "gpt-4(Azure OpenAI)"]:
            vAR_platform = st.selectbox(" ", ("Assistant(Azure OpenAI)",))
        elif vAR_model == "claude-3.5-sonnet(Bedrock)":
            vAR_platform = st.selectbox(" ", ("AWS Bedrock",))
        elif vAR_model == "gemini-1.5(Vertex AI)":
            vAR_platform = st.selectbox(" ", ("Vertex AI(Gemini)",))
        elif vAR_model == "Azure OpenAI(Langchain)":
            vAR_platform = st.selectbox(" ", ("Langchain",))

    # # Row 4: Select LLM as a Judge Model
    # with col1:
    #     st.markdown("<div style='height:5.1rem; line-height:5.1rem; font-weight: bold;'>Select LLM as a Judge Model (Evaluator)</div>", 
    #                 unsafe_allow_html=True)
    # with col2:
    #     vAR_eval_llm = st.selectbox(" ", ("GPT(Default)", "Claude", "Gemini"))

    # # Row 5: Select LLM as a Judge Type
    # with col1:
    #     st.markdown("<div style='height:4rem; line-height:4rem; font-weight: bold;'>Select LLM as a Judge Type</div>", unsafe_allow_html=True)
    # with col2:
    #     vAR_eval_type = st.selectbox(" ", ("Pairwise Comparison(Default)", 
    #                                       "Evaluation by criteria with reference", 
    #                                       "Evaluation by criteria without reference"))
    with m2:
        # Initialize chat history and feedback state
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "user", "content": "We are delighted to have you here in the Live Agent Chat room!"},
                {"role": "assistant", "content": "Hello! How can I assist you today?"}
            ]
        if "feedback_given" not in st.session_state:
            st.session_state.feedback_given = set()
        if "file_copied" not in st.session_state:
            st.session_state.file_copied = False
        if "improved_response" not in st.session_state:
            st.session_state["improved_response"] = None
        if "prompt" not in st.session_state:
            st.session_state["prompt"] = None
        if "first_response" not in st.session_state:
            st.session_state["first_response"] = None


        # Add custom CSS for styling chat
        st.markdown(
            """
            <style>
            .chat-container { display: flex; align-items: center; margin: 10px 0; }
            .chat-container.user { justify-content: flex-end; }
            .chat-container.assistant { justify-content: flex-start; }
            .chat-container img { width: 40px; height: 40px; border-radius: 50%; margin: 0 10px; }
            .chat-bubble { max-width: 70%; padding: 10px; border-radius: 10px; margin: 5px 0; }
            .chat-bubble.user { background-color: #e0e0e0; text-align: right; }
            .chat-bubble.assistant { background-color: #ffffff; text-align: left; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        user_image_url = "https://storage.googleapis.com/macrovector-acl-eu/previews/118720/thumb_118720.webp"
        assistant_image_url = "https://cdn-icons-png.flaticon.com/512/6014/6014401.png"

        # Display chat history
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(
                    f'''
                    <div class="chat-container user">
                        <div class="chat-bubble user">{message["content"]}</div>
                        <img src="{user_image_url}" alt="User">
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'''
                    <div class="chat-container assistant">
                        <img src="{assistant_image_url}" alt="Assistant">
                        <div class="chat-bubble assistant">{message["content"]}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
                
                if "table_heading" in message:
                    st.write(f"### {message['table_heading']}")
                if "table" in message:
                    st.table(message["table"])
                
                if "feedback_table_heading" in message:
                    st.write(f"### {message['feedback_table_heading']}")
                if "feedback_table" in message:
                    st.table(message["feedback_table"])
                

                # Collect feedback for assistant responses
                if i > 1 and i not in st.session_state.feedback_given:
                    feedback_ = streamlit_feedback(
                        align="flex-start",
                        feedback_type="thumbs",
                        optional_text_label="[Optional] Please provide an explanation",
                        key=f"thumbs_{i}"
                    )

                    if feedback_:
                        # Parse feedback details
                        feedback_score = feedback_.get("score", "neutral")
                        feedback_text = feedback_.get("text", "")
                        st.session_state.feedback_given.add(i)
                        
                        if feedback_score == "👎":
                            # Handle thumbs-down feedback
                            prompt = st.session_state['prompt']

                            response = st.session_state['first_response']
                            improved_response = Azure_model_for_human_in_the_loop(prompt, response, feedback_score, feedback_text)
                            st.session_state["improved_response"] = improved_response
                            project_id = "elp-2022-352222"
                            dataset_id = "DMV_ELP"
                            table_name = "prompt_and_response"

                            expected_response = get_expected_response(project_id, dataset_id, table_name, prompt)
                                
                            data_2 = {
                                "Prompt": [prompt],
                                "Expected Response": [expected_response],
                                "GPT Response": [response],
                                "Label" : [feedback_score],
                                "Feedback" : [feedback_text] 
                            }
                            step_2 = pd.DataFrame(data_2)
                            step_3 = step_2.copy()
                            step_3["Optimized LLM Response"] = improved_response
                            
                            # Append improved response and tables with headings
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": "",
                                    "table_heading": "Human-in-the-Loop (HITL):",
                                    "table": step_2,
                                    "feedback_table_heading": "Optimized LLM response based on human feedback:",
                                    "feedback_table": step_3,
                                }
                            )
                            
                        if feedback_score == "👍":
                            user_message = st.session_state['prompt']
                            assistant_message = st.session_state["improved_response"] 
                            

                            if not st.session_state.file_copied:
                                try:
                                    source_file = "DMV_FAQ.docx"
                                    destination_file = "DMV_FAQ_copy.docx"
                                    shutil.copy(source_file, destination_file)
                                    st.session_state.file_copied = True
                                except Exception as e:
                                    st.error(f"Error during file operations: {e}")

                            file_path = "DMV_FAQ_copy.docx"
                            text = f"\n\n{user_message}\n\n{assistant_message}"

                            print("************",text)

                            # Update the .docx file
                            doc = Document(file_path)
                            doc.add_paragraph(text)
                            doc.save(file_path)

                            # Update vector store
                            vector_store_files = st.session_state.client.beta.vector_stores.files.list(
                                vector_store_id=os.getenv("VECTOR_STORE_ID")
                            )
                            if vector_store_files.data:
                                file_ids = vector_store_files.data[0].id
                                st.session_state.client.beta.vector_stores.files.delete(
                                    vector_store_id=os.getenv("VECTOR_STORE_ID"), file_id=file_ids
                                )

                            response = st.session_state.client.files.create(file=open(file_path, "rb"), purpose="assistants")
                            file_id = response.id
                            st.session_state.client.beta.vector_stores.files.create(
                                vector_store_id=os.getenv("VECTOR_STORE_ID"), file_id=file_id
                            )
                            st.session_state.thread = st.session_state.client.beta.threads.create()

                            st.write("User Feedback has been successfully saved to the OpenAI Vector Database")

        # Capture user input
        prompt = st.chat_input("What else can I do to help?")
        if prompt:
            st.session_state["prompt"] = prompt
            # Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate assistant response with a table
            response_text = assistant_call_with_annotation(prompt)
            st.session_state["first_response"] = response_text
            project_id = "elp-2022-352222"
            dataset_id = "DMV_ELP"
            table_name = "prompt_and_response"

            expected_response = get_expected_response(project_id, dataset_id, table_name, prompt)
                
            data = {
                "Prompt": [prompt],
                "Expected Response": [expected_response],
                "GPT Response": [response_text]
            }
            step_1 = pd.DataFrame(data)
            st.session_state.messages.append({"role": "assistant", "content": "", "table_heading": "Model Response:","table": step_1})

            st.rerun()
        if st.button("Reset KnowledgeBase"):
            Update_vector()
            st.session_state.thread = st.session_state.client.beta.threads.create()
            st.write("KnowledgeBase successfully reset")
            time.sleep(2)
            st.rerun() 

