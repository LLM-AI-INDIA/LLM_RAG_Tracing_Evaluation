import streamlit as st
from streamlit_feedback import streamlit_feedback
import pandas as pd
import os
import re
import shutil
from openai import AzureOpenAI
from docx import Document
from src.LLM_source_code.LLM_HumanFeedbackLoop.model import Azure_model_for_human_in_the_loop
from src.LLM_source_code.LLM_HumanFeedbackLoop.annotation_llm import assistant_call_with_annotation

# Function to initialize session state variables
def initialize_session_state():
    if 'feedback_counter' not in st.session_state:
        st.session_state.feedback_counter = 0
    if "step_2" not in st.session_state:
        st.session_state["step_2"] = None
    if "step_3" not in st.session_state:
        st.session_state["step_3"] = None   
    if "to_improve" not in st.session_state:
        st.session_state["to_improve"] = None 
    if "file_copied" not in st.session_state:
        st.session_state.file_copied = False
    if "file_deleted" not in st.session_state:
        st.session_state.file_deleted = False
    
def empty_session_state():
    st.session_state["step_2"] = None
    st.session_state["step_3"] = None

def Update_vector():
    # Ensure Azure client is initialized only once
    if "client" not in st.session_state:
        try:
            st.session_state.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-05-01-preview",
            )
        except Exception as e:
            st.error(f"Error initializing Azure client: {e}")
            return

    # File management: Update vector store if file is not deleted yet
    if not st.session_state.file_deleted:
        try:
            file_path = "DMV_FAQ.docx"
            vector_store_files = st.session_state.client.beta.vector_stores.files.list(
                vector_store_id=os.getenv("VECTOR_STORE_ID")
            )

            if vector_store_files.data:
                file_ids = vector_store_files.data[0].id
                st.session_state.client.beta.vector_stores.files.delete(
                    vector_store_id=os.getenv("VECTOR_STORE_ID"), file_id=file_ids
                )

            # Upload updated document to vector store
            response = st.session_state.client.files.create(file=open(file_path, "rb"), purpose="assistants")
            file_id = response.id
            st.session_state.client.beta.vector_stores.files.create(
                vector_store_id=os.getenv("VECTOR_STORE_ID"), file_id=file_id
            )

            st.session_state.file_deleted = True
            print("Base File Updated")
        except Exception as e:
            st.error(f"Error updating vector store: {e}")
            return
        
def response_with_feedback():
    initialize_session_state()  # Ensure session state variables are initialized
    Update_vector()

    # Generate unique keys using the counter
    feedback_1_key = f'feedback_1_{st.session_state.feedback_counter}'
    feedback_2_key = f'feedback_2_{st.session_state.feedback_counter}'
    

    w1, col1, w2, col2, w3 = st.columns([1, 5, 1, 5, 1])  # Keep original column split
    m1, m2, m3 = st.columns([0.3, 6, 0.3])

    # Row 1: Select Use Case
    with col1:
        st.markdown("<div style='height:4rem; line-height:4rem; font-weight: bold;'>Select Use Case</div>", unsafe_allow_html=True)
    with col2:
        vAR_usecase = st.selectbox(" ", ("Policy Guru", "Multimodal RAG"))

    # Row 2: Select LLM
    with col1:
        st.markdown("<div style='height:5.3rem; line-height:5.3rem; font-weight: bold;'>Select LLM</div>", unsafe_allow_html=True)
    with col2:
        vAR_model = st.selectbox(" ", ("All", "gpt-4o(Azure OpenAI)", "gpt-4(Azure OpenAI)", 
                                      "claude-3.5-sonnet(Bedrock)", "gemini-1.5(Vertex AI)", "Azure OpenAI(Langchain)"))

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

    # Row 4: Select LLM as a Judge Model
    with col1:
        st.markdown("<div style='height:5.1rem; line-height:5.1rem; font-weight: bold;'>Select LLM as a Judge Model (Evaluator)</div>", 
                    unsafe_allow_html=True)
    with col2:
        vAR_eval_llm = st.selectbox(" ", ("GPT(Default)", "Claude", "Gemini"))

    # Row 5: Select LLM as a Judge Type
    with col1:
        st.markdown("<div style='height:4rem; line-height:4rem; font-weight: bold;'>Select LLM as a Judge Type</div>", unsafe_allow_html=True)
    with col2:
        vAR_eval_type = st.selectbox(" ", ("Pairwise Comparison(Default)", 
                                          "Evaluation by criteria with reference", 
                                          "Evaluation by criteria without reference"))
    with col1:
        st.markdown("<div style='height:4rem; line-height:4rem; font-weight: bold;'>Enter the Prompt</div>", unsafe_allow_html=True)
    with col2:
        vAR_prompt = st.text_input("", placeholder="Provide the prompt and hit enter")
    st.write("")
    if vAR_prompt:
        response = assistant_call_with_annotation(vAR_prompt)

        excepted_resp = pd.read_excel("expected_response.xlsx")

        expected_response = excepted_resp.loc[excepted_resp['Prompt'] == vAR_prompt, 'Expected Response']

        if not expected_response.empty:
            # Extract the value
            expected_response = expected_response.iloc[0]
        data = {
            "Prompt": [vAR_prompt],
            "Expected Response": [expected_response],
            "GPT Response": [response]
        }
        step_1 = pd.DataFrame(data)
        with m2:
            st.write("### Model Response:")
            st.table(step_1)
            feedback_ = streamlit_feedback(
                align="flex-start",
                feedback_type="thumbs",
                optional_text_label="[ Human Feedback Optional ] Please provide an explanation",
                key=feedback_1_key
            )
    
            # Process first feedback
            if feedback_:
                feedback_score = feedback_.get("score", "neutral")
                feedback_text = feedback_.get("text", "")
                
                if feedback_score == "👎":
                    # Generate improved response
                    st.session_state["to_improve"] = Azure_model_for_human_in_the_loop(vAR_prompt, response, feedback_score, feedback_text)
                    step_2 = step_1.copy()

                    # Add new columns to the copied DataFrame
                    step_2["Label"] = [feedback_score]  
                    step_2["Feedback"] = [feedback_text] 

                    # Save to session state to persist the second table
                    st.session_state["step_2"] = step_2

                    step_3 = step_2.copy()
                    step_3["Improved Response"] = [st.session_state["to_improve"]]
                    st.session_state["step_3"] = step_3
                                
        # Display second table if it exists in session state
        if st.session_state["step_3"] is not None:
            with m2:
                st.write("### Feedback captured:")
                st.table(st.session_state["step_2"])
                st.write("### Re-Generated Response based on user feedback:")
                st.table(st.session_state["step_3"])
                feedback_2 = streamlit_feedback(
                    align="flex-start",
                    feedback_type="thumbs",
                    optional_text_label="[ Human Feedback Optional ] Please provide an explanation",
                    key=feedback_2_key
                )
                
                # Process second feedback
                if feedback_2:
                    feedback_score_2 = feedback_2.get("score", "neutral")
                    feedback_text_2 = feedback_2.get("text", "")

                    if feedback_score_2 == "👍":
                        user_message = vAR_prompt
                        assistant_message = st.session_state["to_improve"]
                        user_message = re.sub(r"[\[\]',]", "", str(user_message)).strip()
                        if not st.session_state.file_copied:
                            try:
                                source_file = "DMV_FAQ.docx"
                                destination_file = "DMV_FAQ_copy.docx"
                                shutil.copy(source_file, destination_file)
                                st.session_state.file_copied = True
                            except Exception as e:
                                st.error(f"Error during file operations: {e}")
                        print(f"Old Thread ID: {st.session_state.thread.id}")
                        st.session_state.thread = st.session_state.client.beta.threads.create()
                        print(f"New Thread ID: {st.session_state.thread.id}")
                        file_path = "DMV_FAQ_copy.docx"
                        text = f"\n\n{user_message}\n\n{assistant_message}"

                        # Load and update the .docx file
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
                        print("Session File Updated")
                        st.write("File successfully updated and added to vector store.")

                        empty_session_state()
                        st.session_state.feedback_counter = st.session_state.feedback_counter +1

                    # st.write(f"Final Feedback: {feedback_text_2}")

                    # # Use `to_improve` from session state
                    # if st.session_state["to_improve"] is not None:
                    #     text_to_add = vAR_prompt + " " + st.session_state["to_improve"]
                        
                        
                    #     add_to_vectorstore([text_to_add])
                    #     st.write("Added to vectorstore successfully!")
                    #     response, reference = langchain_rag(vAR_prompt)
                    #     data_2 = {
                    #         "Prompt": [vAR_prompt],
                    #         "Expected Response": [reference],
                    #         "GPT Response": [response]
                    #     }
                    #     step_4 = pd.DataFrame(data_2)
                    #     st.table(step_4)
                    # else:
                    #     st.warning("No improved response available to add to vectorstore.")

