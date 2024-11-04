import streamlit as st
from streamlit_chat import message
import os
from src.LLM_source_code.LLM_RAG.LLM_Assistant_Call import  assistant_call
from src.LLM_source_code.LLM_RAG.LLM_Bedrock_Call import  retrieve_generated
from src.LLM_source_code.LLM_RAG.LLM_VertexAI_Call import generate
import datetime
from google.cloud import bigquery
import concurrent.futures
import pandas as pd
import webbrowser




from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from src.LLM_source_code.LLM_RAG.LLM_Custom_Eval import Custom_Eval_Context_Precision,Custom_Eval_Context_Recall





def Conversation(vAR_knowledge_base):
    
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
    vAR_response = None
    px_client = None
    phoenix_df = None

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            vAR_user_input = st.text_input("Prompt:", placeholder="How can I help you?", key='input')
            submit_button = st.form_submit_button(label='Interact with LLM')

        if submit_button and vAR_user_input and vAR_user_input!='':
            # Generate response from the agent
            with concurrent.futures.ThreadPoolExecutor() as executor:

                vAR_assistant = executor.submit(assistant_call,vAR_user_input)
                vAR_bedrock = executor.submit(retrieve_generated,vAR_user_input)
                vAR_vertex = executor.submit(generate,vAR_user_input)

                px_client,vAR_assistant_phoenix_df,vAR_assistant_response,assistant_thread_id,assistant_request_id,assistant_response_id = vAR_assistant.result()
                vAR_response_bedrock,bedrock_thread_id,bedrock_request_id,bedrock_response_id = vAR_bedrock.result()
                vAR_response_vertex,vertex_thread_id,vertex_request_id,vertex_response_id = vAR_vertex.result()

                print("vAR_assistant_result - ",vAR_assistant_response)
                print("vAR_bedrock_result - ",vAR_response_bedrock)
                print("vAR_vertex_result - ",vAR_response_vertex)
            

            st.session_state.vAR_assistant_response_list.append(vAR_assistant_response)
            st.session_state.vAR_bedrock_response_list.append(vAR_response_bedrock)
            st.session_state.vAR_vertex_response_list.append(vAR_response_vertex)
            
            vAR_final_df = pd.DataFrame({"OpenAI":st.session_state.vAR_assistant_response_list,"Bedrock":st.session_state.vAR_bedrock_response_list,"Google":st.session_state.vAR_vertex_response_list})

            st.write("")
            col1,col2,col3,col4 = st.columns([5.5,2,5,1])
            with col1:
                st.markdown("<h3 style='font-size:18px;'>Response Summary</h3>", unsafe_allow_html=True)
            with col2:
                st.link_button("Report View","https://lookerstudio.google.com/reporting/f7586dea-e417-44c9-bc6b-f5ba3dee09ee")

            # vAR_final_df = vAR_final_df.reset_index().rename(columns={'index': '#'})
            
            # vAR_final_df.style.hide(axis="index")

            st.table(vAR_final_df)
            # st.subheader("Tracing Dataframe")
            # st.dataframe(vAR_assistant_phoenix_df)


            # queries_df = get_qa_with_reference(px_client)
            # retrieved_documents_df = get_retrieved_documents(px_client)


            eval_model = OpenAIModel()
                
            hallucination_evaluator = HallucinationEvaluator(eval_model)
            qa_correctness_evaluator = QAEvaluator(eval_model)
            relevance_evaluator = RelevanceEvaluator(eval_model)

            
            vAR_responses = [("GPT",vAR_assistant_response,assistant_thread_id),("Claude",vAR_response_bedrock,bedrock_thread_id),("Gemini",vAR_response_vertex,vertex_thread_id)]
            vAR_eval_df_list = []
            vAR_eval_df_list2 = []
            for item in vAR_responses:
                queries_df = pd.DataFrame({"input":[vAR_user_input],"output":[item[1]],"reference":[item[1]]})
                retrieved_documents_df = pd.DataFrame({"input":[vAR_user_input],"reference":[item[1]]})
                
                # Generation Evaluation
                hallucination_eval_df, qa_correctness_eval_df = run_evals(
                    dataframe=queries_df,
                    evaluators=[hallucination_evaluator, qa_correctness_evaluator],
                    provide_explanation=True,
                )

                # Retrieval Evaluation
                relevance_eval_df = run_evals(
                    dataframe=retrieved_documents_df,
                    evaluators=[relevance_evaluator],
                    provide_explanation=True,
                )[0]

                try:

                    context_precision_eval_df = Custom_Eval_Context_Precision(queries_df)
                    context_recall_eval_df = Custom_Eval_Context_Recall(queries_df)
                
                except BaseException as e:
                    vAR_err = str(e)
                    st.error("Error in Custom Evaluation - "+vAR_err)
                    print("Error in Custom Evaluation - "+vAR_err)

                hallucination_eval_df["Metrics Category"] = "Generation"
                qa_correctness_eval_df["Metrics Category"] = "Generation"
                relevance_eval_df["Metrics Category"] = "Retrieval"
                context_precision_eval_df["Metrics Category"] = "Retrieval"
                context_recall_eval_df["Metrics Category"] = "Retrieval"

                # Concatenate the DataFrames
                merged_df = pd.concat([hallucination_eval_df, qa_correctness_eval_df, relevance_eval_df,context_precision_eval_df,context_recall_eval_df], ignore_index=True)
                merged_df.rename(columns={'label':'Metrics','score':'Score','explanation':'Explanation'},inplace=True)
                
                merged_df["Model"] = item[0]

                merged_df = merged_df.reindex(['Model','Metrics Category','Metrics','Score','Explanation'], axis=1)
                vAR_eval_df_list.append(merged_df)

                merged_df2 = pd.concat([hallucination_eval_df, qa_correctness_eval_df, relevance_eval_df,context_precision_eval_df,context_recall_eval_df], ignore_index=True)
                merged_df2["MODEL_NAME"] = item[0]
                merged_df2["MODEL_RESPONSE"] = item[1]
                merged_df2["REQUEST_ID"] = assistant_request_id
                merged_df2["RESPONSE_ID"] = assistant_response_id
                merged_df2["THREAD_ID"] = item[2]
                merged_df2["PROMPT"] = vAR_user_input
                vAR_eval_df_list2.append(merged_df2)

            vAR_final_eval_df = pd.concat(vAR_eval_df_list,ignore_index=True)
            # vAR_final_eval_df = vAR_final_eval_df.reset_index().rename(columns={'index': '#'})
            # vAR_final_eval_df.style.hide(axis="index")

            vAR_final_eval_df2 = pd.concat(vAR_eval_df_list2,ignore_index=True)

            st.write("")
            col1,col2,col3,col4 = st.columns([5.5,2,5,1])
            with col1:
                st.markdown("<h3 style='font-size:18px;'>Generation & Retrieval Evaluation Metrics</h3>", unsafe_allow_html=True)
            with col2:
                st.link_button("Report View","https://lookerstudio.google.com/reporting/f7586dea-e417-44c9-bc6b-f5ba3dee09ee/page/p_55e0w4zfmd")

            st.table(vAR_final_eval_df)


            
            

            # Bigquery Insert
            Bigquery_Insert(assistant_thread_id,vAR_user_input,vAR_assistant_response,assistant_request_id,assistant_response_id,"OpenAI-Assistant-GPT-4o")
            Bigquery_Insert(bedrock_thread_id,vAR_user_input,vAR_response_bedrock,bedrock_request_id,bedrock_response_id,"Anthropic-Claude-3.5-Sonnet")
            Bigquery_Insert(vertex_thread_id,vAR_user_input,vAR_response_vertex,vertex_request_id,vertex_response_id,"Gemini-1.5-Flash")
            
            # # Eval Insertion
            Bigquery_Eval_Insert(vAR_final_eval_df2)

            st.session_state['past'].append(vAR_user_input)
            st.session_state['generated'].append(vAR_response_bedrock)
        
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

        print("custom_css - ",custom_css)
        st.markdown(custom_css,unsafe_allow_html=True)


        if st.session_state['generated']:
            with response_container:
                st.write("")
                st.write("")
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    print("before message print response - ",st.session_state["generated"][i])
                    message(st.session_state["generated"][i], key=str(i+55), avatar_style="thumbs")

                    # Skip feedback for predefined messages
                    if st.session_state["past"][i] != st.session_state["past"][0] and st.session_state["generated"][i] != st.session_state["generated"][0]:
                        feedback = st.feedback("thumbs", key=f"feedback_{i}")
                        if feedback is not None:
                            feedback_text = "thumbs up" if feedback == 1 else "thumbs down"
                            feedback_class = "thumbs-up" if feedback == 1 else "thumbs-down"
                            st.markdown(f'<div class="{feedback_class}">Thank you for your feedback! You rated this response with a {feedback_text}.</div>', unsafe_allow_html=True)
    return vAR_response,px_client, phoenix_df


def LLM_RAG_Impl():
    
    col1,col2,col3,col4,col5 = st.columns([1,5,1,5,1])
    col21,col22,col23,col24,col25 = st.columns([1,5,1,5,1])
    col16,col17,col18,col19,col20 = st.columns([1,5,1,5,1])


    col6,col7,col8,col9,col10 = st.columns([1,5,1,5,1])
    col11,col12,col13,col14,col15 = st.columns([1,5,1,5,1])
    col26,col27,col28,col29,col30 = st.columns([1,5,1,5,1])


    vAR_model = None
    vAR_questions = None
    vAR_knowledge_base = None

    with col2:
        st.write("")
        st.markdown("<h3 style='font-size:18px;'>Select Use Case</h3>", unsafe_allow_html=True)
    with col4:
        vAR_usecase = st.selectbox(" ",("Policy Guru","Multimodal RAG"))
        st.write("")

    if vAR_usecase=="Policy Guru":
        with col22:
            st.write("")
            st.markdown("<h3 style='font-size:18px;'>Upload Knowledge Base</h3>", unsafe_allow_html=True)
        with col24:
            vAR_knowledge_base = st.file_uploader(" ",accept_multiple_files=True)
            
            
            

    if vAR_knowledge_base:
        with open("DMV-FAQ.pdf", mode='wb') as w:
            w.write(vAR_knowledge_base[0].getvalue())
        with col17:
            st.write("")
            st.markdown("<h3 style='font-size:18px;'>Select LLM</h3>", unsafe_allow_html=True)
        with col19:
            vAR_model = st.selectbox(" ",("All","gpt-4o","gpt-4o-mini","gpt-3.5","claude-3.5-sonnet","llama3","gemini-1.5"))
            st.write("")

        with col7:
            st.write("")
            st.markdown("<h3 style='font-size:18px;'>Select Platform</h3>", unsafe_allow_html=True)
        with col9:
            vAR_platform = st.selectbox(" ",("All","Assistant(OpenAI)","Assistant(Azure OpenAI)","AWS Bedrock","Vertex AI(Gemini)"))
            st.write("")

        vAR_response,px_client, phoenix_df = Conversation(vAR_knowledge_base)

        # col31,col32,col33 = st.columns([2,3,2])

        # with col32:
        #     vAR_function = st.radio("Select Functionality",["Observability","Evaluation"],horizontal=True)
        # if vAR_function=="Observability":
        #     LLM_RAG_Tracing(vAR_function,px_client, phoenix_df)
        #     pass
        # else:
        #     pass
        # col34,col35,col36,col37,col38 = st.columns([1,5,1,5,1])
        # col39,col40,col41 = st.columns([1,5,1])
        # if vAR_function=="Evaluation":
        #     with col35:
        #         st.subheader("Select Metrics")
        #     with col37:
        #         selected_metrics = st.multiselect("Select metrics",["All","Hallucination Score","QA Correctness Score","Retrieval Relevance Score","Bias","Noise Sensitivity","Context Recall"])
        #         if "All" in selected_metrics:
        #             selected_metrics = ["Hallucination Score","QA Correctness Score","Retrieval Relevance Score","Bias","Noise Sensitivity","Context Recall"]
        #     with col40:
        #         st.write("")
        #         st.write("")
        #         with st.expander("Evaluation Metrics Definition"):
        #             st.write("**Hallucination :** This Eval is specifically designed to detect hallucinations in generated answers from private or retrieved data. The Eval detects if an AI answer to a question is a hallucination based on the reference data used to generate the answer.")
        #             st.write("**QA Correctness :** This Eval evaluates whether a question was correctly answered by the system based on the retrieved data. In contrast to retrieval Evals that are checks on chunks of data returned, this check is a system level check of a correct Q&A.")
        #             st.write("**Retrieval Relevance :** This Eval evaluates whether a retrieved chunk contains an answer to the query. It's extremely useful for evaluating retrieval systems.")
        #             st.write("**Context Recall :** Context Recall measures how many of the relevant documents (or pieces of information) were successfully retrieved. It focuses on not missing important results. Higher recall means fewer relevant documents were left out.")
        #             st.write("**Noise Sensitivity :** NoiseSensitivity measures how often a system makes errors by providing incorrect responses when utilizing either relevant or irrelevant retrieved documents. The score ranges from 0 to 1, with lower values indicating better performance.")
                
        #     if selected_metrics:
        #         LLM_RAG_Tracing(vAR_function,px_client, phoenix_df)
        #         st.write("")






def LLM_RAG_Tracing(vAR_function,px_client, phoenix_df):
    
    if vAR_function=="Observability":
        st.write("Tracing/Span Dataframe")
        st.dataframe(phoenix_df)

    if vAR_function!="Observability":
        # spans = query_spans(px_client)
        # queries_df = get_qa_with_reference(px_client)
        retrieved_documents_df = get_retrieved_documents(px_client)


        # Evaluation Metrics
        # https://docs.arize.com/phoenix/evaluation/concepts-evals/evaluation
        # https://docs.arize.com/phoenix/api/evaluation-models#openaimodel

        eval_model = OpenAIModel()
        
        hallucination_evaluator = HallucinationEvaluator(eval_model)
        qa_correctness_evaluator = QAEvaluator(eval_model)
        relevance_evaluator = RelevanceEvaluator(eval_model)

        # Generation Evaluation
        hallucination_eval_df, qa_correctness_eval_df = run_evals(
            dataframe=queries_df,
            evaluators=[hallucination_evaluator, qa_correctness_evaluator],
            provide_explanation=True,
        )
        # Retrieval Evaluation
        relevance_eval_df = run_evals(
            dataframe=retrieved_documents_df,
            evaluators=[relevance_evaluator],
            provide_explanation=True,
        )[0]

        st.write("Queries")
        st.dataframe(queries_df)

        st.write("Retrieved Docs")
        st.dataframe(retrieved_documents_df)

        st.write("Hallucination Result")
        st.dataframe(hallucination_eval_df)

        st.write("Q&A Correctness")
        st.dataframe(qa_correctness_eval_df)

        st.write("Retrieval Relevance Score")
        st.dataframe(relevance_eval_df)



def Bigquery_Insert(thread_id,vAR_user_input,vAR_response,request_id,response_id,model_name):
    
    vAR_response = vAR_response.replace("\n","").replace("\"","'")
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