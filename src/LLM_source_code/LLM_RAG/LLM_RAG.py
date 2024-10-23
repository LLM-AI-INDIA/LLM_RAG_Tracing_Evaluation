import streamlit as st
from streamlit_chat import message
import os
import phoenix as px





from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents

# https://python.langchain.com/docs/tutorials/pdf_qa/



def RAG_Core_Impl(vAR_user_input, vAR_model):
    # Move imports inside the function to avoid circular dependency
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    file_path = "DMVFAQ.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    llm = ChatOpenAI(model=vAR_model)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-ada-002"))

    retriever = vectorstore.as_retriever()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        metadata={"application_type": "question_answering"},
    )

    PHOENIX_API_KEY = os.environ["PHOENIX_API_KEY"]
    os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

    # Configuration is picked up from your environment variables
    tracer_provider = register()
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    result = chain.invoke(vAR_user_input)
    print("Result - ", result)

    px_client = px.Client()

    phoenix_df = px_client.get_spans_dataframe()

    return result["result"], px_client, phoenix_df


def Conversation(vAR_model):
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Greetings! I am LLMAI Live Agent. How can I help you?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["We are delighted to have you here in the LLMAI Live Agent Chat room!"]

    # Container for the chat history
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            vAR_user_input = st.text_input("Prompt:", placeholder="How can I help you?", key='input')
            submit_button = st.form_submit_button(label='Interact with LLM')

        if submit_button and vAR_user_input and vAR_user_input!='':
            # Generate response from the agent
            vAR_response, px_client, phoenix_df = RAG_Core_Impl(vAR_user_input, vAR_model)
            st.session_state['past'].append(vAR_user_input)
            print(" vAR_response before append - ",vAR_response)
            st.session_state['generated'].append(vAR_response)
            

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
    return vAR_user_input


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
        st.subheader("Select Use Case")
        st.write("")
    with col4:
        vAR_usecase = st.selectbox("Select Usecase",("Policy Guru","Multimodal RAG"))
        st.write("")

    if vAR_usecase=="Policy Guru":
        with col22:
            st.subheader("Upload Knowledge Base")
        with col24:
            vAR_knowledge_base = st.file_uploader("Choose any type of file",accept_multiple_files=True)

    if vAR_knowledge_base:
        with col17:
            st.subheader("Select LLM")
            st.write("")
        with col19:
            vAR_model = st.selectbox("Select LLM",("gpt-4o","gpt-4o-mini","gpt-3.5","claude-3.5-sonnet","llama3"))
            st.write("")

        with col7:
            st.subheader("Select Vector DB")
            st.write("")
        with col9:
            vAR_vector_db = st.selectbox("Select Vector DB",("Chroma","Pinecone","Weaviate","Qdrant","Milvus","Opensearch"))
            st.write("")

        vAR_user_input = Conversation(vAR_model)

        col31,col32,col33 = st.columns([2,3,2])

        with col32:
            vAR_function = st.radio("Select Functionality",["Observability","Evaluation"],horizontal=True)
        if vAR_function=="Observability":
            LLM_RAG_Tracing(vAR_function,vAR_user_input,vAR_model)
            pass
        else:
            pass
        col34,col35,col36,col37,col38 = st.columns([1,5,1,5,1])
        col39,col40,col41 = st.columns([1,5,1])
        if vAR_function=="Evaluation":
            with col35:
                st.subheader("Select Metrics")
            with col37:
                selected_metrics = st.multiselect("Select metrics",["All","Hallucination Score","QA Correctness Score","Retrieval Relevance Score","Bias","Noise Sensitivity","Context Recall"])
                if "All" in selected_metrics:
                    selected_metrics = ["Hallucination Score","QA Correctness Score","Retrieval Relevance Score","Bias","Noise Sensitivity","Context Recall"]
            with col40:
                st.write("")
                st.write("")
                with st.expander("Evaluation Metrics Definition"):
                    st.write("**Hallucination :** This Eval is specifically designed to detect hallucinations in generated answers from private or retrieved data. The Eval detects if an AI answer to a question is a hallucination based on the reference data used to generate the answer.")
                    st.write("**QA Correctness :** This Eval evaluates whether a question was correctly answered by the system based on the retrieved data. In contrast to retrieval Evals that are checks on chunks of data returned, this check is a system level check of a correct Q&A.")
                    st.write("**Retrieval Relevance :** This Eval evaluates whether a retrieved chunk contains an answer to the query. It's extremely useful for evaluating retrieval systems.")
                    st.write("**Context Recall :** Context Recall measures how many of the relevant documents (or pieces of information) were successfully retrieved. It focuses on not missing important results. Higher recall means fewer relevant documents were left out.")
                    st.write("**Noise Sensitivity :** NoiseSensitivity measures how often a system makes errors by providing incorrect responses when utilizing either relevant or irrelevant retrieved documents. The score ranges from 0 to 1, with lower values indicating better performance.")
                
            if selected_metrics:
                LLM_RAG_Tracing(vAR_function)
                st.write("")







def LLM_RAG_Tracing(vAR_function,vAR_user_input,vAR_model):
    
    result,px_client,phoenix_df = RAG_Core_Impl(vAR_user_input,vAR_model)
    if vAR_function=="Observability":
        st.write("Tracing/Span Dataframe")
        st.dataframe(phoenix_df)

    if vAR_function!="Observability":
        queries_df = get_qa_with_reference(px_client)
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