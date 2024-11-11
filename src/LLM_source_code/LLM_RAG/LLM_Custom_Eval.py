import pandas as pd
from deepeval.metrics import ContextualPrecisionMetric,AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric
# from langchain_openai import AzureOpenAI
import os
import streamlit as st
import traceback
# from src.LLM_source_code.LLM_RAG.LLM_DeepEvalBaseLLM import AzureOpenAIClass

# custom_model = AzureOpenAI(
#     openai_api_version="2023-03-15-preview",
#     azure_deployment="GPT-35-DEV-Deployment",
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
#     max_tokens=7000,
# )

# azure_openai = AzureOpenAIClass(model=custom_model)


def Custom_Eval_Context_Precision(queries_df):

    try:
        # Initialize the metric
        metric = ContextualPrecisionMetric(
            threshold=0.7,
            include_reason=True,
            model="gpt-4o"
        )

        # Prepare test cases
        test_cases = []
        for _, query_row in queries_df.iterrows():

            # Filter retrieved documents with the same input text
            retrieval_context = [query_row['reference']]  # Collect all relevant references

            # Create test case
            test_case = LLMTestCase(
                input=query_row['input'],            # Use 'input' from queries_df
                actual_output=query_row['output'],    # Output from queries_df
                expected_output=retrieval_context,  # Expected output from queries_df's reference - later change it.
                retrieval_context=retrieval_context    # All relevant references from retrieved_documents_df
            )
            test_cases.append(test_case)

            # Measure each test case
        scores = []
        reasons = []
        for test_case in test_cases:
            result = metric.measure(test_case)
            print("cprecision result - ",metric)
            scores.append(metric.score)
            reasons.append(metric.reason)

        # Create a new DataFrame to store the results
        precision_eval_df = pd.DataFrame({
            "score": scores,
            "explanation": reasons,
            "label" : "Context Precision"
        })

        return precision_eval_df

    except BaseException as e:
        vAR_err = str(e)
        print(traceback.format_exc())
        print("Error in Custom Evaluation - "+vAR_err)
        return pd.DataFrame({
            "score": [""],
            "explanation": ["N/A"],
            "label" : "Context Precision"
        })




def Custom_Eval_Context_Recall(queries_df):

    try:
        # Initialize the metric
        recall_metric = ContextualRecallMetric(
            threshold=0.7,
            include_reason=True,
            model="gpt-4o"
        )

        
        # Prepare test cases and lists to hold recall scores and reasons
        recall_scores = []
        recall_reasons = []

        # Loop through each row in queries_df to measure Contextual Recall
        for _, query_row in queries_df.iterrows():

            # Filter retrieved documents with the same input text
            retrieval_context = [query_row['reference']]  # Collect all relevant references

            # Create the test case for Contextual Recall
            test_case = LLMTestCase(
                input=query_row['input'],               # Use 'input' from queries_df
                actual_output=query_row['output'],       # Output from queries_df
                expected_output=retrieval_context,  # Expected output from queries_df's reference - later change it.
                retrieval_context=retrieval_context      # All relevant references from retrieved_documents_df
            )

            # Measure the recall and store results
            result = recall_metric.measure(test_case)
            print("crecall result - ",recall_metric)
            recall_scores.append(recall_metric.score)
            recall_reasons.append(recall_metric.reason)

        # Create a new DataFrame to store the Contextual Recall results
        recall_eval_df = pd.DataFrame({
            "score": recall_scores,
            "explanation": recall_reasons,
            "label":"Context Recall"
        })

        return recall_eval_df
    
    except BaseException as e:
        vAR_err = str(e)
        print(traceback.format_exc())
        print("Error in Custom Evaluation - "+vAR_err)
        return pd.DataFrame({
            "score": [""],
            "explanation": ["N/A"],
            "label" : "Context Precision"
        })