import pandas as pd
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric


def Custom_Eval_Context_Precision(queries_df):
    # Initialize the metric
    metric = ContextualPrecisionMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
    )

    
    # Prepare test cases
    test_cases = []
    for _, query_row in queries_df.iterrows():

        # Filter retrieved documents with the same input text
        retrieval_context = list(query_row['reference'])  # Collect all relevant references

        # Create test case
        test_case = LLMTestCase(
            input=query_row['input'],            # Use 'input' from queries_df
            actual_output=query_row['output'],    # Output from queries_df
            expected_output=query_row['reference'],  # Expected output from queries_df's reference
            retrieval_context=retrieval_context    # All relevant references from retrieved_documents_df
        )
        test_cases.append(test_case)

        # Measure each test case
    scores = []
    reasons = []
    for test_case in test_cases:
        metric.measure(test_case)
        scores.append(metric.score)
        reasons.append(metric.reason)

    # Create a new DataFrame to store the results
    precision_eval_df = pd.DataFrame({
        "score": scores,
        "explanation": reasons,
        "label" : "Context Precision"
    })

    return precision_eval_df




def Custom_Eval_Context_Recall(queries_df):
    # Initialize the metric
    recall_metric = ContextualRecallMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
    )

    
    # Prepare test cases and lists to hold recall scores and reasons
    recall_scores = []
    recall_reasons = []

    # Loop through each row in queries_df to measure Contextual Recall
    for _, query_row in queries_df.iterrows():
        # Get the input to use as the key for matching rows in retrieved_documents_df
        input_text = query_row['input']

        # Filter retrieved documents with the same input text
        retrieval_context = list(query_row['reference'])  # Collect all relevant references

        # Create the test case for Contextual Recall
        test_case = LLMTestCase(
            input=query_row['input'],               # Use 'input' from queries_df
            actual_output=query_row['output'],       # Output from queries_df
            expected_output=query_row['reference'],  # Expected output from queries_df's reference
            retrieval_context=retrieval_context      # All relevant references from retrieved_documents_df
        )

        # Measure the recall and store results
        recall_metric.measure(test_case)
        recall_scores.append(recall_metric.score)
        recall_reasons.append(recall_metric.reason)

    # Create a new DataFrame to store the Contextual Recall results
    recall_eval_df = pd.DataFrame({
        "score": recall_scores,
        "explanation": recall_reasons,
        "label":"Context Recall"
    })

    return recall_eval_df