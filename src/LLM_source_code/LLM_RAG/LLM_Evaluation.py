import pandas as pd
from multiprocessing import Pool
from functools import partial

from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)

import os
from src.LLM_source_code.LLM_RAG.LLM_Custom_Eval import Custom_Eval_Context_Precision,Custom_Eval_Context_Recall


eval_model = OpenAIModel(
                model="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-05-01-preview",
            )
                
hallucination_evaluator = HallucinationEvaluator(eval_model)
qa_correctness_evaluator = QAEvaluator(eval_model)
relevance_evaluator = RelevanceEvaluator(eval_model)
def process_single_response(item, vAR_user_input, assistant_request_id, assistant_response_id):
    """
    Process a single response tuple and return the evaluation DataFrames
    """
    try:
        # Create input DataFrames
        queries_df = pd.DataFrame({
            "input": [vAR_user_input],
            "output": [item[1]],
            "reference": [item[1]]
        })
        
        retrieved_documents_df = pd.DataFrame({
            "input": [vAR_user_input],
            "reference": [item[1]]
        })
        
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

        # Custom Evaluations
        context_precision_eval_df = Custom_Eval_Context_Precision(queries_df)
        context_recall_eval_df = Custom_Eval_Context_Recall(queries_df)
        
        # Add categories
        hallucination_eval_df["Metrics Category"] = "Generation"
        qa_correctness_eval_df["Metrics Category"] = "Generation"
        relevance_eval_df["Metrics Category"] = "Retrieval"
        context_precision_eval_df["Metrics Category"] = "Retrieval"
        context_recall_eval_df["Metrics Category"] = "Retrieval"

        # Create first merged DataFrame
        merged_df = pd.concat([
            hallucination_eval_df, 
            qa_correctness_eval_df, 
            relevance_eval_df,
            context_precision_eval_df,
            context_recall_eval_df
        ], ignore_index=True)
        
        merged_df.rename(columns={
            'label': 'Metrics',
            'score': 'Score',
            'explanation': 'Explanation'
        }, inplace=True)
        
        merged_df["Model"] = item[0]
        merged_df = merged_df.reindex(['Model', 'Metrics Category', 'Metrics', 'Score', 'Explanation'], axis=1)

        # Create second merged DataFrame
        merged_df2 = pd.concat([
            hallucination_eval_df, 
            qa_correctness_eval_df, 
            relevance_eval_df,
            context_precision_eval_df,
            context_recall_eval_df
        ], ignore_index=True)
        
        merged_df2["MODEL_NAME"] = item[0]
        merged_df2["MODEL_RESPONSE"] = item[1]
        merged_df2["REQUEST_ID"] = assistant_request_id
        merged_df2["RESPONSE_ID"] = assistant_response_id
        merged_df2["THREAD_ID"] = item[2]
        merged_df2["PROMPT"] = vAR_user_input

        return merged_df, merged_df2
    
    except Exception as e:
        print(f"Error processing response for {item[0]}: {str(e)}")
        return None, None
