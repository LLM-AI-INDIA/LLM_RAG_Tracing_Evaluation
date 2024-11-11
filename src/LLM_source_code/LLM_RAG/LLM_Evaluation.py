
from phoenix.evals import (
    
    run_evals
)
from src.LLM_source_code.LLM_RAG.LLM_Custom_Eval import Custom_Eval_Context_Precision,Custom_Eval_Context_Recall




from concurrent.futures import ProcessPoolExecutor, as_completed


import multiprocessing as mp
import pandas as pd
import traceback

from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
import os

def process_response(item,vAR_user_input, assistant_request_id, assistant_response_id,vAR_eval_df_list):
    
    eval_model = OpenAIModel(
                model="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-05-01-preview",
            )
                
    hallucination_evaluator = HallucinationEvaluator(eval_model)
    qa_correctness_evaluator = QAEvaluator(eval_model)
    relevance_evaluator = RelevanceEvaluator(eval_model)

    queries_df = pd.DataFrame({"input": [vAR_user_input], "output": [item[1]], "reference": [item[3]]})
    retrieved_documents_df = pd.DataFrame({"input": [vAR_user_input], "reference": [item[3]]})

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
        print(f"Error in Custom Evaluation - {vAR_err}")
        print(traceback.format_exc())

    hallucination_eval_df["Metrics Category"] = "Generation"
    qa_correctness_eval_df["Metrics Category"] = "Generation"
    relevance_eval_df["Metrics Category"] = "Retrieval"
    context_precision_eval_df["Metrics Category"] = "Retrieval"
    context_recall_eval_df["Metrics Category"] = "Retrieval"

    # Concatenate the DataFrames
    merged_df = pd.concat(
        [hallucination_eval_df, qa_correctness_eval_df, relevance_eval_df, context_precision_eval_df, context_recall_eval_df],
        ignore_index=True,
    )
    merged_df.rename(columns={'label': 'Metrics', 'score': 'Score', 'explanation': 'Explanation'}, inplace=True)
    merged_df["Model"] = item[0]
    merged_df = merged_df.reindex(['Model', 'Metrics Category', 'Metrics', 'Score', 'Explanation'], axis=1)
    vAR_eval_df_list.append(merged_df)


    # merged_df2 = pd.concat(
    #     [hallucination_eval_df, qa_correctness_eval_df, relevance_eval_df, context_precision_eval_df, context_recall_eval_df],
    #     ignore_index=True,
    # )
    # merged_df2["MODEL_NAME"] = item[0]
    # merged_df2["MODEL_RESPONSE"] = item[1]
    # merged_df2["REQUEST_ID"] = assistant_request_id
    # merged_df2["RESPONSE_ID"] = assistant_response_id
    # merged_df2["THREAD_ID"] = item[2]
    # merged_df2["PROMPT"] = vAR_user_input

    # merged_df2["RESPONSE_ID"] = [assistant_response_id] * len(merged_df2)
    # merged_df2["MODEL_NAME"] = [item[0]]* len(merged_df2)
    # merged_df2["MODEL_RESPONSE"] = [item[1]]* len(merged_df2)
    # merged_df2["REQUEST_ID"] = [assistant_request_id]* len(merged_df2)
    # merged_df2["THREAD_ID"] = [item[2]]* len(merged_df2)
    # merged_df2["PROMPT"] = [vAR_user_input]* len(merged_df2)
    # vAR_eval_df_list2.append(merged_df2)



    return vAR_eval_df_list

