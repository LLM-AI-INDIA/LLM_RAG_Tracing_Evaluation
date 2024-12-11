import pandas as pd
from src.LLM_source_code.LLM_HumanFeedbackLoop_v2.LLM_Eval_Model import multiturn_generate_content_rel,multiturn_generate_content_indirect,multiturn_generate_correctness,multiturn_generate_faithfull


def eval(data_eval_1,data_eval_2):
    vAR_eval_df1_rel = multiturn_generate_content_rel(data_eval_1)
    vAR_eval_df1_indirect = multiturn_generate_content_indirect(data_eval_1)
    vAR_eval_df1_faith = multiturn_generate_faithfull(data_eval_1)
    vAR_eval_df1_correctness = multiturn_generate_correctness(data_eval_1)

    vAR_eval_df2_rel = multiturn_generate_content_rel(data_eval_2)
    vAR_eval_df2_indirect = multiturn_generate_content_indirect(data_eval_2)
    vAR_eval_df2_faith = multiturn_generate_faithfull(data_eval_2)
    vAR_eval_df2_correctness = multiturn_generate_correctness(data_eval_2)

    concatenated_df = pd.concat([vAR_eval_df1_rel, vAR_eval_df1_indirect,vAR_eval_df1_faith,vAR_eval_df1_correctness,vAR_eval_df2_rel,vAR_eval_df2_indirect,vAR_eval_df2_faith,vAR_eval_df2_correctness], ignore_index=True)
    return concatenated_df