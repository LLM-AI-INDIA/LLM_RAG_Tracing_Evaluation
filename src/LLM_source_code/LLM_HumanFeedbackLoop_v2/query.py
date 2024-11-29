from google.cloud import bigquery

def get_expected_response(project_id, dataset_id, table_name, vAR_prompt):
    # Initialize BigQuery client
    client = bigquery.Client()

    # Define the query
    query = f"""
        SELECT `Expected_Response`
        FROM `{project_id}.{dataset_id}.{table_name}`
        WHERE `Prompt` = @prompt
    """

    # Set query parameters
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("prompt", "STRING", vAR_prompt)
        ]
    )

    # Execute the query
    query_job = client.query(query, job_config=job_config)
    results = query_job.result()

    # Convert results to DataFrame
    df = results.to_dataframe()

    if not df.empty:
        # Extract the first value
        expected_response = df.iloc[0]['Expected_Response']
        return expected_response
    else:
        return None