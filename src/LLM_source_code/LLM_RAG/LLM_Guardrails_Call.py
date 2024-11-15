import boto3
import os
import streamlit as st

client = boto3.client("bedrock-runtime",aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
                      aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],region_name='us-west-2')

def guardrails_call(input,source):

    response = client.apply_guardrail(
    guardrailIdentifier=os.environ["GUARDRAIL_ID"],
    guardrailVersion='DRAFT',
    source=source,
    content=[
        {
            'text': {
                'text': input
            }
        },
    ]
)
    
    print("Guardrail Response - ",response)
    return response