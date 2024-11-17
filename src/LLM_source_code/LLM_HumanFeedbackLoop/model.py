from openai import AzureOpenAI
import os
from src.LLM_source_code.LLM_HumanFeedbackLoop.prompt_template import prompt_template


def Azure_model_for_human_in_the_loop(request, model_response, feedback, feedback_text):
    """
    Integrates Azure OpenAI's GPT model into a human-in-the-loop workflow for refining AI outputs 
    based on user feedback.

    Args:
        request (str): The original question or input provided to the AI model.
        model_response (str): The previous response generated by the AI model.
        feedback (str): Feedback from the user (e.g., 'thmbsup', 'thmbsdown').
        feedback_text (str): Additional textual feedback provided by the user to elaborate on their response.

    Returns:
        str: The updated response generated by the AI model based on the provided inputs and feedback.
    """
    # Initialize the Azure OpenAI client using environment variables for secure configuration.
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # Azure OpenAI endpoint
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # API key for authentication
        api_version="2024-09-01-preview"  # API version
    )

    # Generate a chat completion request with the given inputs.
    response = client.chat.completions.create(
        model="gpt-4o",  # Specifies the model to be used (GPT-4 Optimized)
        messages=prompt_template(request, model_response, feedback, feedback_text)  # Builds the prompt
    )

    # Log the inputs for debugging purposes.
    print(f"Inputs: \n- Question: {request} \n- Previous Response: {model_response} \n- User Feedback: {feedback} \n- User Feedback Text: {feedback_text}")

    # Return the content of the model's response.
    return response.choices[0].message.content
