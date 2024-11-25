



def prompt_template(request, model_response, feedback, feedback_text):
    """
    Constructs a structured prompt template for Azure OpenAI's chat completions API, enabling iterative 
    improvement of model responses based on user feedback."""

    messages=[
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": """your a helpful assistant to improve the response based on user feedback, using the request, model_response, feedback, and feedback_text:
Task:
- Identify specific areas in the previous response that require improvement based on the feedback.
- Clearly explain how the revisions address the identified issues without meaning changes.
- Generate a improved response that:
1. Is clear and easy to understand.
2. Aligns with the question and user expectations.
3. Effectively incorporates the user's feedback.

Must Follow these two rules:
1. Presents the response in numberd bullet points, with each numbered bullet points on a new line.
2. Revise only responses that received negative feedback or thumbs-down indicators.
"""
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Inputs: \n- Question: {What do I need to do if I'm moving out of state, but the vehicle registered to me will continue to be registered in California?} \n- Previous Response: {If you are moving out of state but the vehicle registered to you will continue to be registered in California, you need to report your change of address to the DMV within 10 days of moving. You can change your address online to notify the DMV of a change of residence or mailing address for your vehicle, vessel, driver’s license, or identification card records【6:0†source】.} \n- User Feedback: {thumbs-down}"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "1. Notify the California DMV of your change of address within 10 days of moving out of state. This can be done online.\n\n2. Ensure that your vehicle insurance meets California's requirements, even if you are residing in another state.\n\n3. Check if your vehicle needs to comply with California's emissions standards, especially if it will be driven in California.\n\n4. Consider any tax implications or registration fees that may apply when keeping a vehicle registered in California while living elsewhere.\n\n5. Contact the California DMV directly for any specific questions or additional requirements related to your situation."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": f"Inputs: \n- Question: {request} \n- Previous Response: {model_response} \n- User Feedback: {feedback} \n- User Feedback Text: {feedback_text}"
        }
      ]
    },
  ]
    return messages





# 3. Don't add any extra meaning or remove the meaning; moreover, don't give additional information.