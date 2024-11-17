# from openai import OpenAI
# import os


# def human_in_the_loop(request, model_response, feedback):
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#     {
#       "role": "system",
#       "content": [
#         {
#           "type": "text",
#           "text": """your a helpful assistant to Improve the response using the following questions and response to improve alignment with user feedback:
# Task:
# - Identify specific areas in the previous response that require improvement based on the feedback.
# - Clearly explain how the revisions address the identified issues without meaning changes.
# - Generate a improved response that:
# 1. Is clear and easy to understand.
# 2. Aligns with the question and user expectations.
# 3. Effectively incorporates the user's feedback.

# Must Follow these two rules:
# 1. Presents the response in numberd bullet points, with each numbered bullet points on a new line.
# 2. Revise only responses that received negative feedback or thumbs-down indicators.
# 3. Don't add any extra meaning or remove the meaning; moreover, don't give additional information."""
#         }
#       ]
#     },
#     {
#       "role": "user",
#       "content": [
#         {
#           "type": "text",
#           "text": "Inputs: \n- Question: {What do I need to do if I'm moving out of state, but the vehicle registered to me will continue to be registered in California?} \n- Previous Response: {If you are moving out of state but the vehicle registered to you will continue to be registered in California, you need to report your change of address to the DMV within 10 days of moving. You can change your address online to notify the DMV of a change of residence or mailing address for your vehicle, vessel, driver’s license, or identification card records【6:0†source】.} \n- User Feedback: {thumbs-down}"
#         }
#       ]
#     },
#     {
#       "role": "assistant",
#       "content": [
#         {
#           "type": "text",
#           "text": "1. Notify the California DMV of your change of address within 10 days of moving out of state. This can be done online.\n\n2. Ensure that your vehicle insurance meets California's requirements, even if you are residing in another state.\n\n3. Check if your vehicle needs to comply with California's emissions standards, especially if it will be driven in California.\n\n4. Consider any tax implications or registration fees that may apply when keeping a vehicle registered in California while living elsewhere.\n\n5. Contact the California DMV directly for any specific questions or additional requirements related to your situation."
#         }
#       ]
#     },
#     {
#       "role": "user",
#       "content": [
#         {
#           "type": "text",
#           "text": f"Inputs: \n- Question: {request} \n- Previous Response: {model_response} \n- User Feedback: {feedback}"
#         }
#       ]
#     },
#   ],
#     temperature=1,
#     max_tokens=2048,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0,
#     response_format={
#         "type": "text"
#     }
#     )
#     return response.choices[0].message.content
  
  

from openai import AzureOpenAI
import os
def Azure_model_for_human_in_the_loop(request, model_response, feedback):
  client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-09-01-preview"
  )
  response = client.chat.completions.create(
    model="gpt-4o", # replace with the model deployment name of your o1-preview, or o1-mini model
    messages=[
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": """your a helpful assistant to Improve the response using the following questions and response to improve alignment with user feedback:
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
3. Don't add any extra meaning or remove the meaning; moreover, don't give additional information."""
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
          "text": f"Inputs: \n- Question: {request} \n- Previous Response: {model_response} \n- User Feedback: {feedback}"
        }
      ]
    },
  ]

  )

  print(response.choices[0].message.content)
  return response.choices[0].message.content