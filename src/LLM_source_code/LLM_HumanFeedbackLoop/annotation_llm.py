import os
import time
import streamlit as st
from openai import AzureOpenAI
from google.cloud import bigquery

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Admin\Downloads\elp-2022-352222-230ab152d47c.json"

# Initialize the BigQuery client
biq_query_client = bigquery.Client()

# Specify the table ID (use environment variable for security)
table_id = os.getenv("BIGQUERY_TABLE_ID")


def assistant_call_with_annotation(user_input):
    try:
        # Initialize the Azure OpenAI client
        if "client" not in st.session_state:
            st.session_state.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-05-01-preview",
            )
        client = st.session_state.client

        # Create a new thread if it doesn't exist
        if "thread" not in st.session_state:
            st.session_state.thread = client.beta.threads.create()

        # Add a user question to the thread
        client.beta.threads.messages.create(
            thread_id=st.session_state.thread.id,
            role="user",
            content=user_input,
        )

        # Initiate a run
        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread.id,
            assistant_id=os.getenv("ASSISTANT_ID"),
        )

        # Poll until the run completes or fails
        while run.status in ["queued", "in_progress", "cancelling"]:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread.id, run_id=run.id
            )

        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=st.session_state.thread.id)

            # Loop through messages and return content based on role
            for msg in messages.data:
                content = msg.content[0].text.value
                return content
        else:
            st.error("The assistant was unable to complete your request. Please try again.")
            return "An error occurred during the assistant's processing."

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "The assistant encountered an error. Please try again later."


from streamlit_feedback import streamlit_feedback

def text_based():

    prompt = None  # Initialize prompt with a default value
    m1, m2, m3 = st.columns([1, 5, 1])

    # Initialize chat history and feedback tracking
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "user", "content": "We are delighted to have you here in the Live Agent Chat room!"},
            {"role": "assistant", "content": "Hello! How can I assist you today?"},
        ]
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()  # Track feedback for each assistant message

    # Custom CSS for alignment and styling
    st.markdown(
        """
        <style>
        .chat-container { display: flex; align-items: flex-start; margin: 10px 0; }
        .chat-container.user { justify-content: flex-end; }
        .chat-container.assistant { justify-content: flex-start; }
        .chat-container img { width: 40px; height: 40px; border-radius: 50%; margin: 0 10px; }
        .chat-bubble { max-width: 70%; padding: 10px; border-radius: 10px; margin: 5px 0; }
        .chat-bubble.user { background-color: #e0e0e0; text-align: right; }
        .chat-bubble.assistant { background-color: #ffffff; text-align: left; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    user_image_url = "https://storage.googleapis.com/macrovector-acl-eu/previews/118720/thumb_118720.webp"
    assistant_image_url = "https://cdn-icons-png.flaticon.com/512/6014/6014401.png"

    with m2:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(
                    f'''
                    <div class="chat-container user">
                        <div class="chat-bubble user">{message["content"]}</div>
                        <img src="{user_image_url}" alt="User">
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'''
                    <div class="chat-container assistant">
                        <img src="{assistant_image_url}" alt="Assistant">
                        <div class="chat-bubble assistant">{message["content"]}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

                # Exclude feedback for the first assistant message
                if i == 1:
                    continue

                # Feedback mechanism for all assistant responses, including improved ones
                if i not in st.session_state.feedback_given:
                    feedback_ = streamlit_feedback(
                        align="flex-start",
                        feedback_type="thumbs",
                        optional_text_label="[Optional] Please provide an explanation",
                        key=f"thumbs_{i}"  # Unique key for each feedback element
                    )

                    if feedback_:
                        # Parse the feedback output
                        feedback_type = feedback_.get("type", "unknown")
                        feedback_score = feedback_.get("score", "neutral")
                        feedback_text = feedback_.get("text", "")

                        st.write(f"Thank you for your feedback! You rated this response with {feedback_score}.")
                        if feedback_text:
                            st.write(f"Optional feedback provided: {feedback_text}")

                        # Insert feedback into BigQuery
                        rows_to_insert = [
                            {
                                "request_prompt": st.session_state.messages[i - 1]["content"] if i > 0 else None,
                                "model_response": message["content"],
                                "feedback_type": feedback_type,
                                "feedback_score": feedback_score,
                                "feedback_text": feedback_text,
                            }
                        ]

                        # Insert rows into the BigQuery table
                        errors = biq_query_client.insert_rows_json(table_id, rows_to_insert)

                        # Check for errors
                        if errors == []:
                            st.write("Your feedback has been successfully recorded.")
                        else:
                            st.write("There was an error recording your feedback. Please try again.")
                            st.error(f"BigQuery Insertion Errors: {errors}")

                        # Mark feedback as given for this message
                        st.session_state.feedback_given.add(i)

                        # Process thumbs-down feedback
                        if feedback_score == "👎":
                            # Directly use request, response, and feedback variables
                            request = [st.session_state.messages[i - 1]["content"] if i > 0 else None]
                            response = [message["content"]]
                            feedback = [feedback_score]

                            # Format the user message as a table
                            formatted_message = f"""
                            | Feedback                           | Action                                              |
                            |------------------------------------|-----------------------------------------------------|
                            | {feedback_text}                    | Tuning Model and Re-generating Response in Progress |
                            """
                            st.markdown(formatted_message)
                            
                            # Generate an improved response
                            from src.LLM_source_code.LLM_HumanFeedbackLoop.model import Azure_model_for_human_in_the_loop
                            improved_response = Azure_model_for_human_in_the_loop(request, response, feedback, feedback_text)
                            
                            improved_response = "Here is the improved response based on the given feedback:\n\n"+ improved_response


                            # Update the current assistant message with the improved response
                            st.session_state.messages.append({"role": "assistant", "content": improved_response})
                            st.session_state.messages[i]["content"] = improved_response

                            # Notify the user
                            st.write("Response has been updated based on your feedback.")

        # User input
        prompt = st.chat_input("What else can I do to help?")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = assistant_call_with_annotation(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            time.sleep(1)
            st.rerun()