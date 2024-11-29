import streamlit as st
import os
from openai import AzureOpenAI

def Update_vector():
    if "file_copied" not in st.session_state:
        st.session_state.file_copied = False
    if "file_deleted" not in st.session_state:
        st.session_state.file_deleted = False
    # Ensure Azure client is initialized only once
    if "client" not in st.session_state:
        try:
            st.session_state.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-05-01-preview",
            )
        except Exception as e:
            st.error(f"Error initializing Azure client: {e}")
            return

    # File management: Update vector store if file is not deleted yet
    if not st.session_state.file_deleted:
        try:
            file_path = "DMV_FAQ.docx"
            vector_store_files = st.session_state.client.beta.vector_stores.files.list(
                vector_store_id=os.getenv("VECTOR_STORE_ID")
            )

            if vector_store_files.data:
                file_ids = vector_store_files.data[0].id
                st.session_state.client.beta.vector_stores.files.delete(
                    vector_store_id=os.getenv("VECTOR_STORE_ID"), file_id=file_ids
                )

            # Upload updated document to vector store
            response = st.session_state.client.files.create(file=open(file_path, "rb"), purpose="assistants")
            file_id = response.id
            st.session_state.client.beta.vector_stores.files.create(
                vector_store_id=os.getenv("VECTOR_STORE_ID"), file_id=file_id
            )

            st.session_state.file_deleted = True
            print("Base File Updated")
        except Exception as e:
            st.error(f"Error updating vector store: {e}")
            return