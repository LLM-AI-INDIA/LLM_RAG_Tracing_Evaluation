import os
from langchain_openai import AzureChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredWordDocumentLoader
import shutil
from google.cloud import storage
import chromadb.api

# GCP Bucket and Local Path Configurations
BUCKET_NAME = "persist_storage"         # Name of your GCP bucket
GCP_DB_PATH = "chroma_store"           # Path in the bucket for the Chroma DB folder
LOCAL_DB_PATH = "./chroma_store"       # Local directory to store Chroma DB

# GCP Utility Functions
def upload_to_gcp(local_path=LOCAL_DB_PATH, gcp_path=GCP_DB_PATH):
    """Uploads local Chroma DB to GCP bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    for root, _, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            blob_path = os.path.relpath(local_file_path, local_path)
            blob = bucket.blob(f"{gcp_path}/{blob_path}")
            blob.upload_from_filename(local_file_path)
    print("Chroma DB uploaded to GCP bucket.")

def download_from_gcp(local_path=LOCAL_DB_PATH, gcp_path=GCP_DB_PATH):
    """Downloads Chroma DB from GCP bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=gcp_path)
    shutil.rmtree(local_path, ignore_errors=True)  # Clear local DB folder
    os.makedirs(local_path, exist_ok=True)
    for blob in blobs:
        local_file_path = os.path.join(local_path, os.path.relpath(blob.name, gcp_path))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
    print("Chroma DB downloaded from GCP bucket.")

# Function 1: Process prompt and retrieve content
def langchain_rag(input_prompt):
    """Retrieve and generate content using RAG."""
    ## Check if local Chroma DB exists; download only if it doesn't
    if not os.path.exists(LOCAL_DB_PATH) or not os.listdir(LOCAL_DB_PATH):
        print("Local Chroma DB not found. Downloading from GCP...")
        download_from_gcp()
    else:
        print("Using local Chroma DB.")

    # Initialize the language model
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-05-01-preview"
    )

    # Load the vector store
    vectorstore = Chroma(
        persist_directory=LOCAL_DB_PATH,
        embedding_function=OpenAIEmbeddings()
    )

    # Set up the retriever
    retriever = vectorstore.as_retriever()

    # Define the prompt template
    prompt = hub.pull("rlm/rag-prompt")

    # Function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Set up the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Retrieve relevant documents
    retrieval_docs = retriever.invoke(input_prompt)
    retrieved_content = format_docs(retrieval_docs)

    # Generate response
    response = rag_chain.invoke(input_prompt)

    return response, retrieved_content

# Function 2: Append new text to the vector store
def append_to_vectorstore(new_text):
    """Append new text to Chroma DB and sync with GCP."""

    # Load the vector store from the local directory
    vectorstore = Chroma(
        persist_directory=LOCAL_DB_PATH,
        embedding_function=OpenAIEmbeddings()
    )

    # Split the new text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_documents = text_splitter.split_text(new_text)

    # Add new documents to the vector store
    vectorstore.add_texts(new_documents)

    # Upload the updated vector store to GCP
    upload_to_gcp()
    print("New data appended to Chroma DB and uploaded to GCP.")

# Function 3: Reset the vector store to base knowledge
def reset_vectorstore(base_knowledge_path, vectorstore_path="./chroma_store"):
    """Reset Chroma DB to base knowledge and sync with GCP."""
    # Initialize the ChromaDB client
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    # Clear the existing vector store
    if os.path.exists(vectorstore_path):
        try:
            for root, dirs, files in os.walk(vectorstore_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except PermissionError:
                        print(f"Permission denied for file: {file_path}")
            shutil.rmtree(vectorstore_path, ignore_errors=True)
        except Exception as e:
            raise PermissionError(f"Unable to delete vector store: {e}")

    # Load the base knowledge document
    loader = UnstructuredWordDocumentLoader(base_knowledge_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Recreate the vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=vectorstore_path
    )

    # Upload updated vector store
    upload_to_gcp()
    print("Reset vector store and uploaded to GCP.")
