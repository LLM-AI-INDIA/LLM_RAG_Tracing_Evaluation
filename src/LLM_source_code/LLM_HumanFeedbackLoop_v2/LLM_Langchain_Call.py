import os
from langchain_openai import AzureChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import chromadb.api
import shutil

# Initialize the ChromaDB client
chromadb.api.client.SharedSystemClient.clear_system_cache()

# Function 1: Process prompt and retrieve content
def langchain_rag(input_prompt, vectorstore_path="chroma_store"):
    # Initialize the language model
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-05-01-preview"
    )

    # Load the vector store
    vectorstore = Chroma(
        persist_directory=vectorstore_path,
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
def append_to_vectorstore(new_text, vectorstore_path="chroma_store"):
    # Load the vector store
    vectorstore = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=OpenAIEmbeddings()
    )

    # Split the new text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_documents = text_splitter.split_text(new_text)

    # Add new documents to the vector store
    vectorstore.add_texts(new_documents)
    # No need to call persist() as persistence is automatic

# Function 3: Reset the vector store to base knowledge
def reset_vectorstore(base_knowledge_path, vectorstore_path="chroma_store"):
    # Clear the existing vector store
    if os.path.exists(vectorstore_path):
        shutil.rmtree(vectorstore_path)

    # Load the base knowledge document
    loader = UnstructuredWordDocumentLoader(base_knowledge_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Create a new vector store with the base knowledge
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=vectorstore_path
    )
    
